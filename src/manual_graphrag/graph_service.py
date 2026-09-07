from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .chunking import TextChunk


SCHEMA_CONTEXT_LIMIT = 30_000
SCHEMA_MERGE_LIMIT = 12_000
SCHEMA_DESCRIPTION_LIMIT = 120
EXTRACTION_BATCH_LIMIT = 12_000


@dataclass(frozen=True)
class SchemaPlan:
    schema: dict[str, Any]
    analyzed_chunks: int
    total_chunks: int
    batch_count: int
    merge_rounds: int


@dataclass(frozen=True)
class GraphExtraction:
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    processed_chunks: int


def _api_url(base_url: str, resource: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("請先填寫模型 API Base URL")
    return f"{base}/{resource.lstrip('/')}"


def _post_json(
    url: str, payload: dict[str, Any], api_key: str, timeout: int = 120
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"模型 API 回傳 HTTP {exc.code}：{detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f"無法連線模型 API：{exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("模型 API 回傳的不是有效 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("模型 API 回傳格式不正確")
    return result


def check_model_connection(base_url: str, api_key: str) -> None:
    url = _api_url(base_url, "models")
    headers = {}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if not 200 <= response.status < 300:
                raise ValueError(f"模型服務回傳 HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"模型服務回傳 HTTP {exc.code}：{detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"模型服務連線失敗：{exc}") from exc


def _extract_json_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("模型未回傳有效的 JSON 結果")


def _chat_response_content(response: dict[str, Any]) -> tuple[str, str]:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        finish_reason = str(choice.get("finish_reason") or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("模型 API 回傳缺少 choices/message/content") from exc
    if not isinstance(content, str):
        raise ValueError("模型回傳內容格式不正確")
    return content, finish_reason


def _chat_json(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0,
    max_output_tokens: int = 2048,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not 0 <= temperature <= 2:
        raise ValueError("temperature 必須介於 0 到 2")
    if max_output_tokens < 1:
        raise ValueError("最大輸出 tokens 必須大於 0")
    if not model.strip():
        raise ValueError("請選擇 LLM 模型")

    url = _api_url(base_url, "chat/completions")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def request_json(request_messages: list[dict[str, str]], request_temperature: float):
        return _post_json(
            url,
            {
                "model": model.strip(),
                "temperature": request_temperature,
                "max_tokens": int(max_output_tokens),
                "messages": request_messages,
            },
            api_key,
        )

    def parse_and_validate(raw_content: str) -> dict[str, Any]:
        parsed = _extract_json_text(raw_content)
        return validator(parsed) if validator else parsed

    response = request_json(messages, temperature)
    content, finish_reason = _chat_response_content(response)
    try:
        return parse_and_validate(content)
    except ValueError as first_error:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "上一個回覆無法通過 JSON 解析或必要結構驗證。"
                    f"驗證錯誤：{first_error}。請修正並只輸出一個完整 JSON 物件，"
                    "不得加入 Markdown code fence 或說明文字，也不得省略必要欄位或回傳空陣列。"
                ),
            },
        ]
        repaired_response = request_json(repair_messages, 0)
        repaired_content, repaired_finish_reason = _chat_response_content(repaired_response)
        try:
            return parse_and_validate(repaired_content)
        except ValueError as exc:
            if finish_reason in {"length", "max_tokens"} or repaired_finish_reason in {
                "length",
                "max_tokens",
            }:
                raise ValueError(
                    "模型 JSON 連續兩次無法通過解析或結構驗證，且輸出可能被截斷；"
                    "請提高最大輸出 tokens 或減少 Schema 類型數量"
                ) from exc
            raise ValueError(
                f"模型 JSON 連續兩次無法通過解析或結構驗證：{exc}"
            ) from first_error


def _chunk_label(chunk: TextChunk) -> str:
    pages = ",".join(map(str, chunk.pages))
    return f"[CHUNK {chunk.number}; PAGES {pages}]\n{chunk.text}"


def _chunk_batches(
    chunks: list[TextChunk], limit: int
) -> list[list[TextChunk]]:
    batches: list[list[TextChunk]] = []
    current: list[TextChunk] = []
    size = 0
    for chunk in chunks:
        chunk_size = len(_chunk_label(chunk)) + 2
        if current and size + chunk_size > limit:
            batches.append(current)
            current = []
            size = 0
        current.append(chunk)
        size += chunk_size
    if current:
        batches.append(current)
    return batches


def _schema_groups(
    schemas: list[dict[str, Any]], limit: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for schema in schemas:
        schema_size = len(json.dumps(schema, ensure_ascii=False)) + 2
        if current and size + schema_size > limit:
            groups.append(current)
            current = []
            size = 0
        current.append(schema)
        size += schema_size
    if current:
        groups.append(current)
    if len(groups) == len(schemas) and len(schemas) > 1:
        return [schemas[index : index + 2] for index in range(0, len(schemas), 2)]
    return groups



def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = validate_schema(schema)

    def compact_description(value: Any) -> str:
        return " ".join(str(value or "").split())[:SCHEMA_DESCRIPTION_LIMIT]

    entity_types = [
        {
            "name": str(item["name"]).strip(),
            "description": compact_description(item.get("description")),
        }
        for item in schema["entity_types"]
    ]
    relationship_types = []
    for item in schema["relationship_types"]:
        relationship = {
            "name": str(item["name"]).strip(),
            "description": compact_description(item.get("description")),
        }
        for field in ("source_types", "target_types"):
            values = item.get(field, [])
            relationship[field] = (
                list(
                    dict.fromkeys(
                        str(value).strip() for value in values if str(value).strip()
                    )
                )
                if isinstance(values, list)
                else []
            )
        relationship_types.append(relationship)
    return {
        "entity_types": entity_types,
        "relationship_types": relationship_types,
    }



def validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    entity_types = schema.get("entity_types")
    relationship_types = schema.get("relationship_types")
    if not isinstance(entity_types, list) or not entity_types:
        raise ValueError("schema 必須包含非空的 entity_types 陣列")
    if not isinstance(relationship_types, list) or not relationship_types:
        raise ValueError("schema 必須包含非空的 relationship_types 陣列")
    for group_name, values in (
        ("entity_types", entity_types),
        ("relationship_types", relationship_types),
    ):
        for value in values:
            if not isinstance(value, dict) or not str(value.get("name", "")).strip():
                raise ValueError(f"{group_name} 每一項都必須包含 name")
    return schema


def plan_graph_schema(
    base_url: str,
    api_key: str,
    llm_model: str,
    chunks: list[TextChunk],
    temperature: float = 0,
    max_output_tokens: int = 2048,
    progress_callback: Callable[[float, str], None] | None = None,
    schema_granularity: str = "平衡",
    max_entity_types: int = 15,
    max_relationship_types: int = 20,
    max_concurrent_requests: int = 3,
) -> SchemaPlan:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    granularity_guidance = {
        "粗略": "只保留最核心的跨章節概念，積極合併上下位與近義類型。",
        "平衡": "保留支援主要查詢所需的通用類型，合併過細的上下位與近義類型。",
        "詳細": "可保留有明確查詢價值的專業子類型，但仍不得把具體實例當成類型。",
    }
    if schema_granularity not in granularity_guidance:
        raise ValueError("Schema 粒度必須是粗略、平衡或詳細")
    if int(max_entity_types) < 1:
        raise ValueError("最大實體類型數必須大於 0")
    if int(max_relationship_types) < 1:
        raise ValueError("最大關係類型數必須大於 0")
    if int(max_concurrent_requests) < 1:
        raise ValueError("最大並行請求數必須大於 0")
    max_entity_types = int(max_entity_types)
    max_relationship_types = int(max_relationship_types)
    max_concurrent_requests = int(max_concurrent_requests)

    def validate_planned_schema(schema: dict[str, Any]) -> dict[str, Any]:
        schema = validate_schema(schema)
        if len(schema["entity_types"]) > max_entity_types:
            raise ValueError(f"entity_types 不得超過 {max_entity_types} 個")
        if len(schema["relationship_types"]) > max_relationship_types:
            raise ValueError(f"relationship_types 不得超過 {max_relationship_types} 個")
        return schema

    planning_rules = (
        f"Schema 粒度：{schema_granularity}。{granularity_guidance[schema_granularity]}"
        "只建立可重複使用、可泛化的類型；具體名稱、型號、編號、人物、組織或章節"
        "應在抽取階段成為實體，不得直接成為類型。只有重複出現且具有獨立查詢或關係"
        "價值的概念才建立類型；相近概念應合併到較高階類型。"
        f"實體類型最多 {max_entity_types} 個，關係類型最多 {max_relationship_types} 個；"
        "若超過上限，應依重要性合併較細類型，不可任意截斷。"
    )
    batches = _chunk_batches(chunks, SCHEMA_CONTEXT_LIMIT)
    candidates: list[dict[str, Any] | None] = [None] * len(batches)

    def plan_batch(batch: list[TextChunk]) -> dict[str, Any]:
        context = "\n\n".join(_chunk_label(chunk) for chunk in batch)
        candidate = _chat_json(
            base_url,
            api_key,
            llm_model,
            "你是知識圖譜 schema 設計專家。只輸出 JSON，不要 Markdown 或說明文字。",
            "請根據這一批文件內容提出候選實體與關係類型。"
            f"{planning_rules}"
            "名稱使用英文大寫 snake case，說明使用繁體中文。"
            "輸出格式：{\"entity_types\":[{\"name\":\"...\",\"description\":\"...\"}],"
            "\"relationship_types\":[{\"name\":\"...\",\"description\":\"...\","
            "\"source_types\":[\"...\"],\"target_types\":[\"...\"]}]}。\n\n"
            f"文件 chunks：\n{context}",
            temperature,
            max_output_tokens,
            validate_planned_schema,
        )
        return _compact_schema(candidate)

    analyzed = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        futures = {
            executor.submit(plan_batch, batch): (index, batch)
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            index, batch = futures[future]
            try:
                candidates[index] = future.result()
            except ValueError as exc:
                raise ValueError(
                    f"Schema 規劃第 {index + 1} / {len(batches)} 批失敗：{exc}"
                ) from exc
            analyzed += len(batch)
            if progress_callback:
                progress_callback(
                    0.75 * analyzed / len(chunks),
                    f"已完成 {sum(item is not None for item in candidates)} / "
                    f"{len(batches)} 批（已分析 {analyzed} / {len(chunks)} chunks）",
                )
    candidates = [candidate for candidate in candidates if candidate is not None]

    merge_rounds = 0
    while len(candidates) > 1:
        merge_rounds += 1
        groups = _schema_groups(candidates, SCHEMA_MERGE_LIMIT)
        merged: list[dict[str, Any] | None] = [None] * len(groups)

        def merge_group(group: list[dict[str, Any]]) -> dict[str, Any]:
            if len(group) == 1:
                return group[0]
            result = _chat_json(
                base_url,
                api_key,
                llm_model,
                "你是知識圖譜 schema 整合專家。只輸出 JSON，不要 Markdown 或說明文字。",
                "合併以下候選 Schema；這不是候選類型的聯集。"
                f"{planning_rules}"
                "請積極去除重複、統一同義名稱並合併上下位與近義類型。"
                "只保留類型名稱、簡短說明，以及關係的 source_types "
                "與 target_types；不要輸出範例、屬性或其他欄位。"
                "輸出格式必須維持 entity_types 與 relationship_types。\n\n"
                f"候選 Schema：\n{json.dumps(group, ensure_ascii=False)}",
                temperature,
                max_output_tokens,
                validate_planned_schema,
            )
            return _compact_schema(result)

        completed_groups = 0
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            futures = {
                executor.submit(merge_group, group): index
                for index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    merged[index] = future.result()
                except ValueError as exc:
                    raise ValueError(
                        f"Schema 第 {merge_rounds} 輪整合第 "
                        f"{index + 1} / {len(groups)} 組失敗：{exc}"
                    ) from exc
                completed_groups += 1
                if progress_callback:
                    progress_callback(
                        min(
                            0.99,
                            0.75
                            + 0.24 * (1 - 0.5 ** (merge_rounds - 1))
                            + 0.24
                            * (0.5**merge_rounds)
                            * completed_groups
                            / len(groups),
                        ),
                        f"第 {merge_rounds} 輪 Schema 整合："
                        f"已完成 {completed_groups} / {len(groups)} 組",
                    )
        candidates = [item for item in merged if item is not None]
    if progress_callback:
        progress_callback(1.0, f"已分析全部 {len(chunks)} / {len(chunks)} chunks")
    return SchemaPlan(
        candidates[0], len(chunks), len(chunks), len(batches), merge_rounds
    )


def _batches(chunks: list[TextChunk]) -> list[list[TextChunk]]:
    return _chunk_batches(chunks, EXTRACTION_BATCH_LIMIT)


def extract_graph(
    base_url: str,
    api_key: str,
    llm_model: str,
    chunks: list[TextChunk],
    schema: dict[str, Any],
    temperature: float = 0,
    max_output_tokens: int = 2048,
) -> GraphExtraction:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    schema = validate_schema(schema)
    allowed_entity_types = {
        str(item["name"]).casefold() for item in schema["entity_types"]
    }
    allowed_relationship_types = {
        str(item["name"]).casefold() for item in schema["relationship_types"]
    }
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    relationships: dict[tuple[str, str, str], dict[str, Any]] = {}
    chunk_lookup = {chunk.number: chunk for chunk in chunks}

    for batch in _batches(chunks):
        context = "\n\n".join(_chunk_label(chunk) for chunk in batch)
        result = _chat_json(
            base_url,
            api_key,
            llm_model,
            "你是知識圖譜資訊抽取器。只能依據提供的文件內容抽取，禁止臆測。只輸出 JSON。",
            "依照 schema 抽取實體與關係。entity.type 與 relationship.type 必須來自 schema；"
            "source_chunk_numbers 必須引用提供的 CHUNK 編號。關係的 source 與 target 使用實體 name。"
            "輸出格式：{\"entities\":[{\"name\":\"...\",\"type\":\"...\","
            "\"description\":\"...\",\"source_chunk_numbers\":[1]}],"
            "\"relationships\":[{\"source\":\"...\",\"target\":\"...\","
            "\"type\":\"...\",\"description\":\"...\",\"source_chunk_numbers\":[1]}]}。\n\n"
            f"schema：\n{json.dumps(schema, ensure_ascii=False)}\n\n文件：\n{context}",
            temperature,
            max_output_tokens,
        )
        for item in result.get("entities", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("type", "")).strip()
            if (
                not name
                or not entity_type
                or entity_type.casefold() not in allowed_entity_types
            ):
                continue
            numbers = _source_numbers(item, chunk_lookup)
            key = (entity_type.casefold(), name.casefold())
            current = entities.setdefault(
                key,
                {
                    "name": name,
                    "type": entity_type,
                    "description": str(item.get("description", "")).strip(),
                    "source_chunk_numbers": [],
                    "source_pages": [],
                },
            )
            _merge_sources(current, numbers, chunk_lookup)
        for item in result.get("relationships", []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation_type = str(item.get("type", "")).strip()
            if (
                not source
                or not target
                or not relation_type
                or relation_type.casefold() not in allowed_relationship_types
            ):
                continue
            numbers = _source_numbers(item, chunk_lookup)
            key = (source.casefold(), relation_type.casefold(), target.casefold())
            current = relationships.setdefault(
                key,
                {
                    "source": source,
                    "type": relation_type,
                    "target": target,
                    "description": str(item.get("description", "")).strip(),
                    "source_chunk_numbers": [],
                    "source_pages": [],
                },
            )
            _merge_sources(current, numbers, chunk_lookup)
    return GraphExtraction(list(entities.values()), list(relationships.values()), len(chunks))


def _source_numbers(
    item: dict[str, Any], chunk_lookup: dict[int, TextChunk]
) -> list[int]:
    raw = item.get("source_chunk_numbers", [])
    if not isinstance(raw, list):
        return []
    numbers: list[int] = []
    for value in raw:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number in chunk_lookup and number not in numbers:
            numbers.append(number)
    return numbers


def _merge_sources(
    item: dict[str, Any], numbers: list[int], chunk_lookup: dict[int, TextChunk]
) -> None:
    for number in numbers:
        if number not in item["source_chunk_numbers"]:
            item["source_chunk_numbers"].append(number)
        for page in chunk_lookup[number].pages:
            if page not in item["source_pages"]:
                item["source_pages"].append(page)
