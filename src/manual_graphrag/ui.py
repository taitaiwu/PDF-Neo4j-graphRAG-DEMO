from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

from .chunking import TextChunk, chunk_pages, preview_rows_for_page
from .config import public_settings
from .env_store import load_env, save_env
from .graph_service import (
    extract_graph,
    plan_graph_schema,
    check_model_connection,
    validate_schema,
)
from .neo4j_service import (
    check_neo4j_connection,
    import_extraction,
    load_latest_graph,
    search_graph_evidence,
)
from .pdf_service import extract_pdf, get_pdf_page_count
from .project_store import append_question, create_project, list_projects, load_project, save_project
from .qa_service import answer_graph_question, embedding_vectors
from .storage import write_json


def connection_summary(
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    model_endpoint: str,
    api_key: str,
) -> tuple[str, dict[str, object]]:
    missing = [
        label
        for label, value in {
            "Neo4j URI": neo4j_uri,
            "Database": neo4j_database,
            "Username": neo4j_username,
            "Password": neo4j_password,
            "模型端點": model_endpoint,
        }.items()
        if not value.strip()
    ]
    if missing:
        return f"⚠️ 尚未填寫：{'、'.join(missing)}", {}
    settings = public_settings(
        {
            "neo4j_uri": neo4j_uri,
            "neo4j_database": neo4j_database,
            "neo4j_username": neo4j_username,
            "neo4j_password": neo4j_password,
            "model_endpoint": model_endpoint,
            "api_key": api_key,
        }
    )
    return "✅ 欄位格式已通過初步檢查；實際連線將於下一版接入。", settings


def check_neo4j_for_ui(
    uri: str, database: str, username: str, password: str
) -> str:
    try:
        check_neo4j_connection(uri, database, username, password)
    except ValueError as exc:
        return f"❌ {exc}"
    return "✅ Neo4j 連線成功，且可存取指定 Database。"


def check_model_service_for_ui(base_url: str, api_key: str) -> str:
    try:
        check_model_connection(base_url, api_key)
    except ValueError as exc:
        return f"❌ {exc}"
    return "✅ 模型服務連線成功。"


def persist_env_settings(
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    model_endpoint: str,
    api_key: str,
    build_model: str,
    embedding_model: str,
    answer_model: str,
) -> str:
    save_env(
        {
            "NEO4J_URI": neo4j_uri,
            "NEO4J_DATABASE": neo4j_database,
            "NEO4J_USERNAME": neo4j_username,
            "NEO4J_PASSWORD": neo4j_password,
            "MODEL_API_BASE": model_endpoint,
            "MODEL_API_KEY": api_key,
            "BUILD_MODEL": build_model,
            "EMBEDDING_MODEL": embedding_model,
            "ANSWER_MODEL": answer_model,
        }
    )
    return "✅ 已自動儲存至 .env"


def reload_env_settings() -> tuple[str, ...]:
    settings = load_env()
    return (
        settings["NEO4J_URI"],
        settings["NEO4J_DATABASE"],
        settings["NEO4J_USERNAME"],
        settings["NEO4J_PASSWORD"],
        settings["MODEL_API_BASE"],
        settings["MODEL_API_KEY"],
        settings["BUILD_MODEL"],
        settings["EMBEDDING_MODEL"],
        settings["ANSWER_MODEL"],
        "✅ 已重新讀取 .env",
    )



def _project_choices() -> list[tuple[str, str]]:
    return list_projects()


def create_project_for_ui(name: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        project = create_project(name)
    except ValueError as exc:
        return gr.update(), {}, f"❌ {exc}"
    return gr.update(choices=_project_choices(), value=project["project_id"]), project, f"✅ 已建立專案「{project['name']}」。"


def refresh_projects_for_ui() -> dict[str, Any]:
    return gr.update(choices=_project_choices())


def _chunk_dicts(chunks: list[TextChunk]) -> list[dict[str, Any]]:
    return [{"number": c.number, "text": c.text, "pages": list(c.pages)} for c in chunks]


def _stored_chunks(items: list[dict[str, Any]]) -> list[TextChunk]:
    return [TextChunk(int(i["number"]), str(i["text"]), tuple(i.get("pages") or [])) for i in items]


def _history_rows(questions: list[dict[str, Any]]) -> list[list[object]]:
    return [[i.get("asked_at", ""), i.get("question", ""), i.get("answer", ""),
             i.get("retrieval_mode", ""), i.get("document", "")] for i in reversed(questions)]


def save_project_for_ui(
    project_id: str, pdf_file: str | None, preview_state: dict[str, Any],
    chunks: list[TextChunk], graph_state: dict[str, Any],
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    model_endpoint: str, api_key: str, graph_llm_model: str,
    graph_embedding_model: str, answer_model: str, start_page: int,
    end_page: int | None, chunk_size: int, chunk_overlap: int,
    graph_temperature: float, graph_max_output_tokens: int,
    schema_granularity: str, max_entity_types: int, max_relationship_types: int,
    max_concurrent_requests: int, schema_sampling_mode: str,
    schema_sample_page_count: int, extraction_llm_model: str,
    extraction_max_concurrent_requests: int, import_mode: str,
    retrieval_mode: str, top_k: int, schema_text: str,
) -> tuple[dict[str, Any], str]:
    if not project_id:
        return {}, "❌ 請先建立或載入專案。"
    settings = {
        "neo4j_uri": neo4j_uri, "neo4j_database": neo4j_database,
        "neo4j_username": neo4j_username, "neo4j_password": neo4j_password,
        "model_endpoint": model_endpoint, "api_key": api_key,
        "graph_llm_model": graph_llm_model, "graph_embedding_model": graph_embedding_model,
        "answer_model": answer_model, "start_page": int(start_page),
        "end_page": None if end_page is None else int(end_page),
        "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
        "graph_temperature": float(graph_temperature),
        "graph_max_output_tokens": int(graph_max_output_tokens),
        "schema_granularity": schema_granularity,
        "max_entity_types": int(max_entity_types), "max_relationship_types": int(max_relationship_types),
        "max_concurrent_requests": int(max_concurrent_requests),
        "schema_sampling_mode": schema_sampling_mode,
        "schema_sample_page_count": int(schema_sample_page_count),
        "extraction_llm_model": extraction_llm_model,
        "extraction_max_concurrent_requests": int(extraction_max_concurrent_requests),
        "import_mode": import_mode, "retrieval_mode": retrieval_mode,
        "top_k": int(top_k), "schema_text": schema_text or "",
    }
    try:
        project = save_project(project_id, {
            "settings": settings, "preview_state": preview_state or {},
            "chunks": _chunk_dicts(chunks or []), "graph_state": graph_state or {},
        }, pdf_file)
    except (OSError, ValueError) as exc:
        return {}, f"❌ {exc}"
    return project, f"✅ 專案「{project['name']}」已保存。"


def load_project_for_ui(project_id: str) -> tuple[Any, ...]:
    try:
        project = load_project(project_id)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc))
    settings = project.get("settings") or {}
    env, get = load_env(), settings.get
    document_path = (project.get("document") or {}).get("path")
    if document_path and not Path(document_path).is_file():
        document_path = None
    preview = project.get("preview_state") or {}
    chunks = _stored_chunks(project.get("chunks") or [])
    graph = project.get("graph_state") or {}
    page_start = int(preview.get("page_start", get("start_page", 1)))
    page_end = int(preview.get("page_end", get("end_page") or page_start))
    rows = preview_rows_for_page(chunks, page_start) if chunks else []
    return (
        project, f"✅ 已載入專案「{project['name']}」。", document_path,
        get("neo4j_uri", env["NEO4J_URI"]), get("neo4j_database", env["NEO4J_DATABASE"]),
        get("neo4j_username", env["NEO4J_USERNAME"]), get("neo4j_password", env["NEO4J_PASSWORD"]),
        get("model_endpoint", env["MODEL_API_BASE"]), get("api_key", env["MODEL_API_KEY"]),
        get("graph_llm_model", env["BUILD_MODEL"]), get("graph_embedding_model", env["EMBEDDING_MODEL"]),
        get("answer_model", env["ANSWER_MODEL"]), get("start_page", 1), get("end_page"),
        get("chunk_size", 1500), get("chunk_overlap", 200), get("graph_temperature", 0),
        get("graph_max_output_tokens", 4096), get("schema_granularity", "平衡"),
        get("max_entity_types", 15), get("max_relationship_types", 20),
        get("max_concurrent_requests", 3),
        gr.update(
            value=get("schema_sampling_mode", "全部頁面"),
            visible=get("schema_sampling_mode", "全部頁面") == "隨機抽取 N 頁",
        ),
        get("schema_sample_page_count", 10), get("extraction_llm_model", env["BUILD_MODEL"]),
        get("extraction_max_concurrent_requests", 3), get("import_mode", "保留既有圖譜"),
        get("retrieval_mode", "GraphRAG"), get("top_k", 8), get("schema_text", ""),
        preview, chunks, graph,
        gr.update(minimum=page_start, maximum=page_end, value=page_start, interactive=bool(chunks)),
        rows, _page_status(page_start, page_end, len(rows)) if chunks else "請先解析 PDF。",
        _history_rows(project.get("questions") or []),
    )


def answer_question_for_project_ui(project_id: str, *args: Any) -> tuple[Any, ...]:
    status, answer, sources = answer_question_for_ui(*args)
    if not status.startswith("✅") or not project_id:
        note = "" if project_id else "⚠️ 未選擇專案，問答未加入專案紀錄。"
        return status, answer, sources, gr.update(), note
    try:
        current = load_project(project_id)
        record = {
            "question": str(args[7]).strip(), "answer": answer,
            "answer_model": str(args[6]), "retrieval_mode": str(args[8]),
            "top_k": int(args[9]), "sources": sources,
            "document": (current.get("graph_state") or {}).get("document", ""),
        }
        project = append_question(project_id, record)
    except (OSError, ValueError) as exc:
        return status, answer, sources, gr.update(), f"⚠️ 回答成功，但專案紀錄保存失敗：{exc}"
    return status, answer, sources, _history_rows(project.get("questions") or []), "✅ 問答紀錄已加入目前專案。"

def initialize_page_range(
    file_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not file_path:
        return (
            gr.update(value=1),
            gr.update(value=None),
            "尚未解析 PDF。",
        )
    try:
        page_count = get_pdf_page_count(file_path)
    except ValueError as exc:
        return gr.update(value=1), gr.update(value=None), f"❌ {exc}"
    return (
        gr.update(value=1, maximum=page_count),
        gr.update(value=page_count, maximum=page_count),
        f"已偵測到 {page_count} 頁；解析結束頁預設為第 {page_count} 頁。",
    )

def preview_pdf(
    file_path: str | None,
    chunk_size: int,
    chunk_overlap: int,
    start_page: int | float = 1,
    end_page: int | float | None = None,
) -> tuple[
    str, list[list[object]], dict[str, Any], list[TextChunk], dict[str, Any], str
]:
    if not file_path:
        return "請先上傳 PDF。", [], {}, [], gr.update(), "尚未解析 PDF。"
    try:
        parsed_chunk_size = int(chunk_size)
        parsed_chunk_overlap = int(chunk_overlap)
        if not 100 <= parsed_chunk_size <= 10000:
            raise ValueError("chunk_size 必須介於 100 到 10,000")
        if not 0 <= parsed_chunk_overlap < parsed_chunk_size:
            raise ValueError("chunk_overlap 必須大於等於 0 且小於 chunk_size")
        requested_start = int(start_page)
        requested_end = None if end_page is None else int(end_page)
        pages, empty_pages = extract_pdf(file_path, requested_start, requested_end)
        chunks = chunk_pages(pages, parsed_chunk_size, parsed_chunk_overlap)
        if not chunks:
            return (
                "PDF 沒有可解析文字；掃描文件需在後續版本加入 OCR。",
                [],
                {},
                [],
                gr.update(),
                "沒有可預覽的頁面。",
            )
        parsed_start = pages[0].page
        parsed_end = pages[-1].page
        state = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "page_count": len(pages),
            "page_start": parsed_start,
            "page_end": parsed_end,
            "empty_pages": empty_pages,
            "chunk_count": len(chunks),
            "config": {
                "chunk_size": parsed_chunk_size,
                "chunk_overlap": parsed_chunk_overlap,
            },
        }
        note = f"已解析第 {parsed_start}–{parsed_end} 頁，產生 {len(chunks)} 個 chunk。"
        if empty_pages:
            note += f" 無文字頁面：{', '.join(map(str, empty_pages))}。"
        first_page_rows = preview_rows_for_page(chunks, parsed_start)
        return (
            note,
            first_page_rows,
            state,
            chunks,
            gr.update(
                minimum=parsed_start,
                maximum=parsed_end,
                value=parsed_start,
                interactive=True,
            ),
            _page_status(parsed_start, parsed_end, len(first_page_rows)),
        )
    except (ValueError, TypeError) as exc:
        return f"❌ {exc}", [], {}, [], gr.update(), "無法預覽頁面。"


def _page_status(page_number: int, page_count: int, chunk_count: int) -> str:
    if chunk_count:
        detail = f"顯示 {chunk_count} 個相關 chunk。"
    else:
        detail = "本頁沒有可解析文字或相關 chunk。"
    return f"第 {page_number} / {page_count} 頁；{detail}"


def preview_page(
    page_number: int | float, chunks: list[TextChunk], state: dict[str, Any]
) -> tuple[str, list[list[object]]]:
    if not state or not chunks:
        return "請先解析 PDF。", []
    page_start = max(1, int(state.get("page_start", 1)))
    page_end = max(page_start, int(state.get("page_end", state.get("page_count", 1))))
    page = max(page_start, min(int(page_number), page_end))
    rows = preview_rows_for_page(chunks, page)
    return _page_status(page, page_end, len(rows)), rows


def _move_page(page_number: int | float, state: dict[str, Any], offset: int) -> int:
    page_start = max(1, int(state.get("page_start", 1))) if state else 1
    page_end = max(page_start, int(state.get("page_end", state.get("page_count", 1))))
    return max(page_start, min(int(page_number) + offset, page_end))


def previous_page(page_number: int | float, state: dict[str, Any]) -> int:
    return _move_page(page_number, state, -1)


def next_page(page_number: int | float, state: dict[str, Any]) -> int:
    return _move_page(page_number, state, 1)


def save_config(state: dict[str, Any]) -> tuple[str, str | None]:
    if not state:
        return "請先成功預覽 PDF。", None
    output = write_json(Path("data/exports/latest-config.json"), state)
    return f"設定已儲存：{output}", str(output)


def _select_schema_planning_chunks(
    chunks: list[TextChunk], sampling_mode: str, sample_page_count: int
) -> tuple[list[TextChunk], list[int]]:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    available_pages = sorted({page for chunk in chunks for page in chunk.pages})
    if sampling_mode == "全部頁面":
        return chunks, available_pages
    if sampling_mode != "隨機抽取 N 頁":
        raise ValueError("不支援的 Schema 規劃範圍")
    sample_page_count = int(sample_page_count)
    if sample_page_count < 1:
        raise ValueError("隨機抽取頁數必須至少為 1")
    if sample_page_count > len(available_pages):
        raise ValueError(
            f"隨機抽取頁數不可超過可用頁數 {len(available_pages)}"
        )
    sampled_pages = sorted(random.sample(available_pages, sample_page_count))
    sampled_page_set = set(sampled_pages)
    selected_chunks = [
        chunk for chunk in chunks if sampled_page_set.intersection(chunk.pages)
    ]
    return selected_chunks, sampled_pages


def plan_schema_for_ui(
    model_endpoint: str,
    api_key: str,
    llm_model: str,
    temperature: float,
    max_output_tokens: int,
    schema_granularity: str,
    max_entity_types: int,
    max_relationship_types: int,
    max_concurrent_requests: int,
    sampling_mode: str,
    sample_page_count: int,
    chunks: list[TextChunk],
    progress=gr.Progress(),
) -> tuple[str, str]:
    try:
        planning_chunks, selected_pages = _select_schema_planning_chunks(
            chunks, sampling_mode, sample_page_count
        )
        plan = plan_graph_schema(
            model_endpoint,
            api_key,
            llm_model,
            planning_chunks,
            float(temperature),
            int(max_output_tokens),
            lambda value, description: progress(value, desc=description),
            schema_granularity,
            int(max_entity_types),
            int(max_relationship_types),
            int(max_concurrent_requests),
        )
    except ValueError as exc:
        return f"❌ {exc}", ""
    if sampling_mode == "全部頁面":
        scope_note = f"全部 {len(selected_pages)} 頁"
    else:
        sampled_page_text = ", ".join(map(str, selected_pages))
        scope_note = (
            f"隨機抽取 {len(selected_pages)} 頁（頁碼：{sampled_page_text}）"
        )
    note = (
        f"✅ 已使用 {llm_model} 規劃 schema；參考 {scope_note}、"
        f"{plan.analyzed_chunks} 個 chunk，"
        f"共 {plan.batch_count} 批、{plan.merge_rounds} 輪整合。"
        f"粒度：{schema_granularity}；實體／關係類型上限："
        f"{int(max_entity_types)}／{int(max_relationship_types)}。"
        f"最大並行請求數：{int(max_concurrent_requests)}。"
        "請確認或編輯後再進行抽取。"
    )
    return note, json.dumps(plan.schema, ensure_ascii=False, indent=2)


def extract_graph_for_ui(
    model_endpoint: str,
    api_key: str,
    llm_model: str,
    temperature: float,
    max_output_tokens: int,
    max_concurrent_requests: int,
    chunks: list[TextChunk],
    schema_text: str,
    preview_state: dict[str, Any],
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], list[list[object]], dict[str, Any]]:
    try:
        raw_schema = json.loads(schema_text)
        if not isinstance(raw_schema, dict):
            raise ValueError("schema 必須是 JSON 物件")
        schema = validate_schema(raw_schema)
        extraction = extract_graph(
            model_endpoint,
            api_key,
            llm_model,
            chunks,
            schema,
            float(temperature),
            int(max_output_tokens),
            int(max_concurrent_requests),
            lambda value, description: progress(value, desc=description),
        )
    except json.JSONDecodeError:
        return "❌ schema 不是有效 JSON。", [], [], {}
    except (ValueError, RuntimeError) as exc:
        return f"❌ {exc}", [], [], {}

    entity_rows = [
        [
            item["name"],
            item["type"],
            item["description"],
            ", ".join(map(str, item["source_chunk_numbers"])),
            ", ".join(map(str, item["source_pages"])),
        ]
        for item in extraction.entities
    ]
    relationship_rows = [
        [
            item["source"],
            item["type"],
            item["target"],
            item["description"],
            ", ".join(map(str, item["source_chunk_numbers"])),
            ", ".join(map(str, item["source_pages"])),
        ]
        for item in extraction.relationships
    ]
    run_id = str(uuid4())
    document_name = preview_state.get("file_name", "")
    graph_state = {
        "run_id": run_id,
        "document": document_name,
        "llm_model": llm_model,
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "max_concurrent_requests": int(max_concurrent_requests),
        "schema": schema,
        "entities": extraction.entities,
        "relationships": extraction.relationships,
        "chunks": [
            {"number": chunk.number, "text": chunk.text, "pages": list(chunk.pages)}
            for chunk in chunks
        ],
    }
    graph_state["neo4j_imported"] = False
    status = (
        f"✅ 已處理 {extraction.processed_chunks} 個 chunk，抽取 "
        f"{len(extraction.entities)} 個實體與 "
        f"{len(extraction.relationships)} 筆關係（最大並行請求數：{int(max_concurrent_requests)}）。請確認結果後進行 Embedding 並匯入 Neo4j。"
    )
    return status, entity_rows, relationship_rows, graph_state


def _build_graph_evidence(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = []
    for item in chunks:
        number = int(item.get("number", 0))
        evidence.append({
            "evidence_id": f"chunk-{number}", "kind": "原文",
            "name": "", "source": "", "target": "",
            "text": str(item.get("text", "")),
            "source_pages": item.get("pages", []),
            "source_chunk_numbers": [number],
        })
    for index, item in enumerate(entities):
        evidence.append({
            "evidence_id": f"entity-{index}", "kind": "實體",
            "name": item.get("name", ""), "source": "", "target": "",
            "text": "實體：{}；類型：{}；說明：{}".format(
                item.get("name", ""), item.get("type", ""), item.get("description", "")
            ),
            "source_pages": item.get("source_pages", []),
            "source_chunk_numbers": item.get("source_chunk_numbers", []),
        })
    for index, item in enumerate(relationships):
        evidence.append({
            "evidence_id": f"relationship-{index}", "kind": "關係", "name": "",
            "source": item.get("source", ""), "target": item.get("target", ""),
            "text": "關係：{} -[{}]-> {}；說明：{}".format(
                item.get("source", ""), item.get("type", ""),
                item.get("target", ""), item.get("description", "")
            ),
            "source_pages": item.get("source_pages", []),
            "source_chunk_numbers": item.get("source_chunk_numbers", []),
        })
    return evidence


def import_graph_for_ui(
    model_endpoint: str,
    api_key: str,
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    embedding_model: str,
    import_mode: str,
    destructive_confirmed: bool,
    graph_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not graph_state or not graph_state.get("run_id"):
        return "❌ 請先完成知識圖譜抽取。", graph_state or {}
    if not embedding_model.strip():
        return "❌ 請選擇 Embedding 模型。", graph_state
    if import_mode != "保留既有圖譜" and not destructive_confirmed:
        return "❌ 取代或清空資料前必須勾選確認。", graph_state
    updated_state = dict(graph_state)
    updated_state["embedding_model"] = embedding_model.strip()
    try:
        evidence = _build_graph_evidence(
            updated_state["entities"],
            updated_state["relationships"],
            updated_state.get("chunks", []),
        )
        if not evidence:
            raise ValueError("沒有可建立向量索引的原文、實體或關係")
        vectors = embedding_vectors(
            model_endpoint, api_key, embedding_model,
            [item["text"] for item in evidence],
        )
        for item, vector in zip(evidence, vectors):
            item["embedding"] = vector
        imported = import_extraction(
            neo4j_uri,
            neo4j_database,
            neo4j_username,
            neo4j_password,
            updated_state["run_id"],
            updated_state.get("document", ""),
            updated_state.get("llm_model", ""),
            updated_state.get("embedding_model", ""),
            updated_state["schema"],
            updated_state["entities"],
            updated_state["relationships"],
            evidence,
            import_mode,
            bool(destructive_confirmed),
        )
    except (KeyError, ValueError) as exc:
        updated_state["neo4j_imported"] = False
        updated_state["neo4j_error"] = str(exc)
        return f"❌ {exc}", updated_state

    updated_state["neo4j_imported"] = True
    updated_state.pop("neo4j_error", None)
    return (
        f"✅ 匯入模式：{import_mode}；已建立 {len(evidence)} 筆向量證據、Vector Index 與 Full-text Index；已匯入 Neo4j {imported.entity_count} 個實體與 "
        f"{imported.relationship_count} 筆關係。",
        updated_state,
    )


def answer_question_for_ui(
    model_endpoint: str,
    api_key: str,
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    answer_model: str,
    question: str,
    retrieval_mode: str,
    top_k: int,
) -> tuple[str, str, list[list[object]]]:
    if not question.strip():
        return "請輸入問題。", "", []
    try:
        graph_state = load_latest_graph(
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password
        )
        question_vector = embedding_vectors(
            model_endpoint, api_key, graph_state.get("embedding_model", ""),
            [question.strip()],
        )[0]
        evidence = search_graph_evidence(
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            graph_state["run_id"], question.strip(), question_vector,
            retrieval_mode, int(top_k),
        )
        result = answer_graph_question(
            model_endpoint, api_key, answer_model, question, retrieval_mode, evidence
        )
    except ValueError as exc:
        return f"❌ {exc}", "", []
    rows = [
        [
            item["kind"],
            item["text"],
            ", ".join(item.get("matched_by", [])),
            f"{item.get('fusion_score', 0.0):.4f}",
            ", ".join(map(str, item.get("source_pages", []))),
            ", ".join(map(str, item.get("source_chunk_numbers", []))),
        ]
        for item in result["evidence"]
    ]
    record = {
        "run_id": graph_state.get("run_id"),
        "document": graph_state.get("document", ""),
        "answer_model": answer_model,
        "embedding_model": graph_state.get("embedding_model", ""),
        "question": question.strip(),
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "answer": result["answer"],
        "evidence": result["evidence"],
    }
    output = write_json(Path("data/qa") / f"{uuid4()}.json", record)
    status = (
        f"✅ {retrieval_mode} 已使用 {len(rows)} 筆證據完成回答；"
        f"文件：{graph_state.get('document', '未知')}；紀錄：{output}。"
    )
    return status, result["answer"], rows


def build_app() -> gr.Blocks:
    env = load_env()
    with gr.Blocks(title="PDF GraphRAG 測試工具", fill_width=True) as app:
        gr.Markdown(
            "# PDF GraphRAG 測試工具\n"
            "上傳使用手冊、調整建圖參數，並測試 Neo4j GraphRAG。"
        )
        project_state = gr.State({})
        preview_state = gr.State({})
        chunk_state = gr.State([])
        graph_state = gr.State({})

        with gr.Tab("0. 專案設定") as project_tab:
            gr.Markdown("### 專案工作區\n建立或載入專案後，可保存本頁面所有連線、模型、參數、Chunk、文件、建圖狀態與問答紀錄。")
            with gr.Row():
                project_selector = gr.Dropdown(
                    choices=_project_choices(), label="現有專案", interactive=True
                )
                load_project_button = gr.Button("載入專案", variant="primary")
            with gr.Row():
                new_project_name = gr.Textbox(label="新專案名稱", placeholder="例如：ALCX17 使用手冊")
                create_project_button = gr.Button("建立新專案", variant="primary")
            project_status = gr.Markdown("尚未選擇專案；載入後，設定與處理結果都會自動保存。")
            gr.Markdown("⚠️ 專案設定保存在本機 `data/projects/`，其中 Password 與 API Key 為明文；請勿分享或提交該目錄。")

        with gr.Tab("1. 連線設定"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Neo4j")
                    neo4j_uri = gr.Textbox(label="URI", value=env["NEO4J_URI"])
                    neo4j_database = gr.Textbox(label="Database", value=env["NEO4J_DATABASE"])
                    neo4j_username = gr.Textbox(label="Username", value=env["NEO4J_USERNAME"])
                    neo4j_password = gr.Textbox(label="Password", value=env["NEO4J_PASSWORD"], type="password")
                    neo4j_test_button = gr.Button("測試 Neo4j 連線", variant="primary")
                    neo4j_connection_status = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### 模型服務")
                    model_endpoint = gr.Textbox(label="API Base URL", value=env["MODEL_API_BASE"])
                    api_key = gr.Textbox(label="API Key", value=env["MODEL_API_KEY"], type="password")
                    model_test_button = gr.Button("測試模型服務連線", variant="primary")
                    model_connection_status = gr.Markdown()
                    reload_button = gr.Button("重新讀取 .env")
            gr.Markdown("⚠️ Password 與 API Key 會以明文寫入本機 `.env`；請勿提交此檔案。")
            env_status = gr.Markdown("啟動時已讀取 .env；欄位修改後會自動儲存。")

        with gr.Tab("2. PDF 與參數"):
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_file = gr.File(label="PDF 使用手冊", file_types=[".pdf"], type="filepath")
                    with gr.Row():
                        start_page = gr.Number(
                            value=1, minimum=1, precision=0, label="解析起始頁"
                        )
                        end_page = gr.Number(
                            value=None,
                            minimum=1,
                            precision=0,
                            label="解析結束頁（上傳後自動設為最後一頁）",
                        )
                    chunk_size = gr.Slider(100, 10000, value=1500, step=100, label="Chunk size（字元）")
                    chunk_overlap = gr.Slider(0, 2000, value=200, step=50, label="Chunk overlap（字元）")
                    preview_button = gr.Button("解析並預覽 Chunk", variant="primary")
                    export_button = gr.Button("匯出目前設定")
                    export_file = gr.File(label="設定 JSON", interactive=False)
                with gr.Column(scale=3):
                    preview_status = gr.Markdown("尚未解析 PDF。")
                    with gr.Row():
                        previous_button = gr.Button("上一頁", scale=1)
                        page_selector = gr.Slider(
                            1, 1, value=1, step=1, label="PDF 頁碼", interactive=False, scale=8
                        )
                        next_button = gr.Button("下一頁", scale=1)
                    page_status = gr.Markdown("請先解析 PDF。")
                    chunk_table = gr.Dataframe(
                        headers=["編號", "頁碼", "字元數", "內容"],
                        datatype=["number", "str", "number", "str"],
                        interactive=False,
                        wrap=True,
                        max_height=750,
                        column_widths=[80, 120, 100, 900],
                    )

        with gr.Tab("3. 建圖"):
            gr.Markdown("### 規劃並抽取知識圖譜")
            with gr.Group():
                gr.Markdown("#### ① 規劃 Schema")
                gr.Markdown(
                    "選擇 LLM，從上一頁產生的 chunks 規劃實體與關係類型。"
                )
                with gr.Row():
                    graph_llm_model = gr.Dropdown(
                        choices=list(dict.fromkeys([env["BUILD_MODEL"], env["ANSWER_MODEL"]])),
                        value=env["BUILD_MODEL"],
                        allow_custom_value=True,
                        label="Schema 規劃 LLM",
                    )
                    graph_temperature = gr.Slider(
                        0, 2, value=0, step=0.1, label="Temperature"
                    )
                    graph_max_output_tokens = gr.Number(
                        value=4096, minimum=1, precision=0, label="最大輸出 tokens"
                    )
                with gr.Row():
                    schema_granularity = gr.Radio(
                        ["粗略", "平衡", "詳細"],
                        value="平衡",
                        label="Schema 粒度",
                    )
                    max_entity_types = gr.Number(
                        value=15, minimum=1, precision=0, label="最大實體類型數"
                    )
                    max_relationship_types = gr.Number(
                        value=20, minimum=1, precision=0, label="最大關係類型數"
                    )
                    max_concurrent_requests = gr.Number(
                        value=3, minimum=1, precision=0, label="最大並行請求數"
                    )
                with gr.Row():
                    schema_sampling_mode = gr.Radio(
                        ["全部頁面", "隨機抽取 N 頁"],
                        value="全部頁面",
                        label="Schema 規劃範圍",
                    )
                    schema_sample_page_count = gr.Number(
                        value=10, minimum=1, precision=0, label="隨機抽取頁數 N", visible=False
                    )
                plan_schema_button = gr.Button(
                    "分析文件並規劃 Schema", variant="primary"
                )
                plan_status = gr.Markdown("請先在 PDF 頁面解析並產生 chunks。")
                gr.HTML(
                    """
                    <style>
                    .schema-scroll-editor .cm-content {
                        font-size: 17px;
                        line-height: 1.6;
                    }
                    </style>
                    """,
                    padding=False,
                )
                schema_editor = gr.Code(
                    label="實體與關係 Schema（可編輯 JSON）",
                    language="json",
                    interactive=True,
                    lines=18,
                    max_lines=18,
                    wrap_lines=False,
                    elem_classes="schema-scroll-editor",
                )

            with gr.Group():
                gr.Markdown("#### ② 確認 Schema 並抽取知識圖譜")
                gr.Markdown(
                    "確認上方 JSON 後執行全部 chunks；檢查抽取結果後，再手動匯入 Neo4j。"
                )
                extraction_llm_model = gr.Dropdown(
                    choices=list(
                        dict.fromkeys([env["BUILD_MODEL"], env["ANSWER_MODEL"]])
                    ),
                    value=env["BUILD_MODEL"],
                    allow_custom_value=True,
                    label="知識圖譜抽取 LLM",
                )
                extraction_max_concurrent_requests = gr.Number(
                    value=3,
                    minimum=1,
                    precision=0,
                    label="最大並行請求數",
                )
                generate_graph_button = gr.Button(
                    "確認 Schema 並抽取", variant="primary"
                )
                build_status = gr.Markdown("尚未執行抽取。")
                gr.Markdown("##### 抽取結果")
                entity_table = gr.Dataframe(
                    headers=["實體", "類型", "說明", "來源 Chunks", "來源頁碼"],
                    interactive=False,
                    wrap=True,
                )
                relationship_table = gr.Dataframe(
                    headers=["來源實體", "關係", "目標實體", "說明", "來源 Chunks", "來源頁碼"],
                    interactive=False,
                    wrap=True,
                )

            with gr.Group():
                gr.Markdown("#### ③ Embedding 並匯入 Neo4j")
                gr.Markdown(
                    "確認上方抽取結果後，選擇 Embedding 模型並匯入 Neo4j。"
                )
                graph_embedding_model = gr.Dropdown(
                    choices=[env["EMBEDDING_MODEL"]],
                    value=env["EMBEDDING_MODEL"],
                    allow_custom_value=True,
                    label="Embedding 模型",
                )
                import_mode = gr.Radio(
                    ["保留既有圖譜", "取代最近一次圖譜", "清空本工具所有圖譜"],
                    value="保留既有圖譜",
                    label="匯入模式",
                )
                destructive_confirmed = gr.Checkbox(
                    value=False,
                    label="我確認取代或清空操作會刪除既有圖譜資料",
                )
                import_graph_button = gr.Button(
                    "Embedding 並匯入 Neo4j", variant="primary"
                )
                import_status = gr.Markdown("尚未執行 Embedding 與匯入。")

        with gr.Tab("4. 問答測試"):
            gr.Markdown(
                "直接使用連線設定中的 Neo4j；預設查詢最近更新的建圖結果。"
            )
            answer_model = gr.Textbox(label="問答 LLM", value=env["ANSWER_MODEL"])
            question = gr.Textbox(label="問題", placeholder="例如：設備出現 E01 時該如何處理？")
            with gr.Row():
                retrieval_mode = gr.Radio(["GraphRAG", "向量 RAG"], value="GraphRAG", label="檢索模式")
                top_k = gr.Slider(1, 50, value=8, step=1, label="Top K")
            ask_button = gr.Button("送出問題", variant="primary")
            answer_status = gr.Markdown()
            gr.HTML(
                """
                <style>
                .answer-panel {
                    border: 2px solid var(--border-color-primary);
                    border-radius: 12px;
                    padding: 18px 22px;
                    background: var(--background-fill-secondary);
                }
                .answer-content,
                .answer-content p,
                .answer-content li {
                    font-size: 20px !important;
                    line-height: 1.75 !important;
                }
                </style>
                """,
                padding=False,
            )
            with gr.Group(elem_classes="answer-panel"):
                gr.Markdown("### 回答")
                answer = gr.Markdown(elem_classes="answer-content")
            gr.Markdown("### 檢索來源")
            answer_sources = gr.Dataframe(
                headers=[
                    "類型", "證據", "Retriever", "官方混合分數",
                    "來源頁碼", "來源 Chunks",
                ],
                interactive=False,
                wrap=True,
            )

        with gr.Tab("5. 歷史紀錄"):
            gr.Markdown("目前專案的問答紀錄；成功問答後會自動追加並保存。")
            project_history_status = gr.Markdown()
            history_table = gr.Dataframe(
                headers=["時間", "問題", "回答", "模式", "文件"],
                interactive=False, wrap=True,
            )

        project_setting_inputs = [
            project_selector, pdf_file, preview_state, chunk_state, graph_state,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, start_page, end_page, chunk_size, chunk_overlap,
            graph_temperature, graph_max_output_tokens, schema_granularity,
            max_entity_types, max_relationship_types, max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, import_mode, retrieval_mode,
            top_k, schema_editor,
        ]
        project_load_outputs = [
            project_state, project_status, pdf_file,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, start_page, end_page, chunk_size, chunk_overlap,
            graph_temperature, graph_max_output_tokens, schema_granularity,
            max_entity_types, max_relationship_types, max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, import_mode, retrieval_mode,
            top_k, schema_editor, preview_state, chunk_state, graph_state,
            page_selector, chunk_table, page_status, history_table,
        ]
        project_tab.select(refresh_projects_for_ui, outputs=project_selector)
        create_project_button.click(
            create_project_for_ui, inputs=new_project_name,
            outputs=[project_selector, project_state, project_status],
        )
        load_project_button.click(
            load_project_for_ui, inputs=project_selector, outputs=project_load_outputs,
        )
        auto_save_components = [
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, start_page, end_page, chunk_size, chunk_overlap,
            graph_temperature, graph_max_output_tokens, schema_granularity,
            max_entity_types, max_relationship_types, max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, import_mode, retrieval_mode,
            top_k, schema_editor,
        ]
        for component in auto_save_components:
            component.input(
                save_project_for_ui, inputs=project_setting_inputs,
                outputs=[project_state, project_status], show_progress="hidden",
            )

        neo4j_test_button.click(
            check_neo4j_for_ui,
            inputs=[neo4j_uri, neo4j_database, neo4j_username, neo4j_password],
            outputs=neo4j_connection_status,
        )
        model_test_button.click(
            check_model_service_for_ui,
            inputs=[model_endpoint, api_key],
            outputs=model_connection_status,
        )
        env_inputs = [
            neo4j_uri,
            neo4j_database,
            neo4j_username,
            neo4j_password,
            model_endpoint,
            api_key,
            graph_llm_model,
            graph_embedding_model,
            answer_model,
        ]
        for component in env_inputs:
            component.change(persist_env_settings, inputs=env_inputs, outputs=env_status)
        reload_button.click(reload_env_settings, outputs=[*env_inputs, env_status])
        pdf_upload_event = pdf_file.upload(
            initialize_page_range,
            inputs=pdf_file,
            outputs=[start_page, end_page, preview_status],
        )
        pdf_upload_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        preview_event = preview_button.click(
            preview_pdf,
            inputs=[
                pdf_file,
                chunk_size,
                chunk_overlap,
                start_page,
                end_page,
            ],
            outputs=[
                preview_status,
                chunk_table,
                preview_state,
                chunk_state,
                page_selector,
                page_status,
            ],
        )
        preview_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        page_selector.change(
            preview_page,
            inputs=[page_selector, chunk_state, preview_state],
            outputs=[page_status, chunk_table],
        )
        previous_button.click(
            previous_page, inputs=[page_selector, preview_state], outputs=page_selector
        )
        next_button.click(next_page, inputs=[page_selector, preview_state], outputs=page_selector)
        export_button.click(save_config, inputs=[preview_state], outputs=[preview_status, export_file])
        schema_sampling_mode.change(
            lambda mode: gr.update(visible=mode == "隨機抽取 N 頁"),
            inputs=schema_sampling_mode,
            outputs=schema_sample_page_count,
        )
        plan_schema_button.click(
            plan_schema_for_ui,
            inputs=[
                model_endpoint,
                api_key,
                graph_llm_model,
                graph_temperature,
                graph_max_output_tokens,
                schema_granularity,
                max_entity_types,
                max_relationship_types,
                max_concurrent_requests,
                schema_sampling_mode,
                schema_sample_page_count,
                chunk_state,
            ],
            outputs=[plan_status, schema_editor],
        )
        extraction_event = generate_graph_button.click(
            extract_graph_for_ui,
            inputs=[
                model_endpoint,
                api_key,
                extraction_llm_model,
                graph_temperature,
                graph_max_output_tokens,
                extraction_max_concurrent_requests,
                chunk_state,
                schema_editor,
                preview_state,
            ],
            outputs=[build_status, entity_table, relationship_table, graph_state],
            show_progress="minimal",
        )
        extraction_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        import_event = import_graph_button.click(
            import_graph_for_ui,
            inputs=[
                model_endpoint,
                api_key,
                neo4j_uri,
                neo4j_database,
                neo4j_username,
                neo4j_password,
                graph_embedding_model,
                import_mode,
                destructive_confirmed,
                graph_state,
            ],
            outputs=[import_status, graph_state],
        )
        import_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        ask_button.click(
            answer_question_for_project_ui,
            inputs=[
                project_selector,
                model_endpoint,
                api_key,
                neo4j_uri,
                neo4j_database,
                neo4j_username,
                neo4j_password,
                answer_model,
                question,
                retrieval_mode,
                top_k,
            ],
            outputs=[answer_status, answer, answer_sources, history_table, project_history_status],
        )
    return app
