from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr

from .chunking import TextChunk, chunk_pages, preview_rows
from .config import (
    OPENAI_EMBEDDING_MODELS,
    OPENAI_LLM_MODELS,
    model_choices,
    public_settings,
)
from .env_store import load_env, save_env
from .evaluation_service import generate_evaluation_questions, judge_evaluation_answer
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
    vector_index_name,
)
from .pdf_service import extract_pdf
from .project_store import (
    append_question,
    create_project,
    delete_project,
    list_projects,
    load_project,
    remove_document,
    save_project,
)
from .qa_service import answer_graph_question, check_embedding_connection, embedding_vectors
from .storage import write_json


DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MAX_ENTITY_TYPES = 15
DEFAULT_MAX_RELATIONSHIP_TYPES = 20


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


def check_embedding_service_for_ui(base_url: str, api_key: str, model: str) -> str:
    try:
        check_embedding_connection(base_url, api_key, model)
    except ValueError as exc:
        return f"❌ {exc}"
    return "✅ Embedding 服務連線成功。"


def persist_env_settings(
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    model_endpoint: str,
    api_key: str,
    embedding_api_base: str,
    embedding_api_key: str,
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
            "EMBEDDING_API_BASE": embedding_api_base,
            "EMBEDDING_API_KEY": embedding_api_key,
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
        settings["EMBEDDING_API_BASE"],
        settings["EMBEDDING_API_KEY"],
        settings["BUILD_MODEL"],
        settings["EMBEDDING_MODEL"],
        settings["ANSWER_MODEL"],
        "✅ 已重新讀取 .env",
    )



def unlock_project_tabs_for_ui(project_id: str) -> tuple[dict[str, Any], ...]:
    enabled = bool(project_id)
    return tuple(gr.update(interactive=enabled) for _ in range(5))


def delete_project_for_ui(
    project_id: str,
) -> tuple[Any, ...]:
    try:
        name = delete_project(project_id)
    except (OSError, ValueError) as exc:
        return (
            gr.update(), gr.update(), f"❌ {exc}",
            *unlock_project_tabs_for_ui(project_id), False,
        )
    return (
        gr.update(choices=_project_choices(), value=None), {},
        f"✅ 已刪除專案「{name}」。", *unlock_project_tabs_for_ui(""), True,
    )


def refresh_projects_after_delete_for_ui(deleted: bool) -> dict[str, Any]:
    if not deleted:
        return gr.update()
    return gr.update(choices=_project_choices(), value=None)


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
    return [
        {"number": c.number, "text": c.text, "pages": list(c.pages), "document": c.document}
        for c in chunks
    ]


def _stored_chunks(items: list[dict[str, Any]]) -> list[TextChunk]:
    return [
        TextChunk(
            int(i["number"]), str(i["text"]), tuple(i.get("pages") or []),
            str(i.get("document", "")),
        )
        for i in items
    ]


def _document_rows(documents: list[dict[str, Any]]) -> list[list[object]]:
    return [
        [
            doc.get("file_name", ""),
            f"{doc.get('page_start', '')}–{doc.get('page_end', '')}",
            doc.get("chunk_count", 0),
        ]
        for doc in documents
    ]


def _document_choices(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return gr.update(
        choices=[doc.get("file_name", "") for doc in documents], value=[]
    )


def _chunk_rows(chunks: list[TextChunk]) -> list[list[object]]:
    return preview_rows(chunks, limit=len(chunks))


def _document_status(index: int, total: int, doc: dict[str, Any], chunk_count: int) -> str:
    if not total:
        return "尚未解析任何 PDF。"
    name = doc.get("file_name", "")
    page_range = f"{doc.get('page_start', '')}–{doc.get('page_end', '')}"
    return f"文件 {index + 1} / {total}：{name}（第 {page_range} 頁），共 {chunk_count} 個 chunk。"


def _history_rows(questions: list[dict[str, Any]]) -> list[list[object]]:
    return [[i.get("asked_at", ""), i.get("question", ""), i.get("answer", ""),
             i.get("retrieval_mode", ""), i.get("document", "")] for i in reversed(questions)]


def _display_retrieval_mode(value: str | None) -> str:
    return {
        "GraphRAG": "關聯擴展檢索",
        "向量 RAG": "基本檢索",
    }.get(value or "", value or "關聯擴展檢索")


def _graph_rows(graph: dict[str, Any]) -> tuple[list[list[object]], list[list[object]]]:
    entities = [[
        item.get("name", ""), item.get("type", ""), item.get("description", ""),
        ", ".join(map(str, item.get("source_chunk_numbers", []))),
        ", ".join(map(str, item.get("source_pages", []))),
        "、".join(item.get("source_documents", [])),
    ] for item in graph.get("entities", [])]
    relationships = [[
        item.get("source", ""), item.get("type", ""), item.get("target", ""),
        item.get("description", ""),
        ", ".join(map(str, item.get("source_chunk_numbers", []))),
        ", ".join(map(str, item.get("source_pages", []))),
        "、".join(item.get("source_documents", [])),
    ] for item in graph.get("relationships", [])]
    return entities, relationships


def save_project_for_ui(
    project_id: str, documents: list[dict[str, Any]],
    chunks: list[TextChunk], graph_state: dict[str, Any],
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    model_endpoint: str, api_key: str, graph_llm_model: str,
    graph_embedding_model: str, answer_model: str,
    chunk_size: int, chunk_overlap: int,
    graph_temperature: float,
    schema_granularity: str,
    max_concurrent_requests: int, schema_sampling_mode: str,
    schema_sample_page_count: int, extraction_llm_model: str,
    extraction_max_concurrent_requests: int,
    retrieval_mode: str, top_k: int, schema_text: str,
) -> tuple[dict[str, Any], str]:
    if not project_id:
        return {}, "❌ 請先建立或載入專案。"
    settings = {
        "neo4j_uri": neo4j_uri, "neo4j_database": neo4j_database,
        "neo4j_username": neo4j_username, "neo4j_password": neo4j_password,
        "model_endpoint": model_endpoint, "api_key": api_key,
        "graph_llm_model": graph_llm_model, "graph_embedding_model": graph_embedding_model,
        "answer_model": answer_model,
        "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
        "graph_temperature": float(graph_temperature),
        "schema_granularity": schema_granularity,
        "max_concurrent_requests": int(max_concurrent_requests),
        "schema_sampling_mode": schema_sampling_mode,
        "schema_sample_page_count": int(schema_sample_page_count),
        "extraction_llm_model": extraction_llm_model,
        "extraction_max_concurrent_requests": int(extraction_max_concurrent_requests),
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k), "schema_text": schema_text or "",
    }
    documents = documents or []
    try:
        project = save_project(project_id, {
            "settings": settings, "documents_meta": documents,
            "chunks": _chunk_dicts(chunks or []), "graph_state": graph_state or {},
        }, [doc["file_path"] for doc in documents if doc.get("file_path")])
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
    documents = project.get("documents_meta") or []
    chunks = _stored_chunks(project.get("chunks") or [])
    graph = project.get("graph_state") or {}
    active_preview = documents[-1] if documents else {}
    active_chunks = [
        chunk for chunk in chunks if chunk.document == active_preview.get("file_name")
    ] if active_preview else []
    document_status = (
        _document_status(len(documents) - 1, len(documents), active_preview, len(active_chunks))
        if documents else "請先解析 PDF。"
    )
    entity_rows, relationship_rows = _graph_rows(graph)
    if graph.get("run_id"):
        graph_status = (
            f"✅ 已載入專案保存的圖譜：{len(entity_rows)} 個實體、"
            f"{len(relationship_rows)} 筆關係。"
        )
        import_status = (
            "✅ 此圖譜已匯入 Neo4j。" if graph.get("neo4j_imported")
            else "此圖譜尚未匯入 Neo4j。"
        )
    else:
        graph_status, import_status = "尚未執行抽取。", "尚未執行 Embedding 與匯入。"
    return (
        project, f"✅ 已載入專案「{project['name']}」。",
        get("neo4j_uri", env["NEO4J_URI"]), get("neo4j_database", env["NEO4J_DATABASE"]),
        get("neo4j_username", env["NEO4J_USERNAME"]), get("neo4j_password", env["NEO4J_PASSWORD"]),
        get("model_endpoint", env["MODEL_API_BASE"]), get("api_key", env["MODEL_API_KEY"]),
        get("graph_llm_model", env["BUILD_MODEL"]), get("graph_embedding_model", env["EMBEDDING_MODEL"]),
        get("answer_model", env["ANSWER_MODEL"]),
        get("chunk_size", 1500), get("chunk_overlap", 200), get("graph_temperature", 0),
        get("schema_granularity", "平衡"),
        get("max_concurrent_requests", 3),
        gr.update(
            value=get("schema_sampling_mode", "全部頁面"),
            visible=get("schema_sampling_mode", "全部頁面") == "隨機抽取 N 頁",
        ),
        get("schema_sample_page_count", 10), get("extraction_llm_model", env["BUILD_MODEL"]),
        get("extraction_max_concurrent_requests", 3),
        _display_retrieval_mode(get("retrieval_mode")), get("top_k", 8), get("schema_text", ""),
        documents, chunks, graph, active_preview, active_chunks,
        _document_rows(documents), _document_choices(documents),
        _chunk_rows(active_chunks), document_status,
        _history_rows(project.get("questions") or []),
        entity_rows, relationship_rows, graph_status, import_status,
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


def _evaluation_question_rows(questions: list[dict[str, Any]]) -> list[list[object]]:
    return [[item["number"], item["question"], item["expected_answer"],
             ", ".join(map(str, item.get("source_pages", []))),
             item.get("document", "")] for item in questions]


def _questions_from_rows(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    if not isinstance(rows, list) or not rows:
        raise ValueError("題目不可為空")
    questions = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError(f"第 {index} 列格式不正確")
        question = str(row[1] or "").strip()
        answer = str(row[2] or "").strip()
        if not question or not answer:
            raise ValueError(f"第 {index} 題的問題與標準答案不可為空")
        raw_pages = row[3] if len(row) > 3 else ""
        if isinstance(raw_pages, (list, tuple)):
            page_values = raw_pages
        else:
            page_values = str(raw_pages or "").replace("，", ",").split(",")
        try:
            pages = [int(value) for value in page_values if str(value).strip()]
        except ValueError as exc:
            raise ValueError(f"第 {index} 題的來源頁碼必須是逗號分隔的整數") from exc
        document = str(row[4]).strip() if len(row) > 4 and row[4] is not None else ""
        questions.append({
            "number": index, "question": question,
            "expected_answer": answer, "source_pages": pages,
            "document": document,
        })
    return questions


def save_evaluation_questions_for_ui(
    project_id: str, rows: Any, evaluation: dict[str, Any]
) -> tuple[str, dict[str, Any], list[list[object]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", evaluation or {}, []
    try:
        questions = _questions_from_rows(rows)
        updated = dict(evaluation or {})
        updated.update({"questions": questions, "results": [], "dirty": False})
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        failed = dict(evaluation or {})
        if "questions" in locals():
            failed.update({"questions": questions, "results": [], "dirty": True})
        return f"❌ {exc}；自動儲存失敗。", failed, []
    return f"✅ 已自動儲存 {len(questions)} 道題目。", updated, []


def import_evaluation_questions_for_ui(
    project_id: str, file_path: str | None, evaluation: dict[str, Any]
) -> tuple[str, list[list[object]], dict[str, Any], list[list[object]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", [], evaluation or {}, []
    if not file_path:
        return "❌ 請選擇 JSON 或 CSV 題目檔。", [], evaluation or {}, []
    try:
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload.get("questions") if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                raise ValueError("JSON 必須是題目陣列或包含 questions 陣列")
            rows = [[item.get("number", index), item.get("question", ""),
                     item.get("expected_answer", ""), item.get("source_pages", []),
                     item.get("document", "")]
                    for index, item in enumerate(items, start=1) if isinstance(item, dict)]
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                items = list(csv.DictReader(handle))
            rows = [[item.get("number", index), item.get("question", ""),
                     item.get("expected_answer", ""), item.get("source_pages", ""),
                     item.get("document", "")]
                    for index, item in enumerate(items, start=1)]
        else:
            raise ValueError("只支援 .json 或 .csv 題目檔")
        questions = _questions_from_rows(rows)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return f"❌ 匯入失敗：{exc}", [], evaluation or {}, []
    updated = dict(evaluation or {})
    updated.update({"questions": questions, "results": [], "dirty": False})
    try:
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        updated["dirty"] = True
        return f"❌ 匯入成功但自動儲存失敗：{exc}", _evaluation_question_rows(questions), updated, []
    return (f"✅ 已匯入並自動儲存 {len(questions)} 道題目。",
            _evaluation_question_rows(questions), updated, [])


def export_evaluation_questions_for_ui(
    project_id: str, rows: Any
) -> tuple[str, str | None]:
    if not project_id:
        return "❌ 請先建立或載入專案。", None
    try:
        questions = _questions_from_rows(rows)
        output = write_json(
            Path("data/projects") / project_id / "exports" / "questions.json",
            {"questions": questions},
        )
    except (OSError, ValueError) as exc:
        return f"❌ 匯出失敗：{exc}", None
    return f"✅ 已匯出 {len(questions)} 道題目。", str(output)


def _evaluation_result_rows(results: list[dict[str, Any]]) -> list[list[object]]:
    return [[item["number"], item["question"], item["expected_answer"],
             item.get("actual_answer", ""), "✅ 通過" if item.get("passed") else "❌ 未通過",
             item.get("reason", "")] for item in results]


def load_evaluation_for_ui(project_id: str) -> tuple[Any, ...]:
    if not project_id:
        return {}, [], [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "請先選擇專案。"
    try:
        project = load_project(project_id)
    except (OSError, ValueError) as exc:
        return {}, [], [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"❌ {exc}"
    evaluation = dict(project.get("evaluation") or {})
    evaluation.setdefault("dirty", False)
    preferences = evaluation.get("preferences") or {}
    questions = evaluation.get("questions") or []
    results = evaluation.get("results") or []
    env = load_env()
    legacy_model = preferences.get("model", env["ANSWER_MODEL"])
    return (
        evaluation, _evaluation_question_rows(questions), _evaluation_result_rows(results),
        preferences.get("generation_model", legacy_model),
        preferences.get("test_model", legacy_model), preferences.get("question_count", 10),
        _display_retrieval_mode(preferences.get("retrieval_mode")), preferences.get("top_k", 8),
        f"已載入 {len(questions)} 道題目與 {len(results)} 筆測試結果。",
    )


def save_evaluation_preferences_for_ui(
    project_id: str, generation_model: str, test_model: str, question_count: int,
    retrieval_mode: str, top_k: int,
) -> str:
    if not project_id:
        return "⚠️ 請先選擇專案。"
    try:
        project = load_project(project_id)
        evaluation = dict(project.get("evaluation") or {})
        evaluation["preferences"] = {
            "generation_model": generation_model, "test_model": test_model,
            "question_count": int(question_count),
            "retrieval_mode": retrieval_mode, "top_k": int(top_k),
        }
        save_project(project_id, {"evaluation": evaluation})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 自動保存失敗：{exc}"
    return "✅ 自動測試設定已保存。"


def generate_evaluation_for_ui(
    project_id: str, model_endpoint: str, api_key: str, generation_model: str,
    test_model: str, question_count: int, retrieval_mode: str, top_k: int,
    chunks: list[TextChunk],
) -> tuple[str, list[list[object]], dict[str, Any], list[list[object]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", [], {}, []
    try:
        questions = generate_evaluation_questions(
            model_endpoint, api_key, generation_model, chunks, int(question_count)
        )
        evaluation = {
            "preferences": {"generation_model": generation_model, "test_model": test_model,
                            "question_count": int(question_count),
                            "retrieval_mode": retrieval_mode, "top_k": int(top_k)},
            "questions": questions, "results": [], "dirty": False,
        }
        save_project(project_id, {"evaluation": evaluation})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ {exc}", [], {}, []
    return f"✅ 已從 PDF 建立 {len(questions)} 道題目與標準答案。", _evaluation_question_rows(questions), evaluation, []


def run_evaluation_for_ui(
    project_id: str, model_endpoint: str, api_key: str,
    embedding_api_base: str, embedding_api_key: str,
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    model: str, retrieval_mode: str, top_k: int, evaluation: dict[str, Any],
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], dict[str, Any]]:
    questions = evaluation.get("questions") if evaluation else None
    if not project_id:
        return "❌ 請先建立或載入專案。", [], evaluation or {}
    if not questions:
        return "❌ 請先建立測試題目。", [], evaluation or {}
    if evaluation.get("dirty"):
        return "❌ 題目或答案尚未完成自動儲存，請稍後再試。", [], evaluation
    results = []
    for index, item in enumerate(questions, start=1):
        progress((index - 1) / len(questions), desc=f"測試第 {index} / {len(questions)} 題")
        status, actual, _ = answer_question_for_ui(
            model_endpoint, api_key, embedding_api_base, embedding_api_key,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model, item["question"], retrieval_mode, int(top_k),
        )
        if status.startswith("✅"):
            try:
                judgment = judge_evaluation_answer(
                    model_endpoint, api_key, model, item["question"],
                    item["expected_answer"], actual,
                )
            except ValueError as exc:
                judgment = {"passed": False, "reason": f"評判失敗：{exc}"}
        else:
            judgment = {"passed": False, "reason": status}
        results.append({**item, "actual_answer": actual, **judgment})
    updated = dict(evaluation)
    updated["results"] = results
    try:
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        return f"❌ 測試已完成，但保存失敗：{exc}", _evaluation_result_rows(results), updated
    passed = sum(bool(item["passed"]) for item in results)
    total = len(results)
    failed = total - passed
    accuracy = passed / total * 100 if total else 0
    summary = (
        f"## 測試完成｜答對 {passed} 題 / 共 {total} 題  "
        f"\n答錯：{failed} 題｜正確率：{accuracy:.1f}%"
    )
    return summary, _evaluation_result_rows(results), updated

def _add_single_document(
    file_path: str,
    parsed_chunk_size: int,
    parsed_chunk_overlap: int,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
) -> tuple[str, dict[str, Any] | None, list[TextChunk]]:
    file_name = Path(file_path).name
    if any(doc.get("file_name") == file_name for doc in documents):
        return f"❌「{file_name}」：專案中已有同名文件，請先移除或重新命名後再上傳。", None, []
    try:
        pages, empty_pages = extract_pdf(file_path)
        next_number = max((chunk.number for chunk in chunks), default=0) + 1
        new_chunks = chunk_pages(
            pages, parsed_chunk_size, parsed_chunk_overlap,
            document=file_name, start_number=next_number,
        )
        if not new_chunks:
            raise ValueError("PDF 沒有可解析文字；掃描文件需在後續版本加入 OCR。")
    except (ValueError, TypeError) as exc:
        return f"❌「{file_name}」：{exc}", None, []
    parsed_start, parsed_end = pages[0].page, pages[-1].page
    doc_state = {
        "file_path": file_path,
        "file_name": file_name,
        "page_count": len(pages),
        "page_start": parsed_start,
        "page_end": parsed_end,
        "empty_pages": empty_pages,
        "chunk_count": len(new_chunks),
        "config": {"chunk_size": parsed_chunk_size, "chunk_overlap": parsed_chunk_overlap},
    }
    note = f"✅「{file_name}」第 {parsed_start}–{parsed_end} 頁，產生 {len(new_chunks)} 個 chunk。"
    if empty_pages:
        note += f" 無文字頁面：{', '.join(map(str, empty_pages))}。"
    return note, doc_state, new_chunks


def add_document_for_ui(
    file_paths: list[str] | str | None,
    chunk_size: int,
    chunk_overlap: int,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
) -> tuple[
    str, list[list[object]], list[dict[str, Any]], list[TextChunk],
    dict[str, Any], list[TextChunk], str,
    list[list[object]], dict[str, Any], dict[str, Any],
]:
    documents = list(documents or [])
    chunks = list(chunks or [])
    if not file_paths:
        return (
            "請先上傳 PDF。", [], documents, chunks, {}, [],
            "尚未解析任何 PDF。", _document_rows(documents), gr.update(),
            _document_choices(documents),
        )
    try:
        parsed_chunk_size = int(chunk_size)
        parsed_chunk_overlap = int(chunk_overlap)
        if not 100 <= parsed_chunk_size <= 10000:
            raise ValueError("chunk_size 必須介於 100 到 10,000")
        if not 0 <= parsed_chunk_overlap < parsed_chunk_size:
            raise ValueError("chunk_overlap 必須大於等於 0 且小於 chunk_size")
    except (ValueError, TypeError) as exc:
        return (
            f"❌ {exc}", [], documents, chunks, {}, [],
            "無法解析頁面。", _document_rows(documents), gr.update(),
            _document_choices(documents),
        )

    paths = file_paths if isinstance(file_paths, list) else [file_paths]
    notes: list[str] = []
    last_doc_state: dict[str, Any] = {}
    last_new_chunks: list[TextChunk] = []
    for file_path in paths:
        note, doc_state, new_chunks = _add_single_document(
            file_path, parsed_chunk_size, parsed_chunk_overlap, documents, chunks,
        )
        notes.append(note)
        if doc_state is not None:
            documents = documents + [doc_state]
            chunks = chunks + new_chunks
            last_doc_state, last_new_chunks = doc_state, new_chunks

    summary = f"（專案累計 {len(chunks)} 個 chunk，{len(documents)} 份文件）"
    if len(notes) > 1:
        status = "\n".join(f"- {note}" for note in notes) + f"\n\n{summary}"
    else:
        status = notes[0] + summary
    if not last_doc_state:
        return (
            status, [], documents, chunks, {}, [],
            "無法解析頁面。", _document_rows(documents), gr.update(),
            _document_choices(documents),
        )
    return (
        status,
        _chunk_rows(last_new_chunks),
        documents,
        chunks,
        last_doc_state,
        last_new_chunks,
        _document_status(len(documents) - 1, len(documents), last_doc_state, len(last_new_chunks)),
        _document_rows(documents),
        gr.update(value=None),
        _document_choices(documents),
    )


def remove_document_for_ui(
    project_id: str,
    file_names: list[str] | str | None,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
) -> tuple[
    str, list[dict[str, Any]], list[TextChunk], dict[str, Any], list[TextChunk],
    str, list[list[object]], dict[str, Any], list[list[object]],
]:
    documents = list(documents or [])
    chunks = list(chunks or [])
    names = [
        name for name in (file_names if isinstance(file_names, list) else [file_names])
        if name
    ]
    if not names:
        return (
            "請先選擇要移除的 PDF。", documents, chunks, {}, [],
            "請先解析 PDF。", _document_rows(documents),
            _document_choices(documents), [],
        )
    if project_id:
        for file_name in names:
            try:
                remove_document(project_id, file_name)
            except (OSError, ValueError) as exc:
                return (
                    f"❌ {exc}", documents, chunks, {}, [],
                    "請先解析 PDF。", _document_rows(documents),
                    _document_choices(documents), [],
                )
    name_set = set(names)
    documents = [doc for doc in documents if doc.get("file_name") not in name_set]
    chunks = [chunk for chunk in chunks if chunk.document not in name_set]
    active_preview = documents[-1] if documents else {}
    active_chunks = [
        chunk for chunk in chunks if chunk.document == active_preview.get("file_name")
    ] if active_preview else []
    status = (
        f"✅ 已移除 {len(names)} 份文件「{'、'.join(names)}」"
        f"（專案剩餘 {len(chunks)} 個 chunk、{len(documents)} 份文件）。"
    )
    document_status = (
        _document_status(len(documents) - 1, len(documents), active_preview, len(active_chunks))
        if documents else "請先解析 PDF。"
    )
    return (
        status,
        documents,
        chunks,
        active_preview,
        active_chunks,
        document_status,
        _document_rows(documents),
        _document_choices(documents),
        _chunk_rows(active_chunks),
    )


def switch_document(
    offset: int,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
    active: dict[str, Any],
) -> tuple[dict[str, Any], list[TextChunk], list[list[object]], str]:
    documents = documents or []
    if not documents:
        return {}, [], [], "尚未解析任何 PDF。"
    names = [doc.get("file_name", "") for doc in documents]
    current_name = (active or {}).get("file_name", "")
    current_index = names.index(current_name) if current_name in names else len(documents) - 1
    new_index = (current_index + offset) % len(documents)
    doc = documents[new_index]
    doc_chunks = [chunk for chunk in (chunks or []) if chunk.document == doc.get("file_name")]
    return (
        doc, doc_chunks, _chunk_rows(doc_chunks),
        _document_status(new_index, len(documents), doc, len(doc_chunks)),
    )


def previous_document(
    documents: list[dict[str, Any]], chunks: list[TextChunk], active: dict[str, Any]
) -> tuple[dict[str, Any], list[TextChunk], list[list[object]], str]:
    return switch_document(-1, documents, chunks, active)


def next_document(
    documents: list[dict[str, Any]], chunks: list[TextChunk], active: dict[str, Any]
) -> tuple[dict[str, Any], list[TextChunk], list[list[object]], str]:
    return switch_document(1, documents, chunks, active)


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
    schema_granularity: str,
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
            DEFAULT_MAX_OUTPUT_TOKENS,
            lambda value, description: progress(value, desc=description),
            schema_granularity,
            DEFAULT_MAX_ENTITY_TYPES,
            DEFAULT_MAX_RELATIONSHIP_TYPES,
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
        f"{DEFAULT_MAX_ENTITY_TYPES}／{DEFAULT_MAX_RELATIONSHIP_TYPES}（固定）。"
        f"最大並行請求數：{int(max_concurrent_requests)}。"
        "請確認或編輯後再進行抽取。"
    )
    return note, json.dumps(plan.schema, ensure_ascii=False, indent=2)


def extract_graph_for_ui(
    model_endpoint: str,
    api_key: str,
    llm_model: str,
    temperature: float,
    max_concurrent_requests: int,
    chunks: list[TextChunk],
    schema_text: str,
    documents: list[dict[str, Any]],
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
            DEFAULT_MAX_OUTPUT_TOKENS,
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
            "、".join(item.get("source_documents", [])),
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
            "、".join(item.get("source_documents", [])),
        ]
        for item in extraction.relationships
    ]
    run_id = str(uuid4())
    document_name = "、".join(doc.get("file_name", "") for doc in (documents or []))
    graph_state = {
        "run_id": run_id,
        "document": document_name,
        "llm_model": llm_model,
        "temperature": float(temperature),
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "max_concurrent_requests": int(max_concurrent_requests),
        "schema": schema,
        "entities": extraction.entities,
        "relationships": extraction.relationships,
        "chunks": [
            {
                "number": chunk.number, "text": chunk.text,
                "pages": list(chunk.pages), "document": chunk.document,
            }
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
        document = str(item.get("document", ""))
        evidence.append({
            "evidence_id": f"chunk-{number}", "kind": "原文",
            "name": "", "source": "", "target": "",
            "text": str(item.get("text", "")),
            "source_pages": item.get("pages", []),
            "source_chunk_numbers": [number],
            "source_documents": [document] if document else [],
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
            "source_documents": item.get("source_documents", []),
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
            "source_documents": item.get("source_documents", []),
        })
    return evidence


def import_graph_for_ui(
    embedding_api_base: str,
    embedding_api_key: str,
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    embedding_model: str,
    graph_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not graph_state or not graph_state.get("run_id"):
        return "❌ 請先完成知識圖譜抽取。", graph_state or {}
    if not embedding_model.strip():
        return "❌ 請選擇 Embedding 模型。", graph_state
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
            embedding_api_base, embedding_api_key, embedding_model,
            [item["text"] for item in evidence],
        )
        for item, vector in zip(evidence, vectors):
            item["embedding"] = vector
        updated_state["embedding_dimensions"] = len(vectors[0])
        updated_state["vector_index_name"] = vector_index_name(len(vectors[0]))
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
        )
    except (KeyError, ValueError) as exc:
        updated_state["neo4j_imported"] = False
        updated_state["neo4j_error"] = str(exc)
        return f"❌ {exc}", updated_state

    updated_state["neo4j_imported"] = True
    updated_state.pop("neo4j_error", None)
    return (
        f"✅ 已清空本工具既有圖譜；建立 {len(evidence)} 筆向量證據、Vector Index 與 Full-text Index；已匯入 Neo4j {imported.entity_count} 個實體與 "
        f"{imported.relationship_count} 筆關係。",
        updated_state,
    )


def answer_question_for_ui(
    model_endpoint: str,
    api_key: str,
    embedding_api_base: str,
    embedding_api_key: str,
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
            embedding_api_base, embedding_api_key, graph_state.get("embedding_model", ""),
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
            "、".join(item.get("source_documents") or []),
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
        documents_state = gr.State([])
        active_preview_state = gr.State({})
        active_chunks_state = gr.State([])
        chunk_state = gr.State([])
        graph_state = gr.State({})
        evaluation_state = gr.State({})

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
                delete_project_button = gr.Button("刪除專案", variant="stop")
                delete_project_completed = gr.State(False)
            project_status = gr.Markdown("尚未選擇專案；載入後，設定與處理結果都會自動保存。")
            gr.Markdown("⚠️ 專案設定保存在本機 `data/projects/`，其中 Password 與 API Key 為明文；請勿分享或提交該目錄。")

        with gr.Tab("1. 連線設定") as connection_tab:
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
                    gr.Markdown("### 模型服務（對話／建圖用，預設使用 OpenAI）")
                    model_endpoint = gr.Textbox(label="API Base URL", value=env["MODEL_API_BASE"])
                    api_key = gr.Textbox(label="API Key", value=env["MODEL_API_KEY"], type="password")
                    model_test_button = gr.Button("測試模型服務連線", variant="primary")
                    model_connection_status = gr.Markdown()
                    gr.Markdown("### Embedding 服務（預設使用 OpenAI）")
                    embedding_api_base = gr.Textbox(
                        label="Embedding API Base URL", value=env["EMBEDDING_API_BASE"]
                    )
                    embedding_api_key = gr.Textbox(
                        label="Embedding API Key", value=env["EMBEDDING_API_KEY"], type="password"
                    )
                    embedding_test_model = gr.Textbox(
                        label="測試用 Embedding 模型名稱", value=env["EMBEDDING_MODEL"]
                    )
                    embedding_test_button = gr.Button("測試 Embedding 服務連線", variant="primary")
                    embedding_connection_status = gr.Markdown()
                    reload_button = gr.Button("重新讀取 .env")
            gr.Markdown("⚠️ Password 與 API Key 會以明文寫入本機 `.env`；請勿提交此檔案。")
            env_status = gr.Markdown("啟動時已讀取 .env；欄位修改後會自動儲存。")

        with gr.Tab("2. PDF 與參數", interactive=False) as pdf_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_file = gr.File(
                        label="PDF 使用手冊（可一次選取多個檔案）",
                        file_types=[".pdf"], file_count="multiple", type="filepath",
                    )
                    chunk_size = gr.Slider(100, 10000, value=1500, step=100, label="Chunk size（字元）")
                    chunk_overlap = gr.Slider(0, 2000, value=200, step=50, label="Chunk overlap（字元）")
                    preview_button = gr.Button("解析並加入專案", variant="primary")
                    export_button = gr.Button("匯出目前設定")
                    export_file = gr.File(label="設定 JSON", interactive=False)
                    gr.Markdown("##### 已加入本專案的 PDF")
                    documents_table = gr.Dataframe(
                        headers=["文件", "頁碼範圍", "Chunk 數"],
                        datatype=["str", "str", "number"],
                        interactive=False,
                        wrap=True,
                    )
                    remove_document_selector = gr.Dropdown(
                        label="選擇要移除的 PDF（可多選）", choices=[],
                        multiselect=True, interactive=True,
                    )
                    remove_document_button = gr.Button("移除選定的 PDF", variant="stop")
                with gr.Column(scale=3):
                    preview_status = gr.Markdown("尚未解析 PDF。可重複上傳多份 PDF，逐一加入同一個專案。")
                    with gr.Row():
                        previous_button = gr.Button("上一份文件", scale=1)
                        next_button = gr.Button("下一份文件", scale=1)
                    page_status = gr.Markdown("請先解析 PDF。")
                    chunk_table = gr.Dataframe(
                        headers=["編號", "文件", "頁碼", "字元數", "內容"],
                        datatype=["number", "str", "str", "number", "str"],
                        interactive=False,
                        wrap=True,
                        max_height=750,
                        column_widths=[80, 160, 120, 100, 900],
                    )

        with gr.Tab("3. 建圖", interactive=False) as graph_tab:
            gr.Markdown("### 規劃並抽取知識圖譜")
            with gr.Group():
                gr.Markdown("#### ① 規劃 Schema")
                gr.Markdown(
                    "選擇 LLM，從上一頁產生的 chunks 規劃實體與關係類型。"
                )
                with gr.Row():
                    graph_llm_model = gr.Dropdown(
                        choices=model_choices(
                            env["BUILD_MODEL"], env["ANSWER_MODEL"],
                            defaults=OPENAI_LLM_MODELS,
                        ),
                        value=env["BUILD_MODEL"],
                        allow_custom_value=True,
                        label="Schema 規劃 LLM",
                    )
                    graph_temperature = gr.Slider(
                        0, 2, value=0, step=0.1, label="Temperature"
                    )
                with gr.Row():
                    schema_granularity = gr.Radio(
                        ["粗略", "平衡", "詳細"],
                        value="平衡",
                        label="Schema 粒度",
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
                    choices=model_choices(
                        env["BUILD_MODEL"], env["ANSWER_MODEL"],
                        defaults=OPENAI_LLM_MODELS,
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
                    headers=["實體", "類型", "說明", "來源 Chunks", "來源頁碼", "來源文件"],
                    interactive=False,
                    wrap=True,
                )
                relationship_table = gr.Dataframe(
                    headers=["來源實體", "關係", "目標實體", "說明", "來源 Chunks", "來源頁碼", "來源文件"],
                    interactive=False,
                    wrap=True,
                )

            with gr.Group():
                gr.Markdown("#### ③ Embedding 並匯入 Neo4j")
                gr.Markdown(
                    "確認上方抽取結果後，選擇 Embedding 模型並匯入 Neo4j。"
                )
                graph_embedding_model = gr.Dropdown(
                    choices=model_choices(
                        env["EMBEDDING_MODEL"], defaults=OPENAI_EMBEDDING_MODELS
                    ),
                    value=env["EMBEDDING_MODEL"],
                    allow_custom_value=True,
                    label="Embedding 模型",
                )
                gr.Markdown(
                    "⚠️ 每次匯入都會先清空本工具在目前 Neo4j Database 中建立的圖譜，再寫入本次結果。"
                )
                import_graph_button = gr.Button(
                    "Embedding 並匯入 Neo4j", variant="primary"
                )
                import_status = gr.Markdown("尚未執行 Embedding 與匯入。")

        with gr.Tab("4. 自動問答測試", interactive=False) as evaluation_tab:
            gr.Markdown(
                "### 從 PDF 自動建立問答測試集\n"
                "先建立指定數量的題目與標準答案，再一鍵執行目前的 RAG 並由模型判斷答案是否正確。"
            )
            with gr.Group():
                gr.Markdown("#### 生題設定")
                with gr.Row():
                    evaluation_generation_model = gr.Dropdown(
                        choices=model_choices(
                            env["ANSWER_MODEL"], env["BUILD_MODEL"],
                            defaults=OPENAI_LLM_MODELS,
                        ),
                        value=env["ANSWER_MODEL"],
                        allow_custom_value=True,
                        label="生題模型",
                    )
                    evaluation_question_count = gr.Number(value=10, minimum=1, maximum=100, precision=0, label="題目數量 N")
                generate_evaluation_button = gr.Button("從 PDF 建立題目與答案", variant="primary")
            with gr.Group():
                gr.Markdown("#### 測試模型設定")
                with gr.Row():
                    evaluation_test_model = gr.Dropdown(
                        choices=model_choices(
                            env["ANSWER_MODEL"], env["BUILD_MODEL"],
                            defaults=OPENAI_LLM_MODELS,
                        ),
                        value=env["ANSWER_MODEL"],
                        allow_custom_value=True,
                        label="回答與評判模型",
                    )
                    evaluation_retrieval_mode = gr.Radio(["基本檢索", "關聯擴展檢索"], value="關聯擴展檢索", label="檢索模式")
                    evaluation_top_k = gr.Slider(1, 50, value=8, step=1, label="Top K")
                run_evaluation_button = gr.Button("一鍵測試", variant="primary")
            with gr.Row():
                evaluation_import_file = gr.File(
                    label="匯入題目（JSON／CSV）", file_types=[".json", ".csv"], type="filepath"
                )
                import_evaluation_button = gr.Button("匯入題目")
                export_evaluation_button = gr.Button("匯出題目")
                evaluation_export_file = gr.File(label="題目 JSON", interactive=False)
            gr.HTML(
                """<style>
                .evaluation-metrics-box {
                    border: 2px solid var(--border-color-primary) !important;
                    border-radius: 12px !important;
                    padding: 16px 22px !important;
                    margin: 18px 0 12px !important;
                    background: var(--background-fill-secondary) !important;
                }
                .evaluation-metrics {font-size: 24px !important; line-height: 1.7 !important;}
                .evaluation-table table {font-size: 18px !important;}
                .evaluation-table td, .evaluation-table th {padding: 10px !important;}
                </style>""",
                padding=False,
            )
            gr.Markdown("#### 測試題目")
            evaluation_questions_table = gr.Dataframe(
                headers=["編號", "問題", "標準答案", "來源頁碼", "來源文件"],
                datatype=["number", "str", "str", "str", "str"],
                type="array", interactive=True, wrap=True,
                elem_classes="evaluation-table",
            )
            with gr.Group(elem_classes="evaluation-metrics-box"):
                evaluation_status = gr.Markdown(
                    "請先載入專案並解析 PDF。", elem_classes="evaluation-metrics"
                )
            gr.Markdown("#### 測試結果")
            evaluation_results_table = gr.Dataframe(
                headers=["編號", "問題", "標準答案", "實際答案", "結果", "評判理由"],
                interactive=False, wrap=True, elem_classes="evaluation-table",
            )

        with gr.Tab("5. 問答測試", interactive=False) as qa_tab:
            gr.Markdown(
                "直接使用連線設定中的 Neo4j；預設查詢最近更新的建圖結果。"
            )
            answer_model = gr.Dropdown(
                choices=model_choices(
                    env["ANSWER_MODEL"], env["BUILD_MODEL"],
                    defaults=OPENAI_LLM_MODELS,
                ),
                value=env["ANSWER_MODEL"],
                allow_custom_value=True,
                label="問答 LLM",
            )
            question = gr.Textbox(label="問題", placeholder="例如：設備出現 E01 時該如何處理？")
            with gr.Row():
                retrieval_mode = gr.Radio(["基本檢索", "關聯擴展檢索"], value="關聯擴展檢索", label="檢索模式")
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
                    "來源頁碼", "來源 Chunks", "來源文件",
                ],
                interactive=False,
                wrap=True,
            )

        with gr.Tab("6. 歷史紀錄", interactive=False) as history_tab:
            gr.Markdown("目前專案的問答紀錄；成功問答後會自動追加並保存。")
            project_history_status = gr.Markdown()
            history_table = gr.Dataframe(
                headers=["時間", "問題", "回答", "模式", "文件"],
                interactive=False, wrap=True,
            )

        evaluation_tab.select(
            load_evaluation_for_ui, inputs=project_selector,
            outputs=[evaluation_state, evaluation_questions_table, evaluation_results_table,
                     evaluation_generation_model, evaluation_test_model, evaluation_question_count,
                     evaluation_retrieval_mode, evaluation_top_k, evaluation_status],
        )
        evaluation_preference_inputs = [
            project_selector, evaluation_generation_model, evaluation_test_model, evaluation_question_count,
            evaluation_retrieval_mode, evaluation_top_k,
        ]
        for component in [evaluation_generation_model, evaluation_test_model, evaluation_question_count,
                          evaluation_retrieval_mode, evaluation_top_k]:
            component.input(
                save_evaluation_preferences_for_ui,
                inputs=evaluation_preference_inputs, outputs=evaluation_status,
                show_progress="hidden",
            )
        evaluation_questions_table.input(
            save_evaluation_questions_for_ui,
            inputs=[project_selector, evaluation_questions_table, evaluation_state],
            outputs=[evaluation_status, evaluation_state, evaluation_results_table],
            show_progress="hidden",
        )
        import_evaluation_button.click(
            import_evaluation_questions_for_ui,
            inputs=[project_selector, evaluation_import_file, evaluation_state],
            outputs=[evaluation_status, evaluation_questions_table,
                     evaluation_state, evaluation_results_table],
        )
        export_evaluation_button.click(
            export_evaluation_questions_for_ui,
            inputs=[project_selector, evaluation_questions_table],
            outputs=[evaluation_status, evaluation_export_file],
        )
        generate_evaluation_button.click(
            generate_evaluation_for_ui,
            inputs=[project_selector, model_endpoint, api_key, evaluation_generation_model,
                    evaluation_test_model, evaluation_question_count, evaluation_retrieval_mode,
                    evaluation_top_k, chunk_state],
            outputs=[evaluation_status, evaluation_questions_table,
                     evaluation_state, evaluation_results_table],
        )
        run_evaluation_button.click(
            run_evaluation_for_ui,
            inputs=[project_selector, model_endpoint, api_key,
                    embedding_api_base, embedding_api_key,
                    neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
                    evaluation_test_model, evaluation_retrieval_mode,
                    evaluation_top_k, evaluation_state],
            outputs=[evaluation_status, evaluation_results_table, evaluation_state],
        )

        project_setting_inputs = [
            project_selector, documents_state, chunk_state, graph_state,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
            top_k, schema_editor,
        ]
        project_load_outputs = [
            project_state, project_status,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
            top_k, schema_editor,
            documents_state, chunk_state, graph_state,
            active_preview_state, active_chunks_state,
            documents_table, remove_document_selector,
            chunk_table, page_status, history_table,
            entity_table, relationship_table, build_status, import_status,
        ]
        project_tab.select(refresh_projects_for_ui, outputs=project_selector)
        create_project_event = create_project_button.click(
            create_project_for_ui, inputs=new_project_name,
            outputs=[project_selector, project_state, project_status],
        )
        load_project_event = load_project_button.click(
            load_project_for_ui, inputs=project_selector, outputs=project_load_outputs,
        )
        protected_tabs = [pdf_tab, graph_tab, evaluation_tab, qa_tab, history_tab]
        delete_project_event = delete_project_button.click(
            delete_project_for_ui,
            inputs=project_selector,
            outputs=[project_selector, project_state, project_status,
                     pdf_tab, graph_tab, evaluation_tab, qa_tab, history_tab,
                     delete_project_completed],
            js="""(projectId) => {
                if (!window.confirm('確定要刪除此專案嗎？專案設定、PDF、圖譜、題庫與紀錄都會永久刪除。')) {
                    throw new Error('使用者取消刪除');
                }
                return projectId;
            }""",
        )
        delete_project_event.then(
            refresh_projects_after_delete_for_ui,
            inputs=delete_project_completed,
            outputs=project_selector,
            show_progress="hidden",
        )
        create_project_event.success(
            unlock_project_tabs_for_ui, inputs=project_selector, outputs=protected_tabs,
        )
        load_project_event.success(
            unlock_project_tabs_for_ui, inputs=project_selector, outputs=protected_tabs,
        )
        auto_save_components = [
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
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
        embedding_test_button.click(
            check_embedding_service_for_ui,
            inputs=[embedding_api_base, embedding_api_key, embedding_test_model],
            outputs=embedding_connection_status,
        )
        env_inputs = [
            neo4j_uri,
            neo4j_database,
            neo4j_username,
            neo4j_password,
            model_endpoint,
            api_key,
            embedding_api_base,
            embedding_api_key,
            graph_llm_model,
            graph_embedding_model,
            answer_model,
        ]
        for component in env_inputs:
            component.change(persist_env_settings, inputs=env_inputs, outputs=env_status)
        reload_button.click(reload_env_settings, outputs=[*env_inputs, env_status])
        preview_event = preview_button.click(
            add_document_for_ui,
            inputs=[
                pdf_file,
                chunk_size,
                chunk_overlap,
                documents_state,
                chunk_state,
            ],
            outputs=[
                preview_status,
                chunk_table,
                documents_state,
                chunk_state,
                active_preview_state,
                active_chunks_state,
                page_status,
                documents_table,
                pdf_file,
                remove_document_selector,
            ],
        )
        preview_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        remove_document_event = remove_document_button.click(
            remove_document_for_ui,
            inputs=[project_selector, remove_document_selector, documents_state, chunk_state],
            outputs=[
                preview_status,
                documents_state,
                chunk_state,
                active_preview_state,
                active_chunks_state,
                page_status,
                documents_table,
                remove_document_selector,
                chunk_table,
            ],
        )
        remove_document_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        previous_button.click(
            previous_document,
            inputs=[documents_state, chunk_state, active_preview_state],
            outputs=[active_preview_state, active_chunks_state, chunk_table, page_status],
        )
        next_button.click(
            next_document,
            inputs=[documents_state, chunk_state, active_preview_state],
            outputs=[active_preview_state, active_chunks_state, chunk_table, page_status],
        )
        export_button.click(
            save_config, inputs=[active_preview_state], outputs=[preview_status, export_file]
        )
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
                schema_granularity,
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
                extraction_max_concurrent_requests,
                chunk_state,
                schema_editor,
                documents_state,
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
                embedding_api_base,
                embedding_api_key,
                neo4j_uri,
                neo4j_database,
                neo4j_username,
                neo4j_password,
                graph_embedding_model,
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
                embedding_api_base,
                embedding_api_key,
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
