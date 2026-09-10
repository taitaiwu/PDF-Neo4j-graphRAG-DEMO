import json
import gradio as gr
from manual_graphrag import ui
from manual_graphrag.chunking import PageText, TextChunk

from manual_graphrag.ui import build_app, connection_summary, persist_env_settings


def test_build_app_returns_blocks() -> None:
    assert isinstance(build_app(), gr.Blocks)


def test_plan_schema_button_uses_primary_variant() -> None:
    app = build_app()
    button = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("value") == "分析文件並規劃 Schema"
    )

    assert button["props"]["variant"] == "primary"


def test_build_app_has_manual_neo4j_import_button() -> None:
    app = build_app()

    assert any(
        component.get("props", {}).get("value") == "Embedding 並匯入 Neo4j"
        for component in app.config["components"]
    )



def test_connection_summary_does_not_expose_secrets() -> None:
    status, settings = connection_summary(
        "bolt://localhost:7687", "neo4j", "neo4j", "password", "http://localhost:11434/v1", "key"
    )
    assert status.startswith("✅")
    assert "password" not in settings
    assert "api_key" not in settings


def test_persist_env_settings_writes_all_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    status = persist_env_settings(
        "bolt://db", "neo4j", "user", "pass", "http://models", "key",
        "http://embeddings", "embed-key", "build", "embed", "answer",
    )
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert status.startswith("✅")
    assert 'NEO4J_PASSWORD="pass"' in content
    assert 'ANSWER_MODEL="answer"' in content


def test_preview_page_filters_chunks_and_reports_page() -> None:
    chunks = [
        TextChunk(1, "first", (1,)),
        TextChunk(2, "shared", (1, 2)),
        TextChunk(3, "second", (2,)),
    ]

    status, rows = ui.preview_page(2, chunks, {"page_count": 3})

    assert status == "第 2 / 3 頁；顯示 2 個相關 chunk。"
    assert [row[0] for row in rows] == [2, 3]


def test_preview_page_reports_empty_page() -> None:
    status, rows = ui.preview_page(3, [TextChunk(1, "first", (1,))], {"page_count": 3})

    assert status == "第 3 / 3 頁；本頁沒有可解析文字或相關 chunk。"
    assert rows == []


def test_page_navigation_stays_within_document_bounds() -> None:
    state = {"page_count": 3}

    assert ui.previous_page(1, state) == 1
    assert ui.previous_page(3, state) == 2
    assert ui.next_page(2, state) == 3
    assert ui.next_page(3, state) == 3

    ranged_state = {"page_count": 3, "page_start": 2, "page_end": 4}
    assert ui.previous_page(2, ranged_state) == 2
    assert ui.next_page(3, ranged_state) == 4
    assert ui.next_page(4, ranged_state) == 4


def test_preview_pdf_initializes_page_navigation(monkeypatch) -> None:
    pages = [PageText(2, "two"), PageText(3, "three")]
    chunks = [
        TextChunk(1, "two", (2,)),
        TextChunk(2, "three", (3,)),
    ]
    monkeypatch.setattr(ui, "extract_pdf", lambda path, start, end: (pages, []))
    monkeypatch.setattr(ui, "chunk_pages", lambda *args: chunks)

    result = ui.preview_pdf("manual.pdf", 100, 0, 2, 3)
    status, rows, state, stored_chunks, slider_update, page_status = result

    assert status.startswith("已解析第 2–3 頁，產生 2 個 chunk。")
    assert [row[0] for row in rows] == [1]
    assert state["page_count"] == 2
    assert state["page_start"] == 2
    assert state["page_end"] == 3
    assert "chunks" not in state
    assert stored_chunks is chunks
    assert slider_update["minimum"] == 2
    assert slider_update["maximum"] == 3
    assert slider_update["value"] == 2
    assert page_status == "第 2 / 3 頁；顯示 1 個相關 chunk。"


def test_initialize_page_range_defaults_end_to_last_page(monkeypatch) -> None:
    monkeypatch.setattr(ui, "get_pdf_page_count", lambda path: 326)

    start_update, end_update, status = ui.initialize_page_range("manual.pdf")

    assert start_update["value"] == 1
    assert start_update["maximum"] == 326
    assert end_update["value"] == 326
    assert end_update["maximum"] == 326
    assert status == "已偵測到 326 頁；解析結束頁預設為第 326 頁。"


def test_plan_schema_for_ui_returns_editable_json(monkeypatch) -> None:
    from manual_graphrag.graph_service import SchemaPlan

    monkeypatch.setattr(
        ui,
        "plan_graph_schema",
        lambda *args: SchemaPlan(
            {"entity_types": [{"name": "DEVICE"}], "relationship_types": [{"name": "USES"}]},
            5,
            5,
            2,
            1,
        ),
    )

    status, schema_text = ui.plan_schema_for_ui(
        "http://models/v1",
        "key",
        "llm",
        0.3,
        999,
        "平衡",
        15,
        20,
        3,
        "全部頁面",
        10,
        [TextChunk(1, "text", (1,))],
    )

    assert status.startswith("✅")
    assert "全部 1 頁、5 個 chunk" in status
    assert "共 2 批、1 輪整合" in status
    assert "粒度：平衡；實體／關係類型上限：15／20" in status
    assert json.loads(schema_text)["entity_types"][0]["name"] == "DEVICE"


def test_extract_graph_for_ui_formats_tables_and_state(monkeypatch) -> None:
    from manual_graphrag.graph_service import GraphExtraction

    extraction = GraphExtraction(
        entities=[
            {
                "name": "設備 A",
                "type": "DEVICE",
                "description": "設備",
                "source_chunk_numbers": [1],
                "source_pages": [3],
            }
        ],
        relationships=[
            {
                "source": "設備 A",
                "type": "USES",
                "target": "設備 B",
                "description": "使用",
                "source_chunk_numbers": [1],
                "source_pages": [3],
            }
        ],
        processed_chunks=1,
    )
    monkeypatch.setattr(ui, "extract_graph", lambda *args: extraction)
    monkeypatch.setattr(
        ui,
        "import_extraction",
        lambda *args: (_ for _ in ()).throw(AssertionError("抽取時不應匯入 Neo4j")),
    )
    schema = {
        "entity_types": [{"name": "DEVICE"}],
        "relationship_types": [{"name": "USES"}],
    }

    status, entities, relationships, state = ui.extract_graph_for_ui(
        "http://models/v1",
        "key",
        "llm",
        0.2,
        1500,
        3,
        [TextChunk(1, "text", (3,))],
        json.dumps(schema),
        {"file_name": "manual.pdf"},
    )

    assert status.startswith("✅ 已處理 1 個 chunk")
    assert entities[0][:2] == ["設備 A", "DEVICE"]
    assert relationships[0][:3] == ["設備 A", "USES", "設備 B"]
    assert state["document"] == "manual.pdf"
    assert state["temperature"] == 0.2
    assert state["max_output_tokens"] == 1500
    assert state["max_concurrent_requests"] == 3
    assert state["chunks"][0]["text"] == "text"
    assert state["neo4j_imported"] is False
    assert "進行 Embedding 並匯入 Neo4j" in status


def test_extract_graph_for_ui_rejects_invalid_schema() -> None:
    result = ui.extract_graph_for_ui(
        "http://models/v1", "", "llm", 0.2, 1500, 3, [], "not-json", {}
    )

    assert result == ("❌ schema 不是有效 JSON。", [], [], {})


def _importable_graph_state() -> dict:
    return {
        "run_id": "run-1",
        "document": "manual.pdf",
        "llm_model": "llm",
        "schema": {
            "entity_types": [{"name": "DEVICE"}],
            "relationship_types": [{"name": "USES"}],
        },
        "entities": [{
            "name": "設備 A", "type": "DEVICE", "description": "設備",
            "source_chunk_numbers": [1], "source_pages": [1],
        }],
        "relationships": [],
        "chunks": [{"number": 1, "text": "設備說明", "pages": [1]}],
        "neo4j_imported": False,
    }


def test_import_graph_for_ui_imports_saved_extraction(monkeypatch) -> None:
    from manual_graphrag.neo4j_service import ImportSummary

    captured = {}

    def fake_import(*args):
        captured["args"] = args
        return ImportSummary(2, 1)

    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]] * 2)
    monkeypatch.setattr(ui, "import_extraction", fake_import)

    status, state = ui.import_graph_for_ui(
        "http://models/v1", "key", "bolt://db", "neo4j", "user", "password",
        "embed", "保留既有圖譜", False, _importable_graph_state(),
    )

    assert status.startswith("✅ 匯入模式：保留既有圖譜")
    assert state["neo4j_imported"] is True
    assert state["embedding_model"] == "embed"
    assert captured["args"][4] == "run-1"


def test_import_graph_for_ui_keeps_state_when_import_fails(monkeypatch) -> None:
    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]] * 2)
    monkeypatch.setattr(
        ui,
        "import_extraction",
        lambda *args: (_ for _ in ()).throw(ValueError("Neo4j 寫入失敗：offline")),
    )

    status, state = ui.import_graph_for_ui(
        "http://models/v1", "key", "bolt://db", "neo4j", "user", "password",
        "embed", "保留既有圖譜", False, _importable_graph_state(),
    )

    assert status == "❌ Neo4j 寫入失敗：offline"
    assert state["neo4j_imported"] is False
    assert state["neo4j_error"] == "Neo4j 寫入失敗：offline"


def test_import_graph_for_ui_requires_extraction() -> None:
    assert ui.import_graph_for_ui(
        "http://models/v1", "key", "bolt://db", "neo4j", "user", "password",
        "embed", "保留既有圖譜", False, {},
    ) == ("❌ 請先完成知識圖譜抽取。", {})


def test_answer_question_for_ui_displays_hybrid_scores(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ui,
        "load_latest_graph",
        lambda *args: {
            "run_id": "run-1",
            "document": "manual.pdf",
            "embedding_model": "embed",
        },
    )
    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]])
    captured = {}

    def fake_search(*args):
        captured["args"] = args
        return [{
            "evidence_id": "chunk-1",
            "kind": "原文",
            "text": "E01 排除方式",
            "source_pages": [3],
            "source_chunk_numbers": [2],
            "score": 0.03,
            "fusion_score": 0.03,
            "matched_by": ["official-hybrid"],
        }]

    monkeypatch.setattr(ui, "search_graph_evidence", fake_search)
    monkeypatch.setattr(
        ui,
        "answer_graph_question",
        lambda *args: {"answer": "請重新啟動。", "evidence": args[-1]},
    )

    status, answer, rows = ui.answer_question_for_ui(
        "http://models/v1", "key", "http://embed/v1", "embed-key",
        "bolt://db", "neo4j", "user", "password",
        "answer", " E01 怎麼處理？ ", "向量 RAG", 8,
    )

    assert status.startswith("✅ 向量 RAG")
    assert answer == "請重新啟動。"
    assert captured["args"][5] == "E01 怎麼處理？"
    assert rows == [[
        "原文", "E01 排除方式", "official-hybrid",
        "0.0300", "3", "2",
    ]]
