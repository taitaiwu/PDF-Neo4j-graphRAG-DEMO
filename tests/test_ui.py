import json
from pathlib import Path
import gradio as gr
from manual_graphrag import ui
from manual_graphrag.chunking import PageText, TextChunk

from manual_graphrag.ui import build_app, connection_summary, persist_env_settings


def test_build_app_returns_blocks() -> None:
    assert isinstance(build_app(), gr.Blocks)


def test_model_fields_offer_openai_models_and_allow_custom_values() -> None:
    app = build_app()
    labels = {
        "Schema 規劃 LLM",
        "知識圖譜抽取 LLM",
        "Embedding 模型",
        "生題模型",
        "回答與評判模型",
        "問答 LLM",
    }
    fields = {
        component.get("props", {}).get("label"): component
        for component in app.config["components"]
        if component.get("props", {}).get("label") in labels
    }

    assert set(fields) == labels
    assert all(field["type"] == "dropdown" for field in fields.values())
    assert all(field["props"]["allow_custom_value"] for field in fields.values())
    llm_choices = fields["問答 LLM"]["props"]["choices"]
    embedding_choices = fields["Embedding 模型"]["props"]["choices"]
    assert ("gpt-4.1", "gpt-4.1") in llm_choices
    assert ("gpt-4o-mini", "gpt-4o-mini") in llm_choices
    assert ("text-embedding-3-large", "text-embedding-3-large") in embedding_choices


def test_plan_schema_button_uses_primary_variant() -> None:
    app = build_app()
    button = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("value") == "分析文件並規劃 Schema"
    )

    assert button["props"]["variant"] == "primary"


def test_project_page_uses_automatic_refresh_and_save() -> None:
    app = build_app()
    button_values = {
        component.get("props", {}).get("value")
        for component in app.config["components"]
        if component.get("type") == "button"
    }

    assert "重新整理專案清單" not in button_values
    assert "保存目前專案設定" not in button_values
    assert any(
        dependency.get("api_name") == "refresh_projects_for_ui"
        and any(trigger[1] == "select" for trigger in dependency.get("targets", []))
        for dependency in app.config["dependencies"]
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("save_project_for_ui")
        and any(trigger[1] == "input" for trigger in dependency.get("targets", []))
        for dependency in app.config["dependencies"]
    )


def test_pages_stay_locked_until_project_is_created_or_loaded() -> None:
    app = build_app()
    protected_labels = {
        "2. PDF 與參數", "3. 建圖", "4. 自動問答測試",
        "5. 問答測試", "6. 歷史紀錄",
    }
    tabs = [
        component for component in app.config["components"]
        if component.get("props", {}).get("label") in protected_labels
    ]

    assert len(tabs) == 5
    connection_tab = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("label") == "1. 連線設定"
    )
    assert connection_tab["props"].get("interactive", True) is True
    assert all(tab["props"]["interactive"] is False for tab in tabs)
    assert all(update["interactive"] is False for update in ui.unlock_project_tabs_for_ui(""))
    assert all(update["interactive"] is True for update in ui.unlock_project_tabs_for_ui("project"))
    unlock_dependencies = [
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("unlock_project_tabs_for_ui")
    ]
    assert len(unlock_dependencies) == 2
    assert all(len(dependency["outputs"]) == 5 for dependency in unlock_dependencies)


def test_delete_project_refreshes_list_after_server_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("刪除測試")

    deleted = ui.delete_project_for_ui(project["project_id"])

    assert "已刪除" in deleted[2]
    assert all(update["interactive"] is False for update in deleted[3:-1])
    assert deleted[-1] is True
    refreshed = ui.refresh_projects_after_delete_for_ui(deleted[-1])
    assert refreshed["choices"] == []
    assert refreshed["value"] is None
    assert ui.list_projects() == []


def test_failed_delete_does_not_clear_selection(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = ui.delete_project_for_ui("missing")

    assert result[2].startswith("❌")
    assert result[-1] is False
    assert "choices" not in ui.refresh_projects_after_delete_for_ui(result[-1])


def test_delete_button_uses_browser_confirmation() -> None:
    app = build_app()
    button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "刪除專案"
    )
    dependency = next(
        item for item in app.config["dependencies"]
        if any(target[0] == button["id"] for target in item.get("targets", []))
    )
    assert "window.confirm" in dependency["js"]
    assert "throw new Error" in dependency["js"]
    assert len(dependency["inputs"]) == 1
    assert any(
        str(item.get("api_name", "")).startswith("refresh_projects_after_delete_for_ui")
        and item.get("trigger_after") == dependency["id"]
        for item in app.config["dependencies"]
    )


def test_build_app_has_automatic_evaluation_page() -> None:
    app = build_app()
    values = [
        component.get("props", {}).get("value")
        for component in app.config["components"]
    ]

    assert "從 PDF 建立題目與答案" in values
    assert "一鍵測試" in values
    assert "匯入題目" in values
    assert "儲存題目" not in values
    assert "匯出題目" in values
    assert "#### 生題設定" in values
    assert "#### 測試模型設定" in values
    question_table = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("headers")
        == ["編號", "問題", "標準答案", "來源頁碼"]
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("save_evaluation_questions_for_ui")
        and [question_table["id"], "input"] in dependency.get("targets", [])
        for dependency in app.config["dependencies"]
    )
    assert any(component.get("props", {}).get("label") == "4. 自動問答測試" for component in app.config["components"])
    assert any(component.get("props", {}).get("label") == "5. 問答測試" for component in app.config["components"])
    assert any(component.get("props", {}).get("label") == "6. 歷史紀錄" for component in app.config["components"])
    components = app.config["components"]
    question_table_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("headers") == ["編號", "問題", "標準答案", "來源頁碼"]
    )
    metrics_box_index = next(
        index for index, component in enumerate(components)
        if "evaluation-metrics-box" in component.get("props", {}).get("elem_classes", [])
    )
    result_title_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("value") == "#### 測試結果"
    )
    result_table_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("headers")
        == ["編號", "問題", "標準答案", "實際答案", "結果", "評判理由"]
    )
    assert question_table_index < metrics_box_index < result_title_index < result_table_index


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



def test_project_ui_create_save_and_load(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _, created, status = ui.create_project_for_ui("手冊專案")
    assert status.startswith("✅")
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"pdf")
    values = [
        created["project_id"], str(document), {"file_name": "manual.pdf"},
        [TextChunk(1, "內容", (1,))], {
            "run_id": "run-1", "document": "manual.pdf", "neo4j_imported": True,
            "entities": [{"name": "設備", "type": "DEVICE", "description": "說明",
                          "source_chunk_numbers": [1], "source_pages": [1]}],
            "relationships": [{"source": "設備", "type": "USES", "target": "零件",
                               "description": "使用", "source_chunk_numbers": [1],
                               "source_pages": [1]}],
        },
        "bolt://db", "neo4j", "user", "pass", "http://models", "key",
        "build", "embed", "answer", 1, 5, 1200, 100, 0.2, 3000,
        "詳細", 10, 12, 2, "全部頁面", 4, "extract", 2,
        "關聯擴展檢索", 6, '{"entity_types": []}',
    ]
    saved, save_status = ui.save_project_for_ui(*values)
    loaded = ui.load_project_for_ui(created["project_id"])
    assert save_status.startswith("✅")
    assert Path(saved["document"]["path"]).read_bytes() == b"pdf"
    assert loaded[0]["project_id"] == created["project_id"]
    assert loaded[14:16] == (1200, 100)
    assert loaded[30][0].text == "內容"
    stored_project = ui.load_project(created["project_id"])
    assert stored_project["graph_state"]["entities"][0]["name"] == "設備"
    assert stored_project["graph_state"]["relationships"][0]["type"] == "USES"
    assert loaded[36][0][:2] == ["設備", "DEVICE"]
    assert loaded[37][0][:3] == ["設備", "USES", "零件"]
    assert "1 個實體、1 筆關係" in loaded[38]
    assert "已匯入 Neo4j" in loaded[39]


def test_project_answer_appends_history(monkeypatch) -> None:
    monkeypatch.setattr(ui, "answer_question_for_ui", lambda *args: ("✅ 完成", "答案", [["來源"]]))
    monkeypatch.setattr(ui, "load_project", lambda project_id: {"graph_state": {"document": "manual.pdf"}})
    captured = {}
    def fake_append(project_id, record):
        captured.update(record)
        return {"questions": [record]}
    monkeypatch.setattr(ui, "append_question", fake_append)
    result = ui.answer_question_for_project_ui(
        "project", "endpoint", "key", "bolt", "neo4j", "user", "pass",
        "answer-model", "問題", "關聯擴展檢索", 8,
    )
    assert captured["document"] == "manual.pdf"
    assert captured["sources"] == [["來源"]]
    assert result[3][0][1:3] == ["問題", "答案"]


def test_generate_evaluation_for_ui_saves_questions(monkeypatch) -> None:
    questions = [{"number": 1, "question": "Q", "expected_answer": "A", "source_pages": [1]}]
    monkeypatch.setattr(ui, "generate_evaluation_questions", lambda *args: questions)
    captured = {}
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status, rows, state, results = ui.generate_evaluation_for_ui(
        "project", "endpoint", "key", "generation-model", "test-model", 1, "關聯擴展檢索", 8,
        [TextChunk(1, "text", (1,))],
    )

    assert status.startswith("✅")
    assert rows[0][1:3] == ["Q", "A"]
    assert state["questions"] == questions
    assert state["preferences"]["generation_model"] == "generation-model"
    assert captured["evaluation"]["questions"] == questions
    assert results == []


def test_run_evaluation_for_ui_judges_and_saves(monkeypatch) -> None:
    monkeypatch.setattr(ui, "answer_question_for_ui", lambda *args: ("✅ 完成", "實際答案", []))
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *args: {"passed": True, "reason": "正確"})
    captured = {}
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})
    evaluation = {"questions": [
        {"number": 1, "question": "Q", "expected_answer": "A", "source_pages": [1]}
    ]}

    status, rows, updated = ui.run_evaluation_for_ui(
        "project", "endpoint", "key", "bolt", "neo4j", "user", "pass",
        "model", "關聯擴展檢索", 8, evaluation,
    )

    assert "答對 1 題 / 共 1 題" in status
    assert "答錯：0 題" in status
    assert "正確率：100.0%" in status
    assert rows[0][3:] == ["實際答案", "✅ 通過", "正確"]
    assert updated["results"][0]["passed"] is True
    assert captured["evaluation"] == updated


def test_edit_questions_auto_save_and_clear_results(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status, state, results = ui.save_evaluation_questions_for_ui(
        "project", [[9, "修改後問題", "修改後答案", "2, 3"]],
        {"results": [{"passed": True}]},
    )

    assert status.startswith("✅ 已自動儲存")
    assert state["dirty"] is False
    assert state["results"] == []
    assert state["questions"][0]["number"] == 1
    assert state["questions"][0]["source_pages"] == [2, 3]
    assert captured["evaluation"] == state
    assert results == []


def test_save_and_export_edited_questions(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("project")
    rows = [[1, "問題", "答案", "1, 4"]]

    status, state, _ = ui.save_evaluation_questions_for_ui(
        project["project_id"], rows, {"dirty": True}
    )
    export_status, export_path = ui.export_evaluation_questions_for_ui(
        project["project_id"], rows
    )

    assert status.startswith("✅")
    assert state["dirty"] is False
    assert export_status.startswith("✅")
    payload = json.loads(Path(export_path).read_text(encoding="utf-8"))
    assert payload["questions"][0]["source_pages"] == [1, 4]


def test_import_questions_supports_json_and_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("import-test")
    json_file = tmp_path / "questions.json"
    json_file.write_text(json.dumps({"questions": [
        {"question": "JSON Q", "expected_answer": "JSON A", "source_pages": [2]}
    ]}), encoding="utf-8")
    csv_file = tmp_path / "questions.csv"
    csv_file.write_text(
        "number,question,expected_answer,source_pages\n1,CSV Q,CSV A,3\n",
        encoding="utf-8",
    )

    json_result = ui.import_evaluation_questions_for_ui(project["project_id"], str(json_file), {})
    csv_result = ui.import_evaluation_questions_for_ui(project["project_id"], str(csv_file), {})

    assert json_result[2]["questions"][0]["question"] == "JSON Q"
    assert csv_result[2]["questions"][0]["expected_answer"] == "CSV A"
    assert json_result[2]["dirty"] is False
    assert csv_result[2]["dirty"] is False
    assert "已匯入並自動儲存" in csv_result[0]

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
        "embed", _importable_graph_state(),
    )

    assert status.startswith("✅ 已清空本工具既有圖譜")
    assert state["neo4j_imported"] is True
    assert state["embedding_model"] == "embed"
    assert state["embedding_dimensions"] == 1
    assert state["vector_index_name"] == "graph_evidence_embedding_1"
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
        "embed", _importable_graph_state(),
    )

    assert status == "❌ Neo4j 寫入失敗：offline"
    assert state["neo4j_imported"] is False
    assert state["neo4j_error"] == "Neo4j 寫入失敗：offline"


def test_import_graph_for_ui_requires_extraction() -> None:
    assert ui.import_graph_for_ui(
        "http://models/v1", "key", "bolt://db", "neo4j", "user", "password",
        "embed", {},
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
        "answer", " E01 怎麼處理？ ", "基本檢索", 8,
    )

    assert status.startswith("✅ 基本檢索")
    assert answer == "請重新啟動。"
    assert captured["args"][5] == "E01 怎麼處理？"
    assert rows == [[
        "原文", "E01 排除方式", "official-hybrid",
        "0.0300", "3", "2",
    ]]


def test_evaluation_preferences_keep_generation_and_test_models_separate(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ui, "load_project", lambda project_id: {"evaluation": {}})
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status = ui.save_evaluation_preferences_for_ui(
        "project", "generation-model", "test-model", 12, "基本檢索", 6
    )

    assert status.startswith("✅")
    assert captured["evaluation"]["preferences"] == {
        "generation_model": "generation-model",
        "test_model": "test-model",
        "question_count": 12,
        "retrieval_mode": "基本檢索",
        "top_k": 6,
    }


def test_load_evaluation_supports_legacy_shared_model(monkeypatch) -> None:
    monkeypatch.setattr(ui, "load_project", lambda project_id: {
        "evaluation": {"preferences": {"model": "legacy-model", "retrieval_mode": "GraphRAG"}}
    })

    loaded = ui.load_evaluation_for_ui("project")

    assert loaded[3:5] == ("legacy-model", "legacy-model")
    assert loaded[6] == "關聯擴展檢索"
