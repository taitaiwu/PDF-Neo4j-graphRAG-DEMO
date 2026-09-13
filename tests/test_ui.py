import json
from pathlib import Path
import gradio as gr
import pytest
from manual_graphrag import service_settings as settings, ui
from manual_graphrag.chunking import PageText, TextChunk

from manual_graphrag.ui import build_app, connection_summary, persist_env_settings


def test_build_app_returns_blocks() -> None:
    assert isinstance(build_app(), gr.Blocks)


def test_current_project_banner_reflects_project_name() -> None:
    assert ui._current_project_banner({}) == "### 📁 目前專案：尚未選擇"
    assert ui._current_project_banner({"name": "手冊專案"}) == "### 📁 目前專案：手冊專案"


def test_build_app_wires_project_state_change_to_banner() -> None:
    app = build_app()
    assert any(
        component.get("props", {}).get("value") == "### 📁 目前專案：尚未選擇"
        for component in app.config["components"]
    )
    dependency = next(
        item for item in app.config["dependencies"]
        if str(item.get("api_name", "")).startswith("_current_project_banner")
    )
    assert any(
        (target[1] if isinstance(target, (list, tuple)) else None) == "change"
        for target in dependency.get("targets", [])
    )


def test_model_fields_only_offer_initially_checked_models() -> None:
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
    assert all(not field["props"]["allow_custom_value"] for field in fields.values())
    for field in fields.values():
        for display, value in field["props"]["choices"]:
            assert display in {f"OpenAI｜{value}", f"Ollama｜{value}"}


def test_graph_controls_are_above_schema_and_type_limit_fields_are_removed() -> None:
    app = build_app()
    components = app.config["components"]
    ids_by_value = {
        value: component["id"]
        for component in components
        if isinstance((value := component.get("props", {}).get("value")), str)
    }
    schema_heading = ids_by_value["#### ① 規劃 Schema"]
    assert ids_by_value["⏸ 暫停"] < schema_heading
    assert ids_by_value["⏹ 停止"] < schema_heading
    labels = {component.get("props", {}).get("label") for component in components}
    assert "最大實體類型數" not in labels
    assert "最大關係類型數" not in labels


def test_pages_three_through_five_default_all_llm_fields_to_gpt_4_1_mini(monkeypatch) -> None:
    monkeypatch.setattr(ui, "service_choices", lambda state: ["gpt-4.1-mini", "gpt-4o-mini"])
    monkeypatch.setattr(
        ui, "service_choice_items",
        lambda state: [("OpenAI｜gpt-4.1-mini", "gpt-4.1-mini"), ("OpenAI｜gpt-4o-mini", "gpt-4o-mini")],
    )
    monkeypatch.setattr(ui, "preferred_service_model", lambda state: "gpt-4.1-mini")
    app = build_app()
    labels = {
        "Schema 規劃 LLM", "知識圖譜抽取 LLM", "生題模型", "回答與評判模型", "問答 LLM",
    }
    fields = [
        component for component in app.config["components"]
        if component.get("props", {}).get("label") in labels
    ]
    assert len(fields) == 5
    assert all(field["props"]["value"] == "gpt-4.1-mini" for field in fields)


def test_pause_and_stop_buttons_bypass_the_queue() -> None:
    app = build_app()
    pause_button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "⏸ 暫停"
    )
    stop_button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "⏹ 停止"
    )
    pause_dependency = next(
        item for item in app.config["dependencies"]
        if str(item.get("api_name", "")).startswith("toggle_pause_for_ui")
    )
    stop_dependency = next(
        item for item in app.config["dependencies"]
        if str(item.get("api_name", "")).startswith("request_stop_for_ui")
    )
    assert stop_button["props"]["variant"] == "stop"
    assert pause_dependency["queue"] is False
    assert stop_dependency["queue"] is False
    assert any(target[0] == pause_button["id"] for target in pause_dependency["targets"])
    assert any(target[0] == stop_button["id"] for target in stop_dependency["targets"])


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


def test_pdf_page_range_has_valid_values_before_upload() -> None:
    app = build_app()
    fields = {
        component.get("props", {}).get("label"): component
        for component in app.config["components"]
    }

    assert fields["解析起始頁"]["props"]["value"] == 1
    assert fields["解析結束頁（上傳後自動設為最後一頁）"]["props"]["value"] == 1
    assert fields["解析結束頁（上傳後自動設為最後一頁）"]["props"]["minimum"] == 1


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
    llm = settings.load_service_settings("llm")
    embedding = settings.load_service_settings("embedding")
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("", False, llm, embedding))
    llm["profiles"]["OpenAI"]["connected"] = True
    embedding["profiles"]["OpenAI"]["connected"] = True
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("project", False, llm, embedding))
    assert all(update["interactive"] is True for update in ui.workflow_tabs_for_ui("project", True, llm, embedding))
    gate_dependencies = [
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("workflow_tabs_for_ui")
    ]
    assert len(gate_dependencies) >= 5
    assert all(len(dependency["outputs"]) == 5 for dependency in gate_dependencies)



def test_workflow_gate_requires_project_neo4j_llm_and_embedding() -> None:
    llm = settings.load_service_settings("llm")
    embedding = settings.load_service_settings("embedding")
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("", False, llm, embedding))
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("project", False, llm, embedding))
    llm["profiles"]["OpenAI"]["connected"] = True
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("project", True, llm, embedding))
    embedding["profiles"]["OpenAI"]["connected"] = True
    assert all(update["interactive"] is True for update in ui.workflow_tabs_for_ui("project", True, llm, embedding))


def test_ollama_models_require_current_successful_fetch(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = settings.load_service_settings("llm")
    profile = state["profiles"]["Ollama"]
    state["active"] = "Ollama"
    profile["rows"] = [[True, "local-model"]]
    assert settings.service_choices(state) == []
    monkeypatch.setattr(ui, "list_models", lambda *args: ["local-model"])
    fetched = ui.service_action_for_ui(
        "fetch", "Ollama", state, profile["base_url"], profile["api_key"],
        profile["rows"], *profile["models"],
    )
    assert settings.service_choices(fetched[0]) == ["local-model"]
    monkeypatch.setattr(ui, "list_models", lambda *args: (_ for _ in ()).throw(ValueError("offline")))
    failed = ui.service_action_for_ui(
        "fetch", "Ollama", fetched[0], fetched[1], fetched[2], fetched[3]["value"],
        *[field["value"] for field in fetched[7:]],
    )
    assert settings.service_choices(failed[0]) == []

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
        == ["編號", "問題", "標準答案", "來源頁碼", "來源文件"]
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("save_evaluation_questions_for_ui")
        and any(
            tuple(target) == (question_table["id"], "input")
            for target in dependency.get("targets", [])
        )
        for dependency in app.config["dependencies"]
    )
    assert any(component.get("props", {}).get("label") == "4. 自動問答測試" for component in app.config["components"])
    assert any(component.get("props", {}).get("label") == "5. 問答測試" for component in app.config["components"])
    assert any(component.get("props", {}).get("label") == "6. 歷史紀錄" for component in app.config["components"])
    components = app.config["components"]
    question_table_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("headers") == ["編號", "問題", "標準答案", "來源頁碼", "來源文件"]
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
    assert 'ANSWER_MODEL' not in content
    assert 'BUILD_MODEL' not in content
    assert 'MODEL_SERVICE_PROFILES' not in content




def test_pdf_table_has_checkbox_and_new_project_resets_visible_status() -> None:
    app = build_app()
    components = app.config["components"]
    table = next(
        component for component in components
        if component.get("props", {}).get("headers") == ["選取", "文件", "頁碼範圍", "Chunk 數"]
    )
    assert table["props"]["datatype"] == ["bool", "str", "str", "number"]
    assert table["props"]["interactive"] is True
    create_button = next(
        component for component in components
        if component.get("props", {}).get("value") == "建立新專案"
    )
    create_event = next(
        dependency for dependency in app.config["dependencies"]
        if (create_button["id"], "click") in dependency["targets"]
    )
    assert any(
        dependency.get("trigger_after") == create_event["id"]
        and str(dependency.get("api_name", "")).startswith("reset_new_project_pdf_status_for_ui")
        for dependency in app.config["dependencies"]
    )

def test_build_app_has_independent_provider_switches_and_ollama_tables() -> None:
    app = build_app()
    components = app.config["components"]
    buttons = {c["props"]["value"]: c for c in components if c["type"] == "button"}
    assert "⚡ 套用 Ollama 本機預設（省 token）" not in buttons
    assert not {"重新整理模型清單", "重新整理 Embedding 模型清單"} & buttons.keys()
    for label, provider_label, fetch_label, test_label, count in [
        ("LLM 模型清單", "模型服務來源", "獲得模型清單", "測試模型服務連線", 5),
        ("Embedding 模型清單", "Embedding 服務來源", "獲得 Embedding 模型清單", "測試 Embedding 服務連線", 1),
    ]:
        table = next(c for c in components if c.get("props", {}).get("label") == label)
        provider = next(c for c in components if c.get("props", {}).get("label") == provider_label)
        expected_providers = [("OpenAI", "OpenAI"), ("Ollama", "Ollama")]
        if provider_label == "Embedding 服務來源":
            expected_providers.append(("Voyage", "Voyage"))
        assert provider["props"]["choices"] == expected_providers
        ollama = provider["props"]["value"] == "Ollama"
        assert buttons[fetch_label]["props"]["visible"] == ollama
        assert buttons[test_label]["props"]["visible"] != ollama
        assert table["props"]["visible"] == ollama
        assert table["props"]["datatype"] == ["bool", "str"]
        assert table["props"]["static_columns"] == [1]
        fetch = next(d for d in app.config["dependencies"] if (buttons[fetch_label]["id"], "click") in d["targets"])
        selection = next(d for d in app.config["dependencies"] if (table["id"], "input") in d["targets"])
        switch = next(d for d in app.config["dependencies"] if (provider["id"], "input") in d["targets"])
        assert fetch["outputs"] == selection["outputs"] == switch["outputs"]
        assert len(fetch["outputs"]) == count + 7


def test_project_ui_create_save_and_load(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _, created, status = ui.create_project_for_ui("手冊專案")
    assert status.startswith("✅")
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"pdf")
    values = [
        created["project_id"],
        [{"file_name": "manual.pdf", "file_path": str(document)}],
        [TextChunk(1, "內容", (1,), "manual.pdf")], {
            "run_id": "run-1", "document": "manual.pdf", "neo4j_imported": True,
            "entities": [{"name": "設備", "type": "DEVICE", "description": "說明",
                          "source_chunk_numbers": [1], "source_pages": [1]}],
            "relationships": [{"source": "設備", "type": "USES", "target": "零件",
                               "description": "使用", "source_chunk_numbers": [1],
                               "source_pages": [1]}],
        },
        "bolt://db", "neo4j", "user", "pass", "http://models", "key",
        "build", "embed", "answer", 1200, 100, 0.2,
        "詳細", 2, "全部頁面", 4, "extract", 2,
        "關聯擴展檢索", 6, '{"entity_types": []}',
    ]
    saved, save_status = ui.save_project_for_ui(*values)
    loaded = ui.load_project_for_ui(created["project_id"])
    assert save_status.startswith("✅")
    assert Path(saved["documents"][0]["path"]).read_bytes() == b"pdf"
    assert loaded[0]["project_id"] == created["project_id"]
    assert loaded[11:13] == (1200, 100)
    assert loaded[24][0].text == "內容"
    stored_project = ui.load_project(created["project_id"])
    assert "model_endpoint" not in stored_project["settings"]
    assert "api_key" not in stored_project["settings"]
    assert stored_project["graph_state"]["entities"][0]["name"] == "設備"
    assert stored_project["graph_state"]["relationships"][0]["type"] == "USES"
    assert loaded[33][0][:2] == ["設備", "DEVICE"]
    assert loaded[34][0][:3] == ["設備", "USES", "零件"]
    assert "1 個實體、1 筆關係" in loaded[35]
    assert "已匯入 Neo4j" in loaded[36]


def test_load_project_ignores_legacy_model_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui, "load_env", lambda: {
        "MODEL_OPENAI_API_BASE": "https://current.example/v1",
        "MODEL_OPENAI_API_KEY": "current-key",
        "MODEL_OLLAMA_API_BASE": "http://localhost:11434/v1",
        "MODEL_OLLAMA_API_KEY": "",
        "EMBEDDING_OPENAI_API_BASE": "https://current.example/v1",
        "EMBEDDING_OPENAI_API_KEY": "current-key",
        "EMBEDDING_OLLAMA_API_BASE": "http://localhost:11434/v1",
        "EMBEDDING_OLLAMA_API_KEY": "",
        "EMBEDDING_VOYAGE_API_BASE": "https://api.voyageai.com/v1",
        "EMBEDDING_VOYAGE_API_KEY": "",
        "NEO4J_URI": "bolt://db",
        "NEO4J_DATABASE": "neo4j",
        "NEO4J_USERNAME": "user",
        "NEO4J_PASSWORD": "password",
    })
    _, created, _ = ui.create_project_for_ui("Legacy")
    project_path = tmp_path / "data" / "projects" / created["project_id"] / "project.json"
    ui.write_json(project_path, {
        **created,
        "documents": [],
        "documents_meta": [],
        "settings": {
            "model_endpoint": "https://legacy.example/v1",
            "api_key": "legacy-key",
        },
    })

    loaded = ui.load_project_for_ui(created["project_id"])

    migrated = ui.load_project(created["project_id"])
    assert loaded[6:8] == ("https://current.example/v1", "current-key")
    assert "model_endpoint" not in migrated["settings"]
    assert "api_key" not in migrated["settings"]


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
    monkeypatch.setattr(
        ui, "generate_document_summary",
        lambda *args: {"document": args[3][0].document, "summary": "摘要"},
    )
    captured = {}
    real_executor = ui.ThreadPoolExecutor

    def recording_executor(max_workers):
        captured["generation_workers"] = max_workers
        return real_executor(max_workers=max_workers)

    monkeypatch.setattr(ui, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status, rows, state, results = ui.generate_evaluation_for_ui(
        "project", "endpoint", "key", "generation-model", "test-model", 1, "關聯擴展檢索", 8,
        [TextChunk(1, "text", (1,))], 2, 5,
    )

    assert status.startswith("✅")
    assert rows[0][1:3] == ["Q", "A"]
    assert state["questions"] == questions
    assert state["preferences"]["generation_model"] == "generation-model"
    assert captured["evaluation"]["questions"] == questions
    assert state["preferences"]["generation_max_concurrent_requests"] == 2
    assert captured["generation_workers"] == 2
    assert state["preferences"]["test_max_concurrent_requests"] == 5
    assert results == []


def test_generate_evaluation_distributes_questions_across_documents(monkeypatch) -> None:
    requested_documents = []
    exclusions_by_document = {"a.pdf": [], "b.pdf": []}

    def fake_generate(_endpoint, _key, _model, chunks, count, _excluded=None):
        document = chunks[0].document
        exclusions_by_document[document].append(list(_excluded or []))
        requested_documents.append(document)
        sequence = requested_documents.count(document)
        return [{
            "number": 1,
            "question": f"{document} question {sequence}",
            "expected_answer": "answer",
            "source_pages": [1],
            "document": chunks[0].document,
        }]

    monkeypatch.setattr(ui, "generate_evaluation_questions", fake_generate)
    monkeypatch.setattr(
        ui, "generate_document_summary",
        lambda *args: {"document": args[3][0].document, "summary": "摘要"},
    )
    monkeypatch.setattr(ui, "save_project", lambda *args: {})

    status, _rows, state, _results = ui.generate_evaluation_for_ui(
        "project", "endpoint", "key", "generation-model", "test-model",
        2, "基本檢索", 8,
        [TextChunk(1, "a", (1,), "a.pdf"), TextChunk(2, "b", (1,), "b.pdf")],
        3, 3,
    )

    assert status.startswith("✅")
    assert "2 份 PDF 各建立 2 道題目，共 4 道" in status
    assert requested_documents.count("a.pdf") == 2
    assert requested_documents.count("b.pdf") == 2
    assert [item["document"] for item in state["questions"]] == [
        "a.pdf", "a.pdf", "b.pdf", "b.pdf",
    ]
    assert exclusions_by_document["a.pdf"] == [[], ["a.pdf question 1"]]
    assert exclusions_by_document["b.pdf"] == [[], ["b.pdf question 1"]]


def test_generate_evaluation_refills_duplicate_questions(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_generate(_endpoint, _key, _model, chunks, _count, excluded=None):
        calls["count"] += 1
        question = "相同問題？" if calls["count"] <= 2 else "不同問題？"
        if excluded:
            assert "相同問題？" in excluded
        return [{
            "number": 1,
            "question": question,
            "expected_answer": "答案",
            "source_pages": [1],
            "source_chunk_numbers": [1],
            "document": chunks[0].document,
        }]

    monkeypatch.setattr(ui, "generate_evaluation_questions", fake_generate)
    monkeypatch.setattr(
        ui, "generate_document_summary",
        lambda *args: {"document": args[3][0].document, "summary": "摘要"},
    )
    monkeypatch.setattr(ui, "save_project", lambda *args: {})

    status, _rows, state, _results = ui.generate_evaluation_for_ui(
        "project",
        "endpoint",
        "key",
        "generation-model",
        "test-model",
        2,
        "基本檢索",
        8,
        [TextChunk(1, "內容", (1,), "manual.pdf")],
        2,
        3,
    )

    assert status.startswith("✅")
    assert [item["question"] for item in state["questions"]] == ["相同問題？", "不同問題？"]
    assert calls["count"] == 3


def test_run_evaluation_for_ui_judges_and_saves(monkeypatch) -> None:
    monkeypatch.setattr(ui, "answer_question_for_ui", lambda *args: ("✅ 完成", "實際答案", []))
    real_executor = ui.ThreadPoolExecutor

    def recording_executor(max_workers):
        captured["test_workers"] = max_workers
        return real_executor(max_workers=max_workers)

    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *args: {"passed": True, "reason": "正確"})
    captured = {}
    monkeypatch.setattr(ui, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})
    evaluation = {"questions": [
        {"number": 1, "question": "Q", "expected_answer": "A", "source_pages": [1], "document": "manual.pdf"}
    ]}

    status, rows, updated = ui.run_evaluation_for_ui(
        "project", "endpoint", "key", "embed-endpoint", "embed-key",
        "bolt", "neo4j", "user", "pass",
        "model", "關聯擴展檢索", 8, evaluation, 2,
    )

    assert "總共答對 1 題 / 1 題" in status
    assert "manual.pdf：答對 1 題 / 1 題" in status
    assert "答錯：0 題" in status
    assert "正確率：100.0%" in status
    assert rows[0][3:] == ["實際答案", "✅ 通過", "正確"]
    assert captured["test_workers"] == 2
    assert updated["results"][0]["passed"] is True
    assert captured["evaluation"] == updated


def test_run_evaluation_for_ui_forwards_credentials_to_answer_question_for_ui(monkeypatch) -> None:
    captured = {}

    def fake_answer_question_for_ui(*args):
        captured["args"] = args
        return "✅ 完成", "實際答案", []

    monkeypatch.setattr(ui, "answer_question_for_ui", fake_answer_question_for_ui)
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *args: {"passed": True, "reason": "正確"})
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: {})
    evaluation = {"questions": [
        {"number": 1, "question": "Q", "expected_answer": "A", "source_pages": [1]}
    ]}

    ui.run_evaluation_for_ui(
        "project", "endpoint", "key", "embed-endpoint", "embed-key",
        "bolt", "neo4j", "user", "pass",
        "model", "關聯擴展檢索", 8, evaluation,
    )

    assert captured["args"] == (
        "endpoint", "key", "embed-endpoint", "embed-key",
        "bolt", "neo4j", "user", "pass",
        "model", "Q", "關聯擴展檢索", 8,
    )


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

def test_switch_document_cycles_through_documents() -> None:
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 2},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    chunks = [
        TextChunk(1, "a-text", (1,), "a.pdf"),
        TextChunk(2, "b-text", (1,), "b.pdf"),
    ]

    doc, doc_chunks, rows, status = ui.next_document(documents, chunks, documents[0])

    assert doc["file_name"] == "b.pdf"
    assert [chunk.document for chunk in doc_chunks] == ["b.pdf"]
    assert [row[0] for row in rows] == [2]
    assert status == "文件 2 / 2：b.pdf（第 1–2 頁），共 1 個 chunk。"

    back, back_chunks, back_rows, back_status = ui.previous_document(documents, chunks, doc)
    assert back["file_name"] == "a.pdf"
    assert [chunk.document for chunk in back_chunks] == ["a.pdf"]
    assert back_status == "文件 1 / 2：a.pdf（第 1–2 頁），共 1 個 chunk。"


def test_switch_document_wraps_around_and_handles_empty_list() -> None:
    documents = [{"file_name": "only.pdf", "page_start": 1, "page_end": 1}]
    chunks = [TextChunk(1, "text", (1,), "only.pdf")]

    doc, *_rest = ui.next_document(documents, chunks, documents[0])
    assert doc["file_name"] == "only.pdf"

    doc, doc_chunks, rows, status = ui.previous_document([], [], {})
    assert doc == {}
    assert doc_chunks == []
    assert rows == []
    assert status == "尚未解析任何 PDF。"


def test_add_document_for_ui_initializes_active_document(monkeypatch) -> None:
    pages = [PageText(2, "two"), PageText(3, "three")]
    chunks = [
        TextChunk(1, "two", (2,), "manual.pdf"),
        TextChunk(2, "three", (3,), "manual.pdf"),
    ]
    monkeypatch.setattr(ui, "extract_pdf", lambda path: (pages, []))
    monkeypatch.setattr(ui, "chunk_pages", lambda *args, **kwargs: chunks)

    result = ui.add_document_for_ui("manual.pdf", 100, 0, [], [])
    (
        status, rows, documents, stored_chunks, active_preview, active_chunks,
        document_status, documents_rows, pdf_reset, remove_choices,
    ) = result

    assert status.startswith("✅「manual.pdf」第 2–3 頁，產生 2 個 chunk")
    assert [row[0] for row in rows] == [1, 2]
    assert documents[0]["file_name"] == "manual.pdf"
    assert documents[0]["page_count"] == 2
    assert documents[0]["page_start"] == 2
    assert documents[0]["page_end"] == 3
    assert stored_chunks == chunks
    assert active_chunks == chunks
    assert active_preview["file_name"] == "manual.pdf"
    assert document_status == "文件 1 / 1：manual.pdf（第 2–3 頁），共 2 個 chunk。"
    assert documents_rows == [[False, "manual.pdf", "2–3", 2]]
    assert remove_choices["choices"] == ["manual.pdf"]


def test_add_document_for_ui_rejects_duplicate_document_name(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "extract_pdf", lambda path: ((_ for _ in ()).throw(AssertionError()))
    )
    existing = [{"file_name": "manual.pdf"}]

    status, *_ = ui.add_document_for_ui("manual.pdf", 100, 0, existing, [])

    assert status.startswith("❌")
    assert "同名文件" in status



def test_document_table_selection_removes_checked_pdfs(monkeypatch) -> None:
    removed = []
    monkeypatch.setattr(ui, "remove_document", lambda project_id, name: removed.append(name))
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    table = [[True, "a.pdf", "1–1", 1], [False, "b.pdf", "1–2", 2]]
    result = ui.remove_document_for_ui("project", table, documents, [])
    assert removed == ["a.pdf"]
    assert [doc["file_name"] for doc in result[1]] == ["b.pdf"]
    assert result[6] == [[False, "b.pdf", "1–2", 0]]



def test_document_table_dataframe_selection_removes_checked_pdf(monkeypatch) -> None:
    import pandas as pd

    removed = []
    monkeypatch.setattr(ui, "remove_document", lambda project_id, name: removed.append(name))
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    table = pd.DataFrame([
        [True, "a.pdf", "1–1", 1],
        [False, "b.pdf", "1–2", 2],
    ], columns=["選取", "文件", "頁碼範圍", "Chunk 數"])
    result = ui.remove_document_for_ui("project", table, documents, [])
    assert removed == ["a.pdf"]
    assert [doc["file_name"] for doc in result[1]] == ["b.pdf"]

def test_new_project_resets_pdf_status_message() -> None:
    assert ui.reset_new_project_pdf_status_for_ui() == (
        "尚未解析 PDF。可重複上傳多份 PDF，逐一加入同一個專案。"
    )

def test_remove_document_for_ui_prunes_documents_and_chunks_without_project(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "remove_document", lambda *args: (_ for _ in ()).throw(AssertionError("不應呼叫"))
    )
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 2},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    chunks = [
        TextChunk(1, "a-text", (1,), "a.pdf"),
        TextChunk(2, "b-text", (1,), "b.pdf"),
    ]

    result = ui.remove_document_for_ui("", "a.pdf", documents, chunks)
    (
        status, remaining_documents, remaining_chunks, active_preview, active_chunks,
        document_status, documents_rows, remove_choices, rows,
    ) = result

    assert status.startswith("✅ 已移除 1 份文件「a.pdf」")
    assert [doc["file_name"] for doc in remaining_documents] == ["b.pdf"]
    assert [chunk.document for chunk in remaining_chunks] == ["b.pdf"]
    assert active_preview["file_name"] == "b.pdf"
    assert active_chunks == remaining_chunks
    assert documents_rows == [[False, "b.pdf", "1–2", 0]]
    assert remove_choices["choices"] == ["b.pdf"]


def test_remove_document_for_ui_deletes_from_project_when_loaded(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        ui, "remove_document",
        lambda project_id, file_name: captured.update(project_id=project_id, file_name=file_name),
    )
    documents = [{"file_name": "a.pdf", "page_start": 1, "page_end": 1}]

    status, remaining_documents, *_ = ui.remove_document_for_ui(
        "project-1", "a.pdf", documents, []
    )

    assert captured == {"project_id": "project-1", "file_name": "a.pdf"}
    assert remaining_documents == []
    assert status.startswith("✅")


def test_remove_document_for_ui_supports_multi_select(monkeypatch) -> None:
    removed = []
    monkeypatch.setattr(
        ui, "remove_document",
        lambda project_id, file_name: removed.append(file_name),
    )
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "c.pdf", "page_start": 1, "page_end": 1},
    ]
    chunks = [
        TextChunk(1, "a", (1,), "a.pdf"),
        TextChunk(2, "b", (1,), "b.pdf"),
        TextChunk(3, "c", (1,), "c.pdf"),
    ]

    status, remaining_documents, remaining_chunks, *_ = ui.remove_document_for_ui(
        "project-1", ["a.pdf", "b.pdf"], documents, chunks
    )

    assert sorted(removed) == ["a.pdf", "b.pdf"]
    assert [doc["file_name"] for doc in remaining_documents] == ["c.pdf"]
    assert [chunk.document for chunk in remaining_chunks] == ["c.pdf"]
    assert "2 份文件" in status


def test_add_document_for_ui_accepts_multiple_files_at_once(monkeypatch) -> None:
    def fake_extract_pdf(path):
        name = Path(path).stem
        return [PageText(1, name)], []

    monkeypatch.setattr(ui, "extract_pdf", fake_extract_pdf)

    result = ui.add_document_for_ui(
        ["a.pdf", "b.pdf"], 1500, 200, [], []
    )
    (
        status, rows, documents, chunks, active_preview, active_chunks,
        document_status, documents_rows, pdf_reset, remove_choices,
    ) = result

    assert [doc["file_name"] for doc in documents] == ["a.pdf", "b.pdf"]
    assert [chunk.number for chunk in chunks] == [1, 2]
    assert [chunk.document for chunk in chunks] == ["a.pdf", "b.pdf"]
    assert active_preview["file_name"] == "b.pdf"
    assert "a.pdf" in status and "b.pdf" in status
    assert "2 份文件" in status
    assert status.count("- ✅") == 2
    assert status.splitlines()[0].startswith("- ✅「a.pdf」")
    assert status.splitlines()[1].startswith("- ✅「b.pdf」")


def test_request_stop_for_ui_sets_flag_on_shared_control() -> None:
    control = ui.RunControl()

    status = ui.request_stop_for_ui(control)

    assert status.startswith("⏹")
    with pytest.raises(ui.RunCancelled):
        control.check()


def test_toggle_pause_for_ui_flips_state_and_button_label() -> None:
    control = ui.RunControl()

    status, button_update = ui.toggle_pause_for_ui(control)
    assert status.startswith("⏸")
    assert button_update["value"] == "▶ 繼續"
    assert control.is_paused is True

    status, button_update = ui.toggle_pause_for_ui(control)
    assert status.startswith("▶")
    assert button_update["value"] == "⏸ 暫停"
    assert control.is_paused is False


def test_plan_schema_for_ui_reports_stopped_status(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "plan_graph_schema",
        lambda *args, **kwargs: (_ for _ in ()).throw(ui.RunCancelled("stopped")),
    )

    status, schema_text = ui.plan_schema_for_ui(
        "http://models/v1", "key", "llm", 0.3, "平衡", 3,
        "全部頁面", 10, [TextChunk(1, "text", (1,))], ui.RunControl(),
    )

    assert status.startswith("⏹")
    assert schema_text == ""


def test_extract_graph_for_ui_reports_stopped_status(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "extract_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(ui.RunCancelled("stopped")),
    )
    schema = {"entity_types": [{"name": "DEVICE"}], "relationship_types": [{"name": "USES"}]}

    result = ui.extract_graph_for_ui(
        "http://models/v1", "key", "llm", 0.2, 3,
        [TextChunk(1, "text", (1,))], json.dumps(schema), {}, ui.RunControl(),
    )

    assert result == ("⏹ 已停止（使用者中止抽取）。", [], [], {})


def test_plan_schema_for_ui_returns_editable_json(monkeypatch) -> None:
    from manual_graphrag.graph_service import SchemaPlan

    monkeypatch.setattr(
        ui,
        "plan_graph_schema",
        lambda *args, **kwargs: SchemaPlan(
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
        "平衡",
        3,
        "全部頁面",
        10,
        [TextChunk(1, "text", (1,))],
        ui.RunControl(),
    )

    assert status.startswith("✅")
    assert "全部 1 頁、5 個 chunk" in status
    assert "共 2 批、1 輪整合" in status
    assert "粒度：平衡。" in status
    assert "類型上限" not in status
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
    monkeypatch.setattr(ui, "extract_graph", lambda *args, **kwargs: extraction)
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
        3,
        [TextChunk(1, "text", (3,))],
        json.dumps(schema),
        [{"file_name": "manual.pdf"}],
        ui.RunControl(),
    )

    assert status.startswith("✅ 已處理 1 個 chunk")
    assert entities[0][:2] == ["設備 A", "DEVICE"]
    assert relationships[0][:3] == ["設備 A", "USES", "設備 B"]
    assert state["document"] == "manual.pdf"
    assert state["temperature"] == 0.2
    assert "max_output_tokens" not in state
    assert state["max_concurrent_requests"] == 3
    assert state["chunks"][0]["text"] == "text"
    assert state["neo4j_imported"] is False
    assert "進行 Embedding 並匯入 Neo4j" in status


def test_extract_graph_for_ui_rejects_invalid_schema() -> None:
    result = ui.extract_graph_for_ui(
        "http://models/v1", "", "llm", 0.2, 3, [], "not-json", {}, ui.RunControl()
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
        "0.0300", "3", "2", "",
    ]]


def test_evaluation_preferences_keep_generation_and_test_models_separate(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ui, "load_project", lambda project_id: {"evaluation": {}})
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status = ui.save_evaluation_preferences_for_ui(
        "project", "generation-model", "test-model", 12, "基本檢索", 6, 4, 5
    )

    assert status.startswith("✅")
    assert captured["evaluation"]["preferences"] == {
        "generation_model": "generation-model",
        "test_model": "test-model",
        "question_count": 12,
        "retrieval_mode": "基本檢索",
        "top_k": 6,
        "generation_max_concurrent_requests": 4,
        "test_max_concurrent_requests": 5,
    }


def test_load_evaluation_supports_legacy_shared_model(monkeypatch) -> None:
    monkeypatch.setattr(ui, "load_project", lambda project_id: {
        "evaluation": {"preferences": {"model": "legacy-model", "retrieval_mode": "GraphRAG"}}
    })

    loaded = ui.load_evaluation_for_ui("project")

    assert loaded[8] == 3
    assert loaded[9] == 3
    assert loaded[3:5] == ("legacy-model", "legacy-model")
    assert loaded[6] == "關聯擴展檢索"


def test_project_list_refreshes_on_page_load_and_tab_select_without_focus_rerender(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = build_app()
    selector = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("label") == "現有專案"
    )
    assert selector["props"]["choices"] == []
    dependencies = [
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("refresh_projects_for_ui")
    ]
    triggers = {
        target[1] for dependency in dependencies for target in dependency["targets"]
    }
    assert all(len(dependency["inputs"]) == 1 for dependency in dependencies)
    assert triggers == {"load", "select"}
    assert "focus" not in triggers
    assert all(dependency["outputs"] == [selector["id"]] for dependency in dependencies)

    project = ui.create_project("新增專案")
    for dependency in dependencies:
        update = app.fns[dependency["id"]].fn()
        assert update["choices"] == [("新增專案", project["project_id"])]
        assert update["value"] is None

    selected = ui.refresh_projects_for_ui(project)
    assert selected["choices"] == [("新增專案", project["project_id"])]
    assert selected["value"] == project["project_id"]

    ui.delete_project(project["project_id"])
    deleted = ui.refresh_projects_for_ui(project)
    assert deleted["choices"] == []
    assert deleted["value"] is None


def test_unselected_models_report_actionable_errors_without_network() -> None:
    plan = ui.plan_schema_for_ui(
        "", "", None, 0, "平衡", 1, "全部頁面", 1, [], ui.RunControl(),
    )
    extraction = ui.extract_graph_for_ui("", "", None, 0, 1, [], "{}", [], ui.RunControl())
    generation = ui.generate_evaluation_for_ui("project", "", "", None, None, 1, "基本檢索", 1, [])
    evaluation = ui.run_evaluation_for_ui(
        "project", "", "", "", "", "", "", "", "", None, "基本檢索", 1,
        {"questions": [{"question": "Q"}]},
    )
    answer = ui.answer_question_for_ui("", "", "", "", "", "", "", "", None, "Q", "基本檢索", 1)
    embedding = ui.import_graph_for_ui("", "", "", "", "", "", None, {"run_id": "run"})
    for result in [plan, extraction, generation, evaluation, answer, embedding]:
        assert result[0].startswith("❌")
        assert "選擇" in result[0]


def test_model_selections_are_not_saved_in_env(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ui.persist_env_settings("", "", "", "", "", "", "", "", "build", "embed", "answer")
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BUILD_MODEL" not in content
    assert "EMBEDDING_MODEL" not in content
    assert "ANSWER_MODEL" not in content
