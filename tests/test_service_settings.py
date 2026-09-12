import json

import pytest

from manual_graphrag import service_settings as settings, ui
from manual_graphrag.env_store import DEFAULTS


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def choices(provider, models):
    return [(f"{provider}｜{model}", model) for model in models]


def service(kind="llm"):
    return settings.load_service_settings(kind, dict(DEFAULTS))


def act(state, action, *, provider=None, base=None, key=None, rows=None, models=None):
    p = state["profiles"][state["active"]]
    displayed = ui.render_service_for_ui(state)
    return ui.service_action_for_ui(
        action, provider or state["active"], state,
        p["base_url"] if base is None else base,
        p["api_key"] if key is None else key,
        p["rows"] if rows is None else rows,
        *([u["value"] for u in displayed[7:]] if models is None else models),
    )


@pytest.mark.parametrize("kind,expected", [
    ("llm", ["gpt-4.1-mini", "gpt-4o-mini"]),
    ("embedding", ["text-embedding-3-small", "text-embedding-3-large"]),
])
def test_openai_requires_connection_then_only_offers_json_allowlist(monkeypatch, kind, expected):
    calls = []
    monkeypatch.setattr(ui, "check_model_connection", lambda base, key: calls.append((base, key)))
    monkeypatch.setattr(ui, "list_models", lambda *args: pytest.fail("OpenAI must not fetch the model catalog"))
    state = service(kind)
    initial = ui.render_service_for_ui(state)
    assert all(u["choices"] == [] for u in initial[7:])
    tested = act(state, "test", key="test-key")
    assert calls == [("https://api.openai.com/v1", "test-key")]
    assert tested[4]["visible"] is True
    assert tested[5]["visible"] is False
    assert tested[3]["visible"] is False
    assert all(u["choices"] == choices("OpenAI", expected) for u in tested[7:])
    assert tested[6].startswith("✅")
    # Neither a forged table nor the hidden fetch action may add extra OpenAI models.
    edited = act(tested[0], "edit", rows=[[True, "expensive-model"]], models=["expensive-model"] * settings.MODEL_COUNTS[kind])
    assert all(u["choices"] == choices("OpenAI", expected) and u["value"] is None for u in edited[7:])
    fetched = act(tested[0], "fetch")
    assert all(u["choices"] == choices("OpenAI", expected) for u in fetched[7:])


@pytest.mark.parametrize("kind", ["llm", "embedding"])
def test_switch_preserves_both_profiles_credentials_checks_and_models(monkeypatch, kind):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    monkeypatch.setattr(ui, "list_models", lambda *args: ["local-chat", "local-embed"])
    count = settings.MODEL_COUNTS[kind]
    openai = act(service(kind), "test", key="openai-secret")
    openai = act(openai[0], "edit", models=[settings.openai_models(kind)[1]] * count)
    ollama = act(openai[0], "switch", provider="Ollama")
    assert ollama[1:3] == ("http://localhost:11434/v1", "")
    assert ollama[3]["visible"] is True
    assert ollama[4]["visible"] is False
    assert ollama[5]["visible"] is True
    ollama = act(ollama[0], "edit", base="http://ollama-server:11434/v1", key="local-secret")
    ollama = act(ollama[0], "fetch")
    assert ollama[3]["value"] == [[False, "local-chat"], [False, "local-embed"]]
    chosen = "local-chat" if kind == "llm" else "local-embed"
    rows = [[model == chosen, model] for model in ["local-chat", "local-embed"]]
    ollama = act(ollama[0], "edit", rows=rows)
    ollama = act(ollama[0], "edit", models=[chosen] * count)
    back = act(ollama[0], "switch", provider="OpenAI")
    assert back[1:3] == ("https://api.openai.com/v1", "openai-secret")
    assert all(u["value"] == settings.openai_models(kind)[1] for u in back[7:])
    again = act(back[0], "switch", provider="Ollama")
    assert again[1:3] == ("http://ollama-server:11434/v1", "local-secret")
    assert again[3]["value"] == rows
    assert all(u["choices"] == choices("OpenAI", settings.openai_models(kind)) + choices("Ollama", [chosen]) and u["value"] == chosen for u in again[7:])
    reloaded = settings.load_service_settings(kind)
    assert reloaded["active"] == "Ollama"
    assert reloaded["profiles"]["OpenAI"]["api_key"] == "openai-secret"
    assert reloaded["profiles"]["OpenAI"]["connected"] is False
    assert settings.service_choices(reloaded) == [chosen]
    assert reloaded["profiles"]["Ollama"]["models"] == [chosen] * count


def test_llm_and_embedding_ollama_catalogs_have_independent_checks(monkeypatch):
    monkeypatch.setattr(ui, "list_models", lambda *args: ["chat", "embed"])
    for kind, chosen in [("llm", "chat"), ("embedding", "embed")]:
        switched = act(service(kind), "switch", provider="Ollama")
        fetched = act(switched[0], "fetch")
        act(fetched[0], "edit", rows=[[chosen == name, name] for name in ["chat", "embed"]])
    assert settings.service_choices(settings.load_service_settings("llm")) == ["chat"]
    assert settings.service_choices(settings.load_service_settings("embedding")) == ["embed"]


def test_connection_failure_and_credential_edit_revoke_openai_models(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    connected = act(service(), "test", key="key")
    edited = act(connected[0], "edit", key="different")
    assert all(u["choices"] == [] for u in edited[7:])
    def fail(*args):
        raise ValueError("連線失敗")
    monkeypatch.setattr(ui, "check_model_connection", fail)
    failed = act(connected[0], "test")
    assert failed[6] == "❌ 連線失敗"
    assert all(u["choices"] == [] for u in failed[7:])


def test_failed_ollama_fetch_preserves_checked_catalog(monkeypatch):
    monkeypatch.setattr(ui, "list_models", lambda *args: ["chat", "embed"])
    fetched = act(act(service(), "switch", provider="Ollama")[0], "fetch")
    selected = act(fetched[0], "edit", rows=[[True, "chat"], [False, "embed"]])
    def fail(*args):
        raise ValueError("offline")
    monkeypatch.setattr(ui, "list_models", fail)
    failed = act(selected[0], "fetch")
    assert failed[3]["value"] == [[True, "chat"], [False, "embed"]]
    assert all(u["choices"] == choices("Ollama", ["chat"]) for u in failed[7:])
    assert failed[6] == "❌ offline"


def test_custom_json_controls_allowlist_and_invalid_json_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "openai_models.json"
    monkeypatch.setattr(settings, "OPENAI_MODELS_PATH", path)
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    path.write_text(json.dumps({"llm": ["custom-mini"], "embedding": ["custom-embed"]}))
    assert act(service(), "test")[7]["choices"] == choices("OpenAI", ["custom-mini"])
    path.write_text("{}")
    failed = act(service(), "test")
    assert "設定檔無效" in failed[6]
    assert failed[7]["choices"] == []


def test_saved_project_cannot_bypass_openai_allowlist_or_connection(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    project = ui.create_project("saved")
    ui.save_project(project["project_id"], {"settings": {
        "model_endpoint": "https://api.openai.com/v1", "api_key": "",
        "graph_llm_model": "gpt-expensive", "extraction_llm_model": "gpt-4o-mini",
        "answer_model": "gpt-expensive", "graph_embedding_model": "other-embed",
    }})
    before = ui.load_project_with_services_for_ui(project["project_id"], service(), service("embedding"))
    assert before[8]["choices"] == []
    llm = act(service(), "test")[0]
    embed = act(service("embedding"), "test")[0]
    loaded = ui.load_project_with_services_for_ui(project["project_id"], llm, embed)
    assert loaded[8]["value"] is None
    assert loaded[18]["value"] == "gpt-4o-mini"
    assert loaded[9]["value"] is None
    assert loaded[8]["choices"] == choices("OpenAI", settings.openai_models("llm"))
    assert loaded[9]["choices"] == choices("OpenAI", settings.openai_models("embedding"))


def test_evaluation_cannot_restore_unapproved_models(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    project = ui.create_project("saved")
    ui.save_project(project["project_id"], {"evaluation": {"preferences": {
        "generation_model": "unapproved", "test_model": "gpt-4o-mini",
    }}})
    state = act(service(), "test")[0]
    loaded = ui.load_evaluation_with_services_for_ui(project["project_id"], state)
    assert loaded[3]["value"] is None
    assert loaded[4]["value"] == "gpt-4o-mini"


def test_profile_load_rejects_invalid_json_without_exposing_keys():
    env = dict(DEFAULTS, MODEL_SERVICE_PROFILES='{"secret":"sensitive"}')
    with pytest.raises(ValueError, match="MODEL_SERVICE_PROFILES") as error:
        settings.load_service_settings("llm", env)
    assert "sensitive" not in str(error.value)


def test_gradio_events_switch_connect_filter_and_restore(monkeypatch):
    import asyncio
    from gradio.state_holder import SessionState

    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    monkeypatch.setattr(ui, "list_models", lambda *args: ["chat-local", "embed-local"])
    app = ui.build_app()
    session = SessionState(app)
    components = app.config["components"]
    def component(label):
        return next(c for c in components if c.get("props", {}).get("label") == label)
    def button(value):
        return next(c for c in components if c["type"] == "button" and c["props"]["value"] == value)
    def event(component_id, trigger):
        return next(d for d in app.config["dependencies"] if (component_id, trigger) in d["targets"])
    provider = component("模型服務來源")
    test = event(button("測試模型服務連線")["id"], "click")
    switch = event(provider["id"], "input")
    fetch = event(button("獲得模型清單")["id"], "click")
    edit = event(component("LLM 模型清單")["id"], "input")
    async def run():
        inputs = ["OpenAI", None, "https://api.openai.com/v1", "test-key", {"headers": ["使用", "模型名稱"], "data": []}, *([None] * 5)]
        connected = (await app.process_api(test["id"], inputs, state=session))["data"]
        assert connected[7]["choices"] == [["OpenAI｜gpt-4.1-mini", "gpt-4.1-mini"], ["OpenAI｜gpt-4o-mini", "gpt-4o-mini"]]
        inputs[4] = connected[3]["value"]
        inputs[5:] = [u["value"] for u in connected[7:]]
        inputs[0] = "Ollama"
        switched = (await app.process_api(switch["id"], inputs, state=session))["data"]
        assert switched[1:3] == ["http://localhost:11434/v1", ""]
        inputs[2:5] = [switched[1], switched[2], switched[3]["value"]]
        inputs[5:] = [u["value"] for u in switched[7:]]
        fetched = (await app.process_api(fetch["id"], inputs, state=session))["data"]
        inputs[4] = fetched[3]["value"]
        inputs[4]["data"][0][0] = True
        checked = (await app.process_api(edit["id"], inputs, state=session))["data"]
        assert checked[7]["choices"] == [["OpenAI｜gpt-4.1-mini", "gpt-4.1-mini"], ["OpenAI｜gpt-4o-mini", "gpt-4o-mini"], ["Ollama｜chat-local", "chat-local"]]
        inputs[4] = checked[3]["value"]
        inputs[0] = "OpenAI"
        restored = (await app.process_api(switch["id"], inputs, state=session))["data"]
        assert restored[2] == "test-key"
        assert restored[7]["value"] == "gpt-4.1-mini"
        project = ui.create_project("restored")
        ui.save_project(project["project_id"], {"settings": {
            "model_endpoint": "https://api.openai.com/v1", "api_key": "test-key",
            "graph_llm_model": "gpt-4o-mini", "extraction_llm_model": "outside-list",
        }})
        refresh = event(component("現有專案")["id"], "focus")
        await app.process_api(refresh["id"], [], state=session)
        load = event(button("載入專案")["id"], "click")
        loaded = (await app.process_api(load["id"], [project["project_id"], None, None], state=session))["data"]
        assert loaded[8]["value"] == "gpt-4o-mini"
        assert loaded[18]["value"] is None
        reload = event(button("重新讀取 .env")["id"], "click")
        reloaded = (await app.process_api(reload["id"], [], state=session))["data"]
        assert reloaded[12]["choices"] == [["Ollama｜chat-local", "chat-local"]]
    asyncio.run(run())


def test_reload_restores_both_profiles_and_revokes_openai_connection(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    act(service(), "test", key="llm-key")
    act(service("embedding"), "test", key="embedding-key")
    loaded = ui.reload_env_with_services_for_ui()
    assert loaded[4] == "OpenAI"
    assert loaded[7] == "llm-key"
    assert loaded[17] == "OpenAI"
    assert loaded[20] == "embedding-key"
    assert loaded[12]["choices"] == []
    assert loaded[25]["choices"] == []


def test_invalid_json_after_successful_connection_revokes_models(tmp_path, monkeypatch):
    path = tmp_path / "openai_models.json"
    monkeypatch.setattr(settings, "OPENAI_MODELS_PATH", path)
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    path.write_text(json.dumps({"llm": ["mini"]}))
    connected = act(service(), "test")
    path.write_text("{}")
    failed = ui.service_action_for_ui("test", "OpenAI", connected[0], connected[1], connected[2], [], *(["mini"] * 5))
    assert "設定檔無效" in failed[6]
    assert failed[7]["choices"] == []


def test_both_provider_models_remain_visible_and_route_to_their_own_service(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    monkeypatch.setattr(ui, "list_models", lambda *args: ["local-chat", "local-embed"])
    connected = act(service(), "test", key="openai-key")
    ollama = act(connected[0], "switch", provider="Ollama")
    ollama = act(ollama[0], "edit", base="http://ollama:11434/v1")
    fetched = act(ollama[0], "fetch")
    combined = act(
        fetched[0], "edit",
        rows=[[True, "local-chat"], [False, "local-embed"]],
    )
    assert combined[7]["choices"] == (
        choices("OpenAI", ["gpt-4.1-mini", "gpt-4o-mini"])
        + choices("Ollama", ["local-chat"])
    )
    assert settings.resolve_model_service(combined[0], "gpt-4o-mini") == (
        "https://api.openai.com/v1", "openai-key", "gpt-4o-mini",
    )
    assert settings.resolve_model_service(combined[0], "local-chat") == (
        "http://ollama:11434/v1", "", "local-chat",
    )
    assert ui.resolve_model_credentials_for_ui(
        combined[0], "local-chat"
    ) == ("http://ollama:11434/v1", "")


def test_duplicate_model_name_uses_active_provider_credentials(monkeypatch):
    monkeypatch.setattr(settings, "openai_models", lambda kind: ["same-model"])
    state = service()
    state["profiles"]["OpenAI"].update(connected=True, api_key="openai-key")
    state["profiles"]["Ollama"].update(
        base_url="http://ollama:11434/v1", rows=[[True, "same-model"]],
    )
    state["active"] = "OpenAI"
    assert settings.resolve_model_service(state, "same-model")[:2] == (
        "https://api.openai.com/v1", "openai-key",
    )
    state["active"] = "Ollama"
    assert settings.resolve_model_service(state, "same-model")[:2] == (
        "http://ollama:11434/v1", "",
    )


def test_unavailable_model_cannot_be_routed():
    with pytest.raises(ValueError, match="目前不可用"):
        settings.resolve_model_service(service(), "unknown")
