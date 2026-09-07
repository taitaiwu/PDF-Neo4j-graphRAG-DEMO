from app import launch_settings


def test_launch_settings_default_to_local_browser(monkeypatch) -> None:
    monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    monkeypatch.delenv("GRADIO_INBROWSER", raising=False)

    assert launch_settings() == {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "share": False,
        "inbrowser": True,
    }


def test_launch_settings_accept_docker_environment(monkeypatch) -> None:
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "8000")
    monkeypatch.setenv("GRADIO_INBROWSER", "false")

    settings = launch_settings()

    assert settings["server_name"] == "0.0.0.0"
    assert settings["server_port"] == 8000
    assert settings["inbrowser"] is False
