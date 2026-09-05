from manual_graphrag.env_store import load_env, save_env


def test_env_round_trip_preserves_special_characters(tmp_path) -> None:
    path = tmp_path / ".env"
    save_env(
        {
            "NEO4J_PASSWORD": "p#a ss=word",
            "MODEL_API_KEY": "sk-test",
            "BUILD_MODEL": "model-a",
        },
        path,
    )
    loaded = load_env(path)
    assert loaded["NEO4J_PASSWORD"] == "p#a ss=word"
    assert loaded["MODEL_API_KEY"] == "sk-test"
    assert loaded["BUILD_MODEL"] == "model-a"


def test_save_env_preserves_unmanaged_values(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text("CUSTOM_VALUE=keep\nNEO4J_URI=old\n", encoding="utf-8")
    save_env({"NEO4J_URI": "bolt://new"}, path)
    content = path.read_text(encoding="utf-8")
    assert "CUSTOM_VALUE=keep" in content
    assert load_env(path)["NEO4J_URI"] == "bolt://new"
