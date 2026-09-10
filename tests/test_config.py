import pytest

from manual_graphrag.config import (
    OPENAI_EMBEDDING_MODELS,
    OPENAI_LLM_MODELS,
    BuildConfig,
    model_choices,
    public_settings,
)


def test_build_config_accepts_valid_values() -> None:
    config = BuildConfig("model-a", "embed-a", chunk_size=500, chunk_overlap=50)
    assert config.chunk_size == 500


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 101)])
def test_build_config_rejects_invalid_chunk_values(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        BuildConfig("model-a", "embed-a", chunk_size=size, chunk_overlap=overlap)


def test_public_settings_removes_secrets() -> None:
    safe = public_settings({"api_key": "secret", "neo4j_password": "secret", "uri": "bolt://db"})
    assert safe == {"uri": "bolt://db"}


def test_model_choices_include_openai_options_and_keep_custom_current_value() -> None:
    llm_choices = model_choices("provider/custom", defaults=OPENAI_LLM_MODELS)
    embedding_choices = model_choices(
        "provider/embed", defaults=OPENAI_EMBEDDING_MODELS
    )

    assert llm_choices[0] == "provider/custom"
    assert {"gpt-4.1", "gpt-4o-mini"}.issubset(llm_choices)
    assert embedding_choices[0] == "provider/embed"
    assert {
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    }.issubset(embedding_choices)
