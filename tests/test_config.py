import pytest

from manual_graphrag.config import BuildConfig, public_settings


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
