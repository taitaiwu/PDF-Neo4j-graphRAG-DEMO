from __future__ import annotations

from dataclasses import asdict, dataclass


OPENAI_LLM_MODELS = (
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
)

OPENAI_EMBEDDING_MODELS = (
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
)

# Ollama exposes an OpenAI-compatible API at this base URL by default; pointing
# the existing generic chat/embeddings client at it avoids per-token API costs
# for local models (see 交接紀錄.md).
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def model_choices(*current_models: str, defaults: tuple[str, ...]) -> list[str]:
    """Return unique model choices while retaining provider-specific current values."""
    return list(
        dict.fromkeys(
            model.strip()
            for model in (*current_models, *defaults)
            if model and model.strip()
        )
    )


@dataclass(frozen=True)
class BuildConfig:
    build_model: str
    embedding_model: str
    chunk_size: int = 1500
    chunk_overlap: int = 200
    temperature: float = 0.0
    max_output_tokens: int = 2048

    def __post_init__(self) -> None:
        if not self.build_model.strip():
            raise ValueError("請填寫建圖模型名稱")
        if not self.embedding_model.strip():
            raise ValueError("請填寫 embedding 模型名稱")
        if not 100 <= self.chunk_size <= 10000:
            raise ValueError("chunk_size 必須介於 100 到 10,000")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap 必須大於等於 0 且小於 chunk_size")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必須介於 0 到 2")
        if self.max_output_tokens < 1:
            raise ValueError("最大輸出 tokens 必須大於 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def public_settings(settings: dict[str, object]) -> dict[str, object]:
    """Remove secrets before displaying or exporting settings."""
    blocked = {"password", "api_key", "token", "secret"}
    return {
        key: value
        for key, value in settings.items()
        if key.lower() not in blocked and not any(word in key.lower() for word in blocked)
    }
