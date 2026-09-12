from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import OLLAMA_DEFAULT_BASE_URL
from .env_store import load_env, save_env


OPENAI_MODELS_PATH = Path(__file__).resolve().parents[2] / "config/openai_models.json"
PROFILE_KEYS = {"llm": "MODEL_SERVICE_PROFILES", "embedding": "EMBEDDING_SERVICE_PROFILES"}
MODEL_COUNTS = {"llm": 5, "embedding": 1}


def openai_models(kind: str) -> list[str]:
    try:
        data = json.loads(OPENAI_MODELS_PATH.read_text(encoding="utf-8"))
        models = data[kind]
        if not isinstance(models, list) or not models or any(
            not isinstance(model, str) or not model.strip() for model in models
        ):
            raise ValueError()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"OpenAI 模型設定檔無效：{OPENAI_MODELS_PATH.name}（{kind}）") from exc
    return list(dict.fromkeys(model.strip() for model in models))


def provider_for_url(base_url: str) -> str:
    try:
        return "Ollama" if urlsplit(base_url).port == 11434 else "OpenAI"
    except ValueError:
        return "OpenAI"


def load_service_settings(kind: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    env = load_env() if env is None else env
    prefix = "MODEL" if kind == "llm" else "EMBEDDING"
    base = env[f"{prefix}_API_BASE"]
    active = provider_for_url(base)
    current_models = (
        [env["BUILD_MODEL"], env["BUILD_MODEL"], env["ANSWER_MODEL"], env["ANSWER_MODEL"], env["ANSWER_MODEL"]]
        if kind == "llm" else [env["EMBEDDING_MODEL"]]
    )
    profiles = {
        provider: {"base_url": url, "api_key": "", "rows": [], "models": [None] * MODEL_COUNTS[kind]}
        for provider, url in [("OpenAI", "https://api.openai.com/v1"), ("Ollama", OLLAMA_DEFAULT_BASE_URL)]
    }
    profiles[active].update(base_url=base or profiles[active]["base_url"], api_key=env[f"{prefix}_API_KEY"], models=current_models)
    if active == "Ollama":
        profiles[active]["rows"] = [[True, model] for model in dict.fromkeys(current_models) if model]
    raw = env.get(PROFILE_KEYS[kind], "")
    if raw:
        try:
            saved = json.loads(raw)
            if saved["active"] not in profiles:
                raise ValueError()
            for provider in profiles:
                profile = saved["profiles"][provider]
                if not isinstance(profile["base_url"], str) or not isinstance(profile["api_key"], str):
                    raise ValueError()
                if not isinstance(profile["models"], list) or len(profile["models"]) != MODEL_COUNTS[kind] or any(
                    model is not None and not isinstance(model, str) for model in profile["models"]
                ):
                    raise ValueError()
                if not isinstance(profile["rows"], list) or any(
                    not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], bool)
                    or not isinstance(row[1], str) for row in profile["rows"]
                ):
                    raise ValueError()
                profiles[provider] = {key: deepcopy(profile[key]) for key in ("base_url", "api_key", "rows", "models")}
            active = saved["active"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("服務設定格式錯誤，請檢查 .env 中的 " + PROFILE_KEYS[kind]) from exc
    # Connection success is tied to this session, never trusted from disk.
    for profile in profiles.values():
        profile["connected"] = False
        profile["status"] = "請先測試連線。"
    profiles["OpenAI"]["rows"] = []
    return {"kind": kind, "active": active, "profiles": profiles}


def service_choices(state: dict[str, Any]) -> list[str]:
    profile = state["profiles"][state["active"]]
    if state["active"] == "OpenAI":
        return openai_models(state["kind"]) if profile["connected"] else []
    return list(dict.fromkeys(row[1] for row in profile["rows"] if row[0]))


def save_service_settings(state: dict[str, Any]) -> None:
    saved = {"active": state["active"], "profiles": {
        provider: {key: profile[key] for key in ("base_url", "api_key", "rows", "models")}
        for provider, profile in state["profiles"].items()
    }}
    save_env({PROFILE_KEYS[state["kind"]]: json.dumps(saved, ensure_ascii=False)})


def capture_service_settings(
    state: dict[str, Any], base_url: str, api_key: str, rows: list[list[Any]], models: list[str | None],
) -> dict[str, Any]:
    state = deepcopy(state)
    profile = state["profiles"][state["active"]]
    changed = (profile["base_url"], profile["api_key"]) != (base_url, api_key)
    profile.update(base_url=base_url, api_key=api_key)
    if changed:
        profile.update(connected=False, rows=[], models=[None] * MODEL_COUNTS[state["kind"]], status="設定已變更，請重新測試連線或取得模型清單。")
    else:
        if state["active"] == "Ollama":
            # Only checkboxes from the fetched catalog may change.
            checked = {row[1] for row in rows if len(row) == 2 and row[0] is True}
            profile["rows"] = [[model in checked, model] for _, model in profile["rows"]]
        if state["active"] == "Ollama" or profile["connected"]:
            allowed = service_choices(state) if state["active"] == "Ollama" else models
            profile["models"] = [model if model in allowed else None for model in models]
    return state


def restore_service_settings(
    state: dict[str, Any], base_url: str, api_key: str, models: dict[int, str | None],
) -> dict[str, Any]:
    state = deepcopy(state)
    provider = next((name for name, p in state["profiles"].items() if p["base_url"] == base_url), provider_for_url(base_url))
    state["active"] = provider
    profile = state["profiles"][provider]
    if (profile["base_url"], profile["api_key"]) != (base_url, api_key):
        profile.update(base_url=base_url, api_key=api_key, connected=False, rows=[], status="請先測試連線或取得模型清單。")
    for index, model in models.items():
        profile["models"][index] = model
    # Saved model names cannot expand the OpenAI allowlist or bypass Ollama checkboxes.
    return state
