from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, **payload}
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != 1:
        raise ValueError("不支援的 JSON schema_version")
    return data
