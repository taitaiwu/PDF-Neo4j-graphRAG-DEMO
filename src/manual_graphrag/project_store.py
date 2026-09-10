from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .storage import read_json, write_json


PROJECTS_DIR = Path("data/projects")


def _project_id(name: str) -> str:
    normalized = re.sub(r"[^\w\-]+", "-", name.strip(), flags=re.UNICODE).strip("-_")
    return normalized[:60] or f"project-{uuid4().hex[:8]}"


def list_projects(root: str | Path = PROJECTS_DIR) -> list[tuple[str, str]]:
    projects: list[tuple[str, str]] = []
    base = Path(root)
    if not base.exists():
        return projects
    for config_path in base.glob("*/project.json"):
        try:
            data = read_json(config_path)
            projects.append((str(data.get("name") or config_path.parent.name), config_path.parent.name))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(projects, key=lambda item: item[0].casefold())


def create_project(name: str, root: str | Path = PROJECTS_DIR) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("請輸入專案名稱")
    base = Path(root)
    project_id = _project_id(clean_name)
    target = base / project_id
    if target.exists():
        raise ValueError("已有相同識別碼的專案，請改用其他名稱")
    now = datetime.now(timezone.utc).isoformat()
    project = {
        "project_id": project_id,
        "name": clean_name,
        "created_at": now,
        "updated_at": now,
        "settings": {},
        "document": {},
        "preview_state": {},
        "chunks": [],
        "graph_state": {},
        "questions": [],
    }
    write_json(target / "project.json", project)
    return project


def load_project(project_id: str, root: str | Path = PROJECTS_DIR) -> dict[str, Any]:
    if not project_id or Path(project_id).name != project_id:
        raise ValueError("請選擇有效的專案")
    target = Path(root) / project_id / "project.json"
    if not target.is_file():
        raise ValueError("找不到指定專案")
    return read_json(target)


def delete_project(project_id: str, root: str | Path = PROJECTS_DIR) -> str:
    project = load_project(project_id, root)
    target = Path(root) / project_id
    shutil.rmtree(target)
    return str(project["name"])


def save_project(
    project_id: str,
    payload: dict[str, Any],
    document_path: str | None = None,
    root: str | Path = PROJECTS_DIR,
) -> dict[str, Any]:
    current = load_project(project_id, root)
    project_dir = Path(root) / project_id
    document = dict(payload.get("document") or current.get("document") or {})
    if document_path:
        source = Path(document_path)
        if not source.is_file():
            raise ValueError("找不到要保存的 PDF 文件")
        documents_dir = project_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        destination = documents_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        document = {"name": source.name, "path": str(destination)}
    updated = {
        **current,
        **payload,
        "project_id": project_id,
        "name": current["name"],
        "created_at": current["created_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "document": document,
    }
    write_json(project_dir / "project.json", updated)
    return updated


def append_question(
    project_id: str, record: dict[str, Any], root: str | Path = PROJECTS_DIR
) -> dict[str, Any]:
    project = load_project(project_id, root)
    questions = list(project.get("questions") or [])
    questions.append({"asked_at": datetime.now(timezone.utc).isoformat(), **record})
    return save_project(project_id, {**project, "questions": questions}, root=root)

