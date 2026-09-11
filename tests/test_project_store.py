from pathlib import Path

import pytest

from manual_graphrag.project_store import (
    append_question,
    create_project,
    delete_project,
    list_projects,
    load_project,
    remove_document,
    save_project,
)


def test_project_round_trip_and_listing(tmp_path) -> None:
    project = create_project("設備手冊", tmp_path)
    saved = save_project(
        project["project_id"],
        {"settings": {"api_key": "secret", "chunk_size": 1200}},
        root=tmp_path,
    )

    assert saved["settings"]["api_key"] == "secret"
    assert load_project(project["project_id"], tmp_path)["settings"]["chunk_size"] == 1200
    assert list_projects(tmp_path) == [("設備手冊", project["project_id"])]


def test_save_project_copies_documents(tmp_path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"pdf")
    root = tmp_path / "projects"
    project = create_project("Manual", root)

    saved = save_project(project["project_id"], {}, [str(source)], root)

    stored = Path(saved["documents"][0]["path"])
    assert stored.read_bytes() == b"pdf"
    assert stored.parent.name == "documents"


def test_save_project_accumulates_multiple_documents(tmp_path) -> None:
    first = tmp_path / "a.pdf"
    first.write_bytes(b"a")
    second = tmp_path / "b.pdf"
    second.write_bytes(b"b")
    root = tmp_path / "projects"
    project = create_project("Multi", root)

    save_project(project["project_id"], {}, [str(first)], root)
    saved = save_project(project["project_id"], {}, [str(second)], root)

    assert [doc["name"] for doc in saved["documents"]] == ["a.pdf", "b.pdf"]


def test_remove_document_deletes_file_and_entry(tmp_path) -> None:
    first = tmp_path / "a.pdf"
    first.write_bytes(b"a")
    second = tmp_path / "b.pdf"
    second.write_bytes(b"b")
    root = tmp_path / "projects"
    project = create_project("Multi", root)
    save_project(project["project_id"], {}, [str(first), str(second)], root)

    updated = remove_document(project["project_id"], "a.pdf", root)

    assert [doc["name"] for doc in updated["documents"]] == ["b.pdf"]
    assert not (root / project["project_id"] / "documents" / "a.pdf").exists()


def test_load_project_migrates_legacy_single_document(tmp_path) -> None:
    root = tmp_path / "projects"
    project = create_project("Legacy", root)
    from manual_graphrag.storage import write_json

    project_path = root / project["project_id"] / "project.json"
    legacy = {
        **project,
        "document": {"name": "old.pdf", "path": "old.pdf"},
        "preview_state": {"file_name": "old.pdf"},
    }
    legacy.pop("documents", None)
    legacy.pop("documents_meta", None)
    write_json(project_path, legacy)

    loaded = load_project(project["project_id"], root)

    assert loaded["documents"] == [{"name": "old.pdf", "path": "old.pdf"}]
    assert loaded["documents_meta"] == [{"file_name": "old.pdf"}]


def test_append_question_preserves_history(tmp_path) -> None:
    project = create_project("QA", tmp_path)
    append_question(project["project_id"], {"question": "Q1", "answer": "A1"}, tmp_path)
    updated = append_question(
        project["project_id"], {"question": "Q2", "answer": "A2"}, tmp_path
    )

    assert [item["question"] for item in updated["questions"]] == ["Q1", "Q2"]
    assert all(item["asked_at"] for item in updated["questions"])


def test_create_project_validates_name_and_collision(tmp_path) -> None:
    with pytest.raises(ValueError, match="專案名稱"):
        create_project("  ", tmp_path)
    create_project("same", tmp_path)
    with pytest.raises(ValueError, match="相同識別碼"):
        create_project("same", tmp_path)


def test_delete_project_removes_only_selected_project(tmp_path) -> None:
    first = create_project("first", tmp_path)
    second = create_project("second", tmp_path)

    deleted_name = delete_project(first["project_id"], tmp_path)

    assert deleted_name == "first"
    assert not (tmp_path / first["project_id"]).exists()
    assert load_project(second["project_id"], tmp_path)["name"] == "second"
    with pytest.raises(ValueError, match="找不到"):
        delete_project(first["project_id"], tmp_path)
