import gradio as gr
from manual_graphrag import ui
from manual_graphrag.chunking import PageText, TextChunk

from manual_graphrag.ui import build_app, connection_summary, persist_env_settings


def test_build_app_returns_blocks() -> None:
    assert isinstance(build_app(), gr.Blocks)


def test_connection_summary_does_not_expose_secrets() -> None:
    status, settings = connection_summary(
        "bolt://localhost:7687", "neo4j", "neo4j", "password", "http://localhost:11434/v1", "key"
    )
    assert status.startswith("✅")
    assert "password" not in settings
    assert "api_key" not in settings


def test_persist_env_settings_writes_all_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    status = persist_env_settings(
        "bolt://db", "neo4j", "user", "pass", "http://models", "key", "build", "embed", "answer"
    )
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert status.startswith("✅")
    assert 'NEO4J_PASSWORD="pass"' in content
    assert 'ANSWER_MODEL="answer"' in content


def test_preview_page_filters_chunks_and_reports_page() -> None:
    chunks = [
        TextChunk(1, "first", (1,)),
        TextChunk(2, "shared", (1, 2)),
        TextChunk(3, "second", (2,)),
    ]

    status, rows = ui.preview_page(2, chunks, {"page_count": 3})

    assert status == "第 2 / 3 頁；顯示 2 個相關 chunk。"
    assert [row[0] for row in rows] == [2, 3]


def test_preview_page_reports_empty_page() -> None:
    status, rows = ui.preview_page(3, [TextChunk(1, "first", (1,))], {"page_count": 3})

    assert status == "第 3 / 3 頁；本頁沒有可解析文字或相關 chunk。"
    assert rows == []


def test_page_navigation_stays_within_document_bounds() -> None:
    state = {"page_count": 3}

    assert ui.previous_page(1, state) == 1
    assert ui.previous_page(3, state) == 2
    assert ui.next_page(2, state) == 3
    assert ui.next_page(3, state) == 3


def test_preview_pdf_initializes_page_navigation(monkeypatch) -> None:
    pages = [PageText(1, "one"), PageText(2, ""), PageText(3, "three")]
    chunks = [
        TextChunk(1, "one", (1,)),
        TextChunk(2, "three", (3,)),
    ]
    monkeypatch.setattr(ui, "extract_pdf", lambda path: (pages, [2]))
    monkeypatch.setattr(ui, "chunk_pages", lambda *args: chunks)

    result = ui.preview_pdf("manual.pdf", "build", "embed", 100, 0, 0, 100)
    status, rows, state, stored_chunks, slider_update, page_status = result

    assert status.startswith("已解析 3 頁，產生 2 個 chunk。")
    assert [row[0] for row in rows] == [1]
    assert state["page_count"] == 3
    assert "chunks" not in state
    assert stored_chunks is chunks
    assert slider_update["maximum"] == 3
    assert slider_update["value"] == 1
    assert page_status == "第 1 / 3 頁；顯示 1 個相關 chunk。"
