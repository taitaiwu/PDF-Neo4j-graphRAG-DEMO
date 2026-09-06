import pytest

from manual_graphrag.chunking import (
    PageText,
    TextChunk,
    chunk_pages,
    preview_rows,
    preview_rows_for_page,
)


def test_chunk_pages_preserves_page_provenance() -> None:
    chunks = chunk_pages([PageText(1, "abcd"), PageText(2, "efgh")], 6, 2)
    assert [chunk.text for chunk in chunks] == ["abcd", "efgh"]
    assert [chunk.pages for chunk in chunks] == [(1,), (2,)]


def test_chunk_pages_splits_by_markdown_heading_hierarchy() -> None:
    markdown = "# Guide\n\nIntro\n\n## Install\n\nRun setup"

    chunks = chunk_pages([PageText(3, markdown)], 100, 0)

    assert [chunk.headings for chunk in chunks] == [
        ("Guide",),
        ("Guide", "Install"),
    ]
    assert chunks[0].text == "# Guide\n\nIntro"
    assert chunks[1].text == "# Guide\n\n## Install\n\nRun setup"
    assert all(chunk.pages == (3,) for chunk in chunks)


def test_chunk_pages_keeps_fenced_code_block_together() -> None:
    markdown = "# Example\n\nBefore\n\n```python\nfirst = 1\n\nsecond = 2\n```\n\nAfter"

    chunks = chunk_pages([PageText(1, markdown)], 55, 0)

    code_chunks = [chunk for chunk in chunks if "```python" in chunk.text]
    assert len(code_chunks) == 1
    assert "first = 1\n\nsecond = 2" in code_chunks[0].text


def test_chunk_pages_does_not_treat_code_comments_as_headings() -> None:
    markdown = "# Example\n\n```python\n# not a heading\nvalue = 1\n```"

    chunks = chunk_pages([PageText(1, markdown)], 100, 0)

    assert len(chunks) == 1
    assert chunks[0].headings == ("Example",)
    assert "# not a heading" in chunks[0].text


def test_chunk_pages_only_character_splits_an_oversized_block() -> None:
    markdown = "# Long section\n\n" + "abcdefghij" * 12

    chunks = chunk_pages([PageText(5, markdown)], 50, 10)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert all(chunk.headings == ("Long section",) for chunk in chunks)
    assert all(chunk.pages == (5,) for chunk in chunks)


def test_chunk_pages_keeps_markdown_table_as_one_block() -> None:
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    markdown = f"# Data\n\nBefore\n\n{table}\n\nAfter"

    chunks = chunk_pages([PageText(1, markdown)], 50, 0)

    table_chunks = [chunk for chunk in chunks if "| A | B |" in chunk.text]
    assert len(table_chunks) == 1
    assert table in table_chunks[0].text


def test_chunk_pages_returns_empty_for_blank_pages() -> None:
    assert chunk_pages([PageText(1, "  ")], 100, 10) == []


def test_chunk_pages_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "text")], 10, 10)


def test_preview_rows_for_page_returns_all_related_chunks() -> None:
    chunks = [
        TextChunk(1, "first", (1,)),
        TextChunk(2, "shared", (1, 2)),
        TextChunk(3, "second", (2,)),
    ]

    assert [row[0] for row in preview_rows_for_page(chunks, 2)] == [2, 3]


def test_preview_rows_includes_heading_path() -> None:
    chunk = TextChunk(1, "# Guide\n\n## Install\n\nRun", (2,), ("Guide", "Install"))

    assert preview_rows([chunk]) == [
        [1, "Guide > Install", "2", len(chunk.text), chunk.text]
    ]
