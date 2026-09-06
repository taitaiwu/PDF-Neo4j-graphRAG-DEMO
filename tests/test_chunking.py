import pytest

from manual_graphrag.chunking import (
    PageText,
    TextChunk,
    chunk_pages,
    preview_rows_for_page,
)


def test_chunk_pages_preserves_page_provenance() -> None:
    chunks = chunk_pages([PageText(1, "abcd"), PageText(2, "efgh")], 6, 2)
    assert chunks[0].pages == (1, 2)
    assert chunks[1].pages == (2,)


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
