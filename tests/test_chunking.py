import pytest

from manual_graphrag.chunking import PageText, chunk_pages


def test_chunk_pages_preserves_page_provenance() -> None:
    chunks = chunk_pages([PageText(1, "abcd"), PageText(2, "efgh")], 6, 2)
    assert chunks[0].pages == (1, 2)
    assert chunks[1].pages == (2,)


def test_chunk_pages_returns_empty_for_blank_pages() -> None:
    assert chunk_pages([PageText(1, "  ")], 100, 10) == []


def test_chunk_pages_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "text")], 10, 10)
