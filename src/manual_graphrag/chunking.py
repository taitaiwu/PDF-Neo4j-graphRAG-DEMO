from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    number: int
    text: str
    pages: tuple[int, ...]


def chunk_pages(
    pages: list[PageText], chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必須大於 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必須大於等於 0 且小於 chunk_size")

    joined: list[tuple[str, int]] = []
    for page in pages:
        text = page.text.strip()
        if text:
            if joined:
                joined.append(("\n\n", page.page))
            joined.extend((char, page.page) for char in text)

    if not joined:
        return []

    chunks: list[TextChunk] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(joined):
        section = joined[start : start + chunk_size]
        text = "".join(char for char, _ in section).strip()
        pages_in_chunk = tuple(dict.fromkeys(page for _, page in section))
        if text:
            chunks.append(TextChunk(len(chunks) + 1, text, pages_in_chunk))
        start += step
    return chunks


def preview_rows(chunks: list[TextChunk], limit: int = 10) -> list[list[object]]:
    return [
        [chunk.number, ", ".join(map(str, chunk.pages)), len(chunk.text), chunk.text]
        for chunk in chunks[:limit]
    ]
