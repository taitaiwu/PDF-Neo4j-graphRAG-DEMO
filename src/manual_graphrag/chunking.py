from __future__ import annotations

import re
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
    headings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SourceLine:
    text: str
    page: int


@dataclass(frozen=True)
class _MarkdownBlock:
    source: tuple[tuple[str, int], ...]

    @property
    def text(self) -> str:
        return "".join(character for character, _ in self.source)

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(page for _, page in self.source))


@dataclass(frozen=True)
class _MarkdownSection:
    headings: tuple[_MarkdownBlock, ...]
    body: tuple[_SourceLine, ...]


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")


def _source_lines(pages: list[PageText]) -> list[_SourceLine]:
    lines: list[_SourceLine] = []
    for page in pages:
        page_lines = page.text.strip().splitlines()
        lines.extend(_SourceLine(line.rstrip(), page.page) for line in page_lines)
        if page_lines:
            lines.append(_SourceLine("", page.page))
    return lines


def _block_from_lines(lines: list[_SourceLine]) -> _MarkdownBlock:
    source: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        if index:
            source.append(("\n", line.page))
        source.extend((character, line.page) for character in line.text)
    return _MarkdownBlock(tuple(source))


def _markdown_sections(lines: list[_SourceLine]) -> list[_MarkdownSection]:
    sections: list[_MarkdownSection] = []
    headings: list[_MarkdownBlock] = []
    body: list[_SourceLine] = []
    pending_heading = False
    fence_marker: str | None = None

    for line in lines:
        fence = _FENCE_PATTERN.match(line.text)
        if fence:
            marker = fence.group(1)[0]
            if fence_marker == marker:
                fence_marker = None
            elif fence_marker is None:
                fence_marker = marker
            body.append(line)
            continue

        match = _HEADING_PATTERN.match(line.text) if fence_marker is None else None
        if not match:
            body.append(line)
            continue

        if any(item.text.strip() for item in body):
            sections.append(_MarkdownSection(tuple(headings), tuple(body)))
            body = []

        level = len(match.group(1))
        heading = _block_from_lines([line])
        headings = headings[: level - 1]
        headings.append(heading)
        pending_heading = True

    if any(item.text.strip() for item in body) or pending_heading:
        sections.append(_MarkdownSection(tuple(headings), tuple(body)))
    return sections


def _markdown_blocks(lines: tuple[_SourceLine, ...]) -> list[_MarkdownBlock]:
    blocks: list[_MarkdownBlock] = []
    current: list[_SourceLine] = []
    fence_marker: str | None = None

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(_block_from_lines(current))
            current = []

    for line in lines:
        fence = _FENCE_PATTERN.match(line.text)
        if fence:
            marker = fence.group(1)[0]
            current.append(line)
            if fence_marker == marker:
                fence_marker = None
                flush()
            elif fence_marker is None:
                fence_marker = marker
            continue

        if not line.text.strip() and fence_marker is None:
            flush()
        else:
            current.append(line)
    flush()
    return blocks


def _join_sources(blocks: list[_MarkdownBlock]) -> tuple[tuple[str, int], ...]:
    source: list[tuple[str, int]] = []
    for block in blocks:
        if not block.source:
            continue
        if source:
            separator_page = block.pages[0] if block.pages else source[-1][1]
            source.extend((("\n", separator_page), ("\n", separator_page)))
        source.extend(block.source)
    return tuple(source)


def _heading_titles(headings: tuple[_MarkdownBlock, ...]) -> tuple[str, ...]:
    return tuple(_HEADING_PATTERN.sub(r"\2", heading.text).strip() for heading in headings)


def _make_chunk(
    number: int,
    headings: tuple[_MarkdownBlock, ...],
    body: list[_MarkdownBlock],
) -> TextChunk | None:
    source = _join_sources([*headings, *body])
    text = "".join(character for character, _ in source).strip()
    if not text:
        return None
    pages = tuple(dict.fromkeys(page for _, page in source))
    return TextChunk(number, text, pages, _heading_titles(headings))


def _overlap_blocks(blocks: list[_MarkdownBlock], overlap: int) -> list[_MarkdownBlock]:
    selected: list[_MarkdownBlock] = []
    length = 0
    for block in reversed(blocks):
        added = len(block.text) + (2 if selected else 0)
        if length + added > overlap:
            break
        selected.insert(0, block)
        length += added
    return selected


def _split_oversized_block(
    headings: tuple[_MarkdownBlock, ...],
    block: _MarkdownBlock,
    chunk_size: int,
    chunk_overlap: int,
    first_number: int,
) -> list[TextChunk]:
    prefix = _join_sources(list(headings))
    separator_page = block.pages[0] if block.pages else 1
    separator = (("\n", separator_page), ("\n", separator_page)) if prefix else ()
    available = chunk_size - len(prefix) - len(separator)
    if available < 1:
        combined = _join_sources([*headings, block])
        prefix = ()
        separator = ()
        block_source = combined
        available = chunk_size
    else:
        block_source = block.source

    overlap = min(chunk_overlap, max(0, available - 1))
    step = available - overlap
    chunks: list[TextChunk] = []
    for start in range(0, len(block_source), step):
        section = block_source[start : start + available]
        source = (*prefix, *separator, *section)
        text = "".join(character for character, _ in source).strip()
        if text:
            pages = tuple(dict.fromkeys(page for _, page in source))
            chunks.append(
                TextChunk(
                    first_number + len(chunks),
                    text,
                    pages,
                    _heading_titles(headings),
                )
            )
        if start + available >= len(block_source):
            break
    return chunks


def chunk_pages(
    pages: list[PageText], chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必須大於 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必須大於等於 0 且小於 chunk_size")

    sections = _markdown_sections(_source_lines(pages))
    chunks: list[TextChunk] = []
    for section in sections:
        blocks = _markdown_blocks(section.body)
        if not blocks:
            chunk = _make_chunk(len(chunks) + 1, section.headings, [])
            if chunk:
                chunks.append(chunk)
            continue

        current: list[_MarkdownBlock] = []
        for block in blocks:
            candidate = _make_chunk(len(chunks) + 1, section.headings, [*current, block])
            if candidate and len(candidate.text) <= chunk_size:
                current.append(block)
                continue

            if current:
                chunk = _make_chunk(len(chunks) + 1, section.headings, current)
                if chunk:
                    chunks.append(chunk)
                current = _overlap_blocks(current, chunk_overlap)
                candidate = _make_chunk(len(chunks) + 1, section.headings, [*current, block])
                if candidate and len(candidate.text) <= chunk_size:
                    current.append(block)
                    continue
                current = []

            chunks.extend(
                _split_oversized_block(
                    section.headings,
                    block,
                    chunk_size,
                    chunk_overlap,
                    len(chunks) + 1,
                )
            )

        if current:
            chunk = _make_chunk(len(chunks) + 1, section.headings, current)
            if chunk:
                chunks.append(chunk)
    return chunks


def _preview_row(chunk: TextChunk) -> list[object]:
    return [
        chunk.number,
        " > ".join(chunk.headings),
        ", ".join(map(str, chunk.pages)),
        len(chunk.text),
        chunk.text,
    ]


def preview_rows(chunks: list[TextChunk], limit: int = 10) -> list[list[object]]:
    return [_preview_row(chunk) for chunk in chunks[:limit]]


def preview_rows_for_page(
    chunks: list[TextChunk], page_number: int
) -> list[list[object]]:
    return [_preview_row(chunk) for chunk in chunks if page_number in chunk.pages]
