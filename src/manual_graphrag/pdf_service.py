from __future__ import annotations

from pathlib import Path

import pymupdf
import pymupdf4llm

from .chunking import PageText


def extract_pdf(path: str | Path) -> tuple[list[PageText], list[int]]:
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("僅支援 PDF 檔案")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError("PDF 無法開啟，可能已損毀或加密") from exc

    with document:
        if document.needs_pass:
            raise ValueError("目前不支援加密 PDF")
        try:
            page_chunks = pymupdf4llm.to_markdown(
                document,
                page_chunks=True,
                use_ocr=False,
            )
        except Exception as exc:
            raise ValueError("PDF 無法解析，請確認檔案內容有效") from exc

    if not isinstance(page_chunks, list):
        raise ValueError("PDF 解析結果格式不正確")

    pages: list[PageText] = []
    empty_pages: list[int] = []
    for index, chunk in enumerate(page_chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError("PDF 解析結果格式不正確")
        metadata = chunk.get("metadata", {})
        page_number = (
            metadata.get("page_number", index)
            if isinstance(metadata, dict)
            else index
        )
        if not isinstance(page_number, int) or page_number < 1:
            page_number = index
        text = str(chunk.get("text") or "").strip()
        pages.append(PageText(page_number, text))
        if not text:
            empty_pages.append(page_number)
    return pages, empty_pages
