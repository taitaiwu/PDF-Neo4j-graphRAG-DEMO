from __future__ import annotations

from pathlib import Path

import pymupdf
import pymupdf4llm

from .chunking import PageText


def get_pdf_page_count(path: str | Path) -> int:
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("僅支援 PDF 檔案")

    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                raise ValueError("目前不支援加密 PDF")
            page_count = document.page_count
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 無法開啟，可能已損毀或加密") from exc

    if page_count < 1:
        raise ValueError("PDF 沒有任何頁面")
    return page_count


def _validate_page_range(
    page_count: int, start_page: int, end_page: int | None
) -> tuple[int, int]:
    if page_count < 1:
        raise ValueError("PDF 沒有任何頁面")
    if start_page < 1:
        raise ValueError("解析起始頁必須大於等於 1")
    last_page = page_count if end_page is None else end_page
    if last_page < start_page:
        raise ValueError("解析結束頁不得小於起始頁")
    if start_page > page_count or last_page > page_count:
        raise ValueError(f"解析頁碼超出範圍；PDF 共 {page_count} 頁")
    return start_page, last_page


def extract_pdf(
    path: str | Path, start_page: int = 1, end_page: int | None = None
) -> tuple[list[PageText], list[int]]:
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
        first_page, last_page = _validate_page_range(
            document.page_count, start_page, end_page
        )
        try:
            page_chunks = pymupdf4llm.to_markdown(
                document,
                page_chunks=True,
                header=False,
                footer=False,
                pages=range(first_page - 1, last_page),
                use_ocr=False,
            )
        except Exception as exc:
            raise ValueError("PDF 無法解析，請確認檔案內容有效") from exc

    if not isinstance(page_chunks, list):
        raise ValueError("PDF 解析結果格式不正確")

    pages: list[PageText] = []
    empty_pages: list[int] = []
    for index, chunk in enumerate(page_chunks):
        fallback_page = first_page + index
        if not isinstance(chunk, dict):
            raise ValueError("PDF 解析結果格式不正確")
        metadata = chunk.get("metadata", {})
        page_number = (
            metadata.get("page_number", fallback_page)
            if isinstance(metadata, dict)
            else fallback_page
        )
        if not isinstance(page_number, int) or page_number < 1:
            page_number = fallback_page
        text = str(chunk.get("text") or "").strip()
        pages.append(PageText(page_number, text))
        if not text:
            empty_pages.append(page_number)
    return pages, empty_pages
