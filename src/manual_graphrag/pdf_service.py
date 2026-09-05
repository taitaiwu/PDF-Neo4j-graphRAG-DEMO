from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .chunking import PageText


def extract_pdf(path: str | Path) -> tuple[list[PageText], list[int]]:
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("僅支援 PDF 檔案")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise ValueError("PDF 無法開啟，可能已損毀或加密") from exc

    if reader.is_encrypted:
        raise ValueError("目前不支援加密 PDF")

    pages: list[PageText] = []
    empty_pages: list[int] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(PageText(index, text))
        if not text:
            empty_pages.append(index)
    return pages, empty_pages
