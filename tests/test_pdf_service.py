from __future__ import annotations

from pathlib import Path

import pytest

from manual_graphrag import pdf_service


class FakeDocument:
    def __init__(self, *, needs_pass: bool = False, page_count: int = 2) -> None:
        self.needs_pass = needs_pass
        self.page_count = page_count

    def __enter__(self) -> FakeDocument:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_extract_pdf_returns_page_markdown_and_empty_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    document = FakeDocument()
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)

    def fake_to_markdown(received_document: object, **options: object) -> list[dict[str, object]]:
        assert received_document is document
        assert list(options.pop("pages")) == [0, 1]
        assert options == {"page_chunks": True, "use_ocr": False}
        return [
            {"metadata": {"page_number": 1}, "text": "# 標題\n\n第一頁"},
            {"metadata": {"page_number": 2}, "text": "   "},
        ]

    monkeypatch.setattr(pdf_service.pymupdf4llm, "to_markdown", fake_to_markdown)

    pages, empty_pages = pdf_service.extract_pdf("manual.pdf")

    assert [(page.page, page.text) for page in pages] == [
        (1, "# 標題\n\n第一頁"),
        (2, ""),
    ]
    assert empty_pages == [2]


def test_extract_pdf_falls_back_to_sequence_for_invalid_page_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: FakeDocument())
    monkeypatch.setattr(
        pdf_service.pymupdf4llm,
        "to_markdown",
        lambda document, **options: [{"metadata": {"page_number": 0}, "text": "content"}],
    )

    pages, _ = pdf_service.extract_pdf(Path("manual.PDF"))

    assert pages[0].page == 1


def test_extract_pdf_rejects_non_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_open(path: object) -> None:
        raise AssertionError("non-PDF input must not be opened")

    monkeypatch.setattr(pdf_service.pymupdf, "open", unexpected_open)

    with pytest.raises(ValueError, match="僅支援 PDF 檔案"):
        pdf_service.extract_pdf("manual.txt")


def test_extract_pdf_rejects_encrypted_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pdf_service.pymupdf,
        "open",
        lambda path: FakeDocument(needs_pass=True),
    )

    with pytest.raises(ValueError, match="目前不支援加密 PDF"):
        pdf_service.extract_pdf("encrypted.pdf")


def test_extract_pdf_wraps_parser_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_open(path: object) -> None:
        raise RuntimeError("broken document")

    monkeypatch.setattr(pdf_service.pymupdf, "open", broken_open)

    with pytest.raises(ValueError, match="PDF 無法開啟，可能已損毀或加密"):
        pdf_service.extract_pdf("broken.pdf")


def test_extract_pdf_wraps_markdown_conversion_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: FakeDocument())

    def broken_parser(document: object, **options: object) -> None:
        raise ValueError("layout failed")

    monkeypatch.setattr(pdf_service.pymupdf4llm, "to_markdown", broken_parser)

    with pytest.raises(ValueError, match="PDF 無法解析，請確認檔案內容有效"):
        pdf_service.extract_pdf("broken.pdf")


def test_extract_pdf_rejects_unexpected_parser_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: FakeDocument())
    monkeypatch.setattr(pdf_service.pymupdf4llm, "to_markdown", lambda document, **options: "text")

    with pytest.raises(ValueError, match="PDF 解析結果格式不正確"):
        pdf_service.extract_pdf("manual.pdf")


def test_extract_pdf_with_real_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    document = pdf_service.pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Hello PDF")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "Second page")
    document.new_page()
    document.save(pdf_path)
    document.close()

    assert pdf_service.get_pdf_page_count(pdf_path) == 3
    pages, empty_pages = pdf_service.extract_pdf(pdf_path, 2, 3)

    assert [(page.page, page.text) for page in pages] == [(2, "Second page"), (3, "")]
    assert empty_pages == [3]


def test_extract_pdf_only_converts_requested_page_range(monkeypatch: pytest.MonkeyPatch) -> None:
    document = FakeDocument(page_count=4)
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    captured: dict[str, list[int]] = {}

    def fake_to_markdown(received: object, **options: object) -> list[dict[str, object]]:
        captured["pages"] = list(options["pages"])
        return [
            {"metadata": {"page_number": 2}, "text": "second"},
            {"metadata": {"page_number": 3}, "text": "   "},
        ]

    monkeypatch.setattr(pdf_service.pymupdf4llm, "to_markdown", fake_to_markdown)

    pages, empty_pages = pdf_service.extract_pdf("manual.pdf", 2, 3)

    assert [(page.page, page.text) for page in pages] == [(2, "second"), (3, "")]
    assert empty_pages == [3]
    assert captured["pages"] == [1, 2]


def test_extract_pdf_rejects_invalid_page_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: FakeDocument(page_count=3))

    cases = [
        (0, None, "解析起始頁必須大於等於 1"),
        (3, 2, "解析結束頁不得小於起始頁"),
        (1, 4, "解析頁碼超出範圍；PDF 共 3 頁"),
    ]

    for start_page, end_page, message in cases:
        with pytest.raises(ValueError, match=message):
            pdf_service.extract_pdf("manual.pdf", start_page, end_page)
