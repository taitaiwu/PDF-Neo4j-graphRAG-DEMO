import pytest

from manual_graphrag.chunking import TextChunk
from manual_graphrag import evaluation_service


def test_generate_evaluation_questions_validates_and_numbers(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "CX17 系列的問題一？", "expected_answer": "答案一", "source_pages": [2],
                 "source_chunk_numbers": [1]},
                {"question": "CX17 系列的問題二？", "expected_answer": "答案二", "source_pages": [3],
                 "source_chunk_numbers": [1]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    questions = evaluation_service.generate_evaluation_questions(
        "http://models", "key", "model", [TextChunk(1, "文件", (2, 3), "manual.pdf")], 2, [],
        {"primary_identifier": "CX17 系列", "identifiers": ["Windows 7"]},
    )

    assert [item["number"] for item in questions] == [1, 2]
    assert questions[0]["source_pages"] == [2]
    assert questions[0]["document"] == "manual.pdf"


def test_generate_evaluation_questions_rejects_similar_questions(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [{
                "question": "系統的最大容量是多少？",
                "expected_answer": "100",
                "source_pages": [1],
                "source_chunk_numbers": [1],
            }]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)

    with pytest.raises(ValueError, match="重複或過度相似"):
        evaluation_service.generate_evaluation_questions(
            "url",
            "",
            "model",
            [TextChunk(1, "容量為 100", (1,), "manual.pdf")],
            1,
            ["系統的最大容量是多少"],
            {"identifiers": ["系統"]},
        )


def test_generate_evaluation_questions_tells_model_which_questions_to_avoid(
    monkeypatch,
) -> None:
    captured = {}

    def fake_chat(*args, **kwargs):
        captured["prompt"] = args[4]
        return kwargs["validator"]({
            "questions": [{
                "question": "另一個問題？",
                "expected_answer": "答案",
                "source_pages": [1],
                "source_chunk_numbers": [1],
            }]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    evaluation_service.generate_evaluation_questions(
        "url", "", "model", [TextChunk(1, "文件", (1,))], 1, ["既有問題？"]
    )

    assert "既有問題？" in captured["prompt"]

    assert "不得重複或改寫以下已建立題目" in captured["prompt"]
def test_select_relevant_documents_uses_only_summaries(monkeypatch) -> None:
    captured = {}

    def fake_chat(*args, **kwargs):
        captured["prompt"] = args[4]
        return kwargs["validator"]({
            "documents": ["printer-a.pdf"],
            "reason": "問題指定印表機 A",
            "confidence": 0.9,
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    result = evaluation_service.select_relevant_documents(
        "url",
        "",
        "model",
        "印表機 A 如何查詢 IP？",
        [
            {"document": "printer-a.pdf", "summary": "印表機 A 使用手冊"},
            {"document": "printer-b.pdf", "summary": "印表機 B 使用手冊"},
        ],
    )

    assert result["documents"] == ["printer-a.pdf"]
    assert "printer-a.pdf" in captured["prompt"]
    assert "printer-b.pdf" in captured["prompt"]


def test_select_relevant_documents_rejects_unknown_document(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "_chat_json",
        lambda *args, **kwargs: kwargs["validator"]({
            "documents": ["unknown.pdf"],
            "reason": "錯誤選擇",
            "confidence": 1,
        }),
    )

    with pytest.raises(ValueError, match="不存在"):
        evaluation_service.select_relevant_documents(
            "url",
            "",
            "model",
            "問題",
            [{"document": "manual.pdf", "summary": "摘要"}],
        )



def test_generate_evaluation_questions_rejects_missing_chunk_numbers(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "manual.pdf 的問題一？", "expected_answer": "答案一", "source_pages": [2]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    with pytest.raises(ValueError, match="source_chunk_numbers"):
        evaluation_service.generate_evaluation_questions(
            "http://models", "key", "model", [TextChunk(1, "文件", (2,), "manual.pdf")], 1
        )


def test_generate_document_summary_returns_routing_metadata(monkeypatch) -> None:
    captured = {}

    def fake_chat(*args, **kwargs):
        captured["prompt"] = args[4]
        return kwargs["validator"]({
            "summary": "印表機 A 的網路設定與列印手冊",
            "primary_identifier": "Epson AcuLaser CX17 系列",
            "identifiers": ["Epson AcuLaser CX17", "CX17NF", "CX17WF", "額外一", "額外二"],
            "topics": ["網路設定"],
            "keywords": ["IP 位址"],
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)

    result = evaluation_service.generate_document_summary(
        "url",
        "",
        "model",
        [TextChunk(1, "如何查詢 IP 位址", (1,), "printer-a.pdf")],
    )

    assert result == {
        "document": "printer-a.pdf",
        "summary": "印表機 A 的網路設定與列印手冊",
        "primary_identifier": "Epson AcuLaser CX17 系列",
        "identifiers": ["Epson AcuLaser CX17 系列", "Epson AcuLaser CX17", "CX17NF", "CX17WF", "額外一"],
        "topics": ["網路設定"],
        "keywords": ["IP 位址"],
    }
    assert "printer-a.pdf" in captured["prompt"]
    assert "排除出版商" in captured["prompt"]


def test_generate_evaluation_questions_joins_multiple_source_documents(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "a.pdf 的問題一？", "expected_answer": "答案一", "source_pages": [1],
                 "source_chunk_numbers": [1, 2]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    questions = evaluation_service.generate_evaluation_questions(
        "http://models", "key", "model",
        [TextChunk(1, "文件一", (1,), "a.pdf"), TextChunk(2, "文件二", (1,), "b.pdf")], 1
    )

    assert questions[0]["document"] == "a.pdf、b.pdf"


def test_generate_evaluation_questions_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="先解析"):
        evaluation_service.generate_evaluation_questions("url", "", "model", [], 2)
    with pytest.raises(ValueError, match="1 到 100"):
        evaluation_service.generate_evaluation_questions(
            "url", "", "model", [TextChunk(1, "text", (1,))], 0
        )


def test_judge_evaluation_answer_returns_boolean(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "_chat_json",
        lambda *args, **kwargs: kwargs["validator"]({"passed": True, "reason": "語意相符"}),
    )

    result = evaluation_service.judge_evaluation_answer(
        "url", "key", "model", "問題", "標準", "實際"
    )

    assert result == {"passed": True, "reason": "語意相符"}


def test_judge_rejects_abstention_when_expected_answer_has_fact(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "_chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("明確拒答應由程式直接判錯，不應交給模型")
        ),
    )

    result = evaluation_service.judge_evaluation_answer(
        "url",
        "key",
        "model",
        "Epson AcuLaser CX17 的中央輸出承接盤可容納多少張 A4 紙？",
        "約 100 張 (A4)",
        "根據提供的證據內容，沒有任何資訊提及容量。因此，無法回答此問題。",
    )

    assert result["passed"] is False
    assert "標準答案包含明確事實" in result["reason"]


def test_judge_rejects_empty_answer_without_model_call(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "_chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不應呼叫模型")),
    )

    result = evaluation_service.judge_evaluation_answer(
        "url", "key", "model", "問題", "明確答案", "  "
    )

    assert result["passed"] is False
    assert "實際答案為空" in result["reason"]
