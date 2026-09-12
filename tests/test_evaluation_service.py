import pytest

from manual_graphrag.chunking import TextChunk
from manual_graphrag import evaluation_service


def test_generate_evaluation_questions_validates_and_numbers(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "問題一？", "expected_answer": "答案一", "source_pages": [2],
                 "source_chunk_numbers": [1]},
                {"question": "問題二？", "expected_answer": "答案二", "source_pages": [3],
                 "source_chunk_numbers": [1]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    questions = evaluation_service.generate_evaluation_questions(
        "http://models", "key", "model", [TextChunk(1, "文件", (2, 3), "manual.pdf")], 2
    )

    assert [item["number"] for item in questions] == [1, 2]
    assert questions[0]["source_pages"] == [2]
    assert questions[0]["document"] == "manual.pdf"


def test_generate_evaluation_questions_rejects_missing_chunk_numbers(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "問題一？", "expected_answer": "答案一", "source_pages": [2]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    with pytest.raises(ValueError, match="source_chunk_numbers"):
        evaluation_service.generate_evaluation_questions(
            "http://models", "key", "model", [TextChunk(1, "文件", (2,), "manual.pdf")], 1
        )


def test_generate_evaluation_questions_joins_multiple_source_documents(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "問題一？", "expected_answer": "答案一", "source_pages": [1],
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
