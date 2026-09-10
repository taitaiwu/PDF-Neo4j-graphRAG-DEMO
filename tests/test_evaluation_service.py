import pytest

from manual_graphrag.chunking import TextChunk
from manual_graphrag import evaluation_service


def test_generate_evaluation_questions_validates_and_numbers(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "問題一？", "expected_answer": "答案一", "source_pages": [2]},
                {"question": "問題二？", "expected_answer": "答案二", "source_pages": [3]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    questions = evaluation_service.generate_evaluation_questions(
        "http://models", "key", "model", [TextChunk(1, "文件", (2, 3))], 2
    )

    assert [item["number"] for item in questions] == [1, 2]
    assert questions[0]["source_pages"] == [2]


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
