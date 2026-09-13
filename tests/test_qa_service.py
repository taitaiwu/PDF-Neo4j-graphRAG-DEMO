from manual_graphrag import qa_service


def test_answer_graph_question_omits_max_tokens(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, api_key, **kwargs):
        captured.update(url=url, payload=payload, api_key=api_key)
        return {
            "choices": [{
                "message": {"content": "答案"},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(qa_service, "_post_json", fake_post)

    result = qa_service.answer_graph_question(
        "http://models/v1",
        "secret",
        "model-a",
        "問題",
        "基本檢索",
        [{"text": "證據"}],
    )

    assert result["answer"] == "答案"
    assert captured["url"] == "http://models/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert "max_tokens" not in captured["payload"]
