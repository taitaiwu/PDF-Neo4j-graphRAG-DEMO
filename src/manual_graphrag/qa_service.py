from __future__ import annotations

import json
from typing import Any

from .graph_service import _api_url, _chat_response_content, _post_json


def embedding_vectors(base_url: str, api_key: str, model: str, texts: list[str]) -> list[list[float]]:
    if not model.strip():
        raise ValueError("此建圖結果沒有 Embedding 模型")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        response = _post_json(
            _api_url(base_url, "embeddings"),
            {"model": model.strip(), "input": batch},
            api_key,
        )
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise ValueError("Embedding API 回傳格式或數量不正確")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        try:
            for item in ordered:
                vectors.append([float(value) for value in item["embedding"]])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Embedding API 回傳的向量格式不正確") from exc
    return vectors


def answer_graph_question(
    base_url: str,
    api_key: str,
    answer_model: str,
    question: str,
    retrieval_mode: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("請輸入問題")
    if not answer_model.strip():
        raise ValueError("請選擇問答 LLM")
    if retrieval_mode not in {"GraphRAG", "向量 RAG"}:
        raise ValueError("不支援的檢索模式")
    if not evidence:
        raise ValueError("Neo4j Vector Search 找不到相關證據")
    response = _post_json(
        _api_url(base_url, "chat/completions"),
        {
            "model": answer_model.strip(),
            "temperature": 0,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "system",
                    "content": "你是文件知識圖譜問答助手。只能根據提供的證據回答；證據不足時必須明確說明。使用繁體中文，並在相關敘述後標示來源頁碼。",
                },
                {
                    "role": "user",
                    "content": "問題：{}\n\n證據：\n{}".format(
                        question.strip(), json.dumps(evidence, ensure_ascii=False)
                    ),
                },
            ],
        },
        api_key,
    )
    answer, _ = _chat_response_content(response)
    return {"answer": answer.strip(), "evidence": evidence}
