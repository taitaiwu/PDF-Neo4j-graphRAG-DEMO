from __future__ import annotations

import json
import math
from typing import Any

from .graph_service import _api_url, _chat_response_content, _post_json


def _embedding_vectors(base_url: str, api_key: str, model: str, texts: list[str]) -> list[list[float]]:
    if not model.strip():
        raise ValueError("此建圖結果沒有 Embedding 模型")
    response = _post_json(_api_url(base_url, "embeddings"), {"model": model.strip(), "input": texts}, api_key)
    data = response.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise ValueError("Embedding API 回傳格式或數量不正確")
    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    try:
        return [[float(value) for value in item["embedding"]] for item in ordered]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Embedding API 回傳的向量格式不正確") from exc


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    if not denominator:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def answer_graph_question(base_url: str, api_key: str, answer_model: str, question: str, retrieval_mode: str, top_k: int, graph_state: dict[str, Any]) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("請輸入問題")
    if not answer_model.strip():
        raise ValueError("請選擇問答 LLM")
    if not graph_state or not graph_state.get("run_id"):
        raise ValueError("請先完成知識圖譜抽取與匯入")
    if not graph_state.get("neo4j_imported"):
        raise ValueError("請先將目前知識圖譜匯入 Neo4j")
    if retrieval_mode not in {"GraphRAG", "向量 RAG"}:
        raise ValueError("不支援的檢索模式")
    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("Top K 必須大於 0")

    evidence: list[dict[str, Any]] = []
    for item in graph_state.get("entities", []):
        evidence.append({"kind": "實體", "name": str(item.get("name", "")), "text": "實體：{}；類型：{}；說明：{}".format(item.get("name", ""), item.get("type", ""), item.get("description", "")), "source_pages": item.get("source_pages", []), "source_chunk_numbers": item.get("source_chunk_numbers", [])})
    for item in graph_state.get("relationships", []):
        evidence.append({"kind": "關係", "source": str(item.get("source", "")), "target": str(item.get("target", "")), "text": "關係：{} -[{}]-> {}；說明：{}".format(item.get("source", ""), item.get("type", ""), item.get("target", ""), item.get("description", "")), "source_pages": item.get("source_pages", []), "source_chunk_numbers": item.get("source_chunk_numbers", [])})
    if not evidence:
        raise ValueError("目前建圖結果沒有可供檢索的實體或關係")

    vectors = _embedding_vectors(base_url, api_key, graph_state.get("embedding_model", ""), [question.strip(), *[item["text"] for item in evidence]])
    for item, vector in zip(evidence, vectors[1:]):
        item["score"] = _cosine_similarity(vectors[0], vector)

    if retrieval_mode == "向量 RAG":
        selected = sorted(evidence, key=lambda item: item["score"], reverse=True)[:top_k]
    else:
        entities = [item for item in evidence if item["kind"] == "實體"]
        seeds = sorted(entities, key=lambda item: item["score"], reverse=True)[:top_k]
        names = {item["name"].casefold() for item in seeds}
        related = [item for item in evidence if item["kind"] == "關係" and (item["source"].casefold() in names or item["target"].casefold() in names)]
        related.sort(key=lambda item: item["score"], reverse=True)
        selected = [*seeds, *related[:top_k]]
        if not selected:
            selected = sorted(evidence, key=lambda item: item["score"], reverse=True)[:top_k]

    response = _post_json(_api_url(base_url, "chat/completions"), {"model": answer_model.strip(), "temperature": 0, "max_tokens": 2048, "messages": [{"role": "system", "content": "你是文件知識圖譜問答助手。只能根據提供的證據回答；證據不足時必須明確說明。使用繁體中文，並在相關敘述後標示來源頁碼。"}, {"role": "user", "content": "問題：{}\n\n證據：\n{}".format(question.strip(), json.dumps(selected, ensure_ascii=False))}]}, api_key)
    answer, _ = _chat_response_content(response)
    return {"answer": answer.strip(), "evidence": selected}
