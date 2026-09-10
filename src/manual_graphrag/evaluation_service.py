from __future__ import annotations

import json
from typing import Any

from .chunking import TextChunk
from .graph_service import _chat_json


EVALUATION_CONTEXT_LIMIT = 30_000


def _evaluation_context(chunks: list[TextChunk]) -> str:
    if not chunks:
        return ""
    max_samples = max(1, EVALUATION_CONTEXT_LIMIT // 500)
    if len(chunks) <= max_samples:
        sampled = chunks
    else:
        sampled = [
            chunks[round(index * (len(chunks) - 1) / (max_samples - 1))]
            for index in range(max_samples)
        ]
    per_chunk = max(200, EVALUATION_CONTEXT_LIMIT // len(sampled) - 80)
    return "\n\n".join(
        f"[CHUNK {chunk.number}; PAGES {','.join(map(str, chunk.pages))}]\n{chunk.text[:per_chunk]}"
        for chunk in sampled
    )


def generate_evaluation_questions(
    base_url: str,
    api_key: str,
    model: str,
    chunks: list[TextChunk],
    question_count: int,
) -> list[dict[str, Any]]:
    count = int(question_count)
    if not chunks:
        raise ValueError("請先解析 PDF 並產生 chunks")
    if not 1 <= count <= 100:
        raise ValueError("題目數量必須介於 1 到 100")

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        questions = payload.get("questions")
        if not isinstance(questions, list) or len(questions) != count:
            raise ValueError(f"questions 必須剛好包含 {count} 題")
        normalized = []
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                raise ValueError("每一題必須是 JSON 物件")
            question = str(item.get("question", "")).strip()
            answer = str(item.get("expected_answer", "")).strip()
            if not question or not answer:
                raise ValueError("每一題都必須包含 question 與 expected_answer")
            pages = item.get("source_pages", [])
            if not isinstance(pages, list):
                raise ValueError("source_pages 必須是陣列")
            normalized.append({
                "number": index,
                "question": question,
                "expected_answer": answer,
                "source_pages": [int(page) for page in pages],
            })
        return {"questions": normalized}

    result = _chat_json(
        base_url,
        api_key,
        model,
        "你是文件問答評測資料設計師。只能根據提供的文件內容出題，並只輸出 JSON。",
        f"請建立剛好 {count} 道可由文件明確回答、彼此不重複且涵蓋不同內容的繁體中文問題。"
        "每題提供精確標準答案與來源頁碼。輸出格式："
        '{"questions":[{"question":"...","expected_answer":"...","source_pages":[1]}]}。\n\n'
        f"文件：\n{_evaluation_context(chunks)}",
        temperature=0.2,
        max_output_tokens=max(2048, count * 300),
        validator=validate,
    )
    return result["questions"]


def judge_evaluation_answer(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> dict[str, Any]:
    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        passed = payload.get("passed")
        reason = str(payload.get("reason", "")).strip()
        if not isinstance(passed, bool) or not reason:
            raise ValueError("評判結果必須包含 passed boolean 與 reason")
        return {"passed": passed, "reason": reason}

    return _chat_json(
        base_url,
        api_key,
        model,
        "你是嚴謹的問答評測員。比較語意與關鍵事實，不要求逐字相同，只輸出 JSON。",
        f"問題：{question}\n標準答案：{expected_answer}\n實際答案：{actual_answer}\n"
        '輸出格式：{"passed":true,"reason":"簡短理由"}。',
        validator=validate,
    )
