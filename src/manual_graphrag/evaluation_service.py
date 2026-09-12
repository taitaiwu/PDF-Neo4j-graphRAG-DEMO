from __future__ import annotations

import json
from typing import Any

from .chunking import TextChunk
from .graph_service import _chat_json


EVALUATION_CONTEXT_LIMIT = 30_000


def _evaluation_context(chunks: list[TextChunk]) -> tuple[str, set[int]]:
    if not chunks:
        return "", set()
    max_samples = max(1, EVALUATION_CONTEXT_LIMIT // 500)
    if len(chunks) <= max_samples:
        sampled = chunks
    else:
        sampled = [
            chunks[round(index * (len(chunks) - 1) / (max_samples - 1))]
            for index in range(max_samples)
        ]
    per_chunk = max(200, EVALUATION_CONTEXT_LIMIT // len(sampled) - 80)
    context = "\n\n".join(
        f"[CHUNK {chunk.number}; PAGES {','.join(map(str, chunk.pages))}]\n{chunk.text[:per_chunk]}"
        for chunk in sampled
    )
    return context, {chunk.number for chunk in sampled}


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

    context, available_chunk_numbers = _evaluation_context(chunks)
    chunk_lookup = {chunk.number: chunk for chunk in chunks}

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
            raw_numbers = item.get("source_chunk_numbers", [])
            if not isinstance(raw_numbers, list):
                raise ValueError("source_chunk_numbers 必須是陣列")
            chunk_numbers: list[int] = []
            for value in raw_numbers:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number in available_chunk_numbers and number not in chunk_numbers:
                    chunk_numbers.append(number)
            if not chunk_numbers:
                raise ValueError("每一題都必須包含至少一個有效的 source_chunk_numbers")
            document = "、".join(
                dict.fromkeys(
                    chunk_lookup[number].document
                    for number in chunk_numbers
                    if chunk_lookup[number].document
                )
            )
            normalized.append({
                "number": index,
                "question": question,
                "expected_answer": answer,
                "source_pages": [int(page) for page in pages],
                "source_chunk_numbers": chunk_numbers,
                "document": document,
            })
        return {"questions": normalized}

    result = _chat_json(
        base_url,
        api_key,
        model,
        "你是文件問答評測資料設計師。只能根據提供的文件內容出題，並只輸出 JSON。",
        f"請建立剛好 {count} 道可由文件明確回答、彼此不重複且涵蓋不同內容的繁體中文問題。"
        "每題提供精確標準答案、來源頁碼，以及該題所依據的 CHUNK 編號"
        "（source_chunk_numbers，必須引用下方文件中標示的 CHUNK 編號）。輸出格式："
        '{"questions":[{"question":"...","expected_answer":"...","source_pages":[1],'
        '"source_chunk_numbers":[1]}]}。\n\n'
        f"文件：\n{context}",
        temperature=0.2,
        max_output_tokens=max(2048, count * 300),
        validator=validate,
    )
    return result["questions"]


_ABSTENTION_PHRASES = (
    "無法回答", "無法判斷", "沒有任何資訊", "找不到相關資訊",
    "文件未提及", "證據未提及", "未提供相關", "證據不足",
    "cannot answer", "not enough information", "insufficient evidence",
)


def _is_abstention(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    return any(phrase in normalized for phrase in _ABSTENTION_PHRASES)


def judge_evaluation_answer(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> dict[str, Any]:
    if not actual_answer.strip():
        return {"passed": False, "reason": "實際答案為空，未回答標準答案中的關鍵事實。"}
    if _is_abstention(actual_answer) and not _is_abstention(expected_answer):
        return {
            "passed": False,
            "reason": "實際答案表示無法回答或資料不足，但標準答案包含明確事實。",
        }

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
        "你是嚴謹的問答評測員。只有實際答案包含標準答案的核心事實才能通過。"
        "誠實表示不知道、文件未提及、找不到資訊或證據不足，不等於回答正確；"
        "當標準答案有明確事實而實際答案拒答時，passed 必須為 false。"
        "不要求逐字相同，只輸出 JSON。",
        f"問題：{question}\n標準答案：{expected_answer}\n實際答案：{actual_answer}\n"
        "請逐項檢查標準答案中的數值、單位、名稱、條件與結論是否出現在實際答案。"
        '輸出格式：{"passed":true,"reason":"簡短理由"}。',
        validator=validate,
    )
