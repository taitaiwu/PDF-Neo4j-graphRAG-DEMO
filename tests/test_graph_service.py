import json

import pytest

from manual_graphrag import graph_service
from manual_graphrag.chunking import TextChunk


SCHEMA = {
    "entity_types": [{"name": "DEVICE", "description": "設備"}],
    "relationship_types": [
        {
            "name": "USES",
            "description": "使用",
            "source_types": ["DEVICE"],
            "target_types": ["DEVICE"],
        }
    ],
}


def chat_response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
                }
            }
        ]
    }


def test_plan_graph_schema_parses_fenced_json_and_uses_chunks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, payload, api_key, timeout=120):
        captured.update({"url": url, "payload": payload, "api_key": api_key})
        return chat_response(SCHEMA)

    monkeypatch.setattr(graph_service, "_post_json", fake_post)
    chunks = [TextChunk(1, "設備 A 使用設備 B", (2,))]

    plan = graph_service.plan_graph_schema(
        "http://localhost:11434/v1", "secret", "model-a", chunks, 0.4, 777
    )

    assert plan.schema == SCHEMA
    assert plan.analyzed_chunks == 1
    assert plan.batch_count == 1
    assert plan.merge_rounds == 0
    assert plan.total_chunks == 1
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert captured["payload"]["temperature"] == 0.4
    assert captured["payload"]["max_tokens"] == 777
    assert "[CHUNK 1; PAGES 2]" in captured["payload"]["messages"][1]["content"]


def test_schema_planning_analyzes_all_chunks_and_merges_hierarchically(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 11)]
    prompts: list[str] = []
    progress_updates: list[tuple[float, str]] = []

    def fake_chat(*args, **kwargs):
        prompts.append(args[4])
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    plan = graph_service.plan_graph_schema(
        "http://models/v1",
        "",
        "llm",
        chunks,
        progress_callback=lambda value, description: progress_updates.append(
            (value, description)
        ),
    )

    chunk_prompts = [prompt for prompt in prompts if "文件 chunks" in prompt]
    seen = {
        number
        for number in range(1, 11)
        if any(f"CHUNK {number};" in prompt for prompt in chunk_prompts)
    }
    assert seen == set(range(1, 11))
    assert plan.analyzed_chunks == plan.total_chunks == 10
    assert plan.batch_count == len(chunk_prompts)
    assert plan.batch_count > 1
    assert plan.merge_rounds > 0
    assert [value for value, _ in progress_updates] == sorted(
        value for value, _ in progress_updates
    )
    assert progress_updates[-1] == (1.0, "已分析全部 10 / 10 chunks")



def test_schema_groups_use_smaller_merge_limit(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    monkeypatch.setattr(graph_service, "SCHEMA_MERGE_LIMIT", 150)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 11)]
    merge_group_sizes = []

    def fake_chat(*args, **kwargs):
        prompt = args[4]
        if "候選 Schema" in prompt:
            merge_group_sizes.append(prompt.count('"entity_types"'))
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    plan = graph_service.plan_graph_schema("http://models/v1", "", "llm", chunks)

    assert plan.merge_rounds > 1
    assert merge_group_sizes
    assert all(size <= 2 for size in merge_group_sizes)


def test_compact_schema_keeps_only_required_merge_fields(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_DESCRIPTION_LIMIT", 8)
    schema = {
        "entity_types": [
            {
                "name": " DEVICE ",
                "description": "很長   的設備類型說明",
                "properties": ["serial"],
            }
        ],
        "relationship_types": [
            {
                "name": " USES ",
                "description": "使用關係說明文字",
                "source_types": ["DEVICE", "DEVICE", ""],
                "target_types": [" DEVICE "],
                "example": "A uses B",
            }
        ],
    }

    compact = graph_service._compact_schema(schema)

    assert compact == {
        "entity_types": [{"name": "DEVICE", "description": "很長 的設備類型"}],
        "relationship_types": [
            {
                "name": "USES",
                "description": "使用關係說明文字",
                "source_types": ["DEVICE"],
                "target_types": ["DEVICE"],
            }
        ],
    }



def test_extract_graph_batches_deduplicates_and_keeps_sources(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "EXTRACTION_BATCH_LIMIT", 45)
    chunks = [
        TextChunk(1, "設備 A", (1,)),
        TextChunk(2, "設備 A 使用設備 B", (2,)),
    ]
    responses = iter(
        [
            {
                "entities": [
                    {
                        "name": "設備 A",
                        "type": "DEVICE",
                        "description": "主要設備",
                        "source_chunk_numbers": [1],
                    }
                ],
                "relationships": [],
            },
            {
                "entities": [
                    {
                        "name": "設備 A",
                        "type": "DEVICE",
                        "description": "主要設備",
                        "source_chunk_numbers": [2],
                    },
                    {
                        "name": "設備 B",
                        "type": "DEVICE",
                        "description": "配件",
                        "source_chunk_numbers": [2],
                    },
                    {
                        "name": "忽略",
                        "type": "UNKNOWN",
                        "source_chunk_numbers": [2],
                    },
                ],
                "relationships": [
                    {
                        "source": "設備 A",
                        "target": "設備 B",
                        "type": "USES",
                        "description": "搭配使用",
                        "source_chunk_numbers": [2, 999],
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(graph_service, "_chat_json", lambda *args, **kwargs: next(responses))

    extraction = graph_service.extract_graph(
        "http://models/v1", "", "llm", chunks, SCHEMA
    )

    assert extraction.processed_chunks == 2
    assert len(extraction.entities) == 2
    assert extraction.entities[0]["source_chunk_numbers"] == [1, 2]
    assert extraction.entities[0]["source_pages"] == [1, 2]
    assert extraction.relationships == [
        {
            "source": "設備 A",
            "type": "USES",
            "target": "設備 B",
            "description": "搭配使用",
            "source_chunk_numbers": [2],
            "source_pages": [2],
        }
    ]


def test_validate_schema_rejects_missing_types() -> None:
    with pytest.raises(ValueError, match="entity_types"):
        graph_service.validate_schema({"entity_types": [], "relationship_types": []})


def test_plan_graph_schema_requires_chunks() -> None:
    with pytest.raises(ValueError, match="先在 PDF 頁面解析"):
        graph_service.plan_graph_schema("http://models/v1", "", "llm", [])


def test_chat_json_rejects_invalid_generation_options() -> None:
    with pytest.raises(ValueError, match="temperature"):
        graph_service._chat_json("http://models/v1", "", "llm", "system", "user", 2.1, 100)
    with pytest.raises(ValueError, match="最大輸出 tokens"):
        graph_service._chat_json("http://models/v1", "", "llm", "system", "user", 0, 0)


def test_schema_planning_stops_when_any_batch_fails(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 45)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 4)]
    calls = 0

    def fake_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second batch failed")
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    with pytest.raises(ValueError, match="second batch failed"):
        graph_service.plan_graph_schema("http://models/v1", "", "llm", chunks)
    assert calls == 2
    # Failure is contextualized with the exact batch and no partial plan is returned.


def test_extract_json_text_accepts_prose_around_json() -> None:
    result = graph_service._extract_json_text(
        '以下是結果：\n{"entity_types": [], "relationship_types": []}\n請確認。'
    )

    assert result == {"entity_types": [], "relationship_types": []}


def test_chat_json_retries_invalid_output_once(monkeypatch) -> None:
    responses = iter(
        [
            {
                "choices": [
                    {"message": {"content": "not json"}, "finish_reason": "stop"}
                ]
            },
            chat_response(SCHEMA),
        ]
    )
    payloads = []

    def fake_post(url, payload, api_key, timeout=120):
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr(graph_service, "_post_json", fake_post)

    result = graph_service._chat_json(
        "http://models/v1", "", "llm", "system", "user", 0.7, 2048
    )

    assert result == SCHEMA
    assert len(payloads) == 2
    assert payloads[0]["temperature"] == 0.7
    assert payloads[1]["temperature"] == 0
    assert len(payloads[1]["messages"]) == 4
    assert "請修正" in payloads[1]["messages"][-1]["content"]



def test_chat_json_retries_schema_that_fails_structure_validation(monkeypatch) -> None:
    invalid_schema = {"entity_types": [], "relationship_types": []}
    responses = iter([chat_response(invalid_schema), chat_response(SCHEMA)])
    payloads = []

    def fake_post(url, payload, api_key, timeout=120):
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr(graph_service, "_post_json", fake_post)

    result = graph_service._chat_json(
        "http://models/v1",
        "",
        "llm",
        "system",
        "user",
        0.7,
        2048,
        graph_service.validate_schema,
    )

    assert result == SCHEMA
    assert len(payloads) == 2
    assert payloads[1]["temperature"] == 0
    repair_prompt = payloads[1]["messages"][-1]["content"]
    assert "entity_types" in repair_prompt
    assert "不得省略必要欄位或回傳空陣列" in repair_prompt


def test_chat_json_reports_schema_validation_error_after_retry(monkeypatch) -> None:
    invalid_schema = {"entity_types": [], "relationship_types": []}
    monkeypatch.setattr(
        graph_service, "_post_json", lambda *args, **kwargs: chat_response(invalid_schema)
    )

    with pytest.raises(ValueError, match="entity_types"):
        graph_service._chat_json(
            "http://models/v1",
            "",
            "llm",
            "system",
            "user",
            0,
            2048,
            graph_service.validate_schema,
        )



def test_chat_json_reports_token_truncation_after_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_service,
        "_post_json",
        lambda *args, **kwargs: {
            "choices": [
                {"message": {"content": '{"entity_types": ['}, "finish_reason": "length"}
            ]
        },
    )

    with pytest.raises(ValueError, match="提高最大輸出 tokens"):
        graph_service._chat_json(
            "http://models/v1", "", "llm", "system", "user", 0, 100
        )
