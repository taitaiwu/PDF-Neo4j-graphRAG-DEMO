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
        "http://localhost:11434/v1", "secret", "model-a", chunks
    )

    assert plan.schema == SCHEMA
    assert plan.sampled_chunks == 1
    assert plan.total_chunks == 1
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert "[CHUNK 1; PAGES 2]" in captured["payload"]["messages"][1]["content"]


def test_schema_planning_samples_across_large_document(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 11)]
    prompts: list[str] = []

    def fake_chat(base_url, api_key, model, system_prompt, user_prompt, temperature=0):
        prompts.append(user_prompt)
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    plan = graph_service.plan_graph_schema("http://models/v1", "", "llm", chunks)

    assert 1 < plan.sampled_chunks < plan.total_chunks
    assert "CHUNK 1" in prompts[0]
    assert "CHUNK 10" in prompts[0]


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
