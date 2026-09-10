import pytest

from manual_graphrag import neo4j_service


class FakeResult:
    def __init__(self, count=None, rows=None):
        self.count = count
        self.rows = rows or []
        self.consumed = False

    def consume(self):
        self.consumed = True
        return self

    def single(self):
        return None if self.count is None else {"count": self.count}

    def data(self):
        return self.rows


class FakeTransaction:
    def __init__(self):
        self.calls = []

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "RETURN count(entity)" in query:
            return FakeResult(2)
        if "RETURN count(relation)" in query:
            return FakeResult(1)
        return FakeResult()


class FakeSession:
    def __init__(self, transaction):
        self.transaction = transaction
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return FakeResult()

    def execute_write(self, callback, *args):
        return callback(self.transaction, *args)


class FakeDriver:
    def __init__(self, transaction):
        self.transaction = transaction
        self.verified = False
        self.database = None
        self.session_instance = FakeSession(transaction)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def verify_connectivity(self):
        self.verified = True

    def session(self, *, database):
        self.database = database
        return self.session_instance


def test_import_extraction_writes_document_entities_and_relationships(monkeypatch) -> None:
    transaction = FakeTransaction()
    driver = FakeDriver(transaction)
    captured = {}

    def fake_driver(uri, auth):
        captured.update({"uri": uri, "auth": auth})
        return driver

    monkeypatch.setattr(neo4j_service.GraphDatabase, "driver", fake_driver)
    entities = [
        {
            "name": "設備 A",
            "type": "DEVICE",
            "description": "設備",
            "source_chunk_numbers": [1],
            "source_pages": [2],
        },
        {
            "name": "設備 B",
            "type": "DEVICE",
            "description": "配件",
            "source_chunk_numbers": [1],
            "source_pages": [2],
        },
    ]
    relationships = [
        {
            "source": "設備 A",
            "type": "USES",
            "target": "設備 B",
            "description": "使用",
            "source_chunk_numbers": [1],
            "source_pages": [2],
        }
    ]

    summary = neo4j_service.import_extraction(
        "bolt://db",
        "neo4j",
        "user",
        "password",
        "run-1",
        "manual.pdf",
        "llm",
        "embed",
        {"entity_types": [{"name": "DEVICE"}]},
        entities,
        relationships,
        [{
            "evidence_id": "chunk-1", "kind": "原文", "text": "設備說明",
            "embedding": [0.1], "source_pages": [2],
            "source_chunk_numbers": [1],
        }],
    )

    assert summary == neo4j_service.ImportSummary(2, 1)
    assert captured == {"uri": "bolt://db", "auth": ("user", "password")}
    assert driver.verified is True
    assert driver.database == "neo4j"
    assert len(driver.session(database="neo4j").calls) == 2
    assert "CREATE VECTOR INDEX graph_evidence_embedding_1" in driver.session(database="neo4j").calls[0][0]
    assert "CREATE FULLTEXT INDEX" in driver.session(database="neo4j").calls[1][0]
    assert "fulltext.analyzer" in driver.session(database="neo4j").calls[1][0]
    assert len(transaction.calls) == 4
    assert "GraphDocument" in transaction.calls[0][0]
    assert transaction.calls[0][1]["embedding_dimensions"] == 1
    assert transaction.calls[0][1]["vector_index_name"] == "graph_evidence_embedding_1"
    assert transaction.calls[1][1]["entities"] is entities
    assert transaction.calls[2][1]["relationships"] is relationships
    assert "EXTRACTED_RELATION" in transaction.calls[2][0]
    assert "GraphEvidence" in transaction.calls[3][0]


@pytest.mark.parametrize(
    "values,message",
    [
        (("", "neo4j", "user", "password"), "Neo4j URI"),
        (("bolt://db", "", "user", "password"), "Neo4j Database"),
        (("bolt://db", "neo4j", "", "password"), "Neo4j Username"),
        (("bolt://db", "neo4j", "user", ""), "Neo4j Password"),
    ],
)
def test_import_extraction_validates_connection_fields(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        neo4j_service.import_extraction(
            *values,
            "run-1",
            "manual.pdf",
            "llm",
            "embed",
            {},
            [],
            [],
            [],
        )


def test_import_extraction_wraps_driver_errors(monkeypatch) -> None:
    def broken_driver(uri, auth):
        raise OSError("database unavailable")

    monkeypatch.setattr(neo4j_service.GraphDatabase, "driver", broken_driver)

    with pytest.raises(ValueError, match="Neo4j 寫入失敗"):
        neo4j_service.import_extraction(
            "bolt://db",
            "neo4j",
            "user",
            "password",
            "run-1",
            "manual.pdf",
            "llm",
            "embed",
            {},
            [],
            [],
            [{"embedding": [0.1]}],
        )


def _evidence(evidence_id: str, score: float, kind: str = "原文") -> dict:
    return {
        "evidence_id": evidence_id,
        "kind": kind,
        "name": evidence_id if kind == "實體" else "",
        "source": "",
        "target": "",
        "text": evidence_id,
        "source_pages": [1],
        "source_chunk_numbers": [1],
        "score": score,
    }


def test_vector_index_name_uses_embedding_dimensions() -> None:
    assert neo4j_service.vector_index_name(1536) == "graph_evidence_embedding_1536"
    assert neo4j_service.vector_index_name(3072) == "graph_evidence_embedding_3072"
    with pytest.raises(ValueError, match="維度"):
        neo4j_service.vector_index_name(0)


def test_escape_fulltext_query_escapes_lucene_syntax() -> None:
    assert neo4j_service._escape_fulltext_query("E01 +(重試):A/B") == (
        r"E01 \+\(重試\)\:A\/B"
    )


def test_format_hybrid_record_preserves_evidence_and_official_score() -> None:
    item = neo4j_service._format_hybrid_record({
        "evidence": _evidence("both", 0.0),
        "score": 0.85,
    })

    assert item.content["evidence_id"] == "both"
    assert item.content["matched_by"] == ["official-hybrid"]
    assert item.content["fusion_score"] == 0.85
    assert item.metadata == {"score": 0.85}


class SearchSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "count(node)" in query:
            return FakeResult(3)
        raise AssertionError(f"unexpected query: {query}")


class SearchDriver:
    def __init__(self, session):
        self.session_instance = session

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def session(self, *, database):
        return self.session_instance


class FakeOfficialRetriever:
    initialization = None
    search_arguments = None

    def __init__(self, **kwargs):
        type(self).initialization = kwargs

    def search(self, **kwargs):
        type(self).search_arguments = kwargs
        evidence = [
            _evidence("both", 0.9),
            _evidence("vector-only", 0.8),
            _evidence("keyword-only", 0.7),
        ]
        for item in evidence:
            item.update({
                "matched_by": ["official-hybrid"],
                "fusion_score": item["score"],
            })
        return type("Result", (), {
            "items": [type("Item", (), {"content": item}) for item in evidence]
        })()


def test_search_dimension_error_instructs_user_to_reimport(monkeypatch) -> None:
    session = SearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )

    class BrokenRetriever:
        def __init__(self, **kwargs):
            pass

        def search(self, **kwargs):
            raise ValueError(
                "Vector index has configured dimensionality 3072, "
                "but the provided vector has dimension 1536"
            )

    monkeypatch.setattr(neo4j_service, "HybridCypherRetriever", BrokenRetriever)

    with pytest.raises(ValueError, match="重新執行.*Embedding 並匯入 Neo4j"):
        neo4j_service.search_graph_evidence(
            "bolt://db", "neo4j", "user", "password", "run-1",
            "question", [0.1] * 1536, "向量 RAG", 3,
        )


def test_search_graph_evidence_uses_official_hybrid_retriever(monkeypatch) -> None:
    session = SearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )
    monkeypatch.setattr(
        neo4j_service, "HybridCypherRetriever", FakeOfficialRetriever
    )

    results = neo4j_service.search_graph_evidence(
        "bolt://db", "neo4j", "user", "password", "run-1",
        "E01 +(重試)", [0.1], "向量 RAG", 3,
    )

    assert [item["evidence_id"] for item in results] == [
        "both", "vector-only", "keyword-only",
    ]
    initialization = FakeOfficialRetriever.initialization
    assert initialization["vector_index_name"] == "graph_evidence_embedding_1"
    assert initialization["fulltext_index_name"] == "graph_evidence_fulltext"
    assert initialization["neo4j_database"] == "neo4j"
    assert "$run_id" in initialization["retrieval_query"]
    arguments = FakeOfficialRetriever.search_arguments
    assert arguments["query_text"] == r"E01 \+\(重試\)"
    assert arguments["query_vector"] == [0.1]
    assert arguments["top_k"] == 3
    assert arguments["effective_search_ratio"] == 3
    assert arguments["query_params"] == {"run_id": "run-1"}
    assert arguments["ranker"] == "naive"
