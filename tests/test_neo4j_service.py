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
    assert "CREATE VECTOR INDEX" in driver.session(database="neo4j").calls[0][0]
    assert "CREATE FULLTEXT INDEX" in driver.session(database="neo4j").calls[1][0]
    assert "fulltext.analyzer" in driver.session(database="neo4j").calls[1][0]
    assert len(transaction.calls) == 4
    assert "GraphDocument" in transaction.calls[0][0]
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


def test_escape_fulltext_query_escapes_lucene_syntax() -> None:
    assert neo4j_service._escape_fulltext_query("E01 +(重試):A/B") == (
        r"E01 \+\(重試\)\:A\/B"
    )


def test_fuse_search_results_deduplicates_and_uses_both_rankings() -> None:
    results = neo4j_service._fuse_search_results(
        [_evidence("vector-only", 0.9), _evidence("both", 0.8)],
        [_evidence("both", 4.0), _evidence("keyword-only", 3.0)],
        3,
    )

    assert [item["evidence_id"] for item in results] == [
        "both", "vector-only", "keyword-only",
    ]
    both = results[0]
    assert both["matched_by"] == ["vector", "fulltext"]
    assert both["vector_score"] == 0.8
    assert both["keyword_score"] == 4.0
    assert both["fusion_score"] > results[1]["fusion_score"]


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
        if "vector.queryNodes" in query:
            return FakeResult(rows=[
                {"evidence": _evidence("vector-only", 0.9)},
                {"evidence": _evidence("both", 0.8)},
            ])
        if "fulltext.queryNodes" in query:
            return FakeResult(rows=[
                {"evidence": _evidence("both", 4.0)},
                {"evidence": _evidence("keyword-only", 3.0)},
            ])
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


def test_search_graph_evidence_combines_vector_and_fulltext(monkeypatch) -> None:
    session = SearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )

    results = neo4j_service.search_graph_evidence(
        "bolt://db", "neo4j", "user", "password", "run-1",
        "E01 +(重試)", [0.1], "向量 RAG", 3,
    )

    assert [item["evidence_id"] for item in results] == [
        "both", "vector-only", "keyword-only",
    ]
    fulltext_call = next(
        call for call in session.calls if "fulltext.queryNodes" in call[0]
    )
    assert fulltext_call[1]["run_id"] == "run-1"
    assert fulltext_call[1]["question"] == r"E01 \+\(重試\)"
    assert fulltext_call[1]["candidate_limit"] == 9
