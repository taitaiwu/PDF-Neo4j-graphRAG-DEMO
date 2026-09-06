import pytest

from manual_graphrag import neo4j_service


class FakeResult:
    def __init__(self, count=None):
        self.count = count
        self.consumed = False

    def consume(self):
        self.consumed = True
        return self

    def single(self):
        return None if self.count is None else {"count": self.count}


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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute_write(self, callback, *args):
        return callback(self.transaction, *args)


class FakeDriver:
    def __init__(self, transaction):
        self.transaction = transaction
        self.verified = False
        self.database = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def verify_connectivity(self):
        self.verified = True

    def session(self, *, database):
        self.database = database
        return FakeSession(self.transaction)


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
    )

    assert summary == neo4j_service.ImportSummary(2, 1)
    assert captured == {"uri": "bolt://db", "auth": ("user", "password")}
    assert driver.verified is True
    assert driver.database == "neo4j"
    assert len(transaction.calls) == 3
    assert "GraphDocument" in transaction.calls[0][0]
    assert transaction.calls[1][1]["entities"] is entities
    assert transaction.calls[2][1]["relationships"] is relationships
    assert "EXTRACTED_RELATION" in transaction.calls[2][0]


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
        )
