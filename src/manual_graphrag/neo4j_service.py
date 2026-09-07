from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError


@dataclass(frozen=True)
class ImportSummary:
    entity_count: int
    relationship_count: int


def check_neo4j_connection(
    uri: str, database: str, username: str, password: str
) -> None:
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")
    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                session.run("RETURN 1 AS value").consume()
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("請先填寫"):
            raise
        raise ValueError(f"Neo4j 連線失敗：{exc}") from exc


def load_latest_graph(
    uri: str, database: str, username: str, password: str
) -> dict[str, Any]:
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")
    query = """
    MATCH (document:GraphDocument)
    WITH document ORDER BY document.updated_at DESC LIMIT 1
    OPTIONAL MATCH (entity:ExtractedEntity)-[:IN_DOCUMENT]->(document)
    WITH document, collect(DISTINCT entity {
        .name, .type, .description, .source_chunk_numbers, .source_pages
    }) AS entities
    OPTIONAL MATCH (source:ExtractedEntity)-[relation:EXTRACTED_RELATION]->(target:ExtractedEntity)
    WHERE relation.run_id = document.run_id
    RETURN document.run_id AS run_id, document.file_name AS document,
           document.embedding_model AS embedding_model, entities,
           collect(DISTINCT relation {
               source: source.name, target: target.name, .type, .description,
               .source_chunk_numbers, .source_pages
           }) AS relationships
    """
    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                record = session.run(query).single()
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("請先填寫"):
            raise
        raise ValueError(f"Neo4j 查詢失敗：{exc}") from exc
    if record is None:
        raise ValueError("Neo4j 中沒有可供問答的 GraphDocument")
    result = dict(record)
    result["neo4j_imported"] = True
    result["entities"] = [item for item in result.get("entities", []) if item]
    result["relationships"] = [item for item in result.get("relationships", []) if item]
    return result


def import_extraction(
    uri: str,
    database: str,
    username: str,
    password: str,
    run_id: str,
    document_name: str,
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> ImportSummary:
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")

    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                counts = session.execute_write(
                    _write_graph,
                    run_id,
                    document_name,
                    llm_model,
                    embedding_model,
                    schema,
                    entities,
                    relationships,
                )
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("請先填寫"):
            raise
        raise ValueError(f"Neo4j 寫入失敗：{exc}") from exc
    return ImportSummary(counts["entity_count"], counts["relationship_count"])


def _write_graph(
    transaction: Any,
    run_id: str,
    document_name: str,
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> dict[str, int]:
    transaction.run(
        """
        MERGE (document:GraphDocument {run_id: $run_id})
        SET document.file_name = $document_name,
            document.llm_model = $llm_model,
            document.embedding_model = $embedding_model,
            document.schema_json = $schema_json,
            document.updated_at = datetime()
        """,
        run_id=run_id,
        document_name=document_name,
        llm_model=llm_model,
        embedding_model=embedding_model,
        schema_json=json.dumps(schema, ensure_ascii=False),
    ).consume()
    entity_result = transaction.run(
        """
        MATCH (document:GraphDocument {run_id: $run_id})
        UNWIND $entities AS item
        MERGE (entity:ExtractedEntity {
            run_id: $run_id,
            type: item.type,
            name: item.name
        })
        SET entity.description = item.description,
            entity.source_chunk_numbers = item.source_chunk_numbers,
            entity.source_pages = item.source_pages
        MERGE (entity)-[:IN_DOCUMENT]->(document)
        RETURN count(entity) AS count
        """,
        run_id=run_id,
        entities=entities,
    ).single()
    relationship_result = transaction.run(
        """
        UNWIND $relationships AS item
        MATCH (source:ExtractedEntity {run_id: $run_id, name: item.source})
        MATCH (target:ExtractedEntity {run_id: $run_id, name: item.target})
        WITH item, head(collect(source)) AS source, head(collect(target)) AS target
        MERGE (source)-[relation:EXTRACTED_RELATION {
            run_id: $run_id,
            type: item.type
        }]->(target)
        SET relation.description = item.description,
            relation.source_chunk_numbers = item.source_chunk_numbers,
            relation.source_pages = item.source_pages
        RETURN count(relation) AS count
        """,
        run_id=run_id,
        relationships=relationships,
    ).single()
    return {
        "entity_count": int(entity_result["count"]) if entity_result else 0,
        "relationship_count": int(relationship_result["count"])
        if relationship_result
        else 0,
    }
