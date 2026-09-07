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


def search_graph_evidence(
    uri: str, database: str, username: str, password: str,
    run_id: str, embedding: list[float], retrieval_mode: str, top_k: int,
) -> list[dict[str, Any]]:
    kind = "實體" if retrieval_mode == "GraphRAG" else None
    query = """
    CALL db.index.vector.queryNodes(
        'graph_evidence_embedding', $candidate_count, $embedding
    ) YIELD node, score
    WHERE node.run_id = $run_id AND ($kind IS NULL OR node.kind = $kind)
    RETURN node {
        .kind, .name, .source, .target, .text, .source_pages,
        .source_chunk_numbers, score: score
    } AS evidence
    ORDER BY score DESC LIMIT $top_k
    """
    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            with driver.session(database=database.strip()) as session:
                records = session.run(
                    query, run_id=run_id, embedding=embedding, kind=kind,
                    top_k=int(top_k), candidate_count=max(1000, int(top_k) * 50),
                ).data()
                selected = [record["evidence"] for record in records]
                if retrieval_mode == "GraphRAG" and selected:
                    names = [item.get("name", "") for item in selected]
                    related = session.run(
                        """
                        MATCH (relation:GraphEvidence {run_id: $run_id, kind: '關係'})
                        WHERE relation.source IN $names OR relation.target IN $names
                        RETURN relation {
                            .kind, .name, .source, .target, .text, .source_pages,
                            .source_chunk_numbers, score: 0.0
                        } AS evidence
                        LIMIT $top_k
                        """,
                        run_id=run_id, names=names, top_k=int(top_k),
                    ).data()
                    selected.extend(record["evidence"] for record in related)
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        raise ValueError(f"Neo4j Vector Search 失敗：{exc}") from exc
    return selected


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
    evidence: list[dict[str, Any]],
    import_mode: str = "保留既有圖譜",
    destructive_confirmed: bool = False,
) -> ImportSummary:
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")
    if not evidence or not evidence[0].get("embedding"):
        raise ValueError("沒有可寫入 Neo4j Vector Index 的向量證據")
    allowed_modes = {"保留既有圖譜", "取代最近一次圖譜", "清空本工具所有圖譜"}
    if import_mode not in allowed_modes:
        raise ValueError("不支援的 Neo4j 匯入模式")
    if import_mode != "保留既有圖譜" and not destructive_confirmed:
        raise ValueError("取代或清空資料前必須勾選確認")

    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                dimensions = len(evidence[0]["embedding"])
                session.run(
                    f"CREATE VECTOR INDEX graph_evidence_embedding IF NOT EXISTS "
                    f"FOR (e:GraphEvidence) ON (e.embedding) OPTIONS {{"
                    f"indexConfig: {{`vector.dimensions`: {dimensions}, "
                    f"`vector.similarity_function`: \x27cosine\x27}}}}}"
                ).consume()
                counts = session.execute_write(
                    _write_graph,
                    run_id,
                    document_name,
                    llm_model,
                    embedding_model,
                    schema,
                    entities,
                    relationships,
                    evidence,
                    import_mode,
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
    evidence: list[dict[str, Any]],
    import_mode: str,
) -> dict[str, int]:
    if import_mode == "清空本工具所有圖譜":
        transaction.run(
            """
            MATCH (node)
            WHERE node:GraphDocument OR node:ExtractedEntity OR node:GraphEvidence
            DETACH DELETE node
            """
        ).consume()
    elif import_mode == "取代最近一次圖譜":
        record = transaction.run(
            """
            MATCH (document:GraphDocument)
            RETURN document.run_id AS run_id
            ORDER BY document.updated_at DESC LIMIT 1
            """
        ).single()
        if record:
            transaction.run(
                "MATCH (node) WHERE node.run_id = $run_id DETACH DELETE node",
                run_id=record["run_id"],
            ).consume()

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
    transaction.run(
        """
        MATCH (document:GraphDocument {run_id: $run_id})
        UNWIND $evidence AS item
        MERGE (evidence:GraphEvidence {run_id: $run_id, evidence_id: item.evidence_id})
        SET evidence.kind = item.kind, evidence.name = item.name,
            evidence.source = item.source, evidence.target = item.target,
            evidence.text = item.text, evidence.source_pages = item.source_pages,
            evidence.source_chunk_numbers = item.source_chunk_numbers,
            evidence.embedding = item.embedding
        MERGE (evidence)-[:IN_DOCUMENT]->(document)
        """,
        run_id=run_id,
        evidence=evidence,
    ).consume()
    return {
        "entity_count": int(entity_result["count"]) if entity_result else 0,
        "relationship_count": int(relationship_result["count"])
        if relationship_result
        else 0,
    }
