from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from neo4j_graphrag.exceptions import Neo4jGraphRagError
from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem


@dataclass(frozen=True)
class ImportSummary:
    entity_count: int
    relationship_count: int


def vector_index_name(dimensions: int) -> str:
    if int(dimensions) < 1:
        raise ValueError("Embedding 向量維度必須大於 0")
    return f"graph_evidence_embedding_{int(dimensions)}"


def _escape_fulltext_query(question: str) -> str:
    """Escape Lucene query syntax while preserving terms for the CJK analyzer."""
    return re.sub(
        r"""([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)""",
        r"""\\\1""",
        question.strip(),
    )


def _format_hybrid_record(record: Any) -> RetrieverResultItem:
    evidence = dict(record["evidence"])
    score = float(record["score"])
    evidence.update({
        "score": score,
        "fusion_score": score,
        "matched_by": ["official-hybrid"],
    })
    return RetrieverResultItem(content=evidence, metadata={"score": score})


def _expanded_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        **evidence,
        "score": 0.0,
        "fusion_score": 0.0,
        "matched_by": ["graph"],
    }


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
        .name, .type, .description, .source_chunk_numbers, .source_pages,
        .source_documents
    }) AS entities
    OPTIONAL MATCH (source:ExtractedEntity)-[relation:EXTRACTED_RELATION]->(target:ExtractedEntity)
    WHERE relation.run_id = document.run_id
    RETURN document.run_id AS run_id, document.file_name AS document,
           document.embedding_model AS embedding_model,
           document.embedding_dimensions AS embedding_dimensions,
           document.vector_index_name AS vector_index_name, entities,
           collect(DISTINCT relation {
               source: source.name, target: target.name, .type, .description,
               .source_chunk_numbers, .source_pages, .source_documents
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
    uri: str,
    database: str,
    username: str,
    password: str,
    run_id: str,
    question: str,
    embedding: list[float],
    retrieval_mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    target_vector_index = vector_index_name(len(embedding))
    retrieval_query = """
    WITH node, score
    WHERE node.run_id = $run_id
    RETURN node {
        .evidence_id, .kind, .name, .source, .target, .text, .source_pages,
        .source_chunk_numbers, .source_documents
    } AS evidence, score
    """
    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            with driver.session(database=database.strip()) as session:
                total_record = session.run(
                    "MATCH (node:GraphEvidence) RETURN count(node) AS count"
                ).single()
                candidate_count = max(
                    int(top_k),
                    int(total_record["count"]) if total_record else int(top_k),
                )

            retriever = HybridCypherRetriever(
                driver=driver,
                vector_index_name=target_vector_index,
                fulltext_index_name="graph_evidence_fulltext",
                retrieval_query=retrieval_query,
                result_formatter=_format_hybrid_record,
                neo4j_database=database.strip(),
            )
            result = retriever.search(
                query_text=_escape_fulltext_query(question),
                query_vector=embedding,
                top_k=candidate_count,
                effective_search_ratio=3,
                query_params={"run_id": run_id},
                ranker="naive",
            )
            selected = [
                dict(item.content) for item in result.items
                if isinstance(item.content, dict)
            ][:int(top_k)]

            if retrieval_mode in {"關聯擴展檢索", "GraphRAG"} and selected:
                with driver.session(database=database.strip()) as session:
                    chunk_numbers = sorted({
                        number
                        for item in selected
                        for number in item.get("source_chunk_numbers", [])
                    })
                    names = [
                        item.get("name", "")
                        for item in selected
                        if item.get("kind") == "實體" and item.get("name")
                    ]
                    graph_chunk_numbers = list(dict.fromkeys(
                        number
                        for item in selected
                        if item.get("kind") != "原文"
                        for number in item.get("source_chunk_numbers", [])
                    ))
                    source_chunk_records = session.run(
                        """
                        MATCH (chunk:GraphEvidence {run_id: $run_id, kind: "原文"})
                        WHERE any(
                            number IN coalesce(chunk.source_chunk_numbers, [])
                            WHERE number IN $chunk_numbers
                        )
                        RETURN chunk {
                            .evidence_id, .kind, .name, .source, .target, .text,
                            .source_pages, .source_chunk_numbers, .source_documents
                        } AS evidence
                        """,
                        run_id=run_id,
                        chunk_numbers=graph_chunk_numbers,
                    ).data()
                    source_chunks = [
                        _expanded_evidence(record["evidence"])
                        for record in source_chunk_records
                    ]
                    chunk_priority = {
                        number: index for index, number in enumerate(graph_chunk_numbers)
                    }
                    source_chunks.sort(key=lambda item: min(
                        (
                            chunk_priority[number]
                            for number in item.get("source_chunk_numbers", [])
                            if number in chunk_priority
                        ),
                        default=len(chunk_priority),
                    ))
                    selected.extend(source_chunks[:int(top_k)])
                    selected_ids = {item.get("evidence_id", "") for item in selected}
                    related_entities = session.run(
                        """
                        MATCH (entity:GraphEvidence {run_id: $run_id, kind: '實體'})
                        WHERE entity.name IN $names OR any(
                            number IN coalesce(entity.source_chunk_numbers, [])
                            WHERE number IN $chunk_numbers
                        )
                        RETURN entity {
                            .evidence_id, .kind, .name, .source, .target, .text,
                            .source_pages, .source_chunk_numbers, .source_documents
                        } AS evidence
                        LIMIT $top_k
                        """,
                        run_id=run_id,
                        names=names,
                        chunk_numbers=chunk_numbers,
                        top_k=int(top_k),
                    ).data()
                    entity_evidence = [
                        _expanded_evidence(record["evidence"])
                        for record in related_entities
                    ]
                    selected.extend(
                        item for item in entity_evidence
                        if item.get("evidence_id", "") not in selected_ids
                    )
                    expanded_names = list(dict.fromkeys(
                        names + [item.get("name", "") for item in entity_evidence]
                    ))
                    selected_ids = {item.get("evidence_id", "") for item in selected}
                    related = session.run(
                        """
                        MATCH (relation:GraphEvidence {run_id: $run_id, kind: '關係'})
                        WHERE relation.source IN $names OR relation.target IN $names OR any(
                            number IN coalesce(relation.source_chunk_numbers, [])
                            WHERE number IN $chunk_numbers
                        )
                        RETURN relation {
                            .evidence_id, .kind, .name, .source, .target, .text,
                            .source_pages, .source_chunk_numbers, .source_documents
                        } AS evidence
                        LIMIT $top_k
                        """,
                        run_id=run_id,
                        names=expanded_names,
                        chunk_numbers=chunk_numbers,
                        top_k=int(top_k),
                    ).data()
                    selected.extend(
                        _expanded_evidence(record["evidence"])
                        for record in related
                        if record["evidence"].get("evidence_id", "") not in selected_ids
                    )
    except (DriverError, Neo4jError, Neo4jGraphRagError, OSError, ValueError) as exc:
        detail = str(exc)
        guidance = ""
        if "dimensionality" in detail.casefold() or "dimension" in detail.casefold() or "index" in detail.casefold():
            guidance = " 請使用目前的 Embedding 模型重新執行「Embedding 並匯入 Neo4j」以建立對應維度索引。"
        raise ValueError(f"Neo4j 官方混合檢索失敗：{exc}{guidance}") from exc
    except Exception as exc:
        # neo4j-graphrag 1.19.0 tries to format a missing-index error with the
        # nonexistent `self.index_name` attribute. Replace that implementation
        # detail with the actual index requested by this application.
        if isinstance(exc, AttributeError) and "index_name" in str(exc):
            detail = f"找不到向量索引 `{target_vector_index}`。"
        else:
            detail = str(exc)
        raise ValueError(
            f"Neo4j 官方混合檢索失敗：{detail} "
            "請使用目前的 Embedding 模型重新執行「Embedding 並匯入 Neo4j」。"
        ) from exc
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
    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                dimensions = len(evidence[0]["embedding"])
                index_name = vector_index_name(dimensions)
                # Neo4j permits only one vector index for the same label and
                # property schema. IF NOT EXISTS would silently keep an older
                # index with a different dimension, so remove all indexes
                # owned by this application before creating the current one.
                existing_indexes = session.run(
                    "SHOW VECTOR INDEXES YIELD name "
                    "WHERE name = 'graph_evidence_embedding' "
                    "OR name STARTS WITH 'graph_evidence_embedding_' "
                    "RETURN name"
                ).data()
                for record in existing_indexes:
                    old_index_name = str(record.get("name", ""))
                    if re.fullmatch(r"graph_evidence_embedding(?:_\d+)?", old_index_name):
                        session.run(f"DROP INDEX `{old_index_name}` IF EXISTS").consume()
                session.run(
                    f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                    "FOR (e:GraphEvidence) ON (e.embedding) OPTIONS {"
                    "indexConfig: {`vector.dimensions`: "
                    f"{dimensions}, "
                    "`vector.similarity_function`: 'cosine'}}"
                ).consume()
                session.run(
                    "CREATE FULLTEXT INDEX graph_evidence_fulltext IF NOT EXISTS "
                    "FOR (e:GraphEvidence) ON EACH [e.text, e.name, e.source, e.target] "
                    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}"
                ).consume()
                # Waiting by a just-created index name can race Neo4j schema
                # propagation and incorrectly raise IndexNotFound. Await all
                # indexes only after both DDL statements have committed.
                session.run("CALL db.awaitIndexes(300)").consume()
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
                    dimensions,
                    index_name,
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
    embedding_dimensions: int,
    vector_index: str,
) -> dict[str, int]:
    transaction.run(
        """
        MATCH (node)
        WHERE node:GraphDocument OR node:ExtractedEntity OR node:GraphEvidence
        DETACH DELETE node
        """
    ).consume()

    transaction.run(
        """
        MERGE (document:GraphDocument {run_id: $run_id})
        SET document.file_name = $document_name,
            document.llm_model = $llm_model,
            document.embedding_model = $embedding_model,
            document.embedding_dimensions = $embedding_dimensions,
            document.vector_index_name = $vector_index_name,
            document.schema_json = $schema_json,
            document.updated_at = datetime()
        """,
        run_id=run_id,
        document_name=document_name,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        vector_index_name=vector_index,
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
            entity.source_pages = item.source_pages,
            entity.source_documents = item.source_documents
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
            relation.source_pages = item.source_pages,
            relation.source_documents = item.source_documents
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
            evidence.source_documents = item.source_documents,
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
