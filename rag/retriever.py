"""
SolQuant RAG Pipeline — Retriever
===================================
Performs vector similarity search against the MongoDB collection
using the $vectorSearch aggregation stage.

Returns the top-K most relevant context chunks for a given query.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from pymongo.collection import Collection

from rag.config import rag_settings
from rag.db import get_collection
from rag.embedder import embed_query

logger = logging.getLogger("solquant.rag.retriever")


# ── Result dataclass ────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A single retrieved context chunk with metadata."""

    text: str
    score: float
    source: str = "unknown"
    log_level: str = "UNKNOWN"
    timestamp: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "log_level": self.log_level,
            "timestamp": self.timestamp,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
        }


# ── Public API ──────────────────────────────────────────────────────────────


def retrieve(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    log_level_filter: str | None = None,
    collection: Collection | None = None,
) -> list[RetrievalResult]:
    """
    Embed a query and perform vector search against MongoDB.

    Uses the $vectorSearch aggregation stage (MongoDB Atlas Vector Search).
    Supports optional pre-filtering by source and log_level fields.

    Args:
        query: The natural-language search query.
        top_k: Number of results to return (default from config: 3).
        source_filter: Optional filter on the 'source' field.
        log_level_filter: Optional filter on the 'log_level' field.
        collection: Optional collection override.

    Returns:
        List of RetrievalResult, sorted by descending similarity score.
    """
    if collection is None:
        collection = get_collection()
    if top_k is None:
        top_k = rag_settings.retrieval_top_k

    # ── Embed the query ─────────────────────────────────────────────
    logger.info(f"Embedding query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
    query_vector = embed_query(query)

    # ── Build the $vectorSearch stage ───────────────────────────────
    vector_search_stage = {
        "$vectorSearch": {
            "index": rag_settings.vector_index_name,
            "path": rag_settings.embedding_field,
            "queryVector": query_vector,
            "numCandidates": top_k * 10,  # over-fetch for better recall
            "limit": top_k,
        }
    }

    # Add pre-filters if specified
    filter_conditions = {}
    if source_filter:
        filter_conditions["source"] = source_filter
    if log_level_filter:
        filter_conditions["log_level"] = log_level_filter.upper()
    if filter_conditions:
        vector_search_stage["$vectorSearch"]["filter"] = filter_conditions

    # ── Project the results ─────────────────────────────────────────
    project_stage = {
        "$project": {
            "_id": 0,
            "text": 1,
            "source": 1,
            "log_level": 1,
            "timestamp": 1,
            "chunk_index": 1,
            "total_chunks": 1,
            "extra": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
    }

    # ── Execute the aggregation ─────────────────────────────────────
    pipeline = [vector_search_stage, project_stage]

    logger.info(
        f"Executing $vectorSearch: top_k={top_k}, "
        f"filters={filter_conditions or 'none'}"
    )

    try:
        cursor = collection.aggregate(pipeline)
        results = list(cursor)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise RuntimeError(f"Vector search query failed: {e}") from e

    # ── Parse results ───────────────────────────────────────────────
    retrieved = []
    for doc in results:
        score = doc.get("score", 0.0)

        # Apply minimum score threshold
        if score < rag_settings.retrieval_score_threshold:
            continue

        result = RetrievalResult(
            text=doc.get("text", ""),
            score=score,
            source=doc.get("source", "unknown"),
            log_level=doc.get("log_level", "UNKNOWN"),
            timestamp=doc.get("timestamp", ""),
            chunk_index=doc.get("chunk_index", 0),
            total_chunks=doc.get("total_chunks", 1),
            metadata=doc.get("extra", {}),
        )
        retrieved.append(result)

    logger.info(
        f"✓ Retrieved {len(retrieved)} chunks "
        f"(scores: {[r.score for r in retrieved]})"
    )

    return retrieved


def retrieve_as_context(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    log_level_filter: str | None = None,
    separator: str = "\n---\n",
    collection: Collection | None = None,
) -> str:
    """
    Convenience wrapper: retrieve chunks and concatenate into a single
    context string suitable for LLM prompt injection.

    Args:
        query: The search query.
        top_k: Number of chunks.
        source_filter: Optional source filter.
        log_level_filter: Optional log level filter.
        separator: String to join chunks with.
        collection: Optional collection override.

    Returns:
        Concatenated context string with source attribution.
    """
    results = retrieve(
        query=query,
        top_k=top_k,
        source_filter=source_filter,
        log_level_filter=log_level_filter,
        collection=collection,
    )

    if not results:
        return ""

    blocks = []
    for i, r in enumerate(results, 1):
        header = (
            f"[Context {i}/{len(results)} | "
            f"source={r.source} | level={r.log_level} | "
            f"score={r.score:.3f}]"
        )
        blocks.append(f"{header}\n{r.text}")

    return separator.join(blocks)
