"""
SolQuant RAG Pipeline — Ingestion
===================================
Takes raw system logs (JSON or plain text), chunks them using LangChain's
text splitters, embeds them with all-MiniLM-L6-v2, and stores the results
in MongoDB with metadata for filtered vector search.

Supports two input modes:
  1. JSON log records  — structured with fields like timestamp, level, message
  2. Plain text        — unstructured log dumps, multi-line output, etc.
"""

import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import rag_settings
from rag.db import get_collection
from rag.embedder import embed_texts

logger = logging.getLogger("solquant.rag.ingest")


# ── Text splitter (singleton) ───────────────────────────────────────────────

_splitter: RecursiveCharacterTextSplitter | None = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    """Get or create the text splitter configured for system logs."""
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_settings.chunk_size,
            chunk_overlap=rag_settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",    # paragraph breaks
                "\n",      # line breaks (most common in logs)
                ". ",      # sentence boundaries
                ", ",      # clause boundaries
                " ",       # word boundaries
                "",        # character fallback
            ],
            is_separator_regex=False,
        )
        logger.info(
            f"Text splitter: chunk_size={rag_settings.chunk_size}, "
            f"overlap={rag_settings.chunk_overlap}"
        )
    return _splitter


def _content_hash(text: str) -> str:
    """Generate a SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── JSON log parsing ────────────────────────────────────────────────────────


def _parse_json_log(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Extract text content and metadata from a JSON log record.

    Expected fields (all optional, with fallbacks):
      - message / msg / text: the log message body
      - level / severity: log level (INFO, ERROR, etc.)
      - timestamp / time / ts: when the log was created
      - source / service / component: origin of the log

    Returns:
        (text_content, metadata_dict)
    """
    # Extract the main text
    text = (
        record.get("message")
        or record.get("msg")
        or record.get("text")
        or json.dumps(record)  # fallback: stringify the whole record
    )

    # Extract metadata with fallbacks
    metadata = {
        "log_level": (
            record.get("level")
            or record.get("severity")
            or "UNKNOWN"
        ).upper(),
        "timestamp": (
            record.get("timestamp")
            or record.get("time")
            or record.get("ts")
            or datetime.now(timezone.utc).isoformat()
        ),
        "source": (
            record.get("source")
            or record.get("service")
            or record.get("component")
            or "unknown"
        ),
    }

    # Preserve any extra fields as additional metadata
    known_keys = {
        "message", "msg", "text", "level", "severity",
        "timestamp", "time", "ts", "source", "service", "component",
    }
    extras = {k: v for k, v in record.items() if k not in known_keys}
    if extras:
        metadata["extra"] = extras

    return str(text), metadata


# ── Ingestion functions ─────────────────────────────────────────────────────


def ingest_logs(
    records: list[dict[str, Any]],
    source: str = "system",
    collection: Collection | None = None,
) -> int:
    """
    Ingest structured JSON log records into MongoDB with embeddings.

    Each record is parsed for message content and metadata, then chunked,
    embedded, and stored. Duplicate chunks (by content hash) are skipped.

    Args:
        records: List of JSON log records (dicts).
        source: Default source label if not present in record.
        collection: Optional collection override.

    Returns:
        Number of new chunks inserted.
    """
    if collection is None:
        collection = get_collection()

    all_chunks = []
    all_metadata = []

    for record in records:
        text, meta = _parse_json_log(record)
        if not meta.get("source") or meta["source"] == "unknown":
            meta["source"] = source

        splitter = _get_splitter()
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                **meta,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    return _store_chunks(all_chunks, all_metadata, collection)


def ingest_text(
    text: str,
    source: str = "plaintext",
    log_level: str = "INFO",
    collection: Collection | None = None,
) -> int:
    """
    Ingest plain text (e.g., raw log dump, multi-line output) into MongoDB.

    The text is chunked, embedded, and stored with basic metadata.

    Args:
        text: Raw text content to ingest.
        source: Source label for the log data.
        log_level: Default log level for all chunks.
        collection: Optional collection override.

    Returns:
        Number of new chunks inserted.
    """
    if collection is None:
        collection = get_collection()

    splitter = _get_splitter()
    chunks = splitter.split_text(text)

    metadata_list = [
        {
            "source": source,
            "log_level": log_level.upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i, _ in enumerate(chunks)
    ]

    return _store_chunks(chunks, metadata_list, collection)


def _store_chunks(
    chunks: list[str],
    metadata_list: list[dict],
    collection: Collection,
) -> int:
    """
    Embed chunks and insert them into MongoDB, skipping duplicates.

    Each document stored has the shape:
    {
        "text":         "the chunk content",
        "embedding":    [0.012, -0.034, ...],   # 384-dim vector
        "content_hash": "sha256hex",
        "source":       "system",
        "log_level":    "ERROR",
        "timestamp":    "2026-05-07T...",
        "chunk_index":  0,
        "total_chunks": 3,
        "ingested_at":  "2026-05-07T...",
    }
    """
    if not chunks:
        logger.info("No chunks to ingest")
        return 0

    logger.info(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)

    # Build documents, skipping duplicates
    documents = []
    hashes_seen = set()

    # Pre-fetch existing hashes to avoid duplicate DB queries
    chunk_hashes = [_content_hash(c) for c in chunks]
    existing = set()
    try:
        cursor = collection.find(
            {"content_hash": {"$in": chunk_hashes}},
            {"content_hash": 1},
        )
        existing = {doc["content_hash"] for doc in cursor}
    except Exception as e:
        logger.warning(f"Could not check for duplicates: {e}")

    for chunk, embedding, meta, h in zip(
        chunks, embeddings, metadata_list, chunk_hashes
    ):
        if h in existing or h in hashes_seen:
            continue
        hashes_seen.add(h)

        doc = {
            "text": chunk,
            rag_settings.embedding_field: embedding,
            "content_hash": h,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            **meta,
        }
        documents.append(doc)

    if not documents:
        logger.info("All chunks already exist (deduplicated)")
        return 0

    result = collection.insert_many(documents)
    inserted = len(result.inserted_ids)
    skipped = len(chunks) - inserted
    logger.info(
        f"✓ Inserted {inserted} chunks "
        f"({skipped} duplicates skipped)"
    )
    return inserted
