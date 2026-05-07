"""
SolQuant RAG Pipeline
======================
MongoDB-backed Retrieval-Augmented Generation for system logs.

Modules:
  - db:        MongoDB connection + vector search index management
  - embedder:  Local embedding model (all-MiniLM-L6-v2)
  - ingest:    Log chunking + embedding + storage pipeline
  - retriever: Vector similarity search against stored chunks
"""

from rag.db import get_collection, ensure_vector_index
from rag.embedder import get_embedding_model
from rag.ingest import ingest_logs, ingest_text
from rag.retriever import retrieve

__all__ = [
    "get_collection",
    "ensure_vector_index",
    "get_embedding_model",
    "ingest_logs",
    "ingest_text",
    "retrieve",
]
