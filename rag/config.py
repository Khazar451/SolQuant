"""
SolQuant RAG Pipeline — Configuration
=======================================
All tunables for the MongoDB vector store and embedding pipeline.
"""

from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    """RAG pipeline configuration loaded from environment or .env file."""

    # ── MongoDB ─────────────────────────────────────────────────────────
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "solquant_rag"
    mongo_collection_name: str = "system_logs"

    # ── Vector search index ─────────────────────────────────────────────
    vector_index_name: str = "vector_search_index"
    embedding_field: str = "embedding"       # field storing the vector
    embedding_dimensions: int = 384          # all-MiniLM-L6-v2 output dim
    similarity_metric: str = "cosine"        # cosine | euclidean | dotProduct

    # ── Index creation polling ──────────────────────────────────────────
    index_poll_interval_sec: float = 2.0     # how often to check index status
    index_poll_timeout_sec: float = 120.0    # give up after this many seconds

    # ── Embedding model ─────────────────────────────────────────────────
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"            # cpu | cuda
    embedding_batch_size: int = 64

    # ── Text chunking ───────────────────────────────────────────────────
    chunk_size: int = 512                    # characters per chunk
    chunk_overlap: int = 64                  # overlap between adjacent chunks

    # ── Retrieval ───────────────────────────────────────────────────────
    retrieval_top_k: int = 3                 # number of results to return
    retrieval_score_threshold: float = 0.0   # minimum similarity score

    model_config = {"env_prefix": "RAG_", "env_file": ".env"}


rag_settings = RAGSettings()
