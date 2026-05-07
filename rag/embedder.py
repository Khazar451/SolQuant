"""
SolQuant RAG Pipeline — Embedding Model
=========================================
Loads a lightweight local embedding model (all-MiniLM-L6-v2) via
sentence-transformers for converting text chunks into dense vectors.

Model specs:
  - Name:       all-MiniLM-L6-v2
  - Dimensions: 384
  - Max tokens: 256
  - Size:       ~80 MB
  - Speed:      ~14,000 sentences/sec on GPU, ~1,200/sec on CPU
"""

import logging
from typing import Union

from sentence_transformers import SentenceTransformer

from rag.config import rag_settings

logger = logging.getLogger("solquant.rag.embedder")

# ── Singleton ───────────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load or return the cached embedding model."""
    global _model
    if _model is None:
        logger.info(
            f"Loading embedding model: {rag_settings.embedding_model_name} "
            f"(device={rag_settings.embedding_device})"
        )
        _model = SentenceTransformer(
            rag_settings.embedding_model_name,
            device=rag_settings.embedding_device,
        )
        logger.info(
            f"✓ Embedding model loaded — "
            f"dim={_model.get_sentence_embedding_dimension()}"
        )
    return _model


def embed_texts(texts: Union[str, list[str]]) -> list[list[float]]:
    """
    Embed one or more texts into dense vectors.

    Args:
        texts: A single string or list of strings.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if isinstance(texts, str):
        texts = [texts]

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=rag_settings.embedding_batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,  # unit vectors for cosine similarity
    )

    # Convert numpy arrays to plain lists for MongoDB storage
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Uses the same model as document embedding. For symmetric retrieval
    (query and docs share the same embedding space), this is correct.
    For asymmetric models, you'd use encode_query() instead.

    Args:
        query: The search query text.

    Returns:
        A single embedding vector as a list of floats.
    """
    return embed_texts(query)[0]
