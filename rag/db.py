"""
SolQuant RAG Pipeline — MongoDB Connection & Vector Index
==========================================================
Handles:
  1. Connection to a local MongoDB instance via pymongo.
  2. Creation and async polling of a vector search index using
     SearchIndexModel with cosine similarity.

MongoDB Atlas Search / Atlas Vector Search requires MongoDB 7.0+
with the $vectorSearch aggregation stage. For local dev, you can
use mongodb-atlas-local (Docker) which supports vector search.

Docker one-liner for local Atlas:
  docker run -p 27017:27017 mongodb/mongodb-atlas-local:8.0
"""

import logging
import time

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from rag.config import rag_settings

logger = logging.getLogger("solquant.rag.db")

# ── Module-level singletons ─────────────────────────────────────────────────

_client: MongoClient | None = None
_collection: Collection | None = None


def get_client() -> MongoClient:
    """Get or create the MongoDB client (singleton)."""
    global _client
    if _client is None:
        logger.info(f"Connecting to MongoDB at {rag_settings.mongo_uri}")
        _client = MongoClient(rag_settings.mongo_uri)
        # Verify connectivity
        _client.admin.command("ping")
        logger.info("MongoDB connection established")
    return _client


def get_collection() -> Collection:
    """Get or create the target collection (singleton)."""
    global _collection
    if _collection is None:
        client = get_client()
        db = client[rag_settings.mongo_db_name]
        _collection = db[rag_settings.mongo_collection_name]
        logger.info(
            f"Using collection: "
            f"{rag_settings.mongo_db_name}.{rag_settings.mongo_collection_name}"
        )
    return _collection


# ── Vector Search Index ─────────────────────────────────────────────────────


def _build_index_definition() -> dict:
    """
    Build the vector search index definition.

    Structure follows the MongoDB Atlas Vector Search spec:
    https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
    """
    return {
        "fields": [
            {
                "type": "vector",
                "path": rag_settings.embedding_field,
                "numDimensions": rag_settings.embedding_dimensions,
                "similarity": rag_settings.similarity_metric,
            },
            # Include a filter field for source-based pre-filtering
            {
                "type": "filter",
                "path": "source",
            },
            {
                "type": "filter",
                "path": "log_level",
            },
        ]
    }


def _index_exists(collection: Collection, index_name: str) -> bool:
    """Check if a vector search index with the given name already exists."""
    try:
        existing = list(collection.list_search_indexes(name=index_name))
        return len(existing) > 0
    except Exception as e:
        logger.debug(f"Could not list search indexes: {e}")
        return False


def _poll_index_ready(
    collection: Collection,
    index_name: str,
    poll_interval: float,
    timeout: float,
) -> bool:
    """
    Asynchronously poll until the vector search index is READY.

    MongoDB creates vector search indexes asynchronously. After calling
    create_search_index(), the index transitions through states:
      PENDING → INITIAL_SYNC → READY

    Args:
        collection: The MongoDB collection.
        index_name: Name of the search index.
        poll_interval: Seconds between status checks.
        timeout: Maximum wait time in seconds.

    Returns:
        True if the index reached READY state, False on timeout.
    """
    start = time.time()
    logger.info(
        f"Polling for index '{index_name}' readiness "
        f"(timeout={timeout}s, interval={poll_interval}s)..."
    )

    while time.time() - start < timeout:
        try:
            indexes = list(collection.list_search_indexes(name=index_name))
            if indexes:
                status = indexes[0].get("status", "UNKNOWN")
                queryable = indexes[0].get("queryable", False)
                elapsed = time.time() - start
                logger.info(
                    f"Index '{index_name}': status={status}, "
                    f"queryable={queryable} ({elapsed:.1f}s elapsed)"
                )
                if queryable:
                    logger.info(
                        f"✓ Index '{index_name}' is READY and queryable "
                        f"after {elapsed:.1f}s"
                    )
                    return True
        except Exception as e:
            logger.warning(f"Error polling index status: {e}")

        time.sleep(poll_interval)

    elapsed = time.time() - start
    logger.error(
        f"✗ Index '{index_name}' did not become ready "
        f"within {elapsed:.1f}s timeout"
    )
    return False


def ensure_vector_index(collection: Collection | None = None) -> bool:
    """
    Create the vector search index if it doesn't exist, then poll until ready.

    This is idempotent — safe to call on every startup.

    Args:
        collection: Optional collection override. Uses singleton if None.

    Returns:
        True if the index is ready for queries.
    """
    if collection is None:
        collection = get_collection()

    index_name = rag_settings.vector_index_name

    # Check if already exists
    if _index_exists(collection, index_name):
        logger.info(f"Vector search index '{index_name}' already exists")
        # Still poll to confirm it's queryable
        return _poll_index_ready(
            collection,
            index_name,
            rag_settings.index_poll_interval_sec,
            timeout=10.0,  # short timeout for existing index
        )

    # Create the index
    logger.info(f"Creating vector search index '{index_name}'...")

    search_index_model = SearchIndexModel(
        definition=_build_index_definition(),
        name=index_name,
        type="vectorSearch",
    )

    try:
        result = collection.create_search_index(model=search_index_model)
        logger.info(f"Index creation initiated: {result}")
    except Exception as e:
        # Handle "index already exists" race condition
        if "already exists" in str(e).lower():
            logger.info(f"Index '{index_name}' was created concurrently")
        else:
            logger.error(f"Failed to create search index: {e}")
            raise

    # Poll until the index is ready
    return _poll_index_ready(
        collection,
        index_name,
        rag_settings.index_poll_interval_sec,
        rag_settings.index_poll_timeout_sec,
    )


def close():
    """Close the MongoDB connection."""
    global _client, _collection
    if _client is not None:
        _client.close()
        _client = None
        _collection = None
        logger.info("MongoDB connection closed")
