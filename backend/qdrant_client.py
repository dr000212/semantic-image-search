from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.config import Config
from backend.logger import GLOBAL_LOGGER as log
from backend.exception.custom_exception import SemanticImageSearchException


class QdrantClientManager:
    """
    Qdrant Client Manager (Singleton)
    """

    _client = None

    @classmethod
    def _extract_default_vector_params(cls, vectors_config):
        """
        Normalize Qdrant vectors config to return params for named vector 'default'.
        Returns None when the collection does not use named vectors or has no 'default'.
        """
        if vectors_config is None:
            return None

        # Single-vector collection (unnamed): incompatible with our named-vector usage.
        if isinstance(vectors_config, models.VectorParams):
            return None

        # Named vectors config in various possible SDK representations.
        if isinstance(vectors_config, dict):
            return vectors_config.get("default")

        if hasattr(vectors_config, "get"):
            return vectors_config.get("default")

        if hasattr(vectors_config, "vectors"):
            inner = getattr(vectors_config, "vectors")
            if isinstance(inner, dict):
                return inner.get("default")
            if hasattr(inner, "get"):
                return inner.get("default")

        return None

    @classmethod
    def _validate_collection_schema(cls, client):
        info = client.get_collection(collection_name=Config.QDRANT_COLLECTION)
        vectors_cfg = info.config.params.vectors
        default_params = cls._extract_default_vector_params(vectors_cfg)

        if default_params is None:
            msg = (
                "Existing collection schema mismatch: expected named vector 'default'. "
                "Please reset/recreate collection and reindex."
            )
            log.error(msg, collection=Config.QDRANT_COLLECTION)
            raise SemanticImageSearchException(msg)

        actual_size = getattr(default_params, "size", None)
        actual_distance = getattr(default_params, "distance", None)

        if actual_size != Config.VECTOR_SIZE:
            msg = (
                f"Collection vector size mismatch: expected {Config.VECTOR_SIZE}, got {actual_size}. "
                "Please reset/recreate collection and reindex."
            )
            log.error(
                "Collection vector size mismatch",
                collection=Config.QDRANT_COLLECTION,
                expected=Config.VECTOR_SIZE,
                actual=actual_size,
            )
            raise SemanticImageSearchException(msg)

        if actual_distance != models.Distance.COSINE:
            msg = (
                f"Collection distance mismatch: expected {models.Distance.COSINE}, got {actual_distance}. "
                "Please reset/recreate collection and reindex."
            )
            log.error(
                "Collection distance mismatch",
                collection=Config.QDRANT_COLLECTION,
                expected=str(models.Distance.COSINE),
                actual=str(actual_distance),
            )
            raise SemanticImageSearchException(msg)

    @classmethod
    def _create_collection(cls, client):
        client.create_collection(
            collection_name=Config.QDRANT_COLLECTION,
            vectors_config={
                "default": models.VectorParams(
                    size=Config.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                    on_disk=True,   # important for large datasets
                )
            },
        )

    @classmethod
    def _recreate_collection(cls, client):
        log.warning(
            "Recreating Qdrant collection due to schema mismatch",
            collection=Config.QDRANT_COLLECTION,
        )
        client.delete_collection(collection_name=Config.QDRANT_COLLECTION)
        cls._create_collection(client)
        log.info(
            "Qdrant collection recreated successfully",
            collection=Config.QDRANT_COLLECTION,
            vector_size=Config.VECTOR_SIZE,
            distance="COSINE",
        )

    @classmethod
    def get_client(cls) -> QdrantClient:
        """Lazy initialize the Qdrant client"""
        if cls._client is None:

            if not Config.QDRANT_URL:
                log.warning("QDRANT_URL missing in environment")

            if not Config.QDRANT_API_KEY:
                log.warning("QDRANT_API_KEY missing in environment")

            log.info(
                "Initializing Qdrant client",
                url=Config.QDRANT_URL,
                using_api_key=bool(Config.QDRANT_API_KEY)
            )

            try:
                cls._client = QdrantClient(
                    url=Config.QDRANT_URL,
                    api_key=Config.QDRANT_API_KEY,
                )
                log.info("Qdrant client initialized successfully")

            except Exception as e:
                log.error("Failed to initialize Qdrant client", error=str(e))
                raise SemanticImageSearchException("Failed to init Qdrant client", e)

        return cls._client

    @classmethod
    def ensure_collection(cls):
        """Ensure Qdrant collection exists"""

        try:
            client = cls.get_client()

            log.info("Fetching existing Qdrant collections...")

            all_collections = client.get_collections().collections
            existing = {c.name for c in all_collections}

            if Config.QDRANT_COLLECTION not in existing:

                log.info(
                    "Creating new Qdrant collection",
                    collection=Config.QDRANT_COLLECTION,
                    vector_size=Config.VECTOR_SIZE,
                    distance="COSINE",
                )

                cls._create_collection(client)

                log.info("Qdrant collection created", collection=Config.QDRANT_COLLECTION)

            else:
                try:
                    cls._validate_collection_schema(client)
                except SemanticImageSearchException:
                    cls._recreate_collection(client)
                log.info(
                    "Using existing Qdrant collection",
                    collection=Config.QDRANT_COLLECTION,
                )

        except Exception as e:
            log.error("Failed to ensure Qdrant collection", error=str(e))
            raise SemanticImageSearchException("Failed to ensure Qdrant collection", e)
