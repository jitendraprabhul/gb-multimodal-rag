"""
Qdrant vector database client.

Handles:
- Collection management
- Vector upsert and search
- Filtering and hybrid search
- Batch operations
"""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient as QdrantClientBase
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.exceptions import VectorDBError
from src.core.logging import LoggerMixin
from src.core.types import Chunk, Modality


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    id: str
    score: float
    payload: dict[str, Any]


class QdrantClient(LoggerMixin):
    """
    Async-compatible Qdrant client for vector storage.

    Manages multiple collections for different modalities
    and provides efficient search operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        collection_text: str = "text_chunks",
        collection_table: str = "table_chunks",
        collection_image: str = "image_chunks",
        prefer_grpc: bool = True,
        **config: Any,
    ) -> None:
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: HTTP port
            grpc_port: gRPC port
            collection_text: Text collection name
            collection_table: Table collection name
            collection_image: Image collection name
            prefer_grpc: Use gRPC if available
            **config: Additional configuration
        """
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.collection_text = collection_text
        self.collection_table = collection_table
        self.collection_image = collection_image
        self.prefer_grpc = prefer_grpc

        self._client: QdrantClientBase | None = None
        self._initialized = False
        self._dimensions: dict[str, int] = {}

    async def initialize(
        self,
        text_dimension: int = 768,
        image_dimension: int = 512,
    ) -> None:
        """
        Initialize client and create collections.

        Args:
            text_dimension: Text embedding dimension
            image_dimension: Image embedding dimension
        """
        if self._initialized:
            return

        try:
            self._client = QdrantClientBase(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
            )

            # Store dimensions
            self._dimensions = {
                self.collection_text: text_dimension,
                self.collection_table: text_dimension,  # Tables use text embeddings
                self.collection_image: image_dimension,
            }

            # Create collections
            await self._create_collection(self.collection_text, text_dimension)
            await self._create_collection(self.collection_table, text_dimension)
            await self._create_collection(self.collection_image, image_dimension)

            self._initialized = True
            self.logger.info(
                "Qdrant client initialized",
                host=self.host,
                port=self.port,
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to initialize Qdrant: {e}",
                cause=e,
            )

    async def _create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "Cosine",
    ) -> None:
        """Create a collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self._client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if name in existing_names:
                self.logger.debug(f"Collection {name} already exists")
                return

            # Create collection
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE
                    if distance == "Cosine"
                    else models.Distance.EUCLID,
                ),
            )

            # Create payload indexes
            self._client.create_payload_index(
                collection_name=name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=name,
                field_name="modality",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            self.logger.info(f"Created collection: {name}", dimension=dimension)

        except UnexpectedResponse as e:
            if "already exists" not in str(e):
                raise VectorDBError(
                    f"Failed to create collection {name}: {e}",
                    collection=name,
                    cause=e,
                )

    async def cleanup(self) -> None:
        """Close client connection."""
        if self._client:
            self._client.close()
            self._client = None
        self._initialized = False

    def _get_collection_for_modality(self, modality: Modality) -> str:
        """Get collection name for modality."""
        if modality == Modality.TEXT:
            return self.collection_text
        elif modality == Modality.TABLE:
            return self.collection_table
        elif modality == Modality.IMAGE:
            return self.collection_image
        return self.collection_text

    # =========================================================================
    # Upsert Operations
    # =========================================================================

    async def upsert_chunk(self, chunk: Chunk) -> str:
        """
        Upsert a single chunk.

        Args:
            chunk: Chunk with embedding

        Returns:
            Point ID
        """
        if not self._initialized:
            raise VectorDBError("Client not initialized")

        if chunk.embedding is None:
            raise VectorDBError(
                "Chunk has no embedding",
                details={"chunk_id": chunk.id},
            )

        collection = self._get_collection_for_modality(chunk.modality)

        try:
            point = models.PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.id,
                    "content": chunk.content[:1000],  # Truncate for storage
                    "modality": chunk.modality.value,
                    "doc_id": chunk.metadata.doc_id,
                    "page_number": chunk.metadata.page_number,
                    "section": chunk.metadata.section,
                    "table_id": chunk.metadata.table_id,
                    "image_id": chunk.metadata.image_id,
                    "source_file": chunk.metadata.source_file,
                    "entity_ids": chunk.entity_ids,
                },
            )

            self._client.upsert(
                collection_name=collection,
                points=[point],
            )

            return chunk.id

        except Exception as e:
            raise VectorDBError(
                f"Failed to upsert chunk: {e}",
                collection=collection,
                operation="upsert",
                cause=e,
            )

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
    ) -> list[str]:
        """
        Batch upsert multiple chunks.

        Args:
            chunks: Chunks with embeddings
            batch_size: Batch size for upsert

        Returns:
            List of point IDs
        """
        if not self._initialized:
            raise VectorDBError("Client not initialized")

        # Group by collection
        by_collection: dict[str, list[Chunk]] = {}

        for chunk in chunks:
            if chunk.embedding is None:
                continue

            collection = self._get_collection_for_modality(chunk.modality)
            if collection not in by_collection:
                by_collection[collection] = []
            by_collection[collection].append(chunk)

        ids = []

        for collection, collection_chunks in by_collection.items():
            # Process in batches
            for i in range(0, len(collection_chunks), batch_size):
                batch = collection_chunks[i : i + batch_size]

                points = [
                    models.PointStruct(
                        id=chunk.id,
                        vector=chunk.embedding,
                        payload={
                            "chunk_id": chunk.id,
                            "content": chunk.content[:1000],
                            "modality": chunk.modality.value,
                            "doc_id": chunk.metadata.doc_id,
                            "page_number": chunk.metadata.page_number,
                            "section": chunk.metadata.section,
                            "table_id": chunk.metadata.table_id,
                            "image_id": chunk.metadata.image_id,
                            "source_file": chunk.metadata.source_file,
                            "entity_ids": chunk.entity_ids,
                        },
                    )
                    for chunk in batch
                ]

                try:
                    self._client.upsert(
                        collection_name=collection,
                        points=points,
                    )
                    ids.extend([p.id for p in points])

                except Exception as e:
                    self.logger.error(
                        "Batch upsert failed",
                        collection=collection,
                        batch_size=len(batch),
                        error=str(e),
                    )

        return ids

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query_vector: np.ndarray | list[float],
        collection: str | None = None,
        modality: Modality | None = None,
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            collection: Collection to search (or infer from modality)
            modality: Content modality
            top_k: Number of results
            filter_conditions: Filter conditions
            score_threshold: Minimum score threshold

        Returns:
            List of search results
        """
        if not self._initialized:
            raise VectorDBError("Client not initialized")

        if collection is None:
            if modality:
                collection = self._get_collection_for_modality(modality)
            else:
                collection = self.collection_text

        # Build filter
        query_filter = None
        if filter_conditions:
            must_conditions = []

            for key, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )

            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

        try:
            # Convert to list if numpy array
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            results = self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )

            return [
                VectorSearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results
            ]

        except Exception as e:
            raise VectorDBError(
                f"Search failed: {e}",
                collection=collection,
                operation="search",
                cause=e,
            )

    async def search_all_collections(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search across all collections and merge results.

        Args:
            query_vector: Query embedding
            top_k: Number of results per collection
            filter_conditions: Filter conditions

        Returns:
            Merged and sorted search results
        """
        all_results = []

        for collection in [
            self.collection_text,
            self.collection_table,
            # self.collection_image,  # Skip if dimensions differ
        ]:
            try:
                results = await self.search(
                    query_vector=query_vector,
                    collection=collection,
                    top_k=top_k,
                    filter_conditions=filter_conditions,
                )
                all_results.extend(results)
            except Exception as e:
                self.logger.warning(
                    f"Search in {collection} failed: {e}"
                )

        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.score, reverse=True)

        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)

        return unique_results[:top_k]

    async def search_by_entity_ids(
        self,
        entity_ids: list[str],
        collection: str | None = None,
        top_k: int = 20,
    ) -> list[VectorSearchResult]:
        """
        Find chunks containing specific entities.

        Args:
            entity_ids: Entity IDs to search for
            collection: Collection to search
            top_k: Maximum results

        Returns:
            Chunks containing the entities
        """
        if not self._initialized:
            raise VectorDBError("Client not initialized")

        collection = collection or self.collection_text

        try:
            # Use scroll to find all matching points
            results = self._client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="entity_ids",
                            match=models.MatchAny(any=entity_ids),
                        )
                    ]
                ),
                limit=top_k,
            )

            points, _ = results

            return [
                VectorSearchResult(
                    id=str(p.id),
                    score=1.0,  # No score for filter-based search
                    payload=p.payload or {},
                )
                for p in points
            ]

        except Exception as e:
            raise VectorDBError(
                f"Entity search failed: {e}",
                collection=collection,
                operation="scroll",
                cause=e,
            )

    # =========================================================================
    # Collection Management
    # =========================================================================

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all vectors for a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of deleted points
        """
        total_deleted = 0

        for collection in [
            self.collection_text,
            self.collection_table,
            self.collection_image,
        ]:
            try:
                self._client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="doc_id",
                                    match=models.MatchValue(value=doc_id),
                                )
                            ]
                        )
                    ),
                )
                # Note: Qdrant doesn't return count, estimate from collection info
                total_deleted += 1

            except Exception as e:
                self.logger.warning(
                    f"Delete from {collection} failed: {e}"
                )

        return total_deleted

    async def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Get collection information."""
        try:
            info = self._client.get_collection(collection)
            return {
                "name": collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value if info.status else "unknown",
                "dimension": self._dimensions.get(collection),
            }
        except Exception as e:
            self.logger.warning(f"Failed to get collection info: {e}")
            return {}

    async def get_stats(self) -> dict[str, Any]:
        """Get stats for all collections."""
        stats = {}
        for collection in [
            self.collection_text,
            self.collection_table,
            self.collection_image,
        ]:
            stats[collection] = await self.get_collection_info(collection)
        return stats

    async def get_detailed_stats(self) -> dict[str, Any]:
        """Get detailed stats including point counts for all collections."""
        return await self.get_stats()

    async def __aenter__(self) -> "QdrantClient":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
