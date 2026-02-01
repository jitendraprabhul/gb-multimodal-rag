"""
Tests for the Qdrant vector database client.

Covers:
- Initialization and connection
- Collection creation and management
- Upsert operations (single and batch)
- Search operations (single collection, all collections, by entity)
- Document deletion
- Stats and collection info
- Error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass

import numpy as np
import pytest

from src.core.exceptions import VectorDBError
from src.core.types import Chunk, ChunkMetadata, Modality
from src.vector.qdrant_client import QdrantClient, VectorSearchResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def qdrant():
    """Create QdrantClient with default params (not initialized)."""
    return QdrantClient(
        host="localhost",
        port=6333,
        grpc_port=6334,
        collection_text="text_chunks",
        collection_table="table_chunks",
        collection_image="image_chunks",
    )


@pytest.fixture
def mock_qdrant_base():
    """Create a mock QdrantClientBase."""
    mock = MagicMock()
    # get_collections returns CollectionDescription list
    collection_desc = MagicMock()
    collection_desc.collections = []
    mock.get_collections.return_value = collection_desc
    mock.create_collection.return_value = None
    mock.create_payload_index.return_value = None
    return mock


@pytest.fixture
async def initialized_qdrant(qdrant, mock_qdrant_base):
    """Create an initialized QdrantClient with mocked underlying client."""
    with patch("src.vector.qdrant_client.QdrantClientBase", return_value=mock_qdrant_base):
        await qdrant.initialize()
    return qdrant


def _make_chunk(
    content="Test content",
    modality=Modality.TEXT,
    doc_id="doc-1",
    embedding=None,
    entity_ids=None,
    chunk_id=None,
):
    """Helper to create a Chunk."""
    chunk = Chunk(
        content=content,
        modality=modality,
        metadata=ChunkMetadata(doc_id=doc_id),
        embedding=embedding or [0.1] * 768,
        entity_ids=entity_ids or [],
    )
    if chunk_id:
        chunk.id = chunk_id
    return chunk


# =============================================================================
# VectorSearchResult Tests
# =============================================================================


class TestVectorSearchResult:
    def test_creation(self):
        result = VectorSearchResult(id="r1", score=0.95, payload={"content": "test"})
        assert result.id == "r1"
        assert result.score == 0.95
        assert result.payload["content"] == "test"

    def test_empty_payload(self):
        result = VectorSearchResult(id="r1", score=0.5, payload={})
        assert result.payload == {}


# =============================================================================
# Initialization Tests
# =============================================================================


class TestQdrantClientInit:
    def test_default_params(self):
        client = QdrantClient()
        assert client.host == "localhost"
        assert client.port == 6333
        assert client.grpc_port == 6334
        assert client.collection_text == "text_chunks"
        assert client.collection_table == "table_chunks"
        assert client.collection_image == "image_chunks"
        assert client.prefer_grpc is True
        assert client._initialized is False

    def test_custom_params(self):
        client = QdrantClient(
            host="qdrant-server",
            port=7333,
            grpc_port=7334,
            collection_text="my_text",
            collection_table="my_table",
            collection_image="my_image",
            prefer_grpc=False,
        )
        assert client.host == "qdrant-server"
        assert client.port == 7333
        assert client.collection_text == "my_text"

    async def test_initialize_creates_collections(self, qdrant, mock_qdrant_base):
        with patch("src.vector.qdrant_client.QdrantClientBase", return_value=mock_qdrant_base):
            await qdrant.initialize()

        assert qdrant._initialized is True
        # 3 collections created
        assert mock_qdrant_base.create_collection.call_count == 3
        # 2 payload indexes per collection (doc_id + modality)
        assert mock_qdrant_base.create_payload_index.call_count == 6

    async def test_initialize_skips_existing_collections(self, qdrant, mock_qdrant_base):
        # Simulate all collections existing
        existing = [MagicMock(name=n) for n in ["text_chunks", "table_chunks", "image_chunks"]]
        for coll, name in zip(existing, ["text_chunks", "table_chunks", "image_chunks"]):
            coll.name = name
        mock_qdrant_base.get_collections.return_value.collections = existing

        with patch("src.vector.qdrant_client.QdrantClientBase", return_value=mock_qdrant_base):
            await qdrant.initialize()

        assert qdrant._initialized is True
        assert mock_qdrant_base.create_collection.call_count == 0

    async def test_initialize_idempotent(self, qdrant, mock_qdrant_base):
        with patch("src.vector.qdrant_client.QdrantClientBase", return_value=mock_qdrant_base):
            await qdrant.initialize()
            await qdrant.initialize()  # Second call should be no-op

        # QdrantClientBase constructor should be called only once
        assert qdrant._initialized is True

    async def test_initialize_failure_raises_vector_db_error(self, qdrant):
        with patch(
            "src.vector.qdrant_client.QdrantClientBase",
            side_effect=ConnectionError("Connection refused"),
        ):
            with pytest.raises(VectorDBError, match="Failed to initialize Qdrant"):
                await qdrant.initialize()

    async def test_cleanup(self, initialized_qdrant):
        await initialized_qdrant.cleanup()
        assert initialized_qdrant._initialized is False
        assert initialized_qdrant._client is None

    async def test_cleanup_when_not_initialized(self, qdrant):
        await qdrant.cleanup()  # Should not raise
        assert qdrant._initialized is False

    async def test_context_manager(self, mock_qdrant_base):
        with patch("src.vector.qdrant_client.QdrantClientBase", return_value=mock_qdrant_base):
            async with QdrantClient() as client:
                assert client._initialized is True
            assert client._initialized is False


# =============================================================================
# Modality Mapping Tests
# =============================================================================


class TestModalityMapping:
    def test_text_modality(self, qdrant):
        assert qdrant._get_collection_for_modality(Modality.TEXT) == "text_chunks"

    def test_table_modality(self, qdrant):
        assert qdrant._get_collection_for_modality(Modality.TABLE) == "table_chunks"

    def test_image_modality(self, qdrant):
        assert qdrant._get_collection_for_modality(Modality.IMAGE) == "image_chunks"


# =============================================================================
# Upsert Tests
# =============================================================================


class TestUpsertChunk:
    async def test_upsert_single_text_chunk(self, initialized_qdrant):
        chunk = _make_chunk()
        result_id = await initialized_qdrant.upsert_chunk(chunk)

        assert result_id == chunk.id
        initialized_qdrant._client.upsert.assert_called_once()
        call_kwargs = initialized_qdrant._client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "text_chunks"

    async def test_upsert_table_chunk(self, initialized_qdrant):
        chunk = _make_chunk(modality=Modality.TABLE)
        await initialized_qdrant.upsert_chunk(chunk)

        call_kwargs = initialized_qdrant._client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "table_chunks"

    async def test_upsert_image_chunk(self, initialized_qdrant):
        chunk = _make_chunk(modality=Modality.IMAGE, embedding=[0.1] * 512)
        await initialized_qdrant.upsert_chunk(chunk)

        call_kwargs = initialized_qdrant._client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "image_chunks"

    async def test_upsert_without_init_raises(self, qdrant):
        chunk = _make_chunk()
        with pytest.raises(VectorDBError, match="Client not initialized"):
            await qdrant.upsert_chunk(chunk)

    async def test_upsert_without_embedding_raises(self, initialized_qdrant):
        chunk = _make_chunk()
        chunk.embedding = None
        with pytest.raises(VectorDBError, match="Chunk has no embedding"):
            await initialized_qdrant.upsert_chunk(chunk)

    async def test_upsert_payload_contents(self, initialized_qdrant):
        chunk = _make_chunk(
            content="Financial report summary",
            doc_id="doc-42",
            entity_ids=["e1", "e2"],
        )
        chunk.metadata.page_number = 3
        chunk.metadata.section = "Summary"

        await initialized_qdrant.upsert_chunk(chunk)

        call_args = initialized_qdrant._client.upsert.call_args
        points = call_args.kwargs["points"]
        payload = points[0].payload

        assert payload["chunk_id"] == chunk.id
        assert payload["content"] == "Financial report summary"
        assert payload["modality"] == "text"
        assert payload["doc_id"] == "doc-42"
        assert payload["page_number"] == 3
        assert payload["section"] == "Summary"
        assert payload["entity_ids"] == ["e1", "e2"]

    async def test_upsert_truncates_content(self, initialized_qdrant):
        long_content = "x" * 2000
        chunk = _make_chunk(content=long_content)
        await initialized_qdrant.upsert_chunk(chunk)

        call_args = initialized_qdrant._client.upsert.call_args
        points = call_args.kwargs["points"]
        assert len(points[0].payload["content"]) == 1000

    async def test_upsert_failure_raises_vector_db_error(self, initialized_qdrant):
        initialized_qdrant._client.upsert.side_effect = Exception("Upsert failed")
        chunk = _make_chunk()
        with pytest.raises(VectorDBError, match="Failed to upsert chunk"):
            await initialized_qdrant.upsert_chunk(chunk)


class TestUpsertChunks:
    async def test_batch_upsert(self, initialized_qdrant):
        chunks = [_make_chunk(content=f"Chunk {i}") for i in range(5)]
        ids = await initialized_qdrant.upsert_chunks(chunks)

        assert len(ids) == 5
        assert initialized_qdrant._client.upsert.call_count == 1  # All in one batch

    async def test_batch_upsert_groups_by_collection(self, initialized_qdrant):
        chunks = [
            _make_chunk(content="Text chunk", modality=Modality.TEXT),
            _make_chunk(content="Table chunk", modality=Modality.TABLE),
            _make_chunk(content="Image chunk", modality=Modality.IMAGE, embedding=[0.1] * 512),
        ]
        ids = await initialized_qdrant.upsert_chunks(chunks)

        assert len(ids) == 3
        assert initialized_qdrant._client.upsert.call_count == 3  # One per collection

    async def test_batch_upsert_skips_no_embedding(self, initialized_qdrant):
        chunks = [
            _make_chunk(content="With embedding"),
            _make_chunk(content="Without embedding"),
        ]
        chunks[1].embedding = None

        ids = await initialized_qdrant.upsert_chunks(chunks)
        assert len(ids) == 1

    async def test_batch_upsert_respects_batch_size(self, initialized_qdrant):
        chunks = [_make_chunk(content=f"Chunk {i}") for i in range(5)]
        await initialized_qdrant.upsert_chunks(chunks, batch_size=2)

        # 5 chunks / batch_size 2 = 3 calls (2+2+1)
        assert initialized_qdrant._client.upsert.call_count == 3

    async def test_batch_upsert_not_initialized_raises(self, qdrant):
        chunks = [_make_chunk()]
        with pytest.raises(VectorDBError, match="Client not initialized"):
            await qdrant.upsert_chunks(chunks)


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    async def test_basic_search(self, initialized_qdrant):
        mock_result = MagicMock()
        mock_result.id = "point-1"
        mock_result.score = 0.95
        mock_result.payload = {"content": "test", "doc_id": "doc-1"}
        initialized_qdrant._client.search.return_value = [mock_result]

        results = await initialized_qdrant.search(
            query_vector=[0.1] * 768,
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].id == "point-1"
        assert results[0].score == 0.95
        assert results[0].payload["content"] == "test"

    async def test_search_with_numpy_vector(self, initialized_qdrant):
        initialized_qdrant._client.search.return_value = []

        query = np.random.randn(768).astype(np.float32)
        await initialized_qdrant.search(query_vector=query)

        call_args = initialized_qdrant._client.search.call_args
        assert isinstance(call_args.kwargs["query_vector"], list)

    async def test_search_with_modality(self, initialized_qdrant):
        initialized_qdrant._client.search.return_value = []

        await initialized_qdrant.search(
            query_vector=[0.1] * 768,
            modality=Modality.TABLE,
        )

        call_args = initialized_qdrant._client.search.call_args
        assert call_args.kwargs["collection_name"] == "table_chunks"

    async def test_search_defaults_to_text_collection(self, initialized_qdrant):
        initialized_qdrant._client.search.return_value = []

        await initialized_qdrant.search(query_vector=[0.1] * 768)

        call_args = initialized_qdrant._client.search.call_args
        assert call_args.kwargs["collection_name"] == "text_chunks"

    async def test_search_with_filter_conditions(self, initialized_qdrant):
        initialized_qdrant._client.search.return_value = []

        await initialized_qdrant.search(
            query_vector=[0.1] * 768,
            filter_conditions={"doc_id": "doc-1"},
        )

        call_args = initialized_qdrant._client.search.call_args
        assert call_args.kwargs["query_filter"] is not None

    async def test_search_with_list_filter(self, initialized_qdrant):
        initialized_qdrant._client.search.return_value = []

        await initialized_qdrant.search(
            query_vector=[0.1] * 768,
            filter_conditions={"doc_id": ["doc-1", "doc-2"]},
        )

        call_args = initialized_qdrant._client.search.call_args
        assert call_args.kwargs["query_filter"] is not None

    async def test_search_not_initialized_raises(self, qdrant):
        with pytest.raises(VectorDBError, match="Client not initialized"):
            await qdrant.search(query_vector=[0.1] * 768)

    async def test_search_failure_raises_vector_db_error(self, initialized_qdrant):
        initialized_qdrant._client.search.side_effect = Exception("Search failed")

        with pytest.raises(VectorDBError, match="Search failed"):
            await initialized_qdrant.search(query_vector=[0.1] * 768)

    async def test_search_null_payload(self, initialized_qdrant):
        mock_result = MagicMock()
        mock_result.id = "point-1"
        mock_result.score = 0.5
        mock_result.payload = None
        initialized_qdrant._client.search.return_value = [mock_result]

        results = await initialized_qdrant.search(query_vector=[0.1] * 768)
        assert results[0].payload == {}


class TestSearchAllCollections:
    async def test_merges_results(self, initialized_qdrant):
        mock_r1 = MagicMock()
        mock_r1.id = "r1"
        mock_r1.score = 0.9
        mock_r1.payload = {"content": "text"}

        mock_r2 = MagicMock()
        mock_r2.id = "r2"
        mock_r2.score = 0.8
        mock_r2.payload = {"content": "table"}

        # First call (text collection) returns r1, second (table) returns r2
        initialized_qdrant._client.search.side_effect = [[mock_r1], [mock_r2]]

        results = await initialized_qdrant.search_all_collections(
            query_vector=[0.1] * 768,
            top_k=10,
        )

        assert len(results) == 2
        # Sorted by score descending
        assert results[0].score == 0.9
        assert results[1].score == 0.8

    async def test_deduplicates_results(self, initialized_qdrant):
        mock_r1 = MagicMock()
        mock_r1.id = "same-id"
        mock_r1.score = 0.9
        mock_r1.payload = {"content": "text"}

        mock_r2 = MagicMock()
        mock_r2.id = "same-id"
        mock_r2.score = 0.8
        mock_r2.payload = {"content": "text"}

        initialized_qdrant._client.search.side_effect = [[mock_r1], [mock_r2]]

        results = await initialized_qdrant.search_all_collections(
            query_vector=[0.1] * 768,
        )

        assert len(results) == 1

    async def test_respects_top_k(self, initialized_qdrant):
        mock_results = []
        for i in range(5):
            r = MagicMock()
            r.id = f"r{i}"
            r.score = 0.9 - i * 0.1
            r.payload = {}
            mock_results.append(r)

        initialized_qdrant._client.search.side_effect = [mock_results, []]

        results = await initialized_qdrant.search_all_collections(
            query_vector=[0.1] * 768,
            top_k=3,
        )

        assert len(results) == 3

    async def test_handles_collection_failure_gracefully(self, initialized_qdrant):
        mock_r = MagicMock()
        mock_r.id = "r1"
        mock_r.score = 0.9
        mock_r.payload = {}

        # First collection succeeds, second fails
        initialized_qdrant._client.search.side_effect = [[mock_r], Exception("Failed")]

        results = await initialized_qdrant.search_all_collections(
            query_vector=[0.1] * 768,
        )

        assert len(results) == 1


class TestSearchByEntityIds:
    async def test_basic_entity_search(self, initialized_qdrant):
        mock_point = MagicMock()
        mock_point.id = "p1"
        mock_point.payload = {"content": "test", "entity_ids": ["e1"]}
        initialized_qdrant._client.scroll.return_value = ([mock_point], None)

        results = await initialized_qdrant.search_by_entity_ids(
            entity_ids=["e1"],
        )

        assert len(results) == 1
        assert results[0].id == "p1"
        assert results[0].score == 1.0  # Fixed score for filter-based search

    async def test_entity_search_not_initialized_raises(self, qdrant):
        with pytest.raises(VectorDBError, match="Client not initialized"):
            await qdrant.search_by_entity_ids(entity_ids=["e1"])

    async def test_entity_search_failure_raises(self, initialized_qdrant):
        initialized_qdrant._client.scroll.side_effect = Exception("Scroll failed")
        with pytest.raises(VectorDBError, match="Entity search failed"):
            await initialized_qdrant.search_by_entity_ids(entity_ids=["e1"])


# =============================================================================
# Collection Management Tests
# =============================================================================


class TestDeleteByDocId:
    async def test_deletes_from_all_collections(self, initialized_qdrant):
        result = await initialized_qdrant.delete_by_doc_id("doc-1")

        assert initialized_qdrant._client.delete.call_count == 3
        assert result == 3  # One "deleted" per collection

    async def test_handles_delete_failure_gracefully(self, initialized_qdrant):
        initialized_qdrant._client.delete.side_effect = [
            None,
            Exception("Failed"),
            None,
        ]

        result = await initialized_qdrant.delete_by_doc_id("doc-1")
        # Two successful + one failed
        assert result == 2


class TestCollectionInfo:
    async def test_get_collection_info(self, initialized_qdrant):
        mock_info = MagicMock()
        mock_info.points_count = 100
        mock_info.vectors_count = 100
        mock_info.status = MagicMock(value="green")
        initialized_qdrant._client.get_collection.return_value = mock_info

        info = await initialized_qdrant.get_collection_info("text_chunks")

        assert info["name"] == "text_chunks"
        assert info["points_count"] == 100
        assert info["status"] == "green"

    async def test_get_collection_info_failure_returns_empty(self, initialized_qdrant):
        initialized_qdrant._client.get_collection.side_effect = Exception("Not found")
        info = await initialized_qdrant.get_collection_info("nonexistent")
        assert info == {}

    async def test_get_stats(self, initialized_qdrant):
        mock_info = MagicMock()
        mock_info.points_count = 50
        mock_info.vectors_count = 50
        mock_info.status = MagicMock(value="green")
        initialized_qdrant._client.get_collection.return_value = mock_info

        stats = await initialized_qdrant.get_stats()

        assert "text_chunks" in stats
        assert "table_chunks" in stats
        assert "image_chunks" in stats
