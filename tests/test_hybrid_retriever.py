"""
Tests for the hybrid retriever.

Covers:
- Full retrieval pipeline
- Entity-based retrieval
- Entity context fetching
- Initialization and context manager
- Error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.core.types import (
    Chunk,
    ChunkMetadata,
    Entity,
    EntityType,
    GraphPath,
    Modality,
    RetrievalResult,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vector.qdrant_client import VectorSearchResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_service():
    svc = MagicMock()
    svc.initialize = AsyncMock()
    svc.embed_query = AsyncMock(return_value=np.random.randn(768))
    svc.embed_chunks = AsyncMock(side_effect=lambda chunks: chunks)
    return svc


@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.search_all_collections = AsyncMock(return_value=[])
    client.search_by_entity_ids = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_neo4j():
    client = MagicMock()
    client.get_subgraph = AsyncMock(return_value=([], []))
    client.find_paths = AsyncMock(return_value=[])
    client.get_entity = AsyncMock(return_value=None)
    client.get_entity_neighborhood = AsyncMock(return_value=([], []))
    client.search_entities = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_ner():
    ner = MagicMock()
    ner.initialize = AsyncMock()
    ner.extract = AsyncMock(return_value=[])
    return ner


@pytest.fixture
def retriever(mock_embedding_service, mock_qdrant, mock_neo4j, mock_ner):
    return HybridRetriever(
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant,
        neo4j_client=mock_neo4j,
        ner_extractor=mock_ner,
        top_k_vector=10,
        top_k_final=3,
        graph_hops=2,
        max_graph_nodes=50,
    )


def _make_search_result(id, score=0.9, entity_ids=None, doc_id="doc-1"):
    return VectorSearchResult(
        id=id,
        score=score,
        payload={
            "content": f"Content of {id}",
            "modality": "text",
            "doc_id": doc_id,
            "page_number": 1,
            "section": "Summary",
            "entity_ids": entity_ids or [],
        },
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHybridRetrieverInit:
    def test_default_construction(self, retriever):
        assert retriever.top_k_vector == 10
        assert retriever.top_k_final == 3
        assert retriever.graph_hops == 2
        assert retriever._initialized is False

    async def test_initialize(self, retriever, mock_embedding_service, mock_ner):
        await retriever.initialize()

        assert retriever._initialized is True
        mock_embedding_service.initialize.assert_called_once()
        mock_ner.initialize.assert_called_once()

    async def test_initialize_idempotent(self, retriever, mock_embedding_service):
        await retriever.initialize()
        await retriever.initialize()

        mock_embedding_service.initialize.assert_called_once()

    async def test_initialize_without_ner(self, mock_embedding_service, mock_qdrant, mock_neo4j):
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant,
            neo4j_client=mock_neo4j,
            ner_extractor=None,
        )
        await retriever.initialize()
        assert retriever._initialized is True

    async def test_context_manager(self, mock_embedding_service, mock_qdrant, mock_neo4j):
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            qdrant_client=mock_qdrant,
            neo4j_client=mock_neo4j,
        )
        async with retriever as r:
            assert r._initialized is True


# =============================================================================
# Retrieve Pipeline Tests
# =============================================================================


class TestRetrieve:
    async def test_basic_retrieve(self, retriever, mock_qdrant, mock_embedding_service):
        mock_qdrant.search_all_collections.return_value = [
            _make_search_result("r1", 0.9),
            _make_search_result("r2", 0.8),
        ]

        result = await retriever.retrieve("What is the revenue?")

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) <= 3  # top_k_final
        assert result.retrieval_time_ms > 0
        mock_embedding_service.embed_query.assert_called_once()

    async def test_retrieve_extracts_query_entities(self, retriever, mock_ner):
        entity = Entity(
            id="e1",
            name="Apple",
            normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        mock_ner.extract.return_value = [entity]

        await retriever.retrieve("What is Apple's revenue?")

        mock_ner.extract.assert_called_once_with("What is Apple's revenue?")

    async def test_retrieve_with_doc_filter(self, retriever, mock_qdrant):
        mock_qdrant.search_all_collections.return_value = []

        await retriever.retrieve(
            "test query",
            filter_doc_ids=["doc-1", "doc-2"],
        )

        call_kwargs = mock_qdrant.search_all_collections.call_args.kwargs
        assert call_kwargs["filter_conditions"]["doc_id"] == ["doc-1", "doc-2"]

    async def test_retrieve_with_modality_filter(self, retriever, mock_qdrant):
        mock_qdrant.search_all_collections.return_value = []

        await retriever.retrieve(
            "test query",
            modalities=[Modality.TEXT, Modality.TABLE],
        )

        call_kwargs = mock_qdrant.search_all_collections.call_args.kwargs
        assert "text" in call_kwargs["filter_conditions"]["modality"]
        assert "table" in call_kwargs["filter_conditions"]["modality"]

    async def test_retrieve_builds_subgraph(self, retriever, mock_qdrant, mock_neo4j):
        mock_qdrant.search_all_collections.return_value = [
            _make_search_result("r1", 0.9, entity_ids=["e1", "e2"]),
        ]
        mock_neo4j.get_subgraph.return_value = (
            [{"id": "e1"}, {"id": "e2"}],
            [{"source": "e1", "target": "e2", "type": "ACQUIRED"}],
        )

        await retriever.retrieve("test query")

        mock_neo4j.get_subgraph.assert_called_once()

    async def test_retrieve_finds_paths(self, retriever, mock_qdrant, mock_neo4j):
        mock_qdrant.search_all_collections.return_value = [
            _make_search_result("r1", 0.9, entity_ids=["e1", "e2"]),
        ]
        mock_neo4j.get_subgraph.return_value = (
            [{"id": "e1"}, {"id": "e2"}],
            [{"source": "e1", "target": "e2"}],
        )

        await retriever.retrieve("test query")

        mock_neo4j.find_paths.assert_called()

    async def test_retrieve_handles_ner_failure(self, retriever, mock_ner, mock_qdrant):
        mock_ner.extract.side_effect = Exception("NER failed")
        mock_qdrant.search_all_collections.return_value = [
            _make_search_result("r1", 0.9),
        ]

        result = await retriever.retrieve("test query")

        # Should still return results despite NER failure
        assert isinstance(result, RetrievalResult)

    async def test_retrieve_limits_final_results(self, retriever, mock_qdrant):
        mock_qdrant.search_all_collections.return_value = [
            _make_search_result(f"r{i}", 0.9 - i * 0.1) for i in range(10)
        ]

        result = await retriever.retrieve("test query")
        assert len(result.chunks) <= 3  # top_k_final

    async def test_retrieve_converts_to_chunks(self, retriever, mock_qdrant):
        mock_qdrant.search_all_collections.return_value = [
            _make_search_result("r1", 0.9, doc_id="doc-42"),
        ]

        result = await retriever.retrieve("test query")

        if result.chunks:
            chunk = result.chunks[0]
            assert chunk.id == "r1"
            assert chunk.modality == Modality.TEXT
            assert chunk.metadata.doc_id == "doc-42"

    async def test_auto_initializes(self, retriever):
        retriever._initialized = False

        result = await retriever.retrieve("test query")
        assert retriever._initialized is True


# =============================================================================
# Entity-Based Retrieval Tests
# =============================================================================


class TestRetrieveByEntities:
    async def test_basic_entity_retrieval(self, retriever, mock_neo4j, mock_qdrant):
        entity = Entity(
            id="e1",
            name="Apple",
            normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        mock_neo4j.search_entities.return_value = [entity]
        mock_neo4j.get_subgraph.return_value = (
            [{"id": "e1"}, {"id": "e2"}],
            [],
        )
        mock_qdrant.search_by_entity_ids.return_value = [
            _make_search_result("r1", 1.0, entity_ids=["e1"]),
        ]

        result = await retriever.retrieve_by_entities(["Apple"])

        assert isinstance(result, RetrievalResult)
        assert len(result.entities) == 1

    async def test_no_entities_found(self, retriever, mock_neo4j):
        mock_neo4j.search_entities.return_value = []

        result = await retriever.retrieve_by_entities(["NonExistent"])

        assert result.chunks == []
        assert result.entities == []

    async def test_without_graph_expansion(self, retriever, mock_neo4j, mock_qdrant):
        entity = Entity(
            id="e1",
            name="Apple",
            normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        mock_neo4j.search_entities.return_value = [entity]
        mock_qdrant.search_by_entity_ids.return_value = []

        await retriever.retrieve_by_entities(["Apple"], expand_graph=False)

        mock_neo4j.get_subgraph.assert_not_called()


# =============================================================================
# Entity Context Tests
# =============================================================================


class TestGetEntityContext:
    async def test_basic_context(self, retriever, mock_neo4j, mock_qdrant):
        entity = Entity(
            id="e1",
            name="Apple",
            normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        mock_neo4j.get_entity.return_value = entity
        mock_neo4j.get_entity_neighborhood.return_value = (
            [{"id": "e2", "name": "Tim Cook"}],
            [{"source": "e1", "target": "e2", "type": "HAS_CEO"}],
        )
        mock_qdrant.search_by_entity_ids.return_value = [
            _make_search_result("r1", 1.0),
        ]

        context = await retriever.get_entity_context("e1")

        assert "entity" in context
        assert "neighbors" in context
        assert "relations" in context
        assert "chunks" in context

    async def test_entity_not_found(self, retriever, mock_neo4j):
        mock_neo4j.get_entity.return_value = None

        context = await retriever.get_entity_context("nonexistent")
        assert context == {}
