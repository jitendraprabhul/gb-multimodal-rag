"""
Tests for knowledge graph builder.

Covers document processing, NetworkX operations, entity caching, and graph analysis.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import networkx as nx

from src.core.types import (
    Chunk, ChunkMetadata, Document, DocumentType, Entity, EntityType,
    Modality, Relation, RelationType,
)
from src.kg.graph_builder import GraphBuilder


@pytest.fixture
def mock_neo4j():
    client = AsyncMock()
    client.initialize = AsyncMock()
    client.cleanup = AsyncMock()
    client.create_document_node = AsyncMock()
    client.create_chunk_node = AsyncMock()
    client.upsert_entity = AsyncMock(return_value="eid")
    client.link_entity_to_chunk = AsyncMock()
    client.upsert_relation = AsyncMock(return_value="rid")
    client.get_entity = AsyncMock(return_value=None)
    client.get_entity_neighborhood = AsyncMock(return_value=([], []))
    return client


@pytest.fixture
def builder(mock_neo4j):
    b = GraphBuilder(domain="finance")
    b.neo4j_client = mock_neo4j
    b.ner_extractor = AsyncMock()
    b.ner_extractor.initialize = AsyncMock()
    b.ner_extractor.cleanup = AsyncMock()
    b.ner_extractor.extract_from_chunk = AsyncMock(return_value=[])
    b.pattern_extractor = AsyncMock()
    b.pattern_extractor.initialize = AsyncMock()
    b.pattern_extractor.cleanup = AsyncMock()
    b.pattern_extractor.extract_from_chunk = AsyncMock(return_value=[])
    return b


class TestGraphBuilderInit:
    def test_finance_domain(self):
        with patch("src.kg.graph_builder.Neo4jClient"):
            builder = GraphBuilder(domain="finance")
            assert builder.domain == "finance"

    def test_healthcare_domain(self):
        with patch("src.kg.graph_builder.Neo4jClient"):
            builder = GraphBuilder(domain="healthcare")
            assert builder.domain == "healthcare"

    def test_llm_relations_disabled_by_default(self):
        with patch("src.kg.graph_builder.Neo4jClient"):
            builder = GraphBuilder()
            assert builder.llm_extractor is None


class TestGraphBuilderLifecycle:
    @pytest.mark.asyncio
    async def test_initialize(self, builder):
        await builder.initialize()
        assert builder._initialized is True

    @pytest.mark.asyncio
    async def test_double_initialize(self, builder):
        await builder.initialize()
        await builder.initialize()  # Should not fail
        assert builder._initialized is True

    @pytest.mark.asyncio
    async def test_cleanup(self, builder):
        await builder.initialize()
        await builder.cleanup()
        assert builder._initialized is False


class TestProcessDocument:
    @pytest.mark.asyncio
    async def test_process_empty_document(self, builder, mock_neo4j):
        await builder.initialize()
        doc = Document(filename="test.txt", doc_type=DocumentType.TXT, chunks=[])
        entities, relations = await builder.process_document(doc)
        assert entities == []
        assert relations == []
        mock_neo4j.create_document_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_with_chunks(self, builder, mock_neo4j):
        await builder.initialize()

        entity = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY, source_chunk_ids=[],
        )
        builder.ner_extractor.extract_from_chunk = AsyncMock(return_value=[entity])

        chunk = Chunk(
            content="Apple reported revenue.", modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
        )
        doc = Document(
            filename="test.txt", doc_type=DocumentType.TXT, chunks=[chunk],
        )
        entities, relations = await builder.process_document(doc)
        assert len(entities) == 1
        assert entities[0].name == "Apple"
        mock_neo4j.create_chunk_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_deduplication(self, builder):
        await builder.initialize()

        e1 = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY, confidence=0.8,
        )
        e2 = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY, confidence=0.9,
        )

        # First chunk returns e1
        # Second chunk returns e2 (same entity)
        builder.ner_extractor.extract_from_chunk = AsyncMock(
            side_effect=[[e1], [e2]]
        )

        chunks = [
            Chunk(content="Apple is great.", modality=Modality.TEXT,
                  metadata=ChunkMetadata(doc_id="doc-1")),
            Chunk(content="Apple reported.", modality=Modality.TEXT,
                  metadata=ChunkMetadata(doc_id="doc-1")),
        ]
        doc = Document(filename="test.txt", doc_type=DocumentType.TXT, chunks=chunks)
        entities, _ = await builder.process_document(doc)
        # Should be deduplicated to 1 entity
        assert len(entities) == 1
        assert entities[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_process_multiple_documents(self, builder):
        await builder.initialize()
        docs = [
            Document(filename="a.txt", doc_type=DocumentType.TXT, chunks=[]),
            Document(filename="b.txt", doc_type=DocumentType.TXT, chunks=[]),
        ]
        entities, relations = await builder.process_documents(docs)
        assert isinstance(entities, list)
        assert isinstance(relations, list)


class TestNetworkXOperations:
    def test_empty_graph_stats(self, builder):
        stats = builder.get_graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_update_networkx_graph(self, builder):
        entities = [
            Entity(id="e1", name="Apple", normalized_name="apple", entity_type=EntityType.COMPANY),
            Entity(id="e2", name="Google", normalized_name="google", entity_type=EntityType.COMPANY),
        ]
        relations = [
            Relation(
                source_entity_id="e1", target_entity_id="e2",
                relation_type=RelationType.COMPETITOR_OF,
            ),
        ]
        builder._update_networkx_graph(entities, relations)
        stats = builder.get_graph_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1

    def test_get_neighbors(self, builder):
        entities = [
            Entity(id="e1", name="A", normalized_name="a", entity_type=EntityType.COMPANY),
            Entity(id="e2", name="B", normalized_name="b", entity_type=EntityType.COMPANY),
        ]
        relations = [
            Relation(
                source_entity_id="e1", target_entity_id="e2",
                relation_type=RelationType.ACQUIRED,
            ),
        ]
        builder._update_networkx_graph(entities, relations)
        neighbors = builder.get_neighbors("e1")
        assert "e2" in neighbors

    def test_get_neighbors_nonexistent_node(self, builder):
        assert builder.get_neighbors("nonexistent") == []

    def test_get_neighbors_with_type_filter(self, builder):
        entities = [
            Entity(id="e1", name="A", normalized_name="a", entity_type=EntityType.COMPANY),
            Entity(id="e2", name="B", normalized_name="b", entity_type=EntityType.COMPANY),
            Entity(id="e3", name="C", normalized_name="c", entity_type=EntityType.COMPANY),
        ]
        relations = [
            Relation(source_entity_id="e1", target_entity_id="e2", relation_type=RelationType.ACQUIRED),
            Relation(source_entity_id="e1", target_entity_id="e3", relation_type=RelationType.MENTIONS),
        ]
        builder._update_networkx_graph(entities, relations)
        neighbors = builder.get_neighbors("e1", relation_types=["ACQUIRED"])
        assert "e2" in neighbors
        assert "e3" not in neighbors

    def test_find_shortest_path(self, builder):
        entities = [
            Entity(id="e1", name="A", normalized_name="a", entity_type=EntityType.COMPANY),
            Entity(id="e2", name="B", normalized_name="b", entity_type=EntityType.COMPANY),
            Entity(id="e3", name="C", normalized_name="c", entity_type=EntityType.COMPANY),
        ]
        relations = [
            Relation(source_entity_id="e1", target_entity_id="e2", relation_type=RelationType.ACQUIRED),
            Relation(source_entity_id="e2", target_entity_id="e3", relation_type=RelationType.ACQUIRED),
        ]
        builder._update_networkx_graph(entities, relations)
        path = builder.find_shortest_path("e1", "e3")
        assert path is not None
        assert path[0] == "e1"
        assert path[-1] == "e3"

    def test_find_shortest_path_no_path(self, builder):
        entities = [
            Entity(id="e1", name="A", normalized_name="a", entity_type=EntityType.COMPANY),
            Entity(id="e2", name="B", normalized_name="b", entity_type=EntityType.COMPANY),
        ]
        builder._update_networkx_graph(entities, [])
        path = builder.find_shortest_path("e1", "e2")
        assert path is None

    def test_find_shortest_path_nonexistent_node(self, builder):
        path = builder.find_shortest_path("x", "y")
        assert path is None

    def test_get_subgraph(self, builder):
        entities = [
            Entity(id=f"e{i}", name=f"N{i}", normalized_name=f"n{i}", entity_type=EntityType.COMPANY)
            for i in range(5)
        ]
        relations = [
            Relation(source_entity_id="e0", target_entity_id="e1", relation_type=RelationType.ACQUIRED),
            Relation(source_entity_id="e1", target_entity_id="e2", relation_type=RelationType.ACQUIRED),
            Relation(source_entity_id="e2", target_entity_id="e3", relation_type=RelationType.ACQUIRED),
            Relation(source_entity_id="e3", target_entity_id="e4", relation_type=RelationType.ACQUIRED),
        ]
        builder._update_networkx_graph(entities, relations)
        subgraph = builder.get_subgraph(["e0"], hops=1)
        assert "e0" in subgraph.nodes
        assert "e1" in subgraph.nodes

    def test_get_entity_by_name(self, builder):
        entity = Entity(id="e1", name="Apple Inc.", normalized_name="apple inc.", entity_type=EntityType.COMPANY)
        builder._entity_cache["apple inc.:COMPANY"] = entity
        result = builder.get_entity_by_name("Apple Inc.")
        assert result is entity

    def test_get_entity_by_name_not_found(self, builder):
        assert builder.get_entity_by_name("Nonexistent") is None

    def test_export_to_gexf(self, builder, tmp_path):
        entities = [Entity(id="e1", name="A", normalized_name="a", entity_type=EntityType.COMPANY)]
        builder._update_networkx_graph(entities, [])
        path = str(tmp_path / "graph.gexf")
        builder.export_to_gexf(path)
        assert (tmp_path / "graph.gexf").exists()
