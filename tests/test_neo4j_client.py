"""
Tests for Neo4j client.

Uses mocks for all Neo4j operations since we don't have a live database.
"""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.core.exceptions import GraphError
from src.core.types import Entity, EntityType, GraphPath, Relation, RelationType
from src.kg.neo4j_client import Neo4jClient


@pytest.fixture
def mock_driver():
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    session = AsyncMock()
    session.run = AsyncMock()
    session.close = AsyncMock()
    driver.session = MagicMock(return_value=session)
    return driver, session


@pytest.fixture
def client():
    return Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test",
        database="neo4j",
    )


class TestNeo4jClientInit:
    def test_defaults(self):
        client = Neo4jClient()
        assert client.uri == "bolt://localhost:7687"
        assert client.user == "neo4j"
        assert client.database == "neo4j"
        assert client._initialized is False

    def test_custom_params(self):
        client = Neo4jClient(
            uri="bolt://custom:7687",
            user="admin",
            password="secret",
            database="mydb",
            max_connection_pool_size=100,
        )
        assert client.uri == "bolt://custom:7687"
        assert client.user == "admin"
        assert client.database == "mydb"
        assert client.max_connection_pool_size == 100


class TestNeo4jClientLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_success(self, client, mock_driver):
        driver, session = mock_driver
        result_mock = AsyncMock()
        session.run = AsyncMock(return_value=result_mock)

        with patch("src.kg.neo4j_client.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)
            await client.initialize()
            assert client._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self, client):
        with patch("src.kg.neo4j_client.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(side_effect=Exception("Connection refused"))
            with pytest.raises(GraphError, match="Failed to connect"):
                await client.initialize()

    @pytest.mark.asyncio
    async def test_double_initialize(self, client, mock_driver):
        driver, session = mock_driver
        session.run = AsyncMock(return_value=AsyncMock())

        with patch("src.kg.neo4j_client.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)
            await client.initialize()
            await client.initialize()  # Should skip
            assert mock_gdb.driver.call_count == 1

    @pytest.mark.asyncio
    async def test_cleanup(self, client):
        mock_driver = AsyncMock()
        mock_driver.close = AsyncMock()
        client._driver = mock_driver
        client._initialized = True

        await client.cleanup()
        assert client._initialized is False
        assert client._driver is None
        mock_driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_driver(self, client):
        await client.cleanup()  # Should not raise
        assert client._initialized is False


class TestNeo4jUpsertEntity:
    @pytest.mark.asyncio
    async def test_upsert_entity(self, client):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "e1"})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        entity = Entity(
            id="e1", name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        result = await client.upsert_entity(entity)
        assert result == "e1"

    @pytest.mark.asyncio
    async def test_upsert_entity_failure(self, client):
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=Exception("DB error"))
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        entity = Entity(
            id="e1", name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        with pytest.raises(GraphError, match="Failed to upsert entity"):
            await client.upsert_entity(entity)


class TestNeo4jGetEntity:
    @pytest.mark.asyncio
    async def test_get_entity_found(self, client):
        mock_node = {
            "id": "e1", "name": "Apple",
            "normalized_name": "apple", "entity_type": "COMPANY",
            "confidence": 0.9,
        }
        mock_record = {"e": mock_node}
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        entity = await client.get_entity("e1")
        assert entity is not None
        assert entity.name == "Apple"
        assert entity.entity_type == EntityType.COMPANY

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, client):
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        entity = await client.get_entity("nonexistent")
        assert entity is None


class TestNeo4jUpsertRelation:
    @pytest.mark.asyncio
    async def test_upsert_relation(self, client):
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"id": "r1"})
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        relation = Relation(
            id="r1", source_entity_id="e1", target_entity_id="e2",
            relation_type=RelationType.ACQUIRED,
        )
        result = await client.upsert_relation(relation)
        assert result == "r1"


class TestNeo4jDocumentOperations:
    @pytest.mark.asyncio
    async def test_create_document_node(self, client):
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        await client.create_document_node("doc-1", "test.pdf", {"pages": 10})
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_chunk_node(self, client):
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        await client.create_chunk_node("c1", "doc-1", "preview text", "text")
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_entity_to_chunk(self, client):
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        await client.link_entity_to_chunk("e1", "c1")
        mock_session.run.assert_called_once()


class TestNeo4jGraphTraversal:
    @pytest.mark.asyncio
    async def test_get_subgraph_empty(self, client):
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        nodes, edges = await client.get_subgraph(["e1"])
        assert nodes == []
        assert edges == []

    @pytest.mark.asyncio
    async def test_find_paths_empty(self, client):
        mock_result = AsyncMock()
        mock_result.__aiter__ = MagicMock(return_value=iter([]))
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        paths = await client.find_paths("e1", "e2")
        assert paths == []


class TestNeo4jStats:
    @pytest.mark.asyncio
    async def test_get_stats_failure(self, client):
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=Exception("Query failed"))
        mock_session.close = AsyncMock()

        client._initialized = True
        client._driver = MagicMock()
        client._driver.session = MagicMock(return_value=mock_session)

        stats = await client.get_stats()
        assert stats == {}
