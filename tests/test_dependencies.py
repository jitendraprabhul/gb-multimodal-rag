"""
Tests for the service container and FastAPI dependencies.

Covers:
- ServiceContainer singleton pattern
- Service property accessors
- Initialization order
- Cleanup
- FastAPI dependency functions
"""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from config.settings import Settings
from src.api.dependencies import (
    ServiceContainer,
    get_container,
    get_ollama,
    get_embeddings,
    get_qdrant,
    get_neo4j,
    get_graph_builder,
    get_etl,
    get_retriever,
    get_reasoner,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        app_env="development",
        debug=True,
        log_level="DEBUG",
        domain="finance",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test_password",
        qdrant_host="localhost",
        qdrant_port=6333,
        ollama_host="http://mock-ollama:11434",
        text_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset ServiceContainer singleton between tests."""
    ServiceContainer._instance = None
    yield
    ServiceContainer._instance = None


# =============================================================================
# ServiceContainer Tests
# =============================================================================


class TestServiceContainerInit:
    def test_construction(self, test_settings):
        container = ServiceContainer(test_settings)
        assert container.settings is test_settings
        assert container._initialized is False

    def test_services_initially_none(self, test_settings):
        container = ServiceContainer(test_settings)
        assert container._ollama_client is None
        assert container._embedding_service is None
        assert container._qdrant_client is None
        assert container._neo4j_client is None
        assert container._ner_extractor is None
        assert container._graph_builder is None
        assert container._etl_pipeline is None
        assert container._hybrid_retriever is None
        assert container._reasoner is None


class TestServiceContainerSingleton:
    def test_get_instance_creates_singleton(self, test_settings):
        instance1 = ServiceContainer.get_instance(test_settings)
        instance2 = ServiceContainer.get_instance()
        assert instance1 is instance2

    def test_get_instance_uses_provided_settings(self, test_settings):
        instance = ServiceContainer.get_instance(test_settings)
        assert instance.settings is test_settings

    def test_singleton_reset(self, test_settings):
        instance1 = ServiceContainer.get_instance(test_settings)
        ServiceContainer._instance = None
        instance2 = ServiceContainer.get_instance(test_settings)
        assert instance1 is not instance2


class TestServiceContainerProperties:
    def test_ollama_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_client = MagicMock()
        container._ollama_client = mock_client
        assert container.ollama is mock_client

    def test_embeddings_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_service = MagicMock()
        container._embedding_service = mock_service
        assert container.embeddings is mock_service

    def test_qdrant_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_client = MagicMock()
        container._qdrant_client = mock_client
        assert container.qdrant is mock_client

    def test_neo4j_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_client = MagicMock()
        container._neo4j_client = mock_client
        assert container.neo4j is mock_client

    def test_ner_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_ner = MagicMock()
        container._ner_extractor = mock_ner
        assert container.ner is mock_ner

    def test_graph_builder_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_builder = MagicMock()
        container._graph_builder = mock_builder
        assert container.graph_builder is mock_builder

    def test_etl_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_pipeline = MagicMock()
        container._etl_pipeline = mock_pipeline
        assert container.etl is mock_pipeline

    def test_retriever_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_retriever = MagicMock()
        container._hybrid_retriever = mock_retriever
        assert container.retriever is mock_retriever

    def test_reasoner_property(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_reasoner = MagicMock()
        container._reasoner = mock_reasoner
        assert container.reasoner is mock_reasoner


class TestServiceContainerInitialize:
    async def test_initialize_calls_all_init_methods(self, test_settings):
        container = ServiceContainer(test_settings)

        # Mock all init methods
        container._init_ollama = AsyncMock()
        container._init_embeddings = AsyncMock()
        container._init_qdrant = AsyncMock()
        container._init_neo4j = AsyncMock()
        container._init_ner = AsyncMock()
        container._init_graph_builder = AsyncMock()
        container._init_etl = AsyncMock()
        container._init_retriever = AsyncMock()
        container._init_reasoner = AsyncMock()

        await container.initialize()

        assert container._initialized is True
        container._init_ollama.assert_called_once()
        container._init_embeddings.assert_called_once()
        container._init_qdrant.assert_called_once()
        container._init_neo4j.assert_called_once()
        container._init_ner.assert_called_once()
        container._init_graph_builder.assert_called_once()
        container._init_etl.assert_called_once()
        container._init_retriever.assert_called_once()
        container._init_reasoner.assert_called_once()

    async def test_initialize_idempotent(self, test_settings):
        container = ServiceContainer(test_settings)
        container._init_ollama = AsyncMock()
        container._init_embeddings = AsyncMock()
        container._init_qdrant = AsyncMock()
        container._init_neo4j = AsyncMock()
        container._init_ner = AsyncMock()
        container._init_graph_builder = AsyncMock()
        container._init_etl = AsyncMock()
        container._init_retriever = AsyncMock()
        container._init_reasoner = AsyncMock()

        await container.initialize()
        await container.initialize()  # Second call should be no-op

        container._init_ollama.assert_called_once()

    async def test_init_ollama(self, test_settings):
        container = ServiceContainer(test_settings)

        with patch("src.api.dependencies.OllamaClient") as MockOllama:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockOllama.return_value = mock_instance

            await container._init_ollama()

            MockOllama.assert_called_once_with(
                host=test_settings.ollama_host,
                model=test_settings.ollama_model,
                timeout=test_settings.ollama_timeout,
            )
            mock_instance.initialize.assert_called_once()
            assert container._ollama_client is mock_instance

    async def test_init_embeddings(self, test_settings):
        container = ServiceContainer(test_settings)

        with patch("src.api.dependencies.EmbeddingService") as MockEmbed:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockEmbed.return_value = mock_instance

            await container._init_embeddings()

            MockEmbed.assert_called_once_with(
                text_model=test_settings.text_embedding_model,
                image_model=test_settings.image_embedding_model,
                device=test_settings.embedding_device,
                batch_size=test_settings.embedding_batch_size,
            )
            mock_instance.initialize.assert_called_once()

    async def test_init_ner_finance(self, test_settings):
        container = ServiceContainer(test_settings)

        with patch("src.api.dependencies.FinanceNERExtractor") as MockNER:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockNER.return_value = mock_instance

            await container._init_ner()

            MockNER.assert_called_once()
            mock_instance.initialize.assert_called_once()

    async def test_init_ner_healthcare(self):
        settings = Settings(
            app_env="development",
            domain="healthcare",
            neo4j_password="test",
        )
        container = ServiceContainer(settings)

        with patch("src.api.dependencies.HealthcareNERExtractor") as MockNER:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            MockNER.return_value = mock_instance

            await container._init_ner()

            MockNER.assert_called_once()

    async def test_init_reasoner(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_ollama = MagicMock()
        container._ollama_client = mock_ollama

        with patch("src.api.dependencies.GraphAwareReasoner") as MockReasoner:
            mock_instance = MagicMock()
            MockReasoner.return_value = mock_instance

            await container._init_reasoner()

            MockReasoner.assert_called_once_with(
                ollama_client=mock_ollama,
                model=test_settings.ollama_model,
                max_tokens=1024,
                temperature=0.3,
                include_reasoning_chain=True,
            )


class TestServiceContainerCleanup:
    async def test_cleanup_calls_all_services(self, test_settings):
        container = ServiceContainer(test_settings)
        container._initialized = True

        # Set up mock services with cleanup methods
        container._ollama_client = MagicMock()
        container._ollama_client.cleanup = AsyncMock()
        container._embedding_service = MagicMock()
        container._embedding_service.cleanup = AsyncMock()
        container._qdrant_client = MagicMock()
        container._qdrant_client.cleanup = AsyncMock()
        container._neo4j_client = MagicMock()
        container._neo4j_client.cleanup = AsyncMock()
        container._graph_builder = MagicMock()
        container._graph_builder.cleanup = AsyncMock()
        container._etl_pipeline = MagicMock()
        container._etl_pipeline.cleanup = AsyncMock()

        await container.cleanup()

        assert container._initialized is False
        container._ollama_client.cleanup.assert_called_once()
        container._embedding_service.cleanup.assert_called_once()
        container._qdrant_client.cleanup.assert_called_once()
        container._neo4j_client.cleanup.assert_called_once()
        container._graph_builder.cleanup.assert_called_once()
        container._etl_pipeline.cleanup.assert_called_once()

    async def test_cleanup_handles_none_services(self, test_settings):
        container = ServiceContainer(test_settings)
        # All services are None; cleanup should not raise
        await container.cleanup()
        assert container._initialized is False


# =============================================================================
# FastAPI Dependency Function Tests
# =============================================================================


class TestDependencyFunctions:
    async def test_get_ollama(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_client = MagicMock()
        container._ollama_client = mock_client

        result = await get_ollama(container)
        assert result is mock_client

    async def test_get_embeddings(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_service = MagicMock()
        container._embedding_service = mock_service

        result = await get_embeddings(container)
        assert result is mock_service

    async def test_get_qdrant(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_client = MagicMock()
        container._qdrant_client = mock_client

        result = await get_qdrant(container)
        assert result is mock_client

    async def test_get_neo4j(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_client = MagicMock()
        container._neo4j_client = mock_client

        result = await get_neo4j(container)
        assert result is mock_client

    async def test_get_graph_builder(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_builder = MagicMock()
        container._graph_builder = mock_builder

        result = await get_graph_builder(container)
        assert result is mock_builder

    async def test_get_etl(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_pipeline = MagicMock()
        container._etl_pipeline = mock_pipeline

        result = await get_etl(container)
        assert result is mock_pipeline

    async def test_get_retriever(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_retriever = MagicMock()
        container._hybrid_retriever = mock_retriever

        result = await get_retriever(container)
        assert result is mock_retriever

    async def test_get_reasoner(self, test_settings):
        container = ServiceContainer(test_settings)
        mock_reasoner = MagicMock()
        container._reasoner = mock_reasoner

        result = await get_reasoner(container)
        assert result is mock_reasoner
