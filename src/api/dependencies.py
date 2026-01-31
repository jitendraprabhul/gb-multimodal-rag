"""
FastAPI dependencies for dependency injection.

Manages singleton instances of services and provides
them to route handlers.
"""

from functools import lru_cache
from typing import Any, AsyncGenerator

from fastapi import Depends

from config.settings import Settings, get_settings
from src.core.logging import setup_logging
from src.etl.pipeline import ETLPipeline
from src.kg.graph_builder import GraphBuilder
from src.kg.neo4j_client import Neo4jClient
from src.kg.ner_extractor import FinanceNERExtractor, HealthcareNERExtractor
from src.llm.ollama_client import OllamaClient
from src.llm.reasoning import GraphAwareReasoner
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vector.embeddings import EmbeddingService
from src.vector.qdrant_client import QdrantClient


class ServiceContainer:
    """
    Container for all service instances.

    Implements the Service Locator pattern for centralized
    dependency management.
    """

    _instance: "ServiceContainer | None" = None

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._initialized = False

        # Service instances (lazy initialized)
        self._ollama_client: OllamaClient | None = None
        self._embedding_service: EmbeddingService | None = None
        self._qdrant_client: QdrantClient | None = None
        self._neo4j_client: Neo4jClient | None = None
        self._ner_extractor = None
        self._graph_builder: GraphBuilder | None = None
        self._etl_pipeline: ETLPipeline | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._reasoner: GraphAwareReasoner | None = None

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> "ServiceContainer":
        """Get singleton instance."""
        if cls._instance is None:
            if settings is None:
                settings = get_settings()
            cls._instance = cls(settings)
        return cls._instance

    async def initialize(self) -> None:
        """Initialize all services."""
        if self._initialized:
            return

        # Setup logging
        setup_logging(
            level=self.settings.log_level,
            json_format=self.settings.is_production,
        )

        # Initialize services in order
        await self._init_ollama()
        await self._init_embeddings()
        await self._init_qdrant()
        await self._init_neo4j()
        await self._init_ner()
        await self._init_graph_builder()
        await self._init_etl()
        await self._init_retriever()
        await self._init_reasoner()

        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup all services."""
        if self._ollama_client:
            await self._ollama_client.cleanup()
        if self._embedding_service:
            await self._embedding_service.cleanup()
        if self._qdrant_client:
            await self._qdrant_client.cleanup()
        if self._neo4j_client:
            await self._neo4j_client.cleanup()
        if self._graph_builder:
            await self._graph_builder.cleanup()
        if self._etl_pipeline:
            await self._etl_pipeline.cleanup()

        self._initialized = False

    async def _init_ollama(self) -> None:
        """Initialize Ollama client."""
        self._ollama_client = OllamaClient(
            host=self.settings.ollama_host,
            model=self.settings.ollama_model,
            timeout=self.settings.ollama_timeout,
        )
        await self._ollama_client.initialize()

    async def _init_embeddings(self) -> None:
        """Initialize embedding service."""
        self._embedding_service = EmbeddingService(
            text_model=self.settings.text_embedding_model,
            image_model=self.settings.image_embedding_model,
            device=self.settings.embedding_device,
            batch_size=self.settings.embedding_batch_size,
        )
        await self._embedding_service.initialize()

    async def _init_qdrant(self) -> None:
        """Initialize Qdrant client."""
        self._qdrant_client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            collection_text=self.settings.qdrant_collection_text,
            collection_table=self.settings.qdrant_collection_table,
            collection_image=self.settings.qdrant_collection_image,
        )
        await self._qdrant_client.initialize(
            text_dimension=self._embedding_service.text_dimension,
            image_dimension=self._embedding_service.image_dimension,
        )

    async def _init_neo4j(self) -> None:
        """Initialize Neo4j client."""
        self._neo4j_client = Neo4jClient(
            uri=self.settings.neo4j_uri,
            user=self.settings.neo4j_user,
            password=self.settings.neo4j_password,
            database=self.settings.neo4j_database,
        )
        await self._neo4j_client.initialize()

    async def _init_ner(self) -> None:
        """Initialize NER extractor based on domain."""
        if self.settings.domain.value == "finance":
            self._ner_extractor = FinanceNERExtractor()
        else:
            self._ner_extractor = HealthcareNERExtractor()

        await self._ner_extractor.initialize()

    async def _init_graph_builder(self) -> None:
        """Initialize graph builder."""
        self._graph_builder = GraphBuilder(
            domain=self.settings.domain.value,
            neo4j_uri=self.settings.neo4j_uri,
            neo4j_user=self.settings.neo4j_user,
            neo4j_password=self.settings.neo4j_password,
            neo4j_database=self.settings.neo4j_database,
            use_llm_relations=False,  # Start without LLM relations
            ollama_client=self._ollama_client,
        )
        await self._graph_builder.initialize()

    async def _init_etl(self) -> None:
        """Initialize ETL pipeline."""
        self._etl_pipeline = ETLPipeline(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            extract_images=True,
            extract_tables=True,
            enable_ocr=True,
            ocr_lang=self.settings.ocr_lang,
            use_gpu=self.settings.ocr_use_gpu,
            max_concurrent=self.settings.max_concurrent_docs,
        )
        await self._etl_pipeline.initialize()

    async def _init_retriever(self) -> None:
        """Initialize hybrid retriever."""
        self._hybrid_retriever = HybridRetriever(
            embedding_service=self._embedding_service,
            qdrant_client=self._qdrant_client,
            neo4j_client=self._neo4j_client,
            ner_extractor=self._ner_extractor,
            top_k_vector=self.settings.top_k_vector,
            top_k_final=self.settings.top_k_final,
            graph_hops=self.settings.graph_hop_limit,
            max_graph_nodes=self.settings.max_graph_nodes,
        )
        await self._hybrid_retriever.initialize()

    async def _init_reasoner(self) -> None:
        """Initialize reasoning engine."""
        self._reasoner = GraphAwareReasoner(
            ollama_client=self._ollama_client,
            model=self.settings.ollama_model,
            max_tokens=1024,
            temperature=0.3,
            include_reasoning_chain=True,
        )

    # Properties for accessing services
    @property
    def ollama(self) -> OllamaClient:
        return self._ollama_client

    @property
    def embeddings(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def qdrant(self) -> QdrantClient:
        return self._qdrant_client

    @property
    def neo4j(self) -> Neo4jClient:
        return self._neo4j_client

    @property
    def ner(self):
        return self._ner_extractor

    @property
    def graph_builder(self) -> GraphBuilder:
        return self._graph_builder

    @property
    def etl(self) -> ETLPipeline:
        return self._etl_pipeline

    @property
    def retriever(self) -> HybridRetriever:
        return self._hybrid_retriever

    @property
    def reasoner(self) -> GraphAwareReasoner:
        return self._reasoner


# Global container instance
_container: ServiceContainer | None = None


async def get_container() -> ServiceContainer:
    """Get initialized service container."""
    global _container
    if _container is None:
        settings = get_settings()
        _container = ServiceContainer(settings)
        await _container.initialize()
    return _container


# FastAPI dependencies
async def get_settings_dep() -> Settings:
    """Get settings dependency."""
    return get_settings()


async def get_ollama(
    container: ServiceContainer = Depends(get_container),
) -> OllamaClient:
    """Get Ollama client dependency."""
    return container.ollama


async def get_embeddings(
    container: ServiceContainer = Depends(get_container),
) -> EmbeddingService:
    """Get embedding service dependency."""
    return container.embeddings


async def get_qdrant(
    container: ServiceContainer = Depends(get_container),
) -> QdrantClient:
    """Get Qdrant client dependency."""
    return container.qdrant


async def get_neo4j(
    container: ServiceContainer = Depends(get_container),
) -> Neo4jClient:
    """Get Neo4j client dependency."""
    return container.neo4j


async def get_graph_builder(
    container: ServiceContainer = Depends(get_container),
) -> GraphBuilder:
    """Get graph builder dependency."""
    return container.graph_builder


async def get_etl(
    container: ServiceContainer = Depends(get_container),
) -> ETLPipeline:
    """Get ETL pipeline dependency."""
    return container.etl


async def get_retriever(
    container: ServiceContainer = Depends(get_container),
) -> HybridRetriever:
    """Get hybrid retriever dependency."""
    return container.retriever


async def get_reasoner(
    container: ServiceContainer = Depends(get_container),
) -> GraphAwareReasoner:
    """Get reasoning engine dependency."""
    return container.reasoner
