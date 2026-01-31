"""
Hybrid retriever combining vector search and graph traversal.

Implements the core GraphRAG retrieval pipeline:
1. Query understanding and entity extraction
2. Vector-based coarse retrieval
3. Graph subgraph construction
4. Graph-aware re-ranking
5. Result aggregation with explainability
"""

import time
from typing import Any

from src.core.logging import LoggerMixin, log_operation
from src.core.types import (
    Chunk,
    ChunkMetadata,
    Entity,
    GraphPath,
    Modality,
    RetrievalResult,
)
from src.kg.neo4j_client import Neo4jClient
from src.kg.ner_extractor import NERExtractor
from src.retrieval.graph_traversal import GraphTraversal
from src.retrieval.reranker import GraphAwareReranker, HybridReranker
from src.vector.embeddings import EmbeddingService
from src.vector.qdrant_client import QdrantClient, VectorSearchResult


class HybridRetriever(LoggerMixin):
    """
    Hybrid retriever for GraphRAG.

    Combines dense vector retrieval with knowledge graph
    traversal for relationship-aware, explainable retrieval.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_client: QdrantClient,
        neo4j_client: Neo4jClient,
        ner_extractor: NERExtractor | None = None,
        top_k_vector: int = 20,
        top_k_final: int = 5,
        graph_hops: int = 2,
        max_graph_nodes: int = 100,
        graph_weight: float = 0.4,
        vector_weight: float = 0.6,
        use_cross_encoder: bool = False,
        **config: Any,
    ) -> None:
        """
        Initialize hybrid retriever.

        Args:
            embedding_service: Service for query embedding
            qdrant_client: Vector database client
            neo4j_client: Graph database client
            ner_extractor: NER extractor for query entities
            top_k_vector: Number of vector results to retrieve
            top_k_final: Final number of results to return
            graph_hops: Maximum graph traversal depth
            max_graph_nodes: Maximum nodes in subgraph
            graph_weight: Weight for graph-based scores
            vector_weight: Weight for vector similarity
            use_cross_encoder: Whether to use cross-encoder reranking
            **config: Additional configuration
        """
        self.embedding_service = embedding_service
        self.qdrant_client = qdrant_client
        self.neo4j_client = neo4j_client
        self.ner_extractor = ner_extractor

        self.top_k_vector = top_k_vector
        self.top_k_final = top_k_final
        self.graph_hops = graph_hops
        self.max_graph_nodes = max_graph_nodes

        # Initialize components
        self.graph_traversal = GraphTraversal(
            neo4j_client=neo4j_client,
            max_depth=graph_hops,
            max_nodes=max_graph_nodes,
        )

        self.reranker = HybridReranker(
            use_cross_encoder=use_cross_encoder,
            graph_weight=graph_weight,
            vector_weight=vector_weight,
        )

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        await self.embedding_service.initialize()

        if self.ner_extractor:
            await self.ner_extractor.initialize()

        self._initialized = True
        self.logger.info("Hybrid retriever initialized")

    async def retrieve(
        self,
        query: str,
        filter_doc_ids: list[str] | None = None,
        modalities: list[Modality] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Pipeline:
        1. Extract entities from query
        2. Generate query embedding
        3. Vector search for coarse retrieval
        4. Build graph subgraph from results
        5. Find relevant graph paths
        6. Re-rank with graph information
        7. Return enriched results

        Args:
            query: User query
            filter_doc_ids: Optional document ID filter
            modalities: Optional modality filter

        Returns:
            RetrievalResult with chunks, entities, and paths
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self.logger.info("Starting hybrid retrieval", query=query[:100])

        # Step 1: Extract entities from query
        query_entities = []
        if self.ner_extractor:
            try:
                query_entities = await self.ner_extractor.extract(query)
                self.logger.debug(
                    "Query entities extracted",
                    count=len(query_entities),
                )
            except Exception as e:
                self.logger.warning(f"Query NER failed: {e}")

        # Step 2: Generate query embedding
        query_embedding = await self.embedding_service.embed_query(query)

        # Step 3: Vector search
        filter_conditions = {}
        if filter_doc_ids:
            filter_conditions["doc_id"] = filter_doc_ids
        if modalities:
            filter_conditions["modality"] = [m.value for m in modalities]

        vector_results = await self.qdrant_client.search_all_collections(
            query_vector=query_embedding,
            top_k=self.top_k_vector,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        self.logger.debug(
            "Vector search completed",
            results=len(vector_results),
        )

        # Step 4: Build graph subgraph
        # Collect entity IDs from results
        result_entity_ids = set()
        for result in vector_results:
            entity_ids = result.payload.get("entity_ids", [])
            result_entity_ids.update(entity_ids)

        # Add query entity IDs
        query_entity_ids = [e.id for e in query_entities]
        all_entity_ids = list(result_entity_ids | set(query_entity_ids))

        # Get subgraph
        graph_nodes, graph_edges = [], []
        if all_entity_ids:
            graph_nodes, graph_edges = await self.graph_traversal.get_subgraph(
                entity_ids=all_entity_ids[:20],  # Limit starting points
                depth=self.graph_hops,
            )

            self.logger.debug(
                "Subgraph retrieved",
                nodes=len(graph_nodes),
                edges=len(graph_edges),
            )

        # Step 5: Find relevant graph paths
        graph_paths = []
        if len(all_entity_ids) >= 2:
            graph_paths = await self.graph_traversal.find_connecting_paths(
                entity_ids=all_entity_ids[:10],
                max_path_length=3,
            )

        # Calculate entity centrality for reranking
        entity_centrality = {}
        if graph_nodes and graph_edges:
            centrality_ranking = self.graph_traversal.rank_entities_by_centrality(
                graph_nodes, graph_edges
            )
            entity_centrality = dict(centrality_ranking)

        # Step 6: Re-rank with graph information
        reranked_results = await self.reranker.rerank(
            query=query,
            results=vector_results,
            query_entities=query_entities,
            graph_paths=[p.model_dump() for p in graph_paths],
            entity_centrality=entity_centrality,
        )

        # Step 7: Build final result
        final_results = reranked_results[:self.top_k_final]

        # Convert to chunks
        chunks = []
        vector_scores = {}
        graph_scores = {}

        for result in final_results:
            chunk = Chunk(
                id=result.id,
                content=result.payload.get("content", ""),
                modality=Modality(result.payload.get("modality", "text")),
                metadata=ChunkMetadata(
                    doc_id=result.payload.get("doc_id", ""),
                    page_number=result.payload.get("page_number"),
                    section=result.payload.get("section"),
                    table_id=result.payload.get("table_id"),
                    image_id=result.payload.get("image_id"),
                    source_file=result.payload.get("source_file"),
                ),
                entity_ids=result.payload.get("entity_ids", []),
            )
            chunks.append(chunk)
            vector_scores[result.id] = result.score

        # Get entities from results
        result_entities = []
        seen_entity_ids = set()

        for chunk in chunks:
            for entity_id in chunk.entity_ids:
                if entity_id not in seen_entity_ids:
                    seen_entity_ids.add(entity_id)
                    entity = await self.neo4j_client.get_entity(entity_id)
                    if entity:
                        result_entities.append(entity)

        retrieval_time_ms = (time.time() - start_time) * 1000

        log_operation(
            "hybrid_retrieval",
            success=True,
            duration_ms=retrieval_time_ms,
            query_length=len(query),
            results=len(chunks),
            entities=len(result_entities),
            paths=len(graph_paths),
        )

        return RetrievalResult(
            chunks=chunks,
            entities=result_entities,
            graph_paths=graph_paths,
            vector_scores=vector_scores,
            graph_scores=graph_scores,
            retrieval_time_ms=retrieval_time_ms,
        )

    async def retrieve_by_entities(
        self,
        entity_names: list[str],
        expand_graph: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve chunks related to specific entities.

        Args:
            entity_names: Entity names to search for
            expand_graph: Whether to expand to related entities

        Returns:
            RetrievalResult with related chunks
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Find entities by name
        entities = []
        for name in entity_names:
            search_results = await self.neo4j_client.search_entities(
                query=name,
                limit=3,
            )
            entities.extend(search_results)

        if not entities:
            return RetrievalResult(
                chunks=[],
                entities=[],
                graph_paths=[],
                vector_scores={},
                graph_scores={},
                retrieval_time_ms=(time.time() - start_time) * 1000,
            )

        entity_ids = [e.id for e in entities]

        # Expand graph if requested
        if expand_graph:
            nodes, edges = await self.graph_traversal.get_subgraph(
                entity_ids=entity_ids,
                depth=1,
            )
            entity_ids = [n["id"] for n in nodes]

        # Find chunks containing these entities
        results = await self.qdrant_client.search_by_entity_ids(
            entity_ids=entity_ids,
            top_k=self.top_k_vector,
        )

        # Convert to chunks
        chunks = []
        for result in results[:self.top_k_final]:
            chunk = Chunk(
                id=result.id,
                content=result.payload.get("content", ""),
                modality=Modality(result.payload.get("modality", "text")),
                metadata=ChunkMetadata(
                    doc_id=result.payload.get("doc_id", ""),
                    page_number=result.payload.get("page_number"),
                    section=result.payload.get("section"),
                ),
                entity_ids=result.payload.get("entity_ids", []),
            )
            chunks.append(chunk)

        # Find paths between entities
        graph_paths = []
        if len(entity_ids) >= 2:
            graph_paths = await self.graph_traversal.find_connecting_paths(
                entity_ids=entity_ids[:5],
                max_path_length=3,
            )

        return RetrievalResult(
            chunks=chunks,
            entities=entities,
            graph_paths=graph_paths,
            vector_scores={r.id: r.score for r in results},
            graph_scores={},
            retrieval_time_ms=(time.time() - start_time) * 1000,
        )

    async def get_entity_context(
        self,
        entity_id: str,
    ) -> dict[str, Any]:
        """
        Get full context for an entity.

        Returns entity details, related chunks, and graph neighborhood.

        Args:
            entity_id: Entity ID

        Returns:
            Context dictionary
        """
        # Get entity
        entity = await self.neo4j_client.get_entity(entity_id)
        if not entity:
            return {}

        # Get neighborhood
        nodes, edges = await self.neo4j_client.get_entity_neighborhood(
            entity_id=entity_id,
            hops=2,
        )

        # Get related chunks
        chunk_results = await self.qdrant_client.search_by_entity_ids(
            entity_ids=[entity_id],
            top_k=10,
        )

        return {
            "entity": entity.model_dump(),
            "neighbors": nodes,
            "relations": edges,
            "chunks": [
                {
                    "id": r.id,
                    "content": r.payload.get("content", ""),
                    "doc_id": r.payload.get("doc_id"),
                }
                for r in chunk_results
            ],
        }

    async def __aenter__(self) -> "HybridRetriever":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
