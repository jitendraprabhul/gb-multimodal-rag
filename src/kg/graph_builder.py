"""
Knowledge Graph builder.

Orchestrates the complete graph construction pipeline:
- Entity extraction (NER)
- Relation extraction
- Graph persistence (Neo4j)
- In-memory graph analysis (NetworkX)
"""

from typing import Any

import networkx as nx

from src.core.logging import LoggerMixin, log_operation
from src.core.types import Chunk, Document, Entity, Relation
from src.kg.neo4j_client import Neo4jClient
from src.kg.ner_extractor import (
    FinanceNERExtractor,
    HealthcareNERExtractor,
    NERExtractor,
)
from src.kg.relation_extractor import PatternRelationExtractor, RelationExtractor


class GraphBuilder(LoggerMixin):
    """
    Knowledge graph builder and manager.

    Coordinates entity extraction, relation extraction,
    and persistence to Neo4j. Also maintains an in-memory
    NetworkX graph for analysis and debugging.
    """

    def __init__(
        self,
        domain: str = "finance",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "graphrag_password",
        neo4j_database: str = "neo4j",
        use_llm_relations: bool = False,
        ollama_client: Any = None,
        **config: Any,
    ) -> None:
        """
        Initialize graph builder.

        Args:
            domain: Domain ("finance" or "healthcare")
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            use_llm_relations: Whether to use LLM for relation extraction
            ollama_client: Ollama client for LLM relations
            **config: Additional configuration
        """
        self.domain = domain
        self.use_llm_relations = use_llm_relations

        # Initialize components
        self.neo4j_client = Neo4jClient(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )

        # Select domain-specific NER
        if domain == "finance":
            self.ner_extractor: NERExtractor = FinanceNERExtractor()
        else:
            self.ner_extractor = HealthcareNERExtractor()

        # Initialize relation extractors
        self.pattern_extractor = PatternRelationExtractor(domain=domain)
        self.llm_extractor: RelationExtractor | None = None

        if use_llm_relations:
            from src.kg.relation_extractor import LLMRelationExtractor

            self.llm_extractor = LLMRelationExtractor(
                ollama_client=ollama_client,
                domain=domain,
            )

        # In-memory graph for analysis
        self._graph = nx.MultiDiGraph()
        self._entity_cache: dict[str, Entity] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        self.logger.info("Initializing graph builder", domain=self.domain)

        await self.neo4j_client.initialize()
        await self.ner_extractor.initialize()
        await self.pattern_extractor.initialize()

        if self.llm_extractor:
            await self.llm_extractor.initialize()

        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.neo4j_client.cleanup()
        await self.ner_extractor.cleanup()
        await self.pattern_extractor.cleanup()

        if self.llm_extractor:
            await self.llm_extractor.cleanup()

        self._initialized = False

    async def process_document(
        self,
        document: Document,
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Process a document and build graph.

        Args:
            document: Document to process

        Returns:
            Tuple of (entities, relations)
        """
        if not self._initialized:
            await self.initialize()

        import time
        start_time = time.time()

        self.logger.info(
            "Processing document for graph",
            doc_id=document.id,
            chunks=len(document.chunks),
        )

        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        # Create document node in Neo4j
        await self.neo4j_client.create_document_node(
            doc_id=document.id,
            filename=document.filename,
            metadata=document.metadata,
        )

        # Process each chunk
        for chunk in document.chunks:
            # Create chunk node
            await self.neo4j_client.create_chunk_node(
                chunk_id=chunk.id,
                doc_id=document.id,
                content_preview=chunk.content[:500],
                modality=chunk.modality.value,
            )

            # Extract entities
            entities = await self.ner_extractor.extract_from_chunk(chunk)

            # Deduplicate and merge entities
            for entity in entities:
                existing = self._entity_cache.get(
                    f"{entity.normalized_name}:{entity.entity_type.value}"
                )
                if existing:
                    # Merge with existing
                    existing.source_chunk_ids.extend(entity.source_chunk_ids)
                    existing.confidence = max(existing.confidence, entity.confidence)
                else:
                    self._entity_cache[
                        f"{entity.normalized_name}:{entity.entity_type.value}"
                    ] = entity
                    all_entities.append(entity)

            # Extract relations
            chunk_entities = [
                self._entity_cache.get(f"{e.normalized_name}:{e.entity_type.value}", e)
                for e in entities
            ]

            # Pattern-based relations
            pattern_relations = await self.pattern_extractor.extract_from_chunk(
                chunk, chunk_entities
            )
            all_relations.extend(pattern_relations)

            # LLM-based relations (if enabled)
            if self.llm_extractor and len(chunk_entities) >= 2:
                llm_relations = await self.llm_extractor.extract_from_chunk(
                    chunk, chunk_entities
                )
                all_relations.extend(llm_relations)

        # Persist to Neo4j
        await self._persist_to_neo4j(document, all_entities, all_relations)

        # Update in-memory graph
        self._update_networkx_graph(all_entities, all_relations)

        duration_ms = (time.time() - start_time) * 1000
        log_operation(
            "process_document_graph",
            success=True,
            duration_ms=duration_ms,
            doc_id=document.id,
            entities=len(all_entities),
            relations=len(all_relations),
        )

        return all_entities, all_relations

    async def _persist_to_neo4j(
        self,
        document: Document,
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Persist entities and relations to Neo4j."""
        # Upsert entities
        for entity in entities:
            await self.neo4j_client.upsert_entity(entity)

            # Link to source chunks
            for chunk_id in entity.source_chunk_ids:
                await self.neo4j_client.link_entity_to_chunk(entity.id, chunk_id)

        # Upsert relations
        for relation in relations:
            await self.neo4j_client.upsert_relation(relation)

    def _update_networkx_graph(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Update in-memory NetworkX graph."""
        # Add nodes
        for entity in entities:
            self._graph.add_node(
                entity.id,
                name=entity.name,
                normalized_name=entity.normalized_name,
                entity_type=entity.entity_type.value,
                confidence=entity.confidence,
            )

        # Add edges
        for relation in relations:
            self._graph.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                key=relation.id,
                relation_type=relation.relation_type.value,
                confidence=relation.confidence,
            )

    async def process_documents(
        self,
        documents: list[Document],
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Process multiple documents.

        Args:
            documents: Documents to process

        Returns:
            Tuple of (all_entities, all_relations)
        """
        all_entities = []
        all_relations = []

        for document in documents:
            entities, relations = await self.process_document(document)
            all_entities.extend(entities)
            all_relations.extend(relations)

        return all_entities, all_relations

    # =========================================================================
    # Graph Analysis (NetworkX)
    # =========================================================================

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Get entity from cache by name."""
        name_lower = name.lower()

        for key, entity in self._entity_cache.items():
            if entity.normalized_name == name_lower or entity.name.lower() == name_lower:
                return entity

        return None

    def get_neighbors(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
    ) -> list[str]:
        """Get neighbor entity IDs from in-memory graph."""
        if entity_id not in self._graph:
            return []

        neighbors = set()

        # Outgoing edges
        for _, target, data in self._graph.out_edges(entity_id, data=True):
            if relation_types is None or data.get("relation_type") in relation_types:
                neighbors.add(target)

        # Incoming edges
        for source, _, data in self._graph.in_edges(entity_id, data=True):
            if relation_types is None or data.get("relation_type") in relation_types:
                neighbors.add(source)

        return list(neighbors)

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> list[str] | None:
        """Find shortest path between entities in memory."""
        try:
            # Use undirected view for path finding
            undirected = self._graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_subgraph(
        self,
        entity_ids: list[str],
        hops: int = 2,
    ) -> nx.MultiDiGraph:
        """Get subgraph around entities from in-memory graph."""
        # Get all nodes within k hops
        nodes_to_include = set(entity_ids)

        for _ in range(hops):
            new_nodes = set()
            for node in nodes_to_include:
                if node in self._graph:
                    # Add successors and predecessors
                    new_nodes.update(self._graph.successors(node))
                    new_nodes.update(self._graph.predecessors(node))
            nodes_to_include.update(new_nodes)

        # Extract subgraph
        return self._graph.subgraph(nodes_to_include).copy()

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the in-memory graph."""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph),
            "is_connected": nx.is_weakly_connected(self._graph)
            if self._graph.number_of_nodes() > 0
            else False,
        }

    def export_to_gexf(self, filepath: str) -> None:
        """Export graph to GEXF format for visualization."""
        nx.write_gexf(self._graph, filepath)

    # =========================================================================
    # Combined Queries (Neo4j + NetworkX)
    # =========================================================================

    async def get_entity_context(
        self,
        entity_id: str,
        hops: int = 2,
        use_neo4j: bool = True,
    ) -> dict[str, Any]:
        """
        Get entity context including neighbors and relations.

        Args:
            entity_id: Entity ID
            hops: Number of hops to include
            use_neo4j: Use Neo4j (True) or NetworkX (False)

        Returns:
            Entity context dictionary
        """
        if use_neo4j:
            nodes, edges = await self.neo4j_client.get_entity_neighborhood(
                entity_id, hops=hops
            )
        else:
            subgraph = self.get_subgraph([entity_id], hops=hops)
            nodes = [
                {
                    "id": n,
                    "name": subgraph.nodes[n].get("name"),
                    "type": subgraph.nodes[n].get("entity_type"),
                }
                for n in subgraph.nodes()
            ]
            edges = [
                {
                    "source": u,
                    "target": v,
                    "type": d.get("relation_type"),
                }
                for u, v, d in subgraph.edges(data=True)
            ]

        # Get entity details
        entity = await self.neo4j_client.get_entity(entity_id)

        return {
            "entity": entity.model_dump() if entity else None,
            "neighbors": nodes,
            "relations": edges,
            "subgraph_size": len(nodes),
        }

    async def __aenter__(self) -> "GraphBuilder":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
