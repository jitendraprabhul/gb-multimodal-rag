"""
Graph traversal algorithms for knowledge graph exploration.

Implements:
- BFS/DFS traversal
- Weighted path finding
- Subgraph extraction
- Path ranking
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from src.core.logging import LoggerMixin
from src.core.types import Entity, GraphPath, RelationType
from src.kg.neo4j_client import Neo4jClient


@dataclass
class TraversalNode:
    """Node in traversal with metadata."""

    entity_id: str
    entity_name: str
    entity_type: str
    depth: int
    path: list[str] = field(default_factory=list)
    score: float = 1.0


@dataclass
class TraversalEdge:
    """Edge in traversal with metadata."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0


class GraphTraversal(LoggerMixin):
    """
    Graph traversal algorithms for the knowledge graph.

    Provides efficient traversal methods for extracting
    relevant subgraphs around query-related entities.
    """

    # Relation weights by type (higher = more relevant)
    DEFAULT_RELATION_WEIGHTS = {
        # Finance - high weight for direct relationships
        RelationType.FILED.value: 1.0,
        RelationType.MENTIONS_METRIC.value: 0.9,
        RelationType.REPORTED.value: 0.9,
        RelationType.ACQUIRED.value: 0.8,
        RelationType.MERGED_WITH.value: 0.8,
        RelationType.SUBSIDIARY_OF.value: 0.7,

        # Healthcare - high weight for clinical relationships
        RelationType.HAS_CONDITION.value: 1.0,
        RelationType.TREATED_WITH.value: 0.95,
        RelationType.PRESCRIBED.value: 0.9,
        RelationType.DIAGNOSED_WITH.value: 0.9,
        RelationType.CONTRAINDICATES.value: 0.85,
        RelationType.INTERACTS_WITH.value: 0.8,

        # Generic
        RelationType.MENTIONS.value: 0.5,
        RelationType.RELATED_TO.value: 0.4,
        RelationType.PART_OF.value: 0.6,
    }

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        max_depth: int = 2,
        max_nodes: int = 100,
        relation_weights: dict[str, float] | None = None,
        **config: Any,
    ) -> None:
        """
        Initialize graph traversal.

        Args:
            neo4j_client: Neo4j client for graph queries
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to retrieve
            relation_weights: Custom relation type weights
            **config: Additional configuration
        """
        self.neo4j_client = neo4j_client
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.relation_weights = relation_weights or self.DEFAULT_RELATION_WEIGHTS

    async def get_subgraph(
        self,
        entity_ids: list[str],
        relation_types: list[RelationType] | None = None,
        depth: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Get subgraph around specified entities.

        Args:
            entity_ids: Starting entity IDs
            relation_types: Allowed relation types
            depth: Traversal depth (default: max_depth)

        Returns:
            Tuple of (nodes, edges)
        """
        depth = depth or self.max_depth

        nodes, edges = await self.neo4j_client.get_subgraph(
            entity_ids=entity_ids,
            hops=depth,
            max_nodes=self.max_nodes,
            relation_types=relation_types,
        )

        return nodes, edges

    async def find_connecting_paths(
        self,
        entity_ids: list[str],
        max_path_length: int = 3,
    ) -> list[GraphPath]:
        """
        Find paths connecting multiple entities.

        Args:
            entity_ids: Entity IDs to connect
            max_path_length: Maximum path length

        Returns:
            List of connecting paths
        """
        if len(entity_ids) < 2:
            return []

        all_paths = []

        # Find paths between each pair
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                paths = await self.neo4j_client.find_paths(
                    source_id=entity_ids[i],
                    target_id=entity_ids[j],
                    max_hops=max_path_length,
                    limit=3,
                )
                all_paths.extend(paths)

        # Score and sort paths
        for path in all_paths:
            path.relevance_score = self._score_path(path)

        all_paths.sort(key=lambda p: p.relevance_score, reverse=True)

        return all_paths

    def _score_path(self, path: GraphPath) -> float:
        """
        Score a path based on relation types and length.

        Shorter paths with high-weight relations score higher.
        """
        if not path.edges:
            return 0.0

        # Base score from path length (shorter = better)
        length_score = 1.0 / (len(path.edges) + 1)

        # Relation type score
        relation_scores = []
        for edge in path.edges:
            rel_type = edge.get("type", "")
            weight = self.relation_weights.get(rel_type, 0.3)
            relation_scores.append(weight)

        avg_relation_score = sum(relation_scores) / len(relation_scores)

        # Combine scores
        return length_score * 0.4 + avg_relation_score * 0.6

    async def expand_entity_context(
        self,
        entity_id: str,
        target_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Expand context around an entity.

        Gets all directly connected entities, optionally
        filtered by target entity type.

        Args:
            entity_id: Entity to expand from
            target_types: Filter for target entity types

        Returns:
            List of connected entity dictionaries
        """
        nodes, edges = await self.neo4j_client.get_entity_neighborhood(
            entity_id=entity_id,
            hops=1,
            limit=50,
        )

        if target_types:
            nodes = [n for n in nodes if n.get("type") in target_types]

        return nodes

    async def get_entity_evidence_paths(
        self,
        entity_id: str,
        chunk_ids: list[str],
    ) -> list[GraphPath]:
        """
        Find paths from entity to chunks (evidence).

        Used to explain why an entity is relevant to
        specific document chunks.

        Args:
            entity_id: Entity ID
            chunk_ids: Chunk IDs that mention the entity

        Returns:
            Paths connecting entity to chunks
        """
        # This would require Chunk nodes in the graph
        # For now, return direct connection info
        paths = []

        for chunk_id in chunk_ids:
            path = GraphPath(
                nodes=[
                    {"id": entity_id, "type": "Entity"},
                    {"id": chunk_id, "type": "Chunk"},
                ],
                edges=[
                    {"type": "MENTIONED_IN", "source": entity_id, "target": chunk_id}
                ],
                path_text=f"Entity mentioned in chunk {chunk_id[:8]}...",
                relevance_score=1.0,
            )
            paths.append(path)

        return paths

    def rank_entities_by_centrality(
        self,
        nodes: list[dict],
        edges: list[dict],
    ) -> list[tuple[str, float]]:
        """
        Rank entities by their centrality in the subgraph.

        Uses a simplified PageRank-like algorithm.

        Args:
            nodes: Subgraph nodes
            edges: Subgraph edges

        Returns:
            List of (entity_id, score) tuples, sorted by score
        """
        if not nodes:
            return []

        # Build adjacency
        node_ids = {n["id"] for n in nodes}
        incoming = defaultdict(list)
        outgoing = defaultdict(int)

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")

            if source in node_ids and target in node_ids:
                incoming[target].append(source)
                outgoing[source] += 1

        # Initialize scores
        n = len(nodes)
        scores = {node["id"]: 1.0 / n for node in nodes}

        # Iterate (simplified PageRank)
        damping = 0.85
        for _ in range(10):  # 10 iterations
            new_scores = {}

            for node in nodes:
                node_id = node["id"]
                rank_sum = sum(
                    scores[src] / max(outgoing[src], 1)
                    for src in incoming[node_id]
                )
                new_scores[node_id] = (1 - damping) / n + damping * rank_sum

            scores = new_scores

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def filter_relevant_edges(
        self,
        edges: list[dict],
        relation_types: list[str] | None = None,
        min_weight: float = 0.3,
    ) -> list[dict]:
        """
        Filter edges by relevance.

        Args:
            edges: All edges
            relation_types: Allowed types
            min_weight: Minimum relation weight

        Returns:
            Filtered edges
        """
        filtered = []

        for edge in edges:
            rel_type = edge.get("type", "")

            # Type filter
            if relation_types and rel_type not in relation_types:
                continue

            # Weight filter
            weight = self.relation_weights.get(rel_type, 0.3)
            if weight < min_weight:
                continue

            filtered.append(edge)

        return filtered
