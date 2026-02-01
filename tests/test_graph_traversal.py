"""
Tests for graph traversal algorithms.

Covers:
- Subgraph extraction
- Path finding and scoring
- Entity context expansion
- Entity evidence paths
- Centrality ranking (PageRank-like)
- Edge filtering
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import Entity, EntityType, GraphPath, RelationType
from src.retrieval.graph_traversal import GraphTraversal, TraversalNode, TraversalEdge


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_neo4j():
    """Create a mock Neo4jClient."""
    client = MagicMock()
    client.get_subgraph = AsyncMock(return_value=([], []))
    client.find_paths = AsyncMock(return_value=[])
    client.get_entity_neighborhood = AsyncMock(return_value=([], []))
    return client


@pytest.fixture
def traversal(mock_neo4j):
    """Create GraphTraversal with mock client."""
    return GraphTraversal(
        neo4j_client=mock_neo4j,
        max_depth=2,
        max_nodes=100,
    )


# =============================================================================
# Data Class Tests
# =============================================================================


class TestTraversalNode:
    def test_creation(self):
        node = TraversalNode(
            entity_id="e1",
            entity_name="Apple",
            entity_type="COMPANY",
            depth=1,
        )
        assert node.entity_id == "e1"
        assert node.score == 1.0
        assert node.path == []

    def test_with_path(self):
        node = TraversalNode(
            entity_id="e2",
            entity_name="John",
            entity_type="PERSON",
            depth=2,
            path=["e0", "e1", "e2"],
            score=0.8,
        )
        assert len(node.path) == 3
        assert node.score == 0.8


class TestTraversalEdge:
    def test_creation(self):
        edge = TraversalEdge(
            source_id="e1",
            target_id="e2",
            relation_type="ACQUIRED",
        )
        assert edge.weight == 1.0

    def test_custom_weight(self):
        edge = TraversalEdge(
            source_id="e1",
            target_id="e2",
            relation_type="MENTIONS",
            weight=0.5,
        )
        assert edge.weight == 0.5


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGraphTraversalInit:
    def test_default_params(self, mock_neo4j):
        gt = GraphTraversal(neo4j_client=mock_neo4j)
        assert gt.max_depth == 2
        assert gt.max_nodes == 100
        assert gt.relation_weights == GraphTraversal.DEFAULT_RELATION_WEIGHTS

    def test_custom_params(self, mock_neo4j):
        custom_weights = {"MENTIONS": 0.1}
        gt = GraphTraversal(
            neo4j_client=mock_neo4j,
            max_depth=3,
            max_nodes=50,
            relation_weights=custom_weights,
        )
        assert gt.max_depth == 3
        assert gt.max_nodes == 50
        assert gt.relation_weights == custom_weights

    def test_default_relation_weights(self):
        weights = GraphTraversal.DEFAULT_RELATION_WEIGHTS
        assert weights[RelationType.FILED.value] == 1.0
        assert weights[RelationType.MENTIONS.value] == 0.5
        assert weights[RelationType.HAS_CONDITION.value] == 1.0


# =============================================================================
# Subgraph Tests
# =============================================================================


class TestGetSubgraph:
    async def test_calls_neo4j(self, traversal, mock_neo4j):
        nodes = [{"id": "e1", "type": "COMPANY"}]
        edges = [{"source": "e1", "target": "e2", "type": "ACQUIRED"}]
        mock_neo4j.get_subgraph.return_value = (nodes, edges)

        result_nodes, result_edges = await traversal.get_subgraph(
            entity_ids=["e1"],
        )

        assert len(result_nodes) == 1
        assert len(result_edges) == 1
        mock_neo4j.get_subgraph.assert_called_once_with(
            entity_ids=["e1"],
            hops=2,
            max_nodes=100,
            relation_types=None,
        )

    async def test_custom_depth(self, traversal, mock_neo4j):
        mock_neo4j.get_subgraph.return_value = ([], [])

        await traversal.get_subgraph(
            entity_ids=["e1"],
            depth=3,
        )

        mock_neo4j.get_subgraph.assert_called_once_with(
            entity_ids=["e1"],
            hops=3,
            max_nodes=100,
            relation_types=None,
        )

    async def test_with_relation_types(self, traversal, mock_neo4j):
        mock_neo4j.get_subgraph.return_value = ([], [])

        await traversal.get_subgraph(
            entity_ids=["e1"],
            relation_types=[RelationType.ACQUIRED],
        )

        mock_neo4j.get_subgraph.assert_called_once_with(
            entity_ids=["e1"],
            hops=2,
            max_nodes=100,
            relation_types=[RelationType.ACQUIRED],
        )


# =============================================================================
# Path Finding Tests
# =============================================================================


class TestFindConnectingPaths:
    async def test_returns_empty_for_single_entity(self, traversal):
        paths = await traversal.find_connecting_paths(entity_ids=["e1"])
        assert paths == []

    async def test_finds_paths_between_pairs(self, traversal, mock_neo4j):
        path = GraphPath(
            nodes=[{"id": "e1"}, {"id": "e2"}],
            edges=[{"type": "ACQUIRED", "source": "e1", "target": "e2"}],
            path_text="e1 -> e2",
            relevance_score=0.0,
        )
        mock_neo4j.find_paths.return_value = [path]

        paths = await traversal.find_connecting_paths(
            entity_ids=["e1", "e2"],
        )

        assert len(paths) == 1
        assert paths[0].relevance_score > 0

    async def test_multiple_entity_pairs(self, traversal, mock_neo4j):
        mock_neo4j.find_paths.return_value = []

        await traversal.find_connecting_paths(
            entity_ids=["e1", "e2", "e3"],
        )

        # 3 choose 2 = 3 pairs: (e1,e2), (e1,e3), (e2,e3)
        assert mock_neo4j.find_paths.call_count == 3

    async def test_paths_sorted_by_relevance(self, traversal, mock_neo4j):
        path1 = GraphPath(
            nodes=[{"id": "e1"}, {"id": "e2"}],
            edges=[{"type": "MENTIONS"}],
            path_text="e1 -> e2",
            relevance_score=0.0,
        )
        path2 = GraphPath(
            nodes=[{"id": "e1"}, {"id": "e2"}],
            edges=[{"type": "FILED"}],
            path_text="e1 -> e2",
            relevance_score=0.0,
        )
        mock_neo4j.find_paths.return_value = [path1, path2]

        paths = await traversal.find_connecting_paths(
            entity_ids=["e1", "e2"],
        )

        # FILED has higher weight than MENTIONS, so path2 should be first
        assert paths[0].relevance_score >= paths[1].relevance_score


# =============================================================================
# Path Scoring Tests
# =============================================================================


class TestScorePath:
    def test_empty_edges(self, traversal):
        path = GraphPath(
            nodes=[],
            edges=[],
            path_text="empty",
            relevance_score=0.0,
        )
        assert traversal._score_path(path) == 0.0

    def test_single_high_weight_edge(self, traversal):
        path = GraphPath(
            nodes=[{"id": "e1"}, {"id": "e2"}],
            edges=[{"type": RelationType.FILED.value}],
            path_text="e1 -> e2",
            relevance_score=0.0,
        )
        score = traversal._score_path(path)
        # length_score = 1/(1+1) = 0.5, relation_score = 1.0
        # total = 0.5 * 0.4 + 1.0 * 0.6 = 0.2 + 0.6 = 0.8
        assert abs(score - 0.8) < 0.01

    def test_longer_path_lower_score(self, traversal):
        short_path = GraphPath(
            nodes=[{"id": "e1"}, {"id": "e2"}],
            edges=[{"type": RelationType.MENTIONS.value}],
            path_text="short",
            relevance_score=0.0,
        )
        long_path = GraphPath(
            nodes=[{"id": f"e{i}"} for i in range(4)],
            edges=[{"type": RelationType.MENTIONS.value}] * 3,
            path_text="long",
            relevance_score=0.0,
        )

        short_score = traversal._score_path(short_path)
        long_score = traversal._score_path(long_path)

        assert short_score > long_score

    def test_unknown_relation_uses_default_weight(self, traversal):
        path = GraphPath(
            nodes=[{"id": "e1"}, {"id": "e2"}],
            edges=[{"type": "UNKNOWN_RELATION"}],
            path_text="test",
            relevance_score=0.0,
        )
        score = traversal._score_path(path)
        # default weight = 0.3
        # length_score = 0.5, relation_score = 0.3
        expected = 0.5 * 0.4 + 0.3 * 0.6
        assert abs(score - expected) < 0.01


# =============================================================================
# Entity Context Expansion Tests
# =============================================================================


class TestExpandEntityContext:
    async def test_returns_neighbors(self, traversal, mock_neo4j):
        nodes = [
            {"id": "e2", "type": "COMPANY"},
            {"id": "e3", "type": "PERSON"},
        ]
        mock_neo4j.get_entity_neighborhood.return_value = (nodes, [])

        result = await traversal.expand_entity_context("e1")

        assert len(result) == 2
        mock_neo4j.get_entity_neighborhood.assert_called_once_with(
            entity_id="e1",
            hops=1,
            limit=50,
        )

    async def test_filters_by_target_types(self, traversal, mock_neo4j):
        nodes = [
            {"id": "e2", "type": "COMPANY"},
            {"id": "e3", "type": "PERSON"},
            {"id": "e4", "type": "COMPANY"},
        ]
        mock_neo4j.get_entity_neighborhood.return_value = (nodes, [])

        result = await traversal.expand_entity_context(
            "e1",
            target_types=["COMPANY"],
        )

        assert len(result) == 2
        assert all(n["type"] == "COMPANY" for n in result)

    async def test_no_target_types_returns_all(self, traversal, mock_neo4j):
        nodes = [{"id": "e2", "type": "PERSON"}]
        mock_neo4j.get_entity_neighborhood.return_value = (nodes, [])

        result = await traversal.expand_entity_context("e1")
        assert len(result) == 1


# =============================================================================
# Entity Evidence Paths Tests
# =============================================================================


class TestGetEntityEvidencePaths:
    async def test_creates_paths(self, traversal):
        paths = await traversal.get_entity_evidence_paths(
            entity_id="e1",
            chunk_ids=["c1", "c2"],
        )

        assert len(paths) == 2
        assert paths[0].relevance_score == 1.0
        assert len(paths[0].nodes) == 2
        assert len(paths[0].edges) == 1

    async def test_empty_chunk_ids(self, traversal):
        paths = await traversal.get_entity_evidence_paths(
            entity_id="e1",
            chunk_ids=[],
        )
        assert paths == []

    async def test_path_text_format(self, traversal):
        paths = await traversal.get_entity_evidence_paths(
            entity_id="e1",
            chunk_ids=["abcdefgh-1234"],
        )
        assert "abcdefgh" in paths[0].path_text


# =============================================================================
# Centrality Ranking Tests
# =============================================================================


class TestRankEntitiesByCentrality:
    def test_empty_graph(self, traversal):
        result = traversal.rank_entities_by_centrality([], [])
        assert result == []

    def test_single_node(self, traversal):
        nodes = [{"id": "e1"}]
        result = traversal.rank_entities_by_centrality(nodes, [])
        assert len(result) == 1
        assert result[0][0] == "e1"

    def test_hub_node_gets_higher_score(self, traversal):
        nodes = [{"id": "hub"}, {"id": "a"}, {"id": "b"}, {"id": "c"}]
        edges = [
            {"source": "a", "target": "hub"},
            {"source": "b", "target": "hub"},
            {"source": "c", "target": "hub"},
        ]

        result = traversal.rank_entities_by_centrality(nodes, edges)

        # Hub should be ranked first (most incoming edges)
        assert result[0][0] == "hub"

    def test_returns_sorted_by_score(self, traversal):
        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "c", "target": "b"},
        ]

        result = traversal.rank_entities_by_centrality(nodes, edges)

        # Scores should be descending
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_ignores_edges_outside_node_set(self, traversal):
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "external", "target": "b"},  # external not in nodes
        ]

        result = traversal.rank_entities_by_centrality(nodes, edges)
        assert len(result) == 2


# =============================================================================
# Edge Filtering Tests
# =============================================================================


class TestFilterRelevantEdges:
    def test_no_filters(self, traversal):
        edges = [
            {"type": RelationType.FILED.value},
            {"type": RelationType.MENTIONS.value},
        ]
        filtered = traversal.filter_relevant_edges(edges)
        assert len(filtered) == 2

    def test_filter_by_relation_types(self, traversal):
        edges = [
            {"type": RelationType.FILED.value},
            {"type": RelationType.MENTIONS.value},
            {"type": RelationType.ACQUIRED.value},
        ]
        filtered = traversal.filter_relevant_edges(
            edges,
            relation_types=[RelationType.FILED.value],
        )
        assert len(filtered) == 1
        assert filtered[0]["type"] == RelationType.FILED.value

    def test_filter_by_min_weight(self, traversal):
        edges = [
            {"type": RelationType.FILED.value},      # weight 1.0
            {"type": RelationType.MENTIONS.value},    # weight 0.5
            {"type": RelationType.RELATED_TO.value},  # weight 0.4
        ]
        filtered = traversal.filter_relevant_edges(edges, min_weight=0.6)
        assert len(filtered) == 1
        assert filtered[0]["type"] == RelationType.FILED.value

    def test_unknown_type_uses_default_weight(self, traversal):
        edges = [{"type": "UNKNOWN"}]
        # Default weight is 0.3, min_weight default is 0.3
        filtered = traversal.filter_relevant_edges(edges, min_weight=0.3)
        assert len(filtered) == 1

    def test_filter_excludes_below_min_weight(self, traversal):
        edges = [{"type": "UNKNOWN"}]
        # Default weight 0.3, asking for min 0.4
        filtered = traversal.filter_relevant_edges(edges, min_weight=0.4)
        assert len(filtered) == 0

    def test_empty_edges(self, traversal):
        filtered = traversal.filter_relevant_edges([])
        assert filtered == []
