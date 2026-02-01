"""
Tests for re-ranking strategies.

Covers:
- GraphAwareReranker (entity/path/centrality boosting)
- ReciprocalRankFusion (RRF merging)
- CrossEncoderReranker (model-based scoring)
- HybridReranker (pipeline combining all strategies)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.core.types import Entity, EntityType
from src.vector.qdrant_client import VectorSearchResult
from src.retrieval.reranker import (
    GraphAwareReranker,
    ReciprocalRankFusion,
    CrossEncoderReranker,
    HybridReranker,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_result(id, score, entity_ids=None, content="test content"):
    return VectorSearchResult(
        id=id,
        score=score,
        payload={
            "content": content,
            "entity_ids": entity_ids or [],
        },
    )


def _make_entity(id, name="Test"):
    return Entity(
        id=id,
        name=name,
        normalized_name=name.lower(),
        entity_type=EntityType.COMPANY,
    )


# =============================================================================
# GraphAwareReranker Tests
# =============================================================================


class TestGraphAwareReranker:
    def test_default_params(self):
        reranker = GraphAwareReranker()
        assert reranker.graph_weight == 0.4
        assert reranker.vector_weight == 0.6
        assert reranker.entity_boost == 0.2
        assert reranker.path_boost == 0.15

    async def test_empty_results(self):
        reranker = GraphAwareReranker()
        results = await reranker.rerank("query", [])
        assert results == []

    async def test_preserves_order_without_graph_info(self):
        reranker = GraphAwareReranker()
        results = [
            _make_result("r1", 0.9),
            _make_result("r2", 0.8),
            _make_result("r3", 0.7),
        ]

        reranked = await reranker.rerank("query", results)

        # Without graph info, vector scores dominate
        assert reranked[0].id == "r1"
        assert reranked[1].id == "r2"
        assert reranked[2].id == "r3"

    async def test_entity_overlap_boosts_score(self):
        reranker = GraphAwareReranker()
        query_entities = [_make_entity("e1"), _make_entity("e2")]

        results = [
            _make_result("r1", 0.7, entity_ids=["e1", "e2"]),  # matches both
            _make_result("r2", 0.9, entity_ids=[]),              # no match but higher vector
        ]

        reranked = await reranker.rerank(
            "query",
            results,
            query_entities=query_entities,
        )

        # r1 should be boosted by entity overlap
        r1 = next(r for r in reranked if r.id == "r1")
        r2 = next(r for r in reranked if r.id == "r2")

        # r1's graph component should be > 0 because of entity match
        assert r1.score > 0.7 * 0.6  # More than just vector weight

    async def test_path_overlap_boosts_score(self):
        reranker = GraphAwareReranker()
        graph_paths = [
            {"nodes": [{"id": "e1"}, {"id": "e2"}], "edges": []}
        ]

        results = [
            _make_result("r1", 0.7, entity_ids=["e1"]),
            _make_result("r2", 0.7, entity_ids=[]),
        ]

        reranked = await reranker.rerank(
            "query",
            results,
            graph_paths=graph_paths,
        )

        r1 = next(r for r in reranked if r.id == "r1")
        r2 = next(r for r in reranked if r.id == "r2")
        assert r1.score > r2.score

    async def test_centrality_boosts_score(self):
        reranker = GraphAwareReranker()
        entity_centrality = {"e1": 0.9, "e2": 0.1}

        results = [
            _make_result("r1", 0.7, entity_ids=["e1"]),  # high centrality entity
            _make_result("r2", 0.7, entity_ids=["e2"]),  # low centrality entity
        ]

        reranked = await reranker.rerank(
            "query",
            results,
            entity_centrality=entity_centrality,
        )

        r1 = next(r for r in reranked if r.id == "r1")
        r2 = next(r for r in reranked if r.id == "r2")
        assert r1.score >= r2.score

    async def test_score_capped_at_one(self):
        reranker = GraphAwareReranker(graph_weight=0.9, vector_weight=0.9)
        query_entities = [_make_entity("e1")]
        graph_paths = [{"nodes": [{"id": "e1"}], "edges": []}]
        entity_centrality = {"e1": 1.0}

        results = [_make_result("r1", 1.0, entity_ids=["e1"])]

        reranked = await reranker.rerank(
            "query",
            results,
            query_entities=query_entities,
            graph_paths=graph_paths,
            entity_centrality=entity_centrality,
        )

        assert reranked[0].score <= 1.0

    async def test_results_sorted_descending(self):
        reranker = GraphAwareReranker()
        results = [
            _make_result("r1", 0.5),
            _make_result("r2", 0.9),
            _make_result("r3", 0.7),
        ]

        reranked = await reranker.rerank("query", results)

        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# ReciprocalRankFusion Tests
# =============================================================================


class TestReciprocalRankFusion:
    def test_default_k(self):
        rrf = ReciprocalRankFusion()
        assert rrf.k == 60

    def test_custom_k(self):
        rrf = ReciprocalRankFusion(k=30)
        assert rrf.k == 30

    async def test_single_result_list(self):
        rrf = ReciprocalRankFusion()
        results = [
            _make_result("r1", 0.9),
            _make_result("r2", 0.8),
        ]

        reranked = await rrf.rerank("query", results)

        assert len(reranked) == 2
        # First result: 1/(60+1) = ~0.0164
        assert reranked[0].id == "r1"
        assert abs(reranked[0].score - 1.0 / 61) < 0.001

    async def test_fusion_of_two_lists(self):
        rrf = ReciprocalRankFusion()
        primary = [
            _make_result("r1", 0.9),
            _make_result("r2", 0.8),
        ]
        additional = [[
            _make_result("r2", 0.95),
            _make_result("r3", 0.85),
        ]]

        reranked = await rrf.rerank("query", primary, additional_results=additional)

        assert len(reranked) == 3  # r1, r2, r3

        # r2 appears in both lists so it should have the highest RRF score
        assert reranked[0].id == "r2"

    async def test_duplicate_results_merged(self):
        rrf = ReciprocalRankFusion()
        primary = [_make_result("r1", 0.9)]
        additional = [[_make_result("r1", 0.8)]]

        reranked = await rrf.rerank("query", primary, additional_results=additional)

        assert len(reranked) == 1
        # Score should be sum of RRF contributions from both lists
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert abs(reranked[0].score - expected) < 0.001

    async def test_empty_results(self):
        rrf = ReciprocalRankFusion()
        reranked = await rrf.rerank("query", [])
        assert reranked == []

    async def test_sorted_by_rrf_score(self):
        rrf = ReciprocalRankFusion()
        results = [
            _make_result("r1", 0.5),
            _make_result("r2", 0.9),
        ]

        reranked = await rrf.rerank("query", results)

        # Regardless of original scores, RRF uses rank positions
        assert reranked[0].id == "r1"  # Rank 1 has higher RRF


# =============================================================================
# CrossEncoderReranker Tests
# =============================================================================


class TestCrossEncoderReranker:
    def test_default_params(self):
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._initialized is False

    async def test_empty_results(self):
        reranker = CrossEncoderReranker()
        results = await reranker.rerank("query", [])
        assert results == []

    async def test_fallback_when_model_unavailable(self):
        reranker = CrossEncoderReranker()
        reranker._initialized = True
        reranker._model = None  # Model failed to load

        results = [
            _make_result("r1", 0.9),
            _make_result("r2", 0.8),
        ]

        reranked = await reranker.rerank("query", results)

        # Should return original order
        assert reranked[0].id == "r1"
        assert reranked[1].id == "r2"

    async def test_with_model(self):
        reranker = CrossEncoderReranker()
        reranker._initialized = True
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9]  # r2 scores higher
        reranker._model = mock_model

        results = [
            _make_result("r1", 0.9, content="first doc"),
            _make_result("r2", 0.5, content="second doc"),
        ]

        reranked = await reranker.rerank("query", results)

        # r2 should now be first (cross-encoder gave it 0.9)
        assert reranked[0].id == "r2"
        assert reranked[0].score == 0.9
        mock_model.predict.assert_called_once()

    async def test_handles_model_failure_gracefully(self):
        reranker = CrossEncoderReranker()
        reranker._initialized = True
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Inference failed")
        reranker._model = mock_model

        results = [_make_result("r1", 0.9)]
        reranked = await reranker.rerank("query", results)

        # Should fall back to original results
        assert len(reranked) == 1
        assert reranked[0].id == "r1"


# =============================================================================
# HybridReranker Tests
# =============================================================================


class TestHybridReranker:
    def test_default_no_cross_encoder(self):
        reranker = HybridReranker()
        assert reranker.use_cross_encoder is False
        assert reranker.cross_encoder is None
        assert isinstance(reranker.rrf_reranker, ReciprocalRankFusion)
        assert isinstance(reranker.graph_reranker, GraphAwareReranker)

    def test_with_cross_encoder(self):
        reranker = HybridReranker(use_cross_encoder=True)
        assert reranker.cross_encoder is not None

    async def test_basic_reranking(self):
        reranker = HybridReranker()
        results = [
            _make_result("r1", 0.9),
            _make_result("r2", 0.8),
        ]

        reranked = await reranker.rerank("query", results)
        assert len(reranked) == 2

    async def test_with_additional_results_triggers_rrf(self):
        reranker = HybridReranker()
        reranker.rrf_reranker.rerank = AsyncMock(
            return_value=[_make_result("r1", 0.5)]
        )
        reranker.graph_reranker.rerank = AsyncMock(
            return_value=[_make_result("r1", 0.5)]
        )

        results = [_make_result("r1", 0.9)]
        additional = [[_make_result("r2", 0.8)]]

        await reranker.rerank("query", results, additional_results=additional)

        reranker.rrf_reranker.rerank.assert_called_once()

    async def test_without_additional_results_skips_rrf(self):
        reranker = HybridReranker()
        reranker.rrf_reranker.rerank = AsyncMock()
        reranker.graph_reranker.rerank = AsyncMock(
            return_value=[_make_result("r1", 0.5)]
        )

        results = [_make_result("r1", 0.9)]
        await reranker.rerank("query", results)

        reranker.rrf_reranker.rerank.assert_not_called()

    async def test_passes_graph_info_to_graph_reranker(self):
        reranker = HybridReranker()
        reranker.graph_reranker.rerank = AsyncMock(
            return_value=[_make_result("r1", 0.5)]
        )

        entities = [_make_entity("e1")]
        paths = [{"nodes": [{"id": "e1"}]}]
        centrality = {"e1": 0.9}

        await reranker.rerank(
            "query",
            [_make_result("r1", 0.9)],
            query_entities=entities,
            graph_paths=paths,
            entity_centrality=centrality,
        )

        call_kwargs = reranker.graph_reranker.rerank.call_args.kwargs
        assert call_kwargs["query_entities"] == entities
        assert call_kwargs["graph_paths"] == paths
        assert call_kwargs["entity_centrality"] == centrality

    async def test_empty_results(self):
        reranker = HybridReranker()
        reranked = await reranker.rerank("query", [])
        assert reranked == []
