"""
Re-ranking strategies for hybrid retrieval.

Implements:
- Graph-aware re-ranking
- Reciprocal rank fusion
- Cross-encoder re-ranking (optional)
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.core.logging import LoggerMixin
from src.core.types import Chunk, Entity
from src.vector.qdrant_client import VectorSearchResult


class Reranker(ABC, LoggerMixin):
    """Abstract base class for re-rankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """
        Re-rank search results.

        Args:
            query: Original query
            results: Initial search results
            **kwargs: Additional context

        Returns:
            Re-ranked results
        """
        pass


class GraphAwareReranker(Reranker):
    """
    Re-ranker that incorporates graph structure.

    Boosts results that are:
    - Connected to query entities
    - Part of relevant graph paths
    - Highly central in the graph
    """

    def __init__(
        self,
        graph_weight: float = 0.4,
        vector_weight: float = 0.6,
        entity_boost: float = 0.2,
        path_boost: float = 0.15,
        **config: Any,
    ) -> None:
        """
        Initialize graph-aware re-ranker.

        Args:
            graph_weight: Weight for graph-based scores
            vector_weight: Weight for vector similarity scores
            entity_boost: Score boost for entity matches
            path_boost: Score boost for path relevance
            **config: Additional configuration
        """
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight
        self.entity_boost = entity_boost
        self.path_boost = path_boost

    async def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        query_entities: list[Entity] | None = None,
        graph_paths: list[dict] | None = None,
        entity_centrality: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """
        Re-rank results using graph information.

        Args:
            query: Original query
            results: Vector search results
            query_entities: Entities extracted from query
            graph_paths: Relevant graph paths
            entity_centrality: Entity centrality scores
            **kwargs: Additional context

        Returns:
            Re-ranked results
        """
        if not results:
            return []

        # Get query entity IDs
        query_entity_ids = set()
        if query_entities:
            query_entity_ids = {e.id for e in query_entities}

        # Get path entity IDs
        path_entity_ids = set()
        if graph_paths:
            for path in graph_paths:
                for node in path.get("nodes", []):
                    if node.get("id"):
                        path_entity_ids.add(node["id"])

        # Score each result
        scored_results = []

        for result in results:
            # Base vector score
            vector_score = result.score

            # Graph-based score components
            entity_score = 0.0
            path_score = 0.0
            centrality_score = 0.0

            # Check entity overlap
            result_entity_ids = set(result.payload.get("entity_ids", []))

            if result_entity_ids:
                # Entity match score
                query_overlap = len(result_entity_ids & query_entity_ids)
                if query_overlap > 0:
                    entity_score = min(query_overlap * self.entity_boost, 0.5)

                # Path overlap score
                path_overlap = len(result_entity_ids & path_entity_ids)
                if path_overlap > 0:
                    path_score = min(path_overlap * self.path_boost, 0.3)

                # Centrality score
                if entity_centrality:
                    centrality_scores = [
                        entity_centrality.get(eid, 0)
                        for eid in result_entity_ids
                    ]
                    if centrality_scores:
                        centrality_score = np.mean(centrality_scores) * 0.2

            # Combine scores
            graph_score = entity_score + path_score + centrality_score
            final_score = (
                self.vector_weight * vector_score
                + self.graph_weight * graph_score
            )

            # Update result score
            result.score = min(final_score, 1.0)
            scored_results.append(result)

        # Sort by new score
        scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results


class ReciprocalRankFusion(Reranker):
    """
    Reciprocal Rank Fusion (RRF) for combining multiple result lists.

    Useful for combining vector search results from different
    collections or different retrieval methods.
    """

    def __init__(
        self,
        k: int = 60,
        **config: Any,
    ) -> None:
        """
        Initialize RRF re-ranker.

        Args:
            k: RRF constant (default 60)
            **config: Additional configuration
        """
        self.k = k

    async def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        additional_results: list[list[VectorSearchResult]] | None = None,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """
        Re-rank using RRF.

        Args:
            query: Original query
            results: Primary search results
            additional_results: Additional result lists to fuse
            **kwargs: Additional context

        Returns:
            Fused and re-ranked results
        """
        all_result_lists = [results]
        if additional_results:
            all_result_lists.extend(additional_results)

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, VectorSearchResult] = {}

        for result_list in all_result_lists:
            for rank, result in enumerate(result_list, start=1):
                result_id = result.id

                # RRF score contribution
                rrf_score = 1.0 / (self.k + rank)

                if result_id in rrf_scores:
                    rrf_scores[result_id] += rrf_score
                else:
                    rrf_scores[result_id] = rrf_score
                    result_map[result_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final result list
        final_results = []
        for result_id in sorted_ids:
            result = result_map[result_id]
            result.score = rrf_scores[result_id]
            final_results.append(result)

        return final_results


class CrossEncoderReranker(Reranker):
    """
    Re-ranker using cross-encoder model.

    More accurate but slower than bi-encoder approaches.
    Uses a transformer model to score query-document pairs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
        **config: Any,
    ) -> None:
        """
        Initialize cross-encoder re-ranker.

        Args:
            model_name: Cross-encoder model name
            device: Device for computation
            batch_size: Batch size for inference
            **config: Additional configuration
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load cross-encoder model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
            self._initialized = True
            self.logger.info(
                "Cross-encoder initialized",
                model=self.model_name,
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to load cross-encoder: {e}"
            )
            self._model = None

    async def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """
        Re-rank using cross-encoder scores.

        Args:
            query: Original query
            results: Vector search results
            **kwargs: Additional context

        Returns:
            Re-ranked results
        """
        if not results:
            return []

        if not self._initialized:
            await self.initialize()

        if self._model is None:
            # Fallback: return original order
            return results

        try:
            # Prepare query-document pairs
            pairs = [
                (query, r.payload.get("content", ""))
                for r in results
            ]

            # Get cross-encoder scores
            scores = self._model.predict(pairs, batch_size=self.batch_size)

            # Update result scores
            for result, score in zip(results, scores):
                result.score = float(score)

            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)

            return results

        except Exception as e:
            self.logger.warning(f"Cross-encoder reranking failed: {e}")
            return results


class HybridReranker(Reranker):
    """
    Combines multiple re-ranking strategies.

    Pipeline:
    1. Reciprocal Rank Fusion (if multiple result lists)
    2. Graph-aware re-ranking
    3. Optional cross-encoder re-ranking
    """

    def __init__(
        self,
        use_cross_encoder: bool = False,
        graph_weight: float = 0.4,
        vector_weight: float = 0.6,
        **config: Any,
    ) -> None:
        """
        Initialize hybrid re-ranker.

        Args:
            use_cross_encoder: Whether to use cross-encoder
            graph_weight: Weight for graph scores
            vector_weight: Weight for vector scores
            **config: Additional configuration
        """
        self.rrf_reranker = ReciprocalRankFusion()
        self.graph_reranker = GraphAwareReranker(
            graph_weight=graph_weight,
            vector_weight=vector_weight,
        )
        self.use_cross_encoder = use_cross_encoder

        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker()
        else:
            self.cross_encoder = None

    async def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        additional_results: list[list[VectorSearchResult]] | None = None,
        query_entities: list[Entity] | None = None,
        graph_paths: list[dict] | None = None,
        entity_centrality: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """
        Re-rank using hybrid strategy.

        Args:
            query: Original query
            results: Primary search results
            additional_results: Additional result lists
            query_entities: Query entities
            graph_paths: Graph paths
            entity_centrality: Entity centrality scores
            **kwargs: Additional context

        Returns:
            Re-ranked results
        """
        # Step 1: RRF if multiple result lists
        if additional_results:
            results = await self.rrf_reranker.rerank(
                query=query,
                results=results,
                additional_results=additional_results,
            )

        # Step 2: Graph-aware re-ranking
        results = await self.graph_reranker.rerank(
            query=query,
            results=results,
            query_entities=query_entities,
            graph_paths=graph_paths,
            entity_centrality=entity_centrality,
        )

        # Step 3: Cross-encoder (optional)
        if self.cross_encoder and self.use_cross_encoder:
            results = await self.cross_encoder.rerank(
                query=query,
                results=results,
            )

        return results
