"""
Query service for handling user questions.

Orchestrates:
- Hybrid retrieval
- LLM reasoning
- Answer generation
- Caching
"""

import time
from typing import Any

from src.core.logging import LoggerMixin, log_operation
from src.core.types import QueryResult, RetrievalResult
from src.llm.reasoning import GraphAwareReasoner, MultiStepReasoner
from src.retrieval.hybrid_retriever import HybridRetriever


class QueryService(LoggerMixin):
    """
    Service for handling question answering queries.

    Provides a unified interface for:
    - Simple queries (single-step reasoning)
    - Complex queries (multi-step reasoning)
    - Entity-focused queries
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reasoner: GraphAwareReasoner,
        multi_step_reasoner: MultiStepReasoner | None = None,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        **config: Any,
    ) -> None:
        """
        Initialize query service.

        Args:
            retriever: Hybrid retriever
            reasoner: Graph-aware reasoner
            multi_step_reasoner: Optional multi-step reasoner
            enable_caching: Enable result caching
            cache_ttl: Cache TTL in seconds
            **config: Additional configuration
        """
        self.retriever = retriever
        self.reasoner = reasoner
        self.multi_step_reasoner = multi_step_reasoner
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Simple in-memory cache
        self._cache: dict[str, tuple[QueryResult, float]] = {}

    async def ask(
        self,
        question: str,
        top_k: int = 5,
        filter_doc_ids: list[str] | None = None,
        use_multi_step: bool = False,
    ) -> QueryResult:
        """
        Answer a question.

        Args:
            question: User question
            top_k: Number of results
            filter_doc_ids: Filter to specific documents
            use_multi_step: Use multi-step reasoning

        Returns:
            QueryResult with answer and sources
        """
        start_time = time.time()

        # Check cache
        cache_key = f"{question}:{top_k}:{filter_doc_ids}"
        if self.enable_caching:
            cached = self._get_cached(cache_key)
            if cached:
                self.logger.debug("Cache hit", question=question[:50])
                return cached

        self.logger.info("Processing query", question=question[:100])

        try:
            # Retrieve context
            retrieval_result = await self.retriever.retrieve(
                query=question,
                filter_doc_ids=filter_doc_ids,
            )

            # Select reasoner
            if use_multi_step and self.multi_step_reasoner:
                reasoner = self.multi_step_reasoner
            else:
                reasoner = self.reasoner

            # Generate answer
            result = await reasoner.reason(
                query=question,
                retrieval_result=retrieval_result,
            )

            # Update latency
            result.latency_ms = (time.time() - start_time) * 1000

            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, result)

            log_operation(
                "query",
                success=True,
                duration_ms=result.latency_ms,
                question_length=len(question),
                sources=len(result.sources),
            )

            return result

        except Exception as e:
            self.logger.error("Query failed", question=question[:100], error=str(e))
            raise

    async def ask_about_entities(
        self,
        entity_names: list[str],
        question: str | None = None,
    ) -> QueryResult:
        """
        Answer questions about specific entities.

        Args:
            entity_names: Entity names to focus on
            question: Optional additional question

        Returns:
            QueryResult focused on entities
        """
        start_time = time.time()

        try:
            # Retrieve entity-focused context
            retrieval_result = await self.retriever.retrieve_by_entities(
                entity_names=entity_names,
                expand_graph=True,
            )

            # Generate question if not provided
            if not question:
                question = f"What information is available about {', '.join(entity_names)}?"

            # Generate answer
            result = await self.reasoner.reason(
                query=question,
                retrieval_result=retrieval_result,
            )

            result.latency_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            self.logger.error("Entity query failed", entities=entity_names, error=str(e))
            raise

    async def get_related_questions(
        self,
        question: str,
        n: int = 5,
    ) -> list[str]:
        """
        Generate related questions.

        Args:
            question: Original question
            n: Number of suggestions

        Returns:
            List of related questions
        """
        # This would use the LLM to generate related questions
        # For now, return empty list
        return []

    def _get_cached(self, key: str) -> QueryResult | None:
        """Get cached result if valid."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self._cache[key]
        return None

    def _cache_result(self, key: str, result: QueryResult) -> None:
        """Cache a result."""
        self._cache[key] = (result, time.time())

        # Simple cache cleanup - remove old entries
        if len(self._cache) > 1000:
            current_time = time.time()
            expired_keys = [
                k for k, (_, ts) in self._cache.items()
                if current_time - ts > self.cache_ttl
            ]
            for k in expired_keys:
                del self._cache[k]

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
