"""
Explanation service for answer explainability.

Provides detailed explanations of:
- How answers were derived
- Evidence chains
- Graph reasoning paths
- Confidence factors
"""

from typing import Any

from src.core.logging import LoggerMixin
from src.core.types import GraphPath, QueryResult, SourceSnippet
from src.llm.ollama_client import OllamaClient


class ExplanationService(LoggerMixin):
    """
    Service for generating answer explanations.

    Provides explainability features required for
    transparent, trustworthy QA systems.
    """

    EXPLANATION_PROMPT = """Analyze the following question, answer, and supporting evidence to provide a detailed explanation.

Question: {question}

Answer: {answer}

Supporting Sources:
{sources}

Graph Reasoning Paths:
{paths}

Please provide:
1. **Evidence Summary**: Key facts from the sources that support the answer
2. **Reasoning Chain**: How the answer was logically derived from the evidence
3. **Graph Connections**: How entity relationships contributed to the answer
4. **Confidence Analysis**: Factors that increase or decrease confidence
5. **Limitations**: What the answer might be missing or uncertain about

Detailed Explanation:"""

    def __init__(
        self,
        ollama_client: OllamaClient,
        **config: Any,
    ) -> None:
        """
        Initialize explanation service.

        Args:
            ollama_client: Ollama client for LLM
            **config: Additional configuration
        """
        self.ollama_client = ollama_client

    async def explain_answer(
        self,
        query_result: QueryResult,
    ) -> dict[str, Any]:
        """
        Generate detailed explanation for an answer.

        Args:
            query_result: Query result to explain

        Returns:
            Explanation dictionary
        """
        # Format sources
        sources_text = self._format_sources(query_result.sources)

        # Format graph paths
        paths_text = self._format_paths(query_result.graph_paths)

        # Generate explanation
        prompt = self.EXPLANATION_PROMPT.format(
            question=query_result.query,
            answer=query_result.answer,
            sources=sources_text,
            paths=paths_text,
        )

        try:
            explanation = await self.ollama_client.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,
            )
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            explanation = "Unable to generate detailed explanation."

        # Parse and structure the explanation
        return {
            "explanation": explanation,
            "evidence_summary": self._extract_evidence_summary(query_result),
            "confidence_factors": self._calculate_confidence_factors(query_result),
            "graph_summary": self._summarize_graph_paths(query_result.graph_paths),
        }

    def _format_sources(self, sources: list[SourceSnippet]) -> str:
        """Format sources for prompt."""
        if not sources:
            return "No specific sources available."

        parts = []
        for i, source in enumerate(sources[:5], 1):
            info = f"Source {i}"
            if source.page_number:
                info += f" (Page {source.page_number})"
            if source.section:
                info += f" [{source.section}]"

            parts.append(f"[{info}]\n{source.content[:300]}...")

        return "\n\n".join(parts)

    def _format_paths(self, paths: list[GraphPath]) -> str:
        """Format graph paths for prompt."""
        if not paths:
            return "No graph paths found."

        return "\n".join(
            f"{i}. {p.path_text}"
            for i, p in enumerate(paths[:5], 1)
        )

    def _extract_evidence_summary(
        self,
        query_result: QueryResult,
    ) -> list[dict[str, Any]]:
        """Extract key evidence points."""
        evidence = []

        for source in query_result.sources[:5]:
            evidence.append({
                "content": source.content[:200],
                "relevance": source.relevance_score,
                "source_type": source.modality.value,
                "location": {
                    "page": source.page_number,
                    "section": source.section,
                },
            })

        return evidence

    def _calculate_confidence_factors(
        self,
        query_result: QueryResult,
    ) -> dict[str, float]:
        """Calculate confidence contributing factors."""
        factors = {}

        # Source coverage factor
        if query_result.sources:
            avg_relevance = sum(s.relevance_score for s in query_result.sources) / len(
                query_result.sources
            )
            factors["source_relevance"] = round(avg_relevance, 2)
        else:
            factors["source_relevance"] = 0.0

        # Source count factor
        source_count = len(query_result.sources)
        if source_count >= 3:
            factors["source_coverage"] = 0.9
        elif source_count >= 1:
            factors["source_coverage"] = 0.6
        else:
            factors["source_coverage"] = 0.2

        # Graph support factor
        path_count = len(query_result.graph_paths)
        if path_count >= 2:
            factors["graph_support"] = 0.85
        elif path_count >= 1:
            factors["graph_support"] = 0.6
        else:
            factors["graph_support"] = 0.4

        # Reasoning chain factor
        if query_result.reasoning_chain and len(query_result.reasoning_chain) > 100:
            factors["reasoning_depth"] = 0.8
        else:
            factors["reasoning_depth"] = 0.5

        return factors

    def _summarize_graph_paths(
        self,
        paths: list[GraphPath],
    ) -> dict[str, Any]:
        """Summarize graph path information."""
        if not paths:
            return {"connected": False, "path_count": 0}

        # Collect all entity types
        entity_types = set()
        relation_types = set()

        for path in paths:
            for node in path.nodes:
                if "type" in node:
                    entity_types.add(node["type"])
            for edge in path.edges:
                if "type" in edge:
                    relation_types.add(edge["type"])

        return {
            "connected": True,
            "path_count": len(paths),
            "avg_path_length": sum(len(p.edges) for p in paths) / len(paths),
            "entity_types": list(entity_types),
            "relation_types": list(relation_types),
        }

    async def compare_answers(
        self,
        question: str,
        answer1: str,
        answer2: str,
    ) -> dict[str, Any]:
        """
        Compare two answers for the same question.

        Useful for A/B testing or debugging.

        Args:
            question: Original question
            answer1: First answer
            answer2: Second answer

        Returns:
            Comparison analysis
        """
        prompt = f"""Compare these two answers to the same question:

Question: {question}

Answer 1: {answer1}

Answer 2: {answer2}

Analyze:
1. Which answer is more accurate?
2. Which provides more relevant information?
3. What are the key differences?
4. Are there any contradictions?

Comparison:"""

        try:
            comparison = await self.ollama_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
            )
        except Exception as e:
            comparison = f"Comparison failed: {e}"

        return {
            "comparison": comparison,
            "answer1_length": len(answer1),
            "answer2_length": len(answer2),
        }
