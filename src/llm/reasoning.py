"""
Reasoning engine for GraphRAG.

Implements:
- Graph-aware prompting
- Chain-of-thought reasoning
- Answer generation with explainability
- Source attribution
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.logging import LoggerMixin, log_operation
from src.core.types import (
    Chunk,
    Entity,
    GraphPath,
    QueryResult,
    RetrievalResult,
    SourceSnippet,
)
from src.llm.ollama_client import OllamaClient


class ReasoningEngine(ABC, LoggerMixin):
    """Abstract base class for reasoning engines."""

    @abstractmethod
    async def reason(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        **kwargs: Any,
    ) -> QueryResult:
        """
        Generate answer with reasoning.

        Args:
            query: User query
            retrieval_result: Retrieved context
            **kwargs: Additional parameters

        Returns:
            Query result with answer and explanation
        """
        pass


class GraphAwareReasoner(ReasoningEngine):
    """
    Graph-aware reasoning engine.

    Uses retrieved chunks, entities, and graph paths
    to generate explainable answers with chain-of-thought.
    """

    # Prompt templates
    SYSTEM_PROMPT = """You are an expert analyst assistant that provides accurate, well-sourced answers.

Your task is to answer questions using the provided context, which includes:
1. Relevant text excerpts from documents
2. Entities mentioned in the documents
3. Relationships between entities (graph paths)

Guidelines:
- Base your answer ONLY on the provided context
- Cite specific sources when making claims
- If the context doesn't contain enough information, say so
- Use the entity relationships to understand connections
- Explain your reasoning step by step
- Be concise but thorough"""

    ANSWER_PROMPT = """Context Information:

## Retrieved Documents:
{documents}

## Relevant Entities:
{entities}

## Entity Relationships (Graph Paths):
{graph_paths}

---

Question: {query}

Please answer the question based on the context above. Structure your response as:

1. **Answer**: Your direct answer to the question
2. **Reasoning**: Brief explanation of how you arrived at this answer, referencing specific sources
3. **Confidence**: High/Medium/Low based on how well the context supports the answer

Answer:"""

    EXTRACTION_PROMPT = """Based on this context and question, extract the key information needed to answer.

Context:
{context}

Question: {query}

Extract:
1. Key facts relevant to the question
2. Entities involved
3. Relationships between entities
4. Any numerical data or dates

Extracted Information:"""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        include_reasoning_chain: bool = True,
        **config: Any,
    ) -> None:
        """
        Initialize graph-aware reasoner.

        Args:
            ollama_client: Ollama client for LLM inference
            model: Model to use (default: client's default)
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            include_reasoning_chain: Include chain-of-thought
            **config: Additional configuration
        """
        self.ollama_client = ollama_client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.include_reasoning_chain = include_reasoning_chain

    async def reason(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        **kwargs: Any,
    ) -> QueryResult:
        """
        Generate answer using graph-aware reasoning.

        Args:
            query: User query
            retrieval_result: Retrieved context
            **kwargs: Additional parameters

        Returns:
            QueryResult with answer and explanation
        """
        start_time = time.time()

        self.logger.info(
            "Starting reasoning",
            query=query[:100],
            chunks=len(retrieval_result.chunks),
            entities=len(retrieval_result.entities),
        )

        # Format context for prompt
        documents_text = self._format_documents(retrieval_result.chunks)
        entities_text = self._format_entities(retrieval_result.entities)
        paths_text = self._format_graph_paths(retrieval_result.graph_paths)

        # Build prompt
        prompt = self.ANSWER_PROMPT.format(
            documents=documents_text,
            entities=entities_text,
            graph_paths=paths_text,
            query=query,
        )

        # Generate answer
        try:
            response = await self.ollama_client.generate(
                prompt=prompt,
                model=self.model,
                system=self.SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            response = "I apologize, but I was unable to generate an answer due to a technical issue."

        # Parse response
        answer, reasoning, confidence = self._parse_response(response)

        # Build source snippets
        sources = self._build_sources(retrieval_result)

        latency_ms = (time.time() - start_time) * 1000

        log_operation(
            "reasoning",
            success=True,
            duration_ms=latency_ms,
            query_length=len(query),
            answer_length=len(answer),
        )

        return QueryResult(
            query=query,
            answer=answer,
            confidence=confidence,
            sources=sources,
            graph_paths=retrieval_result.graph_paths,
            reasoning_chain=reasoning if self.include_reasoning_chain else None,
            metadata={
                "model": self.model or self.ollama_client.model,
                "retrieval_time_ms": retrieval_result.retrieval_time_ms,
                "chunks_used": len(retrieval_result.chunks),
                "entities_used": len(retrieval_result.entities),
            },
            latency_ms=latency_ms,
        )

    def _format_documents(self, chunks: list[Chunk]) -> str:
        """Format chunks for prompt."""
        if not chunks:
            return "No relevant documents found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = []
            if chunk.metadata.source_file:
                source_info.append(f"File: {chunk.metadata.source_file}")
            if chunk.metadata.page_number:
                source_info.append(f"Page: {chunk.metadata.page_number}")
            if chunk.metadata.section:
                source_info.append(f"Section: {chunk.metadata.section}")

            source_str = " | ".join(source_info) if source_info else "Unknown source"

            parts.append(f"[Source {i}: {source_str}]\n{chunk.content[:1000]}")

        return "\n\n".join(parts)

    def _format_entities(self, entities: list[Entity]) -> str:
        """Format entities for prompt."""
        if not entities:
            return "No specific entities identified."

        parts = []
        for entity in entities[:15]:  # Limit entities
            attrs = []
            if entity.attributes:
                for k, v in list(entity.attributes.items())[:3]:
                    attrs.append(f"{k}: {v}")
            attrs_str = " | ".join(attrs) if attrs else ""

            parts.append(
                f"- {entity.name} ({entity.entity_type.value})"
                + (f" [{attrs_str}]" if attrs_str else "")
            )

        return "\n".join(parts)

    def _format_graph_paths(self, paths: list[GraphPath]) -> str:
        """Format graph paths for prompt."""
        if not paths:
            return "No entity relationships found."

        parts = []
        for i, path in enumerate(paths[:5], 1):  # Limit paths
            parts.append(f"{i}. {path.path_text}")

        return "\n".join(parts)

    def _parse_response(
        self,
        response: str,
    ) -> tuple[str, str | None, float]:
        """
        Parse LLM response into answer, reasoning, and confidence.

        Returns:
            Tuple of (answer, reasoning, confidence_score)
        """
        answer = response
        reasoning = None
        confidence = 0.7  # Default medium confidence

        # Try to parse structured response
        lines = response.split("\n")

        answer_lines = []
        reasoning_lines = []
        current_section = "answer"

        for line in lines:
            line_lower = line.lower().strip()

            if line_lower.startswith("**answer**") or line_lower.startswith("answer:"):
                current_section = "answer"
                # Extract content after header
                content = line.split(":", 1)[-1].strip()
                if content:
                    answer_lines.append(content)
            elif line_lower.startswith("**reasoning**") or line_lower.startswith("reasoning:"):
                current_section = "reasoning"
                content = line.split(":", 1)[-1].strip()
                if content:
                    reasoning_lines.append(content)
            elif line_lower.startswith("**confidence**") or line_lower.startswith("confidence:"):
                current_section = "confidence"
                content = line.split(":", 1)[-1].strip().lower()
                if "high" in content:
                    confidence = 0.9
                elif "medium" in content:
                    confidence = 0.7
                elif "low" in content:
                    confidence = 0.4
            elif current_section == "answer" and line.strip():
                answer_lines.append(line)
            elif current_section == "reasoning" and line.strip():
                reasoning_lines.append(line)

        if answer_lines:
            answer = "\n".join(answer_lines).strip()
        if reasoning_lines:
            reasoning = "\n".join(reasoning_lines).strip()

        return answer, reasoning, confidence

    def _build_sources(self, retrieval_result: RetrievalResult) -> list[SourceSnippet]:
        """Build source snippets from retrieval result."""
        sources = []

        for chunk in retrieval_result.chunks:
            score = retrieval_result.vector_scores.get(chunk.id, 0.5)

            source = SourceSnippet(
                chunk_id=chunk.id,
                content=chunk.content[:500],  # Truncate for response
                modality=chunk.modality,
                relevance_score=score,
                page_number=chunk.metadata.page_number,
                section=chunk.metadata.section,
                doc_id=chunk.metadata.doc_id,
            )
            sources.append(source)

        return sources


class MultiStepReasoner(ReasoningEngine):
    """
    Multi-step reasoning for complex queries.

    Decomposes complex questions into sub-questions,
    answers each, and synthesizes the final answer.
    """

    DECOMPOSITION_PROMPT = """Given this complex question, break it down into simpler sub-questions that can be answered individually.

Question: {query}

List 2-4 sub-questions that, when answered, will help answer the main question:"""

    SYNTHESIS_PROMPT = """Based on the following sub-questions and their answers, synthesize a comprehensive answer to the main question.

Main Question: {query}

Sub-Questions and Answers:
{sub_answers}

Provide a synthesized answer that combines all the information:"""

    def __init__(
        self,
        ollama_client: OllamaClient,
        base_reasoner: GraphAwareReasoner,
        max_sub_questions: int = 3,
        **config: Any,
    ) -> None:
        """
        Initialize multi-step reasoner.

        Args:
            ollama_client: Ollama client
            base_reasoner: Base reasoner for sub-questions
            max_sub_questions: Maximum sub-questions
            **config: Additional configuration
        """
        self.ollama_client = ollama_client
        self.base_reasoner = base_reasoner
        self.max_sub_questions = max_sub_questions

    async def reason(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        **kwargs: Any,
    ) -> QueryResult:
        """
        Multi-step reasoning for complex queries.

        Args:
            query: User query
            retrieval_result: Retrieved context
            **kwargs: Additional parameters

        Returns:
            QueryResult with synthesized answer
        """
        start_time = time.time()

        # Check if decomposition is needed
        if self._is_simple_query(query):
            # Use base reasoner directly
            return await self.base_reasoner.reason(query, retrieval_result)

        # Decompose query
        sub_questions = await self._decompose_query(query)

        if not sub_questions:
            # Fallback to base reasoner
            return await self.base_reasoner.reason(query, retrieval_result)

        # Answer sub-questions
        sub_answers = []
        for sub_q in sub_questions[:self.max_sub_questions]:
            # Use the same retrieval result for all sub-questions
            result = await self.base_reasoner.reason(sub_q, retrieval_result)
            sub_answers.append((sub_q, result.answer))

        # Synthesize final answer
        final_answer = await self._synthesize_answers(query, sub_answers)

        # Build result
        latency_ms = (time.time() - start_time) * 1000

        return QueryResult(
            query=query,
            answer=final_answer,
            confidence=0.75,  # Multi-step has moderate confidence
            sources=self.base_reasoner._build_sources(retrieval_result),
            graph_paths=retrieval_result.graph_paths,
            reasoning_chain="\n".join(
                f"Sub-Q: {q}\nA: {a}" for q, a in sub_answers
            ),
            metadata={
                "reasoning_type": "multi_step",
                "sub_questions": len(sub_questions),
            },
            latency_ms=latency_ms,
        )

    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple enough for direct answering."""
        # Simple heuristics
        complex_indicators = [
            " and ",
            " or ",
            "compare",
            "difference",
            "relationship",
            "how does",
            "why does",
            "what is the impact",
            "multiple",
        ]

        query_lower = query.lower()
        return not any(ind in query_lower for ind in complex_indicators)

    async def _decompose_query(self, query: str) -> list[str]:
        """Decompose query into sub-questions."""
        prompt = self.DECOMPOSITION_PROMPT.format(query=query)

        try:
            response = await self.ollama_client.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.3,
            )

            # Parse sub-questions
            sub_questions = []
            for line in response.split("\n"):
                line = line.strip()
                if line and (
                    line[0].isdigit()
                    or line.startswith("-")
                    or line.startswith("•")
                ):
                    # Remove numbering/bullets
                    cleaned = line.lstrip("0123456789.-•) ").strip()
                    if cleaned and "?" in cleaned:
                        sub_questions.append(cleaned)

            return sub_questions

        except Exception as e:
            self.logger.warning(f"Query decomposition failed: {e}")
            return []

    async def _synthesize_answers(
        self,
        query: str,
        sub_answers: list[tuple[str, str]],
    ) -> str:
        """Synthesize final answer from sub-answers."""
        sub_answers_text = "\n\n".join(
            f"Q: {q}\nA: {a}" for q, a in sub_answers
        )

        prompt = self.SYNTHESIS_PROMPT.format(
            query=query,
            sub_answers=sub_answers_text,
        )

        try:
            response = await self.ollama_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
            )
            return response.strip()

        except Exception as e:
            self.logger.warning(f"Answer synthesis failed: {e}")
            # Fallback: concatenate sub-answers
            return "\n\n".join(a for _, a in sub_answers)
