"""
Tests for reasoning engines.

Covers:
- GraphAwareReasoner (prompt formatting, response parsing, source building)
- MultiStepReasoner (query decomposition, synthesis, simple query detection)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import (
    Chunk,
    ChunkMetadata,
    Entity,
    EntityType,
    GraphPath,
    Modality,
    QueryResult,
    RetrievalResult,
    SourceSnippet,
)
from src.llm.reasoning import GraphAwareReasoner, MultiStepReasoner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ollama():
    client = MagicMock()
    client.model = "mistral:7b-instruct"
    client.generate = AsyncMock(return_value="Test answer")
    return client


@pytest.fixture
def reasoner(mock_ollama):
    return GraphAwareReasoner(
        ollama_client=mock_ollama,
        model="mistral:7b-instruct",
        max_tokens=1024,
        temperature=0.3,
    )


def _make_retrieval_result(
    num_chunks=2,
    num_entities=1,
    num_paths=0,
):
    chunks = []
    for i in range(num_chunks):
        chunk = Chunk(
            id=f"chunk-{i}",
            content=f"This is test content for chunk {i}. It discusses revenue growth.",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(
                doc_id="doc-1",
                page_number=i + 1,
                section="Financial Summary",
                source_file="report.pdf",
            ),
            entity_ids=[f"e{i}"],
        )
        chunks.append(chunk)

    entities = []
    for i in range(num_entities):
        entity = Entity(
            id=f"e{i}",
            name=f"Entity {i}",
            normalized_name=f"entity {i}",
            entity_type=EntityType.COMPANY,
            attributes={"sector": "Technology"},
        )
        entities.append(entity)

    paths = []
    for i in range(num_paths):
        path = GraphPath(
            nodes=[{"id": "e0"}, {"id": "e1"}],
            edges=[{"type": "ACQUIRED"}],
            path_text=f"Entity 0 acquired Entity 1",
            relevance_score=0.9,
        )
        paths.append(path)

    return RetrievalResult(
        chunks=chunks,
        entities=entities,
        graph_paths=paths,
        vector_scores={f"chunk-{i}": 0.9 - i * 0.1 for i in range(num_chunks)},
        graph_scores={},
        retrieval_time_ms=150.0,
    )


# =============================================================================
# GraphAwareReasoner Tests
# =============================================================================


class TestGraphAwareReasonerInit:
    def test_default_params(self, mock_ollama):
        reasoner = GraphAwareReasoner(ollama_client=mock_ollama)
        assert reasoner.model is None
        assert reasoner.max_tokens == 1024
        assert reasoner.temperature == 0.3
        assert reasoner.include_reasoning_chain is True

    def test_custom_params(self, mock_ollama):
        reasoner = GraphAwareReasoner(
            ollama_client=mock_ollama,
            model="llama2:7b",
            max_tokens=2048,
            temperature=0.7,
            include_reasoning_chain=False,
        )
        assert reasoner.model == "llama2:7b"
        assert reasoner.max_tokens == 2048
        assert reasoner.include_reasoning_chain is False


class TestGraphAwareReasonerReason:
    async def test_basic_reasoning(self, reasoner, mock_ollama):
        mock_ollama.generate.return_value = (
            "**Answer**: Revenue was $100M\n"
            "**Reasoning**: Based on the financial report\n"
            "**Confidence**: High"
        )

        retrieval = _make_retrieval_result()
        result = await reasoner.reason("What is the revenue?", retrieval)

        assert isinstance(result, QueryResult)
        assert result.query == "What is the revenue?"
        assert "Revenue" in result.answer or "100M" in result.answer
        assert result.confidence == 0.9  # High
        assert result.latency_ms >= 0

    async def test_reasoning_calls_ollama(self, reasoner, mock_ollama):
        retrieval = _make_retrieval_result()
        await reasoner.reason("test query", retrieval)

        mock_ollama.generate.assert_called_once()
        call_kwargs = mock_ollama.generate.call_args.kwargs
        assert "test query" in call_kwargs["prompt"]
        assert call_kwargs["model"] == "mistral:7b-instruct"
        assert call_kwargs["system"] == GraphAwareReasoner.SYSTEM_PROMPT

    async def test_reasoning_includes_sources(self, reasoner, mock_ollama):
        retrieval = _make_retrieval_result(num_chunks=3)
        result = await reasoner.reason("test query", retrieval)

        assert len(result.sources) == 3
        assert all(isinstance(s, SourceSnippet) for s in result.sources)

    async def test_reasoning_includes_metadata(self, reasoner, mock_ollama):
        retrieval = _make_retrieval_result()
        result = await reasoner.reason("test query", retrieval)

        assert result.metadata["model"] == "mistral:7b-instruct"
        assert result.metadata["retrieval_time_ms"] == 150.0
        assert result.metadata["chunks_used"] == 2
        assert result.metadata["entities_used"] == 1

    async def test_reasoning_handles_llm_failure(self, reasoner, mock_ollama):
        mock_ollama.generate.side_effect = Exception("LLM error")

        retrieval = _make_retrieval_result()
        result = await reasoner.reason("test query", retrieval)

        assert "unable to generate" in result.answer.lower()

    async def test_reasoning_without_chain(self, mock_ollama):
        reasoner = GraphAwareReasoner(
            ollama_client=mock_ollama,
            include_reasoning_chain=False,
        )
        mock_ollama.generate.return_value = "Simple answer"

        retrieval = _make_retrieval_result()
        result = await reasoner.reason("test query", retrieval)

        assert result.reasoning_chain is None


# =============================================================================
# Prompt Formatting Tests
# =============================================================================


class TestFormatDocuments:
    def test_empty_chunks(self, reasoner):
        result = reasoner._format_documents([])
        assert "No relevant documents" in result

    def test_formats_with_source_info(self, reasoner):
        chunks = [
            Chunk(
                content="Revenue grew 25%",
                modality=Modality.TEXT,
                metadata=ChunkMetadata(
                    doc_id="d1",
                    source_file="report.pdf",
                    page_number=5,
                    section="Financials",
                ),
            )
        ]
        result = reasoner._format_documents(chunks)

        assert "Source 1" in result
        assert "report.pdf" in result
        assert "Page: 5" in result
        assert "Financials" in result
        assert "Revenue grew 25%" in result

    def test_truncates_long_content(self, reasoner):
        chunks = [
            Chunk(
                content="x" * 2000,
                modality=Modality.TEXT,
                metadata=ChunkMetadata(doc_id="d1"),
            )
        ]
        result = reasoner._format_documents(chunks)
        # Content should be truncated to 1000 chars
        assert len(result) < 2000


class TestFormatEntities:
    def test_empty_entities(self, reasoner):
        result = reasoner._format_entities([])
        assert "No specific entities" in result

    def test_formats_entities(self, reasoner):
        entities = [
            Entity(
                name="Apple Inc.",
                normalized_name="apple inc.",
                entity_type=EntityType.COMPANY,
                attributes={"ticker": "AAPL", "sector": "Technology"},
            )
        ]
        result = reasoner._format_entities(entities)

        assert "Apple Inc." in result
        assert "COMPANY" in result
        assert "ticker: AAPL" in result

    def test_limits_to_15_entities(self, reasoner):
        entities = [
            Entity(
                name=f"Entity {i}",
                normalized_name=f"entity {i}",
                entity_type=EntityType.COMPANY,
            )
            for i in range(20)
        ]
        result = reasoner._format_entities(entities)

        # Should only include 15
        assert result.count("- ") == 15


class TestFormatGraphPaths:
    def test_empty_paths(self, reasoner):
        result = reasoner._format_graph_paths([])
        assert "No entity relationships" in result

    def test_formats_paths(self, reasoner):
        paths = [
            GraphPath(
                nodes=[],
                edges=[],
                path_text="Apple acquired Beats",
                relevance_score=0.9,
            )
        ]
        result = reasoner._format_graph_paths(paths)
        assert "Apple acquired Beats" in result

    def test_limits_to_5_paths(self, reasoner):
        paths = [
            GraphPath(
                nodes=[],
                edges=[],
                path_text=f"Path {i}",
                relevance_score=0.5,
            )
            for i in range(10)
        ]
        result = reasoner._format_graph_paths(paths)
        # Numbered 1-5
        assert "5." in result
        assert "6." not in result


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestParseResponse:
    def test_structured_response(self, reasoner):
        response = (
            "**Answer**: The revenue was $100M.\n"
            "**Reasoning**: Based on the Q4 2023 financial report.\n"
            "**Confidence**: High"
        )
        answer, reasoning, confidence = reasoner._parse_response(response)

        assert "100M" in answer
        assert "Q4 2023" in reasoning
        assert confidence == 0.9

    def test_medium_confidence(self, reasoner):
        response = "**Answer**: Maybe\n**Confidence**: Medium"
        _, _, confidence = reasoner._parse_response(response)
        assert confidence == 0.7

    def test_low_confidence(self, reasoner):
        response = "**Answer**: Unsure\n**Confidence**: Low"
        _, _, confidence = reasoner._parse_response(response)
        assert confidence == 0.4

    def test_unstructured_response(self, reasoner):
        response = "This is just a plain text answer without any structure."
        answer, reasoning, confidence = reasoner._parse_response(response)

        assert answer == response
        assert reasoning is None
        assert confidence == 0.7  # Default

    def test_colon_format(self, reasoner):
        response = "Answer: The revenue is $50M\nReasoning: From page 3\nConfidence: High"
        answer, reasoning, confidence = reasoner._parse_response(response)

        assert "50M" in answer
        assert "page 3" in reasoning
        assert confidence == 0.9

    def test_multiline_answer(self, reasoner):
        response = (
            "**Answer**: Revenue was $100M.\n"
            "This represents a 25% increase.\n"
            "**Reasoning**: Based on report.\n"
            "**Confidence**: High"
        )
        answer, reasoning, confidence = reasoner._parse_response(response)
        assert "25% increase" in answer


# =============================================================================
# Source Building Tests
# =============================================================================


class TestBuildSources:
    def test_builds_sources_from_retrieval(self, reasoner):
        retrieval = _make_retrieval_result(num_chunks=3)
        sources = reasoner._build_sources(retrieval)

        assert len(sources) == 3
        assert all(isinstance(s, SourceSnippet) for s in sources)

    def test_source_includes_metadata(self, reasoner):
        retrieval = _make_retrieval_result(num_chunks=1)
        sources = reasoner._build_sources(retrieval)

        source = sources[0]
        assert source.chunk_id == "chunk-0"
        assert source.modality == Modality.TEXT
        assert source.page_number == 1
        assert source.section == "Financial Summary"
        assert source.doc_id == "doc-1"

    def test_source_truncates_content(self, reasoner):
        retrieval = _make_retrieval_result()
        retrieval.chunks[0].content = "x" * 1000
        sources = reasoner._build_sources(retrieval)

        assert len(sources[0].content) <= 500


# =============================================================================
# MultiStepReasoner Tests
# =============================================================================


class TestMultiStepReasoner:
    @pytest.fixture
    def multi_reasoner(self, mock_ollama, reasoner):
        return MultiStepReasoner(
            ollama_client=mock_ollama,
            base_reasoner=reasoner,
            max_sub_questions=3,
        )

    async def test_simple_query_delegates_to_base(self, multi_reasoner, mock_ollama):
        mock_ollama.generate.return_value = "Simple answer"

        retrieval = _make_retrieval_result()
        result = await multi_reasoner.reason("What is the revenue?", retrieval)

        assert isinstance(result, QueryResult)

    async def test_complex_query_decomposes(self, multi_reasoner, mock_ollama):
        # First call: decompose query → sub-questions
        # Subsequent calls: answer sub-questions and synthesize
        call_count = [0]

        async def mock_generate(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Decomposition
                return "1. What is Apple's revenue?\n2. What is Google's revenue?"
            elif call_count[0] <= 3:
                # Sub-question answers
                return "**Answer**: $100M\n**Confidence**: High"
            else:
                # Synthesis
                return "Combined answer about both companies"

        mock_ollama.generate = AsyncMock(side_effect=mock_generate)

        retrieval = _make_retrieval_result()
        result = await multi_reasoner.reason(
            "Compare Apple and Google's revenue",
            retrieval,
        )

        assert isinstance(result, QueryResult)
        assert result.metadata.get("reasoning_type") == "multi_step"

    async def test_decomposition_failure_falls_back(self, multi_reasoner, mock_ollama):
        call_count = [0]

        async def mock_generate(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Decomposition failed")
            return "Fallback answer"

        mock_ollama.generate = AsyncMock(side_effect=mock_generate)

        retrieval = _make_retrieval_result()
        result = await multi_reasoner.reason(
            "Compare X and Y",
            retrieval,
        )

        assert isinstance(result, QueryResult)


class TestIsSimpleQuery:
    @pytest.fixture
    def multi_reasoner(self, mock_ollama, reasoner):
        return MultiStepReasoner(
            ollama_client=mock_ollama,
            base_reasoner=reasoner,
        )

    def test_simple_queries(self, multi_reasoner):
        assert multi_reasoner._is_simple_query("What is the revenue?") is True
        assert multi_reasoner._is_simple_query("When was it founded?") is True
        assert multi_reasoner._is_simple_query("Who is the CEO?") is True

    def test_complex_queries(self, multi_reasoner):
        assert multi_reasoner._is_simple_query("Compare Apple and Google") is False
        assert multi_reasoner._is_simple_query("What is the difference between X and Y?") is False
        assert multi_reasoner._is_simple_query("How does this relate to that?") is False
        assert multi_reasoner._is_simple_query("What is the impact of climate change?") is False
        assert multi_reasoner._is_simple_query("Why does revenue or profit matter?") is False


class TestDecomposeQuery:
    @pytest.fixture
    def multi_reasoner(self, mock_ollama, reasoner):
        return MultiStepReasoner(
            ollama_client=mock_ollama,
            base_reasoner=reasoner,
        )

    async def test_parses_numbered_list(self, multi_reasoner, mock_ollama):
        mock_ollama.generate.return_value = (
            "1. What is Apple's revenue?\n"
            "2. What is Google's revenue?\n"
            "3. How do they compare?"
        )

        questions = await multi_reasoner._decompose_query("Compare Apple and Google")
        assert len(questions) == 3

    async def test_parses_bulleted_list(self, multi_reasoner, mock_ollama):
        mock_ollama.generate.return_value = (
            "- What is X?\n"
            "- What is Y?\n"
        )

        questions = await multi_reasoner._decompose_query("What is X and Y?")
        assert len(questions) == 2

    async def test_handles_failure(self, multi_reasoner, mock_ollama):
        mock_ollama.generate.side_effect = Exception("Failed")

        questions = await multi_reasoner._decompose_query("test")
        assert questions == []

    async def test_filters_non_questions(self, multi_reasoner, mock_ollama):
        mock_ollama.generate.return_value = (
            "Here are the sub-questions:\n"
            "1. What is revenue?\n"
            "2. Not a question\n"
            "3. How did it change?"
        )

        questions = await multi_reasoner._decompose_query("test")
        # Only lines with "?" are kept
        assert len(questions) == 2


class TestSynthesizeAnswers:
    @pytest.fixture
    def multi_reasoner(self, mock_ollama, reasoner):
        return MultiStepReasoner(
            ollama_client=mock_ollama,
            base_reasoner=reasoner,
        )

    async def test_synthesis(self, multi_reasoner, mock_ollama):
        mock_ollama.generate.return_value = "Combined answer about both topics."

        sub_answers = [
            ("What is X?", "X is 100"),
            ("What is Y?", "Y is 200"),
        ]

        result = await multi_reasoner._synthesize_answers("Compare X and Y", sub_answers)
        assert "Combined answer" in result

    async def test_synthesis_failure_concatenates(self, multi_reasoner, mock_ollama):
        mock_ollama.generate.side_effect = Exception("Failed")

        sub_answers = [
            ("Q1?", "A1"),
            ("Q2?", "A2"),
        ]

        result = await multi_reasoner._synthesize_answers("test", sub_answers)
        assert "A1" in result
        assert "A2" in result
