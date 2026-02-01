"""
Tests for relation extraction.

Covers pattern-based, co-occurrence, and LLM-based extraction.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import (
    Chunk, ChunkMetadata, Entity, EntityType, Modality, RelationType,
)
from src.kg.relation_extractor import (
    LLMRelationExtractor, PatternRelationExtractor,
)


class TestPatternRelationExtractor:
    def test_init_finance(self):
        ext = PatternRelationExtractor(domain="finance")
        assert ext.domain == "finance"
        assert len(ext.patterns) > 0

    def test_init_healthcare(self):
        ext = PatternRelationExtractor(domain="healthcare")
        assert ext.domain == "healthcare"

    @pytest.mark.asyncio
    async def test_initialize(self):
        ext = PatternRelationExtractor()
        await ext.initialize()
        assert ext._initialized is True

    @pytest.mark.asyncio
    async def test_extract_empty_entities(self):
        ext = PatternRelationExtractor()
        await ext.initialize()
        relations = await ext.extract("Some text", [])
        assert isinstance(relations, list)

    @pytest.mark.asyncio
    async def test_extract_cooccurrence(self):
        ext = PatternRelationExtractor(domain="finance")
        await ext.initialize()

        company = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY, source_chunk_ids=["c1"],
        )
        metric = Entity(
            name="revenue", normalized_name="revenue",
            entity_type=EntityType.METRIC, source_chunk_ids=["c1"],
        )
        relations = await ext.extract("Apple reported revenue", [company, metric])
        cooc = [r for r in relations if r.attributes.get("extraction_method") == "cooccurrence"]
        assert len(cooc) >= 1

    @pytest.mark.asyncio
    async def test_no_cooccurrence_different_chunks(self):
        ext = PatternRelationExtractor(domain="finance")
        await ext.initialize()

        company = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY, source_chunk_ids=["c1"],
        )
        metric = Entity(
            name="revenue", normalized_name="revenue",
            entity_type=EntityType.METRIC, source_chunk_ids=["c2"],
        )
        relations = await ext.extract("text", [company, metric])
        cooc = [r for r in relations if r.attributes.get("extraction_method") == "cooccurrence"]
        assert len(cooc) == 0

    def test_build_entity_lookup(self):
        ext = PatternRelationExtractor()
        # Note: suffix stripping in _build_entity_lookup looks for " inc" without period
        entity = Entity(
            name="Apple Inc", normalized_name="apple inc",
            entity_type=EntityType.COMPANY,
        )
        lookup = ext._build_entity_lookup([entity])
        assert "apple inc" in lookup
        assert "apple" in lookup

    def test_find_entity_direct(self):
        ext = PatternRelationExtractor()
        entity = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        lookup = {"apple": entity}
        assert ext._find_entity("Apple", lookup, [EntityType.COMPANY]) is entity

    def test_find_entity_wrong_type(self):
        ext = PatternRelationExtractor()
        entity = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        lookup = {"apple": entity}
        assert ext._find_entity("Apple", lookup, [EntityType.PERSON]) is None

    @pytest.mark.asyncio
    async def test_extract_from_chunk(self):
        ext = PatternRelationExtractor()
        await ext.initialize()
        chunk = Chunk(
            id="chunk-1", content="Apple reported revenue",
            modality=Modality.TEXT, metadata=ChunkMetadata(doc_id="doc-1"),
        )
        entities = [Entity(name="Apple", normalized_name="apple", entity_type=EntityType.COMPANY)]
        relations = await ext.extract_from_chunk(chunk, entities)
        for rel in relations:
            assert "chunk-1" in rel.source_chunk_ids

    @pytest.mark.asyncio
    async def test_cleanup(self):
        ext = PatternRelationExtractor()
        await ext.initialize()
        await ext.cleanup()
        assert ext._initialized is False


class TestLLMRelationExtractor:
    def test_init_finance(self):
        ext = LLMRelationExtractor(domain="finance")
        assert RelationType.FILED in ext.valid_relations

    def test_init_healthcare(self):
        ext = LLMRelationExtractor(domain="healthcare")
        assert RelationType.HAS_CONDITION in ext.valid_relations

    @pytest.mark.asyncio
    async def test_extract_too_few_entities(self):
        ext = LLMRelationExtractor(ollama_client=MagicMock())
        ext._initialized = True
        result = await ext.extract("text", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_single_entity(self):
        ext = LLMRelationExtractor(ollama_client=MagicMock())
        ext._initialized = True
        e = Entity(name="A", normalized_name="a", entity_type=EntityType.COMPANY)
        result = await ext.extract("text", [e])
        assert result == []

    def test_parse_no_relations(self):
        ext = LLMRelationExtractor(domain="finance")
        assert ext._parse_llm_response("NO_RELATIONS", []) == []

    def test_parse_valid_relation(self):
        ext = LLMRelationExtractor(domain="finance")
        entities = [
            Entity(name="Apple", normalized_name="apple", entity_type=EntityType.COMPANY),
            Entity(name="10-K", normalized_name="10-k", entity_type=EntityType.FILING),
        ]
        response = "RELATION: Apple -> FILED -> 10-K"
        result = ext._parse_llm_response(response, entities)
        assert len(result) == 1
        assert result[0].relation_type == RelationType.FILED

    def test_parse_unknown_entity(self):
        ext = LLMRelationExtractor(domain="finance")
        entities = [Entity(name="Apple", normalized_name="apple", entity_type=EntityType.COMPANY)]
        result = ext._parse_llm_response("RELATION: Apple -> FILED -> Unknown", entities)
        assert len(result) == 0

    def test_parse_invalid_relation_type(self):
        ext = LLMRelationExtractor(domain="finance")
        entities = [
            Entity(name="Apple", normalized_name="apple", entity_type=EntityType.COMPANY),
            Entity(name="Google", normalized_name="google", entity_type=EntityType.COMPANY),
        ]
        result = ext._parse_llm_response("RELATION: Apple -> FAKE_TYPE -> Google", entities)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_extract_with_mock(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="NO_RELATIONS")
        ext = LLMRelationExtractor(ollama_client=mock_client, domain="finance")
        ext._initialized = True

        entities = [
            Entity(name="A", normalized_name="a", entity_type=EntityType.COMPANY),
            Entity(name="B", normalized_name="b", entity_type=EntityType.COMPANY),
        ]
        result = await ext.extract("A and B", entities)
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_llm_failure(self):
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(side_effect=Exception("LLM error"))
        ext = LLMRelationExtractor(ollama_client=mock_client)
        ext._initialized = True

        entities = [
            Entity(name="A", normalized_name="a", entity_type=EntityType.COMPANY),
            Entity(name="B", normalized_name="b", entity_type=EntityType.COMPANY),
        ]
        result = await ext.extract("text", entities)
        assert result == []
