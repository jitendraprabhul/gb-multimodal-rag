"""
Tests for NER extraction.

Covers SpaCy NER, Finance NER, Healthcare NER, and edge cases.
"""

from unittest.mock import MagicMock

import pytest

from src.core.types import Chunk, ChunkMetadata, Entity, EntityType, Modality
from src.kg.ner_extractor import (
    FinanceNERExtractor,
    HealthcareNERExtractor,
    SpaCyNERExtractor,
)


class TestSpaCyNERExtractor:
    def test_init(self):
        extractor = SpaCyNERExtractor(model_name="en_core_web_sm")
        assert extractor.model_name == "en_core_web_sm"
        assert extractor._initialized is False

    def test_label_map(self):
        assert SpaCyNERExtractor.LABEL_MAP["PERSON"] == EntityType.PERSON
        assert SpaCyNERExtractor.LABEL_MAP["ORG"] == EntityType.ORGANIZATION
        assert SpaCyNERExtractor.LABEL_MAP["GPE"] == EntityType.LOCATION
        assert SpaCyNERExtractor.LABEL_MAP["MONEY"] == EntityType.MONEY

    @pytest.mark.asyncio
    async def test_extract_empty_text(self):
        extractor = SpaCyNERExtractor()
        extractor._initialized = True
        result = await extractor.extract("")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_whitespace(self):
        extractor = SpaCyNERExtractor()
        extractor._initialized = True
        result = await extractor.extract("   ")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_with_mock(self):
        extractor = SpaCyNERExtractor()
        extractor._initialized = True

        mock_ent = MagicMock()
        mock_ent.text = "Apple Inc."
        mock_ent.label_ = "ORG"
        mock_ent.start_char = 0
        mock_ent.end_char = 10

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        extractor._nlp = MagicMock(return_value=mock_doc)

        entities = await extractor.extract("Apple Inc. reported earnings.")
        assert len(entities) == 1
        assert entities[0].name == "Apple Inc."
        assert entities[0].entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_extract_unmapped_label_skipped(self):
        extractor = SpaCyNERExtractor()
        extractor._initialized = True

        mock_ent = MagicMock()
        mock_ent.text = "something"
        mock_ent.label_ = "UNKNOWN_LABEL"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        extractor._nlp = MagicMock(return_value=mock_doc)

        entities = await extractor.extract("something happened")
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_from_chunk(self):
        extractor = SpaCyNERExtractor()
        extractor._initialized = True

        mock_ent = MagicMock()
        mock_ent.text = "John"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 4

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        extractor._nlp = MagicMock(return_value=mock_doc)

        chunk = Chunk(
            id="chunk-1",
            content="John is the CEO.",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
        )
        entities = await extractor.extract_from_chunk(chunk)
        assert len(entities) == 1
        assert "chunk-1" in entities[0].source_chunk_ids

    @pytest.mark.asyncio
    async def test_extract_batch(self):
        extractor = SpaCyNERExtractor()
        extractor._initialized = True
        mock_doc = MagicMock()
        mock_doc.ents = []
        extractor._nlp = MagicMock(return_value=mock_doc)

        results = await extractor.extract_batch(["text1", "text2"])
        assert len(results) == 2

    def test_normalize_entity(self):
        extractor = SpaCyNERExtractor()
        assert extractor.normalize_entity("  Hello World  ", EntityType.PERSON) == "hello world"

    @pytest.mark.asyncio
    async def test_cleanup(self):
        extractor = SpaCyNERExtractor()
        extractor._initialized = True
        await extractor.cleanup()
        assert extractor._initialized is False


class TestFinanceNERExtractor:
    def test_init(self):
        extractor = FinanceNERExtractor()
        assert extractor.spacy_model == "en_core_web_sm"

    def test_label_map_finance_types(self):
        assert FinanceNERExtractor.LABEL_MAP["COMPANY"] == EntityType.COMPANY
        assert FinanceNERExtractor.LABEL_MAP["TICKER"] == EntityType.TICKER
        assert FinanceNERExtractor.LABEL_MAP["FILING"] == EntityType.FILING

    def test_extract_patterns_ticker(self):
        extractor = FinanceNERExtractor()
        entities = extractor._extract_patterns("AAPL stock rose today.")
        tickers = [e for e in entities if e.entity_type == EntityType.TICKER]
        assert any(e.name == "AAPL" for e in tickers)

    def test_extract_patterns_common_word_not_ticker(self):
        extractor = FinanceNERExtractor()
        entities = extractor._extract_patterns("I am here.")
        tickers = [e for e in entities if e.entity_type == EntityType.TICKER]
        for t in tickers:
            assert t.name.lower() not in {"i", "am"}

    def test_extract_patterns_filing(self):
        extractor = FinanceNERExtractor()
        entities = extractor._extract_patterns("The company filed a 10-K report.")
        filings = [e for e in entities if e.entity_type == EntityType.FILING]
        assert any(e.name == "10-K" for e in filings)

    def test_extract_patterns_metric(self):
        extractor = FinanceNERExtractor()
        entities = extractor._extract_patterns("The revenue grew by 20%.")
        metrics = [e for e in entities if e.entity_type == EntityType.METRIC]
        assert any("revenue" in e.name.lower() for e in metrics)

    def test_deduplicate_entities(self):
        extractor = FinanceNERExtractor()
        e1 = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY, confidence=0.8,
            source_chunk_ids=["c1"],
        )
        e2 = Entity(
            name="Apple", normalized_name="apple",
            entity_type=EntityType.COMPANY, confidence=0.9,
            source_chunk_ids=["c2"],
        )
        result = extractor._deduplicate_entities([e1, e2])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_deduplicate_different_types_kept(self):
        extractor = FinanceNERExtractor()
        e1 = Entity(name="Apple", normalized_name="apple", entity_type=EntityType.COMPANY)
        e2 = Entity(name="AAPL", normalized_name="aapl", entity_type=EntityType.TICKER)
        result = extractor._deduplicate_entities([e1, e2])
        assert len(result) == 2

    def test_normalize_ticker(self):
        extractor = FinanceNERExtractor()
        assert extractor.normalize_entity("aapl", EntityType.TICKER) == "AAPL"

    def test_normalize_company(self):
        extractor = FinanceNERExtractor()
        assert extractor.normalize_entity("Apple Inc.", EntityType.COMPANY) == "Apple"

    def test_normalize_metric(self):
        extractor = FinanceNERExtractor()
        assert extractor.normalize_entity("net income", EntityType.METRIC) == "net_income"

    @pytest.mark.asyncio
    async def test_extract_empty(self):
        extractor = FinanceNERExtractor()
        extractor._initialized = True
        extractor._spacy_extractor = MagicMock()
        result = await extractor.extract("")
        assert result == []


class TestHealthcareNERExtractor:
    def test_init(self):
        extractor = HealthcareNERExtractor()
        assert extractor.model_name == "en_ner_bc5cdr_md"

    def test_label_map(self):
        assert HealthcareNERExtractor.LABEL_MAP["DISEASE"] == EntityType.CONDITION
        assert HealthcareNERExtractor.LABEL_MAP["CHEMICAL"] == EntityType.CHEMICAL
        assert HealthcareNERExtractor.LABEL_MAP["DRUG"] == EntityType.DRUG

    @pytest.mark.asyncio
    async def test_extract_empty(self):
        extractor = HealthcareNERExtractor()
        extractor._initialized = True
        result = await extractor.extract("")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_with_mock(self):
        extractor = HealthcareNERExtractor()
        extractor._initialized = True

        mock_ent = MagicMock()
        mock_ent.text = "diabetes"
        mock_ent.label_ = "DISEASE"
        mock_ent.start_char = 0
        mock_ent.end_char = 8

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        extractor._nlp = MagicMock(return_value=mock_doc)

        entities = await extractor.extract("Patient has diabetes.")
        assert len(entities) == 1
        assert entities[0].entity_type == EntityType.CONDITION

    def test_normalize_condition_plural(self):
        extractor = HealthcareNERExtractor()
        assert extractor.normalize_entity("tumors", EntityType.CONDITION) == "tumor"

    def test_normalize_no_strip_double_s(self):
        extractor = HealthcareNERExtractor()
        assert extractor.normalize_entity("stress", EntityType.CONDITION) == "stress"
