"""
Tests for core data types and models.

Covers all Pydantic models, enums, validators, and edge cases.
"""

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from src.core.types import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    Entity,
    EntityType,
    GraphPath,
    ImageRegion,
    Modality,
    ProcessingJob,
    ProcessingStatus,
    QueryResult,
    Relation,
    RelationType,
    RetrievalResult,
    SourceSnippet,
    Table,
    TableCell,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestModality:
    def test_values(self):
        assert Modality.TEXT == "text"
        assert Modality.TABLE == "table"
        assert Modality.IMAGE == "image"

    def test_all_values_present(self):
        assert len(Modality) == 3


class TestDocumentType:
    def test_values(self):
        assert DocumentType.PDF == "pdf"
        assert DocumentType.TXT == "txt"
        assert DocumentType.CSV == "csv"
        assert DocumentType.XLSX == "xlsx"
        assert DocumentType.IMAGE == "image"
        assert DocumentType.HTML == "html"
        assert DocumentType.JSON == "json"

    def test_all_values_present(self):
        assert len(DocumentType) == 7


class TestEntityType:
    def test_common_types(self):
        assert EntityType.PERSON == "PERSON"
        assert EntityType.ORGANIZATION == "ORGANIZATION"
        assert EntityType.LOCATION == "LOCATION"
        assert EntityType.DATE == "DATE"
        assert EntityType.MONEY == "MONEY"

    def test_finance_types(self):
        assert EntityType.COMPANY == "COMPANY"
        assert EntityType.FILING == "FILING"
        assert EntityType.METRIC == "METRIC"
        assert EntityType.TICKER == "TICKER"

    def test_healthcare_types(self):
        assert EntityType.PATIENT == "PATIENT"
        assert EntityType.CONDITION == "CONDITION"
        assert EntityType.DRUG == "DRUG"
        assert EntityType.GENE == "GENE"


class TestRelationType:
    def test_common_types(self):
        assert RelationType.MENTIONS == "MENTIONS"
        assert RelationType.RELATED_TO == "RELATED_TO"
        assert RelationType.PART_OF == "PART_OF"

    def test_finance_types(self):
        assert RelationType.FILED == "FILED"
        assert RelationType.ACQUIRED == "ACQUIRED"
        assert RelationType.REPORTED == "REPORTED"

    def test_healthcare_types(self):
        assert RelationType.HAS_CONDITION == "HAS_CONDITION"
        assert RelationType.TREATED_WITH == "TREATED_WITH"
        assert RelationType.PRESCRIBED == "PRESCRIBED"


# =============================================================================
# ChunkMetadata Tests
# =============================================================================


class TestChunkMetadata:
    def test_minimal(self):
        meta = ChunkMetadata(doc_id="doc-1")
        assert meta.doc_id == "doc-1"
        assert meta.page_number is None
        assert meta.section is None

    def test_full(self):
        meta = ChunkMetadata(
            doc_id="doc-1",
            page_number=5,
            section="Introduction",
            table_id="tbl-1",
            image_id="img-1",
            start_char=0,
            end_char=100,
            source_file="test.pdf",
        )
        assert meta.page_number == 5
        assert meta.source_file == "test.pdf"

    def test_extra_fields_allowed(self):
        meta = ChunkMetadata(doc_id="doc-1", custom_field="custom_value")
        assert meta.custom_field == "custom_value"


# =============================================================================
# Chunk Tests
# =============================================================================


class TestChunk:
    def test_create_minimal(self):
        chunk = Chunk(
            content="Hello world",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
        )
        assert chunk.content == "Hello world"
        assert chunk.modality == Modality.TEXT
        assert chunk.embedding is None
        assert chunk.entity_ids == []
        # Auto-generated UUID
        UUID(chunk.id)

    def test_content_stripped(self):
        chunk = Chunk(
            content="  Hello world  ",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
        )
        assert chunk.content == "Hello world"

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError, match="Chunk content cannot be empty"):
            Chunk(
                content="",
                modality=Modality.TEXT,
                metadata=ChunkMetadata(doc_id="doc-1"),
            )

    def test_whitespace_only_content_raises(self):
        with pytest.raises(ValidationError, match="Chunk content cannot be empty"):
            Chunk(
                content="   ",
                modality=Modality.TEXT,
                metadata=ChunkMetadata(doc_id="doc-1"),
            )

    def test_token_count(self):
        chunk = Chunk(
            content="one two three four five",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
        )
        assert chunk.token_count == int(5 * 1.3)

    def test_with_embedding(self):
        chunk = Chunk(
            content="test",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_with_entity_ids(self):
        chunk = Chunk(
            content="test",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
            entity_ids=["ent-1", "ent-2"],
        )
        assert chunk.entity_ids == ["ent-1", "ent-2"]

    def test_created_at_auto(self):
        chunk = Chunk(
            content="test",
            modality=Modality.TEXT,
            metadata=ChunkMetadata(doc_id="doc-1"),
        )
        assert isinstance(chunk.created_at, datetime)

    def test_all_modalities(self):
        for modality in Modality:
            chunk = Chunk(
                content="test",
                modality=modality,
                metadata=ChunkMetadata(doc_id="doc-1"),
            )
            assert chunk.modality == modality


# =============================================================================
# Document Tests
# =============================================================================


class TestDocument:
    def test_create_minimal(self):
        doc = Document(filename="test.pdf", doc_type=DocumentType.PDF)
        assert doc.filename == "test.pdf"
        assert doc.chunks == []
        assert doc.metadata == {}
        assert doc.chunk_count == 0
        UUID(doc.id)

    def test_with_chunks(self):
        chunks = [
            Chunk(
                content=f"Chunk {i}",
                modality=Modality.TEXT,
                metadata=ChunkMetadata(doc_id="doc-1"),
            )
            for i in range(3)
        ]
        doc = Document(
            filename="test.pdf",
            doc_type=DocumentType.PDF,
            chunks=chunks,
        )
        assert doc.chunk_count == 3

    def test_with_metadata(self):
        doc = Document(
            filename="test.pdf",
            doc_type=DocumentType.PDF,
            metadata={"author": "Test", "pages": 10},
        )
        assert doc.metadata["author"] == "Test"

    def test_all_document_types(self):
        for doc_type in DocumentType:
            doc = Document(filename=f"test.{doc_type.value}", doc_type=doc_type)
            assert doc.doc_type == doc_type


# =============================================================================
# Entity Tests
# =============================================================================


class TestEntity:
    def test_create_minimal(self):
        entity = Entity(
            name="Apple Inc.",
            normalized_name="apple inc.",
            entity_type=EntityType.COMPANY,
        )
        assert entity.name == "Apple Inc."
        assert entity.normalized_name == "apple inc."
        assert entity.confidence == 1.0
        assert entity.source_chunk_ids == []

    def test_auto_normalize_name(self):
        entity = Entity(
            name="Apple Inc.",
            normalized_name="",
            entity_type=EntityType.COMPANY,
        )
        # When normalized_name is empty string, validator uses name
        assert entity.normalized_name == "apple inc."

    def test_normalized_name_stripped_lowered(self):
        entity = Entity(
            name="Test",
            normalized_name="  HELLO WORLD  ",
            entity_type=EntityType.PERSON,
        )
        assert entity.normalized_name == "hello world"

    def test_confidence_bounds(self):
        entity = Entity(
            name="Test",
            normalized_name="test",
            entity_type=EntityType.PERSON,
            confidence=0.5,
        )
        assert entity.confidence == 0.5

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            Entity(
                name="Test",
                normalized_name="test",
                entity_type=EntityType.PERSON,
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            Entity(
                name="Test",
                normalized_name="test",
                entity_type=EntityType.PERSON,
                confidence=1.1,
            )

    def test_merge_with_same_name(self):
        e1 = Entity(
            name="Apple",
            normalized_name="apple",
            entity_type=EntityType.COMPANY,
            source_chunk_ids=["c1"],
            confidence=0.8,
            attributes={"ticker": "AAPL"},
        )
        e2 = Entity(
            name="Apple",
            normalized_name="apple",
            entity_type=EntityType.COMPANY,
            source_chunk_ids=["c2"],
            confidence=0.9,
            attributes={"sector": "Tech"},
        )
        merged = e1.merge_with(e2)
        assert merged.id == e1.id
        assert set(merged.source_chunk_ids) == {"c1", "c2"}
        assert merged.confidence == 0.9
        assert merged.attributes["ticker"] == "AAPL"
        assert merged.attributes["sector"] == "Tech"

    def test_merge_different_names_raises(self):
        e1 = Entity(
            name="Apple",
            normalized_name="apple",
            entity_type=EntityType.COMPANY,
        )
        e2 = Entity(
            name="Google",
            normalized_name="google",
            entity_type=EntityType.COMPANY,
        )
        with pytest.raises(ValueError, match="Cannot merge entities"):
            e1.merge_with(e2)

    def test_merge_keeps_earlier_created_at(self):
        earlier = datetime(2023, 1, 1)
        later = datetime(2024, 1, 1)
        e1 = Entity(
            name="Test",
            normalized_name="test",
            entity_type=EntityType.PERSON,
            created_at=earlier,
        )
        e2 = Entity(
            name="Test",
            normalized_name="test",
            entity_type=EntityType.PERSON,
            created_at=later,
        )
        merged = e1.merge_with(e2)
        assert merged.created_at == earlier


# =============================================================================
# Relation Tests
# =============================================================================


class TestRelation:
    def test_create(self):
        rel = Relation(
            source_entity_id="e1",
            target_entity_id="e2",
            relation_type=RelationType.ACQUIRED,
        )
        assert rel.source_entity_id == "e1"
        assert rel.target_entity_id == "e2"
        assert rel.confidence == 1.0

    def test_as_triple(self):
        rel = Relation(
            source_entity_id="e1",
            target_entity_id="e2",
            relation_type=RelationType.MENTIONS,
        )
        assert rel.as_triple == ("e1", "MENTIONS", "e2")

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Relation(
                source_entity_id="e1",
                target_entity_id="e2",
                relation_type=RelationType.MENTIONS,
                confidence=1.5,
            )


# =============================================================================
# SourceSnippet Tests
# =============================================================================


class TestSourceSnippet:
    def test_create(self):
        snippet = SourceSnippet(
            chunk_id="c1",
            content="some content",
            modality=Modality.TEXT,
            relevance_score=0.85,
        )
        assert snippet.relevance_score == 0.85

    def test_relevance_score_bounds(self):
        with pytest.raises(ValidationError):
            SourceSnippet(
                chunk_id="c1",
                content="test",
                modality=Modality.TEXT,
                relevance_score=1.5,
            )


# =============================================================================
# GraphPath Tests
# =============================================================================


class TestGraphPath:
    def test_create(self):
        path = GraphPath(
            nodes=[{"id": "n1"}, {"id": "n2"}],
            edges=[{"type": "MENTIONS"}],
            path_text="n1 -> n2",
            relevance_score=0.9,
        )
        assert path.length == 1

    def test_empty_path(self):
        path = GraphPath(
            nodes=[],
            edges=[],
            path_text="empty",
            relevance_score=0.0,
        )
        assert path.length == 0

    def test_multi_hop_path(self):
        path = GraphPath(
            nodes=[{"id": f"n{i}"} for i in range(4)],
            edges=[{"type": "REL"} for _ in range(3)],
            path_text="n0 -> n1 -> n2 -> n3",
            relevance_score=0.7,
        )
        assert path.length == 3


# =============================================================================
# RetrievalResult Tests
# =============================================================================


class TestRetrievalResult:
    def test_create_minimal(self):
        result = RetrievalResult(
            chunks=[],
            entities=[],
            retrieval_time_ms=100.5,
        )
        assert result.retrieval_time_ms == 100.5
        assert result.graph_paths == []
        assert result.vector_scores == {}


# =============================================================================
# QueryResult Tests
# =============================================================================


class TestQueryResult:
    def test_create(self):
        result = QueryResult(
            query="What is revenue?",
            answer="Revenue is $100M",
            confidence=0.9,
            sources=[],
            graph_paths=[],
            latency_ms=250.0,
        )
        assert result.query == "What is revenue?"
        assert result.reasoning_chain is None

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            QueryResult(
                query="test",
                answer="answer",
                confidence=2.0,
                sources=[],
                graph_paths=[],
                latency_ms=0.0,
            )


# =============================================================================
# Table Tests
# =============================================================================


class TestTable:
    def test_create_minimal(self):
        table = Table(headers=["A", "B"], rows=[["1", "2"]])
        assert len(table.headers) == 2
        assert len(table.rows) == 1

    def test_as_markdown(self):
        table = Table(
            headers=["Name", "Value"],
            rows=[["Revenue", "$100M"], ["Profit", "$20M"]],
        )
        md = table.as_markdown
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| Revenue | $100M |" in md

    def test_as_markdown_empty(self):
        table = Table(headers=[], rows=[])
        assert table.as_markdown == ""

    def test_as_markdown_no_headers(self):
        table = Table(headers=[], rows=[["a", "b"]])
        md = table.as_markdown
        assert "| a | b |" in md
        assert "---" not in md

    def test_as_text_with_headers(self):
        table = Table(
            headers=["Name", "Value"],
            rows=[["Revenue", "$100M"]],
            caption="Financial Data",
        )
        text = table.as_text
        assert "Table: Financial Data" in text
        assert "Name: Revenue" in text

    def test_as_text_without_headers(self):
        table = Table(headers=[], rows=[["a", "b", "c"]])
        text = table.as_text
        assert "a, b, c" in text

    def test_as_text_skips_empty_values(self):
        table = Table(
            headers=["A", "B"],
            rows=[["value", "  "]],
        )
        text = table.as_text
        assert "A: value" in text


# =============================================================================
# TableCell Tests
# =============================================================================


class TestTableCell:
    def test_create(self):
        cell = TableCell(row=0, col=0, value="test")
        assert cell.is_header is False

    def test_negative_row_raises(self):
        with pytest.raises(ValidationError):
            TableCell(row=-1, col=0, value="test")

    def test_negative_col_raises(self):
        with pytest.raises(ValidationError):
            TableCell(row=0, col=-1, value="test")


# =============================================================================
# ImageRegion Tests
# =============================================================================


class TestImageRegion:
    def test_create_minimal(self):
        img = ImageRegion(image_path="/path/to/img.png")
        assert img.caption is None
        assert img.ocr_text is None
        assert img.bbox is None

    def test_create_full(self):
        img = ImageRegion(
            image_path="/path/to/img.png",
            caption="A chart showing revenue",
            ocr_text="Revenue 2023",
            bbox=(10, 20, 100, 200),
            page_number=3,
            doc_id="doc-1",
            image_type="chart",
        )
        assert img.bbox == (10, 20, 100, 200)
        assert img.image_type == "chart"


# =============================================================================
# ProcessingJob Tests
# =============================================================================


class TestProcessingJob:
    def test_create_default(self):
        job = ProcessingJob(document_path="/path/to/doc.pdf")
        assert job.status == ProcessingStatus.PENDING
        assert job.progress == 0.0
        assert job.error_message is None
        assert job.result is None

    def test_progress_bounds(self):
        with pytest.raises(ValidationError):
            ProcessingJob(document_path="/test", progress=101.0)

    def test_progress_negative_raises(self):
        with pytest.raises(ValidationError):
            ProcessingJob(document_path="/test", progress=-1.0)

    def test_status_values(self):
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
