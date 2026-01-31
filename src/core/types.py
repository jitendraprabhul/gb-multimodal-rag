"""
Core data types and models for the GraphRAG system.

Uses Pydantic models for validation, serialization, and documentation.
Supports both finance and healthcare domains with appropriate entity types.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
import numpy as np


# =============================================================================
# Enums
# =============================================================================


class Modality(str, Enum):
    """Content modality types."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    TXT = "txt"
    CSV = "csv"
    XLSX = "xlsx"
    IMAGE = "image"
    HTML = "html"
    JSON = "json"


class EntityType(str, Enum):
    """
    Entity types across domains.

    Finance domain:
    - COMPANY, FILING, SECTION, METRIC, EVENT, PERSON, DATE, MONEY

    Healthcare domain:
    - PATIENT, CONDITION, DRUG, LABTEST, STUDY, PROCEDURE, ANATOMY
    """

    # Common
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    QUANTITY = "QUANTITY"

    # Finance specific
    COMPANY = "COMPANY"
    FILING = "FILING"
    SECTION = "SECTION"
    METRIC = "METRIC"
    EVENT = "EVENT"
    TICKER = "TICKER"

    # Healthcare specific
    PATIENT = "PATIENT"
    CONDITION = "CONDITION"
    DRUG = "DRUG"
    LABTEST = "LABTEST"
    STUDY = "STUDY"
    PROCEDURE = "PROCEDURE"
    ANATOMY = "ANATOMY"
    GENE = "GENE"
    CHEMICAL = "CHEMICAL"


class RelationType(str, Enum):
    """
    Relationship types across domains.

    Finance:
    - FILED, HAS_SECTION, MENTIONS_METRIC, INVOLVED_IN, CITES, etc.

    Healthcare:
    - HAS_CONDITION, UNDERWENT, TREATED_WITH, MENTIONS, etc.
    """

    # Common
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"
    HAS_DATE = "HAS_DATE"

    # Finance
    FILED = "FILED"
    HAS_SECTION = "HAS_SECTION"
    MENTIONS_METRIC = "MENTIONS_METRIC"
    INVOLVED_IN = "INVOLVED_IN"
    CITES = "CITES"
    ACQUIRED = "ACQUIRED"
    MERGED_WITH = "MERGED_WITH"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    COMPETITOR_OF = "COMPETITOR_OF"
    REPORTED = "REPORTED"

    # Healthcare
    HAS_CONDITION = "HAS_CONDITION"
    UNDERWENT = "UNDERWENT"
    TREATED_WITH = "TREATED_WITH"
    DIAGNOSED_WITH = "DIAGNOSED_WITH"
    PRESCRIBED = "PRESCRIBED"
    CONTRAINDICATES = "CONTRAINDICATES"
    INTERACTS_WITH = "INTERACTS_WITH"
    TESTED_FOR = "TESTED_FOR"
    AFFECTS = "AFFECTS"


# =============================================================================
# Document Models
# =============================================================================


class ChunkMetadata(BaseModel):
    """Metadata associated with a document chunk."""

    doc_id: str = Field(description="Parent document ID")
    page_number: int | None = Field(default=None, description="Page number in source")
    section: str | None = Field(default=None, description="Section name/title")
    table_id: str | None = Field(default=None, description="Table ID if from table")
    image_id: str | None = Field(default=None, description="Image ID if from image")
    start_char: int | None = Field(default=None, description="Start character position")
    end_char: int | None = Field(default=None, description="End character position")
    source_file: str | None = Field(default=None, description="Original file path")

    class Config:
        extra = "allow"  # Allow additional metadata fields


class Chunk(BaseModel):
    """A chunk of content from a document."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk ID")
    content: str = Field(description="Text content of the chunk")
    modality: Modality = Field(description="Content modality")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
    embedding: list[float] | None = Field(default=None, description="Vector embedding")
    entity_ids: list[str] = Field(default_factory=list, description="Associated entity IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()

    @property
    def token_count(self) -> int:
        """Approximate token count (words * 1.3)."""
        return int(len(self.content.split()) * 1.3)


class Document(BaseModel):
    """A processed document with its chunks and metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique document ID")
    filename: str = Field(description="Original filename")
    doc_type: DocumentType = Field(description="Document type")
    title: str | None = Field(default=None, description="Document title")
    chunks: list[Chunk] = Field(default_factory=list, description="Document chunks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    file_hash: str | None = Field(default=None, description="File content hash")

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the document."""
        return len(self.chunks)


# =============================================================================
# Entity & Relation Models
# =============================================================================


class Entity(BaseModel):
    """A named entity extracted from content."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entity ID")
    name: str = Field(description="Entity name/label")
    normalized_name: str = Field(description="Normalized/canonical name")
    entity_type: EntityType = Field(description="Entity type")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunks where entity was found"
    )
    confidence: float = Field(default=1.0, ge=0, le=1, description="Extraction confidence")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("normalized_name", mode="before")
    @classmethod
    def set_normalized_name(cls, v: str | None, info) -> str:
        """Set normalized name from name if not provided."""
        if v:
            return v.lower().strip()
        # Get name from the data being validated
        name = info.data.get("name", "")
        return name.lower().strip()

    def merge_with(self, other: "Entity") -> "Entity":
        """Merge with another entity of the same type."""
        if self.normalized_name != other.normalized_name:
            raise ValueError("Cannot merge entities with different normalized names")

        return Entity(
            id=self.id,
            name=self.name,
            normalized_name=self.normalized_name,
            entity_type=self.entity_type,
            attributes={**self.attributes, **other.attributes},
            source_chunk_ids=list(set(self.source_chunk_ids + other.source_chunk_ids)),
            confidence=max(self.confidence, other.confidence),
            created_at=min(self.created_at, other.created_at),
        )


class Relation(BaseModel):
    """A relationship between two entities."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique relation ID")
    source_entity_id: str = Field(description="Source entity ID")
    target_entity_id: str = Field(description="Target entity ID")
    relation_type: RelationType = Field(description="Relationship type")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Relation attributes")
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Chunks where relation was found"
    )
    confidence: float = Field(default=1.0, ge=0, le=1, description="Extraction confidence")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def as_triple(self) -> tuple[str, str, str]:
        """Return as (source, relation, target) triple."""
        return (self.source_entity_id, self.relation_type.value, self.target_entity_id)


# =============================================================================
# Query & Retrieval Models
# =============================================================================


class SourceSnippet(BaseModel):
    """A source snippet supporting an answer."""

    chunk_id: str = Field(description="Source chunk ID")
    content: str = Field(description="Snippet content")
    modality: Modality = Field(description="Content modality")
    relevance_score: float = Field(ge=0, le=1, description="Relevance score")
    page_number: int | None = Field(default=None, description="Page number")
    section: str | None = Field(default=None, description="Section name")
    doc_id: str | None = Field(default=None, description="Document ID")


class GraphPath(BaseModel):
    """A path through the knowledge graph supporting reasoning."""

    nodes: list[dict[str, Any]] = Field(description="Nodes in the path")
    edges: list[dict[str, Any]] = Field(description="Edges in the path")
    path_text: str = Field(description="Human-readable path description")
    relevance_score: float = Field(ge=0, le=1, description="Path relevance score")

    @property
    def length(self) -> int:
        """Number of hops in the path."""
        return len(self.edges)


class RetrievalResult(BaseModel):
    """Result from hybrid retrieval."""

    chunks: list[Chunk] = Field(description="Retrieved chunks")
    entities: list[Entity] = Field(description="Related entities")
    graph_paths: list[GraphPath] = Field(default_factory=list, description="Graph paths")
    vector_scores: dict[str, float] = Field(
        default_factory=dict, description="Vector similarity scores"
    )
    graph_scores: dict[str, float] = Field(
        default_factory=dict, description="Graph relevance scores"
    )
    retrieval_time_ms: float = Field(description="Retrieval time in milliseconds")


class QueryResult(BaseModel):
    """Final result for a user query."""

    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    confidence: float = Field(ge=0, le=1, description="Answer confidence score")
    sources: list[SourceSnippet] = Field(description="Supporting sources")
    graph_paths: list[GraphPath] = Field(description="Graph reasoning paths")
    reasoning_chain: str | None = Field(default=None, description="Chain of thought")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    latency_ms: float = Field(description="Total latency in milliseconds")


# =============================================================================
# Table & Image Models
# =============================================================================


class TableCell(BaseModel):
    """A cell in a table."""

    row: int = Field(ge=0, description="Row index")
    col: int = Field(ge=0, description="Column index")
    value: str = Field(description="Cell value")
    is_header: bool = Field(default=False, description="Is header cell")


class Table(BaseModel):
    """A table extracted from a document."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique table ID")
    headers: list[str] = Field(description="Column headers")
    rows: list[list[str]] = Field(description="Table rows")
    caption: str | None = Field(default=None, description="Table caption")
    page_number: int | None = Field(default=None, description="Page number")
    doc_id: str | None = Field(default=None, description="Parent document ID")

    @property
    def as_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers and not self.rows:
            return ""

        lines = []
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @property
    def as_text(self) -> str:
        """Convert table to plain text (for embedding)."""
        lines = []
        if self.caption:
            lines.append(f"Table: {self.caption}")

        for i, row in enumerate(self.rows):
            if self.headers:
                row_text = ", ".join(
                    f"{h}: {v}" for h, v in zip(self.headers, row) if v.strip()
                )
            else:
                row_text = ", ".join(v for v in row if v.strip())
            if row_text:
                lines.append(row_text)

        return "\n".join(lines)


class ImageRegion(BaseModel):
    """An image region extracted from a document."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique image ID")
    image_path: str = Field(description="Path to extracted image")
    caption: str | None = Field(default=None, description="Image caption")
    ocr_text: str | None = Field(default=None, description="OCR extracted text")
    embedding: list[float] | None = Field(default=None, description="CLIP embedding")
    bbox: tuple[int, int, int, int] | None = Field(
        default=None, description="Bounding box (x1, y1, x2, y2)"
    )
    page_number: int | None = Field(default=None, description="Page number")
    doc_id: str | None = Field(default=None, description="Parent document ID")
    image_type: str | None = Field(default=None, description="Image type (chart, diagram, etc.)")


# =============================================================================
# Processing Status Models
# =============================================================================


class ProcessingStatus(str, Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJob(BaseModel):
    """A document processing job."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Job ID")
    document_path: str = Field(description="Path to document")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    progress: float = Field(default=0.0, ge=0, le=100, description="Progress percentage")
    error_message: str | None = Field(default=None, description="Error message if failed")
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    result: Document | None = Field(default=None, description="Processing result")
