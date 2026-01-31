"""
Pydantic schemas for API request/response models.

Defines the contract for all API endpoints.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    services: dict[str, bool] = Field(description="Individual service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Question Answering
# =============================================================================


class AskRequest(BaseModel):
    """Request for question answering."""

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Question to answer",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return",
    )
    filter_doc_ids: list[str] | None = Field(
        default=None,
        description="Filter to specific documents",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source snippets in response",
    )
    include_graph_paths: bool = Field(
        default=True,
        description="Include graph reasoning paths",
    )


class SourceSnippetResponse(BaseModel):
    """Source snippet in response."""

    chunk_id: str = Field(description="Chunk ID")
    content: str = Field(description="Snippet content")
    modality: str = Field(description="Content type (text/table/image)")
    relevance_score: float = Field(description="Relevance score")
    page_number: int | None = Field(default=None, description="Page number")
    section: str | None = Field(default=None, description="Section")
    doc_id: str | None = Field(default=None, description="Document ID")


class GraphPathResponse(BaseModel):
    """Graph path in response."""

    path_text: str = Field(description="Human-readable path")
    nodes: list[dict[str, Any]] = Field(description="Path nodes")
    edges: list[dict[str, Any]] = Field(description="Path edges")
    relevance_score: float = Field(description="Path relevance")


class AskResponse(BaseModel):
    """Response for question answering."""

    answer: str = Field(description="Generated answer")
    confidence: float = Field(
        ge=0,
        le=1,
        description="Answer confidence score",
    )
    sources: list[SourceSnippetResponse] = Field(
        default_factory=list,
        description="Supporting sources",
    )
    graph_paths: list[GraphPathResponse] = Field(
        default_factory=list,
        description="Graph reasoning paths",
    )
    reasoning: str | None = Field(
        default=None,
        description="Chain of thought reasoning",
    )
    latency_ms: float = Field(description="Total latency in milliseconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


# =============================================================================
# Entity Operations
# =============================================================================


class EntityRequest(BaseModel):
    """Request for entity search."""

    query: str = Field(min_length=1, description="Search query")
    entity_types: list[str] | None = Field(
        default=None,
        description="Filter by entity types",
    )
    limit: int = Field(default=10, ge=1, le=100)


class EntityResponse(BaseModel):
    """Entity response."""

    id: str = Field(description="Entity ID")
    name: str = Field(description="Entity name")
    normalized_name: str = Field(description="Normalized name")
    entity_type: str = Field(description="Entity type")
    confidence: float = Field(description="Confidence score")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Entity attributes",
    )


class EntityContextResponse(BaseModel):
    """Entity context with relationships."""

    entity: EntityResponse = Field(description="Entity details")
    neighbors: list[dict[str, Any]] = Field(
        description="Neighboring entities"
    )
    relations: list[dict[str, Any]] = Field(
        description="Entity relationships"
    )
    chunks: list[dict[str, Any]] = Field(
        description="Related document chunks"
    )


# =============================================================================
# Graph Operations
# =============================================================================


class GraphSubgraphRequest(BaseModel):
    """Request for subgraph retrieval."""

    entity_ids: list[str] = Field(
        min_length=1,
        description="Starting entity IDs",
    )
    hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of hops to traverse",
    )
    max_nodes: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum nodes to return",
    )


class GraphSubgraphResponse(BaseModel):
    """Subgraph response."""

    nodes: list[dict[str, Any]] = Field(description="Graph nodes")
    edges: list[dict[str, Any]] = Field(description="Graph edges")
    node_count: int = Field(description="Total nodes")
    edge_count: int = Field(description="Total edges")


class PathFindRequest(BaseModel):
    """Request for path finding."""

    source_entity_id: str = Field(description="Source entity ID")
    target_entity_id: str = Field(description="Target entity ID")
    max_hops: int = Field(default=3, ge=1, le=5)
    limit: int = Field(default=5, ge=1, le=10)


class PathFindResponse(BaseModel):
    """Path finding response."""

    paths: list[GraphPathResponse] = Field(description="Found paths")
    total_paths: int = Field(description="Total paths found")


# =============================================================================
# Ingestion Operations
# =============================================================================


class IngestRequest(BaseModel):
    """Request for document ingestion."""

    file_path: str = Field(description="Path to file to ingest")
    doc_id: str | None = Field(
        default=None,
        description="Custom document ID",
    )
    extract_entities: bool = Field(
        default=True,
        description="Extract entities and build graph",
    )
    generate_embeddings: bool = Field(
        default=True,
        description="Generate vector embeddings",
    )


class IngestResponse(BaseModel):
    """Response for document ingestion."""

    doc_id: str = Field(description="Document ID")
    filename: str = Field(description="Processed filename")
    chunks: int = Field(description="Number of chunks created")
    entities: int = Field(description="Number of entities extracted")
    relations: int = Field(description="Number of relations created")
    processing_time_ms: float = Field(description="Processing time")
    status: str = Field(description="Processing status")


class BatchIngestRequest(BaseModel):
    """Request for batch ingestion."""

    directory_path: str = Field(description="Directory to ingest")
    file_patterns: list[str] | None = Field(
        default=None,
        description="File patterns (e.g., ['*.pdf', '*.csv'])",
    )
    recursive: bool = Field(
        default=True,
        description="Process subdirectories",
    )


class BatchIngestResponse(BaseModel):
    """Response for batch ingestion."""

    total_files: int = Field(description="Total files processed")
    successful: int = Field(description="Successfully processed")
    failed: int = Field(description="Failed to process")
    results: list[IngestResponse] = Field(description="Individual results")
    total_time_ms: float = Field(description="Total processing time")


# =============================================================================
# Statistics
# =============================================================================


class SystemStatsResponse(BaseModel):
    """System statistics response."""

    graph: dict[str, int] = Field(
        description="Graph statistics",
        example={"entities": 1000, "relations": 5000},
    )
    vector: dict[str, Any] = Field(
        description="Vector store statistics",
    )
    documents: int = Field(description="Total documents")
    chunks: int = Field(description="Total chunks")


# =============================================================================
# Explain Operations
# =============================================================================


class ExplainRequest(BaseModel):
    """Request for answer explanation."""

    question: str = Field(description="Original question")
    answer: str = Field(description="Answer to explain")


class ExplainResponse(BaseModel):
    """Explanation response."""

    explanation: str = Field(description="Detailed explanation")
    evidence: list[dict[str, Any]] = Field(
        description="Supporting evidence"
    )
    confidence_factors: dict[str, float] = Field(
        description="Factors affecting confidence"
    )
