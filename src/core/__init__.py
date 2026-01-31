"""Core module with types, exceptions, and logging utilities."""

from src.core.exceptions import (
    GraphRAGError,
    ConfigurationError,
    DocumentProcessingError,
    EmbeddingError,
    GraphError,
    LLMError,
    RetrievalError,
    ValidationError,
)
from src.core.logging import get_logger, setup_logging
from src.core.types import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    Entity,
    EntityType,
    GraphPath,
    Modality,
    QueryResult,
    Relation,
    RelationType,
    RetrievalResult,
    SourceSnippet,
)

__all__ = [
    # Exceptions
    "GraphRAGError",
    "ConfigurationError",
    "DocumentProcessingError",
    "EmbeddingError",
    "GraphError",
    "LLMError",
    "RetrievalError",
    "ValidationError",
    # Logging
    "get_logger",
    "setup_logging",
    # Types
    "Chunk",
    "ChunkMetadata",
    "Document",
    "DocumentType",
    "Entity",
    "EntityType",
    "GraphPath",
    "Modality",
    "QueryResult",
    "Relation",
    "RelationType",
    "RetrievalResult",
    "SourceSnippet",
]
