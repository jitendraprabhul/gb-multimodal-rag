"""
Custom exceptions for the GraphRAG system.

Provides a hierarchy of specific exceptions for better error handling
and debugging across all system components.
"""

from typing import Any


class GraphRAGError(Exception):
    """
    Base exception for all GraphRAG errors.

    All custom exceptions inherit from this class, allowing for
    catch-all error handling when needed.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        base = self.message
        if self.details:
            base += f" | Details: {self.details}"
        if self.cause:
            base += f" | Caused by: {self.cause}"
        return base


class ConfigurationError(GraphRAGError):
    """
    Raised when there's a configuration error.

    Examples:
    - Missing required environment variables
    - Invalid configuration values
    - Unable to connect to required services
    """

    pass


class DocumentProcessingError(GraphRAGError):
    """
    Raised when document processing fails.

    Examples:
    - PDF parsing failures
    - OCR errors
    - Unsupported document format
    - Table extraction failures
    """

    def __init__(
        self,
        message: str,
        document_path: str | None = None,
        document_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if document_path:
            details["document_path"] = document_path
        if document_type:
            details["document_type"] = document_type
        super().__init__(message, details, cause)


class EmbeddingError(GraphRAGError):
    """
    Raised when embedding generation fails.

    Examples:
    - Model loading failures
    - Out of memory errors
    - Invalid input format
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        modality: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        if modality:
            details["modality"] = modality
        super().__init__(message, details, cause)


class GraphError(GraphRAGError):
    """
    Raised when graph operations fail.

    Examples:
    - Neo4j connection errors
    - Cypher query failures
    - Graph traversal errors
    - Entity/relation creation failures
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        node_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if query:
            details["query"] = query
        if node_id:
            details["node_id"] = node_id
        super().__init__(message, details, cause)


class LLMError(GraphRAGError):
    """
    Raised when LLM operations fail.

    Examples:
    - Ollama connection errors
    - Model not found
    - Generation timeout
    - Invalid response format
    """

    def __init__(
        self,
        message: str,
        model: str | None = None,
        prompt_length: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if model:
            details["model"] = model
        if prompt_length:
            details["prompt_length"] = prompt_length
        super().__init__(message, details, cause)


class RetrievalError(GraphRAGError):
    """
    Raised when retrieval operations fail.

    Examples:
    - Vector search failures
    - Hybrid retrieval errors
    - Reranking failures
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        retrieval_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if query:
            details["query"] = query[:100] + "..." if len(query or "") > 100 else query
        if retrieval_type:
            details["retrieval_type"] = retrieval_type
        super().__init__(message, details, cause)


class ValidationError(GraphRAGError):
    """
    Raised when input validation fails.

    Examples:
    - Invalid query format
    - Missing required fields
    - Invalid entity/relation types
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]
        super().__init__(message, details, cause)


class VectorDBError(GraphRAGError):
    """
    Raised when vector database operations fail.

    Examples:
    - Qdrant connection errors
    - Collection creation failures
    - Upsert failures
    """

    def __init__(
        self,
        message: str,
        collection: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if collection:
            details["collection"] = collection
        if operation:
            details["operation"] = operation
        super().__init__(message, details, cause)


class NERError(GraphRAGError):
    """
    Raised when NER extraction fails.

    Examples:
    - Model loading errors
    - Processing errors
    - Invalid text input
    """

    def __init__(
        self,
        message: str,
        model: str | None = None,
        text_length: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if model:
            details["model"] = model
        if text_length:
            details["text_length"] = text_length
        super().__init__(message, details, cause)
