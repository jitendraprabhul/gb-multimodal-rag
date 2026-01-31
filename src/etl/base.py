"""
Base processor interface and registry for ETL components.

Implements the Strategy pattern for different document processors
and a Registry pattern for processor management.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, Generic

from src.core.exceptions import DocumentProcessingError
from src.core.logging import LoggerMixin
from src.core.types import Document, DocumentType


T = TypeVar("T")


class BaseProcessor(ABC, LoggerMixin, Generic[T]):
    """
    Abstract base class for all document processors.

    Implements Template Method pattern where subclasses define
    the specific processing logic.
    """

    supported_types: list[DocumentType] = []

    def __init__(self, **config: Any) -> None:
        """
        Initialize processor with configuration.

        Args:
            **config: Processor-specific configuration
        """
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize processor resources (models, connections, etc.).

        Override in subclasses for async initialization.
        """
        self._initialized = True

    async def cleanup(self) -> None:
        """
        Clean up processor resources.

        Override in subclasses for proper cleanup.
        """
        self._initialized = False

    def supports(self, doc_type: DocumentType) -> bool:
        """Check if processor supports the given document type."""
        return doc_type in self.supported_types

    @abstractmethod
    async def process(self, file_path: Path, **kwargs: Any) -> T:
        """
        Process a file and return the result.

        Args:
            file_path: Path to the file to process
            **kwargs: Additional processing options

        Returns:
            Processing result (type depends on processor)

        Raises:
            DocumentProcessingError: If processing fails
        """
        pass

    def validate_file(self, file_path: Path) -> None:
        """
        Validate that the file exists and is readable.

        Args:
            file_path: Path to validate

        Raises:
            DocumentProcessingError: If validation fails
        """
        if not file_path.exists():
            raise DocumentProcessingError(
                f"File not found: {file_path}",
                document_path=str(file_path),
            )
        if not file_path.is_file():
            raise DocumentProcessingError(
                f"Not a file: {file_path}",
                document_path=str(file_path),
            )

    async def __aenter__(self) -> "BaseProcessor[T]":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()


class ProcessorRegistry:
    """
    Registry for document processors.

    Implements the Registry pattern to manage different processors
    and select the appropriate one based on document type.
    """

    _instance: "ProcessorRegistry | None" = None
    _processors: dict[DocumentType, BaseProcessor[Any]]

    def __new__(cls) -> "ProcessorRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._processors = {}
        return cls._instance

    def register(self, processor: BaseProcessor[Any]) -> None:
        """
        Register a processor for its supported types.

        Args:
            processor: Processor instance to register
        """
        for doc_type in processor.supported_types:
            self._processors[doc_type] = processor

    def get_processor(self, doc_type: DocumentType) -> BaseProcessor[Any]:
        """
        Get the processor for a document type.

        Args:
            doc_type: Document type

        Returns:
            Appropriate processor

        Raises:
            DocumentProcessingError: If no processor found
        """
        processor = self._processors.get(doc_type)
        if processor is None:
            raise DocumentProcessingError(
                f"No processor registered for document type: {doc_type}",
                document_type=doc_type.value,
            )
        return processor

    def has_processor(self, doc_type: DocumentType) -> bool:
        """Check if a processor is registered for the type."""
        return doc_type in self._processors

    @property
    def supported_types(self) -> list[DocumentType]:
        """Get all supported document types."""
        return list(self._processors.keys())

    def clear(self) -> None:
        """Clear all registered processors."""
        self._processors.clear()


def get_document_type(file_path: Path) -> DocumentType:
    """
    Determine document type from file extension.

    Args:
        file_path: Path to the file

    Returns:
        DocumentType enum value

    Raises:
        DocumentProcessingError: If type cannot be determined
    """
    extension_map = {
        ".pdf": DocumentType.PDF,
        ".txt": DocumentType.TXT,
        ".csv": DocumentType.CSV,
        ".xlsx": DocumentType.XLSX,
        ".xls": DocumentType.XLSX,
        ".png": DocumentType.IMAGE,
        ".jpg": DocumentType.IMAGE,
        ".jpeg": DocumentType.IMAGE,
        ".tiff": DocumentType.IMAGE,
        ".tif": DocumentType.IMAGE,
        ".bmp": DocumentType.IMAGE,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
        ".json": DocumentType.JSON,
    }

    suffix = file_path.suffix.lower()
    doc_type = extension_map.get(suffix)

    if doc_type is None:
        raise DocumentProcessingError(
            f"Unsupported file type: {suffix}",
            document_path=str(file_path),
            document_type=suffix,
        )

    return doc_type
