"""
Unified ETL pipeline for multimodal document processing.

Orchestrates all processors for end-to-end document ingestion.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Callable

from src.core.exceptions import DocumentProcessingError
from src.core.logging import LoggerMixin, log_operation
from src.core.types import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    ImageRegion,
    Modality,
    ProcessingJob,
    ProcessingStatus,
    Table,
)
from src.etl.base import BaseProcessor, ProcessorRegistry, get_document_type
from src.etl.chunker import ChunkingStrategy, TableChunker, TextChunker
from src.etl.image_processor import ImageProcessor
from src.etl.ocr_processor import OCRProcessor
from src.etl.pdf_processor import PDFProcessor, TextFileProcessor
from src.etl.table_extractor import SpreadsheetProcessor, TableExtractor


class ETLPipeline(LoggerMixin):
    """
    Unified ETL pipeline for document processing.

    Coordinates:
    - Document type detection
    - Multi-processor pipeline execution
    - Text chunking
    - Table and image processing
    - OCR for scanned content
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        extract_images: bool = True,
        extract_tables: bool = True,
        enable_ocr: bool = True,
        ocr_lang: str = "en",
        use_gpu: bool = False,
        image_output_dir: Path | None = None,
        max_concurrent: int = 4,
        **config: Any,
    ) -> None:
        """
        Initialize ETL pipeline.

        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            chunking_strategy: Chunking strategy
            extract_images: Whether to extract images from documents
            extract_tables: Whether to extract tables
            enable_ocr: Whether to enable OCR for scanned content
            ocr_lang: OCR language
            use_gpu: Whether to use GPU acceleration
            image_output_dir: Directory for extracted images
            max_concurrent: Max concurrent document processing
            **config: Additional configuration
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.enable_ocr = enable_ocr
        self.max_concurrent = max_concurrent
        self.image_output_dir = image_output_dir or Path("./data/processed/images")

        # Initialize components
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
        )
        self.table_chunker = TableChunker()

        # Processors
        self.pdf_processor = PDFProcessor(
            extract_images=extract_images,
            extract_tables=extract_tables,
            image_output_dir=self.image_output_dir,
        )
        self.text_processor = TextFileProcessor()
        self.spreadsheet_processor = SpreadsheetProcessor()
        self.image_processor = ImageProcessor(output_dir=self.image_output_dir)
        self.table_extractor = TableExtractor()
        self.ocr_processor = OCRProcessor(lang=ocr_lang, use_gpu=use_gpu)

        # Registry
        self.registry = ProcessorRegistry()
        self._initialized = False

        # Processing state
        self._semaphore: asyncio.Semaphore | None = None
        self._progress_callback: Callable[[ProcessingJob], None] | None = None

    async def initialize(self) -> None:
        """Initialize all processors."""
        if self._initialized:
            return

        self.logger.info("Initializing ETL pipeline")

        # Initialize processors
        await self.pdf_processor.initialize()
        await self.spreadsheet_processor.initialize()
        await self.image_processor.initialize()

        if self.enable_ocr:
            await self.ocr_processor.initialize()

        # Register processors
        self.registry.register(self.pdf_processor)
        self.registry.register(self.text_processor)
        self.registry.register(self.spreadsheet_processor)
        self.registry.register(self.image_processor)

        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        self._initialized = True
        self.logger.info("ETL pipeline initialized")

    async def cleanup(self) -> None:
        """Clean up all processors."""
        await self.pdf_processor.cleanup()
        await self.spreadsheet_processor.cleanup()
        await self.image_processor.cleanup()
        await self.ocr_processor.cleanup()
        self._initialized = False

    def set_progress_callback(
        self,
        callback: Callable[[ProcessingJob], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    async def process_file(
        self,
        file_path: Path,
        doc_id: str | None = None,
    ) -> Document:
        """
        Process a single file.

        Args:
            file_path: Path to file
            doc_id: Optional document ID

        Returns:
            Processed document
        """
        if not self._initialized:
            await self.initialize()

        file_path = Path(file_path)
        self.logger.info("Processing file", path=str(file_path))

        import time
        start_time = time.time()

        try:
            # Determine document type
            doc_type = get_document_type(file_path)

            # Get appropriate processor
            processor = self.registry.get_processor(doc_type)

            # Process document
            document = await processor.process(file_path)

            # Set document ID if provided
            if doc_id:
                document.id = doc_id

            # Update chunk doc_ids
            for chunk in document.chunks:
                chunk.metadata.doc_id = document.id

            # Apply text chunking to large chunks
            document = await self._rechunk_document(document)

            # Process tables if present
            if self.extract_tables and "tables" in document.metadata:
                await self._process_tables(document)

            # Process images if present
            if self.extract_images and "images" in document.metadata:
                await self._process_images(document)

            # OCR for scanned content
            if self.enable_ocr and doc_type == DocumentType.PDF:
                await self._apply_ocr_if_needed(document, file_path)

            duration_ms = (time.time() - start_time) * 1000
            log_operation(
                "process_file",
                success=True,
                duration_ms=duration_ms,
                file=str(file_path),
                chunks=len(document.chunks),
            )

            return document

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_operation(
                "process_file",
                success=False,
                duration_ms=duration_ms,
                file=str(file_path),
                error=str(e),
            )
            raise

    async def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_patterns: list[str] | None = None,
    ) -> list[Document]:
        """
        Process all files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            file_patterns: File patterns to match (e.g., ["*.pdf", "*.csv"])

        Returns:
            List of processed documents
        """
        if not self._initialized:
            await self.initialize()

        directory = Path(directory)
        if not directory.is_dir():
            raise DocumentProcessingError(
                f"Not a directory: {directory}",
                document_path=str(directory),
            )

        # Find files
        files = []
        patterns = file_patterns or ["*.pdf", "*.txt", "*.csv", "*.xlsx"]

        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))

        self.logger.info(
            "Processing directory",
            path=str(directory),
            files=len(files),
        )

        # Process files concurrently
        tasks = [self._process_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        documents = []
        for result in results:
            if isinstance(result, Document):
                documents.append(result)
            elif isinstance(result, Exception):
                self.logger.error("File processing failed", error=str(result))

        return documents

    async def _process_with_semaphore(self, file_path: Path) -> Document:
        """Process file with semaphore for concurrency control."""
        async with self._semaphore:
            return await self.process_file(file_path)

    async def _rechunk_document(self, document: Document) -> Document:
        """Apply chunking to document text."""
        new_chunks = []

        for chunk in document.chunks:
            if chunk.modality != Modality.TEXT:
                new_chunks.append(chunk)
                continue

            # Check if chunk needs splitting
            if len(chunk.content) <= self.chunk_size * 1.5:
                new_chunks.append(chunk)
                continue

            # Split large text chunks
            sub_chunks = self.text_chunker.chunk(
                chunk.content,
                metadata=chunk.metadata,
                modality=Modality.TEXT,
            )
            new_chunks.extend(sub_chunks)

        document.chunks = new_chunks
        return document

    async def _process_tables(self, document: Document) -> None:
        """Convert table data to chunks."""
        tables_data = document.metadata.get("tables", [])

        for table_dict in tables_data:
            table = Table(**table_dict)

            # Create chunks from table
            table_chunks = self.table_chunker.chunk_table(
                headers=table.headers,
                rows=table.rows,
                metadata=ChunkMetadata(
                    doc_id=document.id,
                    table_id=table.id,
                    page_number=table.page_number,
                ),
            )

            document.chunks.extend(table_chunks)

    async def _process_images(self, document: Document) -> None:
        """Process images and create image chunks."""
        images_data = document.metadata.get("images", [])

        for image_dict in images_data:
            region = ImageRegion(**image_dict)

            # Apply OCR if enabled
            if self.enable_ocr and not region.ocr_text:
                region = await self.ocr_processor.process_image_region(region)

            # Create chunk for image
            content = region.ocr_text or region.caption or f"Image: {region.image_type or 'unknown'}"

            if content.strip():
                chunk = Chunk(
                    content=content,
                    modality=Modality.IMAGE,
                    metadata=ChunkMetadata(
                        doc_id=document.id,
                        image_id=region.id,
                        page_number=region.page_number,
                    ),
                )
                document.chunks.append(chunk)

    async def _apply_ocr_if_needed(
        self,
        document: Document,
        file_path: Path,
    ) -> None:
        """Apply OCR to scanned PDF pages if needed."""
        # Check if document appears to be scanned
        is_scanned = await self.ocr_processor.is_scanned_pdf(file_path)

        if not is_scanned:
            return

        self.logger.info("Applying OCR to scanned PDF", file=str(file_path))

        # Get page count
        import fitz
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()

        # OCR each page
        for page_num in range(1, page_count + 1):
            ocr_text = await self.ocr_processor.ocr_pdf_page(file_path, page_num)

            if ocr_text.strip():
                chunk = Chunk(
                    content=ocr_text,
                    modality=Modality.TEXT,
                    metadata=ChunkMetadata(
                        doc_id=document.id,
                        page_number=page_num,
                        source_file=str(file_path),
                    ),
                )
                document.chunks.append(chunk)

    async def create_processing_job(
        self,
        file_path: Path,
    ) -> ProcessingJob:
        """
        Create a processing job for tracking.

        Args:
            file_path: Path to file

        Returns:
            Processing job
        """
        return ProcessingJob(
            document_path=str(file_path),
            status=ProcessingStatus.PENDING,
        )

    async def __aenter__(self) -> "ETLPipeline":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
