"""
PDF document processor using pdfplumber and PyMuPDF.

Extracts:
- Text content with page/section metadata
- Tables
- Images
- Document structure
"""

import hashlib
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from src.core.exceptions import DocumentProcessingError
from src.core.types import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    ImageRegion,
    Modality,
    Table,
)
from src.etl.base import BaseProcessor


class PDFProcessor(BaseProcessor[Document]):
    """
    Processor for PDF documents.

    Uses pdfplumber for text/table extraction and PyMuPDF for images.
    Handles both digital and scanned PDFs (with OCR integration).
    """

    supported_types = [DocumentType.PDF]

    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        min_image_size: int = 100,
        image_output_dir: Path | None = None,
        **config: Any,
    ) -> None:
        """
        Initialize PDF processor.

        Args:
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            min_image_size: Minimum image dimension to extract
            image_output_dir: Directory to save extracted images
            **config: Additional configuration
        """
        super().__init__(**config)
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.min_image_size = min_image_size
        self.image_output_dir = image_output_dir or Path("./data/processed/images")

    async def initialize(self) -> None:
        """Ensure output directory exists."""
        await super().initialize()
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, file_path: Path, **kwargs: Any) -> Document:
        """
        Process a PDF file.

        Args:
            file_path: Path to PDF file
            **kwargs: Additional options

        Returns:
            Processed Document with chunks, tables, and images

        Raises:
            DocumentProcessingError: If processing fails
        """
        self.validate_file(file_path)
        self.logger.info("Processing PDF", file=str(file_path))

        try:
            # Calculate file hash for deduplication
            file_hash = self._calculate_hash(file_path)

            # Extract content
            text_chunks = await self._extract_text(file_path)
            tables = await self._extract_tables(file_path) if self.extract_tables else []
            images = await self._extract_images(file_path) if self.extract_images else []

            # Get document title
            title = await self._extract_title(file_path)

            # Create document
            doc = Document(
                filename=file_path.name,
                doc_type=DocumentType.PDF,
                title=title,
                chunks=text_chunks,
                file_hash=file_hash,
                metadata={
                    "page_count": len(set(c.metadata.page_number for c in text_chunks if c.metadata.page_number)),
                    "table_count": len(tables),
                    "image_count": len(images),
                    "tables": [t.model_dump() for t in tables],
                    "images": [i.model_dump() for i in images],
                },
            )

            self.logger.info(
                "PDF processed",
                file=str(file_path),
                chunks=len(text_chunks),
                tables=len(tables),
                images=len(images),
            )

            return doc

        except Exception as e:
            self.logger.error("PDF processing failed", file=str(file_path), error=str(e))
            raise DocumentProcessingError(
                f"Failed to process PDF: {e}",
                document_path=str(file_path),
                document_type="pdf",
                cause=e,
            )

    async def _extract_text(self, file_path: Path) -> list[Chunk]:
        """Extract text content from PDF pages."""
        chunks = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                if not text.strip():
                    continue

                # Create chunk for the page
                chunk = Chunk(
                    content=text,
                    modality=Modality.TEXT,
                    metadata=ChunkMetadata(
                        doc_id="",  # Will be set by document
                        page_number=page_num,
                        source_file=str(file_path),
                        start_char=0,
                        end_char=len(text),
                    ),
                )
                chunks.append(chunk)

        return chunks

    async def _extract_tables(self, file_path: Path) -> list[Table]:
        """Extract tables from PDF."""
        tables = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()

                for table_idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue

                    # First row as headers
                    headers = [str(cell or "") for cell in table_data[0]]

                    # Rest as rows
                    rows = [
                        [str(cell or "") for cell in row]
                        for row in table_data[1:]
                    ]

                    table = Table(
                        headers=headers,
                        rows=rows,
                        page_number=page_num,
                    )
                    tables.append(table)

        return tables

    async def _extract_images(self, file_path: Path) -> list[ImageRegion]:
        """Extract images from PDF using PyMuPDF."""
        images = []

        doc = fitz.open(file_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()

                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]

                    try:
                        # Extract image
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Check image size
                        img = Image.open(__import__("io").BytesIO(image_bytes))
                        if img.width < self.min_image_size or img.height < self.min_image_size:
                            continue

                        # Save image
                        image_filename = f"{file_path.stem}_p{page_num + 1}_img{img_idx}.{image_ext}"
                        image_path = self.image_output_dir / image_filename
                        img.save(image_path)

                        # Create image region
                        image_region = ImageRegion(
                            image_path=str(image_path),
                            page_number=page_num + 1,
                        )
                        images.append(image_region)

                    except Exception as e:
                        self.logger.warning(
                            "Failed to extract image",
                            page=page_num + 1,
                            error=str(e),
                        )
                        continue

        finally:
            doc.close()

        return images

    async def _extract_title(self, file_path: Path) -> str | None:
        """Extract document title from PDF metadata or first page."""
        try:
            doc = fitz.open(file_path)
            try:
                # Try metadata
                metadata = doc.metadata
                if metadata and metadata.get("title"):
                    return metadata["title"]

                # Try first page header
                if len(doc) > 0:
                    first_page = doc[0]
                    text = first_page.get_text()
                    lines = text.strip().split("\n")
                    if lines:
                        # Return first non-empty line as potential title
                        for line in lines[:5]:
                            line = line.strip()
                            if len(line) > 5 and len(line) < 200:
                                return line

            finally:
                doc.close()

        except Exception:
            pass

        return None

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class TextFileProcessor(BaseProcessor[Document]):
    """Processor for plain text files."""

    supported_types = [DocumentType.TXT]

    async def process(self, file_path: Path, **kwargs: Any) -> Document:
        """Process a text file."""
        self.validate_file(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunk = Chunk(
                content=content,
                modality=Modality.TEXT,
                metadata=ChunkMetadata(
                    doc_id="",
                    source_file=str(file_path),
                ),
            )

            return Document(
                filename=file_path.name,
                doc_type=DocumentType.TXT,
                chunks=[chunk],
            )

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process text file: {e}",
                document_path=str(file_path),
                cause=e,
            )
