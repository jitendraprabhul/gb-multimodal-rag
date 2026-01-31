"""
Multimodal ETL pipeline for document processing.

Handles:
- PDF processing with text extraction
- Table extraction from PDFs and spreadsheets
- Image extraction and processing
- OCR for scanned documents
- Text chunking with metadata
"""

from src.etl.base import BaseProcessor, ProcessorRegistry
from src.etl.chunker import TextChunker, ChunkingStrategy
from src.etl.pdf_processor import PDFProcessor
from src.etl.table_extractor import TableExtractor
from src.etl.image_processor import ImageProcessor
from src.etl.ocr_processor import OCRProcessor
from src.etl.pipeline import ETLPipeline

__all__ = [
    "BaseProcessor",
    "ProcessorRegistry",
    "TextChunker",
    "ChunkingStrategy",
    "PDFProcessor",
    "TableExtractor",
    "ImageProcessor",
    "OCRProcessor",
    "ETLPipeline",
]
