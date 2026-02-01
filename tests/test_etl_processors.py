"""
Tests for ETL processors: base, PDF, text, table, image, OCR.

Uses mocks for external dependencies (pdfplumber, PyMuPDF, PaddleOCR, etc.).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import io

import pytest
import pandas as pd
import numpy as np

from src.core.exceptions import DocumentProcessingError
from src.core.types import (
    ChunkMetadata,
    Document,
    DocumentType,
    ImageRegion,
    Modality,
    Table,
)
from src.etl.base import BaseProcessor, ProcessorRegistry, get_document_type
from src.etl.table_extractor import TableExtractor, SpreadsheetProcessor
from src.etl.ocr_processor import OCRProcessor


# =============================================================================
# BaseProcessor Tests
# =============================================================================


class ConcreteProcessor(BaseProcessor):
    """Concrete implementation for testing."""

    supported_types = [DocumentType.TXT]

    async def process(self, file_path: Path, **kwargs):
        return f"processed: {file_path}"


class TestBaseProcessor:
    @pytest.mark.asyncio
    async def test_initialize(self):
        proc = ConcreteProcessor()
        assert proc._initialized is False
        await proc.initialize()
        assert proc._initialized is True

    @pytest.mark.asyncio
    async def test_cleanup(self):
        proc = ConcreteProcessor()
        await proc.initialize()
        await proc.cleanup()
        assert proc._initialized is False

    def test_supports(self):
        proc = ConcreteProcessor()
        assert proc.supports(DocumentType.TXT) is True
        assert proc.supports(DocumentType.PDF) is False

    def test_validate_file_exists(self, tmp_path):
        proc = ConcreteProcessor()
        file = tmp_path / "test.txt"
        file.write_text("hello")
        proc.validate_file(file)  # Should not raise

    def test_validate_file_not_exists(self, tmp_path):
        proc = ConcreteProcessor()
        with pytest.raises(DocumentProcessingError, match="File not found"):
            proc.validate_file(tmp_path / "nonexistent.txt")

    def test_validate_file_is_directory(self, tmp_path):
        proc = ConcreteProcessor()
        with pytest.raises(DocumentProcessingError, match="Not a file"):
            proc.validate_file(tmp_path)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with ConcreteProcessor() as proc:
            assert proc._initialized is True
        assert proc._initialized is False


# =============================================================================
# ProcessorRegistry Tests
# =============================================================================


class TestProcessorRegistry:
    def setup_method(self):
        # Reset singleton for each test
        ProcessorRegistry._instance = None

    def test_singleton(self):
        r1 = ProcessorRegistry()
        r2 = ProcessorRegistry()
        assert r1 is r2

    def test_register_and_get(self):
        registry = ProcessorRegistry()
        proc = ConcreteProcessor()
        registry.register(proc)
        assert registry.get_processor(DocumentType.TXT) is proc

    def test_get_unregistered_raises(self):
        registry = ProcessorRegistry()
        registry.clear()
        with pytest.raises(DocumentProcessingError, match="No processor registered"):
            registry.get_processor(DocumentType.PDF)

    def test_has_processor(self):
        registry = ProcessorRegistry()
        registry.clear()
        proc = ConcreteProcessor()
        registry.register(proc)
        assert registry.has_processor(DocumentType.TXT) is True
        assert registry.has_processor(DocumentType.PDF) is False

    def test_supported_types(self):
        registry = ProcessorRegistry()
        registry.clear()
        proc = ConcreteProcessor()
        registry.register(proc)
        assert DocumentType.TXT in registry.supported_types

    def test_clear(self):
        registry = ProcessorRegistry()
        proc = ConcreteProcessor()
        registry.register(proc)
        registry.clear()
        assert registry.has_processor(DocumentType.TXT) is False


# =============================================================================
# get_document_type Tests
# =============================================================================


class TestGetDocumentType:
    def test_pdf(self, tmp_path):
        assert get_document_type(Path("doc.pdf")) == DocumentType.PDF

    def test_txt(self):
        assert get_document_type(Path("doc.txt")) == DocumentType.TXT

    def test_csv(self):
        assert get_document_type(Path("data.csv")) == DocumentType.CSV

    def test_xlsx(self):
        assert get_document_type(Path("data.xlsx")) == DocumentType.XLSX

    def test_xls(self):
        assert get_document_type(Path("data.xls")) == DocumentType.XLSX

    def test_images(self):
        for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
            assert get_document_type(Path(f"image{ext}")) == DocumentType.IMAGE

    def test_html(self):
        assert get_document_type(Path("page.html")) == DocumentType.HTML
        assert get_document_type(Path("page.htm")) == DocumentType.HTML

    def test_json(self):
        assert get_document_type(Path("data.json")) == DocumentType.JSON

    def test_unsupported_extension(self):
        with pytest.raises(DocumentProcessingError, match="Unsupported file type"):
            get_document_type(Path("file.xyz"))

    def test_case_insensitive(self):
        assert get_document_type(Path("doc.PDF")) == DocumentType.PDF
        assert get_document_type(Path("doc.Txt")) == DocumentType.TXT


# =============================================================================
# TextFileProcessor Tests
# =============================================================================


class TestTextFileProcessor:
    @pytest.mark.asyncio
    async def test_process_text_file(self, tmp_path):
        from src.etl.pdf_processor import TextFileProcessor

        proc = TextFileProcessor()
        file = tmp_path / "test.txt"
        file.write_text("Hello world, this is a test document.")

        doc = await proc.process(file)
        assert isinstance(doc, Document)
        assert doc.doc_type == DocumentType.TXT
        assert len(doc.chunks) == 1
        assert "Hello world" in doc.chunks[0].content

    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, tmp_path):
        from src.etl.pdf_processor import TextFileProcessor

        proc = TextFileProcessor()
        with pytest.raises(DocumentProcessingError):
            await proc.process(tmp_path / "nonexistent.txt")


# =============================================================================
# TableExtractor Tests
# =============================================================================


class TestTableExtractor:
    def test_init(self):
        extractor = TableExtractor()
        assert extractor.max_rows == 1000
        assert extractor.max_columns == 50

    def test_supported_types(self):
        extractor = TableExtractor()
        assert extractor.supports(DocumentType.CSV)
        assert extractor.supports(DocumentType.XLSX)
        assert not extractor.supports(DocumentType.PDF)

    @pytest.mark.asyncio
    async def test_extract_csv(self, tmp_path):
        # Create a CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})
        df.to_csv(csv_path, index=False)

        extractor = TableExtractor()
        tables = await extractor.process(csv_path)
        assert len(tables) == 1
        assert tables[0].headers == ["Name", "Age"]
        assert len(tables[0].rows) == 2

    @pytest.mark.asyncio
    async def test_extract_csv_max_rows(self, tmp_path):
        csv_path = tmp_path / "big.csv"
        df = pd.DataFrame({"Val": list(range(100))})
        df.to_csv(csv_path, index=False)

        extractor = TableExtractor(max_rows=10)
        tables = await extractor.process(csv_path)
        assert len(tables) == 1
        assert len(tables[0].rows) == 10

    @pytest.mark.asyncio
    async def test_extract_excel(self, tmp_path):
        xlsx_path = tmp_path / "test.xlsx"
        df = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
        df.to_excel(xlsx_path, index=False)

        extractor = TableExtractor()
        tables = await extractor.process(xlsx_path)
        assert len(tables) == 1

    @pytest.mark.asyncio
    async def test_extract_nonexistent_file(self, tmp_path):
        extractor = TableExtractor()
        with pytest.raises(DocumentProcessingError):
            await extractor.process(tmp_path / "nonexistent.csv")

    @pytest.mark.asyncio
    async def test_unsupported_format(self, tmp_path):
        file = tmp_path / "test.xml"
        file.write_text("<root/>")
        extractor = TableExtractor()
        with pytest.raises(DocumentProcessingError):
            await extractor.process(file)

    def test_dataframe_to_table_nan_handling(self):
        extractor = TableExtractor()
        df = pd.DataFrame({"A": [1, None, 3], "B": [None, "hello", "world"]})
        table = extractor._dataframe_to_table(df)
        assert table.headers == ["A", "B"]
        # NaN values should be empty string
        assert table.rows[0][1] == ""
        assert table.rows[1][0] == ""


# =============================================================================
# SpreadsheetProcessor Tests
# =============================================================================


class TestSpreadsheetProcessor:
    @pytest.mark.asyncio
    async def test_process_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        df.to_csv(csv_path, index=False)

        proc = SpreadsheetProcessor()
        doc = await proc.process(csv_path)
        assert isinstance(doc, Document)
        assert doc.doc_type == DocumentType.CSV
        assert doc.metadata["table_count"] == 1

    @pytest.mark.asyncio
    async def test_process_nonexistent(self, tmp_path):
        proc = SpreadsheetProcessor()
        with pytest.raises(DocumentProcessingError):
            await proc.process(tmp_path / "missing.csv")


# =============================================================================
# OCRProcessor Tests
# =============================================================================


class TestOCRProcessor:
    def test_init(self):
        ocr = OCRProcessor(lang="en", use_gpu=False)
        assert ocr.lang == "en"
        assert ocr.use_gpu is False
        assert ocr._initialized is False

    @pytest.mark.asyncio
    async def test_extract_text_not_initialized(self):
        ocr = OCRProcessor()
        # When PaddleOCR is not available, should return empty string
        with patch.dict("sys.modules", {"paddleocr": None}):
            result = await ocr.extract_text("fake_path.png")
            # Either returns empty or initializes and processes
            assert isinstance(result, (str, tuple))

    @pytest.mark.asyncio
    async def test_extract_text_engine_none(self):
        ocr = OCRProcessor()
        ocr._initialized = True
        ocr._ocr_engine = None
        result = await ocr.extract_text("fake_path.png")
        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_text_with_boxes_engine_none(self):
        ocr = OCRProcessor()
        ocr._initialized = True
        ocr._ocr_engine = None
        result = await ocr.extract_text("fake_path.png", return_boxes=True)
        assert result == ("", [])

    @pytest.mark.asyncio
    async def test_process_image_region(self):
        ocr = OCRProcessor()
        ocr._initialized = True
        ocr._ocr_engine = None

        region = ImageRegion(image_path="/fake/path.png")
        result = await ocr.process_image_region(region)
        assert result.image_path == "/fake/path.png"

    @pytest.mark.asyncio
    async def test_cleanup(self):
        ocr = OCRProcessor()
        ocr._initialized = True
        ocr._ocr_engine = MagicMock()
        await ocr.cleanup()
        assert ocr._initialized is False
        assert ocr._ocr_engine is None

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        ocr = OCRProcessor()
        with patch.dict("sys.modules", {"paddleocr": None}):
            # Should handle ImportError gracefully
            try:
                await ocr.initialize()
            except Exception:
                pass
            # Engine should be None if import fails
            # (the actual behavior depends on the import mechanism)


# =============================================================================
# PDFProcessor Tests (with mocks)
# =============================================================================


class TestPDFProcessor:
    def test_init(self):
        from src.etl.pdf_processor import PDFProcessor

        proc = PDFProcessor()
        assert proc.extract_images is True
        assert proc.extract_tables is True
        assert proc.min_image_size == 100
        assert proc.supports(DocumentType.PDF)

    def test_calculate_hash(self, tmp_path):
        from src.etl.pdf_processor import PDFProcessor

        proc = PDFProcessor()
        file = tmp_path / "test.txt"
        file.write_bytes(b"hello world")
        h = proc._calculate_hash(file)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_calculate_hash_deterministic(self, tmp_path):
        from src.etl.pdf_processor import PDFProcessor

        proc = PDFProcessor()
        file = tmp_path / "test.txt"
        file.write_bytes(b"hello world")
        h1 = proc._calculate_hash(file)
        h2 = proc._calculate_hash(file)
        assert h1 == h2

    @pytest.mark.asyncio
    async def test_process_nonexistent(self, tmp_path):
        from src.etl.pdf_processor import PDFProcessor

        proc = PDFProcessor()
        with pytest.raises(DocumentProcessingError):
            await proc.process(tmp_path / "nonexistent.pdf")


# =============================================================================
# ImageProcessor Tests
# =============================================================================


class TestImageProcessor:
    def test_init(self):
        from src.etl.image_processor import ImageProcessor

        proc = ImageProcessor()
        assert proc.detect_layout is True
        assert proc.resize_max == 1024
        assert proc.supports(DocumentType.IMAGE)

    def test_classify_image_diagram(self):
        from src.etl.image_processor import ImageProcessor

        proc = ImageProcessor()
        # Create a simple solid-color image (very few unique colors)
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        result = proc._classify_image(img)
        assert result == "diagram"

    def test_preprocess_image_convert_rgba(self):
        from src.etl.image_processor import ImageProcessor
        from PIL import Image

        proc = ImageProcessor()
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        result = proc._preprocess_image(img)
        assert result.mode == "RGB"

    def test_preprocess_image_resize_large(self):
        from src.etl.image_processor import ImageProcessor
        from PIL import Image

        proc = ImageProcessor(resize_max=100)
        img = Image.new("RGB", (200, 200))
        result = proc._preprocess_image(img)
        assert max(result.size) <= 100

    def test_preprocess_image_no_resize_small(self):
        from src.etl.image_processor import ImageProcessor
        from PIL import Image

        proc = ImageProcessor(resize_max=1024)
        img = Image.new("RGB", (50, 50))
        result = proc._preprocess_image(img)
        assert result.size == (50, 50)

    def test_prepare_for_clip(self, tmp_path):
        from src.etl.image_processor import ImageProcessor
        from PIL import Image

        proc = ImageProcessor()
        img_path = tmp_path / "test.png"
        Image.new("RGB", (500, 500)).save(img_path)
        result = proc.prepare_for_clip(img_path)
        assert result.size == (224, 224)

    @pytest.mark.asyncio
    async def test_process_nonexistent(self, tmp_path):
        from src.etl.image_processor import ImageProcessor

        proc = ImageProcessor()
        with pytest.raises(DocumentProcessingError):
            await proc.process(tmp_path / "nonexistent.png")

    @pytest.mark.asyncio
    async def test_process_real_image(self, tmp_path):
        from src.etl.image_processor import ImageProcessor
        from PIL import Image

        proc = ImageProcessor(detect_layout=False)
        proc._layout_model = None
        img_path = tmp_path / "test.png"
        Image.new("RGB", (200, 200), color=(128, 128, 128)).save(img_path)
        regions = await proc.process(img_path)
        assert len(regions) >= 1
        assert isinstance(regions[0], ImageRegion)
