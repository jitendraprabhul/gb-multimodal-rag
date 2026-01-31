"""
OCR processing using PaddleOCR.

Handles:
- Scanned document text extraction
- Image text detection
- Multi-language OCR support
"""

from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from src.core.exceptions import DocumentProcessingError
from src.core.logging import LoggerMixin
from src.core.types import ImageRegion


class OCRProcessor(LoggerMixin):
    """
    OCR processor using PaddleOCR.

    Extracts text from images and scanned documents with
    high accuracy for multiple languages.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        **config: Any,
    ) -> None:
        """
        Initialize OCR processor.

        Args:
            lang: OCR language code
            use_gpu: Whether to use GPU acceleration
            det_model_dir: Custom detection model directory
            rec_model_dir: Custom recognition model directory
            **config: Additional configuration
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.config = config
        self._ocr_engine = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize PaddleOCR engine."""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
                det_model_dir=self.det_model_dir,
                rec_model_dir=self.rec_model_dir,
            )
            self._initialized = True
            self.logger.info("PaddleOCR initialized", lang=self.lang, gpu=self.use_gpu)

        except ImportError:
            self.logger.warning("PaddleOCR not available")
            self._ocr_engine = None
        except Exception as e:
            self.logger.error("Failed to initialize PaddleOCR", error=str(e))
            self._ocr_engine = None

    async def extract_text(
        self,
        image_path: str | Path,
        return_boxes: bool = False,
    ) -> str | tuple[str, list[dict[str, Any]]]:
        """
        Extract text from an image.

        Args:
            image_path: Path to image file
            return_boxes: Whether to return bounding boxes

        Returns:
            Extracted text (and optionally bounding boxes)
        """
        if not self._initialized:
            await self.initialize()

        if self._ocr_engine is None:
            self.logger.warning("OCR engine not available")
            return ("", []) if return_boxes else ""

        try:
            # Convert to numpy array
            image = Image.open(image_path)
            image_array = np.array(image)

            # Run OCR
            result = self._ocr_engine.ocr(image_array, cls=True)

            if not result or not result[0]:
                return ("", []) if return_boxes else ""

            # Extract text and boxes
            texts = []
            boxes = []

            for line in result[0]:
                if line and len(line) >= 2:
                    box = line[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    text_info = line[1]  # (text, confidence)

                    if text_info and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]

                        texts.append(text)
                        boxes.append({
                            "text": text,
                            "confidence": confidence,
                            "box": box,
                        })

            full_text = "\n".join(texts)

            if return_boxes:
                return full_text, boxes
            return full_text

        except Exception as e:
            self.logger.error("OCR extraction failed", error=str(e))
            return ("", []) if return_boxes else ""

    async def process_image_region(
        self,
        region: ImageRegion,
    ) -> ImageRegion:
        """
        Process an image region and add OCR text.

        Args:
            region: Image region to process

        Returns:
            Updated image region with OCR text
        """
        try:
            ocr_text = await self.extract_text(region.image_path)
            region.ocr_text = ocr_text
            return region

        except Exception as e:
            self.logger.warning(
                "Failed to OCR image region",
                region_id=region.id,
                error=str(e),
            )
            return region

    async def is_scanned_pdf(
        self,
        pdf_path: Path,
        sample_pages: int = 3,
        text_threshold: float = 0.1,
    ) -> bool:
        """
        Check if a PDF is scanned (image-based) vs digital.

        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample
            text_threshold: Min text/area ratio to consider digital

        Returns:
            True if PDF appears to be scanned
        """
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                total_text_ratio = 0
                pages_checked = 0

                for i, page in enumerate(pdf.pages[:sample_pages]):
                    text = page.extract_text() or ""
                    text_len = len(text.strip())

                    # Estimate page area
                    page_area = page.width * page.height

                    if page_area > 0:
                        text_ratio = text_len / page_area
                        total_text_ratio += text_ratio
                        pages_checked += 1

                if pages_checked == 0:
                    return True

                avg_ratio = total_text_ratio / pages_checked
                return avg_ratio < text_threshold

        except Exception as e:
            self.logger.warning("Failed to check PDF type", error=str(e))
            return False

    async def ocr_pdf_page(
        self,
        pdf_path: Path,
        page_number: int,
        dpi: int = 300,
    ) -> str:
        """
        OCR a specific PDF page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            dpi: Resolution for rendering

        Returns:
            Extracted text from page
        """
        try:
            import fitz

            doc = fitz.open(pdf_path)
            try:
                if page_number < 1 or page_number > len(doc):
                    return ""

                page = doc[page_number - 1]

                # Render page to image
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                image = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples,
                )

                # Save temporarily and OCR
                temp_path = Path(f"/tmp/ocr_page_{page_number}.png")
                image.save(temp_path)

                text = await self.extract_text(temp_path)

                # Cleanup
                temp_path.unlink(missing_ok=True)

                return text

            finally:
                doc.close()

        except Exception as e:
            self.logger.error(
                "Failed to OCR PDF page",
                page=page_number,
                error=str(e),
            )
            return ""

    async def cleanup(self) -> None:
        """Clean up OCR resources."""
        self._ocr_engine = None
        self._initialized = False
