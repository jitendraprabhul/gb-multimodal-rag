"""
Image processing and layout detection.

Handles:
- Image loading and preprocessing
- Layout detection (text regions, figures, tables)
- Image classification (chart, diagram, photo, etc.)
- Feature extraction for embeddings
"""

import io
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from src.core.exceptions import DocumentProcessingError
from src.core.types import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    ImageRegion,
    Modality,
)
from src.etl.base import BaseProcessor


class ImageProcessor(BaseProcessor[list[ImageRegion]]):
    """
    Processor for images with layout detection.

    Uses layoutparser for document layout analysis and
    prepares images for CLIP embedding.
    """

    supported_types = [DocumentType.IMAGE]

    def __init__(
        self,
        detect_layout: bool = True,
        resize_max: int = 1024,
        output_dir: Path | None = None,
        **config: Any,
    ) -> None:
        """
        Initialize image processor.

        Args:
            detect_layout: Whether to detect document layout
            resize_max: Maximum dimension for resizing
            output_dir: Directory for processed images
            **config: Additional configuration
        """
        super().__init__(**config)
        self.detect_layout = detect_layout
        self.resize_max = resize_max
        self.output_dir = output_dir or Path("./data/processed/images")
        self._layout_model = None

    async def initialize(self) -> None:
        """Initialize layout detection model."""
        await super().initialize()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.detect_layout:
            try:
                self._load_layout_model()
            except Exception as e:
                self.logger.warning(
                    "Layout detection unavailable",
                    error=str(e),
                )
                self._layout_model = None

    def _load_layout_model(self) -> None:
        """Load layout detection model."""
        try:
            import layoutparser as lp

            # Use PubLayNet model for document layout detection
            self._layout_model = lp.Detectron2LayoutModel(
                config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            )
        except Exception:
            # Fallback: no layout detection
            self._layout_model = None

    async def process(self, file_path: Path, **kwargs: Any) -> list[ImageRegion]:
        """
        Process an image file.

        Args:
            file_path: Path to image
            **kwargs: Additional options

        Returns:
            List of detected image regions
        """
        self.validate_file(file_path)
        self.logger.info("Processing image", file=str(file_path))

        try:
            # Load and preprocess image
            image = Image.open(file_path)
            image = self._preprocess_image(image)

            # Detect layout regions
            if self._layout_model and self.detect_layout:
                regions = await self._detect_layout(image, file_path)
            else:
                # Single region for whole image
                regions = [
                    ImageRegion(
                        image_path=str(file_path),
                        image_type=self._classify_image(image),
                    )
                ]

            self.logger.info(
                "Image processed",
                file=str(file_path),
                regions=len(regions),
            )
            return regions

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process image: {e}",
                document_path=str(file_path),
                cause=e,
            )

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for processing."""
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large
        max_dim = max(image.size)
        if max_dim > self.resize_max:
            ratio = self.resize_max / max_dim
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    async def _detect_layout(
        self,
        image: Image.Image,
        source_path: Path,
    ) -> list[ImageRegion]:
        """Detect layout regions in image."""
        regions = []

        try:
            import layoutparser as lp

            # Convert PIL to numpy
            image_array = np.array(image)

            # Detect layout
            layout = self._layout_model.detect(image_array)

            for i, block in enumerate(layout):
                # Get region type
                region_type = block.type.lower() if hasattr(block, "type") else "unknown"

                # Skip text regions (will be handled by OCR)
                if region_type in ("text", "title", "list"):
                    continue

                # Extract region
                x1, y1, x2, y2 = map(int, block.coordinates)
                region_image = image.crop((x1, y1, x2, y2))

                # Save region
                region_filename = f"{source_path.stem}_region_{i}.png"
                region_path = self.output_dir / region_filename
                region_image.save(region_path)

                region = ImageRegion(
                    image_path=str(region_path),
                    bbox=(x1, y1, x2, y2),
                    image_type=region_type,
                )
                regions.append(region)

        except Exception as e:
            self.logger.warning("Layout detection failed", error=str(e))
            # Return whole image as single region
            regions = [
                ImageRegion(
                    image_path=str(source_path),
                    image_type=self._classify_image(image),
                )
            ]

        return regions

    def _classify_image(self, image: Image.Image) -> str:
        """
        Simple heuristic image classification.

        Returns: 'chart', 'diagram', 'photo', or 'other'
        """
        # Convert to numpy for analysis
        arr = np.array(image)

        # Calculate color statistics
        unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
        total_pixels = arr.shape[0] * arr.shape[1]
        color_ratio = unique_colors / total_pixels

        # Heuristics:
        # - Charts/diagrams: fewer unique colors, more uniform
        # - Photos: many unique colors, complex textures

        if color_ratio < 0.01:
            return "diagram"
        elif color_ratio < 0.1:
            return "chart"
        else:
            return "photo"

    def prepare_for_clip(self, image_path: str | Path) -> Image.Image:
        """
        Prepare image for CLIP embedding.

        Args:
            image_path: Path to image

        Returns:
            Preprocessed PIL Image
        """
        image = Image.open(image_path)
        image = self._preprocess_image(image)

        # CLIP expects 224x224 images
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        return image


class ImageDocumentProcessor(BaseProcessor[Document]):
    """
    Processor for image documents (standalone images).

    Converts images to Document with image chunks.
    """

    supported_types = [DocumentType.IMAGE]

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        self.image_processor = ImageProcessor(**config)

    async def initialize(self) -> None:
        await super().initialize()
        await self.image_processor.initialize()

    async def process(self, file_path: Path, **kwargs: Any) -> Document:
        """Process an image as a document."""
        self.validate_file(file_path)

        try:
            regions = await self.image_processor.process(file_path)

            # Convert regions to chunks
            chunks = []
            for region in regions:
                # Use OCR text or image description as content
                content = region.ocr_text or region.caption or f"Image: {region.image_type}"

                chunk = Chunk(
                    content=content,
                    modality=Modality.IMAGE,
                    metadata=ChunkMetadata(
                        doc_id="",
                        image_id=region.id,
                        source_file=str(file_path),
                    ),
                )
                chunks.append(chunk)

            return Document(
                filename=file_path.name,
                doc_type=DocumentType.IMAGE,
                chunks=chunks,
                metadata={
                    "image_count": len(regions),
                    "images": [r.model_dump() for r in regions],
                },
            )

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process image document: {e}",
                document_path=str(file_path),
                cause=e,
            )
