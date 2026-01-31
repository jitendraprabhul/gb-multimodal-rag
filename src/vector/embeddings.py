"""
Embedding generation for multimodal content.

Supports:
- Text embeddings (BGE, MiniLM)
- Image embeddings (CLIP)
- Batch processing
- Caching
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.core.exceptions import EmbeddingError
from src.core.logging import LoggerMixin
from src.core.types import Chunk, Modality


class BaseEmbedder(ABC, LoggerMixin):
    """Abstract base class for embedders."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        **config: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.config = config
        self._model = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Load embedding model."""
        pass

    @abstractmethod
    async def embed(self, inputs: list[Any]) -> np.ndarray:
        """Generate embeddings for inputs."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._model = None
        self._initialized = False


class TextEmbedder(BaseEmbedder):
    """
    Text embedder using sentence-transformers.

    Supports BGE, MiniLM, and other sentence-transformer models.
    """

    # Default embedding dimensions for common models
    MODEL_DIMENSIONS = {
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-large-en-v1.5": 1024,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        **config: Any,
    ) -> None:
        """
        Initialize text embedder.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cpu, cuda, mps)
            batch_size: Batch size for embedding
            normalize: Whether to normalize embeddings
            **config: Additional configuration
        """
        super().__init__(model_name, device, **config)
        self.batch_size = batch_size
        self.normalize = normalize
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 768)

    async def initialize(self) -> None:
        """Load sentence-transformers model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            # Get actual dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()

            self._initialized = True
            self.logger.info(
                "Text embedder initialized",
                model=self.model_name,
                device=self.device,
                dimension=self._dimension,
            )

        except Exception as e:
            raise EmbeddingError(
                f"Failed to load text embedding model: {e}",
                model_name=self.model_name,
                modality="text",
                cause=e,
            )

    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of shape (n_texts, dimension)
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return np.array([])

        try:
            # Filter out empty texts
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                return np.zeros((len(texts), self._dimension))

            embeddings = self._model.encode(
                valid_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )

            return np.array(embeddings)

        except Exception as e:
            raise EmbeddingError(
                f"Text embedding failed: {e}",
                model_name=self.model_name,
                modality="text",
                cause=e,
            )

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        embeddings = await self.embed([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension


class ImageEmbedder(BaseEmbedder):
    """
    Image embedder using CLIP.

    Generates embeddings for images that can be compared
    with text embeddings in the same vector space.
    """

    MODEL_DIMENSIONS = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-base-patch16": 512,
        "openai/clip-vit-large-patch14": 768,
        "clip-ViT-B-32": 512,
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        batch_size: int = 16,
        **config: Any,
    ) -> None:
        """
        Initialize image embedder.

        Args:
            model_name: CLIP model name
            device: Device to use
            batch_size: Batch size for embedding
            **config: Additional configuration
        """
        super().__init__(model_name, device, **config)
        self.batch_size = batch_size
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 512)
        self._processor = None

    async def initialize(self) -> None:
        """Load CLIP model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # Use sentence-transformers for CLIP
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            self._initialized = True
            self.logger.info(
                "Image embedder initialized",
                model=self.model_name,
                device=self.device,
            )

        except Exception as e:
            # Fallback to transformers CLIP
            try:
                from transformers import CLIPModel, CLIPProcessor

                self._model = CLIPModel.from_pretrained(self.model_name)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)

                if self.device == "cuda":
                    self._model = self._model.cuda()

                self._initialized = True
                self.logger.info(
                    "Image embedder initialized (transformers)",
                    model=self.model_name,
                )

            except Exception as e2:
                raise EmbeddingError(
                    f"Failed to load image embedding model: {e2}",
                    model_name=self.model_name,
                    modality="image",
                    cause=e2,
                )

    async def embed(self, images: list[Image.Image | str | Path]) -> np.ndarray:
        """
        Generate embeddings for images.

        Args:
            images: List of PIL Images or image paths

        Returns:
            Numpy array of shape (n_images, dimension)
        """
        if not self._initialized:
            await self.initialize()

        if not images:
            return np.array([])

        try:
            # Load images if paths provided
            loaded_images = []
            for img in images:
                if isinstance(img, (str, Path)):
                    loaded_images.append(Image.open(img).convert("RGB"))
                else:
                    loaded_images.append(img.convert("RGB"))

            if self._processor:
                # Using transformers CLIP
                import torch

                inputs = self._processor(
                    images=loaded_images,
                    return_tensors="pt",
                    padding=True,
                )

                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model.get_image_features(**inputs)
                    embeddings = outputs.cpu().numpy()

            else:
                # Using sentence-transformers
                embeddings = self._model.encode(
                    loaded_images,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

            return np.array(embeddings)

        except Exception as e:
            raise EmbeddingError(
                f"Image embedding failed: {e}",
                model_name=self.model_name,
                modality="image",
                cause=e,
            )

    async def embed_single(self, image: Image.Image | str | Path) -> np.ndarray:
        """Embed a single image."""
        embeddings = await self.embed([image])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingService(LoggerMixin):
    """
    Unified embedding service for multimodal content.

    Manages text and image embedders, handles batching,
    and provides a consistent interface for embedding generation.
    """

    def __init__(
        self,
        text_model: str = "BAAI/bge-base-en-v1.5",
        image_model: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        batch_size: int = 32,
        **config: Any,
    ) -> None:
        """
        Initialize embedding service.

        Args:
            text_model: Text embedding model
            image_model: Image embedding model
            device: Device for computation
            batch_size: Batch size for embedding
            **config: Additional configuration
        """
        self.text_embedder = TextEmbedder(
            model_name=text_model,
            device=device,
            batch_size=batch_size,
        )
        self.image_embedder = ImageEmbedder(
            model_name=image_model,
            device=device,
            batch_size=batch_size,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all embedders."""
        if self._initialized:
            return

        await self.text_embedder.initialize()
        await self.image_embedder.initialize()

        self._initialized = True
        self.logger.info("Embedding service initialized")

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.text_embedder.cleanup()
        await self.image_embedder.cleanup()
        self._initialized = False

    async def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: Chunks to embed

        Returns:
            Chunks with embeddings populated
        """
        if not self._initialized:
            await self.initialize()

        # Group chunks by modality
        text_chunks = []
        table_chunks = []
        image_chunks = []

        for chunk in chunks:
            if chunk.modality == Modality.TEXT:
                text_chunks.append(chunk)
            elif chunk.modality == Modality.TABLE:
                table_chunks.append(chunk)
            elif chunk.modality == Modality.IMAGE:
                image_chunks.append(chunk)

        # Embed text chunks
        if text_chunks:
            texts = [c.content for c in text_chunks]
            embeddings = await self.text_embedder.embed(texts)
            for chunk, embedding in zip(text_chunks, embeddings):
                chunk.embedding = embedding.tolist()

        # Embed table chunks (as text)
        if table_chunks:
            texts = [c.content for c in table_chunks]
            embeddings = await self.text_embedder.embed(texts)
            for chunk, embedding in zip(table_chunks, embeddings):
                chunk.embedding = embedding.tolist()

        # Embed image chunks
        if image_chunks:
            # For image chunks, we need the image path from metadata
            image_paths = []
            valid_chunks = []

            for chunk in image_chunks:
                image_id = chunk.metadata.image_id
                if image_id:
                    # Try to find image path (would need to be stored)
                    # For now, use content as text fallback
                    valid_chunks.append(chunk)
                    image_paths.append(chunk.content)

            if valid_chunks:
                # Fallback to text embedding for image descriptions
                texts = [c.content for c in valid_chunks]
                embeddings = await self.text_embedder.embed(texts)
                for chunk, embedding in zip(valid_chunks, embeddings):
                    chunk.embedding = embedding.tolist()

        return chunks

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        if not self._initialized:
            await self.initialize()

        return await self.text_embedder.embed_single(query)

    async def embed_image(self, image: Image.Image | str | Path) -> np.ndarray:
        """
        Generate embedding for an image.

        Args:
            image: Image or path

        Returns:
            Image embedding
        """
        if not self._initialized:
            await self.initialize()

        return await self.image_embedder.embed_single(image)

    @property
    def text_dimension(self) -> int:
        """Get text embedding dimension."""
        return self.text_embedder.dimension

    @property
    def image_dimension(self) -> int:
        """Get image embedding dimension."""
        return self.image_embedder.dimension

    async def __aenter__(self) -> "EmbeddingService":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
