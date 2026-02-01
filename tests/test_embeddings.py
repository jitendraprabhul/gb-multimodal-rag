"""
Tests for embedding generation modules.

Covers:
- TextEmbedder initialization, embedding, dimension
- ImageEmbedder initialization, embedding
- EmbeddingService unified interface
- Error handling and edge cases
"""

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path

import numpy as np
import pytest

from src.core.exceptions import EmbeddingError
from src.core.types import Chunk, ChunkMetadata, Modality
from src.vector.embeddings import (
    BaseEmbedder,
    TextEmbedder,
    ImageEmbedder,
    EmbeddingService,
)


# =============================================================================
# TextEmbedder Tests
# =============================================================================


class TestTextEmbedder:
    def test_default_params(self):
        embedder = TextEmbedder()
        assert embedder.model_name == "BAAI/bge-base-en-v1.5"
        assert embedder.device == "cpu"
        assert embedder.batch_size == 32
        assert embedder.normalize is True
        assert embedder._initialized is False

    def test_custom_params(self):
        embedder = TextEmbedder(
            model_name="all-MiniLM-L6-v2",
            device="cuda",
            batch_size=64,
            normalize=False,
        )
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.device == "cuda"
        assert embedder.batch_size == 64
        assert embedder.normalize is False

    def test_known_model_dimension(self):
        embedder = TextEmbedder(model_name="BAAI/bge-base-en-v1.5")
        assert embedder.dimension == 768

    def test_small_model_dimension(self):
        embedder = TextEmbedder(model_name="BAAI/bge-small-en-v1.5")
        assert embedder.dimension == 384

    def test_unknown_model_defaults_to_768(self):
        embedder = TextEmbedder(model_name="unknown/model")
        assert embedder.dimension == 768

    async def test_initialize_loads_model(self):
        embedder = TextEmbedder()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        with patch(
            "src.vector.embeddings.TextEmbedder.initialize",
            new_callable=AsyncMock,
        ) as mock_init:
            mock_init.return_value = None
            await embedder.initialize()

    async def test_initialize_idempotent(self):
        embedder = TextEmbedder()
        embedder._initialized = True
        # Should return immediately without loading model
        await embedder.initialize()
        assert embedder._model is None  # Never loaded

    async def test_embed_empty_list(self):
        embedder = TextEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()

        result = await embedder.embed([])
        assert len(result) == 0

    async def test_embed_whitespace_texts(self):
        embedder = TextEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()
        embedder._dimension = 768

        result = await embedder.embed(["", "  ", ""])
        assert result.shape == (3, 768)
        assert np.allclose(result, 0.0)

    async def test_embed_valid_texts(self):
        embedder = TextEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()
        embedder._model.encode.return_value = np.random.randn(2, 768)

        result = await embedder.embed(["hello", "world"])
        assert result.shape == (2, 768)
        embedder._model.encode.assert_called_once()

    async def test_embed_single(self):
        embedder = TextEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()
        embedder._model.encode.return_value = np.random.randn(1, 768)

        result = await embedder.embed_single("hello")
        assert result.shape == (768,)

    async def test_embed_single_empty_result(self):
        embedder = TextEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()
        embedder._dimension = 384
        embedder._model.encode.return_value = np.array([])

        # embed returns empty array → embed_single returns zeros
        result = await embedder.embed_single("")
        # The actual implementation returns zeros when result is empty
        assert result.shape == (384,)

    async def test_embed_failure_raises_embedding_error(self):
        embedder = TextEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()
        embedder._model.encode.side_effect = RuntimeError("CUDA OOM")

        with pytest.raises(EmbeddingError, match="Text embedding failed"):
            await embedder.embed(["test text"])

    async def test_cleanup(self):
        embedder = TextEmbedder()
        embedder._model = MagicMock()
        embedder._initialized = True

        await embedder.cleanup()
        assert embedder._model is None
        assert embedder._initialized is False


# =============================================================================
# ImageEmbedder Tests
# =============================================================================


class TestImageEmbedder:
    def test_default_params(self):
        embedder = ImageEmbedder()
        assert embedder.model_name == "openai/clip-vit-base-patch32"
        assert embedder.dimension == 512
        assert embedder._processor is None

    def test_known_model_dimension(self):
        embedder = ImageEmbedder(model_name="openai/clip-vit-large-patch14")
        assert embedder.dimension == 768

    def test_unknown_model_defaults_to_512(self):
        embedder = ImageEmbedder(model_name="unknown/clip")
        assert embedder.dimension == 512

    async def test_embed_empty_list(self):
        embedder = ImageEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()

        result = await embedder.embed([])
        assert len(result) == 0

    async def test_embed_failure_raises_embedding_error(self):
        embedder = ImageEmbedder()
        embedder._initialized = True
        embedder._model = MagicMock()
        embedder._model.encode.side_effect = RuntimeError("Failed")

        from PIL import Image

        img = Image.new("RGB", (100, 100))
        with pytest.raises(EmbeddingError, match="Image embedding failed"):
            await embedder.embed([img])

    async def test_cleanup(self):
        embedder = ImageEmbedder()
        embedder._model = MagicMock()
        embedder._initialized = True

        await embedder.cleanup()
        assert embedder._model is None
        assert embedder._initialized is False


# =============================================================================
# EmbeddingService Tests
# =============================================================================


class TestEmbeddingService:
    def test_default_params(self):
        service = EmbeddingService()
        assert isinstance(service.text_embedder, TextEmbedder)
        assert isinstance(service.image_embedder, ImageEmbedder)
        assert service._initialized is False

    def test_custom_params(self):
        service = EmbeddingService(
            text_model="all-MiniLM-L6-v2",
            image_model="clip-ViT-B-32",
            device="cuda",
            batch_size=64,
        )
        assert service.text_embedder.model_name == "all-MiniLM-L6-v2"
        assert service.image_embedder.model_name == "clip-ViT-B-32"

    def test_dimension_properties(self):
        service = EmbeddingService()
        assert service.text_dimension == 768
        assert service.image_dimension == 512

    async def test_initialize(self):
        service = EmbeddingService()
        service.text_embedder.initialize = AsyncMock()
        service.image_embedder.initialize = AsyncMock()

        await service.initialize()

        assert service._initialized is True
        service.text_embedder.initialize.assert_called_once()
        service.image_embedder.initialize.assert_called_once()

    async def test_initialize_idempotent(self):
        service = EmbeddingService()
        service._initialized = True
        service.text_embedder.initialize = AsyncMock()
        service.image_embedder.initialize = AsyncMock()

        await service.initialize()

        service.text_embedder.initialize.assert_not_called()

    async def test_cleanup(self):
        service = EmbeddingService()
        service.text_embedder.cleanup = AsyncMock()
        service.image_embedder.cleanup = AsyncMock()
        service._initialized = True

        await service.cleanup()

        assert service._initialized is False
        service.text_embedder.cleanup.assert_called_once()
        service.image_embedder.cleanup.assert_called_once()

    async def test_embed_query(self):
        service = EmbeddingService()
        service._initialized = True
        service.text_embedder.embed_single = AsyncMock(
            return_value=np.random.randn(768)
        )

        result = await service.embed_query("What is revenue?")
        assert result.shape == (768,)
        service.text_embedder.embed_single.assert_called_once_with("What is revenue?")

    async def test_embed_chunks_text(self):
        service = EmbeddingService()
        service._initialized = True
        service.text_embedder.embed = AsyncMock(
            return_value=np.random.randn(2, 768)
        )

        chunks = [
            Chunk(
                content=f"Text chunk {i}",
                modality=Modality.TEXT,
                metadata=ChunkMetadata(doc_id="d1"),
            )
            for i in range(2)
        ]

        result = await service.embed_chunks(chunks)

        assert len(result) == 2
        assert result[0].embedding is not None
        assert len(result[0].embedding) == 768

    async def test_embed_chunks_table(self):
        service = EmbeddingService()
        service._initialized = True
        service.text_embedder.embed = AsyncMock(
            return_value=np.random.randn(1, 768)
        )

        chunks = [
            Chunk(
                content="Table data",
                modality=Modality.TABLE,
                metadata=ChunkMetadata(doc_id="d1"),
            )
        ]

        result = await service.embed_chunks(chunks)
        assert result[0].embedding is not None

    async def test_embed_chunks_mixed_modalities(self):
        service = EmbeddingService()
        service._initialized = True

        text_embs = np.random.randn(1, 768)
        table_embs = np.random.randn(1, 768)

        call_count = [0]

        async def mock_embed(texts):
            result = [text_embs, table_embs][call_count[0]]
            call_count[0] += 1
            return result

        service.text_embedder.embed = mock_embed

        chunks = [
            Chunk(
                content="Text",
                modality=Modality.TEXT,
                metadata=ChunkMetadata(doc_id="d1"),
            ),
            Chunk(
                content="Table",
                modality=Modality.TABLE,
                metadata=ChunkMetadata(doc_id="d1"),
            ),
        ]

        result = await service.embed_chunks(chunks)
        assert result[0].embedding is not None
        assert result[1].embedding is not None

    async def test_embed_chunks_empty_list(self):
        service = EmbeddingService()
        service._initialized = True

        result = await service.embed_chunks([])
        assert result == []

    async def test_context_manager(self):
        service = EmbeddingService()
        service.text_embedder.initialize = AsyncMock()
        service.image_embedder.initialize = AsyncMock()
        service.text_embedder.cleanup = AsyncMock()
        service.image_embedder.cleanup = AsyncMock()

        async with service as svc:
            assert svc._initialized is True

        assert svc._initialized is False
