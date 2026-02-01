"""
Tests for ETL pipeline.
"""

import pytest

from src.etl.chunker import TextChunker, ChunkingStrategy


class TestTextChunking:
    """Tests for text chunking."""

    def test_fixed_size_chunking(self):
        """Test fixed-size chunking."""
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10,  # Lower threshold for testing
            strategy=ChunkingStrategy.FIXED,
        )
        text = "This is a test. " * 20

        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        # Check chunks are within expected size range
        assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)

    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        chunker = TextChunker(
            chunk_size=50,
            min_chunk_size=10,  # Lower threshold for testing
            strategy=ChunkingStrategy.SENTENCE,
        )
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Chunks should contain sentences (periods)
        assert all("." in chunk.content or "sentence" in chunk.content.lower() for chunk in chunks)


@pytest.mark.asyncio
class TestETLPipeline:
    """Tests for ETL pipeline."""

    async def test_process_text_file(self, sample_text_file, test_settings):
        """Test processing a text file."""
        from src.etl.pipeline import ETLPipeline
        from src.core.types import DocumentType

        pipeline = ETLPipeline(settings=test_settings)
        document = await pipeline.process_file(sample_text_file)

        assert document is not None
        assert document.filename == "test.txt"
        assert len(document.chunks) > 0
        assert document.doc_type == DocumentType.TXT
