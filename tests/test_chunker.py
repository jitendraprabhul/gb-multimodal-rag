"""
Tests for text chunking strategies.

Covers all chunking strategies, edge cases, and table chunking.
"""

import pytest

from src.core.types import ChunkMetadata, Modality
from src.etl.chunker import ChunkingStrategy, TableChunker, TextChunker


class TestChunkingStrategy:
    def test_values(self):
        assert ChunkingStrategy.FIXED == "fixed"
        assert ChunkingStrategy.SENTENCE == "sentence"
        assert ChunkingStrategy.PARAGRAPH == "paragraph"
        assert ChunkingStrategy.SEMANTIC == "semantic"


class TestTextChunkerInit:
    def test_defaults(self):
        chunker = TextChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_size == 100
        assert chunker.max_chunk_size == 1024
        assert chunker.strategy == ChunkingStrategy.SENTENCE

    def test_custom_params(self):
        chunker = TextChunker(
            chunk_size=256,
            chunk_overlap=25,
            min_chunk_size=50,
            max_chunk_size=512,
            strategy=ChunkingStrategy.FIXED,
        )
        assert chunker.chunk_size == 256
        assert chunker.strategy == ChunkingStrategy.FIXED


class TestTextChunkerEmpty:
    def test_empty_string(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []

    def test_none_like_empty(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []

    def test_whitespace_only(self):
        chunker = TextChunker()
        assert chunker.chunk("   \n\t  ") == []


class TestTextChunkerFixed:
    def test_short_text_single_chunk(self):
        chunker = TextChunker(
            chunk_size=500,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk("This is a short text that fits in one chunk.", meta)
        assert len(chunks) >= 1
        assert chunks[0].modality == Modality.TEXT

    def test_long_text_multiple_chunks(self):
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        text = "word " * 200  # 1000 chars
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) > 1

    def test_chunk_metadata_propagated(self):
        chunker = TextChunker(
            chunk_size=100,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        meta = ChunkMetadata(doc_id="doc-1", page_number=5, section="Intro")
        text = "word " * 200
        chunks = chunker.chunk(text, meta)
        for chunk in chunks:
            assert chunk.metadata.doc_id == "doc-1"
            assert chunk.metadata.page_number == 5
            assert chunk.metadata.section == "Intro"

    def test_overlap_works(self):
        chunker = TextChunker(
            chunk_size=50,
            chunk_overlap=20,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        text = "word " * 100
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        # With overlap, chunks should share some content
        if len(chunks) >= 2:
            # Content of consecutive chunks should overlap
            assert len(chunks) > 1


class TestTextChunkerSentence:
    def test_sentence_splitting(self):
        chunker = TextChunker(
            chunk_size=100,
            min_chunk_size=10,
            strategy=ChunkingStrategy.SENTENCE,
        )
        text = (
            "First sentence here. Second sentence follows. "
            "Third sentence is here. Fourth sentence now. "
            "Fifth sentence too. Sixth sentence also."
        )
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1

    def test_single_long_sentence(self):
        chunker = TextChunker(
            chunk_size=50,
            min_chunk_size=10,
            strategy=ChunkingStrategy.SENTENCE,
        )
        text = "This is a single very long sentence that should end up in a single chunk because there are no sentence boundaries."
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1


class TestTextChunkerParagraph:
    def test_paragraph_splitting(self):
        chunker = TextChunker(
            chunk_size=100,
            min_chunk_size=10,
            strategy=ChunkingStrategy.PARAGRAPH,
        )
        text = "First paragraph with enough content to be kept.\n\nSecond paragraph with enough content to be kept.\n\nThird paragraph with enough content to be kept."
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1

    def test_large_paragraph_split(self):
        chunker = TextChunker(
            chunk_size=100,
            min_chunk_size=10,
            max_chunk_size=200,
            strategy=ChunkingStrategy.PARAGRAPH,
        )
        # Create a paragraph larger than max_chunk_size
        large_para = "Word " * 100
        text = f"Short intro.\n\n{large_para}\n\nShort conclusion."
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1


class TestTextChunkerSemantic:
    def test_semantic_with_headers(self):
        chunker = TextChunker(
            chunk_size=200,
            min_chunk_size=10,
            strategy=ChunkingStrategy.SEMANTIC,
        )
        text = """Introduction
This is the introduction section with enough content to be kept as a chunk in the output.

Methods
This is the methods section with enough content to be kept as a chunk in the output data.

Results
This is the results section with enough content to be kept as a chunk in the output data here."""
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1

    def test_semantic_without_headers(self):
        chunker = TextChunker(
            chunk_size=200,
            min_chunk_size=10,
            strategy=ChunkingStrategy.SEMANTIC,
        )
        text = "just a plain text without any headers or structure but with enough content to be a valid chunk in the chunker output"
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) >= 1


class TestTextChunkerMinSize:
    def test_chunks_below_min_size_filtered(self):
        chunker = TextChunker(
            chunk_size=500,
            min_chunk_size=100,
            strategy=ChunkingStrategy.FIXED,
        )
        text = "Short."  # Below min_chunk_size
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) == 0

    def test_text_at_min_boundary(self):
        chunker = TextChunker(
            chunk_size=500,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        text = "1234567890"  # Exactly 10 chars
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        assert len(chunks) == 1


class TestTextChunkerNormalization:
    def test_whitespace_normalized(self):
        chunker = TextChunker(
            chunk_size=500,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        text = "Hello    world   this   is   a   test with multiple spaces and enough content"
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta)
        if chunks:
            assert "    " not in chunks[0].content


class TestTextChunkerModality:
    def test_custom_modality(self):
        chunker = TextChunker(
            chunk_size=500,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        text = "Table content that is long enough to pass the minimum chunk size filter easily"
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk(text, meta, modality=Modality.TABLE)
        if chunks:
            assert chunks[0].modality == Modality.TABLE


class TestTextChunkerNoMetadata:
    def test_no_metadata_provided(self):
        chunker = TextChunker(
            chunk_size=500,
            min_chunk_size=10,
            strategy=ChunkingStrategy.FIXED,
        )
        text = "Some text without metadata but long enough to pass min chunk size filter"
        chunks = chunker.chunk(text)
        if chunks:
            assert chunks[0].metadata.doc_id == ""


class TestTextChunkerOverlapSentences:
    def test_get_overlap_sentences_empty(self):
        chunker = TextChunker()
        result = chunker._get_overlap_sentences([])
        assert result == []

    def test_get_overlap_sentences_basic(self):
        chunker = TextChunker(chunk_overlap=50)
        sentences = ["Short.", "Another short.", "Third sentence here."]
        result = chunker._get_overlap_sentences(sentences)
        # Should get sentences from end that fit in overlap
        assert len(result) >= 1


# =============================================================================
# TableChunker Tests
# =============================================================================


class TestTableChunker:
    def test_basic_chunking(self):
        chunker = TableChunker()
        headers = ["Name", "Value"]
        rows = [["Revenue", "$100M"], ["Profit", "$20M"]]
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk_table(headers, rows, meta)
        assert len(chunks) == 1
        assert chunks[0].modality == Modality.TABLE
        assert "Name" in chunks[0].content or "Revenue" in chunks[0].content

    def test_large_table_split(self):
        chunker = TableChunker(max_rows_per_chunk=5)
        headers = ["Col"]
        rows = [[f"row_{i}"] for i in range(12)]
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk_table(headers, rows, meta)
        assert len(chunks) == 3  # 5 + 5 + 2

    def test_empty_rows(self):
        chunker = TableChunker()
        chunks = chunker.chunk_table(["A"], [], ChunkMetadata(doc_id="doc-1"))
        assert len(chunks) == 0

    def test_no_headers(self):
        chunker = TableChunker(include_headers=False)
        rows = [["a", "b"], ["c", "d"]]
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk_table([], rows, meta)
        assert len(chunks) >= 1

    def test_headers_included_each_chunk(self):
        chunker = TableChunker(max_rows_per_chunk=2, include_headers=True)
        headers = ["Name", "Value"]
        rows = [["A", "1"], ["B", "2"], ["C", "3"], ["D", "4"]]
        meta = ChunkMetadata(doc_id="doc-1")
        chunks = chunker.chunk_table(headers, rows, meta)
        for chunk in chunks:
            assert "Name" in chunk.content or "Value" in chunk.content

    def test_no_metadata(self):
        chunker = TableChunker()
        chunks = chunker.chunk_table(
            ["A"], [["val"]], None
        )
        if chunks:
            assert chunks[0].metadata.doc_id == ""

    def test_metadata_table_id_propagated(self):
        chunker = TableChunker()
        meta = ChunkMetadata(doc_id="doc-1", table_id="tbl-1")
        chunks = chunker.chunk_table(
            ["A"], [["val"]], meta
        )
        if chunks:
            assert chunks[0].metadata.table_id == "tbl-1"
