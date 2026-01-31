"""
Text chunking with various strategies.

Supports:
- Fixed-size chunking with overlap
- Semantic chunking (sentence-based)
- Document structure-aware chunking
- Table-aware chunking
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from src.core.logging import LoggerMixin
from src.core.types import Chunk, ChunkMetadata, Modality


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class BaseChunker(ABC, LoggerMixin):
    """Abstract base class for chunkers."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: ChunkMetadata,
    ) -> list[Chunk]:
        """Split text into chunks."""
        pass


class TextChunker(LoggerMixin):
    """
    Main text chunker with configurable strategies.

    Implements multiple chunking strategies and handles
    overlap, metadata propagation, and edge cases.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        **config: Any,
    ) -> None:
        """
        Initialize text chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            strategy: Chunking strategy to use
            **config: Additional configuration
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.strategy = strategy

    def chunk(
        self,
        text: str,
        metadata: ChunkMetadata | None = None,
        modality: Modality = Modality.TEXT,
    ) -> list[Chunk]:
        """
        Split text into chunks using configured strategy.

        Args:
            text: Text to chunk
            metadata: Base metadata for chunks
            modality: Content modality

        Returns:
            List of chunks
        """
        if not text or not text.strip():
            return []

        # Normalize text
        text = self._normalize_text(text)

        # Select strategy
        if self.strategy == ChunkingStrategy.FIXED:
            raw_chunks = self._chunk_fixed(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            raw_chunks = self._chunk_by_sentence(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            raw_chunks = self._chunk_by_paragraph(text)
        else:
            raw_chunks = self._chunk_semantic(text)

        # Create Chunk objects
        chunks = []
        char_offset = 0

        for content in raw_chunks:
            if len(content.strip()) < self.min_chunk_size:
                continue

            # Create metadata for this chunk
            chunk_metadata = ChunkMetadata(
                doc_id=metadata.doc_id if metadata else "",
                page_number=metadata.page_number if metadata else None,
                section=metadata.section if metadata else None,
                source_file=metadata.source_file if metadata else None,
                start_char=char_offset,
                end_char=char_offset + len(content),
            )

            chunk = Chunk(
                content=content.strip(),
                modality=modality,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

            # Update offset (accounting for removed chars in normalization)
            char_offset += len(content)

        return chunks

    def _normalize_text(self, text: str) -> str:
        """Normalize text for chunking."""
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)

        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _chunk_fixed(self, text: str) -> list[str]:
        """Fixed-size chunking with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Don't split in the middle of a word
            if end < len(text):
                # Find the last space before the end
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end]
            chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _chunk_by_sentence(self, text: str) -> list[str]:
        """Sentence-based chunking."""
        # Split by sentence boundaries
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If adding this sentence exceeds max, save current chunk
            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_len

        # Add remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> list[str]:
        """Paragraph-based chunking."""
        # Split by paragraph breaks
        paragraphs = re.split(r"\n\n+", text)

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # If single paragraph exceeds max, split it
            if para_len > self.max_chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sub_chunks = self._chunk_by_sentence(para)
                chunks.extend(sub_chunks)
                continue

            # If adding this paragraph exceeds max, save current chunk
            if current_size + para_len > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_len

        # Add remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _chunk_semantic(self, text: str) -> list[str]:
        """
        Semantic chunking based on topic coherence.

        Uses paragraph structure and heading detection.
        """
        # Detect section headers
        header_pattern = r"^(?:#{1,6}\s+)?(?:[A-Z][^.!?]*(?:\n|$))"

        sections = []
        current_section = []
        current_header = None

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if line looks like a header
            if re.match(header_pattern, line) and len(line) < 100:
                if current_section:
                    sections.append((current_header, "\n".join(current_section)))
                current_header = line
                current_section = []
            else:
                current_section.append(line)

        if current_section:
            sections.append((current_header, "\n".join(current_section)))

        # Now chunk each section
        chunks = []
        for header, content in sections:
            if header:
                content = f"{header}\n\n{content}"

            if len(content) > self.chunk_size:
                # Split large sections by paragraph
                sub_chunks = self._chunk_by_paragraph(content)
                chunks.extend(sub_chunks)
            else:
                chunks.append(content)

        return chunks

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        """Get sentences for overlap from the end of chunk."""
        if not sentences:
            return []

        overlap_size = 0
        overlap_sentences = []

        for sentence in reversed(sentences):
            if overlap_size + len(sentence) > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_size += len(sentence)

        return overlap_sentences


class TableChunker(LoggerMixin):
    """
    Chunker for table content.

    Converts tables to text representations that can be embedded.
    """

    def __init__(
        self,
        max_rows_per_chunk: int = 20,
        include_headers: bool = True,
        **config: Any,
    ) -> None:
        """
        Initialize table chunker.

        Args:
            max_rows_per_chunk: Maximum rows per chunk
            include_headers: Include headers in each chunk
            **config: Additional configuration
        """
        self.max_rows_per_chunk = max_rows_per_chunk
        self.include_headers = include_headers

    def chunk_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        metadata: ChunkMetadata | None = None,
    ) -> list[Chunk]:
        """
        Split a table into chunks.

        Args:
            headers: Column headers
            rows: Table rows
            metadata: Base metadata

        Returns:
            List of chunks
        """
        chunks = []

        for i in range(0, len(rows), self.max_rows_per_chunk):
            chunk_rows = rows[i : i + self.max_rows_per_chunk]

            # Create text representation
            lines = []

            if self.include_headers and headers:
                lines.append(" | ".join(headers))
                lines.append("-" * 40)

            for row in chunk_rows:
                if headers:
                    # Key: value format
                    row_text = ", ".join(
                        f"{h}: {v}" for h, v in zip(headers, row) if v.strip()
                    )
                else:
                    row_text = " | ".join(row)

                if row_text:
                    lines.append(row_text)

            content = "\n".join(lines)

            if content.strip():
                chunk_metadata = ChunkMetadata(
                    doc_id=metadata.doc_id if metadata else "",
                    table_id=metadata.table_id if metadata else None,
                    source_file=metadata.source_file if metadata else None,
                )

                chunk = Chunk(
                    content=content,
                    modality=Modality.TABLE,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)

        return chunks
