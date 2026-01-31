"""
Ingestion service for document processing and indexing.

Orchestrates the complete ingestion pipeline:
1. Document processing (ETL)
2. Entity extraction and graph building
3. Embedding generation
4. Vector indexing
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Callable

from src.core.logging import LoggerMixin, log_operation
from src.core.types import Document, Entity, Relation
from src.etl.pipeline import ETLPipeline
from src.kg.graph_builder import GraphBuilder
from src.vector.embeddings import EmbeddingService
from src.vector.qdrant_client import QdrantClient


class IngestionResult:
    """Result of document ingestion."""

    def __init__(
        self,
        doc_id: str,
        filename: str,
        chunks: int,
        entities: int,
        relations: int,
        processing_time_ms: float,
        status: str = "completed",
        error: str | None = None,
    ) -> None:
        self.doc_id = doc_id
        self.filename = filename
        self.chunks = chunks
        self.entities = entities
        self.relations = relations
        self.processing_time_ms = processing_time_ms
        self.status = status
        self.error = error


class IngestionService(LoggerMixin):
    """
    Service for orchestrating document ingestion.

    Provides a high-level interface for processing documents
    through the complete ETL -> Graph -> Vector pipeline.
    """

    def __init__(
        self,
        etl_pipeline: ETLPipeline,
        graph_builder: GraphBuilder,
        embedding_service: EmbeddingService,
        qdrant_client: QdrantClient,
        max_concurrent: int = 4,
        **config: Any,
    ) -> None:
        """
        Initialize ingestion service.

        Args:
            etl_pipeline: ETL pipeline for document processing
            graph_builder: Graph builder for entity/relation extraction
            embedding_service: Embedding service
            qdrant_client: Vector database client
            max_concurrent: Max concurrent ingestions
            **config: Additional configuration
        """
        self.etl_pipeline = etl_pipeline
        self.graph_builder = graph_builder
        self.embedding_service = embedding_service
        self.qdrant_client = qdrant_client
        self.max_concurrent = max_concurrent

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._progress_callbacks: list[Callable[[str, float], None]] = []

    def add_progress_callback(
        self,
        callback: Callable[[str, float], None],
    ) -> None:
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)

    def _notify_progress(self, doc_id: str, progress: float) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(doc_id, progress)
            except Exception:
                pass

    async def ingest_file(
        self,
        file_path: Path | str,
        doc_id: str | None = None,
        extract_entities: bool = True,
        generate_embeddings: bool = True,
    ) -> IngestionResult:
        """
        Ingest a single file.

        Args:
            file_path: Path to file
            doc_id: Optional custom document ID
            extract_entities: Whether to extract entities and build graph
            generate_embeddings: Whether to generate embeddings

        Returns:
            IngestionResult with processing details
        """
        async with self._semaphore:
            return await self._ingest_file_impl(
                file_path=Path(file_path),
                doc_id=doc_id,
                extract_entities=extract_entities,
                generate_embeddings=generate_embeddings,
            )

    async def _ingest_file_impl(
        self,
        file_path: Path,
        doc_id: str | None,
        extract_entities: bool,
        generate_embeddings: bool,
    ) -> IngestionResult:
        """Implementation of file ingestion."""
        start_time = time.time()

        self.logger.info("Starting ingestion", file=str(file_path))

        try:
            # Step 1: Process document (20%)
            self._notify_progress(str(file_path), 0.1)
            document = await self.etl_pipeline.process_file(file_path, doc_id=doc_id)
            self._notify_progress(document.id, 0.2)

            entities_count = 0
            relations_count = 0

            # Step 2: Extract entities and build graph (50%)
            if extract_entities:
                self._notify_progress(document.id, 0.3)
                entities, relations = await self.graph_builder.process_document(document)
                entities_count = len(entities)
                relations_count = len(relations)
                self._notify_progress(document.id, 0.5)

            # Step 3: Generate embeddings (80%)
            if generate_embeddings:
                self._notify_progress(document.id, 0.6)
                document.chunks = await self.embedding_service.embed_chunks(document.chunks)
                self._notify_progress(document.id, 0.8)

                # Step 4: Store in vector DB (100%)
                await self.qdrant_client.upsert_chunks(document.chunks)
                self._notify_progress(document.id, 1.0)

            processing_time_ms = (time.time() - start_time) * 1000

            log_operation(
                "ingest_file",
                success=True,
                duration_ms=processing_time_ms,
                file=str(file_path),
                chunks=len(document.chunks),
                entities=entities_count,
            )

            return IngestionResult(
                doc_id=document.id,
                filename=document.filename,
                chunks=len(document.chunks),
                entities=entities_count,
                relations=relations_count,
                processing_time_ms=processing_time_ms,
                status="completed",
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            log_operation(
                "ingest_file",
                success=False,
                duration_ms=processing_time_ms,
                file=str(file_path),
                error=str(e),
            )

            return IngestionResult(
                doc_id=doc_id or str(file_path),
                filename=file_path.name,
                chunks=0,
                entities=0,
                relations=0,
                processing_time_ms=processing_time_ms,
                status="failed",
                error=str(e),
            )

    async def ingest_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        file_patterns: list[str] | None = None,
        extract_entities: bool = True,
        generate_embeddings: bool = True,
    ) -> list[IngestionResult]:
        """
        Ingest all files in a directory.

        Args:
            directory: Directory path
            recursive: Process subdirectories
            file_patterns: File patterns to match
            extract_entities: Extract entities
            generate_embeddings: Generate embeddings

        Returns:
            List of ingestion results
        """
        directory = Path(directory)
        patterns = file_patterns or ["*.pdf", "*.txt", "*.csv", "*.xlsx"]

        # Find all files
        files = []
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))

        self.logger.info(
            "Starting batch ingestion",
            directory=str(directory),
            files=len(files),
        )

        # Process files concurrently
        tasks = [
            self.ingest_file(
                file_path=f,
                extract_entities=extract_entities,
                generate_embeddings=generate_embeddings,
            )
            for f in files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    IngestionResult(
                        doc_id=str(files[i]),
                        filename=files[i].name,
                        chunks=0,
                        entities=0,
                        relations=0,
                        processing_time_ms=0,
                        status="failed",
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        # Log summary
        successful = sum(1 for r in final_results if r.status == "completed")
        failed = len(final_results) - successful

        self.logger.info(
            "Batch ingestion completed",
            total=len(final_results),
            successful=successful,
            failed=failed,
        )

        return final_results

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all associated data.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            # Delete from vector DB
            await self.qdrant_client.delete_by_doc_id(doc_id)

            # Note: Deleting from graph is more complex as entities
            # might be shared across documents. For now, we leave
            # graph nodes intact.

            self.logger.info("Document deleted", doc_id=doc_id)
            return True

        except Exception as e:
            self.logger.error("Document deletion failed", doc_id=doc_id, error=str(e))
            return False
