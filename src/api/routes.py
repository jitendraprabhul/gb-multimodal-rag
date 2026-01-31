"""
FastAPI routes for GraphRAG API.

Endpoints:
- /ask - Question answering
- /entity - Entity operations
- /graph - Graph operations
- /ingest - Document ingestion
- /health - Health check
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse

from config.settings import Settings
from src import __version__
from src.api.auth import APIKey, verify_api_key, verify_api_key_optional
from src.api.dependencies import (
    get_container,
    get_etl,
    get_graph_builder,
    get_neo4j,
    get_qdrant,
    get_reasoner,
    get_retriever,
    get_settings_dep,
    ServiceContainer,
)
from src.api.schemas import (
    AskRequest,
    AskResponse,
    BatchIngestRequest,
    BatchIngestResponse,
    EntityContextResponse,
    EntityRequest,
    EntityResponse,
    ExplainRequest,
    ExplainResponse,
    GraphPathResponse,
    GraphSubgraphRequest,
    GraphSubgraphResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PathFindRequest,
    PathFindResponse,
    SourceSnippetResponse,
    SystemStatsResponse,
)
from src.core.types import EntityType, Modality
from src.etl.pipeline import ETLPipeline
from src.kg.graph_builder import GraphBuilder
from src.kg.neo4j_client import Neo4jClient
from src.llm.reasoning import GraphAwareReasoner
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vector.qdrant_client import QdrantClient


router = APIRouter()


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    container: ServiceContainer = Depends(get_container),
    api_key: APIKey | None = Depends(verify_api_key_optional),
) -> HealthResponse:
    """
    Check system health and service status.

    Returns status of all connected services.
    Public endpoint (no authentication required).
    """
    services = {
        "ollama": False,
        "qdrant": False,
        "neo4j": False,
        "embeddings": False,
    }

    try:
        # Check Ollama
        if container.ollama and container.ollama._initialized:
            services["ollama"] = True
    except Exception:
        pass

    try:
        # Check Qdrant
        if container.qdrant and container.qdrant._initialized:
            services["qdrant"] = True
    except Exception:
        pass

    try:
        # Check Neo4j
        if container.neo4j and container.neo4j._initialized:
            services["neo4j"] = True
    except Exception:
        pass

    try:
        # Check embeddings
        if container.embeddings and container.embeddings._initialized:
            services["embeddings"] = True
    except Exception:
        pass

    all_healthy = all(services.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        services=services,
        timestamp=datetime.utcnow(),
    )


# =============================================================================
# Question Answering
# =============================================================================


@router.post("/ask", response_model=AskResponse, tags=["QA"])
async def ask_question(
    request: AskRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    reasoner: GraphAwareReasoner = Depends(get_reasoner),
    api_key: APIKey = Depends(verify_api_key),
) -> AskResponse:
    """
    Answer a question using hybrid GraphRAG retrieval.

    Pipeline:
    1. Extract entities from query
    2. Retrieve relevant chunks (vector + graph)
    3. Generate answer with LLM
    4. Return answer with sources and graph paths

    Requires authentication via X-API-Key header.
    """
    try:
        # Retrieve context
        retrieval_result = await retriever.retrieve(
            query=request.question,
            filter_doc_ids=request.filter_doc_ids,
        )

        # Generate answer with reasoning
        query_result = await reasoner.reason(
            query=request.question,
            retrieval_result=retrieval_result,
        )

        # Build response
        sources = []
        if request.include_sources:
            sources = [
                SourceSnippetResponse(
                    chunk_id=s.chunk_id,
                    content=s.content,
                    modality=s.modality.value,
                    relevance_score=s.relevance_score,
                    page_number=s.page_number,
                    section=s.section,
                    doc_id=s.doc_id,
                )
                for s in query_result.sources[:request.top_k]
            ]

        graph_paths = []
        if request.include_graph_paths:
            graph_paths = [
                GraphPathResponse(
                    path_text=p.path_text,
                    nodes=p.nodes,
                    edges=p.edges,
                    relevance_score=p.relevance_score,
                )
                for p in query_result.graph_paths
            ]

        return AskResponse(
            answer=query_result.answer,
            confidence=query_result.confidence,
            sources=sources,
            graph_paths=graph_paths,
            reasoning=query_result.reasoning_chain,
            latency_ms=query_result.latency_ms,
            metadata=query_result.metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Entity Operations
# =============================================================================


@router.get("/entity/search", response_model=list[EntityResponse], tags=["Entities"])
async def search_entities(
    query: str = Query(..., min_length=1, description="Search query"),
    entity_types: list[str] | None = Query(
        default=None, description="Filter by entity types"
    ),
    limit: int = Query(default=10, ge=1, le=100),
    neo4j: Neo4jClient = Depends(get_neo4j),
    api_key: APIKey = Depends(verify_api_key),
) -> list[EntityResponse]:
    """Search for entities by name. Requires authentication."""
    try:
        # Convert string types to EntityType enum
        type_filter = None
        if entity_types:
            type_filter = [EntityType(t) for t in entity_types]

        entities = await neo4j.search_entities(
            query=query,
            entity_types=type_filter,
            limit=limit,
        )

        return [
            EntityResponse(
                id=e.id,
                name=e.name,
                normalized_name=e.normalized_name,
                entity_type=e.entity_type.value,
                confidence=e.confidence,
                attributes=e.attributes,
            )
            for e in entities
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{entity_id}", response_model=EntityResponse, tags=["Entities"])
async def get_entity(
    entity_id: str,
    neo4j: Neo4jClient = Depends(get_neo4j),
    api_key: APIKey = Depends(verify_api_key),
) -> EntityResponse:
    """Get entity details by ID. Requires authentication."""
    try:
        entity = await neo4j.get_entity(entity_id)

        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        return EntityResponse(
            id=entity.id,
            name=entity.name,
            normalized_name=entity.normalized_name,
            entity_type=entity.entity_type.value,
            confidence=entity.confidence,
            attributes=entity.attributes,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/entity/{entity_id}/context",
    response_model=EntityContextResponse,
    tags=["Entities"],
)
async def get_entity_context(
    entity_id: str,
    retriever: HybridRetriever = Depends(get_retriever),
    api_key: APIKey = Depends(verify_api_key),
) -> EntityContextResponse:
    """Get full context for an entity including neighbors and related chunks. Requires authentication."""
    try:
        context = await retriever.get_entity_context(entity_id)

        if not context.get("entity"):
            raise HTTPException(status_code=404, detail="Entity not found")

        entity_data = context["entity"]
        return EntityContextResponse(
            entity=EntityResponse(
                id=entity_data["id"],
                name=entity_data["name"],
                normalized_name=entity_data["normalized_name"],
                entity_type=entity_data["entity_type"],
                confidence=entity_data.get("confidence", 1.0),
                attributes=entity_data.get("attributes", {}),
            ),
            neighbors=context.get("neighbors", []),
            relations=context.get("relations", []),
            chunks=context.get("chunks", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Graph Operations
# =============================================================================


@router.post("/graph/subgraph", response_model=GraphSubgraphResponse, tags=["Graph"])
async def get_subgraph(
    request: GraphSubgraphRequest,
    neo4j: Neo4jClient = Depends(get_neo4j),
    api_key: APIKey = Depends(verify_api_key),
) -> GraphSubgraphResponse:
    """Get subgraph around specified entities. Requires authentication."""
    try:
        nodes, edges = await neo4j.get_subgraph(
            entity_ids=request.entity_ids,
            hops=request.hops,
            max_nodes=request.max_nodes,
        )

        return GraphSubgraphResponse(
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/paths", response_model=PathFindResponse, tags=["Graph"])
async def find_paths(
    request: PathFindRequest,
    neo4j: Neo4jClient = Depends(get_neo4j),
    api_key: APIKey = Depends(verify_api_key),
) -> PathFindResponse:
    """Find paths between two entities. Requires authentication."""
    try:
        paths = await neo4j.find_paths(
            source_id=request.source_entity_id,
            target_id=request.target_entity_id,
            max_hops=request.max_hops,
            limit=request.limit,
        )

        return PathFindResponse(
            paths=[
                GraphPathResponse(
                    path_text=p.path_text,
                    nodes=p.nodes,
                    edges=p.edges,
                    relevance_score=p.relevance_score,
                )
                for p in paths
            ],
            total_paths=len(paths),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Ingestion Operations
# =============================================================================


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    request: IngestRequest,
    etl: ETLPipeline = Depends(get_etl),
    graph_builder: GraphBuilder = Depends(get_graph_builder),
    container: ServiceContainer = Depends(get_container),
    api_key: APIKey = Depends(verify_api_key),
) -> IngestResponse:
    """
    Ingest a document into the system.

    Processes the document, extracts entities, builds graph,
    and creates vector embeddings.

    Requires authentication via X-API-Key header.
    """
    start_time = time.time()

    try:
        file_path = Path(request.file_path)

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Process document
        document = await etl.process_file(file_path, doc_id=request.doc_id)

        entities_count = 0
        relations_count = 0

        # Extract entities and build graph
        if request.extract_entities:
            entities, relations = await graph_builder.process_document(document)
            entities_count = len(entities)
            relations_count = len(relations)

        # Generate embeddings and store in vector DB
        if request.generate_embeddings:
            # Add embeddings to chunks
            document.chunks = await container.embeddings.embed_chunks(document.chunks)

            # Store in Qdrant
            await container.qdrant.upsert_chunks(document.chunks)

        processing_time_ms = (time.time() - start_time) * 1000

        return IngestResponse(
            doc_id=document.id,
            filename=document.filename,
            chunks=len(document.chunks),
            entities=entities_count,
            relations=relations_count,
            processing_time_ms=processing_time_ms,
            status="completed",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/upload", response_model=list[IngestResponse], tags=["Ingestion"])
async def upload_and_ingest(
    files: list[UploadFile] = File(..., description="Files to upload and ingest"),
    extract_entities: bool = Form(True),
    generate_embeddings: bool = Form(True),
    etl: ETLPipeline = Depends(get_etl),
    graph_builder: GraphBuilder = Depends(get_graph_builder),
    container: ServiceContainer = Depends(get_container),
    api_key: APIKey = Depends(verify_api_key),
) -> list[IngestResponse]:
    """
    Upload one or more files and ingest them into the system.

    Accepts multipart/form-data with multiple files. Each file is saved
    to a temporary upload directory, processed, and the results returned.

    Requires authentication via X-API-Key header.
    """
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    results: list[IngestResponse] = []

    for upload_file in files:
        start_time = time.time()
        file_path = upload_dir / upload_file.filename
        try:
            # Save uploaded file to disk
            async with aiofiles.open(file_path, "wb") as f:
                content = await upload_file.read()
                await f.write(content)

            # Process document
            document = await etl.process_file(file_path)

            entities_count = 0
            relations_count = 0

            if extract_entities:
                entities, relations = await graph_builder.process_document(document)
                entities_count = len(entities)
                relations_count = len(relations)

            if generate_embeddings:
                document.chunks = await container.embeddings.embed_chunks(
                    document.chunks
                )
                await container.qdrant.upsert_chunks(document.chunks)

            processing_time_ms = (time.time() - start_time) * 1000

            results.append(
                IngestResponse(
                    doc_id=document.id,
                    filename=document.filename,
                    chunks=len(document.chunks),
                    entities=entities_count,
                    relations=relations_count,
                    processing_time_ms=processing_time_ms,
                    status="completed",
                )
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            results.append(
                IngestResponse(
                    doc_id="",
                    filename=upload_file.filename or "unknown",
                    chunks=0,
                    entities=0,
                    relations=0,
                    processing_time_ms=processing_time_ms,
                    status=f"failed: {str(e)}",
                )
            )

    return results


@router.post("/ingest/batch", response_model=BatchIngestResponse, tags=["Ingestion"])
async def batch_ingest(
    request: BatchIngestRequest,
    etl: ETLPipeline = Depends(get_etl),
    graph_builder: GraphBuilder = Depends(get_graph_builder),
    container: ServiceContainer = Depends(get_container),
    api_key: APIKey = Depends(verify_api_key),
) -> BatchIngestResponse:
    """Batch ingest documents from a directory. Requires authentication."""
    start_time = time.time()

    try:
        directory = Path(request.directory_path)

        if not directory.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found")

        # Process all documents
        documents = await etl.process_directory(
            directory,
            recursive=request.recursive,
            file_patterns=request.file_patterns,
        )

        results = []
        successful = 0
        failed = 0

        for document in documents:
            try:
                # Extract entities and build graph
                entities, relations = await graph_builder.process_document(document)

                # Generate embeddings
                document.chunks = await container.embeddings.embed_chunks(document.chunks)
                await container.qdrant.upsert_chunks(document.chunks)

                results.append(
                    IngestResponse(
                        doc_id=document.id,
                        filename=document.filename,
                        chunks=len(document.chunks),
                        entities=len(entities),
                        relations=len(relations),
                        processing_time_ms=0,
                        status="completed",
                    )
                )
                successful += 1

            except Exception as e:
                results.append(
                    IngestResponse(
                        doc_id=document.id if document else "unknown",
                        filename=document.filename if document else "unknown",
                        chunks=0,
                        entities=0,
                        relations=0,
                        processing_time_ms=0,
                        status=f"failed: {str(e)}",
                    )
                )
                failed += 1

        total_time_ms = (time.time() - start_time) * 1000

        return BatchIngestResponse(
            total_files=len(documents),
            successful=successful,
            failed=failed,
            results=results,
            total_time_ms=total_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Statistics
# =============================================================================


@router.get("/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_stats(
    neo4j: Neo4jClient = Depends(get_neo4j),
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key: APIKey = Depends(verify_api_key),
) -> SystemStatsResponse:
    """Get system statistics. Requires authentication."""
    try:
        graph_stats = await neo4j.get_stats()
        vector_stats = await qdrant.get_stats()

        return SystemStatsResponse(
            graph=graph_stats,
            vector=vector_stats,
            documents=graph_stats.get("documents", 0),
            chunks=graph_stats.get("chunks", 0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Explain
# =============================================================================


@router.post("/explain", response_model=ExplainResponse, tags=["QA"])
async def explain_answer(
    request: ExplainRequest,
    container: ServiceContainer = Depends(get_container),
    api_key: APIKey = Depends(verify_api_key),
) -> ExplainResponse:
    """Get detailed explanation for an answer. Requires authentication."""
    try:
        # Use LLM to generate explanation
        prompt = f"""Given this question and answer, provide a detailed explanation of how the answer was derived.

Question: {request.question}

Answer: {request.answer}

Provide:
1. Step-by-step reasoning
2. Key evidence points
3. Any assumptions made
4. Confidence factors

Explanation:"""

        explanation = await container.ollama.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3,
        )

        return ExplainResponse(
            explanation=explanation,
            evidence=[],
            confidence_factors={
                "source_coverage": 0.8,
                "entity_support": 0.7,
                "graph_connectivity": 0.6,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
