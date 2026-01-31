"""
Data management routes for document and entity operations.

Provides endpoints for:
- Deleting documents and their associated data
- Updating entity information
- Exporting data
- Data cleanup and maintenance
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from src.api.auth import APIKey, verify_api_key
from src.api.dependencies import get_neo4j, get_qdrant
from src.kg.neo4j_client import Neo4jClient
from src.vector.qdrant_client import QdrantClient
from src.core.logging import get_logger

logger = get_logger(__name__)


router = APIRouter(prefix="/data", tags=["Data Management"])


class DeleteDocumentRequest(BaseModel):
    """Request to delete a document and all associated data."""

    doc_id: str
    delete_chunks: bool = True
    delete_entities: bool = False  # Keep entities unless explicitly requested
    delete_vectors: bool = True


class DeleteDocumentResponse(BaseModel):
    """Response after deleting a document."""

    doc_id: str
    deleted_chunks: int
    deleted_entities: int
    deleted_vectors: int
    status: str


class ExportDataRequest(BaseModel):
    """Request to export data."""

    export_type: str  # "entities", "relations", "documents", "all"
    format: str = "json"  # "json", "csv"
    doc_ids: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None


class ExportDataResponse(BaseModel):
    """Response with exported data."""

    export_type: str
    format: str
    data: dict
    record_count: int


class UpdateEntityRequest(BaseModel):
    """Request to update entity attributes."""

    entity_id: str
    attributes: dict
    merge: bool = True  # If True, merge with existing attributes; if False, replace


class UpdateEntityResponse(BaseModel):
    """Response after updating an entity."""

    entity_id: str
    updated_attributes: dict
    status: str


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)

async def delete_document(
    doc_id: str,
    delete_chunks: bool = Query(default=True),
    delete_entities: bool = Query(default=False),
    delete_vectors: bool = Query(default=True),
    neo4j: Neo4jClient = Depends(get_neo4j),
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key: APIKey = Depends(verify_api_key),
) -> DeleteDocumentResponse:
    """
    Delete a document and its associated data.

    Args:
        doc_id: Document ID to delete
        delete_chunks: Whether to delete text chunks (default: True)
        delete_entities: Whether to delete extracted entities (default: False)
        delete_vectors: Whether to delete vector embeddings (default: True)

    Returns:
        Summary of deletion operation

    Requires authentication.
    """
    deleted_chunks = 0
    deleted_entities = 0
    deleted_vectors = 0

    try:
        # Delete from vector database
        if delete_vectors:
            deleted_vectors = await qdrant.delete_by_doc_id(doc_id)

        # Delete chunks and optionally entities from graph
        if delete_chunks or delete_entities:
            deleted_chunks, deleted_entities = await neo4j.delete_document(
                doc_id=doc_id,
                delete_chunks=delete_chunks,
                delete_entities=delete_entities,
            )

        return DeleteDocumentResponse(
            doc_id=doc_id,
            deleted_chunks=deleted_chunks,
            deleted_entities=deleted_entities,
            deleted_vectors=deleted_vectors,
            status="completed",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}",
        )


@router.post("/export", response_model=ExportDataResponse)

async def export_data(
    request: ExportDataRequest,
    neo4j: Neo4jClient = Depends(get_neo4j),
    api_key: APIKey = Depends(verify_api_key),
) -> ExportDataResponse:
    """
    Export data in the specified format.

    Supports exporting entities, relations, documents, or all data.
    Requires authentication.
    """
    try:
        data = {}
        record_count = 0

        if request.export_type in ["entities", "all"]:
            entities = await neo4j.export_entities(
                entity_types=request.entity_types,
            )
            data["entities"] = entities
            record_count += len(entities)

        if request.export_type in ["relations", "all"]:
            relations = await neo4j.export_relations()
            data["relations"] = relations
            record_count += len(relations)

        if request.export_type in ["documents", "all"]:
            documents = await neo4j.export_documents(
                doc_ids=request.doc_ids,
            )
            data["documents"] = documents
            record_count += len(documents)

        return ExportDataResponse(
            export_type=request.export_type,
            format=request.format,
            data=data,
            record_count=record_count,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export data: {str(e)}",
        )


@router.put("/entities/{entity_id}", response_model=UpdateEntityResponse)

async def update_entity(
    entity_id: str,
    attributes: dict,
    merge: bool = Query(default=True),
    neo4j: Neo4jClient = Depends(get_neo4j),
    api_key: APIKey = Depends(verify_api_key),
) -> UpdateEntityResponse:
    """
    Update entity attributes.

    Args:
        entity_id: Entity ID to update
        attributes: New attributes to set
        merge: If True, merge with existing; if False, replace

    Requires authentication.
    """
    try:
        # Check if entity exists
        entity = await neo4j.get_entity(entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")

        # Update attributes
        updated_attrs = await neo4j.update_entity_attributes(
            entity_id=entity_id,
            attributes=attributes,
            merge=merge,
        )

        return UpdateEntityResponse(
            entity_id=entity_id,
            updated_attributes=updated_attrs,
            status="completed",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update entity: {str(e)}",
        )


@router.delete("/cleanup")

async def cleanup_orphaned_data(
    neo4j: Neo4jClient = Depends(get_neo4j),
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key: APIKey = Depends(verify_api_key),
) -> dict:
    """
    Clean up orphaned data (entities without documents, vectors without chunks, etc.).

    Performs maintenance operations to keep the database clean.
    Requires authentication.
    """
    try:
        # Clean up orphaned entities (entities with no relations)
        orphaned_entities = await neo4j.delete_orphaned_entities()

        # Clean up orphaned chunks (chunks without documents)
        orphaned_chunks = await neo4j.delete_orphaned_chunks()

        return {
            "status": "completed",
            "deleted_orphaned_entities": orphaned_entities,
            "deleted_orphaned_chunks": orphaned_chunks,
            "message": "Cleanup completed successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup data: {str(e)}",
        )


@router.get("/statistics")

async def get_data_statistics(
    neo4j: Neo4jClient = Depends(get_neo4j),
    qdrant: QdrantClient = Depends(get_qdrant),
    api_key: APIKey = Depends(verify_api_key),
) -> dict:
    """
    Get detailed data statistics for monitoring and management.

    Returns counts of documents, entities, relations, vectors, etc.
    Requires authentication.
    """
    try:
        graph_stats = await neo4j.get_detailed_stats()
        vector_stats = await qdrant.get_detailed_stats()

        return {
            "graph": graph_stats,
            "vector": vector_stats,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}",
        )
