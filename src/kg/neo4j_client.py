"""
Neo4j client for knowledge graph persistence.

Handles:
- Connection management
- Node and edge CRUD operations
- Cypher query execution
- Graph traversal
"""

from typing import Any
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError

from src.core.exceptions import GraphError
from src.core.logging import LoggerMixin
from src.core.types import Entity, EntityType, GraphPath, Relation, RelationType


def _serialize_neo4j_value(val: Any) -> Any:
    """Convert Neo4j-specific types to JSON-serializable types."""
    if hasattr(val, "iso_format"):  # neo4j.time.DateTime / Date / Time
        return val.iso_format()
    if hasattr(val, "isoformat"):  # Python datetime
        return val.isoformat()
    if isinstance(val, dict):
        return {k: _serialize_neo4j_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_neo4j_value(v) for v in val]
    return val


class Neo4jClient(LoggerMixin):
    """
    Async Neo4j client for knowledge graph operations.

    Provides connection pooling, transaction management,
    and high-level graph operations.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "graphrag_password",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        **config: Any,
    ) -> None:
        """
        Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
            database: Database name
            max_connection_pool_size: Connection pool size
            **config: Additional configuration
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.max_connection_pool_size = max_connection_pool_size

        self._driver: AsyncDriver | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection."""
        if self._initialized:
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=self.max_connection_pool_size,
            )

            # Verify connectivity
            await self._driver.verify_connectivity()

            self._initialized = True
            self.logger.info(
                "Neo4j connected",
                uri=self.uri,
                database=self.database,
            )

            # Create indexes
            await self._create_indexes()

        except Exception as e:
            raise GraphError(
                f"Failed to connect to Neo4j: {e}",
                cause=e,
            )

    async def _create_indexes(self) -> None:
        """Create indexes for efficient querying."""
        indexes = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
        ]

        async with self.session() as session:
            for index_query in indexes:
                try:
                    await session.run(index_query)
                except Neo4jError:
                    pass  # Index may already exist

    async def cleanup(self) -> None:
        """Close database connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
        self._initialized = False

    @asynccontextmanager
    async def session(self):
        """Get a database session."""
        if not self._initialized:
            await self.initialize()

        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            await session.close()

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def upsert_entity(self, entity: Entity) -> str:
        """
        Upsert an entity node.

        Uses MERGE to avoid duplicates based on normalized_name and type.

        Args:
            entity: Entity to upsert

        Returns:
            Entity ID
        """
        query = """
        MERGE (e:Entity {
            normalized_name: $normalized_name,
            entity_type: $entity_type
        })
        ON CREATE SET
            e.id = $id,
            e.name = $name,
            e.attributes = $attributes,
            e.confidence = $confidence,
            e.created_at = datetime()
        ON MATCH SET
            e.name = CASE WHEN $confidence > e.confidence THEN $name ELSE e.name END,
            e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END,
            e.updated_at = datetime()
        RETURN e.id as id
        """

        try:
            async with self.session() as session:
                result = await session.run(
                    query,
                    id=entity.id,
                    name=entity.name,
                    normalized_name=entity.normalized_name,
                    entity_type=entity.entity_type.value,
                    attributes=str(entity.attributes),
                    confidence=entity.confidence,
                )
                record = await result.single()
                return record["id"] if record else entity.id

        except Exception as e:
            raise GraphError(
                f"Failed to upsert entity: {e}",
                node_id=entity.id,
                cause=e,
            )

    async def upsert_entities(self, entities: list[Entity]) -> list[str]:
        """Batch upsert multiple entities."""
        ids = []
        for entity in entities:
            entity_id = await self.upsert_entity(entity)
            ids.append(entity_id)
        return ids

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e
        """

        try:
            async with self.session() as session:
                result = await session.run(query, id=entity_id)
                record = await result.single()

                if not record:
                    return None

                node = record["e"]
                return Entity(
                    id=node["id"],
                    name=node["name"],
                    normalized_name=node["normalized_name"],
                    entity_type=EntityType(node["entity_type"]),
                    confidence=node.get("confidence", 1.0),
                )

        except Exception as e:
            raise GraphError(
                f"Failed to get entity: {e}",
                node_id=entity_id,
                cause=e,
            )

    async def search_entities(
        self,
        query: str,
        entity_types: list[EntityType] | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """
        Search entities by name.

        Args:
            query: Search query
            entity_types: Optional type filter
            limit: Maximum results

        Returns:
            List of matching entities
        """
        type_filter = ""
        if entity_types:
            types_str = ", ".join(f'"{t.value}"' for t in entity_types)
            type_filter = f"AND e.entity_type IN [{types_str}]"

        cypher = f"""
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
            OR toLower(e.normalized_name) CONTAINS toLower($query)
            {type_filter}
        RETURN e
        ORDER BY e.confidence DESC
        LIMIT $limit
        """

        try:
            async with self.session() as session:
                result = await session.run(cypher, parameters={"query": query, "limit": limit})
                records = await result.data()

                entities = []
                for record in records:
                    node = record["e"]
                    entity = Entity(
                        id=node["id"],
                        name=node["name"],
                        normalized_name=node["normalized_name"],
                        entity_type=EntityType(node["entity_type"]),
                        confidence=node.get("confidence", 1.0),
                    )
                    entities.append(entity)

                return entities

        except Exception as e:
            raise GraphError(
                f"Entity search failed: {e}",
                query=query,
                cause=e,
            )

    # =========================================================================
    # Relation Operations
    # =========================================================================

    async def upsert_relation(self, relation: Relation) -> str:
        """
        Upsert a relation edge.

        Args:
            relation: Relation to upsert

        Returns:
            Relation ID
        """
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{relation.relation_type.value}]->(target)
        ON CREATE SET
            r.id = $id,
            r.confidence = $confidence,
            r.attributes = $attributes,
            r.created_at = datetime()
        ON MATCH SET
            r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
            r.updated_at = datetime()
        RETURN r.id as id
        """

        try:
            async with self.session() as session:
                result = await session.run(
                    query,
                    source_id=relation.source_entity_id,
                    target_id=relation.target_entity_id,
                    id=relation.id,
                    confidence=relation.confidence,
                    attributes=str(relation.attributes),
                )
                record = await result.single()
                return record["id"] if record else relation.id

        except Exception as e:
            raise GraphError(
                f"Failed to upsert relation: {e}",
                cause=e,
            )

    async def upsert_relations(self, relations: list[Relation]) -> list[str]:
        """Batch upsert multiple relations."""
        ids = []
        for relation in relations:
            rel_id = await self.upsert_relation(relation)
            ids.append(rel_id)
        return ids

    # =========================================================================
    # Document/Chunk Operations
    # =========================================================================

    async def create_document_node(
        self,
        doc_id: str,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a document node."""
        query = """
        MERGE (d:Document {id: $id})
        ON CREATE SET
            d.filename = $filename,
            d.metadata = $metadata,
            d.created_at = datetime()
        """

        async with self.session() as session:
            await session.run(
                query,
                id=doc_id,
                filename=filename,
                metadata=str(metadata or {}),
            )

    async def create_chunk_node(
        self,
        chunk_id: str,
        doc_id: str,
        content_preview: str,
        modality: str,
    ) -> None:
        """Create a chunk node and link to document."""
        query = """
        MATCH (d:Document {id: $doc_id})
        MERGE (c:Chunk {id: $chunk_id})
        ON CREATE SET
            c.content_preview = $content_preview,
            c.modality = $modality,
            c.created_at = datetime()
        MERGE (d)-[:HAS_CHUNK]->(c)
        """

        async with self.session() as session:
            await session.run(
                query,
                chunk_id=chunk_id,
                doc_id=doc_id,
                content_preview=content_preview[:500],
                modality=modality,
            )

    async def link_entity_to_chunk(
        self,
        entity_id: str,
        chunk_id: str,
    ) -> None:
        """Create edge from chunk to entity."""
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (c)-[:HAS_ENTITY]->(e)
        """

        async with self.session() as session:
            await session.run(
                query,
                chunk_id=chunk_id,
                entity_id=entity_id,
            )

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    async def get_subgraph(
        self,
        entity_ids: list[str],
        hops: int = 2,
        max_nodes: int = 100,
        relation_types: list[RelationType] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Get a subgraph around specified entities.

        Args:
            entity_ids: Starting entity IDs
            hops: Number of hops to traverse
            max_nodes: Maximum nodes to return
            relation_types: Optional relation type filter

        Returns:
            Tuple of (nodes, edges)
        """
        # Build relation filter
        rel_filter = ""
        if relation_types:
            rel_names = "|".join(rt.value for rt in relation_types)
            rel_filter = f"[r:{rel_names}]"
        else:
            rel_filter = "[r]"

        query = f"""
        MATCH (start:Entity)
        WHERE start.id IN $entity_ids
        CALL apoc.path.subgraphAll(start, {{
            maxLevel: $hops,
            relationshipFilter: '{rel_filter.replace('[', '').replace(']', '')}',
            limit: $max_nodes
        }})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """

        # Fallback query if APOC not available
        fallback_query = f"""
        MATCH path = (start:Entity)-{rel_filter}*1..{hops}-(connected:Entity)
        WHERE start.id IN $entity_ids
        WITH collect(DISTINCT start) + collect(DISTINCT connected) as nodes,
             collect(DISTINCT relationships(path)) as rels
        UNWIND nodes as n
        WITH collect(DISTINCT n)[0..$max_nodes] as limited_nodes,
             rels
        UNWIND rels as rel_list
        UNWIND rel_list as r
        RETURN limited_nodes as nodes, collect(DISTINCT r) as relationships
        """

        try:
            async with self.session() as session:
                try:
                    result = await session.run(
                        query,
                        entity_ids=entity_ids,
                        hops=hops,
                        max_nodes=max_nodes,
                    )
                except Neo4jError:
                    # APOC not available, use fallback
                    result = await session.run(
                        fallback_query,
                        entity_ids=entity_ids,
                        max_nodes=max_nodes,
                    )

                record = await result.single()

                if not record:
                    return [], []

                nodes = []
                for node in record.get("nodes", []):
                    nodes.append({
                        "id": node.get("id"),
                        "name": node.get("name"),
                        "type": node.get("entity_type"),
                        "labels": list(node.labels) if hasattr(node, "labels") else [],
                    })

                edges = []
                for rel in record.get("relationships", []):
                    edges.append({
                        "id": rel.get("id"),
                        "type": rel.type,
                        "source": rel.start_node.get("id"),
                        "target": rel.end_node.get("id"),
                    })

                return nodes, edges

        except Exception as e:
            self.logger.warning(f"Subgraph query failed: {e}")
            return [], []

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        limit: int = 5,
    ) -> list[GraphPath]:
        """
        Find paths between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_hops: Maximum path length
            limit: Maximum paths to return

        Returns:
            List of graph paths
        """
        query = f"""
        MATCH path = shortestPath(
            (source:Entity {{id: $source_id}})-[*1..{max_hops}]-(target:Entity {{id: $target_id}})
        )
        RETURN path
        LIMIT $limit
        """

        try:
            async with self.session() as session:
                result = await session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    limit=limit,
                )

                paths = []
                async for record in result:
                    path = record["path"]

                    nodes = []
                    for node in path.nodes:
                        nodes.append({
                            "id": node.get("id"),
                            "name": node.get("name"),
                            "type": node.get("entity_type"),
                        })

                    edges = []
                    for rel in path.relationships:
                        edges.append({
                            "type": rel.type,
                            "source": rel.start_node.get("id"),
                            "target": rel.end_node.get("id"),
                        })

                    # Generate path text
                    path_parts = []
                    for i, node in enumerate(nodes):
                        path_parts.append(f"({node['name']})")
                        if i < len(edges):
                            path_parts.append(f"-[{edges[i]['type']}]->")

                    graph_path = GraphPath(
                        nodes=nodes,
                        edges=edges,
                        path_text=" ".join(path_parts),
                        relevance_score=1.0 / (len(edges) + 1),  # Shorter = more relevant
                    )
                    paths.append(graph_path)

                return paths

        except Exception as e:
            self.logger.warning(f"Path finding failed: {e}")
            return []

    async def get_entity_neighborhood(
        self,
        entity_id: str,
        hops: int = 1,
        limit: int = 50,
    ) -> tuple[list[dict], list[dict]]:
        """
        Get the neighborhood of an entity.

        Args:
            entity_id: Entity ID
            hops: Number of hops
            limit: Maximum neighbors

        Returns:
            Tuple of (nodes, edges)
        """
        return await self.get_subgraph([entity_id], hops=hops, max_nodes=limit)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, int]:
        """Get graph statistics."""
        query = """
        MATCH (e:Entity) WITH count(e) as entity_count
        MATCH (d:Document) WITH entity_count, count(d) as doc_count
        MATCH (c:Chunk) WITH entity_count, doc_count, count(c) as chunk_count
        MATCH ()-[r]->() WITH entity_count, doc_count, chunk_count, count(r) as rel_count
        RETURN entity_count, doc_count, chunk_count, rel_count
        """

        try:
            async with self.session() as session:
                result = await session.run(query)
                record = await result.single()

                return {
                    "entities": record["entity_count"],
                    "documents": record["doc_count"],
                    "chunks": record["chunk_count"],
                    "relations": record["rel_count"],
                }

        except Exception as e:
            self.logger.warning(f"Stats query failed: {e}")
            return {}

    async def get_detailed_stats(self) -> dict[str, Any]:
        """Get detailed graph statistics for the data management dashboard."""
        try:
            stats = await self.get_stats()
            # Add entity type breakdown
            type_query = """
            MATCH (e:Entity)
            RETURN e.entity_type AS type, count(e) AS count
            ORDER BY count DESC
            """
            async with self.session() as session:
                result = await session.run(type_query)
                records = await result.data()
                stats["entity_types"] = {r["type"]: r["count"] for r in records}
            return stats
        except Exception as e:
            self.logger.warning(f"Detailed stats query failed: {e}")
            return await self.get_stats()

    async def export_entities(
        self, entity_types: list[str] | None = None
    ) -> list[dict]:
        """Export all entities, optionally filtered by type."""
        type_filter = ""
        params: dict[str, Any] = {}
        if entity_types:
            type_filter = "WHERE e.entity_type IN $types"
            params["types"] = entity_types

        cypher = f"""
        MATCH (e:Entity)
        {type_filter}
        RETURN e
        ORDER BY e.name
        LIMIT 10000
        """
        try:
            async with self.session() as session:
                result = await session.run(cypher, parameters=params)
                records = await result.data()
                return [_serialize_neo4j_value(dict(r["e"])) for r in records]
        except Exception as e:
            self.logger.warning(f"Export entities failed: {e}")
            return []

    async def export_relations(self) -> list[dict]:
        """Export all relations."""
        cypher = """
        MATCH (a:Entity)-[r]->(b:Entity)
        RETURN a.id AS source_id, a.name AS source_name,
               type(r) AS relation_type,
               b.id AS target_id, b.name AS target_name
        LIMIT 50000
        """
        try:
            async with self.session() as session:
                result = await session.run(cypher)
                records = await result.data()
                return [_serialize_neo4j_value(r) for r in records]
        except Exception as e:
            self.logger.warning(f"Export relations failed: {e}")
            return []

    async def export_documents(
        self, doc_ids: list[str] | None = None
    ) -> list[dict]:
        """Export document metadata."""
        doc_filter = ""
        params: dict[str, Any] = {}
        if doc_ids:
            doc_filter = "WHERE d.id IN $doc_ids"
            params["doc_ids"] = doc_ids

        cypher = f"""
        MATCH (d:Document)
        {doc_filter}
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, count(c) AS chunk_count
        RETURN d.id AS id, d.filename AS filename,
               d.created_at AS created_at, chunk_count
        ORDER BY d.created_at DESC
        LIMIT 10000
        """
        try:
            async with self.session() as session:
                result = await session.run(cypher, parameters=params)
                records = await result.data()
                return [_serialize_neo4j_value(r) for r in records]
        except Exception as e:
            self.logger.warning(f"Export documents failed: {e}")
            return []

    async def delete_document(
        self, doc_id: str, delete_chunks: bool = True, delete_entities: bool = False
    ) -> tuple[int, int]:
        """Delete a document and optionally its chunks and entities."""
        deleted_chunks = 0
        deleted_entities = 0

        try:
            async with self.session() as session:
                if delete_chunks:
                    result = await session.run(
                        """
                        MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                        DETACH DELETE c
                        RETURN count(c) AS deleted
                        """,
                        parameters={"doc_id": doc_id},
                    )
                    record = await result.single()
                    deleted_chunks = record["deleted"] if record else 0

                if delete_entities:
                    result = await session.run(
                        """
                        MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
                        WHERE NOT EXISTS {
                            MATCH (e)<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(other:Document)
                            WHERE other.id <> $doc_id
                        }
                        DETACH DELETE e
                        RETURN count(e) AS deleted
                        """,
                        parameters={"doc_id": doc_id},
                    )
                    record = await result.single()
                    deleted_entities = record["deleted"] if record else 0

                # Delete the document node itself
                await session.run(
                    "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
                    parameters={"doc_id": doc_id},
                )

            return deleted_chunks, deleted_entities

        except Exception as e:
            self.logger.error(f"Delete document failed: {e}")
            raise

    async def delete_orphaned_entities(self) -> int:
        """Delete entities with no relations."""
        cypher = """
        MATCH (e:Entity)
        WHERE NOT (e)-[]-()
        DELETE e
        RETURN count(e) AS deleted
        """
        try:
            async with self.session() as session:
                result = await session.run(cypher)
                record = await result.single()
                return record["deleted"] if record else 0
        except Exception as e:
            self.logger.warning(f"Delete orphaned entities failed: {e}")
            return 0

    async def delete_orphaned_chunks(self) -> int:
        """Delete chunks with no parent document."""
        cypher = """
        MATCH (c:Chunk)
        WHERE NOT (c)<-[:HAS_CHUNK]-(:Document)
        DETACH DELETE c
        RETURN count(c) AS deleted
        """
        try:
            async with self.session() as session:
                result = await session.run(cypher)
                record = await result.single()
                return record["deleted"] if record else 0
        except Exception as e:
            self.logger.warning(f"Delete orphaned chunks failed: {e}")
            return 0

    async def update_entity_attributes(
        self, entity_id: str, attributes: dict, merge: bool = True
    ) -> dict:
        """Update entity attributes."""
        if merge:
            cypher = """
            MATCH (e:Entity {id: $entity_id})
            SET e += $attributes
            RETURN properties(e) AS props
            """
        else:
            cypher = """
            MATCH (e:Entity {id: $entity_id})
            SET e = $attributes
            SET e.id = $entity_id
            RETURN properties(e) AS props
            """
        try:
            async with self.session() as session:
                result = await session.run(
                    cypher,
                    parameters={"entity_id": entity_id, "attributes": attributes},
                )
                record = await result.single()
                return record["props"] if record else {}
        except Exception as e:
            self.logger.error(f"Update entity attributes failed: {e}")
            raise

    async def __aenter__(self) -> "Neo4jClient":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
