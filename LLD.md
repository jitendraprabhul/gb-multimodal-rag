# Low-Level Design (LLD)

## GraphRAG Multimodal RAG System

**Version**: 1.0.0
**Date**: 2026-01-22
**Status**: Production Ready

---

## 1. Introduction

This document provides detailed low-level design specifications for the GraphRAG Multimodal RAG System, including class structures, detailed algorithms, database schemas, API contracts, and implementation details.

---

## 2. Project Structure

```
gb-multimodal-rag/
├── src/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── api/                       # API layer
│   │   ├── __init__.py
│   │   ├── routes.py              # Main API routes (ask, ingest)
│   │   ├── admin_routes.py        # Admin API (key management)
│   │   ├── data_management_routes.py  # Data CRUD operations
│   │   ├── auth.py                # Authentication & authorization
│   │   ├── monitoring.py          # Metrics collection
│   │   └── dependencies.py        # Dependency injection
│   ├── core/                      # Core types and utilities
│   │   ├── __init__.py
│   │   ├── types.py               # Data models (Document, Chunk, Entity)
│   │   ├── config.py              # Configuration management
│   │   └── container.py           # Service container (DI)
│   ├── etl/                       # Document processing pipeline
│   │   ├── __init__.py
│   │   ├── loaders.py             # Document loaders (PDF, TXT, Image)
│   │   ├── extractors.py          # Text, table, image extractors
│   │   ├── chunkers.py            # Text chunking strategies
│   │   ├── ner_extractor.py       # Named entity recognition
│   │   ├── relation_extractor.py  # Relation extraction
│   │   └── pipeline.py            # ETL orchestration
│   ├── kg/                        # Knowledge graph
│   │   ├── __init__.py
│   │   ├── neo4j_client.py        # Neo4j database client
│   │   ├── graph_builder.py       # Graph construction
│   │   └── query_engine.py        # Graph queries
│   ├── vector/                    # Vector database
│   │   ├── __init__.py
│   │   ├── qdrant_client.py       # Qdrant client wrapper
│   │   └── embeddings.py          # Embedding service
│   ├── retrieval/                 # Retrieval components
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py    # Hybrid retrieval
│   │   ├── vector_retriever.py    # Vector search
│   │   └── graph_retriever.py     # Graph traversal
│   ├── llm/                       # LLM integration
│   │   ├── __init__.py
│   │   ├── ollama_client.py       # Ollama API client
│   │   └── prompts.py             # Prompt templates
│   └── services/                  # High-level services
│       ├── __init__.py
│       ├── reasoner.py            # Graph-aware reasoning
│       └── ingestion_service.py   # Document ingestion
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures
│   ├── test_api_auth.py
│   ├── test_api_routes.py
│   ├── test_etl_pipeline.py
│   └── test_monitoring.py
├── scripts/                       # Utility scripts
│   ├── ingest.py                  # CLI for ingestion
│   ├── manage_keys.py             # API key management
│   ├── backup.py                  # Backup utility
│   └── restore.py                 # Restore utility
├── config/                        # Configuration files
│   └── domains/                   # Domain-specific configs
│       ├── finance.yaml
│       └── healthcare.yaml
├── .github/workflows/             # CI/CD
│   └── ci.yml
├── docker-compose.yml             # Production deployment
├── docker-compose.dev.yml         # Development deployment
├── Dockerfile                     # Application image
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Test configuration
├── .env.example                   # Environment template
└── README.md                      # Documentation
```

---

## 3. Core Data Models

### 3.1 Document Model

**File**: `src/core/types.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class Document(BaseModel):
    """Represents a document in the system."""

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    source: str = Field(..., description="Source file path or URL")
    content: str = Field(..., description="Extracted text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

### 3.2 Chunk Model

```python
class Chunk(BaseModel):
    """Represents a text chunk."""

    id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk text content")
    position: int = Field(..., description="Position in document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 3.3 Entity Model

```python
class Entity(BaseModel):
    """Represents a named entity."""

    id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (PERSON, ORG, etc.)")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    source_ids: List[str] = Field(default_factory=list, description="Source chunk IDs")
```

### 3.4 Relation Model

```python
class Relation(BaseModel):
    """Represents a relation between entities."""

    id: str = Field(..., description="Unique relation identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relation type")
    weight: float = Field(1.0, ge=0.0, description="Relation strength")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 3.5 API Request/Response Models

```python
class AskRequest(BaseModel):
    """Request model for question answering."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    domain: Optional[str] = Field(None, description="Domain filter")

class AskResponse(BaseModel):
    """Response model for question answering."""

    answer: str = Field(..., description="Generated answer")
    citations: List[str] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(..., description="Answer confidence score")
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved")
    processing_time_ms: float = Field(..., description="Processing time")
```

---

## 4. Database Schemas

### 4.1 Neo4j Graph Schema

**Node Labels**:

```cypher
// Document node
(:Document {
    id: STRING,              // Unique identifier
    title: STRING,           // Document title
    source: STRING,          // File path or URL
    created_at: DATETIME,    // Creation timestamp
    metadata: MAP            // Additional metadata
})

// Chunk node
(:Chunk {
    id: STRING,              // Unique identifier
    doc_id: STRING,          // Parent document ID
    text: STRING,            // Chunk text content
    position: INTEGER,       // Position in document
    metadata: MAP            // Additional metadata
})

// Entity node
(:Entity {
    id: STRING,              // Unique identifier
    name: STRING,            // Entity name
    type: STRING,            // Entity type (PERSON, ORG, etc.)
    attributes: MAP,         // Entity attributes
    confidence: FLOAT        // Extraction confidence
})
```

**Relationship Types**:

```cypher
// Document to Chunk
(:Document)-[:HAS_CHUNK {position: INTEGER}]->(:Chunk)

// Chunk to Entity (mentions)
(:Chunk)-[:MENTIONS {count: INTEGER}]->(:Entity)

// Entity to Entity (relations)
(:Entity)-[:RELATES_TO {
    type: STRING,           // Relation type
    weight: FLOAT,          // Relation strength
    source_ids: LIST        // Source chunk IDs
}]->(:Entity)
```

**Indexes**:

```cypher
CREATE INDEX document_id FOR (d:Document) ON (d.id)
CREATE INDEX chunk_id FOR (c:Chunk) ON (c.id)
CREATE INDEX chunk_doc_id FOR (c:Chunk) ON (c.doc_id)
CREATE INDEX entity_id FOR (e:Entity) ON (e.id)
CREATE INDEX entity_name FOR (e:Entity) ON (e.name)
CREATE INDEX entity_type FOR (e:Entity) ON (e.type)
```

### 4.2 Qdrant Collection Schema

**Collection: chunks**

```json
{
  "vectors": {
    "size": 768,
    "distance": "Cosine"
  },
  "payload_schema": {
    "doc_id": "keyword",
    "chunk_id": "keyword",
    "text": "text",
    "position": "integer",
    "entity_ids": "keyword[]"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "quantile": 0.99
    }
  }
}
```

**Collection: entities**

```json
{
  "vectors": {
    "size": 768,
    "distance": "Cosine"
  },
  "payload_schema": {
    "entity_id": "keyword",
    "name": "text",
    "type": "keyword",
    "attributes": "text"
  }
}
```

---

## 5. Core Components Implementation

### 5.1 Authentication System

**File**: `src/api/auth.py`

**Class: APIKeyManager**

```python
class APIKeyManager:
    """Manages API keys and authentication."""

    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._rate_limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._create_default_key()

    def create_key(
        self,
        name: str,
        rate_limit: int = 100,
        daily_limit: int = 10000
    ) -> str:
        """
        Create a new API key.

        Args:
            name: Key name/description
            rate_limit: Requests per minute
            daily_limit: Requests per day

        Returns:
            Plain text API key (only shown once)
        """
        key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(key)

        api_key = APIKey(
            key_hash=key_hash,
            name=name,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow(),
            is_active=True
        )

        self._keys[key_hash] = api_key
        return key

    def validate_key(self, key: str) -> Tuple[bool, Optional[APIKey]]:
        """
        Validate an API key.

        Returns:
            Tuple of (is_valid, api_key_object)
        """
        key_hash = self._hash_key(key)
        api_key = self._keys.get(key_hash)

        if not api_key or not api_key.is_active:
            return False, None

        api_key.last_used = datetime.utcnow()
        return True, api_key

    def check_rate_limit(
        self,
        key_hash: str,
        api_key: APIKey
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        rate_info = self._rate_limits[key_hash]
        now = datetime.utcnow()

        # Per-minute rate limiting
        if rate_info.minute_key != now.strftime("%Y%m%d%H%M"):
            rate_info.minute_key = now.strftime("%Y%m%d%H%M")
            rate_info.minute_count = 0

        if rate_info.minute_count >= api_key.rate_limit:
            return False, f"Rate limit exceeded: {api_key.rate_limit} req/min"

        # Daily rate limiting
        if rate_info.day_key != now.strftime("%Y%m%d"):
            rate_info.day_key = now.strftime("%Y%m%d")
            rate_info.day_count = 0

        if rate_info.day_count >= api_key.daily_limit:
            return False, f"Daily limit exceeded: {api_key.daily_limit} req/day"

        # Increment counters
        rate_info.minute_count += 1
        rate_info.day_count += 1

        return True, None

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()
```

**Dependency Function**:

```python
async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key")
) -> APIKey:
    """
    Verify API key from request header.

    Raises:
        HTTPException: If key is invalid or rate limited
    """
    manager = get_api_key_manager()

    is_valid, api_key = manager.validate_key(x_api_key)
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key"
        )

    is_allowed, error = manager.check_rate_limit(api_key.key_hash, api_key)
    if not is_allowed:
        raise HTTPException(status_code=429, detail=error)

    return api_key
```

### 5.2 Monitoring System

**File**: `src/api/monitoring.py`

**Class: MetricsCollector**

```python
class MetricsCollector:
    """Collects and stores application metrics."""

    def __init__(self, max_latency_samples: int = 1000):
        self.request_count: Dict[str, int] = defaultdict(int)
        self.request_latencies: Dict[str, List[float]] = defaultdict(list)
        self.status_codes: Dict[int, int] = defaultdict(int)
        self.component_metrics: Dict[str, ComponentMetrics] = defaultdict(ComponentMetrics)
        self.recent_errors: List[ErrorLog] = []
        self.start_time = datetime.utcnow()
        self.max_latency_samples = max_latency_samples

    def track_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        error: str = None
    ) -> None:
        """Track an API request."""
        endpoint = f"{method} {path}"

        self.request_count[endpoint] += 1
        self.status_codes[status_code] += 1

        # Store latency with circular buffer
        latencies = self.request_latencies[endpoint]
        latencies.append(duration_ms)
        if len(latencies) > self.max_latency_samples:
            latencies.pop(0)

        # Track errors
        if error:
            self.recent_errors.append(ErrorLog(
                timestamp=datetime.utcnow(),
                endpoint=endpoint,
                error=error,
                status_code=status_code
            ))
            if len(self.recent_errors) > 100:
                self.recent_errors.pop(0)

    def track_component(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """Track component-level metrics."""
        metrics = self.component_metrics[component]
        metrics.total_calls += 1
        if success:
            metrics.successful_calls += 1

        metrics.total_duration_ms += duration_ms
        metrics.latencies.append(duration_ms)

        if len(metrics.latencies) > self.max_latency_samples:
            metrics.latencies.pop(0)

    def get_metrics(self) -> dict:
        """Get current metrics snapshot."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "uptime_seconds": uptime,
            "request_count": dict(self.request_count),
            "status_codes": dict(self.status_codes),
            "latency_percentiles": self._calculate_percentiles(),
            "component_metrics": self._format_component_metrics(),
            "recent_errors": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "endpoint": e.endpoint,
                    "error": e.error,
                    "status_code": e.status_code
                }
                for e in self.recent_errors[-10:]
            ]
        }

    def _calculate_percentiles(self) -> dict:
        """Calculate latency percentiles for each endpoint."""
        result = {}
        for endpoint, latencies in self.request_latencies.items():
            if latencies:
                sorted_lat = sorted(latencies)
                result[endpoint] = {
                    "p50": self._percentile(sorted_lat, 50),
                    "p95": self._percentile(sorted_lat, 95),
                    "p99": self._percentile(sorted_lat, 99),
                    "mean": sum(sorted_lat) / len(sorted_lat)
                }
        return result

    @staticmethod
    def _percentile(sorted_list: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not sorted_list:
            return 0.0
        index = int(len(sorted_list) * (percentile / 100.0))
        return sorted_list[min(index, len(sorted_list) - 1)]
```

**Middleware**:

```python
class RequestTimingMiddleware:
    """Middleware to track request timing."""

    def __init__(self, app):
        self.app = app
        self.collector = get_metrics_collector()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration_ms = (time.time() - start_time) * 1000
                status_code = message["status"]

                self.collector.track_request(
                    method=scope["method"],
                    path=scope["path"],
                    status_code=status_code,
                    duration_ms=duration_ms
                )

            await send(message)

        await self.app(scope, receive, send_wrapper)
```

### 5.3 ETL Pipeline

**File**: `src/etl/pipeline.py`

**Class: ETLPipeline**

```python
class ETLPipeline:
    """Orchestrates document processing pipeline."""

    def __init__(
        self,
        loaders: Dict[str, BaseLoader],
        extractor: TextExtractor,
        chunker: BaseChunker,
        ner_extractor: NERExtractor,
        relation_extractor: RelationExtractor,
        embedding_service: EmbeddingService,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient
    ):
        self.loaders = loaders
        self.extractor = extractor
        self.chunker = chunker
        self.ner_extractor = ner_extractor
        self.relation_extractor = relation_extractor
        self.embedding_service = embedding_service
        self.neo4j = neo4j_client
        self.qdrant = qdrant_client

    async def process_document(
        self,
        file_path: Path,
        doc_id: str = None
    ) -> Document:
        """
        Process a single document through the ETL pipeline.

        Steps:
        1. Load document
        2. Extract text
        3. Chunk text
        4. Extract entities and relations
        5. Generate embeddings
        6. Store in Neo4j and Qdrant

        Args:
            file_path: Path to document file
            doc_id: Optional document ID

        Returns:
            Processed Document object
        """
        # Step 1: Load document
        loader = self._get_loader(file_path)
        raw_doc = await loader.load(file_path)

        # Step 2: Extract text
        text = await self.extractor.extract(raw_doc)

        # Step 3: Create document object
        doc_id = doc_id or str(uuid.uuid4())
        document = Document(
            id=doc_id,
            title=file_path.stem,
            source=str(file_path),
            content=text,
            metadata={"file_type": file_path.suffix}
        )

        # Step 4: Chunk text
        chunks = await self.chunker.chunk(text, doc_id)

        # Step 5: Extract entities (parallel with chunking)
        entities = await self.ner_extractor.extract(text)

        # Step 6: Extract relations
        relations = await self.relation_extractor.extract(text, entities)

        # Step 7: Generate embeddings (parallel)
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedding_service.embed_batch(chunk_texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Step 8: Store in databases (parallel)
        await asyncio.gather(
            self._store_in_neo4j(document, chunks, entities, relations),
            self._store_in_qdrant(chunks)
        )

        return document

    async def _store_in_neo4j(
        self,
        document: Document,
        chunks: List[Chunk],
        entities: List[Entity],
        relations: List[Relation]
    ) -> None:
        """Store document data in Neo4j."""
        # Create document node
        await self.neo4j.create_document(document)

        # Create chunk nodes and relationships
        for chunk in chunks:
            await self.neo4j.create_chunk(chunk)
            await self.neo4j.link_chunk_to_document(chunk.id, document.id)

        # Create entity nodes
        for entity in entities:
            await self.neo4j.create_or_update_entity(entity)

            # Link entities to chunks where they appear
            for chunk in chunks:
                if entity.name.lower() in chunk.text.lower():
                    await self.neo4j.link_chunk_to_entity(chunk.id, entity.id)

        # Create relation edges
        for relation in relations:
            await self.neo4j.create_relation(relation)

    async def _store_in_qdrant(self, chunks: List[Chunk]) -> None:
        """Store chunk embeddings in Qdrant."""
        points = [
            {
                "id": chunk.id,
                "vector": chunk.embedding,
                "payload": {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "position": chunk.position
                }
            }
            for chunk in chunks
        ]

        await self.qdrant.upsert_batch("chunks", points)
```

### 5.4 Hybrid Retriever

**File**: `src/retrieval/hybrid_retriever.py`

**Class: HybridRetriever**

```python
class HybridRetriever:
    """Combines vector and graph-based retrieval."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        embedding_service: EmbeddingService,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.embedding_service = embedding_service
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_k: int = 20,
        graph_hops: int = 2
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining vector and graph search.

        Algorithm:
        1. Embed query
        2. Vector search (top_k * 4 results)
        3. Extract entities from top vector results
        4. Graph traversal (2-hop expansion)
        5. Reciprocal Rank Fusion (RRF)
        6. Re-rank and return top_k

        Args:
            query: User question
            top_k: Final number of results
            vector_k: Number of vector results to fetch
            graph_hops: Graph traversal depth

        Returns:
            List of RetrievalResult objects
        """
        # Step 1: Embed query
        query_embedding = await self.embedding_service.embed(query)

        # Step 2: Vector search
        vector_results = await self.vector_retriever.search(
            query_embedding,
            top_k=vector_k
        )

        # Step 3: Extract entities from top results
        entity_ids = await self._extract_entities(vector_results[:10])

        # Step 4: Graph traversal
        graph_results = await self.graph_retriever.traverse(
            entity_ids,
            hops=graph_hops
        )

        # Step 5: Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            graph_results,
            self.vector_weight,
            self.graph_weight
        )

        # Step 6: Return top_k
        return fused_results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        vector_weight: float,
        graph_weight: float,
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion algorithm.

        RRF Score = Σ (weight / (k + rank))

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            vector_weight: Weight for vector scores
            graph_weight: Weight for graph scores
            k: RRF constant (default 60)

        Returns:
            Fused and sorted results
        """
        scores: Dict[str, float] = defaultdict(float)
        chunks: Dict[str, RetrievalResult] = {}

        # Score vector results
        for rank, result in enumerate(vector_results, 1):
            rrf_score = vector_weight / (k + rank)
            scores[result.chunk_id] += rrf_score
            chunks[result.chunk_id] = result

        # Score graph results
        for rank, result in enumerate(graph_results, 1):
            rrf_score = graph_weight / (k + rank)
            scores[result.chunk_id] += rrf_score
            if result.chunk_id not in chunks:
                chunks[result.chunk_id] = result

        # Sort by combined score
        sorted_chunk_ids = sorted(
            scores.keys(),
            key=lambda cid: scores[cid],
            reverse=True
        )

        # Create results with combined scores
        fused = []
        for chunk_id in sorted_chunk_ids:
            result = chunks[chunk_id]
            result.score = scores[chunk_id]
            fused.append(result)

        return fused
```

### 5.5 Graph-Aware Reasoner

**File**: `src/services/reasoner.py`

**Class: GraphAwareReasoner**

```python
class GraphAwareReasoner:
    """Generates answers using LLM with graph context."""

    def __init__(
        self,
        llm_client: OllamaClient,
        prompt_template: str = None
    ):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or self._default_template()

    async def reason(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        graph_context: Optional[dict] = None
    ) -> ReasoningResponse:
        """
        Generate answer using LLM with retrieval context.

        Args:
            question: User question
            retrieval_results: Retrieved chunks and entities
            graph_context: Optional graph structure info

        Returns:
            ReasoningResponse with answer and citations
        """
        # Step 1: Aggregate context
        context = self._aggregate_context(retrieval_results, graph_context)

        # Step 2: Build prompt
        prompt = self._build_prompt(question, context)

        # Step 3: Generate answer
        start_time = time.time()
        llm_response = await self.llm_client.generate(
            prompt,
            temperature=0.3,
            max_tokens=2048
        )
        inference_time = time.time() - start_time

        # Step 4: Extract citations
        citations = self._extract_citations(llm_response, retrieval_results)

        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(
            llm_response,
            retrieval_results
        )

        return ReasoningResponse(
            answer=llm_response,
            citations=citations,
            confidence=confidence,
            inference_time_ms=inference_time * 1000,
            num_chunks_used=len(retrieval_results)
        )

    def _aggregate_context(
        self,
        results: List[RetrievalResult],
        graph_context: Optional[dict]
    ) -> str:
        """Aggregate retrieval results into context string."""
        context_parts = []

        # Add chunk contexts
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {result.text}\n"
                f"    Source: {result.source}\n"
            )

        # Add graph context if available
        if graph_context:
            entities = graph_context.get("entities", [])
            if entities:
                context_parts.append("\nKey Entities:")
                for entity in entities:
                    context_parts.append(
                        f"- {entity['name']} ({entity['type']})"
                    )

            relations = graph_context.get("relations", [])
            if relations:
                context_parts.append("\nRelationships:")
                for rel in relations:
                    context_parts.append(
                        f"- {rel['source']} {rel['type']} {rel['target']}"
                    )

        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build LLM prompt with question and context."""
        return self.prompt_template.format(
            context=context,
            question=question
        )

    @staticmethod
    def _default_template() -> str:
        """Default prompt template."""
        return """You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Provide a clear, concise answer based ONLY on the context
2. If the context doesn't contain enough information, say so
3. Cite sources using [number] notation
4. Be factual and avoid speculation

Answer:"""

    def _extract_citations(
        self,
        response: str,
        results: List[RetrievalResult]
    ) -> List[str]:
        """Extract citation references from response."""
        citations = []
        for i, result in enumerate(results, 1):
            if f"[{i}]" in response:
                citations.append(result.source)
        return citations

    def _calculate_confidence(
        self,
        response: str,
        results: List[RetrievalResult]
    ) -> float:
        """Calculate confidence score for answer."""
        # Simple heuristic based on:
        # - Number of citations
        # - Average retrieval score
        # - Presence of uncertainty phrases

        num_citations = response.count("[")
        avg_score = sum(r.score for r in results) / len(results) if results else 0

        uncertainty_phrases = [
            "i don't know",
            "unclear",
            "not enough information",
            "cannot determine"
        ]
        has_uncertainty = any(
            phrase in response.lower()
            for phrase in uncertainty_phrases
        )

        confidence = min(
            avg_score * (1 + num_citations * 0.1),
            1.0
        )

        if has_uncertainty:
            confidence *= 0.5

        return round(confidence, 2)
```

---

## 6. API Endpoint Specifications

### 6.1 Question Answering Endpoint

**Endpoint**: `POST /api/v1/ask`

**Request**:

```json
{
  "question": "What are the risk factors for diabetes?",
  "top_k": 5,
  "domain": "healthcare"
}
```

**Response**:

```json
{
  "answer": "Based on the documents, key risk factors for diabetes include: [1] family history, [2] obesity, [3] physical inactivity, and [4] age over 45.",
  "citations": ["medical_guidelines.pdf", "diabetes_research_2025.pdf"],
  "confidence": 0.87,
  "retrieved_chunks": 5,
  "processing_time_ms": 1523.45
}
```

**Status Codes**:

- `200`: Success
- `400`: Invalid request
- `401`: Unauthorized
- `429`: Rate limit exceeded
- `500`: Server error

### 6.2 Document Ingestion Endpoint

**Endpoint**: `POST /api/v1/ingest`

**Request** (multipart/form-data):

```
file: document.pdf
metadata: {"domain": "finance", "category": "annual_report"}
```

**Response**:

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "chunks_created": 45,
  "entities_extracted": 123,
  "relations_extracted": 89,
  "processing_time_ms": 5432.1,
  "status": "success"
}
```

### 6.3 Health Check Endpoint

**Endpoint**: `GET /api/v1/health`

**Response**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "neo4j": {
      "status": "up",
      "response_time_ms": 12.3
    },
    "qdrant": {
      "status": "up",
      "response_time_ms": 8.7
    },
    "ollama": {
      "status": "up",
      "model": "llama3.1:8b"
    }
  },
  "uptime_seconds": 86400
}
```

---

## 7. Algorithms

### 7.1 Text Chunking Algorithm

**Strategy**: Recursive Character Splitting with Overlap

```python
def recursive_chunk(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: List[str] = ["\n\n", "\n", ". ", " "]
) -> List[str]:
    """
    Recursively split text using hierarchical separators.

    Algorithm:
    1. Try splitting by first separator
    2. If chunks too large, use next separator
    3. If still too large, split by characters
    4. Add overlap between chunks
    """
    chunks = []

    for sep in separators:
        splits = text.split(sep)
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) <= chunk_size:
                current_chunk += split + sep
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from previous chunk
                    overlap_text = current_chunk[-overlap:] if overlap else ""
                    current_chunk = overlap_text + split + sep
                else:
                    current_chunk = split + sep

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Check if chunking succeeded
        if all(len(c) <= chunk_size for c in chunks):
            return chunks

    # Fallback: character-level splitting
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size - overlap)
    ]
```

### 7.2 Entity Extraction Algorithm

**Strategy**: Hybrid NER (spaCy + Pattern Matching)

```python
async def extract_entities(
    text: str,
    domain: str = "general"
) -> List[Entity]:
    """
    Extract entities using hybrid approach.

    Steps:
    1. Run spaCy NER model
    2. Apply domain-specific patterns
    3. Merge and deduplicate
    4. Calculate confidence scores
    """
    entities = []

    # Step 1: spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        entities.append(Entity(
            id=str(uuid.uuid4()),
            name=ent.text,
            type=ent.label_,
            confidence=0.8,  # Base confidence for spaCy
            attributes={"start": ent.start_char, "end": ent.end_char}
        ))

    # Step 2: Domain-specific patterns
    if domain == "finance":
        # Extract stock tickers: $AAPL, $GOOGL
        ticker_pattern = r'\$([A-Z]{1,5})\b'
        for match in re.finditer(ticker_pattern, text):
            entities.append(Entity(
                id=str(uuid.uuid4()),
                name=match.group(1),
                type="TICKER",
                confidence=0.9,
                attributes={"pattern": "ticker"}
            ))

        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        # ... similar logic

    # Step 3: Deduplicate
    unique_entities = _deduplicate_entities(entities)

    return unique_entities
```

### 7.3 Graph Traversal Algorithm

**Strategy**: Bidirectional BFS with Entity Scoring

```python
async def traverse_graph(
    seed_entity_ids: List[str],
    hops: int = 2,
    max_entities: int = 50
) -> List[str]:
    """
    Traverse graph from seed entities.

    Algorithm:
    1. Start from seed entities
    2. Expand bidirectionally (incoming + outgoing)
    3. Score entities by:
       - Distance from seed
       - Relation weights
       - Entity type importance
    4. Return top-scored entities
    """
    visited = set(seed_entity_ids)
    current_level = seed_entity_ids
    entity_scores = {eid: 1.0 for eid in seed_entity_ids}

    for hop in range(hops):
        next_level = set()
        decay_factor = 0.5 ** (hop + 1)  # Exponential decay

        for entity_id in current_level:
            # Get neighbors (both directions)
            neighbors = await neo4j.get_neighbors(entity_id)

            for neighbor in neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    next_level.add(neighbor.id)

                    # Calculate score
                    score = (
                        decay_factor *
                        neighbor.relation_weight *
                        _entity_type_weight(neighbor.type)
                    )
                    entity_scores[neighbor.id] = score

        current_level = list(next_level)

    # Sort by score and return top entities
    sorted_entities = sorted(
        entity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [eid for eid, score in sorted_entities[:max_entities]]
```

---

## 8. Error Handling

### 8.1 Error Hierarchy

```python
class GraphRAGError(Exception):
    """Base exception for GraphRAG."""
    pass

class DocumentProcessingError(GraphRAGError):
    """Raised when document processing fails."""
    pass

class EntityExtractionError(GraphRAGError):
    """Raised when entity extraction fails."""
    pass

class DatabaseConnectionError(GraphRAGError):
    """Raised when database connection fails."""
    pass

class LLMGenerationError(GraphRAGError):
    """Raised when LLM generation fails."""
    pass

class AuthenticationError(GraphRAGError):
    """Raised when authentication fails."""
    pass

class RateLimitError(GraphRAGError):
    """Raised when rate limit is exceeded."""
    pass
```

### 8.2 Error Handling Strategy

```python
@router.post("/ask")
async def ask_question(request: AskRequest):
    """Question answering endpoint with error handling."""
    try:
        # Main logic
        result = await process_question(request)
        return result

    except AuthenticationError as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail=str(e))

    except RateLimitError as e:
        logger.warning(f"Rate limit: {e}")
        raise HTTPException(status_code=429, detail=str(e))

    except DatabaseConnectionError as e:
        logger.error(f"DB connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database temporarily unavailable"
        )

    except LLMGenerationError as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Answer generation failed"
        )

    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

---

## 9. Configuration Management

### 9.1 Configuration Hierarchy

```
Environment Variables (.env)
    ↓
Config File (config.yaml)
    ↓
Domain Config (config/domains/finance.yaml)
    ↓
Runtime Overrides (API parameters)
```

### 9.2 Settings Class

**File**: `src/core/config.py`

```python
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    app_name: str = "GraphRAG Multimodal RAG"
    app_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Neo4j Settings
    neo4j_uri: str = Field(..., env="NEO4J_URI")
    neo4j_user: str = Field(..., env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")

    # Qdrant Settings
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = "chunks"

    # Ollama Settings
    ollama_base_url: str = Field(..., env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        env="OLLAMA_EMBEDDING_MODEL"
    )

    # ETL Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 32

    # Retrieval Settings
    vector_top_k: int = 20
    graph_hops: int = 2
    final_top_k: int = 5
    vector_weight: float = 0.6
    graph_weight: float = 0.4

    # Authentication
    api_auth_enabled: bool = True
    default_api_key: str = Field(..., env="DEFAULT_API_KEY")
    default_rate_limit: int = 100
    default_daily_limit: int = 10000

    # Monitoring
    metrics_enabled: bool = True
    request_logging: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
```

---

## 10. Testing Strategy

### 10.1 Test Structure

```
tests/
├── unit/                      # Unit tests
│   ├── test_chunkers.py
│   ├── test_extractors.py
│   └── test_retrievers.py
├── integration/               # Integration tests
│   ├── test_etl_pipeline.py
│   ├── test_api_routes.py
│   └── test_database.py
└── e2e/                       # End-to-end tests
    └── test_user_workflows.py
```

### 10.2 Test Examples

**Unit Test**:

```python
@pytest.mark.asyncio
async def test_chunk_text():
    """Test text chunking."""
    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    text = "This is a test. " * 50

    chunks = await chunker.chunk(text, doc_id="test-doc")

    assert len(chunks) > 0
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[0].doc_id == "test-doc"
```

**Integration Test**:

```python
@pytest.mark.asyncio
async def test_document_ingestion(test_client, sample_pdf):
    """Test full document ingestion flow."""
    with open(sample_pdf, "rb") as f:
        response = await test_client.post(
            "/api/v1/ingest",
            files={"file": f},
            headers={"X-API-Key": "test-key"}
        )

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["chunks_created"] > 0
    assert data["entities_extracted"] > 0
```

---

## 11. Deployment Configuration

### 11.1 Docker Compose (Production)

**File**: `docker-compose.yml`

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - neo4j
      - qdrant
      - ollama
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "2"
          memory: 4G

  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  neo4j_data:
  qdrant_data:
  ollama_data:
```

### 11.2 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY scripts/ ./scripts/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 12. Performance Optimizations

### 12.1 Database Query Optimization

**Neo4j Indexes**:

```cypher
CREATE INDEX entity_name_idx FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_idx FOR (e:Entity) ON (e.type);
CREATE FULLTEXT INDEX entity_text_idx FOR (e:Entity) ON EACH [e.name, e.attributes];
```

**Query Optimization**:

```cypher
// Bad: Scans all entities
MATCH (e:Entity)
WHERE e.name CONTAINS 'Apple'
RETURN e

// Good: Uses index
MATCH (e:Entity)
WHERE e.name = 'Apple Inc.'
RETURN e
```

### 12.2 Caching Strategy

```python
from functools import lru_cache
from typing import List

class CachedEmbeddingService:
    """Embedding service with caching."""

    @lru_cache(maxsize=10000)
    async def embed(self, text: str) -> List[float]:
        """Embed text with caching."""
        # LRU cache for frequent queries
        return await self._compute_embedding(text)

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """Batch embedding with parallelization."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.gather(
                *[self._compute_embedding(t) for t in batch]
            )
            results.extend(batch_embeddings)
        return results
```

---

## 13. Security Implementation

### 13.1 Input Validation

```python
from pydantic import validator, Field

class AskRequest(BaseModel):
    """Validated request model."""

    question: str = Field(..., min_length=1, max_length=1000)

    @validator('question')
    def sanitize_question(cls, v):
        """Sanitize user input."""
        # Remove potential injection attempts
        dangerous_patterns = ['<script', 'javascript:', 'onerror=']
        for pattern in dangerous_patterns:
            if pattern in v.lower():
                raise ValueError(f"Invalid input detected: {pattern}")
        return v.strip()
```

### 13.2 SQL Injection Prevention

```python
# Bad: String concatenation
query = f"MATCH (e:Entity {{name: '{user_input}'}}) RETURN e"

# Good: Parameterized query
query = "MATCH (e:Entity {name: $name}) RETURN e"
result = await session.run(query, name=user_input)
```

---

**Document Control**

| Version | Date       | Author   | Changes     |
| ------- | ---------- | -------- | ----------- |
| 1.0.0   | 2026-01-22 | Dev Team | Initial LLD |
