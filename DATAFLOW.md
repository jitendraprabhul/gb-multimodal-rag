# Data Flow Documentation

## GraphRAG Multimodal RAG System

**Version**: 1.0.0
**Date**: 2026-01-22
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#1-overview)
2. [Document Ingestion Flow](#2-document-ingestion-flow)
3. [Question Answering Flow](#3-question-answering-flow)
4. [Authentication Flow](#4-authentication-flow)
5. [Monitoring Data Flow](#5-monitoring-data-flow)
6. [Backup and Restore Flow](#6-backup-and-restore-flow)
7. [Data Management Flow](#7-data-management-flow)
8. [System Initialization Flow](#8-system-initialization-flow)

---

## 1. Overview

This document describes all data flows within the GraphRAG Multimodal RAG System, including request/response patterns, data transformations, and inter-component communication.

### 1.1 Data Flow Layers

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                         │
│              (HTTP Requests/Responses)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│                 API GATEWAY LAYER                       │
│  (Authentication, Rate Limiting, Request Validation)    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│               APPLICATION LAYER                         │
│       (Business Logic, Orchestration)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│                  DATA LAYER                             │
│     (Neo4j, Qdrant, File System)                       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Document Ingestion Flow

### 2.1 High-Level Ingestion Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. POST /api/v1/ingest
     │    (multipart/form-data: file + metadata)
     ↓
┌────────────────────────────────────────────┐
│         API Authentication                 │
│  - Verify API Key                          │
│  - Check Rate Limit                        │
└────┬───────────────────────────────────────┘
     │ 2. Validated Request
     ↓
┌────────────────────────────────────────────┐
│         Document Loader                    │
│  - Detect file type                        │
│  - Load binary content                     │
│  - Initial validation                      │
└────┬───────────────────────────────────────┘
     │ 3. Raw Document
     ↓
┌────────────────────────────────────────────┐
│         Text Extraction                    │
│  - PDF: Extract text + images              │
│  - Images: OCR (Tesseract)                 │
│  - Tables: Structure detection             │
└────┬───────────────────────────────────────┘
     │ 4. Extracted Text
     ↓
     ├──────────────────┬──────────────────┐
     │                  │                  │
     ↓                  ↓                  ↓
┌─────────┐    ┌──────────────┐   ┌────────────┐
│Chunking │    │Entity Extract│   │ Embedding  │
│         │    │              │   │ Generation │
└────┬────┘    └──────┬───────┘   └─────┬──────┘
     │                │                  │
     │ 5. Chunks      │ 6. Entities      │ 7. Vectors
     │                │                  │
     ├────────────────┼──────────────────┘
     │                │
     ↓                ↓
┌─────────────────────────────────────────┐
│      Relation Extraction                │
│  - Dependency parsing                   │
│  - Pattern matching                     │
│  - Entity co-occurrence                 │
└────┬────────────────────────────────────┘
     │ 8. Relations
     │
     ├─────────────────┬─────────────────┐
     │                 │                 │
     ↓                 ↓                 ↓
┌─────────┐    ┌──────────┐    ┌────────────┐
│  Neo4j  │    │ Qdrant   │    │   Files    │
│ Storage │    │ Storage  │    │  Storage   │
└────┬────┘    └────┬─────┘    └─────┬──────┘
     │              │                 │
     └──────────────┴─────────────────┘
                    │
                    ↓ 9. Success Response
            ┌───────────────┐
            │    Client     │
            └───────────────┘
```

### 2.2 Detailed Ingestion Steps

#### Step 1: Client Request

**Input**:

```http
POST /api/v1/ingest HTTP/1.1
Host: localhost:8000
X-API-Key: graphrag_key_abc123
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="document.pdf"
Content-Type: application/pdf

[Binary PDF Data]
--boundary
Content-Disposition: form-data; name="metadata"

{"domain": "finance", "category": "annual_report"}
--boundary--
```

**Data Elements**:

- File: Binary document data
- Metadata: JSON object with domain, category, tags
- API Key: Authentication token

#### Step 2: Authentication & Validation

**Process**:

```python
# Authentication
api_key = request.headers.get("X-API-Key")
is_valid, key_obj = api_key_manager.validate_key(api_key)

# Rate Limiting Check
is_allowed, error = api_key_manager.check_rate_limit(key_hash, key_obj)

# Request Validation
validated_request = IngestRequest(
    file=file,
    metadata=metadata
)
```

**Data Transformation**:

- Input: HTTP headers + form data
- Output: Validated request object

#### Step 3: Document Loading

**Process**:

```python
# Detect file type
file_extension = Path(filename).suffix
loader = loader_factory.get_loader(file_extension)

# Load document
raw_document = await loader.load(file_bytes)
```

**Data Transformation**:

- Input: Binary file data
- Output: RawDocument object with:
  - `content`: Raw bytes
  - `file_type`: MIME type
  - `pages`: Page count (for PDFs)

#### Step 4: Text Extraction

**For PDF Files**:

```python
# Extract text
text_extractor = PDFTextExtractor()
text = await text_extractor.extract(raw_document)

# Extract images
image_extractor = PDFImageExtractor()
images = await image_extractor.extract(raw_document)

# OCR on images (if needed)
for image in images:
    ocr_text = await tesseract_ocr.extract(image)
    text += "\n" + ocr_text
```

**Data Transformation**:

- Input: Raw PDF bytes
- Output: Extracted text string (5,000 - 50,000 words typical)

**For Images**:

```python
# OCR
ocr_result = await tesseract_ocr.extract(image_bytes)
text = ocr_result.text
confidence = ocr_result.confidence
```

**Data Transformation**:

- Input: Image bytes (PNG, JPG)
- Output: OCR text with confidence scores

#### Step 5: Text Chunking

**Process**:

```python
chunker = RecursiveChunker(
    chunk_size=1000,
    overlap=200,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = await chunker.chunk(text, doc_id)

# Example chunks:
# Chunk 1: [0:1000] "Annual Report 2025... financial statements..."
# Chunk 2: [800:1800] "...financial statements... revenue growth..."
# Chunk 3: [1600:2600] "...revenue growth... market analysis..."
```

**Data Transformation**:

- Input: Full document text (10,000 chars)
- Output: List of Chunk objects (10-50 chunks typical)

**Chunk Object Structure**:

```json
{
  "id": "chunk_550e8400",
  "doc_id": "doc_123",
  "text": "The company reported revenue of $1.2B...",
  "position": 0,
  "metadata": {
    "page": 1,
    "section": "financial_summary"
  }
}
```

#### Step 6: Entity Extraction

**Process**:

```python
# spaCy NER
nlp = spacy.load("en_core_web_lg")
doc = nlp(text)

entities = []
for ent in doc.ents:
    entity = Entity(
        id=generate_id(),
        name=ent.text,
        type=ent.label_,  # PERSON, ORG, GPE, MONEY, etc.
        confidence=0.8,
        source_ids=[chunk_id]
    )
    entities.append(entity)

# Domain-specific extraction
if domain == "finance":
    # Extract tickers: $AAPL, $GOOGL
    ticker_entities = extract_tickers(text)
    entities.extend(ticker_entities)
```

**Data Transformation**:

- Input: Text (chunk or full document)
- Output: List of Entity objects (50-200 entities typical)

**Entity Object Structure**:

```json
{
  "id": "entity_apple_inc",
  "name": "Apple Inc.",
  "type": "ORG",
  "attributes": {
    "ticker": "AAPL",
    "industry": "Technology"
  },
  "confidence": 0.92,
  "source_ids": ["chunk_1", "chunk_5", "chunk_12"]
}
```

#### Step 7: Embedding Generation

**Process**:

```python
# Generate embeddings for chunks
embedding_service = OllamaEmbeddingService(
    model="nomic-embed-text",
    dimension=768
)

chunk_texts = [chunk.text for chunk in chunks]
embeddings = await embedding_service.embed_batch(
    texts=chunk_texts,
    batch_size=32
)

# Assign embeddings to chunks
for chunk, embedding in zip(chunks, embeddings):
    chunk.embedding = embedding
```

**Data Transformation**:

- Input: List of text strings
- Output: List of 768-dimensional vectors

**Embedding Vector Example**:

```python
embedding = [
    0.023, -0.145, 0.892, ...,  # 768 dimensions
    -0.234, 0.567, -0.012
]
```

#### Step 8: Relation Extraction

**Process**:

```python
# Extract relations between entities
relation_extractor = DependencyRelationExtractor()

relations = []
for chunk in chunks:
    # Find entity mentions in chunk
    chunk_entities = find_entities_in_chunk(chunk, entities)

    # Extract relations
    chunk_relations = await relation_extractor.extract(
        chunk.text,
        chunk_entities
    )

    relations.extend(chunk_relations)
```

**Data Transformation**:

- Input: Text + Entity list
- Output: List of Relation objects

**Relation Object Structure**:

```json
{
  "id": "rel_123",
  "source_id": "entity_apple_inc",
  "target_id": "entity_tim_cook",
  "type": "CEO_OF",
  "weight": 0.85,
  "metadata": {
    "extracted_from": "chunk_7",
    "context": "Tim Cook, CEO of Apple Inc."
  }
}
```

#### Step 9: Storage (Parallel)

**Neo4j Storage**:

```cypher
// Create document node
CREATE (d:Document {
  id: $doc_id,
  title: $title,
  source: $source,
  created_at: datetime()
})

// Create chunk nodes and relationships
UNWIND $chunks AS chunk
CREATE (c:Chunk {
  id: chunk.id,
  text: chunk.text,
  position: chunk.position
})
CREATE (d)-[:HAS_CHUNK {position: chunk.position}]->(c)

// Create entity nodes (with merge for deduplication)
UNWIND $entities AS entity
MERGE (e:Entity {id: entity.id})
ON CREATE SET
  e.name = entity.name,
  e.type = entity.type,
  e.confidence = entity.confidence

// Create MENTIONS relationships
UNWIND $mentions AS mention
MATCH (c:Chunk {id: mention.chunk_id})
MATCH (e:Entity {id: mention.entity_id})
MERGE (c)-[:MENTIONS]->(e)

// Create RELATES_TO relationships
UNWIND $relations AS rel
MATCH (source:Entity {id: rel.source_id})
MATCH (target:Entity {id: rel.target_id})
MERGE (source)-[:RELATES_TO {
  type: rel.type,
  weight: rel.weight
}]->(target)
```

**Qdrant Storage**:

```python
# Prepare points
points = [
    {
        "id": chunk.id,
        "vector": chunk.embedding,
        "payload": {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.id,
            "text": chunk.text,
            "position": chunk.position,
            "entity_ids": [e.id for e in chunk.entities]
        }
    }
    for chunk in chunks
]

# Upsert to collection
await qdrant_client.upsert(
    collection_name="chunks",
    points=points
)
```

#### Step 10: Response

**Output**:

```json
{
  "document_id": "doc_550e8400",
  "filename": "annual_report_2025.pdf",
  "chunks_created": 45,
  "entities_extracted": 123,
  "relations_extracted": 89,
  "processing_time_ms": 5432.1,
  "status": "success",
  "metadata": {
    "pages": 32,
    "word_count": 12450,
    "unique_entities": 67
  }
}
```

---

## 3. Question Answering Flow

### 3.1 High-Level QA Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. POST /api/v1/ask {"question": "..."}
     ↓
┌────────────────────────────────────────────┐
│      Authentication & Validation           │
└────┬───────────────────────────────────────┘
     │ 2. Validated Question
     ↓
┌────────────────────────────────────────────┐
│      Query Embedding                       │
│  - Embed question with Ollama              │
└────┬───────────────────────────────────────┘
     │ 3. Query Vector [768-dim]
     ↓
     ├─────────────────────┬─────────────────────┐
     │                     │                     │
     ↓                     ↓                     ↓
┌──────────┐      ┌───────────────┐    ┌────────────┐
│  Qdrant  │      │   Neo4j       │    │  Metadata  │
│  Vector  │      │   Graph       │    │  Filters   │
│  Search  │      │   Traversal   │    │            │
└────┬─────┘      └───────┬───────┘    └─────┬──────┘
     │ 4a. Top-20        │ 4b. Related       │
     │     Chunks        │     Entities      │
     │                   │                   │
     └───────────────────┴───────────────────┘
                         │
                         ↓ 5. Combined Results
            ┌────────────────────────┐
            │ Reciprocal Rank Fusion │
            │    (RRF Algorithm)     │
            └────────┬───────────────┘
                     │ 6. Top-5 Ranked Chunks
                     ↓
            ┌────────────────────────┐
            │  Context Aggregation   │
            │  - Chunks              │
            │  - Entities            │
            │  - Relations           │
            └────────┬───────────────┘
                     │ 7. Structured Context
                     ↓
            ┌────────────────────────┐
            │   Prompt Construction  │
            │  - System prompt       │
            │  - Context             │
            │  - Question            │
            └────────┬───────────────┘
                     │ 8. Complete Prompt
                     ↓
            ┌────────────────────────┐
            │   LLM Generation       │
            │   (Ollama - Llama 3.1) │
            └────────┬───────────────┘
                     │ 9. Raw Answer
                     ↓
            ┌────────────────────────┐
            │  Response Processing   │
            │  - Citation extraction │
            │  - Confidence scoring  │
            │  - Formatting          │
            └────────┬───────────────┘
                     │ 10. Final Response
                     ↓
                ┌─────────┐
                │ Client  │
                └─────────┘
```

### 3.2 Detailed QA Steps

#### Step 1: Client Question

**Input**:

```json
{
  "question": "What are the key risk factors for diabetes according to recent medical research?",
  "top_k": 5,
  "domain": "healthcare"
}
```

#### Step 2: Query Embedding

**Process**:

```python
embedding_service = OllamaEmbeddingService()
query_vector = await embedding_service.embed(
    question,
    model="nomic-embed-text"
)
```

**Data Transformation**:

- Input: Question string (50-200 chars)
- Output: 768-dimensional vector

```python
query_vector = [0.123, -0.456, 0.789, ...]  # 768 dimensions
```

#### Step 3: Vector Search (Qdrant)

**Process**:

```python
qdrant_results = await qdrant_client.search(
    collection_name="chunks",
    query_vector=query_vector,
    limit=20,
    filter={
        "must": [
            {"key": "domain", "match": {"value": "healthcare"}}
        ]
    }
)
```

**Data Transformation**:

- Input: Query vector + filters
- Output: Top-20 similar chunks with scores

**Result Structure**:

```json
[
  {
    "id": "chunk_abc123",
    "score": 0.876,
    "payload": {
      "doc_id": "doc_medical_2025",
      "text": "Major risk factors for type 2 diabetes include obesity, physical inactivity, family history...",
      "entity_ids": ["entity_diabetes", "entity_obesity"]
    }
  },
  {
    "id": "chunk_def456",
    "score": 0.834,
    "payload": {
      "text": "Recent studies show that age over 45, high blood pressure...",
      "entity_ids": ["entity_diabetes", "entity_hypertension"]
    }
  }
  // ... 18 more results
]
```

#### Step 4: Graph Traversal (Neo4j)

**Process**:

```cypher
// Extract entity IDs from top vector results
WITH $entity_ids AS seeds

// Find seed entities
MATCH (seed:Entity)
WHERE seed.id IN seeds

// Traverse 2 hops (bidirectional)
MATCH path = (seed)-[:RELATES_TO*1..2]-(related:Entity)

// Get chunks mentioning related entities
MATCH (chunk:Chunk)-[:MENTIONS]->(related)

// Return chunks with relation context
RETURN DISTINCT
  chunk.id AS chunk_id,
  chunk.text AS text,
  collect(DISTINCT related.name) AS entities,
  collect(DISTINCT type(r)) AS relation_types,
  avg(r.weight) AS avg_relation_weight
ORDER BY avg_relation_weight DESC
LIMIT 20
```

**Data Transformation**:

- Input: Seed entity IDs from vector results
- Output: Related chunks with graph context

**Result Structure**:

```json
[
  {
    "chunk_id": "chunk_xyz789",
    "text": "Diabetes complications include cardiovascular disease...",
    "entities": ["Diabetes", "Cardiovascular Disease", "Hypertension"],
    "relation_types": ["CAUSES", "ASSOCIATED_WITH"],
    "avg_relation_weight": 0.82
  }
  // ... more results
]
```

#### Step 5: Reciprocal Rank Fusion (RRF)

**Algorithm**:

```python
def reciprocal_rank_fusion(
    vector_results: List[Result],
    graph_results: List[Result],
    k: int = 60
) -> List[Result]:
    """
    RRF Score = Σ (weight_i / (k + rank_i))

    where:
    - weight_i: importance weight (vector=0.6, graph=0.4)
    - k: constant (60)
    - rank_i: position in result list (1-indexed)
    """
    scores = defaultdict(float)

    # Score vector results
    for rank, result in enumerate(vector_results, 1):
        rrf_score = 0.6 / (60 + rank)
        scores[result.chunk_id] += rrf_score

    # Score graph results
    for rank, result in enumerate(graph_results, 1):
        rrf_score = 0.4 / (60 + rank)
        scores[result.chunk_id] += rrf_score

    # Sort by combined score
    sorted_ids = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    return sorted_ids[:top_k]
```

**Data Transformation**:

- Input: Two ranked lists (20 vector + 20 graph results)
- Output: Single ranked list (top-5 fused results)

**Fused Results**:

```json
[
  {
    "chunk_id": "chunk_abc123",
    "rrf_score": 0.0197, // High in both rankings
    "vector_rank": 1,
    "graph_rank": 3
  },
  {
    "chunk_id": "chunk_xyz789",
    "rrf_score": 0.0183, // High in graph, medium in vector
    "vector_rank": 5,
    "graph_rank": 1
  }
  // ... 3 more
]
```

#### Step 6: Context Aggregation

**Process**:

```python
def aggregate_context(
    chunks: List[Chunk],
    entities: List[Entity],
    relations: List[Relation]
) -> str:
    """Build structured context for LLM."""

    context_parts = []

    # Add chunks with citations
    context_parts.append("## Retrieved Information\n")
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] {chunk.text}\n"
            f"    Source: {chunk.source} (page {chunk.page})\n"
        )

    # Add entity information
    context_parts.append("\n## Key Entities\n")
    for entity in entities:
        context_parts.append(
            f"- {entity.name} ({entity.type})"
        )
        if entity.attributes:
            context_parts.append(f"  {entity.attributes}")

    # Add relationships
    context_parts.append("\n## Entity Relationships\n")
    for rel in relations:
        context_parts.append(
            f"- {rel.source_name} {rel.type} {rel.target_name}"
        )

    return "\n".join(context_parts)
```

**Output Context**:

```
## Retrieved Information

[1] Major risk factors for type 2 diabetes include obesity, physical inactivity, family history of diabetes, age over 45, and high blood pressure.
    Source: Medical Guidelines 2025.pdf (page 12)

[2] Recent epidemiological studies demonstrate that individuals with BMI over 30 have 3x higher risk of developing diabetes compared to normal weight individuals.
    Source: Diabetes Research 2025.pdf (page 5)

[3] Genetic predisposition plays a significant role, with first-degree relatives of diabetic patients having 40% higher lifetime risk.
    Source: Clinical Study.pdf (page 8)

## Key Entities
- Type 2 Diabetes (DISEASE)
- Obesity (CONDITION) - Associated with BMI > 30
- Hypertension (CONDITION) - High blood pressure
- Physical Inactivity (LIFESTYLE_FACTOR)

## Entity Relationships
- Obesity CAUSES Type 2 Diabetes
- Hypertension ASSOCIATED_WITH Type 2 Diabetes
- Family History RISK_FACTOR_FOR Type 2 Diabetes
```

#### Step 7: Prompt Construction

**Template**:

```python
PROMPT_TEMPLATE = """You are a helpful medical AI assistant. Answer the question based on the provided context.

{context}

Question: {question}

Instructions:
1. Provide a clear, evidence-based answer using ONLY the context above
2. Cite sources using [number] notation (e.g., [1], [2])
3. If the context doesn't contain sufficient information, state this clearly
4. Be factual and avoid speculation

Answer:"""
```

**Final Prompt** (2,000-4,000 tokens):

```
You are a helpful medical AI assistant. Answer the question based on the provided context.

## Retrieved Information

[1] Major risk factors for type 2 diabetes include obesity, physical inactivity...
...

Question: What are the key risk factors for diabetes according to recent medical research?

Instructions:
1. Provide a clear, evidence-based answer using ONLY the context above
2. Cite sources using [number] notation
3. If the context doesn't contain sufficient information, state this clearly
4. Be factual and avoid speculation

Answer:
```

#### Step 8: LLM Generation

**Request to Ollama**:

```python
llm_response = await ollama_client.generate(
    model="llama3.1:8b",
    prompt=final_prompt,
    options={
        "temperature": 0.3,  # Low for factual responses
        "top_p": 0.9,
        "max_tokens": 2048,
        "stop": ["Question:", "\n\n\n"]
    }
)
```

**LLM Response** (500-1500 tokens):

```
Based on the provided medical research, the key risk factors for type 2 diabetes are:

1. **Obesity**: Individuals with BMI over 30 have significantly higher risk (3x increase) [2]

2. **Physical Inactivity**: Sedentary lifestyle is a major modifiable risk factor [1]

3. **Family History**: First-degree relatives of diabetic patients have 40% higher lifetime risk [3]

4. **Age**: Risk increases substantially after age 45 [1]

5. **Hypertension**: High blood pressure is strongly associated with diabetes development [1]

These risk factors are well-documented in recent medical literature and represent both modifiable (obesity, physical activity) and non-modifiable (age, genetics) factors.
```

#### Step 9: Response Processing

**Citation Extraction**:

```python
def extract_citations(response: str, chunks: List[Chunk]) -> List[str]:
    """Extract cited sources from response."""
    citations = []
    for i in range(1, len(chunks) + 1):
        if f"[{i}]" in response:
            citations.append(chunks[i-1].source)
    return list(set(citations))  # Deduplicate
```

**Confidence Scoring**:

```python
def calculate_confidence(
    response: str,
    chunks: List[Chunk],
    avg_retrieval_score: float
) -> float:
    """Calculate answer confidence score."""

    # Factor 1: Number of citations
    num_citations = response.count("[")
    citation_score = min(num_citations / 5, 1.0)

    # Factor 2: Average retrieval score
    retrieval_score = avg_retrieval_score

    # Factor 3: Uncertainty detection
    uncertainty_phrases = [
        "i don't know", "unclear", "not enough",
        "cannot determine", "insufficient information"
    ]
    has_uncertainty = any(p in response.lower() for p in uncertainty_phrases)
    uncertainty_penalty = 0.5 if has_uncertainty else 1.0

    # Combined confidence
    confidence = (
        citation_score * 0.3 +
        retrieval_score * 0.7
    ) * uncertainty_penalty

    return round(confidence, 2)
```

#### Step 10: Final Response

**Output**:

```json
{
  "answer": "Based on the provided medical research, the key risk factors for type 2 diabetes are:\n\n1. **Obesity**: Individuals with BMI over 30 have significantly higher risk (3x increase) [2]\n\n2. **Physical Inactivity**: Sedentary lifestyle is a major modifiable risk factor [1]\n\n3. **Family History**: First-degree relatives of diabetic patients have 40% higher lifetime risk [3]\n\n4. **Age**: Risk increases substantially after age 45 [1]\n\n5. **Hypertension**: High blood pressure is strongly associated with diabetes development [1]",
  "citations": [
    "Medical Guidelines 2025.pdf",
    "Diabetes Research 2025.pdf",
    "Clinical Study.pdf"
  ],
  "confidence": 0.87,
  "retrieved_chunks": 5,
  "processing_time_ms": 1523.45,
  "metadata": {
    "vector_search_time_ms": 45.2,
    "graph_traversal_time_ms": 123.8,
    "llm_inference_time_ms": 1245.6,
    "entities_used": 12,
    "relations_used": 8
  }
}
```

---

## 4. Authentication Flow

### 4.1 API Key Creation Flow

```
┌─────────┐
│  Admin  │
└────┬────┘
     │ 1. Request: Create API Key
     │    POST /api/v1/admin/keys/create
     │    {"name": "production-app", "rate_limit": 1000}
     ↓
┌──────────────────────────────────┐
│   Admin Authentication Check     │
│   (Must use master API key)      │
└────┬─────────────────────────────┘
     │ 2. Verified Admin
     ↓
┌──────────────────────────────────┐
│   Generate Random Key            │
│   - secrets.token_urlsafe(32)    │
│   - Result: 43-char string       │
└────┬─────────────────────────────┘
     │ 3. Plain Key: "xK9mP2qR..."
     ↓
┌──────────────────────────────────┐
│   Hash Key (SHA-256)             │
│   - key_hash = sha256(key)       │
└────┬─────────────────────────────┘
     │ 4. Hash: "a3f5b8..."
     ↓
┌──────────────────────────────────┐
│   Create APIKey Object           │
│   - key_hash: hashed value       │
│   - name: "production-app"       │
│   - rate_limit: 1000/min         │
│   - daily_limit: 10000/day       │
│   - created_at: timestamp        │
│   - is_active: true              │
└────┬─────────────────────────────┘
     │ 5. Store in Memory/DB
     ↓
┌──────────────────────────────────┐
│   Return Plain Key (Only Once!)  │
└────┬─────────────────────────────┘
     │ 6. Response with plain key
     ↓
┌─────────┐
│  Admin  │
└─────────┘
```

**Response**:

```json
{
  "api_key": "xK9mP2qR7vN3hL6fJ8dT1wC4yS5bX0zA9mQ2eR",
  "name": "production-app",
  "rate_limit": 1000,
  "daily_limit": 10000,
  "created_at": "2026-01-22T10:30:00Z",
  "warning": "Store this key securely - it cannot be retrieved again!"
}
```

### 4.2 Request Authentication Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. API Request
     │    X-API-Key: xK9mP2qR7vN3hL6...
     ↓
┌────────────────────────────────────┐
│   Extract API Key from Header      │
└────┬───────────────────────────────┘
     │ 2. Plain key
     ↓
┌────────────────────────────────────┐
│   Hash Incoming Key (SHA-256)      │
└────┬───────────────────────────────┘
     │ 3. key_hash = "a3f5b8..."
     ↓
┌────────────────────────────────────┐
│   Lookup Key in Storage            │
│   api_key = keys_db[key_hash]      │
└────┬───────────────────────────────┘
     │ 4. APIKey object (or None)
     ↓
     ├─── Not Found ───> 401 Unauthorized
     │
     ├─── is_active = false ───> 401 Key Revoked
     │
     └─── Valid Key
          │
          ↓
     ┌────────────────────────────────────┐
     │   Check Rate Limits                │
     │   - Per-minute counter             │
     │   - Daily counter                  │
     └────┬───────────────────────────────┘
          │
          ├─── Exceeded ───> 429 Rate Limit
          │
          └─── Within Limit
               │
               ↓
          ┌────────────────────────────────┐
          │   Update Counters              │
          │   - minute_count += 1          │
          │   - daily_count += 1           │
          │   - last_used = now()          │
          └────┬───────────────────────────┘
               │ 5. Authorized!
               ↓
          ┌────────────────────────────────┐
          │   Process Request              │
          └────────────────────────────────┘
```

---

## 5. Monitoring Data Flow

### 5.1 Metrics Collection Flow

```
┌──────────────┐
│API Request   │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Request Timing Middleware           │
│  - Capture start_time                │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Process Request                     │
│  (Business Logic)                    │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Capture Response                    │
│  - status_code                       │
│  - end_time                          │
│  - duration = end - start            │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Metrics Collector                   │
│  .track_request(                     │
│    method="POST",                    │
│    path="/api/v1/ask",               │
│    status_code=200,                  │
│    duration_ms=1523.45               │
│  )                                   │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Update Metrics                      │
│  - request_count["POST /ask"] += 1   │
│  - latencies["POST /ask"].append(..  │
│  - status_codes[200] += 1            │
└──────┬───────────────────────────────┘
       │
       ↓ (periodically or on /metrics request)
┌──────────────────────────────────────┐
│  Calculate Aggregations              │
│  - p50 = percentile(latencies, 50)   │
│  - p95 = percentile(latencies, 95)   │
│  - p99 = percentile(latencies, 99)   │
│  - error_rate = errors / total       │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Expose Metrics (Prometheus Format) │
│  GET /metrics                        │
└──────────────────────────────────────┘
```

**Metrics Output**:

```prometheus
# Request counts
http_requests_total{method="POST",path="/api/v1/ask"} 1523

# Latency percentiles
http_request_duration_seconds{method="POST",path="/api/v1/ask",quantile="0.5"} 1.234
http_request_duration_seconds{method="POST",path="/api/v1/ask",quantile="0.95"} 2.456
http_request_duration_seconds{method="POST",path="/api/v1/ask",quantile="0.99"} 3.789

# Status codes
http_responses_total{status="200"} 1450
http_responses_total{status="401"} 23
http_responses_total{status="429"} 15
http_responses_total{status="500"} 12

# Component metrics
component_calls_total{component="query",operation="embed"} 1523
component_duration_seconds{component="query",operation="embed",quantile="0.95"} 0.045
```

---

## 6. Backup and Restore Flow

### 6.1 Backup Flow

```
┌──────────────┐
│ Cron Job /   │
│ Manual Trigger│
└──────┬───────┘
       │ 1. python scripts/backup.py
       ↓
┌────────────────────────────────────────┐
│  Create Backup Directory               │
│  ./backups/backup-20260122-020000/     │
└──────┬─────────────────────────────────┘
       │
       ├────────────────┬────────────────┐
       │                │                │
       ↓                ↓                ↓
┌───────────┐    ┌──────────┐    ┌──────────────┐
│  Neo4j    │    │ Qdrant   │    │  Metadata    │
│  Export   │    │ Snapshot │    │  Generation  │
└─────┬─────┘    └────┬─────┘    └──────┬───────┘
      │               │                   │
      ↓               ↓                   ↓

┌────────────────────────────────────────┐
│  Neo4j Export                          │
│  MATCH (d:Document)                    │
│  RETURN d                              │
│  → documents.json                      │
│                                        │
│  MATCH (e:Entity)                      │
│  RETURN e                              │
│  → entities.json                       │
│                                        │
│  MATCH ()-[r:RELATES_TO]->()           │
│  RETURN r                              │
│  → relations.json                      │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Qdrant Export                         │
│  GET /collections/chunks/points        │
│  → chunks_vectors.json                 │
│                                        │
│  GET /collections/entities/points      │
│  → entity_vectors.json                 │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Create Metadata                       │
│  {                                     │
│    "timestamp": "2026-01-22...",       │
│    "version": "1.0.0",                 │
│    "documents_count": 150,             │
│    "entities_count": 3450,             │
│    "relations_count": 1200,            │
│    "chunks_count": 2300                │
│  }                                     │
│  → metadata.json                       │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Compress (Optional)                   │
│  tar -czf backup.tar.gz ./backup/      │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Upload to S3 (Optional)               │
│  aws s3 cp backup.tar.gz s3://...      │
└────────────────────────────────────────┘
```

**Backup Directory Structure**:

```
backups/backup-20260122-020000/
├── metadata.json           # Backup metadata
├── documents.json          # Document nodes
├── entities.json           # Entity nodes
├── relations.json          # Relation edges
├── chunks.json             # Chunk nodes
├── chunks_vectors.json     # Vector embeddings
└── entity_vectors.json     # Entity embeddings
```

### 6.2 Restore Flow

```
┌──────────────┐
│  Admin       │
└──────┬───────┘
       │ 1. python scripts/restore.py --input ./backups/...
       ↓
┌────────────────────────────────────────┐
│  Validate Backup                       │
│  - Check metadata.json exists          │
│  - Verify file integrity               │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Confirmation Prompt                   │
│  "This will overwrite existing data.   │
│   Continue? [y/N]"                     │
└──────┬─────────────────────────────────┘
       │ 2. User confirms
       ↓
┌────────────────────────────────────────┐
│  Clear Existing Data (Optional)        │
│  - DROP Neo4j database                 │
│  - DELETE Qdrant collection            │
└──────┬─────────────────────────────────┘
       │
       ├────────────────┬────────────────┐
       │                │                │
       ↓                ↓                ↓
┌───────────┐    ┌──────────┐    ┌──────────────┐
│  Neo4j    │    │ Qdrant   │    │  Verification│
│  Restore  │    │ Restore  │    │              │
└─────┬─────┘    └────┬─────┘    └──────┬───────┘
      │               │                   │

┌────────────────────────────────────────┐
│  Neo4j Restore                         │
│  FOR EACH document IN documents.json:  │
│    CREATE (d:Document {...})           │
│                                        │
│  FOR EACH entity IN entities.json:     │
│    CREATE (e:Entity {...})             │
│                                        │
│  FOR EACH relation IN relations.json:  │
│    CREATE ()-[r:RELATES_TO {...}]->()  │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Qdrant Restore                        │
│  POST /collections/chunks/points       │
│  {points: [...from chunks_vectors]}    │
│                                        │
│  POST /collections/entities/points     │
│  {points: [...from entity_vectors]}    │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Verification                          │
│  - Count documents: 150 ✓              │
│  - Count entities: 3450 ✓              │
│  - Count vectors: 2300 ✓               │
│  - Health checks pass ✓                │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Success Report                        │
│  "Restore completed successfully!"     │
│  "Restored 150 documents, 3450         │
│   entities, 1200 relations"            │
└────────────────────────────────────────┘
```

---

## 7. Data Management Flow

### 7.1 Document Deletion Flow

```
┌──────────────┐
│  Client      │
└──────┬───────┘
       │ DELETE /api/v1/data/documents/{doc_id}
       │ ?delete_chunks=true
       │ &delete_entities=false
       │ &delete_vectors=true
       ↓
┌────────────────────────────────────────┐
│  Authentication                        │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Fetch Document Metadata               │
│  MATCH (d:Document {id: $doc_id})      │
│  RETURN d                              │
└──────┬─────────────────────────────────┘
       │
       ├─── Not Found ───> 404 Error
       │
       └─── Found
            │
            ↓
┌────────────────────────────────────────┐
│  Get Related Data IDs                  │
│  - Chunk IDs: [chunk_1, chunk_2, ...]  │
│  - Entity IDs: [ent_1, ent_2, ...]     │
│  - Vector IDs: [vec_1, vec_2, ...]     │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Cascade Deletion (Parallel)           │
│                                        │
│  IF delete_chunks:                     │
│    MATCH (d:Document)-[:HAS_CHUNK]->(c)│
│    DELETE c                            │
│                                        │
│  IF delete_vectors:                    │
│    DELETE FROM qdrant                  │
│      WHERE id IN chunk_ids             │
│                                        │
│  IF delete_entities:                   │
│    MATCH (e:Entity)<-[:MENTIONS]-(c)   │
│    WHERE c IN chunks                   │
│    DELETE e (if no other refs)         │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Delete Document Node                  │
│  MATCH (d:Document {id: $doc_id})      │
│  DELETE d                              │
└──────┬─────────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────┐
│  Response                              │
│  {                                     │
│    "status": "success",                │
│    "document_id": "doc_123",           │
│    "chunks_deleted": 45,               │
│    "entities_deleted": 0,              │
│    "vectors_deleted": 45               │
│  }                                     │
└────────────────────────────────────────┘
```

---

## 8. System Initialization Flow

### 8.1 Application Startup Flow

```
┌───────────────┐
│ Docker Compose│
│ up -d         │
└───────┬───────┘
        │
        ├────────────┬────────────┬────────────┐
        │            │            │            │
        ↓            ↓            ↓            ↓
   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
   │ Neo4j  │  │Qdrant  │  │ Ollama │  │  API   │
   └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
       │           │           │           │
       ↓           ↓           ↓           ↓
   Initialize  Initialize  Pull Model   Start
   Database    Collections              FastAPI

┌─────────────────────────────────────────────────────┐
│  API Application Startup (src/main.py)              │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  1. Load Configuration                              │
│     - Read .env file                                │
│     - Initialize Settings object                    │
│     - Validate required vars                        │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  2. Initialize Dependency Container                 │
│     - Neo4j client                                  │
│     - Qdrant client                                 │
│     - Ollama client                                 │
│     - Embedding service                             │
│     - ETL pipeline components                       │
│     - Retriever instances                           │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  3. Connect to Databases                            │
│     - Neo4j: Test connection, create indexes        │
│     - Qdrant: Create collections if not exist       │
│     - Ollama: Verify model availability             │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  4. Initialize Authentication                       │
│     - APIKeyManager singleton                       │
│     - Create default master key                     │
│     - Load existing keys (if persistent)            │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  5. Setup Monitoring                                │
│     - MetricsCollector singleton                    │
│     - RequestTimingMiddleware                       │
│     - Health check endpoints                        │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  6. Register Routes                                 │
│     - /api/v1/ask                                   │
│     - /api/v1/ingest                                │
│     - /api/v1/admin/*                               │
│     - /api/v1/data/*                                │
│     - /api/v1/health                                │
│     - /metrics                                      │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  7. Start Uvicorn Server                            │
│     - Host: 0.0.0.0                                 │
│     - Port: 8000                                    │
│     - Workers: 4 (configurable)                     │
│     - Log level: INFO                               │
└──────┬──────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────┐
│  System Ready ✓                                     │
│  - API: http://localhost:8000                       │
│  - Docs: http://localhost:8000/docs                 │
│  - Health: http://localhost:8000/api/v1/health      │
└─────────────────────────────────────────────────────┘
```

---

## 9. Summary: Key Data Transformations

| Process                 | Input             | Transformation     | Output               |
| ----------------------- | ----------------- | ------------------ | -------------------- |
| **Document Loading**    | PDF binary (5MB)  | Extract + OCR      | Text string (50KB)   |
| **Chunking**            | Text (50KB)       | Recursive split    | 45 chunks (1KB each) |
| **Entity Extraction**   | Text chunks       | NER + patterns     | 123 entities         |
| **Relation Extraction** | Text + entities   | Dependency parsing | 89 relations         |
| **Embedding**           | Text chunks       | Neural encoding    | 768-dim vectors      |
| **Vector Search**       | Query vector      | Cosine similarity  | Top-20 results       |
| **Graph Traversal**     | Entity IDs        | BFS expansion      | Related entities     |
| **RRF Fusion**          | 2 ranked lists    | Score combination  | Top-5 results        |
| **Context Aggregation** | Chunks + entities | Template filling   | Structured context   |
| **LLM Generation**      | Prompt (3KB)      | Inference          | Answer (1KB)         |

---

**Document Control**

| Version | Date       | Author   | Changes                        |
| ------- | ---------- | -------- | ------------------------------ |
| 1.0.0   | 2026-01-22 | Dev Team | Initial dataflow documentation |
