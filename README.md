# GraphRAG Multimodal RAG System

A production-grade, local-first, multimodal Retrieval-Augmented Generation (RAG) system that combines knowledge graphs with vector search for enhanced question answering over diverse document types.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

### Core Capabilities

- **Multimodal Document Processing**: PDF, text files, spreadsheets (CSV/XLSX), and images
- **Hybrid Retrieval**: Combines vector similarity search with knowledge graph traversal
- **Domain-Specific**: Pre-configured for Finance and Healthcare domains
- **Knowledge Graph**: Automatic entity and relation extraction with Neo4j
- **Vector Database**: High-performance semantic search with Qdrant
- **Local LLM Integration**: Privacy-first inference with Ollama
- **Production-Ready**: Docker-based deployment with health checks and monitoring

### Advanced Features

- Named Entity Recognition (NER) with domain-specific models
- Relation extraction (pattern-based and LLM-based)
- Graph-aware re-ranking for contextual retrieval
- Table and image extraction from PDFs
- OCR support for scanned documents
- Semantic chunking strategies
- RESTful API with FastAPI
- Async-first architecture for high performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────┬──────────────┬──────────────┐
             │             │              │              │
             v             v              v              v
      ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
      │   ETL    │  │ Knowledge│  │  Vector  │  │   LLM    │
      │ Pipeline │  │  Graph   │  │ Database │  │ Reasoner │
      └──────────┘  └──────────┘  └──────────┘  └──────────┘
             │             │              │              │
             v             v              v              v
      ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
      │ PDFPlumb │  │  Neo4j   │  │  Qdrant  │  │  Ollama  │
      │  PyMuPDF │  │  +APOC   │  │  Vector  │  │   LLM    │
      │ PaddleOCR│  │          │  │    DB    │  │  Server  │
      └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### Component Overview

| Component           | Technology                     | Purpose                                |
| ------------------- | ------------------------------ | -------------------------------------- |
| **API Layer**       | FastAPI, Pydantic              | RESTful endpoints with validation      |
| **ETL Pipeline**    | pdfplumber, PyMuPDF, PaddleOCR | Document processing and chunking       |
| **Knowledge Graph** | Neo4j, spaCy, Transformers     | Entity/relation extraction and storage |
| **Vector DB**       | Qdrant, sentence-transformers  | Semantic search and embeddings         |
| **LLM Integration** | Ollama, Mistral/Llama          | Question answering and reasoning       |
| **Orchestration**   | Docker Compose                 | Multi-service deployment               |

## Quick Start

### Prerequisites

- **Docker** and **Docker Compose** (recommended)
- OR **Python 3.10+** for local development
- **GPU** (optional, for faster embedding and LLM inference)

### Option 1: Docker Deployment (Recommended)

1. Clone the repository:

```bash
git clone <repository-url>
cd gb-multimodal-rag
```

2. Create configuration:

```bash
cp .env.example .env
# Edit .env to customize settings
```

3. Start the stack:

```bash
# Production
docker-compose up -d

# Development (with hot reload)
docker-compose -f docker-compose.dev.yml up -d
```

4. Wait for services to initialize (check logs):

```bash
docker-compose logs -f
```

5. Access the API:

```bash
# API documentation
http://localhost:8000/docs

# Health check
curl http://localhost:8000/api/v1/health
```

### Option 2: Local Development

1. Install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Install spaCy models:

```bash
python -m spacy download en_core_web_trf
```

3. Start required services (Neo4j, Qdrant, Ollama):

```bash
docker-compose up neo4j qdrant ollama -d
```

4. Configure environment:

```bash
cp .env.example .env
# Edit .env with your settings
```

5. Run the API server:

```bash
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Document Ingestion

#### Via API

```bash
# Single file
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@document.pdf"

# Batch ingestion
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.txt" \
  -F "files=@data.csv"
```

#### Via CLI

```bash
# Ingest a single file
python scripts/ingest.py /path/to/document.pdf

# Ingest a directory
python scripts/ingest.py /path/to/documents/ --patterns "*.pdf" "*.txt"

# Recursive directory ingestion
python scripts/ingest.py /path/to/documents/ --recursive
```

### Question Answering

```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings in the financial report?",
    "top_k": 5,
    "use_graph": true
  }'
```

Response:

```json
{
  "question": "What are the key findings in the financial report?",
  "answer": "The financial report highlights...",
  "sources": [
    {
      "content": "Revenue increased by 25%...",
      "doc_id": "doc_123",
      "chunk_id": "chunk_456",
      "score": 0.89
    }
  ],
  "entities": ["Q4 Revenue", "Net Income", "EBITDA"],
  "processing_time_ms": 1250
}
```

### Entity and Graph Operations

```bash
# Search entities
curl "http://localhost:8000/api/v1/entities/search?query=revenue&limit=10"

# Get entity context
curl "http://localhost:8000/api/v1/entities/revenue-q4/context"

# Find graph paths
curl -X POST "http://localhost:8000/api/v1/graph/path" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Company A",
    "target": "Product B",
    "max_depth": 3
  }'
```

## Configuration

### Environment Variables

Copy [.env.example](.env.example) to `.env` and configure:

#### Domain Selection

```bash
DOMAIN=finance  # or "healthcare"
```

#### Database Connections

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

QDRANT_HOST=localhost
QDRANT_PORT=6333
```

#### Model Configuration

```bash
# LLM
OLLAMA_MODEL=mistral:7b-instruct

# Embeddings
TEXT_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
EMBEDDING_DEVICE=cuda  # or "cpu"

# NER
NER_MODEL_FINANCE=dslim/bert-base-NER
NER_MODEL_HEALTHCARE=en_ner_bc5cdr_md
```

#### Performance Tuning

```bash
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_VECTOR=20
TOP_K_FINAL=5
MAX_CONCURRENT_DOCS=4
EMBEDDING_BATCH_SIZE=32
```

### Domain Switching

The system supports two preconfigured domains:

**Finance Domain**:

- Entity types: PERSON, ORG, MONEY, PERCENT, DATE, PRODUCT
- Relations: FILED, MENTIONS_METRIC, PART_OF, ISSUED, etc.
- NER model: dslim/bert-base-NER

**Healthcare Domain**:

- Entity types: DISEASE, CHEMICAL, GENE, SPECIES, CELL_TYPE
- Relations: HAS_CONDITION, TREATED_WITH, CAUSES, AFFECTS
- NER model: en_ner_bc5cdr_md (SciSpaCy)

## API Reference

### Endpoints

| Endpoint                             | Method | Description                        |
| ------------------------------------ | ------ | ---------------------------------- |
| `/api/v1/health`                     | GET    | Health check and service status    |
| `/api/v1/ask`                        | POST   | Question answering with context    |
| `/api/v1/ingest`                     | POST   | Upload and process single document |
| `/api/v1/ingest/batch`               | POST   | Batch document upload              |
| `/api/v1/entities/search`            | GET    | Search entities by name            |
| `/api/v1/entities/{name}`            | GET    | Get entity details                 |
| `/api/v1/entities/{name}/context`    | GET    | Get entity with graph context      |
| `/api/v1/graph/subgraph`             | POST   | Extract subgraph around entities   |
| `/api/v1/graph/path`                 | POST   | Find paths between entities        |
| `/api/v1/stats`                      | GET    | System statistics                  |
| `/api/v1/answer/{answer_id}/explain` | GET    | Get answer explanation             |

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development

### Project Structure

```
gb-multimodal-rag/
├── config/                 # Configuration management
│   └── settings.py        # Pydantic settings
├── src/
│   ├── api/               # FastAPI routes and schemas
│   │   ├── routes.py      # API endpoints
│   │   ├── schemas.py     # Request/response models
│   │   └── dependencies.py # Dependency injection
│   ├── core/              # Core utilities
│   │   ├── types.py       # Type definitions
│   │   ├── exceptions.py  # Custom exceptions
│   │   └── logging.py     # Logging setup
│   ├── etl/               # Document processing
│   │   ├── pipeline.py    # Main ETL orchestrator
│   │   ├── processors/    # Format-specific processors
│   │   └── chunking.py    # Text chunking strategies
│   ├── kg/                # Knowledge graph
│   │   ├── ner_extractor.py      # Entity extraction
│   │   ├── relation_extractor.py # Relation extraction
│   │   └── neo4j_client.py       # Neo4j operations
│   ├── vector/            # Vector database
│   │   ├── embeddings.py  # Embedding generation
│   │   └── qdrant_client.py # Qdrant operations
│   ├── retrieval/         # Hybrid retrieval
│   │   ├── hybrid_retriever.py # Main retriever
│   │   ├── graph_traversal.py  # Graph algorithms
│   │   └── reranker.py         # Re-ranking strategies
│   ├── llm/               # LLM integration
│   │   ├── ollama_client.py # Ollama API client
│   │   └── reasoning.py     # GraphRAG reasoning
│   └── services/          # High-level services
│       ├── ingestion_service.py # Document ingestion
│       └── query_service.py     # Query orchestration
├── scripts/               # Utility scripts
│   ├── ingest.py         # CLI for ingestion
│   └── setup_models.py   # Model download helper
├── docker-compose.yml     # Production stack
├── docker-compose.dev.yml # Development stack
├── Dockerfile             # Container image
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project metadata
└── .env.example          # Configuration template
```

### Code Quality

Run linting and formatting:

```bash
# Format code
black src/ scripts/

# Lint
ruff check src/ scripts/

# Type checking
mypy src/ scripts/
```

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_etl_pipeline.py
```

## Deployment

### Production Checklist

- [ ] Update `.env` with production credentials
- [ ] Change default passwords (Neo4j, etc.)
- [ ] Configure proper CORS origins
- [ ] Set `DEBUG=false`
- [ ] Configure log aggregation
- [ ] Set up backup strategy for Neo4j and Qdrant
- [ ] Configure reverse proxy (Nginx/Traefik)
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and alerting
- [ ] Configure resource limits in docker-compose.yml

### Docker Compose Production

```bash
# Start with specific env file
docker-compose --env-file .env.production up -d

# Scale API workers
docker-compose up -d --scale api=4

# View logs
docker-compose logs -f api

# Restart services
docker-compose restart api
```

### Resource Requirements

**Minimum**:

- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB (+ document storage)

**Recommended**:

- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA with 8GB+ VRAM (optional, for faster inference)
- Disk: 100GB SSD (for embeddings cache and databases)

### Monitoring

Health check endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

Response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "neo4j": "connected",
    "qdrant": "connected",
    "ollama": "connected"
  }
}
```

## Troubleshooting

### Common Issues

**Ollama model not found**:

```bash
docker exec -it ollama ollama pull mistral:7b-instruct
```

**Neo4j authentication failed**:

- Check `NEO4J_PASSWORD` in `.env`
- Reset: `docker-compose down -v && docker-compose up -d`

**Out of memory during ingestion**:

- Reduce `MAX_CONCURRENT_DOCS` in `.env`
- Reduce `EMBEDDING_BATCH_SIZE`
- Increase Docker memory limit

**Slow embedding generation**:

- Set `EMBEDDING_DEVICE=cuda` if GPU available
- Use smaller model: `TEXT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

### Logs

```bash
# API logs
docker-compose logs -f api

# Neo4j logs
docker-compose logs -f neo4j

# All services
docker-compose logs -f
```

## Performance Optimization

### GPU Acceleration

1. Install NVIDIA Container Toolkit
2. Uncomment GPU sections in docker-compose.yml
3. Set in `.env`:

```bash
EMBEDDING_DEVICE=cuda
OCR_USE_GPU=true
```

### Caching

The system automatically caches:

- Embedding models (in `./models`)
- Document embeddings (in Qdrant)
- LLM models (in Ollama volume)

### Scaling

For high-throughput scenarios:

- Increase `API_WORKERS` in `.env`
- Scale API service: `docker-compose up -d --scale api=8`
- Add load balancer (Nginx/Traefik)
- Consider distributed Qdrant deployment

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Run code quality checks: `black . && ruff check . && mypy .`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Neo4j for graph database technology
- Qdrant for vector search
- Ollama for local LLM serving
- FastAPI for the web framework
- All open-source contributors

## Support

- Documentation: http://localhost:8000/docs
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Roadmap

- [ ] Authentication and authorization
- [ ] Multi-tenant support
- [ ] Advanced caching layer (Redis)
- [ ] Streaming responses for large answers
- [ ] Cross-encoder re-ranking
- [ ] LLM-based relation extraction
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics export
- [ ] GraphQL API
- [ ] Web UI for document management

---

**Version**: 1.0.0
**Status**: Production Ready
**Last Updated**: 2026-01-22
