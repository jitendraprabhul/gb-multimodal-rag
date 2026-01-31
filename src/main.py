"""
Main entry point for GraphRAG Multimodal API server.

Starts the FastAPI application with all routes and middleware.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from src import __version__
from src.api.dependencies import get_container, _container
from src.api.routes import router
from src.api.admin_routes import router as admin_router
from src.api.data_management_routes import router as data_router
from src.api.monitoring import RequestTimingMiddleware, get_metrics_collector
from src.core.logging import setup_logging, get_logger
from src.ui.router import router as ui_router


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting GraphRAG API server...")

    try:
        # Initialize service container
        container = await get_container()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GraphRAG API server...")

    try:
        global _container
        if _container:
            await _container.cleanup()
            _container = None
        logger.info("All services cleaned up")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    settings = get_settings()

    # Setup logging
    setup_logging(
        level=settings.log_level,
        json_format=settings.is_production,
    )

    app = FastAPI(
        title="GraphRAG Multimodal API",
        description="""
GraphRAG-style multimodal RAG system combining:
- Knowledge graph over entities and relations
- Vector index over text, table, and image embeddings
- Hybrid retriever (graph traversal + dense retrieval)
- LLM reasoning layer with explainability

**FOSS & Local-First**: All models and databases run locally.
        """,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, configure specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Monitoring middleware
    app.add_middleware(RequestTimingMiddleware)

    # Include routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1")
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(ui_router)

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict:
        return {
            "name": "GraphRAG Multimodal API",
            "version": __version__,
            "ui": "/ui",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    # Metrics endpoint
    @app.get("/metrics", tags=["System"])
    async def metrics() -> dict:
        """Get application metrics (Prometheus-style)."""
        collector = get_metrics_collector()
        return collector.get_metrics()

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": str(exc) if settings.debug else "An error occurred",
            },
        )

    return app


app = create_app()


def main() -> None:
    """Run the API server."""
    settings = get_settings()

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1 if settings.debug else settings.api_workers,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
