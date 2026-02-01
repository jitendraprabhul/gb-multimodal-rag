"""
Pytest configuration and shared fixtures.
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings, get_settings
from src.main import create_app
from src.api.dependencies import ServiceContainer


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        app_env="development",
        debug=True,
        log_level="DEBUG",
        domain="finance",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test_password",
        qdrant_host="localhost",
        qdrant_port=6333,
        ollama_host="http://mock-ollama:11434",
        text_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
    )


@pytest.fixture
def app(test_settings):
    """Create test FastAPI app."""
    # Override settings
    import config.settings as settings_module

    original_get_settings = settings_module.get_settings
    settings_module.get_settings = lambda: test_settings

    app = create_app()

    yield app

    # Restore original settings
    settings_module.get_settings = original_get_settings


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def container(test_settings) -> AsyncGenerator[ServiceContainer, None]:
    """Create and initialize service container for tests."""
    container = ServiceContainer(test_settings)
    await container.initialize()

    yield container

    await container.cleanup()


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a sample PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"

    # Create a simple PDF using reportlab if available
    try:
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "Test Document")
        c.drawString(100, 730, "This is a test document for GraphRAG.")
        c.save()
    except ImportError:
        # If reportlab not available, create a dummy file
        pdf_path.write_text("Dummy PDF content")

    return pdf_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    text_path = tmp_path / "test.txt"
    text_path.write_text(
        """
        GraphRAG Test Document

        This is a test document containing information about financial entities.

        Company ABC reported revenue of $100M in Q4 2023.
        The CEO, John Smith, announced plans for expansion.

        Market analysts predict growth of 25% next year.
        """
    )
    return text_path


@pytest.fixture
def api_key() -> str:
    """Return the default API key for testing."""
    return "graphrag_default_key_CHANGE_IN_PRODUCTION"
