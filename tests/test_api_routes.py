"""
Tests for API routes.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient


@pytest.fixture
def mock_container():
    """Create a mock service container."""
    from src.api.dependencies import ServiceContainer

    mock = MagicMock(spec=ServiceContainer)
    mock.ollama = MagicMock(_initialized=False)
    mock.qdrant = MagicMock(_initialized=False)
    mock.neo4j = MagicMock(_initialized=False)
    mock.embeddings = MagicMock(_initialized=False)
    return mock


@pytest.fixture
def client_with_mock_container(app, mock_container):
    """Create test client with mocked container."""
    from src.api.dependencies import get_container

    async def mock_get_container():
        return mock_container

    app.dependency_overrides[get_container] = mock_get_container

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_container, None)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client_with_mock_container):
        """Test health check endpoint."""
        response = client_with_mock_container.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["name"] == "GraphRAG Multimodal API"
        assert "version" in data
        assert "docs" in data


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics(self, client_with_mock_container):
        """Test metrics endpoint."""
        response = client_with_mock_container.get("/metrics")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "uptime_seconds" in data
        assert "requests" in data
        assert "components" in data


class TestAdminEndpoints:
    """Tests for admin endpoints."""

    def test_create_api_key(self, client, api_key):
        """Test creating an API key."""
        response = client.post(
            "/api/v1/admin/keys/create",
            json={
                "name": "test_key",
                "rate_limit": 100,
                "daily_limit": 1000,
            },
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "api_key" in data
        assert data["name"] == "test_key"
        assert data["rate_limit"] == 100

    def test_list_api_keys(self, client, api_key):
        """Test listing API keys."""
        response = client.get(
            "/api/v1/admin/keys/list",
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "keys" in data
        assert isinstance(data["keys"], list)
