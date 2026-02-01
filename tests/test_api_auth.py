"""
Tests for API authentication and authorization.
"""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.auth import APIKeyManager


class TestAPIKeyManager:
    """Tests for API key management."""

    def test_create_key(self):
        """Test creating a new API key."""
        manager = APIKeyManager()
        key = manager.create_key(name="test_key", rate_limit=100, daily_limit=1000)

        assert key.startswith("graphrag_")
        assert len(key) > 20

    def test_validate_key(self):
        """Test validating an API key."""
        manager = APIKeyManager()
        key = manager.create_key(name="test_key")

        is_valid, api_key_obj = manager.validate_key(key)

        assert is_valid
        assert api_key_obj is not None
        assert api_key_obj.name == "test_key"
        assert api_key_obj.is_active

    def test_validate_invalid_key(self):
        """Test validating an invalid API key."""
        manager = APIKeyManager()

        is_valid, api_key_obj = manager.validate_key("invalid_key")

        assert not is_valid
        assert api_key_obj is None

    def test_revoke_key(self):
        """Test revoking an API key."""
        manager = APIKeyManager()
        key = manager.create_key(name="test_key")

        # Revoke the key
        success = manager.revoke_key(key)
        assert success

        # Validate revoked key
        is_valid, api_key_obj = manager.validate_key(key)
        assert not is_valid

    def test_rate_limiting(self):
        """Test rate limiting."""
        manager = APIKeyManager()
        key = manager.create_key(name="test_key", rate_limit=2, daily_limit=100)

        is_valid, api_key_obj = manager.validate_key(key)
        assert is_valid

        key_hash = manager._hash_key(key)

        # First request
        is_allowed, error = manager.check_rate_limit(key_hash, api_key_obj)
        assert is_allowed

        # Second request
        is_allowed, error = manager.check_rate_limit(key_hash, api_key_obj)
        assert is_allowed

        # Third request (should be rate limited)
        is_allowed, error = manager.check_rate_limit(key_hash, api_key_obj)
        assert not is_allowed
        assert "Rate limit exceeded" in error


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


class TestAPIAuthentication:
    """Tests for API authentication endpoints."""

    def test_health_check_without_auth(self, client_with_mock_container):
        """Health check should work without authentication."""
        response = client_with_mock_container.get("/api/v1/health")
        # Health endpoint should not require auth (no 401)
        assert response.status_code != status.HTTP_401_UNAUTHORIZED
        assert response.status_code == status.HTTP_200_OK

    def test_protected_endpoint_without_auth(self, client_with_mock_container):
        """Protected endpoints should require authentication."""
        response = client_with_mock_container.post(
            "/api/v1/ask",
            json={"question": "test question"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_protected_endpoint_with_auth(self, client_with_mock_container, api_key):
        """Protected endpoints should work with valid API key.

        Note: This will get past auth but may fail due to missing services.
        The important thing is it's not 401 (authentication passed).
        """
        response = client_with_mock_container.post(
            "/api/v1/ask",
            json={"question": "test question"},
            headers={"X-API-Key": api_key},
        )
        # Could be 200, 422, or 500 depending on validation/services
        # The important thing is it's not 401 (authentication passed)
        assert response.status_code != status.HTTP_401_UNAUTHORIZED

    def test_invalid_api_key(self, client_with_mock_container):
        """Test with invalid API key."""
        response = client_with_mock_container.post(
            "/api/v1/ask",
            json={"question": "test question"},
            headers={"X-API-Key": "invalid_key"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
