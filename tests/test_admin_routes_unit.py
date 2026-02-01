"""
Tests for admin routes.

Covers:
- API key creation
- API key listing
- API key revocation
- Authentication requirements
- Error handling
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


# =============================================================================
# Admin Route Schema Tests
# =============================================================================


class TestAdminSchemas:
    def test_create_api_key_request(self):
        from src.api.admin_routes import CreateAPIKeyRequest

        req = CreateAPIKeyRequest(name="test_key")
        assert req.name == "test_key"
        assert req.rate_limit == 100
        assert req.daily_limit == 10000

    def test_create_api_key_request_custom(self):
        from src.api.admin_routes import CreateAPIKeyRequest

        req = CreateAPIKeyRequest(name="custom", rate_limit=50, daily_limit=500)
        assert req.rate_limit == 50
        assert req.daily_limit == 500

    def test_create_api_key_response(self):
        from src.api.admin_routes import CreateAPIKeyResponse

        resp = CreateAPIKeyResponse(
            api_key="graphrag_test_key",
            name="test",
            rate_limit=100,
            daily_limit=10000,
            created_at=datetime.utcnow(),
        )
        assert resp.api_key == "graphrag_test_key"
        assert "Store this key securely" in resp.warning

    def test_revoke_api_key_request(self):
        from src.api.admin_routes import RevokeAPIKeyRequest

        req = RevokeAPIKeyRequest(api_key="some_key")
        assert req.api_key == "some_key"

    def test_list_api_keys_response(self):
        from src.api.admin_routes import ListAPIKeysResponse

        resp = ListAPIKeysResponse(keys=[{"name": "test", "is_active": True}])
        assert len(resp.keys) == 1


# =============================================================================
# Admin Route Tests
# =============================================================================


class TestCreateAPIKeyEndpoint:
    def test_create_key_success(self, client, api_key):
        response = client.post(
            "/api/v1/admin/keys/create",
            json={"name": "new_test_key", "rate_limit": 50, "daily_limit": 5000},
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "new_test_key"
        assert data["rate_limit"] == 50
        assert data["daily_limit"] == 5000
        assert "api_key" in data
        assert data["api_key"].startswith("graphrag_")

    def test_create_key_default_limits(self, client, api_key):
        response = client.post(
            "/api/v1/admin/keys/create",
            json={"name": "default_limits"},
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["rate_limit"] == 100
        assert data["daily_limit"] == 10000

    def test_create_key_requires_auth(self, client):
        response = client.post(
            "/api/v1/admin/keys/create",
            json={"name": "test"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_create_key_invalid_api_key(self, client):
        response = client.post(
            "/api/v1/admin/keys/create",
            json={"name": "test"},
            headers={"X-API-Key": "invalid_key"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestListAPIKeysEndpoint:
    def test_list_keys_success(self, client, api_key):
        response = client.get(
            "/api/v1/admin/keys/list",
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "keys" in data
        assert isinstance(data["keys"], list)
        assert len(data["keys"]) >= 1  # At least the default key

    def test_list_keys_includes_metadata(self, client, api_key):
        # First create a key
        client.post(
            "/api/v1/admin/keys/create",
            json={"name": "listed_key"},
            headers={"X-API-Key": api_key},
        )

        response = client.get(
            "/api/v1/admin/keys/list",
            headers={"X-API-Key": api_key},
        )

        data = response.json()
        for key_info in data["keys"]:
            assert "name" in key_info
            assert "created_at" in key_info
            assert "is_active" in key_info
            assert "rate_limit" in key_info

    def test_list_keys_requires_auth(self, client):
        response = client.get("/api/v1/admin/keys/list")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestRevokeAPIKeyEndpoint:
    def test_revoke_key_success(self, client, api_key):
        # Create a key first
        create_response = client.post(
            "/api/v1/admin/keys/create",
            json={"name": "to_revoke"},
            headers={"X-API-Key": api_key},
        )
        new_key = create_response.json()["api_key"]

        # Revoke it
        response = client.post(
            "/api/v1/admin/keys/revoke",
            json={"api_key": new_key},
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "revoked"

    def test_revoke_nonexistent_key(self, client, api_key):
        response = client.post(
            "/api/v1/admin/keys/revoke",
            json={"api_key": "nonexistent_key_12345"},
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_revoke_key_requires_auth(self, client):
        response = client.post(
            "/api/v1/admin/keys/revoke",
            json={"api_key": "some_key"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_revoked_key_cannot_authenticate(self, client, api_key):
        # Create and revoke a key
        create_response = client.post(
            "/api/v1/admin/keys/create",
            json={"name": "will_revoke"},
            headers={"X-API-Key": api_key},
        )
        new_key = create_response.json()["api_key"]

        client.post(
            "/api/v1/admin/keys/revoke",
            json={"api_key": new_key},
            headers={"X-API-Key": api_key},
        )

        # Try to use the revoked key
        response = client.get(
            "/api/v1/admin/keys/list",
            headers={"X-API-Key": new_key},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
