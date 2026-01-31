"""
Admin routes for API key management and system administration.

These endpoints should be protected with additional admin-level authentication
in production (e.g., only accessible from internal network or with admin tokens).
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.api.auth import APIKey, get_api_key_manager, verify_api_key


router = APIRouter(prefix="/admin", tags=["Admin"])


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str
    rate_limit: int = 100
    daily_limit: int = 10000


class CreateAPIKeyResponse(BaseModel):
    """Response with the new API key."""

    api_key: str
    name: str
    rate_limit: int
    daily_limit: int
    created_at: datetime
    warning: str = "Store this key securely - it cannot be retrieved again"


class ListAPIKeysResponse(BaseModel):
    """Response listing all API keys."""

    keys: List[dict]


class RevokeAPIKeyRequest(BaseModel):
    """Request to revoke an API key by key value or by name."""

    api_key: str | None = None
    name: str | None = None


@router.post("/keys/create", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_key: APIKey = Depends(verify_api_key),
) -> CreateAPIKeyResponse:
    """
    Create a new API key.

    Requires authentication with an existing API key.
    In production, this should require admin-level permissions.
    """
    manager = get_api_key_manager()

    new_key = manager.create_key(
        name=request.name,
        rate_limit=request.rate_limit,
        daily_limit=request.daily_limit,
    )

    return CreateAPIKeyResponse(
        api_key=new_key,
        name=request.name,
        rate_limit=request.rate_limit,
        daily_limit=request.daily_limit,
        created_at=datetime.utcnow(),
    )


@router.get("/keys/list", response_model=ListAPIKeysResponse)
async def list_api_keys(
    current_key: APIKey = Depends(verify_api_key),
) -> ListAPIKeysResponse:
    """
    List all API keys (without revealing the actual keys).

    Returns metadata about all keys for management purposes.
    Requires authentication.
    """
    manager = get_api_key_manager()
    keys = manager.list_keys()

    return ListAPIKeysResponse(
        keys=[
            {
                "name": key.name,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "is_active": key.is_active,
                "rate_limit": key.rate_limit,
                "daily_limit": key.daily_limit,
            }
            for key in keys
        ]
    )


@router.post("/keys/revoke")
async def revoke_api_key(
    request: RevokeAPIKeyRequest,
    current_key: APIKey = Depends(verify_api_key),
) -> dict:
    """
    Revoke an API key by key value or by name.

    Makes the key inactive and prevents further use.
    Requires authentication.
    """
    if not request.api_key and not request.name:
        raise HTTPException(status_code=400, detail="Provide api_key or name")

    manager = get_api_key_manager()

    success = False
    if request.api_key:
        success = manager.revoke_key(request.api_key)
    if not success and request.name:
        success = manager.revoke_key_by_name(request.name)

    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    return {
        "status": "revoked",
        "message": "API key has been successfully revoked",
    }
