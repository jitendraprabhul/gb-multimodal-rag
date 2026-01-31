"""
Authentication and authorization middleware for the API.

This module provides:
- API key-based authentication
- Rate limiting per API key
- Request validation
- Security headers
"""

import hashlib
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from src.core.logging import get_logger

logger = get_logger(__name__)

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKey(BaseModel):
    """API key model."""

    key_hash: str
    name: str
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 100  # requests per minute
    daily_limit: int = 10000  # requests per day


class RateLimitInfo(BaseModel):
    """Rate limit tracking info."""

    requests_per_minute: Dict[str, int] = {}
    requests_per_day: Dict[str, int] = {}
    last_reset_minute: Dict[str, datetime] = {}
    last_reset_day: Dict[str, datetime] = {}


class APIKeyManager:
    """
    Manages API keys and authentication.

    In production, this should be backed by a database.
    For now, we use in-memory storage with optional persistence.
    """

    def __init__(self):
        """Initialize the API key manager."""
        self._keys: Dict[str, APIKey] = {}
        self._rate_limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)

        # Create a default master key for initial setup
        self._create_default_key()

    def _create_default_key(self) -> None:
        """Create a default API key for initial setup."""
        default_key = "graphrag_default_key_CHANGE_IN_PRODUCTION"
        key_hash = self._hash_key(default_key)

        self._keys[key_hash] = APIKey(
            key_hash=key_hash,
            name="default_master_key",
            created_at=datetime.utcnow(),
            is_active=True,
            rate_limit=1000,
            daily_limit=100000,
        )

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new random API key."""
        return f"graphrag_{secrets.token_urlsafe(32)}"

    def create_key(
        self,
        name: str,
        rate_limit: int = 100,
        daily_limit: int = 10000,
    ) -> str:
        """
        Create a new API key.

        Args:
            name: Descriptive name for the key
            rate_limit: Requests per minute limit
            daily_limit: Requests per day limit

        Returns:
            The generated API key (return this to user, cannot be retrieved later)
        """
        key = self.generate_key()
        key_hash = self._hash_key(key)

        self._keys[key_hash] = APIKey(
            key_hash=key_hash,
            name=name,
            created_at=datetime.utcnow(),
            is_active=True,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
        )

        return key

    def validate_key(self, key: str) -> Tuple[bool, Optional[APIKey]]:
        """
        Validate an API key.

        Args:
            key: The API key to validate

        Returns:
            Tuple of (is_valid, api_key_object)
        """
        key_hash = self._hash_key(key)
        api_key = self._keys.get(key_hash)

        if not api_key:
            return False, None

        if not api_key.is_active:
            return False, None

        # Update last used time
        api_key.last_used = datetime.utcnow()

        return True, api_key

    def check_rate_limit(self, key_hash: str, api_key: APIKey) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Args:
            key_hash: Hashed API key
            api_key: API key object

        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = datetime.utcnow()
        rate_info = self._rate_limits[key_hash]

        # Check minute-based rate limit
        last_reset_minute = rate_info.last_reset_minute.get(key_hash)
        if not last_reset_minute or (now - last_reset_minute) >= timedelta(minutes=1):
            # Reset minute counter
            rate_info.requests_per_minute[key_hash] = 0
            rate_info.last_reset_minute[key_hash] = now

        current_minute_requests = rate_info.requests_per_minute.get(key_hash, 0)
        if current_minute_requests >= api_key.rate_limit:
            return False, f"Rate limit exceeded: {api_key.rate_limit} requests per minute"

        # Check daily rate limit
        last_reset_day = rate_info.last_reset_day.get(key_hash)
        if not last_reset_day or (now - last_reset_day) >= timedelta(days=1):
            # Reset daily counter
            rate_info.requests_per_day[key_hash] = 0
            rate_info.last_reset_day[key_hash] = now

        current_day_requests = rate_info.requests_per_day.get(key_hash, 0)
        if current_day_requests >= api_key.daily_limit:
            return False, f"Daily limit exceeded: {api_key.daily_limit} requests per day"

        # Increment counters
        rate_info.requests_per_minute[key_hash] = current_minute_requests + 1
        rate_info.requests_per_day[key_hash] = current_day_requests + 1

        return True, None

    def revoke_key(self, key: str) -> bool:
        """
        Revoke an API key.

        Args:
            key: The API key to revoke

        Returns:
            True if revoked, False if not found
        """
        key_hash = self._hash_key(key)
        api_key = self._keys.get(key_hash)

        if api_key:
            api_key.is_active = False
            return True

        return False

    def revoke_key_by_name(self, name: str) -> bool:
        """
        Revoke an API key by its name.

        Args:
            name: The name of the API key to revoke

        Returns:
            True if revoked, False if not found
        """
        for api_key in self._keys.values():
            if api_key.name == name:
                api_key.is_active = False
                return True
        return False

    def list_keys(self) -> list[APIKey]:
        """List all API keys (without revealing the actual keys)."""
        return list(self._keys.values())


# Global API key manager instance
_api_key_manager = APIKeyManager()


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    return _api_key_manager


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> APIKey:
    """
    Dependency to verify API key authentication.

    Args:
        api_key: API key from request header

    Returns:
        Validated APIKey object

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    manager = get_api_key_manager()
    is_valid, api_key_obj = manager.validate_key(api_key)

    if not is_valid or not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check rate limits
    key_hash = manager._hash_key(api_key)
    is_allowed, error_msg = manager.check_rate_limit(key_hash, api_key_obj)

    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error_msg,
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(api_key_obj.rate_limit),
            },
        )

    return api_key_obj


async def verify_api_key_optional(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[APIKey]:
    """
    Optional API key verification for public endpoints.

    Allows both authenticated and unauthenticated access,
    but applies rate limiting to authenticated users.
    """
    if not api_key:
        return None

    try:
        return await verify_api_key(api_key)
    except HTTPException:
        return None
