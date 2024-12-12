"""Authentication and security middleware for Don trading framework.

This module provides FastAPI middleware and dependencies for:
- API key authentication
- Rate limiting
- Role-based access control
"""

from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_429_TOO_MANY_REQUESTS
)

from .key_rotation import KeyRotationManager
from .rbac import RBACManager, Permission, Role
from .rate_limiting import RateLimiter

# API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for authentication and rate limiting."""

    def __init__(self, app,
                 key_manager: KeyRotationManager,
                 rbac_manager: RBACManager,
                 rate_limiter: RateLimiter):
        """Initialize security middleware.

        Args:
            app: FastAPI application
            key_manager: Key rotation manager
            rbac_manager: RBAC manager
            rate_limiter: Rate limiter
        """
        super().__init__(app)
        self.key_manager = key_manager
        self.rbac_manager = rbac_manager
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Check API key
        api_key = request.headers.get("X-API-Key")
        if not api_key or not self.key_manager.validate_key(api_key):
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        # Get user role
        user_id = request.state.user_id  # Set by auth dependency
        role = self.rbac_manager.get_user_role(user_id)
        if not role:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="User has no assigned role"
            )

        # Check rate limit
        if not self.rate_limiter.check_rate_limit(user_id, role):
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self.rate_limiter.get_remaining_requests(user_id, role)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

def get_current_user(
    api_key: str = Security(API_KEY_HEADER),
    key_manager: KeyRotationManager = Depends()
) -> str:
    """Dependency for getting current authenticated user.

    Args:
        api_key: API key from header
        key_manager: Key rotation manager

    Returns:
        User ID

    Raises:
        HTTPException: If authentication fails
    """
    if not key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    # In a real system, you would decode the user ID from the API key
    # For now, we'll use a placeholder
    return "user-123"

def check_permission(permission: Permission):
    """Dependency factory for checking permissions.

    Args:
        permission: Required permission

    Returns:
        Dependency function
    """
    def permission_dependency(
        user_id: str = Depends(get_current_user),
        rbac_manager: RBACManager = Depends()
    ) -> bool:
        if not rbac_manager.check_permission(user_id, permission):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission.name}"
            )
        return True
    return permission_dependency
