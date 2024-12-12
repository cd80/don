"""Authentication routes for Don trading framework.

This module provides API endpoints for user authentication and management.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..security import RBACManager, Role, Permission
from ..security.middleware import get_current_user

router = APIRouter()

class UserRole(BaseModel):
    """User role assignment request."""
    role: Role

@router.post("/users/{user_id}/role")
async def assign_role(
    user_id: str,
    role_request: UserRole,
    current_user: str = Depends(get_current_user),
    rbac_manager: RBACManager = Depends()
) -> dict:
    """Assign role to user.

    Args:
        user_id: User ID
        role_request: Role to assign
        current_user: Current authenticated user
        rbac_manager: RBAC manager

    Returns:
        Success message
    """
    # Check if current user has permission to manage users
    if not rbac_manager.check_permission(current_user, Permission.MANAGE_USERS):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to manage users"
        )

    rbac_manager.assign_role(user_id, role_request.role)
    return {"message": f"Assigned role {role_request.role.name} to user {user_id}"}

@router.get("/users/me/permissions")
async def get_permissions(
    current_user: str = Depends(get_current_user),
    rbac_manager: RBACManager = Depends()
) -> dict:
    """Get permissions for current user.

    Args:
        current_user: Current authenticated user
        rbac_manager: RBAC manager

    Returns:
        User permissions
    """
    permissions = rbac_manager.get_user_permissions(current_user)
    return {
        "user_id": current_user,
        "permissions": [p.name for p in permissions]
    }
