"""Role-Based Access Control (RBAC) for Don trading framework.

This module implements RBAC with hierarchical roles and granular permissions
for securing API endpoints and system resources.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class Permission(Enum):
    """System permissions."""
    READ_MARKET_DATA = auto()
    WRITE_MARKET_DATA = auto()
    READ_TRADES = auto()
    WRITE_TRADES = auto()
    READ_FEATURES = auto()
    WRITE_FEATURES = auto()
    READ_MODELS = auto()
    WRITE_MODELS = auto()
    MANAGE_USERS = auto()
    MANAGE_SYSTEM = auto()

class Role(Enum):
    """System roles with hierarchical permissions."""
    ADMIN = auto()
    MANAGER = auto()
    TRADER = auto()
    ANALYST = auto()
    VIEWER = auto()

class RBACManager:
    """Manage roles and permissions."""

    def __init__(self):
        """Initialize RBAC manager."""
        self._role_permissions: Dict[Role, Set[Permission]] = {
            Role.ADMIN: {p for p in Permission},  # All permissions
            Role.MANAGER: {
                Permission.READ_MARKET_DATA,
                Permission.WRITE_MARKET_DATA,
                Permission.READ_TRADES,
                Permission.WRITE_TRADES,
                Permission.READ_FEATURES,
                Permission.WRITE_FEATURES,
                Permission.READ_MODELS,
                Permission.WRITE_MODELS,
            },
            Role.TRADER: {
                Permission.READ_MARKET_DATA,
                Permission.READ_TRADES,
                Permission.WRITE_TRADES,
                Permission.READ_FEATURES,
                Permission.READ_MODELS,
            },
            Role.ANALYST: {
                Permission.READ_MARKET_DATA,
                Permission.READ_TRADES,
                Permission.READ_FEATURES,
                Permission.READ_MODELS,
            },
            Role.VIEWER: {
                Permission.READ_MARKET_DATA,
                Permission.READ_TRADES,
                Permission.READ_FEATURES,
            }
        }
        self._user_roles: Dict[str, Role] = {}

    def assign_role(self, user_id: str, role: Role):
        """Assign role to user.

        Args:
            user_id: User identifier
            role: Role to assign
        """
        self._user_roles[user_id] = role
        logger.info(f"Assigned role {role.name} to user {user_id}")

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get permissions for user.

        Args:
            user_id: User identifier

        Returns:
            Set of permissions
        """
        role = self._user_roles.get(user_id)
        if not role:
            return set()
        return self._role_permissions[role]

    def check_permission(self, user_id: str,
                        permission: Permission) -> bool:
        """Check if user has specific permission.

        Args:
            user_id: User identifier
            permission: Permission to check

        Returns:
            True if user has permission
        """
        return permission in self.get_user_permissions(user_id)

    def get_user_role(self, user_id: str) -> Optional[Role]:
        """Get role for user.

        Args:
            user_id: User identifier

        Returns:
            User's role or None
        """
        return self._user_roles.get(user_id)
