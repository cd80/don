"""Rate limiting for Don trading framework API.

This module implements adaptive rate limiting based on user roles
to prevent API abuse and ensure fair resource usage.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

from .rbac import Role

logger = logging.getLogger(__name__)

class RateLimiter:
    """Implement adaptive rate limiting."""

    def __init__(self):
        """Initialize rate limiter."""
        # Requests per minute by role
        self._rate_limits = {
            Role.ADMIN: 1000,
            Role.MANAGER: 500,
            Role.TRADER: 300,
            Role.ANALYST: 200,
            Role.VIEWER: 100
        }
        self._request_counts: Dict[str, Dict] = {}

    def _clean_old_requests(self, user_id: str):
        """Clean requests older than 1 minute."""
        now = datetime.utcnow()
        if user_id in self._request_counts:
            self._request_counts[user_id]['requests'] = [
                ts for ts in self._request_counts[user_id]['requests']
                if now - ts < timedelta(minutes=1)
            ]

    def check_rate_limit(self, user_id: str, role: Role) -> bool:
        """Check if request is within rate limit.

        Args:
            user_id: User identifier
            role: User's role

        Returns:
            True if request is allowed
        """
        now = datetime.utcnow()

        if user_id not in self._request_counts:
            self._request_counts[user_id] = {
                'requests': [now]
            }
            return True

        self._clean_old_requests(user_id)
        request_count = len(self._request_counts[user_id]['requests'])

        if request_count >= self._rate_limits[role]:
            logger.warning(
                f"Rate limit exceeded for user {user_id} with role {role.name}")
            return False

        self._request_counts[user_id]['requests'].append(now)
        return True

    def get_remaining_requests(self, user_id: str, role: Role) -> int:
        """Get remaining requests for current minute.

        Args:
            user_id: User identifier
            role: User's role

        Returns:
            Number of remaining requests
        """
        if user_id not in self._request_counts:
            return self._rate_limits[role]

        self._clean_old_requests(user_id)
        used = len(self._request_counts[user_id]['requests'])
        return max(0, self._rate_limits[role] - used)
