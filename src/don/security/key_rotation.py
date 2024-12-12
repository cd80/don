"""API key rotation management for Don trading framework.

This module implements automatic API key rotation with graceful transitions
to ensure continuous system operation during key changes.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class KeyRotationManager:
    """Manage API key rotation and transitions."""

    def __init__(self, key_store_path: str, rotation_interval: int = 30):
        """Initialize key rotation manager.

        Args:
            key_store_path: Path to encrypted key store
            rotation_interval: Days between key rotations (default: 30)
        """
        self.key_store_path = key_store_path
        self.rotation_interval = rotation_interval
        self.fernet = Fernet(Fernet.generate_key())
        self.current_key: Optional[str] = None
        self.previous_key: Optional[str] = None
        self.next_rotation: Optional[datetime] = None

    def initialize(self):
        """Initialize key store and load existing keys."""
        try:
            with open(self.key_store_path, 'rb') as f:
                encrypted_data = f.read()
                data = self.fernet.decrypt(encrypted_data).decode()
                key_data = eval(data)  # Safe as we control the encrypted content
                self.current_key = key_data.get('current_key')
                self.previous_key = key_data.get('previous_key')
                self.next_rotation = datetime.fromisoformat(
                    key_data.get('next_rotation'))
        except FileNotFoundError:
            # First time initialization
            self._generate_new_key()
        except Exception as e:
            logger.error(f"Failed to initialize key store: {str(e)}")
            raise

    def _generate_new_key(self) -> str:
        """Generate new API key."""
        new_key = Fernet.generate_key().decode()
        self.previous_key = self.current_key
        self.current_key = new_key
        self.next_rotation = datetime.utcnow() + timedelta(
            days=self.rotation_interval)
        self._save_keys()
        return new_key

    def _save_keys(self):
        """Save keys to encrypted store."""
        data = {
            'current_key': self.current_key,
            'previous_key': self.previous_key,
            'next_rotation': self.next_rotation.isoformat()
        }
        encrypted_data = self.fernet.encrypt(str(data).encode())
        with open(self.key_store_path, 'wb') as f:
            f.write(encrypted_data)

    def get_active_keys(self) -> Tuple[str, Optional[str]]:
        """Get current and previous keys.

        Returns:
            Tuple of (current_key, previous_key)
        """
        return self.current_key, self.previous_key

    def check_rotation(self) -> bool:
        """Check if key rotation is needed.

        Returns:
            True if rotation was performed
        """
        if (self.next_rotation and
            datetime.utcnow() >= self.next_rotation):
            self._generate_new_key()
            logger.info("API key rotated successfully")
            return True
        return False

    def validate_key(self, key: str) -> bool:
        """Validate if key is either current or previous.

        Args:
            key: API key to validate

        Returns:
            True if key is valid
        """
        return (key == self.current_key or
                (self.previous_key and key == self.previous_key))
