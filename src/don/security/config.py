"""Secure configuration management for Don trading framework.

This module implements encrypted configuration storage and validation
with secure secret rotation.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class SecureConfig:
    """Manage secure configuration storage."""

    def __init__(self, config_path: str, encryption_key: Optional[str] = None):
        """Initialize secure configuration.

        Args:
            config_path: Path to encrypted config file
            encryption_key: Optional encryption key (generated if None)
        """
        self.config_path = Path(config_path)
        self._fernet = Fernet(encryption_key or Fernet.generate_key())
        self._config: Dict[str, Any] = {}
        self._last_rotation: Optional[datetime] = None
        self._rotation_interval = timedelta(days=30)

    def load_config(self):
        """Load and decrypt configuration."""
        try:
            if self.config_path.exists():
                encrypted_data = self.config_path.read_bytes()
                data = self._fernet.decrypt(encrypted_data)
                config_dict = json.loads(data.decode())
                self._config = config_dict.get('config', {})
                self._last_rotation = datetime.fromisoformat(
                    config_dict.get('last_rotation', datetime.utcnow().isoformat())
                )
            else:
                self._initialize_config()
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise

    def _initialize_config(self):
        """Initialize new configuration."""
        self._config = {}
        self._last_rotation = datetime.utcnow()
        self._save_config()

    def _save_config(self):
        """Save and encrypt configuration."""
        config_dict = {
            'config': self._config,
            'last_rotation': self._last_rotation.isoformat()
        }
        encrypted_data = self._fernet.encrypt(
            json.dumps(config_dict).encode()
        )
        self.config_path.write_bytes(encrypted_data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self._config[key] = value
        self._save_config()

    def validate_config(self, config_model: BaseModel):
        """Validate configuration against Pydantic model.

        Args:
            config_model: Pydantic model for validation

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            config_model(**self._config)
        except ValidationError as e:
            raise ConfigValidationError(str(e))

    def check_rotation(self) -> bool:
        """Check if config rotation is needed.

        Returns:
            True if rotation was performed
        """
        if (datetime.utcnow() - self._last_rotation >
            self._rotation_interval):
            self._rotate_config()
            return True
        return False

    def _rotate_config(self):
        """Rotate configuration encryption."""
        new_key = Fernet.generate_key()
        new_fernet = Fernet(new_key)

        # Re-encrypt with new key
        self._fernet = new_fernet
        self._last_rotation = datetime.utcnow()
        self._save_config()
        logger.info("Configuration encryption rotated successfully")
