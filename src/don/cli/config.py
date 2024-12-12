"""Configuration management for Don trading framework.

This module handles configuration loading, validation, and management
including API keys, database settings, and other parameters.
"""

from pathlib import Path
from typing import Optional

from pydantic import (
    Field,
    PostgresDsn,
    SecretStr,
    field_validator,
    ConfigDict
)
from pydantic_settings import BaseSettings
from rich.console import Console

console = Console()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Binance API settings
    binance_api_key: SecretStr = Field(...)  # Required, no default
    binance_api_secret: SecretStr = Field(...)  # Required, no default
    trading_symbol: str = "BTCUSDT"  # Default trading pair

    # Database settings
    database_url: PostgresDsn = Field(...)  # Required, no default

    # Feature calculation settings
    feature_update_interval: int = 3600  # seconds
    technical_indicators_enabled: bool = True

    # Training settings
    checkpoint_dir: Path = Path("checkpoints")
    dashboard_host: str = "localhost"
    dashboard_port: int = 8501

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=('settings_',),
        validate_default=True
    )

    @field_validator("checkpoint_dir")
    def validate_checkpoint_dir(cls, v: Path) -> Path:
        """Ensure checkpoint directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def check_completeness(self) -> bool:
        """Check if all required configuration is present and valid."""
        try:
            # Check Binance API credentials
            if not self.binance_api_key.get_secret_value():
                msg = "Missing Binance API key"
                console.print(f"[red]{msg}[/red]")
                raise ValueError(msg)
            if not self.binance_api_secret.get_secret_value():
                msg = "Missing Binance API secret"
                console.print(f"[red]{msg}[/red]")
                raise ValueError(msg)

            # Validate database URL
            if not self.database_url:
                msg = "Missing database URL"
                console.print(f"[red]{msg}[/red]")
                raise ValueError(msg)

            console.print("[green]Configuration validation successful![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Configuration validation failed: {str(e)}[/red]")
            raise

    def check_database_connection(self) -> bool:
        """Test PostgreSQL database connection."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(str(self.database_url))
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            console.print("[green]Database connection successful![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Database connection failed: {str(e)}[/red]")
            return False

def load_settings() -> Settings:
    """Load and validate settings from environment."""
    try:
        settings = Settings()
        return settings
    except Exception as e:
        console.print(f"[red]Failed to load settings: {str(e)}[/red]")
        raise
