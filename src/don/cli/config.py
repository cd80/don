"""Configuration management for Don trading framework.

This module handles configuration loading, validation, and management
including API keys, database settings, and other parameters.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, PostgresDsn, SecretStr, validator
from rich.console import Console

console = Console()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Binance API settings
    binance_api_key: SecretStr
    binance_api_secret: SecretStr

    # Database settings
    database_url: PostgresDsn

    # Feature calculation settings
    feature_update_interval: int = 3600  # seconds
    technical_indicators_enabled: bool = True

    # Training settings
    model_checkpoint_dir: Path = Path("checkpoints")
    dashboard_port: int = 8501

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("model_checkpoint_dir")
    def validate_checkpoint_dir(cls, v: Path) -> Path:
        """Ensure checkpoint directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def check_completeness(self) -> bool:
        """Check if all required configuration is present and valid."""
        try:
            # Check Binance API credentials
            if not self.binance_api_key.get_secret_value():
                console.print("[red]Missing Binance API key[/red]")
                return False
            if not self.binance_api_secret.get_secret_value():
                console.print("[red]Missing Binance API secret[/red]")
                return False

            # Validate database URL
            if not self.database_url:
                console.print("[red]Missing database URL[/red]")
                return False

            console.print("[green]Configuration validation successful![/green]")
            return True

        except Exception as e:
            console.print(f"[red]Configuration validation failed: {str(e)}[/red]")
            return False

    def check_database_connection(self) -> bool:
        """Test PostgreSQL database connection."""
        try:
            from sqlalchemy import create_engine

            engine = create_engine(str(self.database_url))
            with engine.connect() as conn:
                conn.execute("SELECT 1")

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
