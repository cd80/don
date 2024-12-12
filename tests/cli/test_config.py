"""Test cases for Don trading framework configuration management."""

import pytest
from unittest.mock import Mock, patch

from don.cli.config import Settings, load_settings

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv('BINANCE_API_KEY', 'test_key')
    monkeypatch.setenv('BINANCE_API_SECRET', 'test_secret')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pass@localhost/test')

def test_load_settings(mock_env_vars):
    """Test settings loading from environment."""
    settings = load_settings()
    assert settings.binance_api_key.get_secret_value() == 'test_key'
    assert settings.binance_api_secret.get_secret_value() == 'test_secret'
    assert str(settings.database_url) == 'postgresql://user:pass@localhost/test'

def test_check_completeness_success(mock_env_vars):
    """Test configuration completeness check with valid settings."""
    settings = load_settings()
    assert settings.check_completeness() is True

def test_check_completeness_failure(monkeypatch):
    """Test configuration completeness check with missing settings."""
    monkeypatch.delenv('BINANCE_API_KEY', raising=False)
    with pytest.raises(Exception):
        Settings()

@pytest.mark.asyncio
async def test_check_database_connection():
    """Test database connection check."""
    settings = Settings(
        binance_api_key='test_key',
        binance_api_secret='test_secret',
        database_url='postgresql://user:pass@localhost/test'
    )

    with patch('sqlalchemy.create_engine') as mock_engine, \
         patch('sqlalchemy.text') as mock_text:
        mock_conn = Mock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_text.return_value = "SELECT 1"

        assert settings.check_database_connection() is True
        mock_text.assert_called_once_with("SELECT 1")
        mock_conn.execute.assert_called_once_with(mock_text.return_value)
