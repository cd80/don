"""Test suite for feature calculation CLI commands."""
import os
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from don.cli.commands import app
from don.cli.config import Settings

runner = CliRunner()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    os.environ["TEST_MODE"] = "1"
    os.environ["BINANCE_API_KEY"] = "test_key"
    os.environ["BINANCE_API_SECRET"] = "test_secret"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
    os.environ["TRADING_SYMBOL"] = "BTCUSDT"
    yield
    os.environ.pop("TEST_MODE", None)
    os.environ.pop("BINANCE_API_KEY", None)
    os.environ.pop("BINANCE_API_SECRET", None)
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("TRADING_SYMBOL", None)

def test_feature_all():
    """Test 'feature --all' command."""
    mock_settings = Mock(spec=Settings)
    mock_settings.database_url = "postgresql://test:test@localhost/test"

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.TechnicalIndicators') as mock_indicators:
        result = runner.invoke(app, ["feature", "--all"])
        assert result.exit_code == 0
        assert "All features calculated successfully" in result.stdout

def test_feature_without_all():
    """Test feature command without --all flag."""
    with patch('don.cli.commands.load_settings') as mock_load_settings:
        result = runner.invoke(app, ["feature"])
        assert result.exit_code == 1
        assert "Please specify --all" in result.stdout
