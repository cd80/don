"""Test suite for training CLI commands."""
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
    os.environ["CHECKPOINT_DIR"] = "/tmp/don/checkpoints"
    yield
    os.environ.pop("TEST_MODE", None)
    os.environ.pop("BINANCE_API_KEY", None)
    os.environ.pop("BINANCE_API_SECRET", None)
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("TRADING_SYMBOL", None)
    os.environ.pop("CHECKPOINT_DIR", None)

def test_train_start():
    """Test 'train start' command."""
    mock_settings = Mock(spec=Settings)
    mock_settings.checkpoint_dir = "/tmp/don/checkpoints"

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.TradingEnvironment') as mock_env:
        result = runner.invoke(app, ["train", "--start"])
        assert result.exit_code == 0
        assert "Training started" in result.stdout

def test_train_without_start():
    """Test train command without --start flag."""
    with patch('don.cli.commands.load_settings') as mock_load_settings:
        result = runner.invoke(app, ["train"])
        assert result.exit_code == 1
        assert "Please use --start" in result.stdout
