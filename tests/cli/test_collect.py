"""Test suite for data collection CLI commands."""
import os
import pytest
from typer.testing import CliRunner
from don.cli.commands import app

runner = CliRunner()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    os.environ["TEST_MODE"] = "1"
    os.environ["BINANCE_API_KEY"] = "test_key"
    os.environ["BINANCE_API_SECRET"] = "test_secret"
    os.environ["TRADING_SYMBOL"] = "BTCUSDT"
    os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/test"
    yield
    os.environ.pop("TEST_MODE", None)
    os.environ.pop("BINANCE_API_KEY", None)
    os.environ.pop("BINANCE_API_SECRET", None)
    os.environ.pop("TRADING_SYMBOL", None)
    os.environ.pop("DATABASE_URL", None)

def test_collect_start():
    """Test 'collect start' command."""
    result = runner.invoke(app, ["collect", "start"])
    assert result.exit_code == 0
    assert "Data collection started successfully" in result.stdout

def test_collect_stop():
    """Test 'collect stop' command."""
    result = runner.invoke(app, ["collect", "stop"])
    assert result.exit_code == 0
    assert "Data collection stopped successfully" in result.stdout

def test_collect_resume():
    """Test 'collect resume' command."""
    result = runner.invoke(app, ["collect", "resume"])
    assert result.exit_code == 0
    assert "Data collection resumed successfully" in result.stdout

def test_collect_invalid_action():
    """Test collect command with invalid action."""
    result = runner.invoke(app, ["collect", "invalid"])
    assert result.exit_code == 1
    assert "Invalid action" in result.stdout

def test_collect_with_custom_symbol():
    """Test collect command with custom symbol."""
    result = runner.invoke(app, ["collect", "start", "--symbol", "ETHUSDT"])
    assert result.exit_code == 0
    assert "Data collection started successfully" in result.stdout
