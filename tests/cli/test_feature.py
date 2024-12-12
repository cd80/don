"""Test suite for feature calculation CLI commands."""
import os
import pytest
from typer.testing import CliRunner
from don.cli.commands import app

runner = CliRunner()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    os.environ["TEST_MODE"] = "1"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
    yield
    os.environ.pop("TEST_MODE", None)

def test_feature_all():
    """Test 'feature --all' command."""
    result = runner.invoke(app, ["feature", "--all"])
    assert result.exit_code == 0
    assert "All features calculated successfully" in result.stdout

def test_feature_without_all():
    """Test feature command without --all flag."""
    result = runner.invoke(app, ["feature"])
    assert result.exit_code == 1
    assert "Please specify --all" in result.stdout
