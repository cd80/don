"""Test suite for training CLI commands."""
import os
import pytest
from typer.testing import CliRunner
from don.cli.commands import app

runner = CliRunner()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    os.environ["TEST_MODE"] = "1"
    os.environ["CHECKPOINT_DIR"] = "/tmp/don/checkpoints"
    yield
    os.environ.pop("TEST_MODE", None)

def test_train_start():
    """Test 'train --start' command."""
    result = runner.invoke(app, ["train", "--start"])
    assert result.exit_code == 0
    assert "Training started" in result.stdout

def test_train_without_start():
    """Test train command without --start flag."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert "Please use --start" in result.stdout
