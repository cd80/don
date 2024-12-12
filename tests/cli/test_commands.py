"""Test cases for Don trading framework CLI commands."""

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch

from don.cli.commands import app
from don.cli.config import Settings

runner = CliRunner()

@pytest.fixture
def mock_settings():
    """Mock settings with valid configuration."""
    settings = Mock(spec=Settings)
    settings.check_completeness.return_value = True
    settings.check_database_connection.return_value = True
    settings.binance_api_key = Mock()
    settings.binance_api_key.get_secret_value.return_value = "test_key"
    settings.binance_api_secret = Mock()
    settings.binance_api_secret.get_secret_value.return_value = "test_secret"
    return settings

@pytest.mark.asyncio
async def test_setup_all(mock_settings):
    """Test 'don setup --all' command."""
    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.create_engine'), \
         patch('don.cli.commands.Base'):
        result = runner.invoke(app, ['setup', '--all'])
        assert result.exit_code == 0
        assert "Setup completed successfully" in result.stdout

@pytest.mark.asyncio
async def test_setup_incomplete_config(mock_settings):
    """Test setup with incomplete configuration."""
    mock_settings.check_completeness.return_value = False
    with patch('don.cli.commands.load_settings', return_value=mock_settings):
        result = runner.invoke(app, ['setup', '--all'])
        assert result.exit_code == 1
        assert "Configuration validation failed" in result.stdout

@pytest.mark.asyncio
async def test_collect_start():
    """Test 'don collect start' command."""
    with patch('don.cli.commands.load_settings', return_value=Mock(spec=Settings)), \
         patch('don.cli.commands.BinanceDataCollector') as mock_collector:
        instance = mock_collector.return_value
        instance.start = Mock()

        result = runner.invoke(app, ['collect', 'start'])
        assert result.exit_code == 0
        instance.start.assert_called_once()
        assert "Data collection started successfully" in result.stdout

@pytest.mark.asyncio
async def test_collect_stop():
    """Test 'don collect stop' command."""
    with patch('don.cli.commands.load_settings', return_value=Mock(spec=Settings)), \
         patch('don.cli.commands.BinanceDataCollector') as mock_collector:
        instance = mock_collector.return_value
        instance.stop = Mock()

        result = runner.invoke(app, ['collect', 'stop'])
        assert result.exit_code == 0
        instance.stop.assert_called_once()
        assert "Data collection stopped successfully" in result.stdout

@pytest.mark.asyncio
async def test_collect_resume():
    """Test 'don collect resume' command."""
    with patch('don.cli.commands.load_settings', return_value=Mock(spec=Settings)), \
         patch('don.cli.commands.BinanceDataCollector') as mock_collector:
        instance = mock_collector.return_value
        instance.resume = Mock()

        result = runner.invoke(app, ['collect', 'resume'])
        assert result.exit_code == 0
        instance.resume.assert_called_once()
        assert "Data collection resumed successfully" in result.stdout

@pytest.mark.asyncio
async def test_feature_all():
    """Test 'don feature --all' command."""
    with patch('don.cli.commands.load_settings', return_value=Mock(spec=Settings)), \
         patch('don.cli.commands.create_engine'), \
         patch('don.cli.commands.sessionmaker'), \
         patch('don.cli.commands.TechnicalIndicators') as mock_calculator:
        instance = mock_calculator.return_value
        instance.calculate_all = Mock()

        result = runner.invoke(app, ['feature', '--all'])
        assert result.exit_code == 0
        instance.calculate_all.assert_called_once()
        assert "All features calculated successfully" in result.stdout

@pytest.mark.asyncio
async def test_train_start():
    """Test 'don train start' command."""
    with patch('don.cli.commands.load_settings', return_value=Mock(spec=Settings)), \
         patch('don.cli.commands.TradingEnvironment') as mock_env:
        instance = mock_env.return_value
        instance.step = Mock(side_effect=KeyboardInterrupt)
        instance.action_space = Mock()
        instance.action_space.sample = Mock()

        result = runner.invoke(app, ['train', '--start'])
        assert result.exit_code == 0
        assert "Training started" in result.stdout
