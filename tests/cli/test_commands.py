"""Test cases for Don trading framework CLI commands."""

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

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
    settings.database_url = "postgresql://user:pass@localhost/testdb"
    settings.trading_symbol = "BTCUSDT"
    return settings

@pytest.mark.asyncio
async def test_setup_all(mock_settings):
    """Test 'don setup --all' command."""
    mock_engine = Mock()
    mock_metadata = MagicMock()

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.create_engine', return_value=mock_engine), \
         patch('don.cli.commands.Base.metadata', mock_metadata):
        result = runner.invoke(app, ['setup', '--all'])
        assert result.exit_code == 0
        assert "Setup completed successfully" in result.stdout
        mock_metadata.create_all.assert_called_once_with(mock_engine)

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
    mock_settings = Mock(spec=Settings)
    mock_settings.binance_api_key = Mock()
    mock_settings.binance_api_key.get_secret_value.return_value = "test_key"
    mock_settings.binance_api_secret = Mock()
    mock_settings.binance_api_secret.get_secret_value.return_value = "test_secret"
    mock_settings.trading_symbol = "BTCUSDT"

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.BinanceDataCollector') as mock_collector:
        instance = mock_collector.return_value
        instance.start = Mock()

        result = runner.invoke(app, ['collect', 'start'])
        assert result.exit_code == 0
        mock_collector.assert_called_once_with(
            symbol="BTCUSDT",
            api_key="test_key",
            api_secret="test_secret"
        )
        instance.start.assert_called_once()
        assert "Data collection started successfully" in result.stdout

@pytest.mark.asyncio
async def test_collect_stop():
    """Test 'don collect stop' command."""
    mock_settings = Mock(spec=Settings)
    mock_settings.binance_api_key = Mock()
    mock_settings.binance_api_key.get_secret_value.return_value = "test_key"
    mock_settings.binance_api_secret = Mock()
    mock_settings.binance_api_secret.get_secret_value.return_value = "test_secret"
    mock_settings.trading_symbol = "BTCUSDT"

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.BinanceDataCollector') as mock_collector:
        instance = mock_collector.return_value
        instance.stop = Mock()

        result = runner.invoke(app, ['collect', 'stop'])
        assert result.exit_code == 0
        mock_collector.assert_called_once_with(
            symbol="BTCUSDT",
            api_key="test_key",
            api_secret="test_secret"
        )
        instance.stop.assert_called_once()
        assert "Data collection stopped successfully" in result.stdout

@pytest.mark.asyncio
async def test_collect_resume():
    """Test 'don collect resume' command."""
    mock_settings = Mock(spec=Settings)
    mock_settings.binance_api_key = Mock()
    mock_settings.binance_api_key.get_secret_value.return_value = "test_key"
    mock_settings.binance_api_secret = Mock()
    mock_settings.binance_api_secret.get_secret_value.return_value = "test_secret"
    mock_settings.trading_symbol = "BTCUSDT"

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.BinanceDataCollector') as mock_collector:
        instance = mock_collector.return_value
        instance.resume = Mock()

        result = runner.invoke(app, ['collect', 'resume'])
        assert result.exit_code == 0
        mock_collector.assert_called_once_with(
            symbol="BTCUSDT",
            api_key="test_key",
            api_secret="test_secret"
        )
        instance.resume.assert_called_once()
        assert "Data collection resumed successfully" in result.stdout

@pytest.mark.asyncio
async def test_feature_all():
    """Test 'don feature --all' command."""
    mock_settings = Mock(spec=Settings)
    mock_settings.database_url = "postgresql://user:pass@localhost/testdb"
    mock_settings.trading_symbol = "BTCUSDT"

    mock_session = Mock()
    mock_session_maker = Mock()
    mock_session_maker.return_value = mock_session
    mock_engine = Mock()
    mock_session.bind = mock_engine

    # Create mock market data
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='h'),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
    })

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.create_engine', return_value=mock_engine), \
         patch('don.cli.commands.sessionmaker', return_value=mock_session_maker), \
         patch('pandas.read_sql_query', return_value=mock_data), \
         patch('don.cli.commands.TechnicalIndicators') as mock_calculator:
        instance = mock_calculator.return_value
        instance.calculate_all = Mock(return_value=pd.DataFrame({
            'sma_20': [101.0] * 5,
            'rsi': [60.0] * 5,
            'macd': [0.5] * 5,
            'macd_signal': [0.3] * 5,
            'macd_hist': [0.2] * 5,
            'bb_upper': [110.0] * 5,
            'bb_middle': [105.0] * 5,
            'bb_lower': [100.0] * 5,
            'obv': [5000.0] * 5,
            'vwap': [103.0] * 5,
            'stoch_k': [70.0] * 5,
            'stoch_d': [65.0] * 5,
            'adx': [30.0] * 5,
            'plus_di': [25.0] * 5,
            'minus_di': [20.0] * 5
        }))

        result = runner.invoke(app, ['feature', '--all'])
        assert result.exit_code == 0
        mock_session_maker.assert_called_once()
        instance.calculate_all.assert_called_once()
        mock_session.commit.assert_called_once()
        assert "All features calculated successfully" in result.stdout

def test_train_start():
    """Test 'train start' command."""
    mock_settings = Mock(spec=Settings)
    mock_settings.checkpoint_dir = "/tmp/don/checkpoints"
    mock_settings.database_url = "postgresql://test:test@localhost/test"
    mock_settings.trading_symbol = "BTCUSDT"
    mock_settings.dashboard_host = "localhost"
    mock_settings.dashboard_port = 8501

    # Mock database session and data
    mock_session = Mock()
    mock_session_maker = Mock(return_value=mock_session)
    mock_engine = Mock()
    mock_session.bind = mock_engine

    # Create mock market data
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='h'),
        'open': [100.0] * 5,
        'high': [105.0] * 5,
        'low': [95.0] * 5,
        'close': [102.0] * 5,
        'volume': [1000.0] * 5,
        'sma_20': [101.0] * 5,
        'rsi': [60.0] * 5,
        'macd': [0.5] * 5,
        'bb_upper': [110.0] * 5,
        'bb_lower': [100.0] * 5,
        'obv': [5000.0] * 5,
        'vwap': [103.0] * 5,
        'adx': [30.0] * 5
    })

    # Mock process for dashboard
    mock_process = Mock()
    mock_process.is_alive.return_value = True

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.create_engine', return_value=mock_engine), \
         patch('don.cli.commands.sessionmaker', return_value=mock_session_maker), \
         patch('pandas.read_sql_query', return_value=mock_data), \
         patch('don.cli.commands.TradingEnvironment') as mock_env, \
         patch('multiprocessing.Process', return_value=mock_process), \
         patch('time.sleep'):  # Mock sleep to speed up test

        # Set up environment mock
        instance = mock_env.return_value
        instance.action_space = Mock()
        instance.action_space.sample = Mock(return_value=0)
        instance.step = Mock(return_value=(None, 0, False, {}))
        instance.reset = Mock()

        # Run command
        result = runner.invoke(app, ["train", "--start"])

        # Verify result
        assert result.exit_code == 0
        assert "Training started" in result.stdout

        # Verify dashboard process was started
        mock_process.start.assert_called_once()

def test_train_without_start():
    """Test train command without --start flag."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
    assert "Please use --start flag to start training" in result.stdout
