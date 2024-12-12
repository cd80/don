"""Test suite for training CLI commands."""
import os
from unittest.mock import Mock, patch
import pandas as pd
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

    # Mock action space
    mock_action_space = Mock()
    mock_action_space.sample.return_value = 0

    # Mock process for dashboard
    mock_process = Mock()
    mock_process.is_alive.return_value = True
    mock_process.terminate = Mock()

    # Mock subprocess for dashboard
    mock_subprocess_run = Mock()

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.create_engine', return_value=mock_engine), \
         patch('don.cli.commands.sessionmaker', return_value=mock_session_maker), \
         patch('pandas.read_sql_query', return_value=mock_data), \
         patch('don.rl.actions.DiscreteActionSpace', return_value=mock_action_space), \
         patch('don.cli.commands.TradingEnvironment') as mock_env, \
         patch('multiprocessing.Process', return_value=mock_process), \
         patch('subprocess.run', mock_subprocess_run), \
         patch('time.sleep', side_effect=KeyboardInterrupt), \
         patch('signal.signal'):  # Mock signal handler

        # Set up environment mock
        instance = mock_env.return_value
        instance.action_space = mock_action_space
        instance.step = Mock(return_value=(None, 0, False, {}))
        instance.reset = Mock()

        # Run command
        result = runner.invoke(app, ["train", "--start"])

        # Verify result
        assert result.exit_code == 0
        assert "Training started" in result.stdout

        # Verify dashboard process was started and terminated
        mock_process.start.assert_called_once()
        mock_process.terminate.assert_called_once()

def test_train_without_start():
    """Test train command without --start flag."""
    with patch('don.cli.commands.load_settings') as mock_load_settings:
        result = runner.invoke(app, ["train"])
        assert result.exit_code == 1
        assert "Please use --start" in result.stdout
