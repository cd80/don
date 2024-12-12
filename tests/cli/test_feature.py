"""Test suite for feature calculation CLI commands."""
import os
from unittest.mock import Mock, patch
import pandas as pd
import pytest
from typer.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from don.cli.commands import app
from don.cli.config import Settings
from don.database.models import Base, MarketData, TechnicalFeatures

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

@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='h'),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
    })

def test_feature_all(mock_market_data):
    """Test 'feature --all' command."""
    mock_settings = Mock(spec=Settings)
    mock_settings.database_url = "postgresql://test:test@localhost/test"
    mock_settings.trading_symbol = "BTCUSDT"

    mock_engine = Mock()
    mock_session = Mock()
    mock_session.bind = mock_engine

    with patch('don.cli.commands.load_settings', return_value=mock_settings), \
         patch('don.cli.commands.create_engine', return_value=mock_engine), \
         patch('don.cli.commands.sessionmaker', return_value=lambda: mock_session), \
         patch('pandas.read_sql_query', return_value=mock_market_data), \
         patch('don.cli.commands.TechnicalIndicators') as mock_indicators:

        # Configure mock TechnicalIndicators
        mock_calc = Mock()
        mock_calc.calculate_all.return_value = pd.DataFrame({
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
        })
        mock_indicators.return_value = mock_calc

        result = runner.invoke(app, ["feature", "--all"])
        assert result.exit_code == 0
        assert "All features calculated successfully" in result.stdout
        mock_calc.calculate_all.assert_called_once()
        mock_session.commit.assert_called_once()

def test_feature_without_all():
    """Test feature command without --all flag."""
    mock_settings = Mock(spec=Settings)
    mock_settings.database_url = "postgresql://test:test@localhost/test"

    with patch('don.cli.commands.load_settings', return_value=mock_settings):
        result = runner.invoke(app, ["feature"])
        assert result.exit_code == 1
        assert "Please specify --all" in result.stdout
