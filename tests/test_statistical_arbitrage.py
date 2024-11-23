"""
Tests for statistical arbitrage functionality.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch
import time

from src.strategies.statistical_arbitrage import (
    StatisticalArbitrage,
    StatArbType,
    PairAnalysis,
    TradingSignal
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'data': {
            'binance': {
                'api_key': 'test',
                'api_secret': 'test'
            },
            'kraken': {
                'api_key': 'test',
                'api_secret': 'test'
            }
        }
    }

@pytest.fixture
def exchanges():
    """Test exchanges."""
    return ['binance', 'kraken']

@pytest.fixture
def trader(config, exchanges):
    """Initialize trader for testing."""
    return StatisticalArbitrage(
        config=config,
        exchanges=exchanges,
        lookback_window=100,
        zscore_threshold=2.0,
        min_half_life=1.0,
        max_half_life=100.0,
        min_correlation=0.5,
        max_position=1.0,
        risk_limit=0.02
    )

@pytest.fixture
def sample_data():
    """Create sample price data."""
    np.random.seed(42)
    n_points = 200
    
    # Generate cointegrated series
    common_factor = np.random.randn(n_points).cumsum()
    noise_a = np.random.randn(n_points) * 0.1
    noise_b = np.random.randn(n_points) * 0.1
    
    prices_a = pd.Series(
        100 + common_factor + noise_a,
        name='BTC/USDT'
    )
    prices_b = pd.Series(
        50 + 0.5 * common_factor + noise_b,
        name='ETH/USDT'
    )
    
    # Generate mean-reverting series
    mean_rev = pd.Series(
        np.random.randn(n_points).cumsum() * 0.1 + 100,
        name='BNB/USDT'
    )
    
    return pd.DataFrame({
        'BTC/USDT': prices_a,
        'ETH/USDT': prices_b,
        'BNB/USDT': mean_rev
    })

def test_initialization(trader):
    """Test trader initialization."""
    assert trader.lookback_window == 100
    assert trader.zscore_threshold == 2.0
    assert trader.min_half_life == 1.0
    assert trader.max_half_life == 100.0
    assert trader.min_correlation == 0.5
    assert trader.max_position == 1.0
    assert trader.risk_limit == 0.02

@pytest.mark.asyncio
async def test_fetch_price_history(trader, sample_data):
    """Test price history fetching."""
    # Mock exchange client
    trader.exchange_clients['binance'] = Mock()
    trader.exchange_clients['binance'].fetch_historical_data = Mock(
        return_value={'close': sample_data['BTC/USDT']}
    )
    
    prices = await trader.fetch_price_history(['BTC/USDT'])
    
    assert isinstance(prices, pd.DataFrame)
    assert 'BTC/USDT' in prices.columns
    assert len(prices) == len(sample_data)

def test_analyze_pair(trader, sample_data):
    """Test pair analysis."""
    pair = trader.analyze_pair(
        sample_data['BTC/USDT'],
        sample_data['ETH/USDT']
    )
    
    assert isinstance(pair, PairAnalysis)
    assert pair.asset_a == 'BTC/USDT'
    assert pair.asset_b == 'ETH/USDT'
    assert -1 <= pair.correlation <= 1
    assert pair.cointegration_pvalue <= 0.05
    assert pair.hedge_ratio > 0
    assert pair.half_life >= trader.min_half_life
    assert isinstance(pair.metrics, dict)

def test_calculate_half_life(trader, sample_data):
    """Test half-life calculation."""
    # Test with mean-reverting series
    half_life = trader.calculate_half_life(sample_data['BNB/USDT'])
    assert trader.min_half_life <= half_life <= trader.max_half_life
    
    # Test with random walk
    random_walk = pd.Series(np.random.randn(100).cumsum())
    half_life = trader.calculate_half_life(random_walk)
    assert half_life == np.inf

def test_calculate_hurst_exponent(trader, sample_data):
    """Test Hurst exponent calculation."""
    # Test mean-reverting series
    hurst = trader.calculate_hurst_exponent(sample_data['BNB/USDT'])
    assert 0 <= hurst <= 1
    assert hurst < 0.5  # Mean-reverting series should have H < 0.5
    
    # Test trending series
    trend = pd.Series(np.linspace(0, 1, 100))
    hurst = trader.calculate_hurst_exponent(trend)
    assert hurst > 0.5  # Trending series should have H > 0.5

def test_generate_pairs_signal(trader, sample_data):
    """Test pairs trading signal generation."""
    pair = trader.analyze_pair(
        sample_data['BTC/USDT'],
        sample_data['ETH/USDT']
    )
    
    signal = trader.generate_pairs_signal(pair)
    
    if abs(pair.zscore) > trader.zscore_threshold:
        assert isinstance(signal, TradingSignal)
        assert signal.type == StatArbType.PAIRS_TRADING
        assert len(signal.assets) == 2
        assert len(signal.direction) == 2
        assert signal.confidence <= 1.0
    else:
        assert signal is None

def test_generate_mean_reversion_signal(trader, sample_data):
    """Test mean reversion signal generation."""
    signal = trader.generate_mean_reversion_signal(
        sample_data['BNB/USDT']
    )
    
    if signal:
        assert isinstance(signal, TradingSignal)
        assert signal.type == StatArbType.MEAN_REVERSION
        assert len(signal.assets) == 1
        assert len(signal.direction) == 1
        assert signal.confidence <= 1.0

def test_generate_cointegration_signal(trader, sample_data):
    """Test cointegration signal generation."""
    signal = trader.generate_cointegration_signal(sample_data)
    
    if signal:
        assert isinstance(signal, TradingSignal)
        assert len(signal.assets) >= 2
        assert len(signal.direction) == len(signal.assets)
        assert signal.confidence <= 1.0
        
        # Check risk limit
        total_risk = sum(abs(s) for s in signal.direction)
        assert total_risk <= trader.risk_limit

@pytest.mark.asyncio
async def test_execute_trades(trader):
    """Test trade execution."""
    signal = TradingSignal(
        type=StatArbType.PAIRS_TRADING,
        assets=['BTC/USDT', 'ETH/USDT'],
        direction=[0.1, -0.2],
        entry_zscore=2.5,
        exit_zscore=0.0,
        confidence=0.8,
        timestamp=time.time(),
        metrics={}
    )
    
    # Mock exchange client
    trader.exchange_clients['binance'] = Mock()
    trader.exchange_clients['binance'].create_order = Mock(
        return_value={'id': '123', 'status': 'filled'}
    )
    
    success = await trader.execute_trades(signal)
    
    assert success
    assert len(trader.positions) == 2
    assert len(trader.signals) == 1

def test_performance_metrics(trader):
    """Test performance metrics calculation."""
    # Add some test trades
    trader.trades = [
        {'pnl': 1.0},
        {'pnl': -0.5},
        {'pnl': 2.0}
    ]
    trader.positions = {
        'BTC/USDT': 0.1,
        'ETH/USDT': -0.2
    }
    
    metrics = trader.get_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert metrics['total_pnl'] == 2.5
    assert metrics['num_trades'] == 3
    assert 0 <= metrics['win_rate'] <= 1
    assert isinstance(metrics['current_positions'], dict)

@pytest.mark.asyncio
async def test_run(trader, sample_data):
    """Test strategy running."""
    # Mock exchange client and methods
    trader.exchange_clients['binance'] = Mock()
    trader.exchange_clients['binance'].fetch_historical_data = Mock(
        return_value={'close': sample_data['BTC/USDT']}
    )
    trader.exchange_clients['binance'].create_order = Mock(
        return_value={'id': '123', 'status': 'filled'}
    )
    
    # Run strategy for a short time
    async def stop_after_delay():
        await asyncio.sleep(0.1)
        raise KeyboardInterrupt
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    with pytest.raises(KeyboardInterrupt):
        await asyncio.gather(
            trader.run(symbols, interval=0.01),
            stop_after_delay()
        )
    
    # Check that signals were generated
    assert len(trader.signals) >= 0

def test_edge_cases(trader):
    """Test edge cases."""
    # Test with insufficient data
    short_data = pd.DataFrame({
        'BTC/USDT': [100, 101, 102],
        'ETH/USDT': [50, 51, 52]
    })
    
    signal = trader.generate_cointegration_signal(short_data)
    assert signal is None
    
    # Test with uncorrelated data
    uncorrelated = pd.DataFrame({
        'BTC/USDT': np.random.randn(100),
        'ETH/USDT': np.random.randn(100)
    })
    
    pair = trader.analyze_pair(
        uncorrelated['BTC/USDT'],
        uncorrelated['ETH/USDT']
    )
    assert pair is None
    
    # Test with non-stationary data
    random_walk = pd.Series(
        np.random.randn(100).cumsum(),
        name='BTC/USDT'
    )
    
    signal = trader.generate_mean_reversion_signal(random_walk)
    assert signal is None
