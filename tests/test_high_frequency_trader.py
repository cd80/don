"""
Tests for high-frequency trading functionality.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch
import time

from src.strategies.high_frequency_trader import (
    HighFrequencyTrader,
    SignalType,
    MarketSignal,
    ExecutionMetrics
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'data': {
            'binance': {
                'api_key': 'test',
                'api_secret': 'test'
            }
        }
    }

@pytest.fixture
def trader(config):
    """Initialize trader for testing."""
    return HighFrequencyTrader(
        config=config,
        exchange='binance',
        max_position=1.0,
        risk_limit=0.02,
        signal_threshold=0.5,
        execution_timeout=0.1,
        buffer_size=1000
    )

@pytest.fixture
def sample_data():
    """Create sample market data."""
    # Generate price series with trend and noise
    np.random.seed(42)
    n_points = 100
    trend = np.linspace(0, 0.1, n_points)
    noise = np.random.normal(0, 0.02, n_points)
    prices = 100 * (1 + trend + noise)
    volumes = np.random.uniform(0.1, 1.0, n_points)
    
    # Generate order book data
    order_flow = []
    for i in range(n_points):
        bid_volume = np.random.uniform(10, 20)
        ask_volume = np.random.uniform(10, 20)
        order_flow.append({
            'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume),
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'spread': np.random.uniform(0.01, 0.05)
        })
    
    return prices, volumes, order_flow

def test_initialization(trader):
    """Test trader initialization."""
    assert trader.max_position == 1.0
    assert trader.risk_limit == 0.02
    assert trader.signal_threshold == 0.5
    assert trader.execution_timeout == 0.1
    assert trader.position == 0.0
    assert len(trader.pending_orders) == 0

@pytest.mark.asyncio
async def test_update_market_data(trader, sample_data):
    """Test market data updates."""
    prices, volumes, order_flow = sample_data
    
    # Mock exchange client
    trader.client = Mock()
    trader.client.fetch_recent_trades = Mock(
        return_value=[{'price': prices[-1], 'amount': volumes[-1]}]
    )
    trader.client.fetch_orderbook = Mock(
        return_value={
            'bids': {prices[-1] * 0.99: order_flow[-1]['bid_volume']},
            'asks': {prices[-1] * 1.01: order_flow[-1]['ask_volume']}
        }
    )
    
    price, volume, flow = await trader.update_market_data('BTC/USDT')
    
    assert isinstance(price, float)
    assert isinstance(volume, float)
    assert isinstance(flow, dict)
    assert len(trader.price_buffer) == 1
    assert len(trader.volume_buffer) == 1
    assert len(trader.order_flow_buffer) == 1

def test_momentum_signal_generation(trader, sample_data):
    """Test momentum signal generation."""
    prices, _, _ = sample_data
    
    # Fill price buffer
    for price in prices:
        trader.price_buffer.append(price)
    
    signal = trader.generate_momentum_signal()
    
    if signal:
        assert isinstance(signal, MarketSignal)
        assert signal.type == SignalType.MOMENTUM
        assert -1 <= signal.direction <= 1
        assert 0 <= signal.strength <= 1
        assert 0 <= signal.confidence <= 1
        assert isinstance(signal.features, dict)

def test_mean_reversion_signal_generation(trader, sample_data):
    """Test mean reversion signal generation."""
    prices, _, _ = sample_data
    
    # Fill price buffer
    for price in prices:
        trader.price_buffer.append(price)
    
    signal = trader.generate_mean_reversion_signal()
    
    if signal:
        assert isinstance(signal, MarketSignal)
        assert signal.type == SignalType.MEAN_REVERSION
        assert -1 <= signal.direction <= 1
        assert 0 <= signal.strength <= 1
        assert 0 <= signal.confidence <= 1
        assert isinstance(signal.features, dict)

def test_order_flow_signal_generation(trader, sample_data):
    """Test order flow signal generation."""
    _, _, order_flow = sample_data
    
    # Fill order flow buffer
    for flow in order_flow:
        trader.order_flow_buffer.append(flow)
    
    signal = trader.generate_order_flow_signal()
    
    if signal:
        assert isinstance(signal, MarketSignal)
        assert signal.type == SignalType.ORDER_FLOW
        assert -1 <= signal.direction <= 1
        assert 0 <= signal.strength <= 1
        assert 0 <= signal.confidence <= 1
        assert isinstance(signal.features, dict)

def test_signal_combination(trader):
    """Test signal combination."""
    signals = [
        MarketSignal(
            type=SignalType.MOMENTUM,
            direction=1.0,
            strength=0.8,
            timestamp=time.time(),
            features={'momentum': 0.1},
            confidence=0.8,
            horizon=5.0
        ),
        MarketSignal(
            type=SignalType.MEAN_REVERSION,
            direction=-0.5,
            strength=0.6,
            timestamp=time.time(),
            features={'z_score': -1.5},
            confidence=0.6,
            horizon=10.0
        )
    ]
    
    combined = trader.combine_signals(signals)
    
    assert isinstance(combined, MarketSignal)
    assert -1 <= combined.direction <= 1
    assert 0 <= combined.strength <= 1
    assert 0 <= combined.confidence <= 1
    assert combined.horizon == 5.0  # Should take minimum horizon

def test_position_size_calculation(trader):
    """Test position size calculation."""
    signal = MarketSignal(
        type=SignalType.TECHNICAL,
        direction=1.0,
        strength=0.8,
        timestamp=time.time(),
        features={},
        confidence=0.7,
        horizon=5.0
    )
    
    size = trader.calculate_position_size(signal, price=50000)
    
    assert isinstance(size, float)
    assert abs(size) <= trader.max_position
    
    # Test risk limit
    high_price = 1000000
    size_with_risk = trader.calculate_position_size(signal, price=high_price)
    assert abs(size_with_risk) < abs(size)

@pytest.mark.asyncio
async def test_order_execution(trader):
    """Test order execution."""
    # Mock exchange client
    trader.client = Mock()
    trader.client.create_order = Mock(
        return_value={
            'id': '123',
            'status': 'open',
            'price': '50000'
        }
    )
    trader.client.get_order = Mock(
        return_value={'status': 'filled'}
    )
    
    success, metrics = await trader.execute_order(
        symbol='BTC/USDT',
        size=0.1,
        price=50000
    )
    
    assert success
    assert isinstance(metrics, ExecutionMetrics)
    assert metrics.latency > 0
    assert metrics.fill_rate == 1.0
    assert metrics.cost > 0

@pytest.mark.asyncio
async def test_order_timeout(trader):
    """Test order execution timeout."""
    # Mock exchange client with delay
    trader.client = Mock()
    trader.client.create_order = Mock(
        return_value={
            'id': '123',
            'status': 'open',
            'price': '50000'
        }
    )
    trader.client.get_order = Mock(
        return_value={'status': 'open'}
    )
    
    success, metrics = await trader.execute_order(
        symbol='BTC/USDT',
        size=0.1,
        price=50000
    )
    
    assert not success
    assert metrics.latency >= trader.execution_timeout
    assert metrics.fill_rate == 0.0

def test_performance_metrics(trader):
    """Test performance metrics calculation."""
    # Add some test metrics
    trader.metrics = [
        ExecutionMetrics(
            latency=0.005,
            slippage=0.0001,
            fill_rate=1.0,
            cost=0.5,
            impact=0.0001,
            timing_cost=0.0001,
            opportunity_cost=0.0
        ),
        ExecutionMetrics(
            latency=0.008,
            slippage=-0.0002,
            fill_rate=1.0,
            cost=0.8,
            impact=0.0002,
            timing_cost=0.0002,
            opportunity_cost=0.0
        )
    ]
    
    metrics = trader.get_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert metrics['avg_latency'] > 0
    assert metrics['num_trades'] == 2
    assert metrics['total_cost'] == 1.3

@pytest.mark.asyncio
async def test_run(trader, sample_data):
    """Test strategy running."""
    prices, volumes, order_flow = sample_data
    
    # Mock exchange client and methods
    trader.client = Mock()
    trader.client.fetch_recent_trades = Mock(
        return_value=[{'price': prices[-1], 'amount': volumes[-1]}]
    )
    trader.client.fetch_orderbook = Mock(
        return_value={
            'bids': {prices[-1] * 0.99: order_flow[-1]['bid_volume']},
            'asks': {prices[-1] * 1.01: order_flow[-1]['ask_volume']}
        }
    )
    trader.client.create_order = Mock(
        return_value={
            'id': '123',
            'status': 'filled',
            'price': str(prices[-1])
        }
    )
    
    # Run strategy for a short time
    async def stop_after_delay():
        await asyncio.sleep(0.1)
        raise KeyboardInterrupt
    
    with pytest.raises(KeyboardInterrupt):
        await asyncio.gather(
            trader.run('BTC/USDT', interval=0.01),
            stop_after_delay()
        )
    
    # Check that data was collected
    assert len(trader.price_buffer) > 0
    assert len(trader.volume_buffer) > 0
    assert len(trader.order_flow_buffer) > 0

def test_edge_cases(trader):
    """Test edge cases."""
    # Test with empty buffers
    assert trader.generate_momentum_signal() is None
    assert trader.generate_mean_reversion_signal() is None
    assert trader.generate_order_flow_signal() is None
    
    # Test with single data point
    trader.price_buffer.append(100)
    trader.volume_buffer.append(1.0)
    trader.order_flow_buffer.append({
        'imbalance': 0,
        'bid_volume': 10,
        'ask_volume': 10,
        'spread': 0.01
    })
    
    assert trader.generate_momentum_signal() is None
    assert trader.generate_mean_reversion_signal() is None
    assert trader.generate_order_flow_signal() is None
    
    # Test signal combination with empty list
    assert trader.combine_signals([]) is None
    
    # Test position size with zero signal
    signal = MarketSignal(
        type=SignalType.TECHNICAL,
        direction=0.0,
        strength=0.0,
        timestamp=time.time(),
        features={},
        confidence=0.0,
        horizon=5.0
    )
    assert trader.calculate_position_size(signal, price=50000) == 0.0
