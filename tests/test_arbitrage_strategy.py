"""
Tests for arbitrage strategy functionality.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch

from src.strategies.arbitrage_strategy import (
    ArbitrageStrategy,
    ArbitrageType,
    ArbitrageOpportunity
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'data': {
            'binance': {'api_key': 'test', 'api_secret': 'test'},
            'kraken': {'api_key': 'test', 'api_secret': 'test'},
            'coinbase': {'api_key': 'test', 'api_secret': 'test'}
        },
        'max_order_time': 60
    }

@pytest.fixture
def exchanges():
    """Test exchanges."""
    return ['binance', 'kraken', 'coinbase']

@pytest.fixture
def strategy(config, exchanges):
    """Initialize strategy for testing."""
    return ArbitrageStrategy(
        config=config,
        exchanges=exchanges,
        min_profit_threshold=0.001,
        max_position_size=1.0
    )

@pytest.fixture
def sample_orderbooks():
    """Create sample orderbook data."""
    def create_orderbook(base_price, spread):
        return pd.DataFrame({
            'bids': {
                base_price * (1 - spread/2): 1.0,
                base_price * (1 - spread): 2.0
            },
            'asks': {
                base_price * (1 + spread/2): 1.0,
                base_price * (1 + spread): 2.0
            }
        })
    
    return {
        'binance': create_orderbook(50000, 0.001),
        'kraken': create_orderbook(50100, 0.001),
        'coinbase': create_orderbook(49900, 0.001)
    }

@pytest.mark.asyncio
async def test_fetch_orderbooks(strategy, sample_orderbooks):
    """Test orderbook fetching."""
    # Mock exchange clients
    for exchange in strategy.exchanges:
        strategy.exchange_clients[exchange] = Mock()
        strategy.exchange_clients[exchange].fetch_orderbook = Mock(
            return_value=sample_orderbooks[exchange]
        )
    
    orderbooks = await strategy.fetch_orderbooks('BTC/USDT')
    
    assert len(orderbooks) == len(strategy.exchanges)
    for exchange in strategy.exchanges:
        assert exchange in orderbooks
        assert isinstance(orderbooks[exchange], pd.DataFrame)

def test_calculate_effective_prices(strategy, sample_orderbooks):
    """Test effective price calculation."""
    orderbook = sample_orderbooks['binance']
    
    # Test with valid volume
    bid_price, ask_price = strategy.calculate_effective_prices(orderbook, 0.5)
    assert bid_price > 0
    assert ask_price > 0
    assert ask_price > bid_price
    
    # Test with excessive volume
    bid_price, ask_price = strategy.calculate_effective_prices(orderbook, 10.0)
    assert bid_price == 0
    assert ask_price == 0

def test_identify_opportunities(strategy, sample_orderbooks):
    """Test arbitrage opportunity identification."""
    opportunities = strategy.identify_opportunities(sample_orderbooks, 0.1)
    
    assert isinstance(opportunities, list)
    for opp in opportunities:
        assert isinstance(opp, ArbitrageOpportunity)
        assert opp.net_profit > strategy.min_profit_threshold
        assert opp.volume_a <= strategy.max_position_size
        assert opp.volume_b <= strategy.max_position_size

def test_calculate_risk_metrics(strategy):
    """Test risk metrics calculation."""
    metrics = strategy.calculate_risk_metrics(50000, 50100, 0.1)
    
    assert isinstance(metrics, dict)
    assert 'price_ratio' in metrics
    assert 'execution_time_risk' in metrics
    assert 'slippage_risk' in metrics
    assert 'liquidity_risk' in metrics
    assert 'counterparty_risk' in metrics
    
    assert metrics['price_ratio'] > 1.0
    assert all(v >= 0 for v in metrics.values())

def test_filter_opportunities(strategy):
    """Test opportunity filtering."""
    opportunities = [
        ArbitrageOpportunity(
            exchange_a='binance',
            exchange_b='kraken',
            symbol='BTC/USDT',
            price_a=50000,
            price_b=50100,
            spread=100,
            timestamp=1234567890,
            volume_a=0.1,
            volume_b=0.1,
            estimated_profit=10,
            transaction_costs=1,
            net_profit=9,
            risk_metrics={
                'execution_time_risk': 0.1,
                'slippage_risk': 0.1,
                'liquidity_risk': 0.1
            }
        ),
        ArbitrageOpportunity(
            exchange_a='kraken',
            exchange_b='coinbase',
            symbol='BTC/USDT',
            price_a=50000,
            price_b=50010,
            spread=10,
            timestamp=1234567890,
            volume_a=0.1,
            volume_b=0.1,
            estimated_profit=1,
            transaction_costs=0.5,
            net_profit=0.5,
            risk_metrics={
                'execution_time_risk': 0.6,  # Too high
                'slippage_risk': 0.1,
                'liquidity_risk': 0.1
            }
        )
    ]
    
    filtered = strategy.filter_opportunities(opportunities)
    
    assert len(filtered) == 1
    assert filtered[0].exchange_a == 'binance'
    assert filtered[0].exchange_b == 'kraken'

@pytest.mark.asyncio
async def test_execute_arbitrage(strategy):
    """Test arbitrage execution."""
    opportunity = ArbitrageOpportunity(
        exchange_a='binance',
        exchange_b='kraken',
        symbol='BTC/USDT',
        price_a=50000,
        price_b=50100,
        spread=100,
        timestamp=1234567890,
        volume_a=0.1,
        volume_b=0.1,
        estimated_profit=10,
        transaction_costs=1,
        net_profit=9,
        risk_metrics={
            'execution_time_risk': 0.1,
            'slippage_risk': 0.1,
            'liquidity_risk': 0.1
        }
    )
    
    # Mock exchange clients
    for exchange in strategy.exchanges:
        strategy.exchange_clients[exchange] = Mock()
        strategy.exchange_clients[exchange].create_order = Mock(
            return_value={'id': '123', 'symbol': 'BTC/USDT'}
        )
    
    success = await strategy.execute_arbitrage(opportunity)
    
    assert success
    assert len(strategy.active_positions) == 2
    assert opportunity in strategy.opportunities

@pytest.mark.asyncio
async def test_monitor_positions(strategy):
    """Test position monitoring."""
    # Setup test positions
    strategy.active_positions = {
        'binance': {
            'symbol': 'BTC/USDT',
            'id': '123',
            'timestamp': 1234567890
        },
        'kraken': {
            'symbol': 'BTC/USDT',
            'id': '456',
            'timestamp': time.time()
        }
    }
    
    # Mock exchange clients
    for exchange in strategy.exchanges:
        strategy.exchange_clients[exchange] = Mock()
        strategy.exchange_clients[exchange].get_order = Mock(
            return_value={'status': 'filled', 'timestamp': time.time()}
        )
    
    await strategy.monitor_positions()
    
    assert len(strategy.active_positions) == 0

def test_performance_metrics(strategy):
    """Test performance metrics calculation."""
    strategy.opportunities = [
        ArbitrageOpportunity(
            exchange_a='binance',
            exchange_b='kraken',
            symbol='BTC/USDT',
            price_a=50000,
            price_b=50100,
            spread=100,
            timestamp=1234567890,
            volume_a=0.1,
            volume_b=0.1,
            estimated_profit=10,
            transaction_costs=1,
            net_profit=9,
            risk_metrics={}
        ),
        ArbitrageOpportunity(
            exchange_a='kraken',
            exchange_b='coinbase',
            symbol='BTC/USDT',
            price_a=50000,
            price_b=50050,
            spread=50,
            timestamp=1234567890,
            volume_a=0.1,
            volume_b=0.1,
            estimated_profit=5,
            transaction_costs=1,
            net_profit=4,
            risk_metrics={}
        )
    ]
    
    metrics = strategy.get_performance_metrics()
    
    assert isinstance(metrics, dict)
    assert metrics['total_profit'] == 13
    assert metrics['num_trades'] == 2
    assert metrics['success_rate'] == 1.0

@pytest.mark.asyncio
async def test_run(strategy, sample_orderbooks):
    """Test strategy running."""
    # Mock exchange clients and methods
    for exchange in strategy.exchanges:
        strategy.exchange_clients[exchange] = Mock()
        strategy.exchange_clients[exchange].fetch_orderbook = Mock(
            return_value=sample_orderbooks[exchange]
        )
        strategy.exchange_clients[exchange].create_order = Mock(
            return_value={'id': '123', 'symbol': 'BTC/USDT'}
        )
        strategy.exchange_clients[exchange].get_order = Mock(
            return_value={'status': 'filled', 'timestamp': time.time()}
        )
    
    # Run strategy for a short time
    async def stop_after_delay():
        await asyncio.sleep(0.1)
        raise KeyboardInterrupt
    
    with pytest.raises(KeyboardInterrupt):
        await asyncio.gather(
            strategy.run('BTC/USDT', 0.1, interval=0.01),
            stop_after_delay()
        )
    
    # Check that opportunities were found and executed
    assert len(strategy.opportunities) > 0
    assert len(strategy.active_positions) == 0

def test_edge_cases(strategy):
    """Test edge cases."""
    # Test with empty orderbooks
    empty_orderbooks = {
        exchange: pd.DataFrame({'bids': {}, 'asks': {}})
        for exchange in strategy.exchanges
    }
    opportunities = strategy.identify_opportunities(empty_orderbooks, 0.1)
    assert len(opportunities) == 0
    
    # Test with invalid prices
    invalid_orderbook = pd.DataFrame({
        'bids': {0: 1.0},
        'asks': {float('inf'): 1.0}
    })
    bid_price, ask_price = strategy.calculate_effective_prices(
        invalid_orderbook,
        0.1
    )
    assert bid_price == 0
    assert ask_price == 0
    
    # Test with no profitable opportunities
    unprofitable_orderbooks = {
        exchange: pd.DataFrame({
            'bids': {100: 1.0},
            'asks': {100: 1.0}
        })
        for exchange in strategy.exchanges
    }
    opportunities = strategy.identify_opportunities(unprofitable_orderbooks, 0.1)
    assert len(opportunities) == 0
