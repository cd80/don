"""
Tests for market making functionality.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch
import time

from src.strategies.market_maker import (
    MarketMaker,
    MarketMakingStyle,
    OrderBook,
    MarketState
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
def market_maker(config):
    """Initialize market maker for testing."""
    return MarketMaker(
        config=config,
        exchange='binance',
        base_spread=0.001,
        min_spread=0.0005,
        max_spread=0.01,
        order_size=0.01,
        max_inventory=1.0,
        risk_aversion=1.0
    )

@pytest.fixture
def sample_orderbook():
    """Create sample orderbook data."""
    return OrderBook(
        bids={
            49900: 1.0,
            49800: 2.0,
            49700: 3.0
        },
        asks={
            50100: 1.0,
            50200: 2.0,
            50300: 3.0
        },
        timestamp=time.time(),
        mid_price=50000,
        spread=0.004,
        depth={
            'bid_depth': 6.0,
            'ask_depth': 6.0
        }
    )

@pytest.fixture
def sample_market_state():
    """Create sample market state."""
    return MarketState(
        price=50000,
        volatility=0.02,
        volume=1.0,
        trend=0.1,
        imbalance=0.0,
        timestamp=time.time()
    )

def test_initialization(market_maker):
    """Test market maker initialization."""
    assert market_maker.base_spread == 0.001
    assert market_maker.min_spread == 0.0005
    assert market_maker.max_spread == 0.01
    assert market_maker.order_size == 0.01
    assert market_maker.max_inventory == 1.0
    assert market_maker.inventory == 0.0
    assert market_maker.realized_pnl == 0.0

@pytest.mark.asyncio
async def test_fetch_market_state(market_maker):
    """Test market state fetching."""
    # Mock exchange client
    market_maker.client = Mock()
    market_maker.client.fetch_orderbook = Mock(
        return_value={
            'bids': {49900: 1.0},
            'asks': {50100: 1.0}
        }
    )
    market_maker.client.fetch_recent_trades = Mock(
        return_value=[
            {'price': 50000, 'amount': 1.0},
            {'price': 50100, 'amount': 1.0}
        ]
    )
    
    order_book, market_state = await market_maker.fetch_market_state('BTC/USDT')
    
    assert isinstance(order_book, OrderBook)
    assert isinstance(market_state, MarketState)
    assert order_book.mid_price == 50000
    assert market_state.price == 50000

def test_calculate_trend(market_maker):
    """Test trend calculation."""
    # Upward trend
    prices = [100, 101, 102, 103, 104]
    trend = market_maker.calculate_trend(prices)
    assert trend > 0
    
    # Downward trend
    prices = [104, 103, 102, 101, 100]
    trend = market_maker.calculate_trend(prices)
    assert trend < 0
    
    # No trend
    prices = [100, 100, 100, 100, 100]
    trend = market_maker.calculate_trend(prices)
    assert abs(trend) < 0.0001

def test_calculate_imbalance(market_maker, sample_orderbook):
    """Test order book imbalance calculation."""
    imbalance = market_maker.calculate_imbalance(sample_orderbook)
    assert -1 <= imbalance <= 1
    
    # Test with more bids
    sample_orderbook.depth['bid_depth'] = 10.0
    imbalance = market_maker.calculate_imbalance(sample_orderbook)
    assert imbalance > 0
    
    # Test with more asks
    sample_orderbook.depth['bid_depth'] = 6.0
    sample_orderbook.depth['ask_depth'] = 10.0
    imbalance = market_maker.calculate_imbalance(sample_orderbook)
    assert imbalance < 0

def test_calculate_optimal_spread(market_maker, sample_market_state):
    """Test optimal spread calculation."""
    spread = market_maker.calculate_optimal_spread(sample_market_state)
    
    assert market_maker.min_spread <= spread <= market_maker.max_spread
    
    # Test with high volatility
    sample_market_state.volatility = 0.05
    high_vol_spread = market_maker.calculate_optimal_spread(sample_market_state)
    assert high_vol_spread > spread
    
    # Test with high volume
    sample_market_state.volatility = 0.02
    sample_market_state.volume = 10.0
    high_vol_spread = market_maker.calculate_optimal_spread(sample_market_state)
    assert high_vol_spread < spread

def test_calculate_order_prices(market_maker, sample_market_state):
    """Test order price calculation."""
    bid_price, ask_price = market_maker.calculate_order_prices(
        sample_market_state
    )
    
    assert bid_price < sample_market_state.price
    assert ask_price > sample_market_state.price
    
    # Test with inventory skew
    market_maker.inventory = 0.5
    skewed_bid, skewed_ask = market_maker.calculate_order_prices(
        sample_market_state
    )
    
    assert skewed_bid < bid_price
    assert skewed_ask < ask_price

def test_calculate_order_sizes(market_maker, sample_market_state):
    """Test order size calculation."""
    bid_size, ask_size = market_maker.calculate_order_sizes(sample_market_state)
    
    assert bid_size > 0
    assert ask_size > 0
    
    # Test with positive inventory
    market_maker.inventory = 0.5
    skewed_bid, skewed_ask = market_maker.calculate_order_sizes(
        sample_market_state
    )
    
    assert skewed_bid > bid_size
    assert skewed_ask < ask_size
    
    # Test with negative inventory
    market_maker.inventory = -0.5
    skewed_bid, skewed_ask = market_maker.calculate_order_sizes(
        sample_market_state
    )
    
    assert skewed_bid < bid_size
    assert skewed_ask > ask_size

@pytest.mark.asyncio
async def test_place_orders(market_maker):
    """Test order placement."""
    # Mock exchange client
    market_maker.client = Mock()
    market_maker.client.create_order = Mock(
        return_value={'id': '123', 'status': 'open'}
    )
    market_maker.client.cancel_order = Mock(return_value=True)
    
    success = await market_maker.place_orders(
        symbol='BTC/USDT',
        bid_price=49900,
        ask_price=50100,
        bid_size=0.01,
        ask_size=0.01
    )
    
    assert success
    assert len(market_maker.current_orders) == 2
    assert 'bid' in market_maker.current_orders
    assert 'ask' in market_maker.current_orders

@pytest.mark.asyncio
async def test_cancel_orders(market_maker):
    """Test order cancellation."""
    # Setup test orders
    market_maker.current_orders = {
        'bid': {'id': '123', 'status': 'open'},
        'ask': {'id': '456', 'status': 'open'}
    }
    
    # Mock exchange client
    market_maker.client = Mock()
    market_maker.client.cancel_order = Mock(return_value=True)
    
    success = await market_maker.cancel_all_orders('BTC/USDT')
    
    assert success
    assert len(market_maker.current_orders) == 0

def test_update_inventory(market_maker):
    """Test inventory updates."""
    # Test buy trade
    trade = {
        'side': 'buy',
        'amount': 0.1,
        'price': 50000,
        'realized_pnl': 10,
        'order_id': '123'
    }
    market_maker.current_orders = {
        'bid': {'id': '123', 'status': 'filled'}
    }
    
    market_maker.update_inventory(trade)
    assert market_maker.inventory == 0.1
    assert market_maker.position_value == 5000
    assert market_maker.realized_pnl == 10
    
    # Test sell trade
    trade = {
        'side': 'sell',
        'amount': 0.05,
        'price': 50100,
        'realized_pnl': 5,
        'order_id': '456'
    }
    market_maker.current_orders = {
        'ask': {'id': '456', 'status': 'filled'}
    }
    
    market_maker.update_inventory(trade)
    assert market_maker.inventory == 0.05
    assert market_maker.realized_pnl == 15

def test_performance_metrics(market_maker):
    """Test performance metrics calculation."""
    # Setup test data
    market_maker.realized_pnl = 100
    market_maker.unrealized_pnl = 50
    market_maker.inventory = 0.1
    market_maker.position_value = 5000
    market_maker.trades = [1, 2, 3]  # Dummy trades
    market_maker.order_books = [
        OrderBook(
            bids={}, asks={}, timestamp=0,
            mid_price=0, spread=0.001, depth={}
        )
        for _ in range(10)
    ]
    market_maker.market_states = [
        MarketState(
            price=0, volatility=0, volume=1.0,
            trend=0, imbalance=0, timestamp=0
        )
        for _ in range(10)
    ]
    
    metrics = market_maker.get_performance_metrics()
    
    assert metrics['realized_pnl'] == 100
    assert metrics['unrealized_pnl'] == 50
    assert metrics['total_pnl'] == 150
    assert metrics['inventory'] == 0.1
    assert metrics['position_value'] == 5000
    assert metrics['num_trades'] == 3

@pytest.mark.asyncio
async def test_run(market_maker, sample_orderbook, sample_market_state):
    """Test strategy running."""
    # Mock exchange client and methods
    market_maker.client = Mock()
    market_maker.client.fetch_orderbook = Mock(
        return_value={
            'bids': sample_orderbook.bids,
            'asks': sample_orderbook.asks
        }
    )
    market_maker.client.fetch_recent_trades = Mock(
        return_value=[
            {'price': 50000, 'amount': 1.0},
            {'price': 50100, 'amount': 1.0}
        ]
    )
    market_maker.client.create_order = Mock(
        return_value={'id': '123', 'status': 'open'}
    )
    market_maker.client.cancel_order = Mock(return_value=True)
    
    # Run strategy for a short time
    async def stop_after_delay():
        await asyncio.sleep(0.1)
        raise KeyboardInterrupt
    
    with pytest.raises(KeyboardInterrupt):
        await asyncio.gather(
            market_maker.run('BTC/USDT', interval=0.01),
            stop_after_delay()
        )
    
    # Check that orders were placed
    assert len(market_maker.order_books) > 0
    assert len(market_maker.market_states) > 0

def test_edge_cases(market_maker):
    """Test edge cases."""
    # Test with zero prices
    state = MarketState(
        price=0,
        volatility=0,
        volume=0,
        trend=0,
        imbalance=0,
        timestamp=0
    )
    spread = market_maker.calculate_optimal_spread(state)
    assert spread >= market_maker.min_spread
    
    # Test with extreme inventory
    market_maker.inventory = market_maker.max_inventory * 2
    bid_size, ask_size = market_maker.calculate_order_sizes(state)
    assert bid_size > 0
    assert ask_size > 0
    
    # Test with extreme volatility
    state.volatility = 1.0
    spread = market_maker.calculate_optimal_spread(state)
    assert spread <= market_maker.max_spread
