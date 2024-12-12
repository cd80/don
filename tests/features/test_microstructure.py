"""Unit tests for market microstructure features."""

import pytest
import pandas as pd
import numpy as np
from don.features.microstructure import MarketMicrostructureFeatures


def test_order_imbalance():
    """Test order book imbalance calculation."""
    calculator = MarketMicrostructureFeatures()

    # Test balanced orderbook
    bids = pd.DataFrame({
        'price': [100.0, 99.0],
        'quantity': [1.0, 1.0]
    })
    asks = pd.DataFrame({
        'price': [101.0, 102.0],
        'quantity': [1.0, 1.0]
    })
    imbalance = calculator._calculate_order_imbalance(bids, asks)
    assert abs(imbalance) < 1e-6  # Should be close to 0

    # Test buy pressure
    bids['quantity'] = [2.0, 2.0]
    imbalance = calculator._calculate_order_imbalance(bids, asks)
    assert imbalance > 0  # Should indicate buying pressure

    # Test sell pressure
    asks['quantity'] = [3.0, 3.0]
    imbalance = calculator._calculate_order_imbalance(bids, asks)
    assert imbalance < 0  # Should indicate selling pressure

    # Test empty orderbook
    empty_df = pd.DataFrame({'price': [], 'quantity': []})
    imbalance = calculator._calculate_order_imbalance(empty_df, empty_df)
    assert imbalance == 0  # Should handle empty orderbook


def test_trade_flow():
    """Test trade flow imbalance calculation."""
    calculator = MarketMicrostructureFeatures()

    # Create sample trade data
    trades = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
        'quantity': [1.0, 2.0, 1.0, 2.0, 1.0],
        'is_buyer_maker': [True, False, True, False, True]
    }).set_index('timestamp')

    result = calculator._calculate_trade_flow(trades, window=3)

    # Basic checks
    assert isinstance(result, pd.Series)
    assert len(result) == len(trades)
    assert all((-1 <= x <= 1) or np.isnan(x) for x in result)

    # Test empty trades
    empty_trades = pd.DataFrame(columns=['quantity', 'is_buyer_maker'])
    empty_result = calculator._calculate_trade_flow(empty_trades)
    assert len(empty_result) == 0


def test_realized_volatility():
    """Test realized volatility calculation."""
    calculator = MarketMicrostructureFeatures()

    # Create sample price data
    prices = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0],
                      index=pd.date_range('2024-01-01', periods=5, freq='1min'))

    result = calculator._calculate_realized_volatility(prices, window=3)

    # Basic checks
    assert isinstance(result, pd.Series)
    assert len(result) == len(prices)
    assert all(x >= 0 or np.isnan(x) for x in result)  # Volatility should be non-negative

    # Test constant prices (zero volatility)
    constant_prices = pd.Series([100.0] * 5,
                              index=pd.date_range('2024-01-01', periods=5, freq='1min'))
    const_result = calculator._calculate_realized_volatility(constant_prices, window=3)
    assert all(x == 0 or np.isnan(x) for x in const_result)


def test_liquidation_impact():
    """Test liquidation impact calculation."""
    calculator = MarketMicrostructureFeatures()

    # Create sample liquidation data
    liquidations = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
        'quantity': [1.0, 2.0, 1.0, 2.0, 1.0],
        'side': ['long', 'short', 'long', 'short', 'long']
    }).set_index('timestamp')

    result = calculator._calculate_liquidation_impact(liquidations, window=3)

    # Check return type and keys
    assert isinstance(result, dict)
    assert all(k in result for k in ['long_liquidation_volume',
                                   'short_liquidation_volume',
                                   'liquidation_imbalance'])

    # Basic checks for each series
    for series in result.values():
        assert isinstance(series, pd.Series)
        assert len(series) == len(liquidations)

    # Test empty liquidations
    empty_liq = pd.DataFrame(columns=['quantity', 'side'])
    empty_result = calculator._calculate_liquidation_impact(empty_liq)
    assert all(len(series) == 0 for series in empty_result.values())


def test_calculate_all_features():
    """Test calculation of all features together."""
    calculator = MarketMicrostructureFeatures()

    # Create sample market data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
        'side': ['bid', 'ask', 'bid', 'ask', 'bid'],
        'price': [100.0, 101.0, 99.0, 102.0, 98.0],
        'quantity': [1.0, 1.0, 2.0, 1.0, 2.0],
        'is_buyer_maker': [True, False, True, False, True],
        'close': [100.0, 101.0, 99.0, 102.0, 98.0]
    }).set_index('timestamp')

    result = calculator.calculate(data)

    # Check that all features are present
    expected_columns = [
        'order_imbalance',
        'trade_flow_imbalance',
        'realized_volatility',
        'long_liquidation_volume',
        'short_liquidation_volume',
        'liquidation_imbalance'
    ]

    assert all(col in result.columns for col in expected_columns)
    assert len(result) == len(data)  # Should preserve length
