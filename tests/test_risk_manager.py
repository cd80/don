"""
Tests for dynamic risk management functionality.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.risk.risk_manager import (
    DynamicRiskManager,
    MarketRegime,
    RiskMetrics
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'var_limit': 0.02,
        'drawdown_limit': 0.20,
        'volatility_limit': 0.03,
        'volatility_window': 20,
        'trend_window': 50,
        'risk_free_rate': 0.0
    }

@pytest.fixture
def risk_manager(config):
    """Initialize risk manager for testing."""
    return DynamicRiskManager(
        config=config,
        initial_capital=100000,
        max_position_size=1.0,
        max_leverage=3.0,
        confidence_level=0.95
    )

@pytest.fixture
def sample_data():
    """Create sample market data."""
    np.random.seed(42)
    n_points = 200
    
    # Generate prices with trend and volatility regimes
    prices = np.zeros(n_points)
    prices[0] = 100
    
    # Add trend
    trend = np.linspace(0, 0.5, n_points)
    
    # Add volatility regimes
    volatility = np.ones(n_points) * 0.02
    volatility[50:100] *= 2  # High volatility regime
    volatility[150:] *= 0.5  # Low volatility regime
    
    # Generate price series
    for i in range(1, n_points):
        prices[i] = prices[i-1] * (1 + np.random.normal(trend[i], volatility[i]))
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    return prices, returns

def test_risk_metrics_calculation(risk_manager, sample_data):
    """Test risk metrics calculation."""
    _, returns = sample_data
    metrics = risk_manager.calculate_risk_metrics(returns, model_confidence=0.8)
    
    assert isinstance(metrics, RiskMetrics)
    assert metrics.var > 0
    assert metrics.es > metrics.var
    assert 0 <= metrics.drawdown <= 1
    assert isinstance(metrics.market_regime, MarketRegime)

def test_market_regime_detection(risk_manager, sample_data):
    """Test market regime detection."""
    prices, returns = sample_data
    
    # Test different market conditions
    regimes = []
    for i in range(risk_manager.trend_window, len(prices), 20):
        regime = risk_manager.detect_market_regime(
            prices[:i],
            returns[:i-1]
        )
        regimes.append(regime)
        assert isinstance(regime, MarketRegime)
    
    # Ensure we detect different regimes
    assert len(set(regimes)) > 1

def test_position_size_calculation(risk_manager):
    """Test position size calculation."""
    # Create test risk metrics
    metrics = RiskMetrics(
        var=0.01,
        es=0.015,
        volatility=0.02,
        drawdown=0.1,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=1.0,
        model_confidence=0.8,
        market_regime=MarketRegime.MEDIUM_VOLATILITY
    )
    
    # Test position sizing
    position = risk_manager.calculate_position_size(
        prediction=0.5,
        confidence=0.8,
        risk_metrics=metrics
    )
    
    assert isinstance(position, float)
    assert -risk_manager.max_leverage <= position <= risk_manager.max_leverage

def test_position_updates(risk_manager):
    """Test position updating and P&L tracking."""
    # Update position
    metrics = risk_manager.update_position(0.5, 100)
    assert metrics['position_size'] == 0.5
    assert metrics['capital'] == risk_manager.initial_capital
    
    # Update with profit
    metrics = risk_manager.update_position(0.7, 105)
    assert metrics['position_size'] == 0.7
    assert metrics['capital'] > risk_manager.initial_capital
    
    # Update with loss
    metrics = risk_manager.update_position(0.3, 95)
    assert metrics['position_size'] == 0.3
    assert 'total_pnl' in metrics
    assert 'return' in metrics

def test_risk_limits_adjustment(risk_manager):
    """Test dynamic risk limit adjustments."""
    initial_limits = risk_manager.get_risk_limits()
    
    # Test adjustment with good performance
    risk_manager.adjust_risk_limits({
        'sharpe_ratio': 2.5,
        'calmar_ratio': 2.0,
        'volatility': 0.01
    })
    
    new_limits = risk_manager.get_risk_limits()
    assert new_limits['var_limit'] > initial_limits['var_limit']
    assert new_limits['max_position_size'] >= initial_limits['max_position_size']
    
    # Test adjustment with poor performance
    risk_manager.adjust_risk_limits({
        'sharpe_ratio': 0.2,
        'calmar_ratio': 0.3,
        'volatility': 0.04
    })
    
    final_limits = risk_manager.get_risk_limits()
    assert final_limits['var_limit'] < new_limits['var_limit']
    assert final_limits['max_position_size'] <= new_limits['max_position_size']

def test_risk_metrics_validation(risk_manager, sample_data):
    """Test risk metrics validation."""
    _, returns = sample_data
    metrics = risk_manager.calculate_risk_metrics(returns, model_confidence=0.8)
    
    # VaR should be positive and greater than mean loss
    assert metrics.var > 0
    assert metrics.var > -np.mean(returns[returns < 0])
    
    # Expected Shortfall should be greater than VaR
    assert metrics.es > metrics.var
    
    # Volatility should be positive
    assert metrics.volatility > 0
    
    # Drawdown should be between 0 and 1
    assert 0 <= metrics.drawdown <= 1

def test_position_size_limits(risk_manager):
    """Test position size limits and leverage constraints."""
    metrics = RiskMetrics(
        var=0.01,
        es=0.015,
        volatility=0.02,
        drawdown=0.1,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=1.0,
        model_confidence=0.8,
        market_regime=MarketRegime.MEDIUM_VOLATILITY
    )
    
    # Test maximum long position
    position = risk_manager.calculate_position_size(1.0, 1.0, metrics)
    assert position <= risk_manager.max_leverage
    
    # Test maximum short position
    position = risk_manager.calculate_position_size(-1.0, 1.0, metrics)
    assert position >= -risk_manager.max_leverage

def test_market_regime_transitions(risk_manager, sample_data):
    """Test market regime transitions."""
    prices, returns = sample_data
    
    regimes = []
    for i in range(risk_manager.trend_window, len(prices)-20, 20):
        # Create different market conditions
        if i < len(prices) // 3:
            # Trending market
            prices[i:i+20] *= np.linspace(1, 1.1, 20)
        elif i < 2 * len(prices) // 3:
            # Volatile market
            prices[i:i+20] *= np.random.normal(1, 0.02, 20)
        else:
            # Range-bound market
            prices[i:i+20] *= np.sin(np.linspace(0, np.pi, 20)) * 0.02 + 1
        
        regime = risk_manager.detect_market_regime(
            prices[:i+20],
            returns[:i+19]
        )
        regimes.append(regime)
    
    # Check regime transitions
    regime_transitions = [(regimes[i], regimes[i+1])
                         for i in range(len(regimes)-1)]
    assert len(set(regime_transitions)) > 1

def test_risk_adjusted_returns(risk_manager):
    """Test risk-adjusted return calculations."""
    # Simulate a series of trades
    prices = [100, 102, 98, 103, 101, 105]
    positions = [0.5, 0.7, 0.3, 0.6, 0.4, 0.0]
    
    metrics = []
    for price, position in zip(prices[1:], positions[1:]):
        metric = risk_manager.update_position(position, price)
        metrics.append(metric)
    
    # Verify metrics
    assert all('return' in m for m in metrics)
    assert all('total_pnl' in m for m in metrics)
    assert metrics[-1]['capital'] != risk_manager.initial_capital

@pytest.mark.parametrize("market_regime", list(MarketRegime))
def test_position_sizing_by_regime(risk_manager, market_regime):
    """Test position sizing for different market regimes."""
    metrics = RiskMetrics(
        var=0.01,
        es=0.015,
        volatility=0.02,
        drawdown=0.1,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=1.0,
        model_confidence=0.8,
        market_regime=market_regime
    )
    
    position = risk_manager.calculate_position_size(0.5, 0.8, metrics)
    
    # Position size should be adjusted based on regime
    if market_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BREAKOUT]:
        assert abs(position) < 0.5  # More conservative
    elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
        assert abs(position) > 0.3  # More aggressive
