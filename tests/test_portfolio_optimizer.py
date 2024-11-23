"""
Tests for portfolio optimization functionality.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.portfolio.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationStrategy,
    PortfolioMetrics
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'covariance_method': 'ledoit_wolf',
        'target_return': 0.1,
        'risk_aversion': 1.0,
        'max_turnover': 0.2,
        'decay_factor': 0.94,
        'bl_tau': 0.05
    }

@pytest.fixture
def optimizer(config):
    """Initialize optimizer for testing."""
    return PortfolioOptimizer(
        config=config,
        risk_free_rate=0.0,
        transaction_costs=0.001,
        min_weight=0.0,
        max_weight=1.0
    )

@pytest.fixture
def sample_data():
    """Create sample market data."""
    np.random.seed(42)
    n_assets = 5
    n_observations = 252
    
    # Generate correlated returns
    correlation = np.array([
        [1.0, 0.5, 0.3, 0.2, 0.1],
        [0.5, 1.0, 0.4, 0.3, 0.2],
        [0.3, 0.4, 1.0, 0.5, 0.3],
        [0.2, 0.3, 0.5, 1.0, 0.4],
        [0.1, 0.2, 0.3, 0.4, 1.0]
    ])
    
    # Generate returns with different means and volatilities
    means = np.array([0.1, 0.08, 0.12, 0.07, 0.09])
    vols = np.array([0.2, 0.15, 0.25, 0.18, 0.22])
    
    L = np.linalg.cholesky(correlation)
    uncorrelated_returns = np.random.normal(0, 1, (n_observations, n_assets))
    correlated_returns = uncorrelated_returns @ L.T
    
    # Scale returns by volatility and add mean
    returns = (correlated_returns * vols) + (means / 252)
    
    return returns

def test_covariance_estimation(optimizer, sample_data):
    """Test covariance matrix estimation methods."""
    methods = ['sample', 'ledoit_wolf', 'exponential']
    
    for method in methods:
        cov = optimizer.estimate_covariance(sample_data, method=method)
        assert isinstance(cov, np.ndarray)
        assert cov.shape == (sample_data.shape[1], sample_data.shape[1])
        assert np.allclose(cov, cov.T)  # Symmetry
        assert np.all(np.linalg.eigvals(cov) > 0)  # Positive definite

def test_mean_variance_optimization(optimizer, sample_data):
    """Test mean-variance optimization."""
    covariance = optimizer.estimate_covariance(sample_data)
    weights = optimizer.mean_variance_optimization(sample_data, covariance)
    
    assert isinstance(weights, np.ndarray)
    assert len(weights) == sample_data.shape[1]
    assert np.isclose(weights.sum(), 1.0)  # Sum to 1
    assert np.all(weights >= optimizer.min_weight)  # Min weight constraint
    assert np.all(weights <= optimizer.max_weight)  # Max weight constraint

def test_risk_parity_optimization(optimizer, sample_data):
    """Test risk parity optimization."""
    covariance = optimizer.estimate_covariance(sample_data)
    weights = optimizer.risk_parity_optimization(covariance)
    
    assert isinstance(weights, np.ndarray)
    assert len(weights) == sample_data.shape[1]
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= optimizer.min_weight)
    assert np.all(weights <= optimizer.max_weight)
    
    # Check risk contribution equality
    portfolio_risk = np.sqrt(weights.T @ covariance @ weights)
    risk_contributions = (covariance @ weights) * weights / portfolio_risk
    assert np.allclose(risk_contributions, risk_contributions[0], rtol=0.1)

def test_hierarchical_risk_parity(optimizer, sample_data):
    """Test hierarchical risk parity optimization."""
    weights = optimizer.hierarchical_risk_parity(sample_data)
    
    assert isinstance(weights, np.ndarray)
    assert len(weights) == sample_data.shape[1]
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)  # Non-negative weights

def test_black_litterman_optimization(optimizer, sample_data):
    """Test Black-Litterman optimization."""
    n_assets = sample_data.shape[1]
    market_caps = np.array([100, 80, 120, 90, 110])
    
    views = {
        (0, 1): 0.02,  # Asset 0 will outperform Asset 1 by 2%
        (2, 3): 0.01   # Asset 2 will outperform Asset 3 by 1%
    }
    view_confidences = {
        (0, 1): 0.6,
        (2, 3): 0.7
    }
    
    weights = optimizer.black_litterman_optimization(
        sample_data,
        market_caps,
        views,
        view_confidences
    )
    
    assert isinstance(weights, np.ndarray)
    assert len(weights) == n_assets
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= optimizer.min_weight)
    assert np.all(weights <= optimizer.max_weight)

def test_portfolio_metrics(optimizer, sample_data):
    """Test portfolio metrics calculation."""
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
    covariance = optimizer.estimate_covariance(sample_data)
    
    metrics = optimizer.calculate_portfolio_metrics(
        weights,
        sample_data,
        covariance
    )
    
    assert isinstance(metrics, PortfolioMetrics)
    assert metrics.volatility > 0
    assert metrics.diversification_ratio > 0
    assert 0 <= metrics.concentration <= 1
    assert metrics.turnover == 0  # No previous weights

def test_optimization_strategies(optimizer, sample_data):
    """Test all optimization strategies."""
    for strategy in OptimizationStrategy:
        if strategy == OptimizationStrategy.BLACK_LITTERMAN:
            # Skip BL as it requires additional parameters
            continue
            
        metrics = optimizer.optimize_portfolio(
            sample_data,
            strategy
        )
        
        assert isinstance(metrics, PortfolioMetrics)
        assert np.isclose(metrics.weights.sum(), 1.0)
        assert np.all(metrics.weights >= optimizer.min_weight)
        assert np.all(metrics.weights <= optimizer.max_weight)

def test_turnover_constraint(optimizer, sample_data):
    """Test turnover constraint in optimization."""
    current_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    
    metrics = optimizer.optimize_portfolio(
        sample_data,
        OptimizationStrategy.MEAN_VARIANCE,
        current_weights=current_weights
    )
    
    assert metrics.turnover <= optimizer.config['max_turnover']

def test_risk_metrics(optimizer, sample_data):
    """Test risk metrics calculation."""
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    covariance = optimizer.estimate_covariance(sample_data)
    
    metrics = optimizer.calculate_portfolio_metrics(
        weights,
        sample_data,
        covariance
    )
    
    # VaR should be positive
    assert metrics.var > 0
    
    # Expected Shortfall should be greater than VaR
    assert metrics.es > metrics.var
    
    # Sharpe ratio should be reasonable
    assert -10 < metrics.sharpe_ratio < 10

def test_parameter_validation(optimizer):
    """Test parameter validation."""
    with pytest.raises(ValueError):
        optimizer.estimate_covariance(np.random.randn(100, 5), method='invalid')
    
    with pytest.raises(ValueError):
        optimizer.optimize_portfolio(
            np.random.randn(100, 5),
            'invalid_strategy'
        )

@pytest.mark.parametrize("risk_aversion", [0.5, 1.0, 2.0])
def test_risk_aversion_impact(optimizer, sample_data, risk_aversion):
    """Test impact of risk aversion parameter."""
    optimizer.risk_aversion = risk_aversion
    metrics = optimizer.optimize_portfolio(
        sample_data,
        OptimizationStrategy.MEAN_VARIANCE
    )
    
    # Higher risk aversion should lead to lower volatility
    if risk_aversion > 1.0:
        assert metrics.volatility < 0.3  # Arbitrary threshold
    
def test_constraints_satisfaction(optimizer, sample_data):
    """Test that optimization satisfies all constraints."""
    metrics = optimizer.optimize_portfolio(
        sample_data,
        OptimizationStrategy.MEAN_VARIANCE
    )
    
    # Sum to 1 constraint
    assert np.isclose(metrics.weights.sum(), 1.0)
    
    # Weight bounds
    assert np.all(metrics.weights >= optimizer.min_weight)
    assert np.all(metrics.weights <= optimizer.max_weight)
    
    # Non-negative weights
    assert np.all(metrics.weights >= 0)

def test_optimization_stability(optimizer, sample_data):
    """Test optimization stability across multiple runs."""
    results = []
    for _ in range(5):
        metrics = optimizer.optimize_portfolio(
            sample_data,
            OptimizationStrategy.MEAN_VARIANCE
        )
        results.append(metrics.weights)
    
    # Check that results are consistent
    for i in range(1, len(results)):
        assert np.allclose(results[0], results[i], rtol=1e-5)
