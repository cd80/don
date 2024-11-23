"""
Tests for enhanced evaluation functionality.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.evaluation.enhanced_evaluator import (
    EnhancedEvaluator,
    EvaluationMetric,
    PerformanceMetrics
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'transaction_cost': 0.001,
        'evaluation': {
            'window_size': 252,
            'confidence_level': 0.95
        }
    }

@pytest.fixture
def sample_data():
    """Create sample market data."""
    np.random.seed(42)
    n_points = 1000
    
    # Generate prices with trend and volatility
    prices = np.zeros(n_points)
    prices[0] = 100
    returns = np.random.normal(0.0001, 0.02, n_points-1)
    
    for i in range(1, n_points):
        prices[i] = prices[i-1] * (1 + returns[i-1])
    
    # Generate predictions and positions
    predictions = np.random.randn(n_points)
    positions = np.sign(predictions)
    
    # Generate benchmark returns
    benchmark_returns = np.random.normal(0.0002, 0.015, n_points-1)
    
    return prices, returns, predictions, positions, benchmark_returns

@pytest.fixture
def evaluator(config, sample_data):
    """Initialize evaluator with sample data."""
    _, _, _, _, benchmark_returns = sample_data
    return EnhancedEvaluator(
        config=config,
        risk_free_rate=0.0,
        benchmark_returns=benchmark_returns
    )

def test_returns_calculation(evaluator, sample_data):
    """Test strategy returns calculation."""
    prices, returns, _, positions, _ = sample_data
    
    # Calculate returns with and without costs
    strategy_returns = evaluator.calculate_returns(
        positions[:-1],
        returns,
        include_costs=True
    )
    strategy_returns_no_costs = evaluator.calculate_returns(
        positions[:-1],
        returns,
        include_costs=False
    )
    
    assert len(strategy_returns) == len(returns)
    assert np.all(strategy_returns <= strategy_returns_no_costs)

def test_metrics_calculation(evaluator, sample_data):
    """Test performance metrics calculation."""
    prices, returns, _, positions, _ = sample_data
    strategy_returns = evaluator.calculate_returns(positions[:-1], returns)
    
    metrics = evaluator.calculate_metrics(strategy_returns, positions[:-1])
    
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    
    # Verify metric values
    assert -1 <= metrics['total_return'] <= np.inf
    assert -np.inf <= metrics['sharpe_ratio'] <= np.inf
    assert -1 <= metrics['max_drawdown'] <= 0
    assert 0 <= metrics['win_rate'] <= 1

def test_rolling_metrics(evaluator, sample_data):
    """Test rolling metrics calculation."""
    prices, returns, _, positions, _ = sample_data
    strategy_returns = evaluator.calculate_returns(positions[:-1], returns)
    
    rolling_metrics = evaluator.calculate_rolling_metrics(
        strategy_returns,
        window=100
    )
    
    assert isinstance(rolling_metrics, pd.DataFrame)
    assert 'return' in rolling_metrics.columns
    assert 'volatility' in rolling_metrics.columns
    assert 'sharpe_ratio' in rolling_metrics.columns
    assert 'drawdown' in rolling_metrics.columns

def test_trade_analysis(evaluator, sample_data):
    """Test trade analysis."""
    prices, returns, _, positions, _ = sample_data
    strategy_returns = evaluator.calculate_returns(positions[:-1], returns)
    
    analysis = evaluator.analyze_trades(strategy_returns, positions[:-1])
    
    assert isinstance(analysis, dict)
    assert 'num_trades' in analysis
    assert 'avg_trade_return' in analysis
    assert 'win_rate' in analysis
    assert analysis['num_trades'] >= 0
    assert -np.inf <= analysis['avg_trade_return'] <= np.inf

def test_risk_metrics(evaluator, sample_data):
    """Test risk metrics calculation."""
    prices, returns, _, positions, _ = sample_data
    strategy_returns = evaluator.calculate_returns(positions[:-1], returns)
    
    risk_metrics = evaluator.calculate_risk_metrics(
        strategy_returns,
        positions[:-1]
    )
    
    assert isinstance(risk_metrics, dict)
    assert 'var_95' in risk_metrics
    assert 'es_95' in risk_metrics
    assert 'time_in_market' in risk_metrics
    assert risk_metrics['var_95'] <= 0
    assert 0 <= risk_metrics['time_in_market'] <= 1

def test_full_evaluation(evaluator, sample_data):
    """Test full evaluation process."""
    prices, returns, predictions, positions, _ = sample_data
    
    performance = evaluator.evaluate(predictions, returns, prices)
    
    assert isinstance(performance, PerformanceMetrics)
    assert isinstance(performance.returns, np.ndarray)
    assert isinstance(performance.metrics, dict)
    assert isinstance(performance.rolling_metrics, pd.DataFrame)
    assert isinstance(performance.trade_analysis, dict)
    assert isinstance(performance.risk_metrics, dict)

def test_plotting(evaluator, sample_data, tmp_path):
    """Test results plotting."""
    prices, returns, predictions, positions, _ = sample_data
    performance = evaluator.evaluate(predictions, returns, prices)
    
    # Test plotting to file
    plot_path = tmp_path / "evaluation_plots.png"
    evaluator.plot_results(performance, str(plot_path))
    assert plot_path.exists()
    
    # Test plotting to display
    evaluator.plot_results(performance)
    plt.close()

def test_report_generation(evaluator, sample_data, tmp_path):
    """Test report generation."""
    prices, returns, predictions, positions, _ = sample_data
    performance = evaluator.evaluate(predictions, returns, prices)
    
    # Test report string generation
    report = evaluator.generate_report(performance)
    assert isinstance(report, str)
    assert "Performance Report" in report
    
    # Test report file generation
    report_path = tmp_path / "evaluation_report.txt"
    evaluator.generate_report(performance, str(report_path))
    assert report_path.exists()

def test_benchmark_comparison(evaluator, sample_data):
    """Test benchmark comparison metrics."""
    prices, returns, predictions, positions, _ = sample_data
    performance = evaluator.evaluate(predictions, returns, prices)
    
    assert 'alpha' in performance.metrics
    assert 'beta' in performance.metrics
    assert 'information_ratio' in performance.metrics

def test_transaction_costs(evaluator, sample_data):
    """Test transaction costs impact."""
    prices, returns, _, positions, _ = sample_data
    
    # Calculate returns with and without costs
    returns_with_costs = evaluator.calculate_returns(
        positions[:-1],
        returns,
        include_costs=True
    )
    returns_without_costs = evaluator.calculate_returns(
        positions[:-1],
        returns,
        include_costs=False
    )
    
    # Verify costs impact
    assert np.mean(returns_with_costs) < np.mean(returns_without_costs)

def test_drawdown_calculation(evaluator, sample_data):
    """Test drawdown calculation."""
    prices, returns, predictions, positions, _ = sample_data
    performance = evaluator.evaluate(predictions, returns, prices)
    
    assert isinstance(performance.drawdown, np.ndarray)
    assert len(performance.drawdown) == len(performance.returns)
    assert np.all(performance.drawdown <= 0)
    assert performance.metrics['max_drawdown'] == np.min(performance.drawdown)

@pytest.mark.parametrize("window", [50, 100, 200])
def test_different_windows(evaluator, sample_data, window):
    """Test different window sizes for rolling metrics."""
    prices, returns, _, positions, _ = sample_data
    strategy_returns = evaluator.calculate_returns(positions[:-1], returns)
    
    rolling_metrics = evaluator.calculate_rolling_metrics(
        strategy_returns,
        window=window
    )
    
    assert len(rolling_metrics) == len(strategy_returns)
    assert not rolling_metrics.iloc[:window-1]['return'].isna().all()

def test_edge_cases(evaluator):
    """Test edge cases."""
    # Test empty data
    with pytest.raises(Exception):
        evaluator.calculate_metrics(np.array([]), np.array([]))
    
    # Test single data point
    single_return = np.array([0.01])
    single_position = np.array([1])
    metrics = evaluator.calculate_metrics(single_return, single_position)
    assert isinstance(metrics, dict)
    
    # Test all zeros
    zero_returns = np.zeros(100)
    zero_positions = np.zeros(100)
    metrics = evaluator.calculate_metrics(zero_returns, zero_positions)
    assert metrics['total_return'] == 0
    assert metrics['volatility'] == 0
