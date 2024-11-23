# Enhanced Model Evaluation Guide

This guide explains how to use the enhanced evaluation capabilities of the Bitcoin Trading RL project to comprehensively assess model performance.

## Overview

Enhanced evaluation enables:

- Comprehensive performance metrics
- Advanced risk analysis
- Trade-level analysis
- Rolling metrics calculation
- Performance visualization
- Detailed reporting

## Quick Start

```python
from src.evaluation.enhanced_evaluator import EnhancedEvaluator

# Initialize evaluator
evaluator = EnhancedEvaluator(
    config=config,
    risk_free_rate=0.0,
    benchmark_returns=benchmark_returns
)

# Evaluate model
performance = evaluator.evaluate(
    predictions=model_predictions,
    targets=true_values,
    prices=asset_prices
)

# Generate visualizations
evaluator.plot_results(performance)

# Generate report
report = evaluator.generate_report(performance)
```

## Configuration

Configure evaluation in `configs/config.yaml`:

```yaml
evaluation:
  enabled: true

  # Performance Metrics
  metrics:
    - accuracy
    - sharpe_ratio
    - sortino_ratio
    - max_drawdown
    - alpha
    - beta
    - information_ratio
    - calmar_ratio
    - win_rate
    - profit_factor
    - kelly_criterion

  # Risk Analysis
  risk:
    var_confidence: 0.95
    es_confidence: 0.95
    stress_test_scenarios: true
    correlation_analysis: true

  # Trade Analysis
  trade:
    min_holding_period: 1
    max_holding_period: null
    transaction_costs: 0.001
    slippage: 0.001

  # Rolling Metrics
  rolling:
    window_size: 252
    min_periods: 126
    metrics:
      - return
      - volatility
      - sharpe_ratio
      - drawdown

  # Visualization
  visualization:
    plot_style: "seaborn"
    figure_size: [15, 10]
    save_format: "png"
    dpi: 300
```

## Performance Metrics

### 1. Return Metrics

Calculate return-based metrics:

```python
def analyze_returns(returns: np.ndarray) -> Dict[str, float]:
    """Analyze return metrics."""
    metrics = {}

    # Total return
    metrics['total_return'] = np.prod(1 + returns) - 1

    # Annualized return
    metrics['annual_return'] = (
        (1 + metrics['total_return']) ** (252/len(returns)) - 1
    )

    # Volatility
    metrics['volatility'] = returns.std() * np.sqrt(252)

    return metrics
```

### 2. Risk-Adjusted Returns

Calculate risk-adjusted metrics:

```python
def analyze_risk_adjusted_returns(
    returns: np.ndarray,
    risk_free_rate: float
) -> Dict[str, float]:
    """Calculate risk-adjusted returns."""
    metrics = {}

    # Excess returns
    excess_returns = returns - risk_free_rate

    # Sharpe ratio
    metrics['sharpe_ratio'] = (
        excess_returns.mean() / returns.std() * np.sqrt(252)
    )

    # Sortino ratio
    downside_returns = returns[returns < 0]
    metrics['sortino_ratio'] = (
        excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    )

    return metrics
```

## Risk Analysis

### 1. Value at Risk

Calculate VaR and Expected Shortfall:

```python
def calculate_risk_metrics(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate risk metrics."""
    metrics = {}

    # Value at Risk
    metrics['var'] = np.percentile(returns, (1-confidence_level)*100)

    # Expected Shortfall
    metrics['es'] = returns[returns <= metrics['var']].mean()

    return metrics
```

### 2. Drawdown Analysis

Analyze drawdowns:

```python
def analyze_drawdowns(returns: np.ndarray) -> Dict[str, float]:
    """Analyze drawdowns."""
    cumulative_returns = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max

    return {
        'max_drawdown': drawdowns.min(),
        'avg_drawdown': drawdowns.mean(),
        'drawdown_duration': calculate_drawdown_duration(drawdowns)
    }
```

## Trade Analysis

### 1. Trade Statistics

Analyze individual trades:

```python
def analyze_trades(
    returns: np.ndarray,
    positions: np.ndarray
) -> Dict[str, float]:
    """Analyze trading performance."""
    trades = np.diff(positions, prepend=0)
    trade_returns = returns[trades != 0]

    return {
        'num_trades': len(trade_returns),
        'win_rate': np.mean(trade_returns > 0),
        'avg_trade': trade_returns.mean(),
        'best_trade': trade_returns.max(),
        'worst_trade': trade_returns.min()
    }
```

### 2. Position Analysis

Analyze position characteristics:

```python
def analyze_positions(positions: np.ndarray) -> Dict[str, float]:
    """Analyze positions."""
    return {
        'avg_position': np.abs(positions).mean(),
        'max_position': np.abs(positions).max(),
        'time_in_market': np.mean(positions != 0),
        'long_ratio': np.mean(positions > 0),
        'short_ratio': np.mean(positions < 0)
    }
```

## Performance Visualization

### 1. Return Plots

Create return visualizations:

```python
def plot_returns(performance: PerformanceMetrics):
    """Plot return metrics."""
    plt.figure(figsize=(15, 10))

    # Cumulative returns
    plt.subplot(2, 2, 1)
    plt.plot(np.cumprod(1 + performance.returns))
    plt.title('Cumulative Returns')

    # Return distribution
    plt.subplot(2, 2, 2)
    plt.hist(performance.returns, bins=50)
    plt.title('Return Distribution')

    # Rolling metrics
    plt.subplot(2, 2, 3)
    performance.rolling_metrics[['return', 'volatility']].plot()
    plt.title('Rolling Metrics')

    # Drawdown
    plt.subplot(2, 2, 4)
    plt.fill_between(range(len(performance.drawdown)),
                    performance.drawdown,
                    0)
    plt.title('Drawdown')

    plt.tight_layout()
```

### 2. Trade Analysis Plots

Visualize trade analysis:

```python
def plot_trade_analysis(performance: PerformanceMetrics):
    """Plot trade analysis."""
    plt.figure(figsize=(15, 5))

    # Position changes
    plt.subplot(1, 3, 1)
    plt.plot(performance.positions)
    plt.title('Positions')

    # Trade returns
    plt.subplot(1, 3, 2)
    trade_returns = performance.returns[
        np.diff(performance.positions, prepend=0) != 0
    ]
    plt.hist(trade_returns, bins=50)
    plt.title('Trade Returns')

    # Trade duration
    plt.subplot(1, 3, 3)
    plt.hist(performance.trade_analysis['trade_durations'], bins=20)
    plt.title('Trade Durations')

    plt.tight_layout()
```

## Best Practices

1. **Metric Selection**

   - Choose relevant metrics
   - Consider multiple perspectives
   - Monitor consistency

2. **Risk Assessment**

   - Use multiple risk measures
   - Consider different timeframes
   - Stress test strategies

3. **Trade Analysis**

   - Analyze trade patterns
   - Monitor costs
   - Track position sizing

4. **Performance Monitoring**
   - Regular evaluation
   - Compare to benchmarks
   - Document insights

## Advanced Topics

### Custom Metrics

Implement custom metrics:

```python
class CustomMetric:
    def __init__(self, name: str, function: Callable):
        self.name = name
        self.function = function

    def calculate(self, returns: np.ndarray) -> float:
        return self.function(returns)
```

### Performance Attribution

Implement performance attribution:

```python
def attribute_performance(
    returns: np.ndarray,
    factors: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Attribute returns to factors."""
    # Factor analysis implementation
    pass
```

## Troubleshooting

### Common Issues

1. **Data Quality**

   ```python
   # Solution: Data validation
   def validate_data(returns: np.ndarray) -> bool:
       return (
           not np.any(np.isnan(returns)) and
           not np.any(np.isinf(returns))
       )
   ```

2. **Metric Stability**

   ```python
   # Solution: Bootstrap analysis
   def bootstrap_metric(
       returns: np.ndarray,
       metric_func: Callable,
       n_samples: int = 1000
   ):
       return [
           metric_func(np.random.choice(returns, len(returns)))
           for _ in range(n_samples)
       ]
   ```

3. **Performance Issues**
   ```python
   # Solution: Optimize calculations
   @numba.jit
   def fast_metric_calculation(returns: np.ndarray):
       # Optimized implementation
       pass
   ```

## Next Steps

1. Implement custom metrics
2. Add performance attribution
3. Create monitoring dashboards
4. Develop stress testing

For API details, see the [Enhanced Evaluation API Reference](../api/enhanced_evaluation.md).
