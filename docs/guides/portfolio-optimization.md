# Portfolio Optimization Guide

This guide explains how to use the portfolio optimization capabilities of the Bitcoin Trading RL project to optimize multi-asset portfolios.

## Overview

Portfolio optimization enables:

- Efficient asset allocation
- Risk management across multiple assets
- Portfolio rebalancing strategies
- Advanced optimization techniques
- Performance monitoring and analysis

## Quick Start

```python
from src.portfolio.portfolio_optimizer import PortfolioOptimizer, OptimizationStrategy

# Initialize optimizer
optimizer = PortfolioOptimizer(
    config=config,
    risk_free_rate=0.0,
    transaction_costs=0.001
)

# Optimize portfolio
metrics = optimizer.optimize_portfolio(
    returns=historical_returns,
    strategy=OptimizationStrategy.MEAN_VARIANCE
)

# Get optimal weights
weights = metrics.weights
```

## Configuration

Configure portfolio optimization in `configs/config.yaml`:

```yaml
portfolio_optimization:
  enabled: true

  # Optimization Parameters
  covariance_method: "ledoit_wolf" # sample, ledoit_wolf, or exponential
  target_return: 0.15 # Target annual return
  risk_aversion: 1.0 # Risk aversion parameter
  max_turnover: 0.2 # Maximum turnover constraint

  # Weight Constraints
  min_weight: 0.0 # Minimum asset weight
  max_weight: 1.0 # Maximum asset weight

  # Risk Parameters
  risk_free_rate: 0.0
  transaction_costs: 0.001
  confidence_level: 0.95

  # Black-Litterman Parameters
  bl_tau: 0.05 # Prior uncertainty parameter

  # Performance Monitoring
  rebalancing:
    frequency: "monthly"
    threshold: 0.05
  monitoring:
    metrics:
      - "sharpe_ratio"
      - "volatility"
      - "var"
      - "turnover"
```

## Optimization Strategies

### 1. Mean-Variance Optimization

Classic Markowitz optimization:

```python
def optimize_mean_variance(returns: np.ndarray) -> np.ndarray:
    """Perform mean-variance optimization."""
    metrics = optimizer.optimize_portfolio(
        returns=returns,
        strategy=OptimizationStrategy.MEAN_VARIANCE
    )
    return metrics.weights
```

### 2. Risk Parity

Equal risk contribution:

```python
def optimize_risk_parity(returns: np.ndarray) -> np.ndarray:
    """Perform risk parity optimization."""
    metrics = optimizer.optimize_portfolio(
        returns=returns,
        strategy=OptimizationStrategy.RISK_PARITY
    )
    return metrics.weights
```

### 3. Black-Litterman

Incorporate views into optimization:

```python
def optimize_black_litterman(
    returns: np.ndarray,
    market_caps: np.ndarray,
    views: Dict[Tuple[int, int], float]
) -> np.ndarray:
    """Perform Black-Litterman optimization."""
    metrics = optimizer.optimize_portfolio(
        returns=returns,
        strategy=OptimizationStrategy.BLACK_LITTERMAN,
        market_caps=market_caps,
        views=views,
        view_confidences={k: 0.6 for k in views}
    )
    return metrics.weights
```

## Portfolio Analysis

### 1. Risk Metrics

Calculate comprehensive risk metrics:

```python
def analyze_portfolio_risk(
    weights: np.ndarray,
    returns: np.ndarray
) -> PortfolioMetrics:
    """Analyze portfolio risk."""
    covariance = optimizer.estimate_covariance(returns)
    metrics = optimizer.calculate_portfolio_metrics(
        weights=weights,
        returns=returns,
        covariance=covariance
    )

    print(f"Expected Return: {metrics.expected_return:.4f}")
    print(f"Volatility: {metrics.volatility:.4f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"VaR (95%): {metrics.var:.4f}")
    print(f"Expected Shortfall: {metrics.es:.4f}")

    return metrics
```

### 2. Portfolio Rebalancing

Implement rebalancing strategy:

```python
def rebalance_portfolio(
    current_weights: np.ndarray,
    returns: np.ndarray,
    threshold: float = 0.05
) -> np.ndarray:
    """Rebalance portfolio if needed."""
    metrics = optimizer.optimize_portfolio(
        returns=returns,
        strategy=OptimizationStrategy.MEAN_VARIANCE,
        current_weights=current_weights
    )

    # Check if rebalancing needed
    turnover = np.abs(metrics.weights - current_weights).sum()
    if turnover > threshold:
        return metrics.weights
    return current_weights
```

## Performance Monitoring

### 1. Portfolio Metrics

Monitor portfolio performance:

```python
def monitor_portfolio(metrics: PortfolioMetrics):
    """Monitor portfolio metrics."""
    plt.figure(figsize=(12, 8))

    # Return and Risk
    plt.subplot(2, 2, 1)
    plt.bar(['Return', 'Risk'],
            [metrics.expected_return, metrics.volatility])
    plt.title('Return and Risk')

    # Asset Weights
    plt.subplot(2, 2, 2)
    plt.pie(metrics.weights,
            labels=[f'Asset {i}' for i in range(len(metrics.weights))])
    plt.title('Portfolio Allocation')

    # Risk Metrics
    plt.subplot(2, 2, 3)
    plt.bar(['VaR', 'ES', 'Vol'],
            [metrics.var, metrics.es, metrics.volatility])
    plt.title('Risk Metrics')

    # Diversification
    plt.subplot(2, 2, 4)
    plt.bar(['Div Ratio', 'Concentration'],
            [metrics.diversification_ratio, metrics.concentration])
    plt.title('Diversification Metrics')

    plt.tight_layout()
    plt.show()
```

### 2. Performance Attribution

Analyze performance attribution:

```python
def analyze_attribution(
    weights: np.ndarray,
    returns: np.ndarray
) -> pd.DataFrame:
    """Analyze performance attribution."""
    asset_returns = returns.mean(axis=0)
    contribution = weights * asset_returns

    return pd.DataFrame({
        'Weight': weights,
        'Return': asset_returns,
        'Contribution': contribution,
        'Contribution %': contribution / contribution.sum() * 100
    })
```

## Best Practices

1. **Risk Management**

   - Set appropriate constraints
   - Monitor risk metrics
   - Regular rebalancing

2. **Optimization Strategy**

   - Choose based on objectives
   - Consider transaction costs
   - Monitor turnover

3. **Data Quality**

   - Clean input data
   - Handle missing values
   - Consider sample size

4. **Performance Monitoring**
   - Regular review
   - Track key metrics
   - Document decisions

## Advanced Topics

### Custom Optimization

Implement custom optimization strategy:

```python
class CustomOptimizer(PortfolioOptimizer):
    def custom_optimization(self, returns: np.ndarray) -> np.ndarray:
        """Implement custom optimization logic."""
        # Custom optimization code
        pass

    def optimize_portfolio(self, returns: np.ndarray, **kwargs) -> PortfolioMetrics:
        weights = self.custom_optimization(returns)
        return self.calculate_portfolio_metrics(weights, returns)
```

### Risk Management Integration

Integrate with risk management:

```python
def optimize_with_risk_limits(
    returns: np.ndarray,
    risk_limits: Dict[str, float]
) -> np.ndarray:
    """Optimize with risk constraints."""
    def risk_constraint(weights):
        metrics = optimizer.calculate_portfolio_metrics(
            weights, returns
        )
        return [
            metrics.volatility - risk_limits['volatility'],
            metrics.var - risk_limits['var']
        ]

    # Add constraints to optimization
    return optimize_with_constraints(returns, risk_constraint)
```

## Troubleshooting

### Common Issues

1. **Optimization Failure**

   ```python
   # Solution: Add constraints gradually
   def add_constraints_gradually(optimizer):
       constraints = []
       for constraint in all_constraints:
           constraints.append(constraint)
           try:
               result = optimizer.optimize_with_constraints(
                   returns, constraints
               )
               if result.success:
                   continue
           except:
               constraints.pop()
       return constraints
   ```

2. **Unstable Solutions**

   ```python
   # Solution: Add regularization
   def optimize_with_regularization(
       returns: np.ndarray,
       lambda_reg: float = 0.1
   ):
       def objective(weights):
           return base_objective(weights) + \
                  lambda_reg * np.sum(weights ** 2)
       return optimize(objective)
   ```

3. **High Turnover**
   ```python
   # Solution: Implement cost-aware optimization
   def optimize_with_costs(
       returns: np.ndarray,
       current_weights: np.ndarray,
       cost_factor: float = 0.01
   ):
       def cost_objective(weights):
           turnover = np.abs(weights - current_weights).sum()
           return base_objective(weights) + \
                  cost_factor * turnover
       return optimize(cost_objective)
   ```

## Next Steps

1. Implement custom optimization strategies
2. Add risk management integration
3. Create monitoring dashboards
4. Develop rebalancing strategies

For API details, see the [Portfolio Optimization API Reference](../api/portfolio_optimization.md).
