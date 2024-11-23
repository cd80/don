# Dynamic Risk Management Guide

This guide explains how to use the dynamic risk management capabilities of the Bitcoin Trading RL project to adaptively manage trading risk.

## Overview

Dynamic risk management enables:

- Adaptive position sizing based on market conditions
- Market regime detection and adaptation
- Risk limit adjustments based on performance
- Comprehensive risk metrics monitoring
- Portfolio protection mechanisms

## Quick Start

```python
from src.risk.risk_manager import DynamicRiskManager

# Initialize risk manager
risk_manager = DynamicRiskManager(
    config=config,
    initial_capital=100000,
    max_position_size=1.0,
    max_leverage=3.0
)

# Calculate risk metrics
metrics = risk_manager.calculate_risk_metrics(
    returns=historical_returns,
    model_confidence=0.8
)

# Get position size
position = risk_manager.calculate_position_size(
    prediction=0.5,
    confidence=0.8,
    risk_metrics=metrics
)
```

## Configuration

Configure risk management in `configs/config.yaml`:

```yaml
risk_management:
  enabled: true

  # Risk Limits
  var_limit: 0.02
  drawdown_limit: 0.20
  volatility_limit: 0.03
  max_position_size: 1.0
  max_leverage: 3.0

  # Market Analysis
  volatility_window: 20
  trend_window: 50
  confidence_level: 0.95

  # Dynamic Adjustments
  position_adjustment:
    enabled: true
    min_confidence: 0.6
    regime_factors:
      low_volatility: 1.0
      medium_volatility: 0.8
      high_volatility: 0.5
      trending_up: 1.2
      trending_down: 1.2
      ranging: 0.7
      breakout: 0.6

  # Performance Monitoring
  performance:
    risk_free_rate: 0.0
    target_sharpe: 2.0
    min_calmar: 0.5
    max_drawdown: 0.20
```

## Risk Metrics

### 1. Value at Risk (VaR)

Calculate VaR:

```python
def calculate_var(returns: np.ndarray, confidence_level: float) -> float:
    """Calculate Value at Risk."""
    return -np.percentile(returns, (1 - confidence_level) * 100)
```

### 2. Expected Shortfall

Calculate Expected Shortfall:

```python
def calculate_es(returns: np.ndarray, var: float) -> float:
    """Calculate Expected Shortfall."""
    return -np.mean(returns[returns < -var])
```

### 3. Dynamic Risk Metrics

Monitor multiple risk dimensions:

```python
metrics = risk_manager.calculate_risk_metrics(returns, model_confidence)
print(f"VaR: {metrics.var:.4f}")
print(f"ES: {metrics.es:.4f}")
print(f"Volatility: {metrics.volatility:.4f}")
print(f"Drawdown: {metrics.drawdown:.4f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

## Market Regime Detection

### 1. Volatility Regimes

Detect volatility regimes:

```python
def analyze_volatility(returns: np.ndarray, window: int = 20) -> str:
    volatility = np.std(returns[-window:])
    if volatility > historical_vol * 1.5:
        return "high_volatility"
    elif volatility < historical_vol * 0.5:
        return "low_volatility"
    return "medium_volatility"
```

### 2. Trend Detection

Identify market trends:

```python
def detect_trend(prices: np.ndarray, window: int = 50) -> str:
    trend = (prices[-1] - prices[-window]) / prices[-window]
    if abs(trend) < 0.02:
        return "ranging"
    return "trending_up" if trend > 0 else "trending_down"
```

## Position Sizing

### 1. Dynamic Sizing

Adjust position size based on conditions:

```python
position_size = risk_manager.calculate_position_size(
    prediction=model_prediction,
    confidence=model_confidence,
    risk_metrics=current_metrics
)
```

### 2. Risk-Adjusted Sizing

Incorporate multiple factors:

```python
def adjust_position_size(base_size: float, metrics: RiskMetrics) -> float:
    # Adjust for volatility
    vol_factor = min(1.0, target_vol / metrics.volatility)

    # Adjust for drawdown
    dd_factor = 1.0 - (metrics.drawdown / max_drawdown)

    # Adjust for model confidence
    conf_factor = max(0.0, (metrics.model_confidence - min_confidence) /
                          (1.0 - min_confidence))

    return base_size * vol_factor * dd_factor * conf_factor
```

## Risk Monitoring

### 1. Performance Tracking

Monitor trading performance:

```python
def track_performance(risk_manager):
    metrics = risk_manager.update_position(new_position, current_price)
    print(f"Position Size: {metrics['position_size']:.2f}")
    print(f"Capital: {metrics['capital']:.2f}")
    print(f"Total P&L: {metrics['total_pnl']:.2f}")
    print(f"Return: {metrics['return']:.2%}")
```

### 2. Risk Limit Monitoring

Monitor risk limits:

```python
def check_risk_limits(metrics: RiskMetrics, limits: Dict) -> bool:
    return (
        metrics.var <= limits['var_limit'] and
        metrics.drawdown <= limits['drawdown_limit'] and
        metrics.volatility <= limits['volatility_limit']
    )
```

## Best Practices

1. **Risk Limit Setting**

   - Start conservative
   - Adjust based on performance
   - Monitor limit utilization

2. **Position Sizing**

   - Consider multiple factors
   - Implement gradual changes
   - Use stop-losses

3. **Market Regime Adaptation**

   - Monitor regime transitions
   - Adjust strategy parameters
   - Validate regime detection

4. **Performance Monitoring**
   - Track multiple metrics
   - Set alert thresholds
   - Regular review periods

## Advanced Topics

### Custom Risk Metrics

Implement custom risk metrics:

```python
class CustomRiskMetrics:
    def __init__(self, returns: np.ndarray):
        self.returns = returns

    def calculate_custom_var(self) -> float:
        # Implement custom VaR calculation
        pass

    def calculate_custom_risk(self) -> float:
        # Implement custom risk measure
        pass
```

### Dynamic Risk Adjustment

Implement dynamic risk adjustment:

```python
class DynamicRiskAdjuster:
    def __init__(self, risk_manager: DynamicRiskManager):
        self.risk_manager = risk_manager

    def adjust_limits(self, performance: Dict[str, float]):
        # Adjust risk limits based on performance
        if performance['sharpe_ratio'] > target_sharpe:
            self.risk_manager.increase_limits()
        elif performance['drawdown'] > max_drawdown:
            self.risk_manager.decrease_limits()
```

## Troubleshooting

### Common Issues

1. **Excessive Risk**

   ```python
   # Solution: Implement circuit breakers
   if metrics.drawdown > emergency_stop:
       risk_manager.close_all_positions()
   ```

2. **Delayed Reactions**

   ```python
   # Solution: Use forward-looking indicators
   regime = risk_manager.detect_market_regime(
       prices,
       returns,
       forward_looking=True
   )
   ```

3. **Position Sizing Issues**
   ```python
   # Solution: Implement smooth transitions
   def smooth_position_change(current: float, target: float,
                            max_change: float = 0.1) -> float:
       return current + np.clip(
           target - current,
           -max_change,
           max_change
       )
   ```

## Next Steps

1. Implement custom risk metrics
2. Add regime-specific strategies
3. Develop risk monitoring dashboards
4. Create alert systems

For API details, see the [Risk Management API Reference](../api/risk_management.md).
