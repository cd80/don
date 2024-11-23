# Statistical Arbitrage Guide

This guide explains how to use the statistical arbitrage capabilities of the Bitcoin Trading RL project to implement pairs trading, mean reversion, and cointegration strategies.

## Overview

Statistical arbitrage enables:

- Pairs trading strategies
- Mean reversion trading
- Cointegration analysis
- Factor-based arbitrage
- Risk-managed execution

## Quick Start

```python
from src.strategies.statistical_arbitrage import StatisticalArbitrage

# Initialize trader
trader = StatisticalArbitrage(
    config=config,
    exchanges=['binance', 'kraken'],
    lookback_window=100,
    zscore_threshold=2.0,
    min_half_life=1.0,
    max_half_life=100.0
)

# Run strategy
await trader.run(
    symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
)
```

## Configuration

Configure statistical arbitrage in `configs/config.yaml`:

```yaml
statistical_arbitrage:
  enabled: true

  # Exchange Configuration
  exchanges:
    - name: "binance"
      api_key: "your_api_key"
      api_secret: "your_api_secret"
      fees:
        maker: 0.001
        taker: 0.001
    - name: "kraken"
      api_key: "your_api_key"
      api_secret: "your_api_secret"
      fees:
        maker: 0.0016
        taker: 0.0026

  # Strategy Parameters
  parameters:
    lookback_window: 100
    zscore_threshold: 2.0
    min_half_life: 1.0
    max_half_life: 100.0
    min_correlation: 0.5
    max_position: 1.0
    risk_limit: 0.02

  # Pair Selection
  pair_selection:
    min_correlation: 0.5
    max_pairs: 10
    rebalance_interval: 86400

  # Signal Generation
  signals:
    pairs_trading:
      enabled: true
      entry_threshold: 2.0
      exit_threshold: 0.0
    mean_reversion:
      enabled: true
      lookback: 50
      entry_threshold: 2.0
    cointegration:
      enabled: true
      significance: 0.05

  # Risk Management
  risk:
    max_drawdown: 0.02
    position_timeout: 86400
    stop_loss: 0.02
    take_profit: 0.05

  # Execution
  execution:
    timeout: 10
    retry_attempts: 3
    min_fill_ratio: 0.9

  # Monitoring
  monitoring:
    metrics:
      - pnl
      - sharpe
      - positions
      - correlations
    alerts:
      slack_webhook: "your_webhook_url"
      email: "your_email"
```

## Pair Analysis

### 1. Correlation Analysis

Analyze pair correlations:

```python
def analyze_correlations(
    prices: pd.DataFrame,
    min_correlation: float = 0.5
) -> pd.DataFrame:
    """Analyze pair correlations."""
    correlations = prices.corr()

    # Find highly correlated pairs
    pairs = []
    for i in range(len(correlations)):
        for j in range(i+1, len(correlations)):
            if abs(correlations.iloc[i, j]) > min_correlation:
                pairs.append({
                    'asset_a': correlations.index[i],
                    'asset_b': correlations.index[j],
                    'correlation': correlations.iloc[i, j]
                })

    return pd.DataFrame(pairs)
```

### 2. Cointegration Testing

Test for cointegration:

```python
def test_cointegration(
    prices_a: pd.Series,
    prices_b: pd.Series
) -> Tuple[float, float]:
    """Test pair cointegration."""
    # Calculate hedge ratio
    model = LinearRegression()
    model.fit(prices_b.values.reshape(-1, 1),
             prices_a.values)
    hedge_ratio = model.coef_[0]

    # Calculate spread
    spread = prices_a - hedge_ratio * prices_b

    # Test stationarity
    adf_pvalue = adfuller(spread)[1]

    return hedge_ratio, adf_pvalue
```

## Signal Generation

### 1. Pairs Trading Signals

Generate pairs trading signals:

```python
def generate_pairs_signal(
    pair: PairAnalysis,
    zscore_threshold: float = 2.0
) -> Optional[TradingSignal]:
    """Generate pairs trading signal."""
    if abs(pair.zscore) > zscore_threshold:
        return TradingSignal(
            type=StatArbType.PAIRS_TRADING,
            assets=[pair.asset_a, pair.asset_b],
            direction=[
                -np.sign(pair.zscore),
                np.sign(pair.zscore) * pair.hedge_ratio
            ],
            confidence=min(
                abs(pair.zscore) / zscore_threshold,
                1.0
            )
        )
    return None
```

### 2. Mean Reversion Signals

Generate mean reversion signals:

```python
def generate_mean_reversion_signal(
    prices: pd.Series,
    lookback: int = 50,
    zscore_threshold: float = 2.0
) -> Optional[TradingSignal]:
    """Generate mean reversion signal."""
    # Calculate z-score
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()
    zscore = (prices - rolling_mean) / rolling_std

    if abs(zscore.iloc[-1]) > zscore_threshold:
        return TradingSignal(
            type=StatArbType.MEAN_REVERSION,
            assets=[prices.name],
            direction=[-np.sign(zscore.iloc[-1])],
            confidence=min(
                abs(zscore.iloc[-1]) / zscore_threshold,
                1.0
            )
        )
    return None
```

## Risk Management

### 1. Position Sizing

Implement position sizing:

```python
def calculate_position_sizes(
    signal: TradingSignal,
    max_position: float,
    risk_limit: float
) -> List[float]:
    """Calculate position sizes."""
    # Base sizes on signal direction
    sizes = [d * max_position for d in signal.direction]

    # Adjust for risk limit
    total_risk = sum(abs(s) for s in sizes)
    if total_risk > risk_limit:
        sizes = [s * risk_limit / total_risk for s in sizes]

    return sizes
```

### 2. Risk Monitoring

Monitor risk metrics:

```python
def monitor_risks(
    positions: Dict[str, float],
    prices: pd.DataFrame
) -> Dict[str, float]:
    """Monitor risk metrics."""
    portfolio_value = sum(
        pos * prices[asset].iloc[-1]
        for asset, pos in positions.items()
    )

    return {
        'portfolio_value': portfolio_value,
        'position_sizes': positions,
        'risk_exposure': sum(abs(p) for p in positions.values())
    }
```

## Performance Analysis

### 1. Strategy Metrics

Calculate performance metrics:

```python
def calculate_metrics(
    returns: pd.Series,
    positions: Dict[str, float]
) -> Dict[str, float]:
    """Calculate performance metrics."""
    return {
        'total_return': (1 + returns).prod() - 1,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
        'max_drawdown': (
            (1 + returns).cumprod() /
            (1 + returns).cumprod().cummax() - 1
        ).min(),
        'current_positions': positions
    }
```

### 2. Pair Analysis

Analyze pair relationships:

```python
def analyze_pairs(
    pairs: List[PairAnalysis]
) -> pd.DataFrame:
    """Analyze pair relationships."""
    return pd.DataFrame([
        {
            'pair': f"{p.asset_a}/{p.asset_b}",
            'correlation': p.correlation,
            'hedge_ratio': p.hedge_ratio,
            'half_life': p.half_life,
            'zscore': p.zscore
        }
        for p in pairs
    ])
```

## Best Practices

1. **Pair Selection**

   - Use correlation thresholds
   - Test cointegration
   - Monitor stability

2. **Signal Generation**

   - Validate signals
   - Use multiple timeframes
   - Consider transaction costs

3. **Risk Management**

   - Set position limits
   - Monitor exposures
   - Use stop losses

4. **Performance Monitoring**
   - Track key metrics
   - Monitor pair stability
   - Regular rebalancing

## Advanced Topics

### Custom Pair Selection

Implement custom pair selection:

```python
class CustomPairSelector:
    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def select_pairs(
        self,
        prices: pd.DataFrame
    ) -> List[Tuple[str, str]]:
        # Custom pair selection logic
        return pairs
```

### Advanced Signal Generation

Implement custom signals:

```python
class CustomSignalGenerator:
    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def generate_signal(
        self,
        pair: PairAnalysis
    ) -> Optional[TradingSignal]:
        # Custom signal generation logic
        return signal
```

## Troubleshooting

### Common Issues

1. **Unstable Pairs**

   ```python
   # Solution: Monitor pair stability
   def monitor_stability(
       pair: PairAnalysis,
       window: int = 50
   ) -> bool:
       return check_pair_stability(pair, window)
   ```

2. **Poor Execution**

   ```python
   # Solution: Implement smart routing
   async def execute_with_routing(
       order: Dict,
       exchanges: List[str]
   ):
       return await route_and_execute(order, exchanges)
   ```

3. **Risk Issues**
   ```python
   # Solution: Implement circuit breakers
   def check_risk_limits(
       positions: Dict[str, float],
       limits: Dict[str, float]
   ) -> bool:
       return validate_risk_limits(positions, limits)
   ```

## Next Steps

1. Implement custom strategies
2. Add pair selection
3. Enhance risk management
4. Create monitoring dashboards

For API details, see the [Statistical Arbitrage API Reference](../api/statistical_arbitrage.md).
