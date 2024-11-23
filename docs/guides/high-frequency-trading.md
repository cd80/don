# High-Frequency Trading Guide

This guide explains how to use the high-frequency trading capabilities of the Bitcoin Trading RL project to implement low-latency trading strategies.

## Overview

High-frequency trading enables:

- Low-latency trading execution
- Real-time signal generation
- Smart order routing
- Efficient order execution
- Performance monitoring

## Quick Start

```python
from src.strategies.high_frequency_trader import HighFrequencyTrader

# Initialize trader
trader = HighFrequencyTrader(
    config=config,
    exchange='binance',
    max_position=1.0,
    risk_limit=0.02,
    signal_threshold=0.5
)

# Run trading strategy
await trader.run(
    symbol='BTC/USDT',
    interval=0.001  # 1ms interval
)
```

## Configuration

Configure high-frequency trading in `configs/config.yaml`:

```yaml
high_frequency_trading:
  enabled: true

  # Exchange Configuration
  exchange:
    name: "binance"
    api_key: "your_api_key"
    api_secret: "your_api_secret"
    fees:
      maker: 0.001
      taker: 0.001

  # Strategy Parameters
  parameters:
    max_position: 1.0
    risk_limit: 0.02
    signal_threshold: 0.5
    execution_timeout: 0.1
    buffer_size: 1000
    update_interval: 0.001

  # Signal Generation
  signals:
    momentum:
      enabled: true
      lookback: 20
      threshold: 0.5
    mean_reversion:
      enabled: true
      lookback: 50
      z_score_threshold: 2.0
    order_flow:
      enabled: true
      lookback: 10
      imbalance_threshold: 0.3

  # Risk Management
  risk:
    max_drawdown: 0.02
    position_timeout: 60
    max_trades_per_second: 10
    max_pending_orders: 5

  # Execution
  execution:
    timeout: 0.1
    retry_attempts: 3
    max_slippage: 0.001
    min_fill_ratio: 0.9

  # Monitoring
  monitoring:
    metrics:
      - latency
      - slippage
      - fill_rate
      - pnl
    alerts:
      slack_webhook: "your_webhook_url"
      email: "your_email"
```

## Signal Generation

### 1. Momentum Signals

Generate momentum-based signals:

```python
def generate_momentum_signal(
    prices: np.ndarray,
    threshold: float = 0.5
) -> Optional[MarketSignal]:
    """Generate momentum signal."""
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]

    # Calculate momentum features
    features = {
        'short_momentum': returns[-5:].mean(),
        'medium_momentum': returns[-10:].mean(),
        'long_momentum': returns.mean(),
        'volatility': returns.std()
    }

    # Calculate signal
    signal = np.sign(features['short_momentum']) * min(
        abs(features['short_momentum'] / features['volatility']),
        1.0
    )

    if abs(signal) > threshold:
        return MarketSignal(
            type=SignalType.MOMENTUM,
            direction=signal,
            strength=abs(signal),
            confidence=min(1.0, abs(signal))
        )

    return None
```

### 2. Mean Reversion Signals

Generate mean reversion signals:

```python
def generate_mean_reversion_signal(
    prices: np.ndarray,
    z_score_threshold: float = 2.0
) -> Optional[MarketSignal]:
    """Generate mean reversion signal."""
    # Calculate z-score
    moving_avg = prices[-20:].mean()
    std_dev = prices[-20:].std()
    z_score = (prices[-1] - moving_avg) / std_dev

    if abs(z_score) > z_score_threshold:
        return MarketSignal(
            type=SignalType.MEAN_REVERSION,
            direction=-np.sign(z_score),
            strength=min(abs(z_score) / 4, 1.0),
            confidence=min(abs(z_score) / 3, 1.0)
        )

    return None
```

## Order Execution

### 1. Smart Order Routing

Implement smart order routing:

```python
async def execute_order(
    symbol: str,
    size: float,
    price: float,
    timeout: float = 0.1
) -> Tuple[bool, ExecutionMetrics]:
    """Execute order with smart routing."""
    start_time = time.time()

    try:
        # Place order
        order = await place_order(symbol, size, price)

        # Wait for fill
        filled = await wait_for_fill(
            order['id'],
            timeout=timeout
        )

        if filled:
            return True, calculate_metrics(
                order,
                start_time,
                price
            )

        return False, ExecutionMetrics(
            latency=timeout,
            fill_rate=0.0
        )

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False, ExecutionMetrics(
            latency=time.time() - start_time,
            fill_rate=0.0
        )
```

### 2. Order Management

Manage active orders:

```python
async def manage_orders(
    pending_orders: Dict[str, Dict]
) -> None:
    """Manage active orders."""
    for order_id, order in pending_orders.items():
        # Check timeout
        if time.time() - order['timestamp'] > max_order_time:
            await cancel_order(order_id)
            continue

        # Check fill status
        status = await get_order_status(order_id)
        if status == 'filled':
            handle_fill(order)
        elif status == 'cancelled':
            handle_cancel(order)
```

## Performance Monitoring

### 1. Latency Monitoring

Monitor execution latency:

```python
def monitor_latency(metrics: List[ExecutionMetrics]) -> Dict[str, float]:
    """Monitor execution latency."""
    latencies = [m.latency for m in metrics]
    return {
        'avg_latency': np.mean(latencies),
        'max_latency': max(latencies),
        'min_latency': min(latencies),
        'latency_std': np.std(latencies)
    }
```

### 2. Execution Analysis

Analyze execution quality:

```python
def analyze_execution(
    metrics: List[ExecutionMetrics]
) -> Dict[str, float]:
    """Analyze execution quality."""
    return {
        'avg_slippage': np.mean([m.slippage for m in metrics]),
        'fill_rate': np.mean([m.fill_rate for m in metrics]),
        'total_cost': sum(m.cost for m in metrics),
        'market_impact': np.mean([m.impact for m in metrics])
    }
```

## Best Practices

1. **Signal Generation**

   - Use multiple signals
   - Validate signal quality
   - Monitor signal performance

2. **Order Execution**

   - Implement timeouts
   - Handle errors
   - Monitor fill rates

3. **Risk Management**

   - Set position limits
   - Monitor exposure
   - Implement circuit breakers

4. **Performance Monitoring**
   - Track key metrics
   - Set alerts
   - Regular review

## Advanced Topics

### Custom Signal Generation

Implement custom signals:

```python
class CustomSignalGenerator:
    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def generate_signal(
        self,
        market_data: Dict
    ) -> Optional[MarketSignal]:
        # Custom signal generation logic
        return signal
```

### Advanced Order Routing

Implement smart routing:

```python
class SmartRouter:
    def __init__(self, exchanges: List[str]):
        self.exchanges = exchanges

    async def route_order(
        self,
        order: Dict
    ) -> Tuple[str, Dict]:
        # Smart routing logic
        return best_exchange, routed_order
```

## Troubleshooting

### Common Issues

1. **High Latency**

   ```python
   # Solution: Implement connection pooling
   async def create_connection_pool(
       size: int = 10
   ):
       return [await create_connection()
               for _ in range(size)]
   ```

2. **Order Failures**

   ```python
   # Solution: Implement retry logic
   async def execute_with_retry(
       order: Dict,
       max_retries: int = 3
   ):
       for i in range(max_retries):
           try:
               return await execute_order(order)
           except Exception:
               await asyncio.sleep(0.01)
   ```

3. **Signal Quality**
   ```python
   # Solution: Implement signal validation
   def validate_signal(
       signal: MarketSignal,
       history: List[MarketSignal]
   ) -> bool:
       return validate_signal_quality(signal, history)
   ```

## Next Steps

1. Implement custom signals
2. Add advanced routing
3. Enhance monitoring
4. Optimize performance

For API details, see the [High-Frequency Trading API Reference](../api/high_frequency_trading.md).
