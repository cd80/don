# Arbitrage Strategies Guide

This guide explains how to use the arbitrage trading capabilities of the Bitcoin Trading RL project to profit from price differences across exchanges.

## Overview

Arbitrage strategies enable:

- Cross-exchange price difference exploitation
- Triangular arbitrage opportunities
- Statistical arbitrage
- Risk-managed execution
- Real-time opportunity monitoring

## Quick Start

```python
from src.strategies.arbitrage_strategy import ArbitrageStrategy

# Initialize strategy
strategy = ArbitrageStrategy(
    config=config,
    exchanges=['binance', 'kraken', 'coinbase'],
    min_profit_threshold=0.001,
    max_position_size=1.0
)

# Run strategy
await strategy.run(
    symbol='BTC/USDT',
    volume=0.1,
    interval=1.0
)
```

## Configuration

Configure arbitrage strategies in `configs/config.yaml`:

```yaml
arbitrage:
  enabled: true

  # Exchange Configuration
  exchanges:
    binance:
      api_key: "your_api_key"
      api_secret: "your_api_secret"
      fees:
        maker: 0.001
        taker: 0.001
    kraken:
      api_key: "your_api_key"
      api_secret: "your_api_secret"
      fees:
        maker: 0.0016
        taker: 0.0026
    coinbase:
      api_key: "your_api_key"
      api_secret: "your_api_secret"
      fees:
        maker: 0.005
        taker: 0.005

  # Strategy Parameters
  parameters:
    min_profit_threshold: 0.001
    max_position_size: 1.0
    max_order_time: 60
    update_interval: 1.0

  # Risk Management
  risk:
    max_drawdown: 0.02
    max_exposure: 0.5
    min_liquidity: 100000
    max_slippage: 0.001

  # Execution
  execution:
    timeout: 5
    retry_attempts: 3
    concurrent_orders: 2

  # Monitoring
  monitoring:
    metrics:
      - profit_loss
      - execution_time
      - slippage
      - success_rate
    alerts:
      slack_webhook: "your_webhook_url"
      email: "your_email"
```

## Strategy Types

### 1. Simple Arbitrage

Direct price difference exploitation:

```python
def identify_simple_arbitrage(orderbooks: Dict[str, pd.DataFrame]) -> List[ArbitrageOpportunity]:
    """Identify simple arbitrage opportunities."""
    opportunities = []

    for exchange_a, book_a in orderbooks.items():
        for exchange_b, book_b in orderbooks.items():
            if exchange_a != exchange_b:
                # Check if buy price on A is lower than sell price on B
                if book_a['asks'][0] < book_b['bids'][0]:
                    opportunities.append(
                        ArbitrageOpportunity(
                            exchange_a=exchange_a,
                            exchange_b=exchange_b,
                            price_a=book_a['asks'][0],
                            price_b=book_b['bids'][0]
                        )
                    )

    return opportunities
```

### 2. Triangular Arbitrage

Multi-pair arbitrage:

```python
def identify_triangular_arbitrage(
    exchange: str,
    pairs: List[str]
) -> List[ArbitrageOpportunity]:
    """Identify triangular arbitrage opportunities."""
    opportunities = []

    # Example: BTC/USD -> ETH/BTC -> ETH/USD
    btc_usd = get_price('BTC/USD')
    eth_btc = get_price('ETH/BTC')
    eth_usd = get_price('ETH/USD')

    # Calculate arbitrage
    synthetic = btc_usd * eth_btc
    if abs(synthetic - eth_usd) > min_profit:
        opportunities.append(
            create_triangular_opportunity(
                pairs, [btc_usd, eth_btc, eth_usd]
            )
        )

    return opportunities
```

## Risk Management

### 1. Pre-trade Analysis

Analyze risks before execution:

```python
def analyze_risks(
    opportunity: ArbitrageOpportunity
) -> Dict[str, float]:
    """Analyze arbitrage risks."""
    return {
        'execution_risk': calculate_execution_risk(opportunity),
        'slippage_risk': estimate_slippage(opportunity),
        'liquidity_risk': assess_liquidity(opportunity),
        'counterparty_risk': get_counterparty_risk(opportunity)
    }
```

### 2. Position Sizing

Implement position sizing:

```python
def calculate_position_size(
    opportunity: ArbitrageOpportunity,
    risks: Dict[str, float]
) -> float:
    """Calculate optimal position size."""
    # Base size on available liquidity
    base_size = min(
        opportunity.volume_a,
        opportunity.volume_b
    )

    # Adjust for risks
    risk_factor = 1.0
    for risk in risks.values():
        risk_factor *= (1.0 - risk)

    return base_size * risk_factor
```

## Execution Management

### 1. Order Execution

Execute arbitrage trades:

```python
async def execute_trades(
    opportunity: ArbitrageOpportunity
) -> bool:
    """Execute arbitrage trades."""
    try:
        # Place entry order
        entry = await place_order(
            exchange=opportunity.exchange_a,
            side='buy',
            price=opportunity.price_a,
            volume=opportunity.volume
        )

        # Place exit order
        exit = await place_order(
            exchange=opportunity.exchange_b,
            side='sell',
            price=opportunity.price_b,
            volume=opportunity.volume
        )

        return True
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False
```

### 2. Position Monitoring

Monitor active positions:

```python
async def monitor_positions(
    positions: Dict[str, Dict]
) -> None:
    """Monitor active positions."""
    for exchange, position in positions.items():
        status = await get_order_status(
            exchange,
            position['order_id']
        )

        if status == 'filled':
            handle_fill(position)
        elif status == 'failed':
            handle_failure(position)
```

## Performance Monitoring

### 1. Metrics Tracking

Track performance metrics:

```python
def track_metrics(
    opportunities: List[ArbitrageOpportunity]
) -> Dict[str, float]:
    """Track strategy metrics."""
    return {
        'total_profit': calculate_total_profit(opportunities),
        'win_rate': calculate_win_rate(opportunities),
        'avg_profit': calculate_avg_profit(opportunities),
        'sharpe_ratio': calculate_sharpe_ratio(opportunities)
    }
```

### 2. Visualization

Create performance visualizations:

```python
def plot_performance(metrics: Dict[str, List[float]]):
    """Plot performance metrics."""
    plt.figure(figsize=(15, 10))

    # Profit plot
    plt.subplot(2, 2, 1)
    plt.plot(metrics['cumulative_profit'])
    plt.title('Cumulative Profit')

    # Win rate plot
    plt.subplot(2, 2, 2)
    plt.plot(metrics['win_rate'])
    plt.title('Win Rate')

    # Volume plot
    plt.subplot(2, 2, 3)
    plt.plot(metrics['volume'])
    plt.title('Trading Volume')

    # Spread plot
    plt.subplot(2, 2, 4)
    plt.plot(metrics['spread'])
    plt.title('Average Spread')

    plt.tight_layout()
```

## Best Practices

1. **Risk Management**

   - Set appropriate thresholds
   - Monitor execution risks
   - Implement circuit breakers

2. **Execution Strategy**

   - Use limit orders when possible
   - Monitor fill rates
   - Implement retry logic

3. **Performance Monitoring**

   - Track key metrics
   - Set alerts
   - Regular review

4. **Exchange Integration**
   - Handle rate limits
   - Monitor API health
   - Backup connections

## Advanced Topics

### Custom Risk Models

Implement custom risk models:

```python
class CustomRiskModel:
    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def evaluate_risk(self, opportunity: ArbitrageOpportunity) -> float:
        # Custom risk evaluation logic
        return risk_score
```

### Advanced Execution

Implement smart execution:

```python
class SmartExecutor:
    def __init__(self, exchanges: List[str]):
        self.exchanges = exchanges

    async def execute_with_retry(
        self,
        opportunity: ArbitrageOpportunity
    ) -> bool:
        # Smart execution logic with retries
        pass
```

## Troubleshooting

### Common Issues

1. **Execution Delays**

   ```python
   # Solution: Implement timeout handling
   async def execute_with_timeout(
       coroutine,
       timeout: float
   ):
       try:
           return await asyncio.wait_for(
               coroutine,
               timeout=timeout
           )
       except asyncio.TimeoutError:
           handle_timeout()
   ```

2. **Price Slippage**

   ```python
   # Solution: Implement slippage protection
   def check_slippage(
       expected_price: float,
       executed_price: float,
       max_slippage: float
   ) -> bool:
       return abs(executed_price - expected_price) <= max_slippage
   ```

3. **API Issues**
   ```python
   # Solution: Implement fallback exchanges
   async def execute_with_fallback(
       opportunity: ArbitrageOpportunity,
       fallback_exchanges: List[str]
   ):
       for exchange in fallback_exchanges:
           try:
               return await execute_on_exchange(exchange)
           except:
               continue
   ```

## Next Steps

1. Implement custom strategies
2. Add risk models
3. Enhance execution logic
4. Create monitoring dashboards

For API details, see the [Arbitrage Strategies API Reference](../api/arbitrage_strategies.md).
