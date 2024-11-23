# Market Making Guide

This guide explains how to use the market making capabilities of the Bitcoin Trading RL project to provide liquidity and profit from bid-ask spreads.

## Overview

Market making enables:

- Liquidity provision
- Spread capture
- Inventory management
- Risk-adjusted pricing
- Real-time market making

## Quick Start

```python
from src.strategies.market_maker import MarketMaker

# Initialize market maker
market_maker = MarketMaker(
    config=config,
    exchange='binance',
    base_spread=0.001,
    min_spread=0.0005,
    max_spread=0.01,
    order_size=0.01,
    max_inventory=1.0
)

# Run market making
await market_maker.run(
    symbol='BTC/USDT',
    interval=1.0
)
```

## Configuration

Configure market making in `configs/config.yaml`:

```yaml
market_making:
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
    base_spread: 0.001
    min_spread: 0.0005
    max_spread: 0.01
    order_size: 0.01
    max_inventory: 1.0
    risk_aversion: 1.0
    update_interval: 1.0

  # Risk Management
  risk:
    max_position: 1.0
    max_drawdown: 0.02
    max_inventory_skew: 0.5
    position_timeout: 300

  # Order Management
  orders:
    min_order_size: 0.001
    max_order_size: 1.0
    price_precision: 2
    size_precision: 6

  # Monitoring
  monitoring:
    metrics:
      - inventory
      - spread
      - pnl
      - volume
    alerts:
      slack_webhook: "your_webhook_url"
      email: "your_email"
```

## Market Making Strategies

### 1. Basic Market Making

Simple spread-based market making:

```python
def calculate_basic_spread(
    mid_price: float,
    base_spread: float
) -> Tuple[float, float]:
    """Calculate basic bid-ask spread."""
    half_spread = base_spread / 2
    bid_price = mid_price * (1 - half_spread)
    ask_price = mid_price * (1 + half_spread)
    return bid_price, ask_price
```

### 2. Adaptive Market Making

Adjust spreads based on market conditions:

```python
def calculate_adaptive_spread(
    mid_price: float,
    volatility: float,
    volume: float
) -> Tuple[float, float]:
    """Calculate adaptive spread."""
    # Adjust spread for volatility
    spread = base_spread * (1 + volatility * 10)

    # Adjust for volume
    volume_factor = volume / base_volume
    spread *= np.clip(1 / np.sqrt(volume_factor), 0.5, 2.0)

    return calculate_prices(mid_price, spread)
```

## Risk Management

### 1. Inventory Management

Manage inventory risk:

```python
def manage_inventory(
    inventory: float,
    max_inventory: float,
    mid_price: float
) -> Tuple[float, float]:
    """Manage inventory risk."""
    # Calculate inventory skew
    inventory_skew = inventory / max_inventory

    # Adjust prices
    price_skew = mid_price * inventory_skew * 0.1
    bid_price -= price_skew
    ask_price -= price_skew

    return bid_price, ask_price
```

### 2. Position Sizing

Implement position sizing:

```python
def calculate_position_sizes(
    base_size: float,
    inventory: float,
    max_inventory: float
) -> Tuple[float, float]:
    """Calculate position sizes."""
    # Adjust for inventory
    inventory_factor = inventory / max_inventory

    bid_size = base_size * (1 + max(0, inventory_factor))
    ask_size = base_size * (1 - min(0, inventory_factor))

    return bid_size, ask_size
```

## Market Analysis

### 1. Order Book Analysis

Analyze order book state:

```python
def analyze_orderbook(
    orderbook: OrderBook
) -> Dict[str, float]:
    """Analyze order book state."""
    return {
        'mid_price': calculate_mid_price(orderbook),
        'spread': calculate_spread(orderbook),
        'depth': calculate_depth(orderbook),
        'imbalance': calculate_imbalance(orderbook)
    }
```

### 2. Market State Analysis

Analyze market conditions:

```python
def analyze_market_state(
    trades: List[Dict],
    orderbook: OrderBook
) -> MarketState:
    """Analyze market state."""
    return MarketState(
        price=calculate_price(trades),
        volatility=calculate_volatility(trades),
        volume=calculate_volume(trades),
        trend=calculate_trend(trades),
        imbalance=calculate_imbalance(orderbook)
    )
```

## Order Execution

### 1. Order Placement

Place market making orders:

```python
async def place_orders(
    bid_price: float,
    ask_price: float,
    bid_size: float,
    ask_size: float
) -> bool:
    """Place market making orders."""
    try:
        # Cancel existing orders
        await cancel_all_orders()

        # Place new orders
        bid_order = await place_bid(bid_price, bid_size)
        ask_order = await place_ask(ask_price, ask_size)

        return True
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return False
```

### 2. Order Management

Manage active orders:

```python
async def manage_orders(
    current_orders: Dict,
    market_state: MarketState
) -> None:
    """Manage active orders."""
    for order in current_orders.values():
        # Check if order needs updating
        if should_update_order(order, market_state):
            await cancel_order(order)
            await place_new_order(
                calculate_new_order(order, market_state)
            )
```

## Performance Monitoring

### 1. Performance Metrics

Track key metrics:

```python
def calculate_metrics(
    trades: List[Dict],
    inventory: float,
    position_value: float
) -> Dict[str, float]:
    """Calculate performance metrics."""
    return {
        'realized_pnl': calculate_realized_pnl(trades),
        'unrealized_pnl': calculate_unrealized_pnl(
            inventory, position_value
        ),
        'inventory': inventory,
        'position_value': position_value,
        'num_trades': len(trades)
    }
```

### 2. Risk Metrics

Monitor risk metrics:

```python
def monitor_risks(
    inventory: float,
    position_value: float,
    market_state: MarketState
) -> Dict[str, float]:
    """Monitor risk metrics."""
    return {
        'inventory_risk': abs(inventory) / max_inventory,
        'position_risk': position_value / capital,
        'market_risk': calculate_market_risk(market_state)
    }
```

## Best Practices

1. **Spread Management**

   - Start conservative
   - Adjust dynamically
   - Monitor competition

2. **Inventory Management**

   - Set appropriate limits
   - Rebalance regularly
   - Consider funding costs

3. **Risk Management**

   - Monitor exposure
   - Set circuit breakers
   - Regular rebalancing

4. **Performance Monitoring**
   - Track key metrics
   - Set alerts
   - Regular review

## Advanced Topics

### Custom Spread Models

Implement custom spread calculation:

```python
class CustomSpreadModel:
    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def calculate_spread(
        self,
        market_state: MarketState
    ) -> float:
        # Custom spread calculation logic
        return optimal_spread
```

### Advanced Inventory Management

Implement inventory optimization:

```python
class InventoryOptimizer:
    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def optimize_inventory(
        self,
        current_inventory: float,
        market_state: MarketState
    ) -> Dict[str, float]:
        # Inventory optimization logic
        return optimal_positions
```

## Troubleshooting

### Common Issues

1. **Wide Spreads**

   ```python
   # Solution: Dynamic spread adjustment
   def adjust_spread(
       base_spread: float,
       market_spread: float
   ) -> float:
       return min(base_spread, market_spread * 0.9)
   ```

2. **Inventory Skew**

   ```python
   # Solution: Aggressive rebalancing
   def rebalance_inventory(
       inventory: float,
       target: float = 0.0
   ) -> Dict[str, float]:
       skew = inventory - target
       return calculate_rebalancing_orders(skew)
   ```

3. **Execution Issues**
   ```python
   # Solution: Implement retry logic
   async def place_order_with_retry(
       order: Dict,
       max_retries: int = 3
   ):
       for i in range(max_retries):
           try:
               return await place_order(order)
           except Exception:
               await asyncio.sleep(1)
   ```

## Next Steps

1. Implement custom strategies
2. Add risk models
3. Enhance execution logic
4. Create monitoring dashboards

For API details, see the [Market Making API Reference](../api/market_making.md).
