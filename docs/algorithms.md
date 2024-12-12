# Don Trading Framework Algorithms

## Reinforcement Learning Components

### Trading Environment (TradingEnvironment)

The trading environment implements a custom OpenAI Gym interface for cryptocurrency trading. It handles the core trading logic and market simulation.

#### Position Sizing
- **Discrete Actions**: Fixed set of position sizes (e.g., [-1.0, -0.5, 0.0, 0.5, 1.0])
- **Continuous Actions**: Any position size within [-1.0, 1.0]
- Position of -1.0 represents maximum short
- Position of 1.0 represents maximum long

#### Trading Step Algorithm
```python
def step(action):
    # 1. Convert action to position
    if discrete_action_space:
        new_position = positions[action_index]
    else:
        new_position = clip(action, min=-1.0, max=1.0)

    # 2. Calculate price change
    price_change = (current_price / previous_price) - 1

    # 3. Calculate trading costs
    trade_size = abs(new_position - current_position)
    trading_cost = trade_size * commission_rate


    # 4. Calculate step PnL
    position_pnl = current_position * price_change
    total_pnl = position_pnl - trading_cost

    # 5. Update state and return
    return next_observation, reward, done, info
```

### Reward Calculation Strategies

#### 1. PnL Reward
Simple profit and loss based reward:
```python
reward = step_pnl  # Direct PnL as reward
```

#### 2. Sharpe Ratio Reward
Risk-adjusted reward using rolling Sharpe ratio:
```python
def calculate_sharpe(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate/252
    if len(returns) < 2:
        return 0.0
    return np.sqrt(252) * (
        np.mean(excess_returns) /
        np.std(excess_returns)
    )
```

#### 3. Risk-Adjusted Reward
PnL with position size penalty:
```python
def calculate_risk_adjusted(pnl, position, penalty):
    position_cost = penalty * abs(position)
    return pnl - position_cost
```

## Feature Calculation

### Technical Indicators
Standard technical analysis features including:
- Simple Moving Average (SMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- On-Balance Volume (OBV)
- Volume-Weighted Average Price (VWAP)

### Market Microstructure Features
Order book and trade-based features:
- Bid-Ask Spread
- Order Book Imbalance
- Trade Flow Imbalance
- Volume Profile

## Data Management

### Data Collection
Real-time and historical data collection from Binance:
1. OHLCV market data
2. Individual trades
3. Order book snapshots
4. Liquidation events

### Database Optimization
- Partitioning by timestamp for efficient queries
- Indexes on common query patterns
- Regular archival of historical data

## Performance Monitoring

### Training Metrics
- Episode rewards
- PnL tracking
- Position statistics
- Trade analysis

### System Metrics
- Data collection latency
- Query performance
- API response times
