# Bitcoin Trading RL

A comprehensive cryptocurrency trading framework using reinforcement learning and advanced trading strategies.

## Features

### Trading Strategies

- **Cross-Exchange Arbitrage**: Exploit price differences across exchanges
- **Market Making**: Provide liquidity and profit from bid-ask spreads
- **High-Frequency Trading**: Low-latency trading with efficient signal generation
- **Statistical Arbitrage**: Pairs trading, mean reversion, and cointegration

### Advanced Capabilities

- **Reinforcement Learning**: Deep RL models for adaptive trading
- **Risk Management**: Dynamic position sizing and risk controls
- **Portfolio Optimization**: Modern portfolio theory implementation
- **Real-time Analysis**: Live market data processing and analysis

## Installation

1. Clone the repository:

```bash
git clone https://github.com/cd80/don.git
cd don
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up configuration:

```bash
cp configs/config.yaml.example configs/config.yaml
# Edit configs/config.yaml with your settings
```

## Quick Start

### 1. Cross-Exchange Arbitrage

```python
from src.strategies.arbitrage_strategy import ArbitrageStrategy

# Initialize strategy
strategy = ArbitrageStrategy(
    config=config,
    exchanges=['binance', 'kraken'],
    min_profit_threshold=0.001,
    max_position_size=1.0
)

# Run strategy
await strategy.run(
    symbol='BTC/USDT',
    interval=1.0
)
```

### 2. Market Making

```python
from src.strategies.market_maker import MarketMaker

# Initialize market maker
market_maker = MarketMaker(
    config=config,
    exchange='binance',
    base_spread=0.001,
    min_spread=0.0005,
    max_spread=0.01
)

# Run market making
await market_maker.run(
    symbol='BTC/USDT',
    interval=1.0
)
```

### 3. High-Frequency Trading

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

# Run trading
await trader.run(
    symbol='BTC/USDT',
    interval=0.001  # 1ms interval
)
```

### 4. Statistical Arbitrage

```python
from src.strategies.statistical_arbitrage import StatisticalArbitrage

# Initialize trader
trader = StatisticalArbitrage(
    config=config,
    exchanges=['binance', 'kraken'],
    lookback_window=100,
    zscore_threshold=2.0
)

# Run strategy
await trader.run(
    symbols=['BTC/USDT', 'ETH/USDT']
)
```

## Documentation

Detailed documentation is available in the `docs` directory:

- [Installation Guide](docs/guides/installation.md)
- [Quick Start Guide](docs/guides/quickstart.md)
- [Configuration Guide](docs/guides/configuration.md)
- [API Reference](docs/api/index.md)
- Strategy Guides:
  - [Arbitrage Strategies](docs/guides/arbitrage-strategies.md)
  - [Market Making](docs/guides/market-making.md)
  - [High-Frequency Trading](docs/guides/high-frequency-trading.md)
  - [Statistical Arbitrage](docs/guides/statistical-arbitrage.md)

## Example Notebooks

Interactive examples are available in the `notebooks` directory:

- [Arbitrage Strategy Example](notebooks/arbitrage_strategy_example.ipynb)
- [Market Making Example](notebooks/market_making_example.ipynb)
- [High-Frequency Trading Example](notebooks/high_frequency_trading_example.ipynb)
- [Statistical Arbitrage Example](notebooks/statistical_arbitrage_example.ipynb)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_arbitrage_strategy.py

# Run with coverage
pytest --cov=src tests/
```

### Code Style

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Run type checking
mypy src/
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Author

- **Kim, Sungwoo** - [cd80](https://github.com/cd80)
- Email: rkwk0112@gmail.com

## Status

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for current development status and roadmap.

## Contact

- Project Link: https://github.com/cd80/don
- Documentation: https://cd80.github.io/don

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this software.
