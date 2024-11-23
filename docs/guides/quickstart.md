# Quick Start Guide

This guide helps you get started with the Bitcoin Trading RL project quickly. For detailed setup instructions, see the [Installation Guide](installation.md).

## 5-Minute Setup

```bash
# Clone repository
git clone https://github.com/yourusername/bitcoin_trading_rl.git
cd bitcoin_trading_rl

# Initialize project
make init

# Activate virtual environment
source venv/bin/activate

# Download and process data
make data-download
make data-process
```

## Basic Usage Examples

### 1. Training a Model

```python
from src.training.trainer import Trainer
from src.models.base_model import BaseModel

# Initialize model and trainer
model = BaseModel()
trainer = Trainer(model)

# Start training
trainer.train()
```

Or using CLI:

```bash
make train
```

### 2. Sentiment Analysis

```python
from src.data.sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Get sentiment features
sentiment_data = await analyzer.fetch_news_sentiment(start_time, end_time)
features = analyzer.calculate_sentiment_features(sentiment_data)
```

### 3. Feature Engineering

```python
from src.features.feature_engineering import FeatureEngineer

# Initialize feature engineer
engineer = FeatureEngineer('data/raw/btc_data.parquet', 'data/processed/')

# Generate features
await engineer.generate_features()
```

## Common Workflows

### 1. Training Pipeline

```bash
# Download data
make data-download

# Process and engineer features
make data-process

# Start training
make train

# Monitor progress
make tensorboard
```

### 2. Evaluation

```bash
# Run evaluation
make evaluate

# View results in TensorBoard
make tensorboard
```

### 3. Development

```bash
# Run tests
make test

# Format code
make format

# Check code quality
make lint
```

## Configuration

Quick configuration changes in `configs/config.yaml`:

```yaml
# Model settings
model:
  type: "hierarchical_rl"
  hardware:
    device: "auto"
    num_gpus: -1

# Training settings
training:
  batch_size: 64
  learning_rate: 0.0001
  epochs: 100

# Sentiment analysis
sentiment:
  enabled: true
  update_interval: 300
```

## Monitoring

### 1. TensorBoard

```bash
make tensorboard
# Visit http://localhost:6006
```

### 2. Jupyter Notebook

```bash
make jupyter
# Visit http://localhost:8888
```

## Docker Deployment

```bash
# Build and run
make docker-build
make docker-run

# Stop containers
make docker-stop
```

## Common Tasks

### Adding New Features

1. Create feature calculation function:

```python
def calculate_custom_feature(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        custom_feature=data['close'].rolling(20).mean()
    )
```

2. Add to feature engineering pipeline:

```python
# src/features/feature_engineering.py
features = calculate_custom_feature(features)
```

### Implementing Custom Strategy

1. Create strategy class:

```python
from src.models.base_model import BaseModel

class CustomStrategy(BaseModel):
    def predict(self, state):
        # Implement strategy logic
        return action
```

2. Use in training:

```python
model = CustomStrategy()
trainer = Trainer(model)
trainer.train()
```

### Adding Data Source

1. Create data fetcher:

```python
from src.data.base_fetcher import BaseFetcher

class CustomFetcher(BaseFetcher):
    async def fetch_data(self):
        # Implement data fetching
        return data
```

2. Add to data pipeline:

```python
fetcher = CustomFetcher()
data = await fetcher.fetch_data()
```

## Next Steps

1. Explore [Model Architecture](model-architecture.md)
2. Learn about [Feature Engineering](feature-engineering.md)
3. Understand [Sentiment Analysis](sentiment-analysis.md)
4. Read [API Documentation](../api/data.md)

## Troubleshooting

### Common Issues

1. GPU not detected:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Data download fails:

```bash
# Check API keys
echo $BINANCE_API_KEY
# Retry download
make data-download
```

3. Training crashes:

```bash
# Check memory usage
nvidia-smi  # GPU memory
free -h     # System memory
```

### Getting Help

- Check [GitHub Issues](https://github.com/yourusername/bitcoin_trading_rl/issues)
- Join [Discussions](https://github.com/yourusername/bitcoin_trading_rl/discussions)
- Review [Documentation](../index.md)
