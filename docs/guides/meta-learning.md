# Meta-Learning Guide

This guide explains how to use the meta-learning capabilities of the Bitcoin Trading RL project to create adaptive trading strategies that quickly adjust to changing market conditions.

## Overview

Meta-learning, or "learning to learn," enables the model to:

- Quickly adapt to new market conditions
- Learn from limited data in new scenarios
- Transfer knowledge between different market regimes
- Improve generalization across different trading conditions

## Quick Start

```python
from src.models.meta_learner import MAMLModel

# Initialize meta-learning model
model = MAMLModel(
    config=config,
    inner_lr=0.01,
    meta_lr=0.001,
    num_inner_steps=5
)

# Train meta-learner
history = model.meta_learn(
    task_generator=task_generator,
    num_tasks=100,
    num_epochs=50
)

# Adapt to current market conditions
adapted_model = model.adapt_to_market(recent_market_data)
```

## Configuration

Configure meta-learning in `configs/config.yaml`:

```yaml
model:
  meta_learning:
    enabled: true
    algorithm: "maml" # Model-Agnostic Meta-Learning
    inner_lr: 0.01 # Learning rate for task adaptation
    meta_lr: 0.001 # Learning rate for meta-update
    num_inner_steps: 5 # Steps per task adaptation
    task_batch_size: 32

    # Task sampling
    task_config:
      window_size: 100
      stride: 20
      min_volatility: 0.1
      max_volatility: 0.5

    # Adaptation settings
    adaptation:
      market_window: "1d"
      update_frequency: "1h"
      min_samples: 48
      max_steps: 10
```

## Task Generation

### Market Regime Tasks

Create tasks based on different market regimes:

```python
def create_market_regime_tasks(data, window_size=100):
    tasks = []

    # Calculate volatility
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=20).std()

    # Identify regimes
    regimes = {
        'low_vol': volatility < volatility.quantile(0.33),
        'med_vol': (volatility >= volatility.quantile(0.33)) &
                  (volatility < volatility.quantile(0.66)),
        'high_vol': volatility >= volatility.quantile(0.66)
    }

    for regime_name, regime_mask in regimes.items():
        regime_data = data[regime_mask]
        tasks.extend(create_tasks_from_data(regime_data, window_size))

    return tasks
```

### Time-Based Tasks

Sample tasks across different time periods:

```python
def create_time_based_tasks(data, window_size=100, stride=20):
    tasks = []

    for i in range(0, len(data) - window_size, stride):
        window = data.iloc[i:i+window_size]

        # Create support and query sets
        support_data = window.iloc[:-20]  # Training data
        query_data = window.iloc[-20:]    # Testing data

        tasks.append((support_data, query_data))

    return tasks
```

## Training Process

### 1. Meta-Training

Train the meta-learner across different market conditions:

```python
# Initialize task generator
task_generator = TaskGenerator(
    data=historical_data,
    window_size=100,
    stride=20
)

# Train meta-learner
history = model.meta_learn(
    task_generator=task_generator,
    num_tasks=100,
    num_epochs=50
)

# Save meta-learned model
model.save_meta_learned('models/meta_learned.pt')
```

### 2. Market Adaptation

Adapt the model to current market conditions:

```python
# Load meta-learned model
model = MAMLModel(config)
model.load_meta_learned('models/meta_learned.pt')

# Get recent market data
recent_data = fetch_recent_data(window='1d')

# Adapt model
adapted_model = model.adapt_to_market(
    market_data=recent_data,
    num_steps=5
)
```

### 3. Online Updates

Continuously update the model:

```python
while True:
    # Get latest market data
    current_data = fetch_latest_data()

    # Adapt model
    adapted_model = model.adapt_to_market(current_data)

    # Make predictions
    predictions = adapted_model.predict(current_data)

    # Execute trades
    execute_trades(predictions)

    # Wait for next update
    time.sleep(update_interval)
```

## Performance Monitoring

### Training Metrics

Monitor meta-learning progress:

```python
def plot_meta_learning_metrics(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot meta-loss
    ax1.plot(history['meta_loss'])
    ax1.set_title('Meta-Learning Loss')

    # Plot adaptation metrics
    ax2.plot(history['adaptation_metrics'])
    ax2.set_title('Adaptation Performance')

    plt.show()
```

### Adaptation Analysis

Analyze model adaptation:

```python
def analyze_adaptation(model, market_data):
    # Pre-adaptation performance
    initial_metrics = model.evaluate(market_data)

    # Adapt model
    adapted_model = model.adapt_to_market(market_data)

    # Post-adaptation performance
    adapted_metrics = adapted_model.evaluate(market_data)

    return {
        'improvement': adapted_metrics['performance'] -
                      initial_metrics['performance'],
        'adaptation_time': adapted_metrics['time'],
        'confidence': adapted_metrics['confidence']
    }
```

## Best Practices

1. **Task Design**

   - Use diverse market conditions
   - Balance task difficulty
   - Include both common and rare scenarios

2. **Adaptation Strategy**

   - Monitor adaptation performance
   - Adjust number of adaptation steps
   - Consider market volatility

3. **Risk Management**

   - Implement confidence thresholds
   - Monitor adaptation stability
   - Use ensemble methods

4. **Performance Optimization**
   - Cache common tasks
   - Use GPU acceleration
   - Implement batch processing

## Advanced Topics

### Custom Meta-Learning

Implement custom meta-learning algorithms:

```python
class CustomMetaLearner(MAMLModel):
    def __init__(self, config):
        super().__init__(config)
        # Custom initialization

    def meta_update(self, task_gradients):
        # Custom meta-update logic
        pass
```

### Ensemble Methods

Combine multiple meta-learned models:

```python
class MetaEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, market_data):
        predictions = []
        for model in self.models:
            adapted_model = model.adapt_to_market(market_data)
            predictions.append(adapted_model.predict(market_data))

        return aggregate_predictions(predictions)
```

## Troubleshooting

### Common Issues

1. **Poor Adaptation**

   ```python
   # Solution: Increase adaptation steps or learning rate
   adapted_model = model.adapt_to_market(
       market_data,
       num_steps=10,
       learning_rate=0.02
   )
   ```

2. **Slow Adaptation**

   ```python
   # Solution: Use GPU acceleration
   model = model.cuda()
   adapted_model = model.adapt_to_market(market_data.cuda())
   ```

3. **Overfitting**
   ```python
   # Solution: Implement early stopping
   adapted_model = model.adapt_to_market(
       market_data,
       early_stopping=True,
       patience=3
   )
   ```

## Next Steps

1. Experiment with different task designs
2. Monitor adaptation performance
3. Implement custom meta-learning algorithms
4. Optimize for your specific use case

For API details, see the [Meta-Learning API Reference](../api/meta_learning.md).
