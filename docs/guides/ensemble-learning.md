# Ensemble Learning Guide

This guide explains how to use the ensemble learning capabilities of the Bitcoin Trading RL project to combine multiple models for improved predictions.

## Overview

Ensemble learning enables:

- Combining multiple models for better predictions
- Reducing prediction variance and bias
- Improving model robustness
- Handling different market conditions

## Quick Start

```python
from src.models.ensemble_learner import create_ensemble
from src.models.base_model import BaseModel

# Create bagging ensemble
ensemble = create_ensemble(
    ensemble_type='bagging',
    base_model_class=BaseModel,
    config=config,
    num_models=5
)

# Train ensemble
metrics = ensemble.update(features, targets)

# Make predictions
predictions = ensemble.predict(features)
```

## Configuration

Configure ensemble learning in `configs/config.yaml`:

```yaml
model:
  ensemble:
    enabled: true
    type: "bagging" # bagging, boosting, stacking, or voting
    num_models: 5

    bagging:
      bootstrap_ratio: 0.8

    boosting:
      learning_rate: 0.1

    stacking:
      meta_model:
        hidden_dims: [64, 32]

    voting:
      method: "soft" # soft or hard
      weights: null # optional custom weights

    training:
      batch_size: 64
      learning_rate: 0.001
      num_epochs: 100
```

## Ensemble Types

### 1. Bagging Ensemble

Uses bootstrap sampling to create diverse models:

```python
from src.models.ensemble_learner import BaggingEnsemble

ensemble = BaggingEnsemble(
    base_model_class=BaseModel,
    config=config,
    num_models=5,
    bootstrap_ratio=0.8
)
```

### 2. Boosting Ensemble

Focuses on hard examples:

```python
from src.models.ensemble_learner import BoostingEnsemble

ensemble = BoostingEnsemble(
    base_model_class=BaseModel,
    config=config,
    num_models=5,
    learning_rate=0.1
)
```

### 3. Stacking Ensemble

Uses meta-model to combine predictions:

```python
from src.models.ensemble_learner import StackingEnsemble

ensemble = StackingEnsemble(
    base_model_class=BaseModel,
    meta_model_class=BaseModel,
    config=config,
    num_models=5
)
```

### 4. Voting Ensemble

Combines pre-trained models:

```python
from src.models.ensemble_learner import VotingEnsemble

ensemble = VotingEnsemble(
    models=trained_models,
    voting='soft',
    weights=[0.4, 0.3, 0.3]
)
```

## Training Process

### 1. Bagging Training

```python
def train_bagging_ensemble(ensemble, data_loader, num_epochs=100):
    history = []

    for epoch in range(num_epochs):
        epoch_metrics = []

        for batch in data_loader:
            features, targets = batch
            metrics = ensemble.update(features, targets)
            epoch_metrics.append(metrics)

        # Average metrics
        avg_metrics = average_metrics(epoch_metrics)
        history.append(avg_metrics)

        print(f"Epoch {epoch}: {avg_metrics}")

    return history
```

### 2. Boosting Training

```python
def train_boosting_ensemble(ensemble, data_loader, num_epochs=100):
    history = []

    for epoch in range(num_epochs):
        for batch in data_loader:
            features, targets = batch
            metrics = ensemble.update(features, targets)
            history.append(metrics)

            # Weights are automatically updated based on errors
            print(f"Model weights: {ensemble.weights}")

    return history
```

### 3. Stacking Training

```python
def train_stacking_ensemble(ensemble, data_loader, num_epochs=100):
    # Train base models
    for model in ensemble.base_models:
        train_model(model, data_loader)

    # Generate meta-features
    meta_features = ensemble.get_meta_features(features)

    # Train meta-model
    train_model(ensemble.meta_model, meta_features, targets)
```

## Performance Monitoring

### Training Progress

```python
def plot_ensemble_progress(history):
    plt.figure(figsize=(15, 5))

    # Model losses
    plt.subplot(1, 3, 1)
    for i, model_losses in enumerate(history['model_losses']):
        plt.plot(model_losses, label=f'Model {i+1}')
    plt.title('Individual Model Losses')
    plt.legend()

    # Ensemble loss
    plt.subplot(1, 3, 2)
    plt.plot(history['ensemble_loss'])
    plt.title('Ensemble Loss')

    # Model weights
    plt.subplot(1, 3, 3)
    plt.plot(history['weights'])
    plt.title('Model Weights')

    plt.tight_layout()
    plt.show()
```

### Model Analysis

```python
def analyze_ensemble(ensemble, test_loader):
    model_predictions = []
    ensemble_predictions = []
    true_values = []

    for features, targets in test_loader:
        # Individual model predictions
        for model in ensemble.models:
            pred = model.predict(features)
            model_predictions.append(pred)

        # Ensemble predictions
        ensemble_pred = ensemble.predict(features)
        ensemble_predictions.append(ensemble_pred)
        true_values.append(targets)

    return {
        'model_predictions': model_predictions,
        'ensemble_predictions': ensemble_predictions,
        'true_values': true_values
    }
```

## Best Practices

1. **Model Selection**

   - Use diverse base models
   - Balance model complexity
   - Consider computational resources

2. **Ensemble Size**

   - Start with 5-10 models
   - Monitor diminishing returns
   - Balance performance and complexity

3. **Training Strategy**

   - Use appropriate batch sizes
   - Monitor individual model performance
   - Adjust learning rates as needed

4. **Weight Management**
   - Monitor weight evolution
   - Consider periodic reweighting
   - Use validation performance

## Advanced Topics

### Custom Base Models

Create specialized models for different conditions:

```python
class MarketRegimeModel(BaseModel):
    def __init__(self, config, regime_type):
        super().__init__(config)
        self.regime_type = regime_type

    def forward(self, x):
        # Add regime-specific logic
        return super().forward(x)
```

### Dynamic Weighting

Implement dynamic weight adjustment:

```python
class DynamicVotingEnsemble(VotingEnsemble):
    def update_weights(self, market_conditions):
        # Adjust weights based on conditions
        self.weights = calculate_weights(
            self.models,
            market_conditions
        )
```

## Troubleshooting

### Common Issues

1. **High Variance**

   ```python
   # Solution: Increase ensemble size
   ensemble = BaggingEnsemble(
       num_models=10,
       bootstrap_ratio=0.8
   )
   ```

2. **Slow Training**

   ```python
   # Solution: Use parallel training
   from torch.multiprocessing import Pool

   with Pool() as p:
       p.map(train_model, ensemble.models)
   ```

3. **Memory Issues**
   ```python
   # Solution: Use model checkpointing
   for model in ensemble.models:
       model.to('cpu')
       torch.save(model.state_dict(), f'model_{i}.pt')
   ```

## Next Steps

1. Experiment with different ensemble types
2. Create custom base models
3. Implement dynamic weighting
4. Monitor and analyze performance

For API details, see the [Ensemble Learning API Reference](../api/ensemble_learning.md).
