# Transfer Learning Guide

This guide explains how to use the transfer learning capabilities of the Bitcoin Trading RL project to leverage pre-trained knowledge for new trading tasks.

## Overview

Transfer learning enables the model to:

- Leverage knowledge from related trading tasks
- Adapt quickly to new market conditions
- Reduce training time and data requirements
- Improve performance on target tasks

## Quick Start

```python
from src.models.transfer_learner import TransferModel

# Initialize transfer model
model = TransferModel(
    config=config,
    source_task="price_prediction",
    target_task="volatility_prediction",
    transfer_config={
        "layer_dims": [128, 64, 32],
        "head_dim": 16
    }
)

# Train model
history = model.transfer_learn(
    source_data=source_loader,
    target_data=target_loader,
    num_epochs=100,
    fine_tune=True
)
```

## Configuration

Configure transfer learning in `configs/config.yaml`:

```yaml
model:
  transfer_learning:
    enabled: true
    shared_architecture:
      layer_dims: [128, 64, 32]
      head_dim: 16
      batch_norm: true
      dropout: 0.1

    training:
      source_epochs: 100
      target_epochs: 50
      fine_tune_epochs: 30
      learning_rate: 0.001
      fine_tune_lr: 0.0001
      batch_size: 64

    tasks:
      source:
        name: "price_prediction"
        type: "regression"
        loss: "mse"
      target:
        name: "volatility_prediction"
        type: "regression"
        loss: "mse"
```

## Task Design

### 1. Source Task Selection

Choose appropriate source tasks:

```python
def create_price_prediction_task(data):
    """Create price prediction source task."""
    return {
        'features': extract_features(data),
        'targets': calculate_returns(data),
        'task_type': 'regression'
    }
```

### 2. Target Task Design

Design target tasks that can benefit from source knowledge:

```python
def create_volatility_prediction_task(data):
    """Create volatility prediction target task."""
    return {
        'features': extract_features(data),
        'targets': calculate_volatility(data),
        'task_type': 'regression'
    }
```

## Training Process

### 1. Pre-training

Train on source task:

```python
# Initialize data loaders
source_loader = create_data_loader(source_data)
target_loader = create_data_loader(target_data)

# Pre-train on source task
history = model.transfer_learn(
    source_data=source_loader,
    target_data=target_loader,
    num_epochs=100
)
```

### 2. Transfer Learning

Transfer to target task:

```python
# Freeze shared layers
model.freeze_shared_layers()

# Train target task head
history = model.transfer_learn(
    source_data=source_loader,
    target_data=target_loader,
    num_epochs=50,
    fine_tune=False
)
```

### 3. Fine-tuning

Fine-tune the entire model:

```python
# Unfreeze shared layers
model.unfreeze_shared_layers()

# Fine-tune
history = model.transfer_learn(
    source_data=source_loader,
    target_data=target_loader,
    num_epochs=30,
    fine_tune=True,
    learning_rate=0.0001
)
```

## Performance Monitoring

### Training Progress

Monitor training metrics:

```python
def plot_transfer_learning_progress(history):
    plt.figure(figsize=(15, 5))

    # Source task training
    plt.subplot(1, 3, 1)
    plt.plot(history['source_loss'])
    plt.title('Source Task Training')

    # Target task training
    plt.subplot(1, 3, 2)
    plt.plot(history['target_loss'])
    plt.title('Target Task Training')

    # Fine-tuning
    plt.subplot(1, 3, 3)
    plt.plot(history['fine_tune_loss'])
    plt.title('Fine-tuning')

    plt.tight_layout()
    plt.show()
```

### Layer Analysis

Analyze layer gradients:

```python
def analyze_layer_gradients(model):
    gradients = model.get_layer_gradients()

    for layer_name, grad in gradients.items():
        print(f"{layer_name}:")
        print(f"  Mean: {grad.mean():.4f}")
        print(f"  Std: {grad.std():.4f}")
```

## Best Practices

1. **Source Task Selection**

   - Choose related tasks
   - Ensure sufficient data
   - Consider task complexity

2. **Architecture Design**

   - Size shared layers appropriately
   - Add task-specific layers
   - Use batch normalization

3. **Training Strategy**

   - Start with frozen layers
   - Gradually unfreeze
   - Use appropriate learning rates

4. **Fine-tuning**
   - Monitor performance
   - Prevent catastrophic forgetting
   - Use early stopping

## Advanced Topics

### Custom Layer Design

Create custom transferable layers:

```python
class CustomTransferableLayer(TransferableLayer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, 4)

    def forward(self, x):
        x = super().forward(x)
        x, _ = self.attention(x, x, x)
        return x
```

### Progressive Transfer

Implement progressive layer unfreezing:

```python
def progressive_unfreeze(model, epochs_per_layer=10):
    """Progressively unfreeze layers from top to bottom."""
    for i, layer in enumerate(reversed(model.shared_layers)):
        layer.unfreeze()
        yield i, layer
```

## Troubleshooting

### Common Issues

1. **Catastrophic Forgetting**

   ```python
   # Solution: Use gradual unfreezing
   for layer_idx, layer in progressive_unfreeze(model):
       train_epoch(model, data_loader)
   ```

2. **Poor Transfer**

   ```python
   # Solution: Adjust layer freezing
   model.freeze_shared_layers()
   model.shared_layers[-1].unfreeze()  # Only unfreeze last layer
   ```

3. **Slow Convergence**
   ```python
   # Solution: Use layer-specific learning rates
   optimizers = {
       'shared': Adam(model.shared_layers.parameters(), lr=0.0001),
       'target': Adam(model.target_head.parameters(), lr=0.001)
   }
   ```

## Next Steps

1. Experiment with different source tasks
2. Try different layer architectures
3. Implement custom transfer strategies
4. Monitor and analyze transfer effectiveness

For API details, see the [Transfer Learning API Reference](../api/transfer_learning.md).
