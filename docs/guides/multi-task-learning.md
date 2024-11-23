# Multi-Task Learning Guide

This guide explains how to use the multi-task learning capabilities of the Bitcoin Trading RL project to simultaneously optimize multiple trading objectives.

## Overview

Multi-task learning enables the model to:

- Learn shared representations across different trading tasks
- Improve generalization through related tasks
- Balance multiple trading objectives
- Transfer knowledge between tasks

## Quick Start

```python
from src.models.multi_task_learner import MultiTaskModel

# Define task configurations
task_configs = {
    "price_prediction": {
        "output_dim": 1,
        "hidden_dims": [64, 32],
        "weight": 1.0
    },
    "volatility_prediction": {
        "output_dim": 1,
        "hidden_dims": [64, 32],
        "weight": 0.5
    },
    "regime_classification": {
        "output_dim": 3,
        "hidden_dims": [64, 32],
        "weight": 0.3
    }
}

# Initialize multi-task model
model = MultiTaskModel(
    config=config,
    task_configs=task_configs,
    shared_dim=128,
    uncertainty_weighting=True
)

# Train model
for batch in data_loader:
    features, targets = batch
    metrics = model.train_step(batch, optimizer)
```

## Configuration

Configure multi-task learning in `configs/config.yaml`:

```yaml
model:
  multi_task:
    enabled: true
    shared_dim: 128
    uncertainty_weighting: true

    tasks:
      price_prediction:
        output_dim: 1
        hidden_dims: [64, 32]
        weight: 1.0
        loss: "mse"
        metrics: ["mae", "rmse"]

      volatility_prediction:
        output_dim: 1
        hidden_dims: [64, 32]
        weight: 0.5
        loss: "mse"
        metrics: ["mae"]

      regime_classification:
        output_dim: 3
        hidden_dims: [64, 32]
        weight: 0.3
        loss: "cross_entropy"
        metrics: ["accuracy"]

    optimization:
      batch_size: 64
      learning_rate: 0.001
      gradient_clipping: 1.0
      scheduler:
        type: "cosine"
        warmup_steps: 1000
```

## Task Design

### 1. Price Prediction Task

```python
def create_price_prediction_task(data, window_size=100):
    """Create price prediction task data."""
    features = data['features'].values
    targets = data['close'].pct_change().shift(-1).values

    return {
        'features': features,
        'targets': targets,
        'task_name': 'price_prediction'
    }
```

### 2. Volatility Prediction Task

```python
def create_volatility_prediction_task(data, window_size=20):
    """Create volatility prediction task data."""
    returns = data['close'].pct_change()
    realized_vol = returns.rolling(window=window_size).std()

    return {
        'features': data['features'].values,
        'targets': realized_vol.values,
        'task_name': 'volatility_prediction'
    }
```

### 3. Regime Classification Task

```python
def create_regime_classification_task(data):
    """Create market regime classification task data."""
    regimes = calculate_market_regimes(data)

    return {
        'features': data['features'].values,
        'targets': regimes,
        'task_name': 'regime_classification'
    }
```

## Training Process

### 1. Data Preparation

```python
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, tasks_data):
        self.tasks_data = tasks_data

    def __len__(self):
        return len(self.tasks_data['features'])

    def __getitem__(self, idx):
        features = self.tasks_data['features'][idx]
        targets = {
            task: self.tasks_data[task]['targets'][idx]
            for task in self.tasks_data['tasks']
        }
        return features, targets

# Create dataset
dataset = MultiTaskDataset(tasks_data)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### 2. Training Loop

```python
def train_multi_task_model(model, data_loader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        epoch_metrics = []

        for batch in data_loader:
            features, targets = batch
            metrics = model.train_step(
                (features, targets),
                optimizer
            )
            epoch_metrics.append(metrics)

        # Log metrics
        avg_metrics = average_metrics(epoch_metrics)
        print(f"Epoch {epoch}: {avg_metrics}")
```

### 3. Evaluation

```python
def evaluate_model(model, data_loader):
    metrics = model.evaluate(data_loader)

    # Analyze task-specific performance
    for task_name, task_metrics in metrics.items():
        print(f"{task_name} Performance:")
        for metric_name, value in task_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
```

## Performance Monitoring

### Task Weights

Monitor task weights during training:

```python
def plot_task_weights(model, history):
    weights = model.get_task_weights()

    plt.figure(figsize=(10, 5))
    for task, weight_history in weights.items():
        plt.plot(weight_history, label=task)
    plt.title('Task Weights Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.legend()
    plt.show()
```

### Loss Analysis

```python
def analyze_losses(history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.title('Total Loss')

    plt.subplot(1, 2, 2)
    for task, losses in history['task_losses'].items():
        plt.plot(losses, label=task)
    plt.title('Task-Specific Losses')
    plt.legend()

    plt.tight_layout()
    plt.show()
```

## Best Practices

1. **Task Selection**

   - Choose related tasks that share useful features
   - Balance task difficulties
   - Consider task relationships

2. **Loss Balancing**

   - Use uncertainty weighting for automatic balancing
   - Monitor task weights evolution
   - Adjust manual weights if needed

3. **Architecture Design**

   - Size shared layers appropriately
   - Add task-specific layers as needed
   - Consider task complexity in head design

4. **Training Strategy**
   - Use appropriate batch sizes
   - Implement learning rate scheduling
   - Monitor task-specific metrics

## Advanced Topics

### Custom Task Heads

Create custom task-specific architectures:

```python
class CustomTaskHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)
```

### Dynamic Task Scheduling

Implement dynamic task sampling:

```python
class TaskScheduler:
    def __init__(self, tasks, schedule_type="uniform"):
        self.tasks = tasks
        self.schedule_type = schedule_type

    def sample_task(self, step):
        if self.schedule_type == "uniform":
            return random.choice(self.tasks)
        elif self.schedule_type == "curriculum":
            return self.get_curriculum_task(step)
```

## Troubleshooting

### Common Issues

1. **Task Dominance**

   ```python
   # Solution: Adjust task weights or use uncertainty weighting
   model = MultiTaskModel(
       task_configs=task_configs,
       uncertainty_weighting=True
   )
   ```

2. **Poor Convergence**

   ```python
   # Solution: Adjust learning rates per task
   optimizer = torch.optim.Adam([
       {'params': model.shared_network.parameters(), 'lr': 0.001},
       {'params': model.task_heads.parameters(), 'lr': 0.01}
   ])
   ```

3. **Memory Issues**
   ```python
   # Solution: Use gradient accumulation
   for i, batch in enumerate(data_loader):
       loss = model.train_step(batch, optimizer)
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

## Next Steps

1. Experiment with different task combinations
2. Implement custom task heads
3. Try different loss balancing strategies
4. Monitor and analyze task relationships

For API details, see the [Multi-Task Learning API Reference](../api/multi_task_learning.md).
