# Curriculum Learning Guide

This guide explains how to use the curriculum learning capabilities of the Bitcoin Trading RL project to progressively train models from simple to complex trading tasks.

## Overview

Curriculum learning enables:

- Progressive learning from simple to complex tasks
- Adaptive task selection based on performance
- Customizable learning paths
- Performance monitoring and progression tracking
- Efficient knowledge transfer

## Quick Start

```python
from src.training.curriculum_learner import (
    CurriculumLearner,
    TaskConfig,
    CurriculumTask,
    TaskDifficulty
)

# Define tasks
tasks = [
    CurriculumTask(
        config=TaskConfig(
            difficulty=TaskDifficulty.EASY,
            market_volatility=(0.01, 0.02),
            trend_strength=(0.01, 0.02),
            time_horizon=10,
            position_constraints={'max_size': 0.5},
            required_accuracy=0.6,
            sampling_weight=1.0
        ),
        data_generator=generate_data,
        performance_metrics=['accuracy', 'profit']
    )
]

# Initialize learner
learner = CurriculumLearner(
    config=config,
    model=model,
    tasks=tasks
)

# Train with curriculum
history = learner.train(
    num_steps=1000,
    optimizer=optimizer
)
```

## Configuration

Configure curriculum learning in `configs/config.yaml`:

```yaml
curriculum_learning:
  enabled: true

  scheduler:
    type: "performance_based" # performance_based, sequential, or adaptive
    temperature: 1.0 # Temperature for task sampling
    eval_frequency: 100 # Steps between evaluations

  tasks:
    very_easy:
      market_volatility: [0.01, 0.02]
      trend_strength: [0.01, 0.02]
      time_horizon: 10
      position_constraints:
        max_size: 0.5
      required_accuracy: 0.6
      sampling_weight: 1.0

    easy:
      market_volatility: [0.02, 0.03]
      trend_strength: [0.02, 0.03]
      time_horizon: 20
      position_constraints:
        max_size: 0.7
      required_accuracy: 0.55
      sampling_weight: 0.8

    medium:
      market_volatility: [0.03, 0.04]
      trend_strength: [0.03, 0.04]
      time_horizon: 30
      position_constraints:
        max_size: 1.0
      required_accuracy: 0.5
      sampling_weight: 0.6

    hard:
      market_volatility: [0.04, 0.05]
      trend_strength: [0.04, 0.05]
      time_horizon: 50
      position_constraints:
        max_size: 1.5
      required_accuracy: 0.45
      sampling_weight: 0.4
```

## Task Design

### 1. Task Configuration

Define task parameters:

```python
def create_task_config(difficulty: TaskDifficulty) -> TaskConfig:
    """Create task configuration."""
    if difficulty == TaskDifficulty.EASY:
        return TaskConfig(
            difficulty=difficulty,
            market_volatility=(0.01, 0.02),
            trend_strength=(0.01, 0.02),
            time_horizon=10,
            position_constraints={'max_size': 0.5},
            required_accuracy=0.6,
            sampling_weight=1.0
        )
    # Add more difficulty levels...
```

### 2. Data Generation

Create task-specific data:

```python
def generate_task_data(
    volatility_range: tuple,
    trend_range: tuple,
    time_horizon: int
) -> tuple:
    """Generate task data."""
    # Generate features
    features = generate_market_features(
        volatility=np.random.uniform(*volatility_range),
        trend=np.random.uniform(*trend_range),
        horizon=time_horizon
    )

    # Generate targets
    targets = calculate_optimal_positions(features)

    return features, targets
```

## Training Process

### 1. Task Progression

Monitor task progression:

```python
def monitor_progression(learner: CurriculumLearner):
    """Monitor learning progression."""
    status = learner.get_curriculum_status()

    print("Completed Tasks:")
    for task in status['completed_tasks']:
        print(f"- {task}")

    print("\nTask History:")
    for task, perf in zip(status['task_history'],
                         status['performance_history']):
        print(f"{task}: {perf['accuracy']:.2f}")
```

### 2. Performance Monitoring

Track performance metrics:

```python
def plot_learning_progress(history: dict):
    """Plot learning progress."""
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')

    # Plot task difficulty
    plt.subplot(1, 3, 2)
    plt.plot(history['task_difficulty'])
    plt.title('Task Difficulty')

    # Plot performance
    plt.subplot(1, 3, 3)
    plt.plot(history['performance'])
    plt.title('Performance')

    plt.tight_layout()
    plt.show()
```

## Best Practices

1. **Task Design**

   - Start simple
   - Gradual complexity increase
   - Clear progression path

2. **Performance Thresholds**

   - Set realistic targets
   - Adjust dynamically
   - Monitor completion rates

3. **Scheduling Strategy**

   - Choose appropriate strategy
   - Tune temperature parameter
   - Monitor task selection

4. **Data Generation**
   - Ensure task consistency
   - Cover edge cases
   - Validate data quality

## Advanced Topics

### Custom Task Generation

Create specialized tasks:

```python
class CustomTask(CurriculumTask):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.custom_params = {}

    def generate_data(self) -> tuple:
        # Custom data generation logic
        return features, targets
```

### Adaptive Scheduling

Implement custom scheduling:

```python
class CustomScheduler(CurriculumScheduler):
    def __init__(self, tasks: List[CurriculumTask]):
        super().__init__(tasks)
        self.custom_metrics = []

    def select_task(self) -> CurriculumTask:
        # Custom task selection logic
        return selected_task
```

## Troubleshooting

### Common Issues

1. **Slow Progress**

   ```python
   # Solution: Adjust difficulty progression
   def adjust_difficulty(task: CurriculumTask):
       if task.completion_rate < 0.2:
           task.config.required_accuracy *= 0.9
   ```

2. **Task Stagnation**

   ```python
   # Solution: Implement task switching
   def force_task_switch(scheduler: CurriculumScheduler):
       if scheduler.consecutive_same_task > 10:
           return scheduler.select_different_task()
   ```

3. **Poor Generalization**
   ```python
   # Solution: Add task variety
   def add_task_variations(task: CurriculumTask):
       variations = create_task_variations(task)
       return variations
   ```

## Next Steps

1. Design custom tasks
2. Implement adaptive scheduling
3. Create monitoring dashboards
4. Optimize progression paths

For API details, see the [Curriculum Learning API Reference](../api/curriculum_learning.md).
