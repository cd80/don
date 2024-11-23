"""
Tests for curriculum learning functionality.
"""

import pytest
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from src.training.curriculum_learner import (
    TaskDifficulty,
    TaskConfig,
    CurriculumTask,
    CurriculumScheduler,
    CurriculumLearner
)

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'curriculum': {
            'scheduler_type': 'performance_based',
            'temperature': 1.0,
            'eval_frequency': 10
        }
    }

@pytest.fixture
def task_configs():
    """Test task configurations."""
    return [
        TaskConfig(
            difficulty=TaskDifficulty.VERY_EASY,
            market_volatility=(0.01, 0.02),
            trend_strength=(0.01, 0.02),
            time_horizon=10,
            position_constraints={'max_size': 0.5},
            required_accuracy=0.6,
            sampling_weight=1.0
        ),
        TaskConfig(
            difficulty=TaskDifficulty.EASY,
            market_volatility=(0.02, 0.03),
            trend_strength=(0.02, 0.03),
            time_horizon=20,
            position_constraints={'max_size': 0.7},
            required_accuracy=0.55,
            sampling_weight=0.8
        ),
        TaskConfig(
            difficulty=TaskDifficulty.MEDIUM,
            market_volatility=(0.03, 0.04),
            trend_strength=(0.03, 0.04),
            time_horizon=30,
            position_constraints={'max_size': 1.0},
            required_accuracy=0.5,
            sampling_weight=0.6
        )
    ]

def generate_test_data(
    volatility_range: tuple,
    trend_range: tuple,
    time_horizon: int
) -> tuple:
    """Generate test data."""
    batch_size = 32
    features = torch.randn(batch_size, 10)
    targets = torch.randn(batch_size, 1)
    return features, targets

@pytest.fixture
def tasks(task_configs):
    """Create test tasks."""
    return [
        CurriculumTask(
            config=config,
            data_generator=generate_test_data,
            performance_metrics=['accuracy', 'mse', 'profit']
        )
        for config in task_configs
    ]

@pytest.fixture
def scheduler(tasks):
    """Initialize scheduler for testing."""
    return CurriculumScheduler(
        tasks=tasks,
        strategy='performance_based',
        temperature=1.0
    )

@pytest.fixture
def learner(config, tasks):
    """Initialize learner for testing."""
    model = SimpleModel()
    return CurriculumLearner(
        config=config,
        model=model,
        tasks=tasks,
        scheduler_type='performance_based'
    )

def test_task_config_initialization(task_configs):
    """Test task configuration initialization."""
    config = task_configs[0]
    assert isinstance(config.difficulty, TaskDifficulty)
    assert len(config.market_volatility) == 2
    assert len(config.trend_strength) == 2
    assert isinstance(config.time_horizon, int)
    assert isinstance(config.position_constraints, dict)
    assert 0 <= config.required_accuracy <= 1
    assert config.sampling_weight > 0

def test_curriculum_task(task_configs):
    """Test curriculum task functionality."""
    task = CurriculumTask(
        config=task_configs[0],
        data_generator=generate_test_data,
        performance_metrics=['accuracy', 'mse', 'profit']
    )
    
    # Test data generation
    features, targets = task.generate_data()
    assert isinstance(features, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    
    # Test performance evaluation
    predictions = torch.randn_like(targets)
    metrics = task.evaluate_performance(predictions, targets)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'mse' in metrics
    assert 'profit' in metrics

def test_scheduler_initialization(scheduler, tasks):
    """Test scheduler initialization."""
    assert len(scheduler.tasks) == len(tasks)
    assert scheduler.strategy == 'performance_based'
    assert scheduler.temperature == 1.0

def test_task_selection(scheduler):
    """Test task selection strategies."""
    # Test sequential selection
    scheduler.strategy = 'sequential'
    task = scheduler.select_task()
    assert isinstance(task, CurriculumTask)
    
    # Test performance-based selection
    scheduler.strategy = 'performance_based'
    task = scheduler.select_task()
    assert isinstance(task, CurriculumTask)
    
    # Test adaptive selection
    scheduler.strategy = 'adaptive'
    task = scheduler.select_task()
    assert isinstance(task, CurriculumTask)

def test_curriculum_learner_initialization(learner):
    """Test curriculum learner initialization."""
    assert isinstance(learner.model, nn.Module)
    assert len(learner.tasks) > 0
    assert isinstance(learner.scheduler, CurriculumScheduler)

def test_training_step(learner):
    """Test single training step."""
    optimizer = torch.optim.Adam(learner.model.parameters())
    metrics = learner.train_step(optimizer)
    
    assert isinstance(metrics, dict)
    assert 'task' in metrics
    assert 'loss' in metrics
    assert 'accuracy' in metrics

def test_full_training(learner):
    """Test full training process."""
    optimizer = torch.optim.Adam(learner.model.parameters())
    history = learner.train(
        num_steps=10,
        optimizer=optimizer,
        eval_frequency=5
    )
    
    assert isinstance(history, dict)
    assert 'loss' in history
    assert 'task_difficulty' in history
    assert 'performance' in history
    assert len(history['loss']) == 10

def test_curriculum_status(learner):
    """Test curriculum status tracking."""
    status = learner.get_curriculum_status()
    
    assert isinstance(status, dict)
    assert 'completed_tasks' in status
    assert 'task_history' in status
    assert 'performance_history' in status

def test_save_load_state(learner, tmp_path):
    """Test saving and loading curriculum state."""
    save_path = tmp_path / "curriculum_state.pt"
    
    # Save state
    learner.save_curriculum_state(str(save_path))
    assert save_path.exists()
    
    # Load state
    new_learner = CurriculumLearner(
        config=learner.config,
        model=SimpleModel(),
        tasks=learner.tasks
    )
    new_learner.load_curriculum_state(str(save_path))
    
    # Check state restoration
    assert len(new_learner.scheduler.task_history) == \
           len(learner.scheduler.task_history)
    assert len(new_learner.scheduler.performance_history) == \
           len(learner.scheduler.performance_history)

def test_performance_based_selection(scheduler):
    """Test performance-based task selection."""
    # Simulate task completion
    for task in scheduler.tasks[:2]:
        task.completion_history.append({'accuracy': 0.9})
        task.current_performance = {'accuracy': 0.9}
    
    # Select task
    selected_task = scheduler.select_task()
    
    # Should prefer incomplete tasks
    assert selected_task.config.difficulty == TaskDifficulty.MEDIUM

def test_adaptive_selection(scheduler):
    """Test adaptive task selection."""
    scheduler.strategy = 'adaptive'
    
    # Simulate learning progress
    for task in scheduler.tasks:
        task.completion_history.extend([
            {'accuracy': 0.5},
            {'accuracy': 0.7}
        ])
        task.current_performance = {'accuracy': 0.7}
    
    # Select task
    selected_task = scheduler.select_task()
    assert isinstance(selected_task, CurriculumTask)

def test_temperature_adaptation(scheduler):
    """Test temperature adaptation in scheduler."""
    initial_temp = scheduler.temperature
    
    # Simulate good performance
    for _ in range(10):
        scheduler.update_progress(
            scheduler.tasks[0],
            {'accuracy': 0.9}
        )
    
    # Temperature should decrease
    assert scheduler.temperature < initial_temp
    
    # Simulate poor performance
    for _ in range(10):
        scheduler.update_progress(
            scheduler.tasks[0],
            {'accuracy': 0.4}
        )
    
    # Temperature should increase
    assert scheduler.temperature > initial_temp

@pytest.mark.parametrize("difficulty", list(TaskDifficulty))
def test_task_difficulty_levels(task_configs, difficulty):
    """Test different task difficulty levels."""
    config = TaskConfig(
        difficulty=difficulty,
        market_volatility=(0.01, 0.02),
        trend_strength=(0.01, 0.02),
        time_horizon=10,
        position_constraints={'max_size': 0.5},
        required_accuracy=0.6,
        sampling_weight=1.0
    )
    
    task = CurriculumTask(
        config=config,
        data_generator=generate_test_data,
        performance_metrics=['accuracy']
    )
    
    assert task.config.difficulty == difficulty
