"""
Tests for meta-learning functionality.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.meta_learner import MAMLModel

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "model": {
            "type": "hierarchical_rl",
            "architecture": {
                "actor_hidden_layers": [64, 32],
                "critic_hidden_layers": [64, 32],
                "attention_heads": 4,
                "transformer_layers": 2
            }
        },
        "training": {
            "inner_lr": 0.01,
            "meta_lr": 0.001,
            "num_inner_steps": 3,
            "task_batch_size": 16
        }
    }

@pytest.fixture
def model(config):
    """Initialize model for testing."""
    return MAMLModel(
        config,
        inner_lr=config["training"]["inner_lr"],
        meta_lr=config["training"]["meta_lr"],
        num_inner_steps=config["training"]["num_inner_steps"],
        task_batch_size=config["training"]["task_batch_size"]
    )

@pytest.fixture
def sample_task():
    """Create sample task data."""
    features = torch.randn(32, 10)  # 32 samples, 10 features
    targets = torch.randn(32, 1)    # 32 samples, 1 target
    return features, targets

@pytest.fixture
def task_generator():
    """Create task generator for meta-learning."""
    def generate_task():
        # Create support set
        support_features = torch.randn(32, 10)
        support_targets = torch.randn(32, 1)
        support_set = (support_features, support_targets)
        
        # Create query set
        query_features = torch.randn(32, 10)
        query_targets = torch.randn(32, 1)
        query_set = (query_features, query_targets)
        
        return support_set, query_set
    
    return generate_task

def test_model_initialization(model):
    """Test model initialization."""
    assert isinstance(model, MAMLModel)
    assert hasattr(model, 'inner_lr')
    assert hasattr(model, 'meta_lr')
    assert hasattr(model, 'num_inner_steps')
    assert hasattr(model, 'meta_optimizer')

def test_model_cloning(model):
    """Test model cloning functionality."""
    clone = model.clone_model()
    assert isinstance(clone, MAMLModel)
    
    # Check parameters are equal but not the same object
    for p1, p2 in zip(model.parameters(), clone.parameters()):
        assert torch.equal(p1.data, p2.data)
        assert p1 is not p2

def test_inner_loop_optimization(model, sample_task):
    """Test inner loop optimization."""
    features, targets = sample_task
    metrics, adapted_model = model.inner_loop(
        support_data=(features, targets)
    )
    
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert isinstance(adapted_model, MAMLModel)
    
    # Check parameters have been updated
    for p1, p2 in zip(model.parameters(), adapted_model.parameters()):
        assert not torch.equal(p1.data, p2.data)

def test_meta_learning(model, task_generator):
    """Test meta-learning process."""
    num_tasks = 5
    num_epochs = 2
    
    history = model.meta_learn(
        task_generator=task_generator,
        num_tasks=num_tasks,
        num_epochs=num_epochs
    )
    
    assert isinstance(history, dict)
    assert 'meta_loss' in history
    assert 'inner_losses' in history
    assert 'adaptation_metrics' in history
    assert len(history['meta_loss']) == num_epochs

def test_market_adaptation(model, sample_task):
    """Test adaptation to market conditions."""
    features, targets = sample_task
    adapted_model = model.adapt_to_market(
        market_data=(features, targets)
    )
    
    assert isinstance(adapted_model, MAMLModel)
    
    # Test prediction with adapted model
    with torch.no_grad():
        predictions, _ = adapted_model(features)
        assert predictions.shape == targets.shape

def test_forward_pass(model, sample_task):
    """Test forward pass with and without targets."""
    features, targets = sample_task
    
    # Test with targets
    loss, metrics = model(features, targets)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    
    # Test without targets
    predictions, metrics = model(features)
    assert predictions.shape == targets.shape
    assert isinstance(metrics, dict)
    assert len(metrics) == 0

def test_accuracy_calculation(model):
    """Test accuracy calculation."""
    predictions = torch.tensor([[0.1], [0.5], [0.9]])
    targets = torch.tensor([[0.15], [0.8], [0.85]])
    
    accuracy = model.calculate_accuracy(predictions, targets)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_save_and_load(model, tmp_path):
    """Test saving and loading meta-learned model."""
    # Save model
    save_path = tmp_path / "meta_model.pt"
    model.save_meta_learned(str(save_path))
    assert save_path.exists()
    
    # Load model
    new_model = MAMLModel(model.config)
    new_model.load_meta_learned(str(save_path))
    
    # Check parameters are equal
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1.data, p2.data)
    
    # Check configuration
    assert model.inner_lr == new_model.inner_lr
    assert model.meta_lr == new_model.meta_lr
    assert model.num_inner_steps == new_model.num_inner_steps

@pytest.mark.parametrize("num_steps", [1, 3, 5])
def test_variable_adaptation_steps(model, sample_task, num_steps):
    """Test adaptation with different numbers of steps."""
    features, targets = sample_task
    adapted_model = model.adapt_to_market(
        market_data=(features, targets),
        num_steps=num_steps
    )
    
    assert isinstance(adapted_model, MAMLModel)

def test_meta_learning_convergence(model, task_generator):
    """Test that meta-learning reduces loss over time."""
    num_tasks = 5
    num_epochs = 5
    
    history = model.meta_learn(
        task_generator=task_generator,
        num_tasks=num_tasks,
        num_epochs=num_epochs
    )
    
    # Check if loss decreases
    meta_losses = history['meta_loss']
    assert meta_losses[-1] < meta_losses[0]

@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_different_batch_sizes(config, batch_size):
    """Test model with different batch sizes."""
    config["training"]["task_batch_size"] = batch_size
    model = MAMLModel(config)
    
    features = torch.randn(batch_size, 10)
    targets = torch.randn(batch_size, 1)
    
    metrics, adapted_model = model.inner_loop(
        support_data=(features, targets)
    )
    
    assert isinstance(metrics, dict)
    assert isinstance(adapted_model, MAMLModel)

def test_gpu_support(model, sample_task):
    """Test GPU support if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = model.cuda()
    features, targets = sample_task
    features = features.cuda()
    targets = targets.cuda()
    
    metrics, adapted_model = model.inner_loop(
        support_data=(features, targets)
    )
    
    assert next(model.parameters()).is_cuda
    assert next(adapted_model.parameters()).is_cuda
