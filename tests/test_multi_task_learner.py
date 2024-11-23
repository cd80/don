"""
Tests for multi-task learning functionality.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.multi_task_learner import MultiTaskModel, TaskHead

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "model": {
            "architecture": {
                "input_dim": 10,
                "shared_dim": 64,
                "hidden_dims": [32, 16]
            }
        }
    }

@pytest.fixture
def task_configs():
    """Test task configurations."""
    return {
        "price_prediction": {
            "output_dim": 1,
            "hidden_dims": [32, 16],
            "weight": 1.0
        },
        "volatility_prediction": {
            "output_dim": 1,
            "hidden_dims": [32, 16],
            "weight": 0.5
        },
        "regime_classification": {
            "output_dim": 3,
            "hidden_dims": [32, 16],
            "weight": 0.3
        }
    }

@pytest.fixture
def model(config, task_configs):
    """Initialize model for testing."""
    return MultiTaskModel(
        config=config,
        task_configs=task_configs,
        shared_dim=64,
        uncertainty_weighting=True
    )

@pytest.fixture
def sample_batch():
    """Create sample batch data."""
    batch_size = 32
    features = torch.randn(batch_size, 10)
    targets = {
        "price_prediction": torch.randn(batch_size, 1),
        "volatility_prediction": torch.randn(batch_size, 1),
        "regime_classification": torch.randint(0, 3, (batch_size, 1)).float()
    }
    return features, targets

def test_task_head_initialization():
    """Test task head initialization."""
    head = TaskHead(input_dim=64, output_dim=1, hidden_dims=[32, 16])
    assert isinstance(head, TaskHead)
    assert isinstance(head.network, torch.nn.Sequential)

def test_model_initialization(model, task_configs):
    """Test model initialization."""
    assert isinstance(model, MultiTaskModel)
    assert hasattr(model, 'shared_network')
    assert hasattr(model, 'task_heads')
    assert len(model.task_heads) == len(task_configs)
    
    if model.uncertainty_weighting:
        assert hasattr(model, 'log_vars')
        assert len(model.log_vars) == len(task_configs)

def test_shared_network(model):
    """Test shared network forward pass."""
    batch_size = 32
    features = torch.randn(batch_size, 10)
    shared_features = model.shared_network(features)
    
    assert shared_features.shape == (batch_size, model.shared_dim)

def test_task_heads(model, task_configs):
    """Test task-specific heads."""
    batch_size = 32
    shared_features = torch.randn(batch_size, model.shared_dim)
    
    for task_name, task_config in task_configs.items():
        output = model.task_heads[task_name](shared_features)
        assert output.shape == (batch_size, task_config['output_dim'])

def test_forward_pass(model, sample_batch):
    """Test forward pass through model."""
    features, _ = sample_batch
    outputs = model(features)
    
    assert isinstance(outputs, dict)
    assert len(outputs) == len(model.task_heads)
    
    for task_name, task_output in outputs.items():
        assert isinstance(task_output, torch.Tensor)
        assert task_output.shape[0] == features.shape[0]

def test_loss_computation(model, sample_batch):
    """Test loss computation."""
    features, targets = sample_batch
    predictions = model(features)
    
    total_loss, task_losses = model.compute_losses(predictions, targets)
    
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(task_losses, dict)
    assert len(task_losses) == len(model.task_heads)
    
    for task_name, loss in task_losses.items():
        assert isinstance(loss, float)

def test_train_step(model, sample_batch):
    """Test training step."""
    optimizer = torch.optim.Adam(model.parameters())
    metrics = model.train_step(sample_batch, optimizer)
    
    assert isinstance(metrics, dict)
    assert 'total_loss' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_evaluation(model, sample_batch):
    """Test model evaluation."""
    features, targets = sample_batch
    dataset = TensorDataset(features, targets)
    data_loader = DataLoader(dataset, batch_size=16)
    
    metrics = model.evaluate(data_loader)
    
    assert isinstance(metrics, dict)
    assert all(isinstance(v, float) for v in metrics.values())

def test_task_weights(model):
    """Test task weight computation."""
    weights = model.get_task_weights()
    
    assert isinstance(weights, dict)
    assert len(weights) == len(model.task_heads)
    assert all(isinstance(v, float) for v in weights.values())

def test_checkpoint_save_load(model, tmp_path):
    """Test saving and loading checkpoints."""
    checkpoint_path = tmp_path / "multi_task_model.pt"
    
    # Save checkpoint
    model.save_checkpoint(str(checkpoint_path))
    assert checkpoint_path.exists()
    
    # Load checkpoint
    new_model = MultiTaskModel(
        config=model.config,
        task_configs=model.task_configs,
        shared_dim=model.shared_dim,
        uncertainty_weighting=model.uncertainty_weighting
    )
    new_model.load_checkpoint(str(checkpoint_path))
    
    # Check parameters are equal
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1.data, p2.data)

@pytest.mark.parametrize("uncertainty_weighting", [True, False])
def test_uncertainty_weighting(config, task_configs, uncertainty_weighting):
    """Test model with and without uncertainty weighting."""
    model = MultiTaskModel(
        config=config,
        task_configs=task_configs,
        uncertainty_weighting=uncertainty_weighting
    )
    
    if uncertainty_weighting:
        assert hasattr(model, 'log_vars')
        assert len(model.log_vars) == len(task_configs)
    else:
        assert not hasattr(model, 'log_vars')

def test_selective_task_computation(model, sample_batch):
    """Test computing only specific tasks."""
    features, _ = sample_batch
    selected_tasks = list(model.task_heads.keys())[:2]
    
    outputs = model(features, tasks=selected_tasks)
    
    assert isinstance(outputs, dict)
    assert len(outputs) == len(selected_tasks)
    assert all(task in outputs for task in selected_tasks)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_support(model, sample_batch):
    """Test GPU support if available."""
    model = model.cuda()
    features, targets = sample_batch
    features = features.cuda()
    targets = {k: v.cuda() for k, v in targets.items()}
    
    outputs = model(features)
    assert all(output.is_cuda for output in outputs.values())
    
    total_loss, _ = model.compute_losses(outputs, targets)
    assert total_loss.is_cuda

def test_gradient_flow(model, sample_batch):
    """Test gradient flow through the model."""
    features, targets = sample_batch
    optimizer = torch.optim.Adam(model.parameters())
    
    # Initial parameters
    initial_params = [p.clone().detach() for p in model.parameters()]
    
    # Training step
    metrics = model.train_step(sample_batch, optimizer)
    
    # Check parameters have been updated
    for p1, p2 in zip(initial_params, model.parameters()):
        assert not torch.equal(p1, p2.data)

def test_batch_normalization(model, sample_batch):
    """Test batch normalization behavior."""
    model.train()
    features, _ = sample_batch
    
    # Training mode
    train_output = model(features)
    
    model.eval()
    # Evaluation mode
    eval_output = model(features)
    
    # Outputs should be different in train and eval modes
    for task_name in train_output.keys():
        assert not torch.equal(train_output[task_name], eval_output[task_name])
