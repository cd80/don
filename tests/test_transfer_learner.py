"""
Tests for transfer learning functionality.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.transfer_learner import TransferModel, TransferableLayer

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "model": {
            "input_dim": 10,
            "output_dim": 1,
            "hidden_dims": [64, 32]
        }
    }

@pytest.fixture
def transfer_config():
    """Test transfer configuration."""
    return {
        "layer_dims": [128, 64, 32],
        "head_dim": 16,
        "layer_config": {
            "batch_norm": True,
            "dropout": 0.1
        }
    }

@pytest.fixture
def model(config, transfer_config):
    """Initialize model for testing."""
    return TransferModel(
        config=config,
        source_task="price_prediction",
        target_task="volatility_prediction",
        transfer_config=transfer_config
    )

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    batch_size = 32
    features = torch.randn(batch_size, 10)
    source_targets = torch.randn(batch_size, 1)
    target_targets = torch.randn(batch_size, 1)
    
    source_dataset = TensorDataset(features, source_targets)
    target_dataset = TensorDataset(features, target_targets)
    
    source_loader = DataLoader(source_dataset, batch_size=16)
    target_loader = DataLoader(target_dataset, batch_size=16)
    
    return source_loader, target_loader

def test_transferable_layer():
    """Test transferable layer functionality."""
    layer = TransferableLayer(64, 32)
    
    # Test forward pass
    x = torch.randn(16, 64)
    output = layer(x)
    assert output.shape == (16, 32)
    
    # Test freezing
    layer.freeze()
    assert layer.frozen
    assert not any(p.requires_grad for p in layer.parameters())
    
    # Test unfreezing
    layer.unfreeze()
    assert not layer.frozen
    assert all(p.requires_grad for p in layer.parameters())

def test_model_initialization(model):
    """Test model initialization."""
    assert isinstance(model, TransferModel)
    assert hasattr(model, 'shared_layers')
    assert hasattr(model, 'source_head')
    assert hasattr(model, 'target_head')

def test_shared_layers(model):
    """Test shared layers structure."""
    assert len(model.shared_layers) > 0
    assert all(isinstance(layer, TransferableLayer)
              for layer in model.shared_layers)

def test_forward_pass(model):
    """Test forward pass through model."""
    batch_size = 16
    features = torch.randn(batch_size, 10)
    
    # Test source task
    source_output = model(features, task="price_prediction")
    assert source_output.shape == (batch_size, 1)
    
    # Test target task
    target_output = model(features, task="volatility_prediction")
    assert target_output.shape == (batch_size, 1)

def test_layer_freezing(model):
    """Test layer freezing functionality."""
    # Test freezing
    model.freeze_shared_layers()
    assert all(layer.frozen for layer in model.shared_layers)
    assert not any(p.requires_grad for layer in model.shared_layers
                  for p in layer.parameters())
    
    # Test unfreezing
    model.unfreeze_shared_layers()
    assert not any(layer.frozen for layer in model.shared_layers)
    assert all(p.requires_grad for layer in model.shared_layers
              for p in layer.parameters())

def test_trainable_params(model):
    """Test getting trainable parameters."""
    # All parameters trainable
    params = model.get_trainable_params()
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
    
    # Freeze shared layers
    model.freeze_shared_layers()
    params = model.get_trainable_params()
    assert len(params) > 0  # Task heads still trainable
    
    # Verify shared layers not in trainable params
    shared_params = set(p for layer in model.shared_layers
                       for p in layer.parameters())
    trainable_params = set(params)
    assert not (shared_params & trainable_params)

def test_transfer_learning(model, sample_data):
    """Test transfer learning process."""
    source_loader, target_loader = sample_data
    
    history = model.transfer_learn(
        source_data=source_loader,
        target_data=target_loader,
        num_epochs=2,
        fine_tune=True,
        fine_tune_epochs=2
    )
    
    assert isinstance(history, dict)
    assert 'source_loss' in history
    assert 'target_loss' in history
    assert 'fine_tune_loss' in history
    assert all(len(losses) > 0 for losses in history.values())

def test_save_load(model, tmp_path):
    """Test saving and loading model."""
    save_path = tmp_path / "transfer_model.pt"
    
    # Save model
    model.save_transfer_model(str(save_path))
    assert save_path.exists()
    
    # Load model
    new_model = TransferModel(
        config=model.config,
        source_task=model.source_task,
        target_task=model.target_task,
        transfer_config=model.transfer_config
    )
    new_model.load_transfer_model(str(save_path))
    
    # Check parameters are equal
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1.data, p2.data)
    
    # Check layer states are preserved
    for l1, l2 in zip(model.shared_layers, new_model.shared_layers):
        assert l1.frozen == l2.frozen

def test_gradient_computation(model):
    """Test gradient computation."""
    features = torch.randn(16, 10)
    targets = torch.randn(16, 1)
    
    # Forward and backward pass
    predictions = model(features)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    
    # Get gradients
    gradients = model.get_layer_gradients()
    assert isinstance(gradients, dict)
    assert len(gradients) > 0
    assert all(isinstance(grad, torch.Tensor) for grad in gradients.values())

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_support(model, sample_data):
    """Test GPU support if available."""
    model = model.cuda()
    source_loader, target_loader = sample_data
    
    # Move data to GPU
    source_loader = [(features.cuda(), targets.cuda())
                    for features, targets in source_loader]
    target_loader = [(features.cuda(), targets.cuda())
                    for features, targets in target_loader]
    
    history = model.transfer_learn(
        source_data=source_loader,
        target_data=target_loader,
        num_epochs=2
    )
    
    assert all(isinstance(loss, float) for losses in history.values()
              for loss in losses)

def test_different_architectures(config, transfer_config):
    """Test different architecture configurations."""
    # Test with different layer dimensions
    transfer_config['layer_dims'] = [256, 128, 64]
    model = TransferModel(
        config=config,
        source_task="price_prediction",
        target_task="volatility_prediction",
        transfer_config=transfer_config
    )
    
    assert len(model.shared_layers) == len(transfer_config['layer_dims']) - 1
    
    # Test forward pass
    features = torch.randn(16, 10)
    output = model(features)
    assert output.shape == (16, 1)

def test_batch_normalization(model):
    """Test batch normalization behavior."""
    features = torch.randn(16, 10)
    
    # Training mode
    model.train()
    train_output = model(features)
    
    # Evaluation mode
    model.eval()
    eval_output = model(features)
    
    # Outputs should be different in train and eval modes
    assert not torch.equal(train_output, eval_output)
