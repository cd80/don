"""
Tests for distributed training functionality.
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from unittest.mock import MagicMock, patch

from src.training.distributed_trainer import DistributedTrainer
from src.models.base_model import BaseModel

class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing."""
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "use_amp": True,
        "grad_clip": 1.0,
        "num_workers": 0,  # Use 0 for testing
        "batch_size": 16,
        "learning_rate": 0.001,
        "num_epochs": 2
    }

@pytest.fixture
def model():
    """Test model."""
    model = BaseModel()
    # Mock forward method to return loss and metrics
    model.forward = MagicMock(return_value=(
        torch.tensor(1.0, requires_grad=True),
        {"loss": torch.tensor(1.0), "accuracy": torch.tensor(0.8)}
    ))
    return model

@pytest.fixture
def dataset():
    """Test dataset."""
    return DummyDataset()

def test_single_gpu_initialization(model, config):
    """Test initialization with single GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    trainer = DistributedTrainer(model, config, world_size=1)
    assert isinstance(trainer.model, torch.nn.Module)
    assert trainer.world_size == 1
    assert trainer.rank == 0

def test_cpu_initialization(model, config):
    """Test initialization on CPU."""
    with patch('torch.cuda.is_available', return_value=False):
        trainer = DistributedTrainer(model, config, world_size=1)
        assert isinstance(trainer.model, torch.nn.Module)
        assert not next(trainer.model.parameters()).is_cuda

def test_data_loader_setup(model, config, dataset):
    """Test data loader configuration."""
    trainer = DistributedTrainer(model, config, world_size=1)
    loader = trainer.setup_data_loader(dataset, batch_size=16)
    
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert loader.batch_size == 16
    assert loader.num_workers == config["num_workers"]

def test_train_step(model, config, dataset):
    """Test single training step."""
    trainer = DistributedTrainer(model, config, world_size=1)
    loader = trainer.setup_data_loader(dataset, batch_size=16)
    optimizer = torch.optim.Adam(trainer.model.parameters())
    
    batch = next(iter(loader))
    metrics = trainer.train_step(batch, optimizer)
    
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_gpu_initialization(model, config):
    """Test initialization with multiple GPUs."""
    world_size = min(2, torch.cuda.device_count())
    if world_size < 2:
        pytest.skip("Less than 2 GPUs available")
    
    trainer = DistributedTrainer(model, config, world_size=world_size)
    assert trainer.world_size == world_size
    assert isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)

def run_distributed_test(rank, world_size, model, config, dataset):
    """Helper function to run distributed tests."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    trainer = DistributedTrainer(model, config, world_size=world_size, rank=rank)
    loader = trainer.setup_data_loader(dataset, batch_size=16)
    optimizer = torch.optim.Adam(trainer.model.parameters())
    
    # Train for one epoch
    metrics = trainer.train_epoch(loader, optimizer)
    
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    
    trainer.cleanup()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_distributed_training(model, config, dataset):
    """Test distributed training across processes."""
    world_size = min(2, torch.cuda.device_count())
    if world_size < 2:
        pytest.skip("Less than 2 GPUs available")
    
    mp.spawn(
        run_distributed_test,
        args=(world_size, model, config, dataset),
        nprocs=world_size,
        join=True
    )

def test_gather_metrics(model, config):
    """Test metric gathering across processes."""
    trainer = DistributedTrainer(model, config, world_size=1)
    metrics = {
        "loss": torch.tensor(1.0),
        "accuracy": torch.tensor(0.8),
        "scalar_metric": 0.5
    }
    
    gathered = trainer.gather_metrics(metrics)
    assert isinstance(gathered, dict)
    assert all(key in gathered for key in metrics.keys())

def test_cleanup(model, config):
    """Test cleanup of distributed resources."""
    trainer = DistributedTrainer(model, config, world_size=1)
    trainer.cleanup()  # Should not raise any errors

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_training(model, config, dataset):
    """Test mixed precision training."""
    trainer = DistributedTrainer(model, config, world_size=1)
    loader = trainer.setup_data_loader(dataset, batch_size=16)
    optimizer = torch.optim.Adam(trainer.model.parameters())
    
    with patch('torch.cuda.amp.autocast') as mock_autocast:
        batch = next(iter(loader))
        trainer.train_step(batch, optimizer)
        assert mock_autocast.called

def test_gradient_clipping(model, config, dataset):
    """Test gradient clipping."""
    config["grad_clip"] = 1.0
    trainer = DistributedTrainer(model, config, world_size=1)
    loader = trainer.setup_data_loader(dataset, batch_size=16)
    optimizer = torch.optim.Adam(trainer.model.parameters())
    
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip_grad:
        batch = next(iter(loader))
        trainer.train_step(batch, optimizer)
        assert mock_clip_grad.called

def test_full_training_loop(model, config, dataset):
    """Test complete training loop."""
    trainer = DistributedTrainer(model, config, world_size=1)
    history = trainer.train(
        dataset,
        dataset,  # Use same dataset for validation
        num_epochs=2,
        batch_size=16,
        learning_rate=0.001
    )
    
    assert isinstance(history, dict)
    assert all(key in history for key in [
        "train_loss", "val_loss", "train_metrics", "val_metrics"
    ])
    assert len(history["train_loss"]) == 2  # 2 epochs
