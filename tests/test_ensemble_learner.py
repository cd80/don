"""
Tests for ensemble learning functionality.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel
from src.models.ensemble_learner import (
    BaggingEnsemble,
    BoostingEnsemble,
    StackingEnsemble,
    VotingEnsemble,
    create_ensemble
)

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "model": {
            "input_dim": 10,
            "output_dim": 1,
            "hidden_dims": [32, 16]
        }
    }

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    batch_size = 32
    features = torch.randn(batch_size, 10)
    targets = torch.randn(batch_size, 1)
    return features, targets

@pytest.fixture
def base_models(config):
    """Create list of base models for testing."""
    return [BaseModel(config) for _ in range(3)]

def test_bagging_ensemble_initialization(config):
    """Test bagging ensemble initialization."""
    ensemble = BaggingEnsemble(
        base_model_class=BaseModel,
        config=config,
        num_models=5
    )
    
    assert len(ensemble.models) == 5
    assert all(isinstance(model, BaseModel) for model in ensemble.models)

def test_bagging_ensemble_prediction(config, sample_data):
    """Test bagging ensemble predictions."""
    ensemble = BaggingEnsemble(
        base_model_class=BaseModel,
        config=config,
        num_models=3
    )
    
    features, _ = sample_data
    predictions = ensemble.predict(features)
    
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(features), 1)

def test_bagging_ensemble_update(config, sample_data):
    """Test bagging ensemble update."""
    ensemble = BaggingEnsemble(
        base_model_class=BaseModel,
        config=config,
        num_models=3
    )
    
    features, targets = sample_data
    metrics = ensemble.update(features, targets)
    
    assert isinstance(metrics, dict)
    assert 'loss' in metrics

def test_boosting_ensemble_initialization(config):
    """Test boosting ensemble initialization."""
    ensemble = BoostingEnsemble(
        base_model_class=BaseModel,
        config=config,
        num_models=5
    )
    
    assert len(ensemble.models) == 5
    assert len(ensemble.weights) == 5
    assert torch.allclose(ensemble.weights.sum(), torch.tensor(1.0))

def test_boosting_ensemble_prediction(config, sample_data):
    """Test boosting ensemble predictions."""
    ensemble = BoostingEnsemble(
        base_model_class=BaseModel,
        config=config,
        num_models=3
    )
    
    features, _ = sample_data
    predictions = ensemble.predict(features)
    
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(features), 1)

def test_boosting_ensemble_update(config, sample_data):
    """Test boosting ensemble update."""
    ensemble = BoostingEnsemble(
        base_model_class=BaseModel,
        config=config,
        num_models=3
    )
    
    features, targets = sample_data
    metrics = ensemble.update(features, targets)
    
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert torch.allclose(ensemble.weights.sum(), torch.tensor(1.0))

def test_stacking_ensemble_initialization(config):
    """Test stacking ensemble initialization."""
    ensemble = StackingEnsemble(
        base_model_class=BaseModel,
        meta_model_class=BaseModel,
        config=config,
        num_models=5
    )
    
    assert len(ensemble.base_models) == 5
    assert isinstance(ensemble.meta_model, BaseModel)

def test_stacking_ensemble_prediction(config, sample_data):
    """Test stacking ensemble predictions."""
    ensemble = StackingEnsemble(
        base_model_class=BaseModel,
        meta_model_class=BaseModel,
        config=config,
        num_models=3
    )
    
    features, _ = sample_data
    predictions = ensemble.predict(features)
    
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(features), 1)

def test_stacking_ensemble_update(config, sample_data):
    """Test stacking ensemble update."""
    ensemble = StackingEnsemble(
        base_model_class=BaseModel,
        meta_model_class=BaseModel,
        config=config,
        num_models=3
    )
    
    features, targets = sample_data
    metrics = ensemble.update(features, targets)
    
    assert isinstance(metrics, dict)
    assert 'base_loss' in metrics
    assert 'meta_loss' in metrics

def test_voting_ensemble_initialization(base_models):
    """Test voting ensemble initialization."""
    ensemble = VotingEnsemble(
        models=base_models,
        voting='soft'
    )
    
    assert len(ensemble.models) == len(base_models)
    assert torch.allclose(ensemble.weights.sum(), torch.tensor(1.0))

def test_voting_ensemble_prediction(base_models, sample_data):
    """Test voting ensemble predictions."""
    ensemble = VotingEnsemble(
        models=base_models,
        voting='soft'
    )
    
    features, _ = sample_data
    predictions = ensemble.predict(features)
    
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(features), 1)

def test_voting_ensemble_update(base_models, sample_data):
    """Test voting ensemble update."""
    ensemble = VotingEnsemble(
        models=base_models,
        voting='soft'
    )
    
    features, targets = sample_data
    metrics = ensemble.update(features, targets)
    
    assert isinstance(metrics, dict)
    assert 'ensemble_loss' in metrics
    assert 'model_losses' in metrics
    assert torch.allclose(ensemble.weights.sum(), torch.tensor(1.0))

def test_ensemble_factory(config):
    """Test ensemble factory function."""
    ensemble_types = ['bagging', 'boosting', 'stacking', 'voting']
    
    for ensemble_type in ensemble_types:
        if ensemble_type == 'voting':
            ensemble = create_ensemble(
                ensemble_type,
                base_model_class=BaseModel,
                config=config,
                models=[BaseModel(config) for _ in range(3)]
            )
        else:
            ensemble = create_ensemble(
                ensemble_type,
                base_model_class=BaseModel,
                config=config
            )
        
        assert isinstance(ensemble, (
            BaggingEnsemble,
            BoostingEnsemble,
            StackingEnsemble,
            VotingEnsemble
        ))

def test_invalid_ensemble_type(config):
    """Test error handling for invalid ensemble type."""
    with pytest.raises(ValueError):
        create_ensemble(
            'invalid_type',
            base_model_class=BaseModel,
            config=config
        )

@pytest.mark.parametrize("voting_type", ['hard', 'soft'])
def test_voting_types(base_models, sample_data, voting_type):
    """Test different voting types."""
    ensemble = VotingEnsemble(
        models=base_models,
        voting=voting_type
    )
    
    features, _ = sample_data
    predictions = ensemble.predict(features)
    
    assert isinstance(predictions, torch.Tensor)
    if voting_type == 'hard':
        assert torch.all((predictions == 0) | (predictions == 1))

def test_bootstrap_sampling(config, sample_data):
    """Test bootstrap sampling in bagging ensemble."""
    ensemble = BaggingEnsemble(
        base_model_class=BaseModel,
        config=config,
        num_models=3,
        bootstrap_ratio=0.8
    )
    
    features, targets = sample_data
    sample_x, sample_y = ensemble.bootstrap_sample(features, targets)
    
    assert len(sample_x) == int(0.8 * len(features))
    assert len(sample_y) == int(0.8 * len(features))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_support(config, sample_data):
    """Test GPU support for ensembles."""
    ensemble_types = ['bagging', 'boosting', 'stacking']
    features, targets = sample_data
    features = features.cuda()
    targets = targets.cuda()
    
    for ensemble_type in ensemble_types:
        ensemble = create_ensemble(
            ensemble_type,
            base_model_class=BaseModel,
            config=config
        )
        
        # Move ensemble to GPU
        if ensemble_type == 'stacking':
            ensemble.meta_model = ensemble.meta_model.cuda()
            for model in ensemble.base_models:
                model = model.cuda()
        else:
            for model in ensemble.models:
                model = model.cuda()
        
        predictions = ensemble.predict(features)
        assert predictions.is_cuda

def test_weighted_voting(base_models, sample_data):
    """Test weighted voting ensemble."""
    weights = [0.5, 0.3, 0.2]
    ensemble = VotingEnsemble(
        models=base_models,
        voting='soft',
        weights=weights
    )
    
    features, _ = sample_data
    predictions = ensemble.predict(features)
    
    assert isinstance(predictions, torch.Tensor)
    assert torch.allclose(ensemble.weights, torch.tensor(weights))
