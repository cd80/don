"""
Ensemble Learning Module for Bitcoin Trading RL.
Implements various ensemble methods to combine multiple models for improved predictions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.models.base_model import BaseModel
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class EnsembleModel(ABC):
    """Abstract base class for ensemble models."""
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using the ensemble."""
        pass
    
    @abstractmethod
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Update the ensemble."""
        pass

class BaggingEnsemble(EnsembleModel):
    """
    Bagging ensemble that combines multiple base models through averaging.
    Uses bootstrap sampling for model diversity.
    """
    
    def __init__(
        self,
        base_model_class: type,
        config: Dict,
        num_models: int = 5,
        bootstrap_ratio: float = 0.8
    ):
        """
        Initialize bagging ensemble.
        
        Args:
            base_model_class: Class of base model to use
            config: Model configuration
            num_models: Number of models in ensemble
            bootstrap_ratio: Ratio of data to sample for each model
        """
        self.models = [base_model_class(config) for _ in range(num_models)]
        self.num_models = num_models
        self.bootstrap_ratio = bootstrap_ratio
        self.config = config
        
        logger.info(
            f"Initialized bagging ensemble with {num_models} models"
        )
    
    def bootstrap_sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create bootstrap sample of data."""
        sample_size = int(len(x) * self.bootstrap_ratio)
        indices = torch.randint(len(x), (sample_size,))
        return x[indices], y[indices]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using averaged ensemble."""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        return torch.mean(torch.stack(predictions), dim=0)
    
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Update ensemble using bootstrap samples."""
        metrics = []
        
        for model in self.models:
            # Create bootstrap sample
            sample_x, sample_y = self.bootstrap_sample(x, y)
            
            # Update model
            loss, model_metrics = model(sample_x, sample_y)
            metrics.append(model_metrics)
        
        # Average metrics across models
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics])
        
        return avg_metrics

class BoostingEnsemble(EnsembleModel):
    """
    Boosting ensemble that combines models sequentially.
    Each model focuses on examples that previous models struggled with.
    """
    
    def __init__(
        self,
        base_model_class: type,
        config: Dict,
        num_models: int = 5,
        learning_rate: float = 0.1
    ):
        """
        Initialize boosting ensemble.
        
        Args:
            base_model_class: Class of base model to use
            config: Model configuration
            num_models: Number of models in ensemble
            learning_rate: Learning rate for model combination
        """
        self.models = [base_model_class(config) for _ in range(num_models)]
        self.weights = torch.ones(num_models) / num_models
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.config = config
        
        logger.info(
            f"Initialized boosting ensemble with {num_models} models"
        )
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using weighted ensemble."""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                pred = model(x)
                predictions.append(weight * pred)
        
        return torch.sum(torch.stack(predictions), dim=0)
    
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Update ensemble using boosting."""
        metrics = []
        errors = torch.zeros(len(x))
        
        for i, model in enumerate(self.models):
            # Weight samples based on errors
            sample_weights = torch.exp(errors)
            sample_weights = sample_weights / sample_weights.sum()
            
            # Update model
            loss, model_metrics = model(x, y, sample_weights)
            metrics.append(model_metrics)
            
            # Update errors
            with torch.no_grad():
                predictions = model(x)
                current_errors = F.mse_loss(predictions, y, reduction='none')
                errors += self.learning_rate * current_errors
            
            # Update model weight
            error_rate = (current_errors * sample_weights).sum()
            self.weights[i] = torch.log((1 - error_rate) / error_rate)
        
        # Normalize weights
        self.weights = F.softmax(self.weights, dim=0)
        
        # Average metrics across models
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics])
        
        return avg_metrics

class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble that uses a meta-model to combine base model predictions.
    """
    
    def __init__(
        self,
        base_model_class: type,
        meta_model_class: type,
        config: Dict,
        num_models: int = 5
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_model_class: Class of base models
            meta_model_class: Class of meta-model
            config: Model configuration
            num_models: Number of base models
        """
        self.base_models = [base_model_class(config) for _ in range(num_models)]
        self.meta_model = meta_model_class(config)
        self.num_models = num_models
        self.config = config
        
        logger.info(
            f"Initialized stacking ensemble with {num_models} base models"
        )
    
    def get_meta_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Generate meta-features from base model predictions."""
        meta_features = []
        
        for model in self.base_models:
            with torch.no_grad():
                pred = model(x)
                meta_features.append(pred)
        
        return torch.cat(meta_features, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using stacked ensemble."""
        meta_features = self.get_meta_features(x)
        return self.meta_model(meta_features)
    
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Update ensemble using stacking."""
        # Update base models
        base_metrics = []
        for model in self.base_models:
            loss, metrics = model(x, y)
            base_metrics.append(metrics)
        
        # Generate meta-features
        meta_features = self.get_meta_features(x)
        
        # Update meta-model
        meta_loss, meta_metrics = self.meta_model(meta_features, y)
        
        # Combine metrics
        metrics = {
            'base_loss': np.mean([m['loss'] for m in base_metrics]),
            'meta_loss': meta_metrics['loss']
        }
        
        return metrics

class VotingEnsemble(EnsembleModel):
    """
    Voting ensemble that combines predictions through voting or averaging.
    """
    
    def __init__(
        self,
        models: List[BaseModel],
        voting: str = 'soft',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize voting ensemble.
        
        Args:
            models: List of pre-trained models
            voting: Voting method ('hard' or 'soft')
            weights: Optional weights for models
        """
        self.models = models
        self.voting = voting
        self.weights = (torch.tensor(weights) if weights is not None
                       else torch.ones(len(models)))
        self.weights = self.weights / self.weights.sum()
        
        logger.info(
            f"Initialized voting ensemble with {len(models)} models"
        )
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions using voting ensemble."""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                pred = model(x)
                if self.voting == 'hard':
                    pred = (pred > 0.5).float()
                predictions.append(weight * pred)
        
        if self.voting == 'hard':
            return (torch.mean(torch.stack(predictions), dim=0) > 0.5).float()
        return torch.mean(torch.stack(predictions), dim=0)
    
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Update ensemble weights based on performance."""
        performances = []
        
        # Evaluate each model
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                loss = F.mse_loss(pred, y)
                performances.append(loss.item())
        
        # Update weights inversely proportional to loss
        performances = torch.tensor(performances)
        self.weights = 1 / (performances + 1e-8)
        self.weights = self.weights / self.weights.sum()
        
        # Calculate ensemble metrics
        ensemble_pred = self.predict(x)
        ensemble_loss = F.mse_loss(ensemble_pred, y)
        
        return {
            'ensemble_loss': ensemble_loss.item(),
            'model_losses': performances.tolist()
        }

def create_ensemble(
    ensemble_type: str,
    base_model_class: type,
    config: Dict,
    **kwargs
) -> EnsembleModel:
    """
    Factory function to create ensemble models.
    
    Args:
        ensemble_type: Type of ensemble to create
        base_model_class: Class of base model to use
        config: Model configuration
        **kwargs: Additional arguments for specific ensemble types
    
    Returns:
        Initialized ensemble model
    """
    ensembles = {
        'bagging': BaggingEnsemble,
        'boosting': BoostingEnsemble,
        'stacking': StackingEnsemble,
        'voting': VotingEnsemble
    }
    
    if ensemble_type not in ensembles:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    return ensembles[ensemble_type](base_model_class, config, **kwargs)
