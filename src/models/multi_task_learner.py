"""
Multi-Task Learning Module for Bitcoin Trading RL.
Implements multi-task learning for simultaneous optimization of multiple trading objectives.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.models.base_model import BaseModel
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class TaskHead(nn.Module):
    """Task-specific head for multi-task learning."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 32]
    ):
        """
        Initialize task head.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through task head."""
        return self.network(x)

class MultiTaskModel(BaseModel):
    """
    Multi-task learning model for trading.
    Handles multiple trading-related tasks simultaneously.
    """
    
    def __init__(
        self,
        config: Dict,
        task_configs: Dict[str, Dict],
        shared_dim: int = 128,
        uncertainty_weighting: bool = True
    ):
        """
        Initialize multi-task model.
        
        Args:
            config: Model configuration
            task_configs: Configuration for each task
            shared_dim: Dimension of shared representation
            uncertainty_weighting: Use uncertainty weighting for loss balancing
        """
        super().__init__(config)
        
        self.task_configs = task_configs
        self.shared_dim = shared_dim
        self.uncertainty_weighting = uncertainty_weighting
        
        # Create shared layers
        self.shared_network = self._create_shared_network()
        
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
        self.log_vars = nn.ParameterDict()  # For uncertainty weighting
        
        for task_name, task_config in task_configs.items():
            self.task_heads[task_name] = TaskHead(
                input_dim=shared_dim,
                output_dim=task_config['output_dim'],
                hidden_dims=task_config.get('hidden_dims', [64, 32])
            )
            
            if uncertainty_weighting:
                self.log_vars[task_name] = nn.Parameter(torch.zeros(1))
        
        logger.info(
            f"Initialized multi-task model with {len(task_configs)} tasks: "
            f"{list(task_configs.keys())}"
        )
    
    def _create_shared_network(self) -> nn.Module:
        """Create shared network layers."""
        return nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, self.shared_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.shared_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model.
        
        Args:
            x: Input features
            tasks: Optional list of tasks to compute
            
        Returns:
            Dictionary of task outputs
        """
        # Get shared representation
        shared_features = self.shared_network(x)
        
        # Compute task-specific outputs
        tasks = tasks or list(self.task_heads.keys())
        outputs = {}
        
        for task_name in tasks:
            outputs[task_name] = self.task_heads[task_name](shared_features)
        
        return outputs
    
    def compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute task-specific losses and total loss.
        
        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
            
        Returns:
            Total loss and dictionary of task-specific losses
        """
        task_losses = {}
        total_loss = 0.0
        
        for task_name in predictions.keys():
            task_loss = F.mse_loss(predictions[task_name], targets[task_name])
            
            if self.uncertainty_weighting:
                # Apply uncertainty weighting
                precision = torch.exp(-self.log_vars[task_name])
                task_losses[task_name] = precision * task_loss + self.log_vars[task_name]
            else:
                # Use task-specific weights from config
                weight = self.task_configs[task_name].get('weight', 1.0)
                task_losses[task_name] = weight * task_loss
            
            total_loss += task_losses[task_name]
        
        return total_loss, {k: v.item() for k, v in task_losses.items()}
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            batch: Tuple of (features, task_targets)
            optimizer: Optimizer for parameter updates
            
        Returns:
            Dictionary of metrics
        """
        features, task_targets = batch
        
        # Forward pass
        predictions = self(features)
        
        # Compute losses
        total_loss, task_losses = self.compute_losses(predictions, task_targets)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Compute metrics
        metrics = {
            'total_loss': total_loss.item(),
            **task_losses
        }
        
        return metrics
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for features, task_targets in data_loader:
                predictions = self(features)
                _, task_losses = self.compute_losses(predictions, task_targets)
                
                # Accumulate metrics
                for task_name, loss in task_losses.items():
                    if task_name not in total_metrics:
                        total_metrics[task_name] = 0.0
                    total_metrics[task_name] += loss
                
                num_batches += 1
        
        # Average metrics
        return {k: v/num_batches for k, v in total_metrics.items()}
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights based on uncertainty."""
        if not self.uncertainty_weighting:
            return {
                task_name: config.get('weight', 1.0)
                for task_name, config in self.task_configs.items()
            }
        
        weights = {}
        for task_name in self.task_heads.keys():
            weights[task_name] = torch.exp(-self.log_vars[task_name]).item()
        
        return weights
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'task_configs': self.task_configs,
            'shared_dim': self.shared_dim,
            'uncertainty_weighting': self.uncertainty_weighting
        }, path)
        logger.info(f"Saved model checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.task_configs = checkpoint['task_configs']
        self.shared_dim = checkpoint['shared_dim']
        self.uncertainty_weighting = checkpoint['uncertainty_weighting']
        logger.info(f"Loaded model checkpoint from {path}")
    
    @property
    def input_dim(self) -> int:
        """Get input dimension from base model."""
        return self.config['model']['architecture']['input_dim']
