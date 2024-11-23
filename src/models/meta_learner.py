"""
Meta-Learning Module for Bitcoin Trading RL.
Implements Model-Agnostic Meta-Learning (MAML) for quick adaptation to new market conditions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class MAMLModel(BaseModel):
    """
    Model-Agnostic Meta-Learning implementation for trading.
    Enables quick adaptation to new market conditions through meta-learning.
    """
    
    def __init__(
        self,
        config: Dict,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        task_batch_size: int = 32
    ):
        """
        Initialize MAML model.
        
        Args:
            config: Model configuration
            inner_lr: Learning rate for inner loop optimization
            meta_lr: Learning rate for meta-optimization
            num_inner_steps: Number of gradient steps in inner loop
            task_batch_size: Batch size for each task
        """
        super().__init__(config)
        
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.task_batch_size = task_batch_size
        
        # Initialize meta-optimizer
        self.meta_optimizer = Adam(self.parameters(), lr=meta_lr)
        
        logger.info(
            f"Initialized MAML model with inner_lr={inner_lr}, "
            f"meta_lr={meta_lr}, num_inner_steps={num_inner_steps}"
        )
    
    def clone_model(self) -> nn.Module:
        """Create a clone of the current model for inner loop optimization."""
        clone = MAMLModel(self.config)
        clone.load_state_dict(self.state_dict())
        return clone
    
    def inner_loop(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[Dict[str, float], Optional[nn.Module]]:
        """
        Perform inner loop optimization on a task.
        
        Args:
            support_data: Training data for the task (features, targets)
            query_data: Optional validation data for the task
            
        Returns:
            Dictionary of metrics and optionally adapted model
        """
        model_clone = self.clone_model()
        features, targets = support_data
        
        # Inner loop optimization
        for step in range(self.num_inner_steps):
            loss, metrics = model_clone(features, targets)
            grads = torch.autograd.grad(
                loss,
                model_clone.parameters(),
                create_graph=True,
                allow_unused=True
            )
            
            # Manual parameter update
            for param, grad in zip(model_clone.parameters(), grads):
                if grad is not None:
                    param.data = param.data - self.inner_lr * grad
        
        # Evaluate on query set if provided
        if query_data is not None:
            query_features, query_targets = query_data
            with torch.no_grad():
                _, query_metrics = model_clone(query_features, query_targets)
            return query_metrics, model_clone
        
        return metrics, model_clone
    
    def meta_learn(
        self,
        task_generator: DataLoader,
        num_tasks: int,
        num_epochs: int
    ) -> Dict[str, List[float]]:
        """
        Perform meta-learning across multiple tasks.
        
        Args:
            task_generator: Generator yielding task data
            num_tasks: Number of tasks to sample per epoch
            num_epochs: Number of meta-training epochs
            
        Returns:
            Dictionary of training history
        """
        history = {
            'meta_loss': [],
            'inner_losses': [],
            'adaptation_metrics': []
        }
        
        for epoch in range(num_epochs):
            epoch_meta_loss = 0.0
            epoch_inner_losses = []
            epoch_adaptation_metrics = []
            
            for _ in range(num_tasks):
                # Sample tasks
                support_data, query_data = next(iter(task_generator))
                
                # Inner loop optimization
                metrics, adapted_model = self.inner_loop(support_data, query_data)
                epoch_inner_losses.append(metrics['loss'])
                epoch_adaptation_metrics.append(metrics)
                
                # Meta-optimization
                query_features, query_targets = query_data
                meta_loss, _ = adapted_model(query_features, query_targets)
                
                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                
                epoch_meta_loss += meta_loss.item()
            
            # Record metrics
            avg_meta_loss = epoch_meta_loss / num_tasks
            history['meta_loss'].append(avg_meta_loss)
            history['inner_losses'].append(np.mean(epoch_inner_losses))
            history['adaptation_metrics'].append({
                k: np.mean([m[k] for m in epoch_adaptation_metrics])
                for k in epoch_adaptation_metrics[0].keys()
            })
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Meta Loss: {avg_meta_loss:.4f}, "
                f"Inner Loss: {history['inner_losses'][-1]:.4f}"
            )
        
        return history
    
    def adapt_to_market(
        self,
        market_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: int = None
    ) -> nn.Module:
        """
        Adapt the model to new market conditions.
        
        Args:
            market_data: Recent market data for adaptation
            num_steps: Optional override for number of adaptation steps
            
        Returns:
            Adapted model for the current market conditions
        """
        num_steps = num_steps or self.num_inner_steps
        _, adapted_model = self.inner_loop(
            market_data,
            num_inner_steps=num_steps
        )
        return adapted_model
    
    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with meta-learning support.
        
        Args:
            features: Input features
            targets: Optional target values
            
        Returns:
            Loss and metrics dictionary
        """
        # Get base model predictions
        predictions = super().forward(features)
        
        if targets is not None:
            # Calculate loss and metrics
            loss = F.mse_loss(predictions, targets)
            metrics = {
                'loss': loss.item(),
                'accuracy': self.calculate_accuracy(predictions, targets)
            }
            return loss, metrics
        
        return predictions, {}
    
    def calculate_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Calculate prediction accuracy."""
        with torch.no_grad():
            correct = torch.abs(predictions - targets) < 0.1
            return correct.float().mean().item()
    
    def save_meta_learned(self, path: str) -> None:
        """
        Save meta-learned model state.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': {
                'inner_lr': self.inner_lr,
                'meta_lr': self.meta_lr,
                'num_inner_steps': self.num_inner_steps
            }
        }, path)
        logger.info(f"Saved meta-learned model to {path}")
    
    def load_meta_learned(self, path: str) -> None:
        """
        Load meta-learned model state.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(
            checkpoint['meta_optimizer_state_dict']
        )
        
        # Update configuration
        config = checkpoint['config']
        self.inner_lr = config['inner_lr']
        self.meta_lr = config['meta_lr']
        self.num_inner_steps = config['num_inner_steps']
        
        logger.info(f"Loaded meta-learned model from {path}")
