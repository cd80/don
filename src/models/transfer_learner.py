"""
Transfer Learning Module for Bitcoin Trading RL.
Implements transfer learning capabilities to leverage pre-trained knowledge for new tasks.
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

class TransferableLayer(nn.Module):
    """
    Layer that can be frozen or fine-tuned during transfer learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        batch_norm: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize transferable layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = [nn.Linear(input_dim, output_dim)]
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        
        if activation is not None:
            layers.append(activation)
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.layer = nn.Sequential(*layers)
        self.frozen = False
    
    def freeze(self):
        """Freeze layer parameters."""
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze layer parameters."""
        self.frozen = False
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer."""
        return self.layer(x)

class TransferModel(BaseModel):
    """
    Transfer learning model for trading.
    Enables knowledge transfer between source and target tasks.
    """
    
    def __init__(
        self,
        config: Dict,
        source_task: str,
        target_task: str,
        transfer_config: Dict = None
    ):
        """
        Initialize transfer learning model.
        
        Args:
            config: Model configuration
            source_task: Name of source task
            target_task: Name of target task
            transfer_config: Transfer learning configuration
        """
        super().__init__(config)
        
        self.source_task = source_task
        self.target_task = target_task
        self.transfer_config = transfer_config or {}
        
        # Create transferable layers
        self.shared_layers = nn.ModuleList([
            TransferableLayer(
                input_dim=dim_in,
                output_dim=dim_out,
                **self.transfer_config.get('layer_config', {})
            )
            for dim_in, dim_out in zip(
                self.transfer_config.get('layer_dims', [128, 64, 32])[:-1],
                self.transfer_config.get('layer_dims', [128, 64, 32])[1:]
            )
        ])
        
        # Task-specific heads
        self.source_head = self._create_task_head(source_task)
        self.target_head = self._create_task_head(target_task)
        
        logger.info(
            f"Initialized transfer model from {source_task} to {target_task}"
        )
    
    def _create_task_head(self, task_name: str) -> nn.Module:
        """Create task-specific head."""
        return nn.Sequential(
            nn.Linear(
                self.transfer_config.get('layer_dims', [128, 64, 32])[-1],
                self.transfer_config.get('head_dim', 32)
            ),
            nn.ReLU(),
            nn.Linear(
                self.transfer_config.get('head_dim', 32),
                self.config['model']['output_dim']
            )
        )
    
    def freeze_shared_layers(self):
        """Freeze shared layers for transfer."""
        for layer in self.shared_layers:
            layer.freeze()
    
    def unfreeze_shared_layers(self):
        """Unfreeze shared layers for fine-tuning."""
        for layer in self.shared_layers:
            layer.unfreeze()
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get currently trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def forward(
        self,
        x: torch.Tensor,
        task: str = None
    ) -> torch.Tensor:
        """
        Forward pass through model.
        
        Args:
            x: Input features
            task: Task to use (source or target)
            
        Returns:
            Model output
        """
        # Pass through shared layers
        features = x
        for layer in self.shared_layers:
            features = layer(features)
        
        # Pass through task-specific head
        if task is None:
            task = self.target_task
        
        head = self.source_head if task == self.source_task else self.target_head
        return head(features)
    
    def transfer_learn(
        self,
        source_data: torch.utils.data.DataLoader,
        target_data: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        fine_tune: bool = True,
        fine_tune_epochs: int = 50,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform transfer learning.
        
        Args:
            source_data: Source task data loader
            target_data: Target task data loader
            num_epochs: Number of pre-training epochs
            learning_rate: Learning rate
            fine_tune: Whether to fine-tune shared layers
            fine_tune_epochs: Number of fine-tuning epochs
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary of training history
        """
        history = {
            'source_loss': [],
            'target_loss': [],
            'fine_tune_loss': []
        }
        
        # Pre-train on source task
        optimizer = Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in source_data:
                features, targets = batch
                
                # Forward pass
                predictions = self(features, task=self.source_task)
                loss = F.mse_loss(predictions, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history['source_loss'].append(avg_loss)
            
            logger.info(f"Source task epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        # Freeze shared layers
        self.freeze_shared_layers()
        
        # Train on target task
        optimizer = Adam(self.target_head.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in target_data:
                features, targets = batch
                
                # Forward pass
                predictions = self(features, task=self.target_task)
                loss = F.mse_loss(predictions, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            history['target_loss'].append(avg_loss)
            
            logger.info(f"Target task epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        # Fine-tuning
        if fine_tune:
            self.unfreeze_shared_layers()
            optimizer = Adam(self.get_trainable_params(), lr=learning_rate * 0.1)
            
            for epoch in range(fine_tune_epochs):
                epoch_loss = 0
                num_batches = 0
                
                for batch in target_data:
                    features, targets = batch
                    
                    # Forward pass
                    predictions = self(features, task=self.target_task)
                    loss = F.mse_loss(predictions, targets)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                history['fine_tune_loss'].append(avg_loss)
                
                logger.info(f"Fine-tuning epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        return history
    
    def save_transfer_model(self, path: str) -> None:
        """
        Save transfer learning model.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'source_task': self.source_task,
            'target_task': self.target_task,
            'transfer_config': self.transfer_config,
            'shared_layers_state': [layer.frozen for layer in self.shared_layers]
        }, path)
        logger.info(f"Saved transfer model to {path}")
    
    def load_transfer_model(self, path: str) -> None:
        """
        Load transfer learning model.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.source_task = checkpoint['source_task']
        self.target_task = checkpoint['target_task']
        self.transfer_config = checkpoint['transfer_config']
        
        # Restore layer states
        for layer, frozen in zip(self.shared_layers,
                               checkpoint['shared_layers_state']):
            if frozen:
                layer.freeze()
            else:
                layer.unfreeze()
        
        logger.info(f"Loaded transfer model from {path}")
    
    def get_layer_gradients(self) -> Dict[str, torch.Tensor]:
        """Get gradients for each layer."""
        gradients = {}
        
        for i, layer in enumerate(self.shared_layers):
            if not layer.frozen:
                for name, param in layer.named_parameters():
                    if param.grad is not None:
                        gradients[f"layer_{i}_{name}"] = param.grad.clone()
        
        return gradients
