"""
Distributed Training Module for Bitcoin Trading RL.
Implements distributed training across multiple GPUs and machines.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from src.models.base_model import BaseModel
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class DistributedTrainer:
    """
    Handles distributed training across multiple GPUs and machines.
    Supports both DataParallel and DistributedDataParallel training.
    """
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict,
        world_size: int = None,
        rank: int = None,
        backend: str = "nccl"
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            world_size: Number of processes for distributed training
            rank: Rank of current process
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        """
        self.model = model
        self.config = config
        self.world_size = world_size or torch.cuda.device_count()
        self.rank = rank
        self.backend = backend
        
        # Set up distributed environment
        self.setup_distributed()
        
        # Initialize distributed model
        self.model = self.setup_model()
        
        # Set up mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get("use_amp", True))
        
        logger.info(f"Initialized distributed trainer with {self.world_size} processes")
    
    def setup_distributed(self) -> None:
        """Set up distributed training environment."""
        if self.world_size > 1:
            if self.rank is None:
                # Single machine, multiple GPUs
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
            
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank or 0
            )
            
            logger.info(f"Initialized process group: rank {self.rank}, world_size {self.world_size}")
    
    def setup_model(self) -> torch.nn.Module:
        """Set up distributed model."""
        if torch.cuda.is_available():
            # Determine GPU device
            if self.world_size > 1:
                gpu_id = self.rank % torch.cuda.device_count()
            else:
                gpu_id = 0
            
            # Move model to GPU
            self.model = self.model.cuda(gpu_id)
            
            if self.world_size > 1:
                # Wrap model with DistributedDataParallel
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[gpu_id],
                    output_device=gpu_id,
                    find_unused_parameters=True
                )
                logger.info(f"Model wrapped with DistributedDataParallel on GPU {gpu_id}")
        
        return self.model
    
    def setup_data_loader(self, dataset, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Set up distributed data loader.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size per GPU
            
        Returns:
            DataLoader configured for distributed training
        """
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        else:
            sampler = None
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True
        )
    
    def train_step(
        self,
        batch: Tuple,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            batch: Batch of training data
            optimizer: Optimizer for parameter updates
            
        Returns:
            Dictionary of metrics
        """
        # Move batch to GPU if available
        if torch.cuda.is_available():
            batch = tuple(t.cuda() if torch.is_tensor(t) else t for t in batch)
        
        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.get("use_amp", True)):
            loss, metrics = self.model(*batch)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.config.get("grad_clip", 0) > 0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_clip"]
            )
        
        # Update parameters
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad()
        
        # Gather metrics from all processes
        if self.world_size > 1:
            metrics = self.gather_metrics(metrics)
        
        return metrics
    
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Gather metrics from all processes.
        
        Args:
            metrics: Dictionary of metrics from current process
            
        Returns:
            Aggregated metrics across all processes
        """
        if not self.world_size > 1:
            return metrics
        
        gathered_metrics = {}
        
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.clone().detach()
                gathered_values = [torch.zeros_like(value) for _ in range(self.world_size)]
                dist.all_gather(gathered_values, value)
                gathered_metrics[key] = torch.mean(torch.stack(gathered_values))
            else:
                gathered_metrics[key] = value
        
        return gathered_metrics
    
    def train(
        self,
        train_dataset,
        val_dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train model in distributed setting.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size per GPU
            learning_rate: Learning rate
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary of training history
        """
        # Set up data loaders
        train_loader = self.setup_data_loader(train_dataset, batch_size)
        val_loader = self.setup_data_loader(val_dataset, batch_size)
        
        # Scale learning rate by world size
        learning_rate *= self.world_size
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            **kwargs.get("optimizer_kwargs", {})
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, optimizer)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_metrics"].append(train_metrics)
            history["val_metrics"].append(val_metrics)
            
            # Log progress
            if self.rank == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"train_loss: {train_metrics['loss']:.4f}, "
                    f"val_loss: {val_metrics['loss']:.4f}"
                )
        
        return history
    
    def train_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train single epoch.
        
        Args:
            data_loader: Training data loader
            optimizer: Optimizer for parameter updates
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_metrics = {}
        
        for batch in data_loader:
            metrics = self.train_step(batch, optimizer)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # Average metrics
        return {k: sum(v)/len(v) for k, v in epoch_metrics.items()}
    
    def validate_epoch(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate single epoch.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        epoch_metrics = {}
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to GPU if available
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() if torch.is_tensor(t) else t for t in batch)
                
                # Forward pass
                _, metrics = self.model(*batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
        
        # Average metrics
        metrics = {k: sum(v)/len(v) for k, v in epoch_metrics.items()}
        
        # Gather metrics from all processes
        if self.world_size > 1:
            metrics = self.gather_metrics(metrics)
        
        return metrics
    
    def cleanup(self):
        """Clean up distributed training resources."""
        if self.world_size > 1:
            dist.destroy_process_group()
            logger.info("Cleaned up distributed training resources")
