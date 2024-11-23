#!/usr/bin/env python3
"""
Distributed Training Script for Bitcoin Trading RL.
Handles multi-GPU and multi-node training setup.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel

from src.models.base_model import BaseModel
from src.training.distributed_trainer import DistributedTrainer
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to use for training"
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Rank of current node"
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="localhost",
        help="Master node address"
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="12355",
        help="Master node port"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend"
    )
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(args, config: dict) -> None:
    """Set up distributed training environment."""
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    
    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    # Set PyTorch settings
    if config["training"]["gpu"]["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = config["training"]["gpu"]["cudnn_benchmark"]

def train(local_rank: int, args, config: dict) -> None:
    """
    Training function for each process.
    
    Args:
        local_rank: Local rank of current process
        args: Command line arguments
        config: Configuration dictionary
    """
    # Calculate world size and rank
    world_size = args.nodes * torch.cuda.device_count()
    rank = args.node_rank * torch.cuda.device_count() + local_rank
    
    try:
        # Initialize process group
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        logger.info(f"Initialized process {rank}/{world_size-1}")
        
        # Create model
        model = BaseModel()
        
        # Initialize distributed trainer
        trainer = DistributedTrainer(
            model=model,
            config=config["training"],
            world_size=world_size,
            rank=rank,
            backend=args.backend
        )
        
        # Load datasets
        train_dataset = None  # TODO: Load your training dataset
        val_dataset = None    # TODO: Load your validation dataset
        
        # Start training
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=config["training"]["num_epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"]
        )
        
        # Save results (only on master process)
        if rank == 0:
            save_results(history, config)
        
    except Exception as e:
        logger.error(f"Error in process {rank}: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

def save_results(history: dict, config: dict) -> None:
    """
    Save training results.
    
    Args:
        history: Training history
        config: Configuration dictionary
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training history
    torch.save(
        history,
        results_dir / f"training_history_{datetime.now():%Y%m%d_%H%M%S}.pt"
    )
    
    logger.info("Training results saved successfully")

def main():
    """Main function to start distributed training."""
    args = parse_args()
    config = load_config(args.config)
    
    try:
        # Verify CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for distributed training.")
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPUs")
        
        # Set up environment
        setup_environment(args, config)
        
        # Start processes
        mp.spawn(
            train,
            args=(args, config),
            nprocs=num_gpus,
            join=True
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
