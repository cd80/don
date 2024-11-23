# Distributed Training Guide

This guide explains how to use the distributed training capabilities of the Bitcoin Trading RL project across multiple GPUs and machines.

## Overview

The distributed training system supports:

- Multi-GPU training on a single machine
- Distributed training across multiple machines
- Mixed precision training
- Gradient accumulation
- Automatic batch size scaling
- Fault tolerance

## Requirements

- CUDA-capable GPUs (for GPU training)
- NCCL or Gloo backend installed
- Network connectivity between nodes (for multi-node training)
- Sufficient GPU memory

## Quick Start

### Single Machine, Multiple GPUs

```bash
# Start training on all available GPUs
python scripts/train_distributed.py --config configs/config.yaml

# Specify number of GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_distributed.py --config configs/config.yaml
```

### Multiple Machines

On the master node:

```bash
python scripts/train_distributed.py \
    --config configs/config.yaml \
    --nodes 2 \
    --node-rank 0 \
    --master-addr "master.example.com" \
    --master-port "12355"
```

On worker nodes:

```bash
python scripts/train_distributed.py \
    --config configs/config.yaml \
    --nodes 2 \
    --node-rank 1 \
    --master-addr "master.example.com" \
    --master-port "12355"
```

## Configuration

### Distributed Training Settings

Configure distributed training in `configs/config.yaml`:

```yaml
training:
  distributed:
    enabled: true
    backend: "nccl" # nccl for GPU, gloo for CPU
    world_size: -1 # -1 for all available GPUs
    num_nodes: 1
    node_rank: 0
    master_addr: "localhost"
    master_port: 12355
    init_method: "env://"
    sync_bn: true
    find_unused_parameters: true

  gpu:
    enabled: true
    devices: "all" # "all" or list of device ids
    data_parallel: true
    mixed_precision: true
    optimization_level: "O2"
    gradient_sync_period: 1
    gradient_accumulation_steps: 1
```

### Performance Optimization

```yaml
training:
  performance:
    num_workers: -1 # -1 for all CPU cores
    pin_memory: true
    prefetch_factor: 2
    persistent_workers: true
    use_amp: true
    grad_clip: 1.0
    benchmark_mode: true
```

## Memory Management

### GPU Memory Optimization

1. Gradient Accumulation:

```yaml
training:
  gpu:
    gradient_accumulation_steps: 4 # Accumulate gradients over 4 steps
```

2. Mixed Precision Training:

```yaml
training:
  gpu:
    mixed_precision: true
    optimization_level: "O2" # O1 for conservative, O2 for aggressive
```

3. Memory Efficient Features:

```yaml
training:
  performance:
    optimize_memory: true
    cuda_graphs: true
```

## Monitoring

### Training Progress

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir results/logs
```

### GPU Utilization

Monitor GPU usage:

```bash
nvidia-smi -l 1  # Update every second
```

### Process Status

Check distributed processes:

```bash
ps aux | grep train_distributed.py
```

## Fault Tolerance

### Checkpointing

Configure checkpointing in `config.yaml`:

```yaml
training:
  checkpointing:
    save_best: true
    save_frequency: 10
    path: "checkpoints"
    max_to_keep: 5
```

### Automatic Recovery

The system automatically handles:

- Process failures
- GPU errors
- Network interruptions

## Performance Tuning

### Batch Size Scaling

The effective batch size is:

```
effective_batch_size = batch_size * num_gpus * gradient_accumulation_steps
```

Scale learning rate accordingly:

```
learning_rate = base_learning_rate * sqrt(effective_batch_size / base_batch_size)
```

### Gradient Synchronization

1. Synchronous Training:

```yaml
training:
  gpu:
    gradient_sync_period: 1 # Sync after every batch
```

2. Asynchronous Training:

```yaml
training:
  gpu:
    gradient_sync_period: 4 # Sync every 4 batches
```

## Best Practices

1. **Data Loading**

   - Use sufficient worker processes
   - Enable pin_memory for faster GPU transfer
   - Use persistent workers for better efficiency

2. **GPU Utilization**

   - Monitor GPU memory usage
   - Adjust batch size based on available memory
   - Use gradient accumulation for large models

3. **Network Communication**

   - Use NCCL backend for GPU-GPU communication
   - Ensure high-speed network connectivity
   - Monitor network bandwidth utilization

4. **Fault Tolerance**
   - Save checkpoints regularly
   - Implement error handling
   - Monitor system resources

## Troubleshooting

### Common Issues

1. **Out of Memory**

   ```bash
   # Solution: Reduce batch size or use gradient accumulation
   python scripts/train_distributed.py --config configs/config.yaml \
       --batch-size 32 --gradient-accumulation-steps 4
   ```

2. **Process Communication Errors**

   ```bash
   # Check network connectivity
   ping master.example.com

   # Verify NCCL installation
   python -c "import torch; print(torch.cuda.nccl.version())"
   ```

3. **GPU Synchronization Issues**
   ```bash
   # Enable deterministic training
   training:
     gpu:
       deterministic: true
       cudnn_benchmark: false
   ```

### Debug Mode

Enable debug logging:

```bash
python scripts/train_distributed.py --config configs/config.yaml --log-level DEBUG
```

## Advanced Topics

### Custom Data Parallelism

Implement custom data parallel logic:

```python
from torch.nn.parallel import DistributedDataParallel

class CustomDataParallel(DistributedDataParallel):
    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)
        # Custom initialization
```

### Pipeline Parallelism

For very large models:

```python
from torch.distributed.pipeline.sync import Pipe

model = Pipe(model, chunks=8)  # Split model into 8 chunks
```

## Examples

### Basic Multi-GPU Training

```bash
# Train on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_distributed.py \
    --config configs/config.yaml \
    --batch-size 64 \
    --nodes 1
```

### Multi-Node Training

```bash
# Node 1 (Master)
python scripts/train_distributed.py \
    --config configs/config.yaml \
    --nodes 2 \
    --node-rank 0 \
    --master-addr "192.168.1.100" \
    --master-port "12355"

# Node 2 (Worker)
python scripts/train_distributed.py \
    --config configs/config.yaml \
    --nodes 2 \
    --node-rank 1 \
    --master-addr "192.168.1.100" \
    --master-port "12355"
```

## Next Steps

1. Experiment with different batch sizes and learning rates
2. Monitor training performance and GPU utilization
3. Implement custom data loading optimizations
4. Fine-tune distributed training parameters

For more details, refer to the [API Documentation](../api/distributed.md).
