#!/usr/bin/env python3
"""
Performance Optimization Script for Bitcoin Trading RL.
This script helps identify and optimize performance bottlenecks.
"""

import os
import sys
import time
import psutil
import logging
import cProfile
import pstats
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import concurrent.futures
from memory_profiler import profile
import torch
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization and monitoring tools."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'latency': [],
            'throughput': []
        }
        self.baseline = None
    
    def measure_system_metrics(self) -> Dict[str, float]:
        """Measure current system metrics."""
        metrics = {}
        
        # CPU usage
        metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used'] = memory.used / (1024 * 1024 * 1024)  # GB
        
        # GPU usage if available
        if torch.cuda.is_available():
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
            metrics['gpu_utilization'] = torch.cuda.utilization()
        
        # Disk I/O
        disk = psutil.disk_io_counters()
        metrics['disk_read'] = disk.read_bytes / (1024 * 1024)  # MB
        metrics['disk_write'] = disk.write_bytes / (1024 * 1024)  # MB
        
        # Network I/O
        net = psutil.net_io_counters()
        metrics['net_sent'] = net.bytes_sent / (1024 * 1024)  # MB
        metrics['net_recv'] = net.bytes_recv / (1024 * 1024)  # MB
        
        return metrics
    
    @profile
    def profile_memory_usage(self, func, *args, **kwargs):
        """Profile memory usage of a function."""
        return func(*args, **kwargs)
    
    def profile_execution_time(self, func, *args, **kwargs) -> Dict[str, float]:
        """Profile execution time of a function."""
        profiler = cProfile.Profile()
        result = profiler.runctx('func(*args, **kwargs)', globals(), locals())
        stats = pstats.Stats(profiler)
        
        metrics = {
            'total_time': stats.total_tt,
            'calls': len(stats.stats),
            'primitive_calls': stats.prim_calls
        }
        
        return metrics
    
    async def measure_network_latency(self, url: str, num_requests: int = 100) -> Dict[str, float]:
        """Measure network latency."""
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for _ in range(num_requests):
                start_time = time.time()
                async with session.get(url) as response:
                    await response.text()
                latencies.append(time.time() - start_time)
        
        return {
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'avg_latency': np.mean(latencies),
            'std_latency': np.std(latencies)
        }
    
    def optimize_batch_size(
        self,
        model,
        data_loader,
        min_batch: int = 1,
        max_batch: int = 1024,
        target_time: float = 0.1
    ) -> int:
        """Find optimal batch size."""
        batch_sizes = [2**i for i in range(int(np.log2(min_batch)), int(np.log2(max_batch))+1)]
        times = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            # Process one batch
            for batch in data_loader:
                model(batch)
                break
            times.append(time.time() - start_time)
        
        # Find closest to target time
        optimal_idx = np.argmin(np.abs(np.array(times) - target_time))
        return batch_sizes[optimal_idx]
    
    def optimize_thread_count(
        self,
        func,
        data,
        min_threads: int = 1,
        max_threads: int = None
    ) -> int:
        """Find optimal thread count."""
        if max_threads is None:
            max_threads = psutil.cpu_count()
        
        thread_counts = range(min_threads, max_threads + 1)
        times = []
        
        for n_threads in thread_counts:
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                executor.map(func, data)
            times.append(time.time() - start_time)
        
        # Find point of diminishing returns
        optimal_idx = np.argmin(np.diff(times))
        return thread_counts[optimal_idx]
    
    def generate_report(self) -> pd.DataFrame:
        """Generate performance report."""
        df = pd.DataFrame(self.metrics)
        
        report = {
            'cpu_usage_avg': df['cpu_usage'].mean(),
            'memory_usage_avg': df['memory_usage'].mean(),
            'latency_avg': df['latency'].mean(),
            'throughput_avg': df['throughput'].mean(),
            'cpu_usage_max': df['cpu_usage'].max(),
            'memory_usage_max': df['memory_usage'].max(),
            'latency_max': df['latency'].max(),
            'throughput_min': df['throughput'].min()
        }
        
        if 'gpu_usage' in df.columns:
            report['gpu_usage_avg'] = df['gpu_usage'].mean()
            report['gpu_usage_max'] = df['gpu_usage'].max()
        
        return pd.DataFrame([report])
    
    def plot_metrics(self):
        """Plot performance metrics."""
        df = pd.DataFrame(self.metrics)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU Usage
        axes[0, 0].plot(df['cpu_usage'])
        axes[0, 0].set_title('CPU Usage')
        axes[0, 0].set_ylabel('Percent')
        
        # Memory Usage
        axes[0, 1].plot(df['memory_usage'])
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('GB')
        
        # Latency
        axes[1, 0].plot(df['latency'])
        axes[1, 0].set_title('Latency')
        axes[1, 0].set_ylabel('Seconds')
        
        # Throughput
        axes[1, 1].plot(df['throughput'])
        axes[1, 1].set_title('Throughput')
        axes[1, 1].set_ylabel('Operations/Second')
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.close()

def main():
    """Main function."""
    optimizer = PerformanceOptimizer()
    
    try:
        # Measure baseline metrics
        logger.info("Measuring baseline system metrics...")
        baseline = optimizer.measure_system_metrics()
        
        # Profile key components
        logger.info("Profiling key components...")
        components = [
            'data_pipeline',
            'model_training',
            'trading_strategies',
            'risk_management'
        ]
        
        for component in components:
            logger.info(f"Profiling {component}...")
            # Add specific profiling logic for each component
        
        # Generate report
        logger.info("Generating performance report...")
        report = optimizer.generate_report()
        print("\nPerformance Report:")
        print(report)
        
        # Plot metrics
        logger.info("Plotting metrics...")
        optimizer.plot_metrics()
        
        logger.info("Optimization analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
