#!/usr/bin/env python3
"""
Latency Optimization Script for Bitcoin Trading RL.
This script helps identify and reduce latency in trading operations.
"""

import os
import sys
import time
import asyncio
import logging
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import socket
import json
import tracemalloc
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LatencyMetrics:
    """Container for latency measurements."""
    network_latency: float
    processing_latency: float
    total_latency: float
    timestamp: datetime

class LatencyOptimizer:
    """Tools for measuring and optimizing trading system latency."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.metrics_history: List[LatencyMetrics] = []
        self.baseline_metrics: Optional[LatencyMetrics] = None
        tracemalloc.start()
    
    async def measure_exchange_latency(
        self,
        exchange_url: str,
        num_requests: int = 100
    ) -> Dict[str, float]:
        """Measure latency to exchange API."""
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for _ in range(num_requests):
                start_time = time.time()
                try:
                    async with session.get(exchange_url) as response:
                        await response.text()
                        latencies.append(time.time() - start_time)
                except Exception as e:
                    logger.error(f"Error measuring exchange latency: {str(e)}")
        
        return {
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'avg_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
    
    def measure_processing_latency(
        self,
        func,
        *args,
        num_runs: int = 100,
        **kwargs
    ) -> Dict[str, float]:
        """Measure processing latency of a function."""
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            try:
                func(*args, **kwargs)
                latencies.append(time.time() - start_time)
            except Exception as e:
                logger.error(f"Error measuring processing latency: {str(e)}")
        
        return {
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'avg_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
    
    def optimize_network_settings(self) -> Dict[str, any]:
        """Optimize network settings for low latency."""
        settings = {}
        
        # TCP settings
        settings['tcp_nodelay'] = True
        settings['tcp_quickack'] = True
        
        # Socket settings
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        
        # Connection pooling settings
        settings['max_connections'] = 100
        settings['keep_alive_timeout'] = 5
        settings['pool_recycle'] = 3600
        
        return settings
    
    def optimize_processing_pipeline(
        self,
        pipeline_func,
        sample_data: pd.DataFrame
    ) -> Dict[str, any]:
        """Optimize data processing pipeline."""
        optimizations = {}
        
        # Measure memory usage
        snapshot1 = tracemalloc.take_snapshot()
        pipeline_func(sample_data)
        snapshot2 = tracemalloc.take_snapshot()
        
        # Find memory bottlenecks
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Optimize based on findings
        optimizations['memory_bottlenecks'] = [
            (stat.traceback, stat.size_diff)
            for stat in top_stats[:10]
        ]
        
        # Recommend optimizations
        optimizations['recommendations'] = []
        
        # Check for memory-intensive operations
        if any(stat.size_diff > 1e6 for stat in top_stats):
            optimizations['recommendations'].append(
                "Consider using generators for large data processing"
            )
        
        # Check for repeated calculations
        if any('calculate' in str(stat.traceback) for stat in top_stats):
            optimizations['recommendations'].append(
                "Consider caching intermediate results"
            )
        
        return optimizations
    
    async def measure_end_to_end_latency(
        self,
        exchange_url: str,
        strategy_func,
        sample_data: pd.DataFrame
    ) -> LatencyMetrics:
        """Measure end-to-end trading latency."""
        # Measure network latency
        network_metrics = await self.measure_exchange_latency(exchange_url)
        network_latency = network_metrics['avg_latency']
        
        # Measure processing latency
        processing_metrics = self.measure_processing_latency(
            strategy_func,
            sample_data
        )
        processing_latency = processing_metrics['avg_latency']
        
        # Calculate total latency
        total_latency = network_latency + processing_latency
        
        metrics = LatencyMetrics(
            network_latency=network_latency,
            processing_latency=processing_latency,
            total_latency=total_latency,
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def analyze_latency_patterns(self) -> pd.DataFrame:
        """Analyze latency patterns over time."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'network_latency': m.network_latency,
                'processing_latency': m.processing_latency,
                'total_latency': m.total_latency
            }
            for m in self.metrics_history
        ])
        
        # Calculate rolling statistics
        df['rolling_avg'] = df['total_latency'].rolling(window=10).mean()
        df['rolling_std'] = df['total_latency'].rolling(window=10).std()
        
        return df
    
    def generate_optimization_report(self) -> Dict[str, any]:
        """Generate comprehensive latency optimization report."""
        df = self.analyze_latency_patterns()
        
        if df.empty:
            return {"error": "No latency data available"}
        
        report = {
            'summary_statistics': {
                'avg_network_latency': df['network_latency'].mean(),
                'avg_processing_latency': df['processing_latency'].mean(),
                'avg_total_latency': df['total_latency'].mean(),
                'p95_total_latency': df['total_latency'].quantile(0.95),
                'p99_total_latency': df['total_latency'].quantile(0.99)
            },
            'trends': {
                'latency_trend': 'improving' if df['rolling_avg'].iloc[-1] < df['rolling_avg'].iloc[0]
                               else 'degrading',
                'volatility': df['rolling_std'].mean()
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if report['summary_statistics']['avg_network_latency'] > 0.1:
            report['recommendations'].append(
                "Consider using closer exchange endpoints or CDN"
            )
        
        if report['summary_statistics']['avg_processing_latency'] > 0.05:
            report['recommendations'].append(
                "Optimize data processing pipeline"
            )
        
        if report['trends']['volatility'] > 0.1:
            report['recommendations'].append(
                "Implement better error handling and retry mechanisms"
            )
        
        return report
    
    def plot_latency_trends(self, save_path: str = 'latency_trends.png'):
        """Plot latency trends over time."""
        df = self.analyze_latency_patterns()
        
        if df.empty:
            logger.warning("No data available for plotting")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot latencies
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['network_latency'], label='Network Latency')
        plt.plot(df['timestamp'], df['processing_latency'], label='Processing Latency')
        plt.plot(df['timestamp'], df['total_latency'], label='Total Latency')
        plt.fill_between(df['timestamp'],
                        df['rolling_avg'] - df['rolling_std'],
                        df['rolling_avg'] + df['rolling_std'],
                        alpha=0.2)
        plt.legend()
        plt.title('Latency Trends')
        plt.ylabel('Latency (seconds)')
        
        # Plot distribution
        plt.subplot(2, 1, 2)
        df['total_latency'].hist(bins=50)
        plt.title('Latency Distribution')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

async def main():
    """Main function."""
    optimizer = LatencyOptimizer()
    
    try:
        # Example exchange URL (replace with actual endpoints)
        exchange_url = "https://api.binance.com/api/v3/time"
        
        # Example strategy function
        def sample_strategy(data):
            time.sleep(0.01)  # Simulate processing
            return data.mean()
        
        # Example data
        sample_data = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        
        logger.info("Measuring baseline latency...")
        baseline = await optimizer.measure_end_to_end_latency(
            exchange_url,
            sample_strategy,
            sample_data
        )
        
        logger.info("Optimizing network settings...")
        network_settings = optimizer.optimize_network_settings()
        
        logger.info("Optimizing processing pipeline...")
        pipeline_opts = optimizer.optimize_processing_pipeline(
            sample_strategy,
            sample_data
        )
        
        # Measure optimized latency
        logger.info("Measuring optimized latency...")
        for _ in range(10):
            await optimizer.measure_end_to_end_latency(
                exchange_url,
                sample_strategy,
                sample_data
            )
        
        # Generate and print report
        logger.info("Generating optimization report...")
        report = optimizer.generate_optimization_report()
        print("\nLatency Optimization Report:")
        print(json.dumps(report, indent=2))
        
        # Plot trends
        logger.info("Plotting latency trends...")
        optimizer.plot_latency_trends()
        
        logger.info("Optimization analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during latency optimization: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
