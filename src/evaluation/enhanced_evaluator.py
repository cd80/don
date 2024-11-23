"""
Enhanced Model Evaluation Module for Bitcoin Trading RL.
Implements comprehensive model assessment and performance analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class EvaluationMetric(Enum):
    """Available evaluation metrics."""
    ACCURACY = 'accuracy'
    SHARPE_RATIO = 'sharpe_ratio'
    SORTINO_RATIO = 'sortino_ratio'
    MAX_DRAWDOWN = 'max_drawdown'
    ALPHA = 'alpha'
    BETA = 'beta'
    INFORMATION_RATIO = 'information_ratio'
    CALMAR_RATIO = 'calmar_ratio'
    WIN_RATE = 'win_rate'
    PROFIT_FACTOR = 'profit_factor'
    KELLY_CRITERION = 'kelly_criterion'

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    returns: np.ndarray
    positions: np.ndarray
    metrics: Dict[str, float]
    drawdown: np.ndarray
    rolling_metrics: pd.DataFrame
    trade_analysis: Dict[str, float]
    risk_metrics: Dict[str, float]

class EnhancedEvaluator:
    """
    Enhanced model evaluation system that provides comprehensive
    performance analysis and visualization.
    """
    
    def __init__(
        self,
        config: Dict,
        risk_free_rate: float = 0.0,
        benchmark_returns: Optional[np.ndarray] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
            risk_free_rate: Risk-free rate for ratio calculations
            benchmark_returns: Optional benchmark returns for comparison
        """
        self.config = config
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        
        logger.info("Initialized enhanced evaluator")
    
    def calculate_returns(
        self,
        positions: np.ndarray,
        price_returns: np.ndarray,
        include_costs: bool = True
    ) -> np.ndarray:
        """
        Calculate strategy returns.
        
        Args:
            positions: Position sizes
            price_returns: Asset returns
            include_costs: Whether to include transaction costs
            
        Returns:
            Strategy returns
        """
        # Calculate basic returns
        strategy_returns = positions * price_returns
        
        if include_costs:
            # Calculate position changes
            position_changes = np.abs(np.diff(positions, prepend=0))
            
            # Apply transaction costs
            transaction_costs = position_changes * self.config.get(
                'transaction_cost',
                0.001
            )
            strategy_returns -= transaction_costs
        
        return strategy_returns
    
    def calculate_metrics(
        self,
        returns: np.ndarray,
        positions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            returns: Strategy returns
            positions: Position sizes
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = np.prod(1 + returns) - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (
            252 / len(returns)
        ) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Risk-adjusted returns
        excess_returns = returns - self.risk_free_rate
        metrics['sharpe_ratio'] = (excess_returns.mean() / returns.std() *
                                 np.sqrt(252))
        
        downside_returns = returns[returns < 0]
        metrics['sortino_ratio'] = (excess_returns.mean() / downside_returns.std() *
                                  np.sqrt(252))
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Calculate alpha and beta if benchmark available
        if self.benchmark_returns is not None:
            covariance = np.cov(returns, self.benchmark_returns)[0, 1]
            variance = np.var(self.benchmark_returns)
            metrics['beta'] = covariance / variance
            metrics['alpha'] = (returns.mean() - self.risk_free_rate -
                              metrics['beta'] * (self.benchmark_returns.mean() -
                                               self.risk_free_rate))
            
            # Information ratio
            active_returns = returns - self.benchmark_returns
            metrics['information_ratio'] = (active_returns.mean() /
                                          active_returns.std() * np.sqrt(252))
        
        # Trading metrics
        trades = np.diff(positions, prepend=0)
        winning_trades = returns[trades != 0][returns[trades != 0] > 0]
        losing_trades = returns[trades != 0][returns[trades != 0] < 0]
        
        metrics['win_rate'] = len(winning_trades) / (
            len(winning_trades) + len(losing_trades)
        )
        
        metrics['profit_factor'] = (abs(winning_trades.sum()) /
                                  abs(losing_trades.sum())
                                  if len(losing_trades) > 0 else np.inf)
        
        # Kelly criterion
        win_prob = metrics['win_rate']
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 1
        metrics['kelly_criterion'] = (win_prob / avg_loss -
                                    (1 - win_prob) / avg_win
                                    if avg_win > 0 else 0)
        
        return metrics
    
    def calculate_rolling_metrics(
        self,
        returns: np.ndarray,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Strategy returns
            window: Rolling window size
            
        Returns:
            DataFrame of rolling metrics
        """
        rolling_metrics = pd.DataFrame()
        
        # Rolling returns
        rolling_returns = pd.Series(returns).rolling(window)
        rolling_metrics['return'] = rolling_returns.mean() * 252
        rolling_metrics['volatility'] = rolling_returns.std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        excess_returns = returns - self.risk_free_rate
        rolling_metrics['sharpe_ratio'] = (
            rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
        )
        
        # Rolling drawdown
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = pd.Series(cumulative_returns).rolling(
            window, min_periods=1
        ).max()
        rolling_metrics['drawdown'] = (cumulative_returns - rolling_max) / rolling_max
        
        return rolling_metrics
    
    def analyze_trades(
        self,
        returns: np.ndarray,
        positions: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze trading performance.
        
        Args:
            returns: Strategy returns
            positions: Position sizes
            
        Returns:
            Dictionary of trade analysis
        """
        trades = np.diff(positions, prepend=0)
        trade_returns = returns[trades != 0]
        
        analysis = {}
        
        # Trade statistics
        analysis['num_trades'] = len(trade_returns)
        analysis['avg_trade_return'] = trade_returns.mean()
        analysis['trade_return_std'] = trade_returns.std()
        
        # Winning trades
        winning_trades = trade_returns[trade_returns > 0]
        analysis['num_winning_trades'] = len(winning_trades)
        analysis['avg_winning_trade'] = winning_trades.mean()
        analysis['max_winning_trade'] = winning_trades.max()
        
        # Losing trades
        losing_trades = trade_returns[trade_returns < 0]
        analysis['num_losing_trades'] = len(losing_trades)
        analysis['avg_losing_trade'] = losing_trades.mean()
        analysis['max_losing_trade'] = losing_trades.min()
        
        # Trade duration
        trade_durations = []
        current_duration = 0
        current_position = 0
        
        for pos in positions:
            if pos != current_position:
                if current_duration > 0:
                    trade_durations.append(current_duration)
                current_duration = 0
                current_position = pos
            current_duration += 1
        
        if trade_durations:
            analysis['avg_trade_duration'] = np.mean(trade_durations)
            analysis['max_trade_duration'] = max(trade_durations)
            analysis['min_trade_duration'] = min(trade_durations)
        
        return analysis
    
    def calculate_risk_metrics(
        self,
        returns: np.ndarray,
        positions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            returns: Strategy returns
            positions: Position sizes
            
        Returns:
            Dictionary of risk metrics
        """
        risk_metrics = {}
        
        # Value at Risk
        risk_metrics['var_95'] = np.percentile(returns, 5)
        risk_metrics['var_99'] = np.percentile(returns, 1)
        
        # Expected Shortfall
        risk_metrics['es_95'] = returns[returns <= risk_metrics['var_95']].mean()
        risk_metrics['es_99'] = returns[returns <= risk_metrics['var_99']].mean()
        
        # Position concentration
        risk_metrics['avg_position_size'] = np.abs(positions).mean()
        risk_metrics['max_position_size'] = np.abs(positions).max()
        
        # Risk exposure
        risk_metrics['time_in_market'] = np.mean(positions != 0)
        risk_metrics['long_exposure'] = np.mean(positions > 0)
        risk_metrics['short_exposure'] = np.mean(positions < 0)
        
        return risk_metrics
    
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prices: np.ndarray
    ) -> PerformanceMetrics:
        """
        Perform comprehensive evaluation.
        
        Args:
            predictions: Model predictions
            targets: True values
            prices: Asset prices
            
        Returns:
            PerformanceMetrics object
        """
        # Calculate positions from predictions
        positions = np.sign(predictions)
        
        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = self.calculate_returns(
            positions[:-1],
            price_returns,
            include_costs=True
        )
        
        # Calculate metrics
        metrics = self.calculate_metrics(strategy_returns, positions[:-1])
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        # Calculate rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(strategy_returns)
        
        # Analyze trades
        trade_analysis = self.analyze_trades(strategy_returns, positions[:-1])
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(strategy_returns, positions[:-1])
        
        return PerformanceMetrics(
            returns=strategy_returns,
            positions=positions[:-1],
            metrics=metrics,
            drawdown=drawdown,
            rolling_metrics=rolling_metrics,
            trade_analysis=trade_analysis,
            risk_metrics=risk_metrics
        )
    
    def plot_results(
        self,
        performance: PerformanceMetrics,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot evaluation results.
        
        Args:
            performance: PerformanceMetrics object
            save_path: Optional path to save plots
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Cumulative returns
        ax1 = plt.subplot(3, 2, 1)
        cumulative_returns = np.cumprod(1 + performance.returns)
        ax1.plot(cumulative_returns)
        if self.benchmark_returns is not None:
            benchmark_cum_returns = np.cumprod(
                1 + self.benchmark_returns[:len(performance.returns)]
            )
            ax1.plot(benchmark_cum_returns, '--', alpha=0.7)
        ax1.set_title('Cumulative Returns')
        ax1.legend(['Strategy', 'Benchmark'])
        
        # Drawdown
        ax2 = plt.subplot(3, 2, 2)
        ax2.fill_between(range(len(performance.drawdown)),
                        performance.drawdown,
                        0,
                        alpha=0.3)
        ax2.set_title('Drawdown')
        
        # Rolling metrics
        ax3 = plt.subplot(3, 2, 3)
        performance.rolling_metrics[['return', 'volatility']].plot(ax=ax3)
        ax3.set_title('Rolling Returns and Volatility')
        
        # Position distribution
        ax4 = plt.subplot(3, 2, 4)
        sns.histplot(performance.positions, ax=ax4)
        ax4.set_title('Position Distribution')
        
        # Trade analysis
        ax5 = plt.subplot(3, 2, 5)
        trade_returns = performance.returns[np.diff(performance.positions,
                                                  prepend=0) != 0]
        sns.histplot(trade_returns, ax=ax5)
        ax5.set_title('Trade Returns Distribution')
        
        # Risk metrics
        ax6 = plt.subplot(3, 2, 6)
        risk_data = pd.Series(performance.risk_metrics)
        risk_data.plot(kind='bar', ax=ax6)
        ax6.set_title('Risk Metrics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def generate_report(
        self,
        performance: PerformanceMetrics,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate performance report.
        
        Args:
            performance: PerformanceMetrics object
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        report = "Performance Report\n"
        report += "=================\n\n"
        
        # Performance metrics
        report += "Performance Metrics:\n"
        report += "------------------\n"
        for metric, value in performance.metrics.items():
            report += f"{metric}: {value:.4f}\n"
        report += "\n"
        
        # Trade analysis
        report += "Trade Analysis:\n"
        report += "--------------\n"
        for metric, value in performance.trade_analysis.items():
            report += f"{metric}: {value:.4f}\n"
        report += "\n"
        
        # Risk metrics
        report += "Risk Metrics:\n"
        report += "------------\n"
        for metric, value in performance.risk_metrics.items():
            report += f"{metric}: {value:.4f}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
