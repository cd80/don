import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Evaluator:
    """
    Comprehensive evaluation and visualization of trading model performance.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_data: pd.DataFrame,
        config: Dict[str, Any],
        output_dir: str = "results/evaluation",
        device: str = "cpu"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained model
            test_data: Test dataset
            config: Configuration dictionary
            output_dir: Directory for saving results
            device: Computing device
        """
        self.model = model
        self.test_data = test_data
        self.config = config
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize metrics storage
        self.metrics = {}
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        
    def calculate_metrics(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
        portfolio_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Array of returns
            positions: Array of positions
            portfolio_values: Array of portfolio values
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        daily_returns = pd.Series(returns).resample('D').sum()
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values)) - 1
        
        # Risk-adjusted returns
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        sortino_ratio = np.mean(returns) / (np.std(downside_returns) + 1e-6) * np.sqrt(252)
        
        # Trading metrics
        position_changes = np.diff(positions)
        trades = np.count_nonzero(position_changes)
        winning_trades = np.sum(returns > 0)
        losing_trades = np.sum(returns < 0)
        win_rate = winning_trades / (winning_trades + losing_trades)
        
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # Calculate maximum consecutive wins/losses
        trade_results = np.where(returns > 0, 1, -1)
        max_consecutive_wins = self._max_consecutive(trade_results, 1)
        max_consecutive_losses = self._max_consecutive(trade_results, -1)
        
        # Calculate profit factor
        gross_profits = np.sum(returns[returns > 0])
        gross_losses = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profits / (gross_losses + 1e-6)
        
        # Statistical metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Compile metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(returns)),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'number_of_trades': trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        return metrics
    
    def _max_consecutive(
        self,
        arr: np.ndarray,
        value: int
    ) -> int:
        """
        Calculate maximum consecutive occurrences of a value.
        
        Args:
            arr: Input array
            value: Value to count
            
        Returns:
            Maximum consecutive count
        """
        mask = np.concatenate(([False], arr == value, [False]))
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        lengths = ends - starts
        return max(lengths) if len(lengths) > 0 else 0
    
    def create_visualizations(self) -> None:
        """
        Create comprehensive performance visualizations.
        """
        # Convert data to pandas Series for easier plotting
        dates = pd.to_datetime(self.test_data.index)
        portfolio_values = pd.Series(self.portfolio_values, index=dates)
        returns = pd.Series(np.diff(np.log(self.portfolio_values)), index=dates[1:])
        positions = pd.Series(self.positions, index=dates)
        
        # Create interactive dashboard using plotly
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Portfolio Value', 'Drawdown',
                'Returns Distribution', 'Position Size',
                'Rolling Sharpe Ratio', 'Rolling Volatility',
                'Trade Analysis', 'Risk Metrics'
            )
        )
        
        # Portfolio Value
        fig.add_trace(
            go.Scatter(x=dates, y=portfolio_values, name='Portfolio Value'),
            row=1, col=1
        )
        
        # Drawdown
        drawdown = (portfolio_values / portfolio_values.cummax() - 1)
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, name='Drawdown', fill='tonexty'),
            row=1, col=2
        )
        
        # Returns Distribution
        fig.add_trace(
            go.Histogram(x=returns, name='Returns Distribution', nbinsx=50),
            row=2, col=1
        )
        
        # Position Size
        fig.add_trace(
            go.Scatter(x=dates, y=positions, name='Position Size'),
            row=2, col=2
        )
        
        # Rolling Metrics
        window = 252  # One year
        rolling_returns = returns.rolling(window=window)
        rolling_sharpe = (
            np.sqrt(252) * rolling_returns.mean() / 
            (rolling_returns.std() + 1e-6)
        )
        rolling_vol = np.sqrt(252) * rolling_returns.std()
        
        fig.add_trace(
            go.Scatter(x=dates[window:], y=rolling_sharpe[window:],
                      name='Rolling Sharpe'),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates[window:], y=rolling_vol[window:],
                      name='Rolling Volatility'),
            row=3, col=2
        )
        
        # Trade Analysis
        trade_returns = returns[positions.shift() != 0]
        fig.add_trace(
            go.Box(y=trade_returns, name='Trade Returns Distribution'),
            row=4, col=1
        )
        
        # Risk Metrics Table
        risk_metrics = pd.DataFrame({
            'Metric': ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown'],
            'Value': [
                self.metrics['var_95'],
                self.metrics['cvar_95'],
                self.metrics['max_drawdown']
            ]
        })
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(risk_metrics.columns)),
                cells=dict(values=[risk_metrics[col] for col in risk_metrics.columns])
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            title_text="Trading Strategy Performance Analysis"
        )
        
        # Save plot
        fig.write_html(os.path.join(self.output_dir, 'performance_dashboard.html'))
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(os.path.join(self.output_dir, 'metrics.csv'))
        
        self.logger.info(f"Visualizations saved to {self.output_dir}")
    
    def backtest(self) -> Dict[str, float]:
        """
        Perform comprehensive backtesting of the model.
        
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("Starting backtesting...")
        
        # Initialize variables
        portfolio_value = self.config['initial_balance']
        position = 0
        returns = []
        
        # Convert data to tensor
        data_tensor = torch.tensor(
            self.test_data.values,
            dtype=torch.float32,
            device=self.device
        )
        
        # Backtest loop
        for i in range(len(self.test_data) - 1):
            # Get state
            state = data_tensor[i:i+1]
            
            # Get action from model
            with torch.no_grad():
                _, _, action_dist, _ = self.model(state)
                action = action_dist.sample()
            
            # Execute action
            new_position = float(action.cpu().numpy()[0])
            position_change = new_position - position
            
            # Calculate transaction costs
            price = self.test_data.iloc[i]['close']
            transaction_cost = abs(position_change) * price * self.config['transaction_fee']
            
            # Calculate returns
            price_change = (
                self.test_data.iloc[i+1]['close'] - price
            ) / price
            
            period_return = (
                position * price_change - transaction_cost / portfolio_value
            )
            
            # Update portfolio value
            portfolio_value *= (1 + period_return)
            
            # Store results
            self.portfolio_values.append(portfolio_value)
            self.positions.append(new_position)
            returns.append(period_return)
            
            # Update position
            position = new_position
        
        # Calculate metrics
        self.metrics = self.calculate_metrics(
            np.array(returns),
            np.array(self.positions),
            np.array(self.portfolio_values)
        )
        
        # Create visualizations
        self.create_visualizations()
        
        self.logger.info("Backtesting completed.")
        self.logger.info(f"Final portfolio value: {portfolio_value:.2f}")
        self.logger.info(f"Sharpe ratio: {self.metrics['sharpe_ratio']:.2f}")
        
        return self.metrics

if __name__ == "__main__":
    # Example usage
    from ..models.base_model import BaseModel
    
    # Load test data
    test_data = pd.read_parquet("data/processed/test_features.parquet")
    
    # Load model
    model = BaseModel(
        state_dim=test_data.shape[1],
        action_dim=1,
        hidden_dim=256
    )
    model.load("results/checkpoints/best_model.pt")
    
    # Configuration
    config = {
        'initial_balance': 100000,
        'transaction_fee': 0.001
    }
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_data=test_data,
        config=config
    )
    
    # Run backtesting
    metrics = evaluator.backtest()
