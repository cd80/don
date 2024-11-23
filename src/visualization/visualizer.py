import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class TradingVisualizer:
    """
    Visualization tools for trading data and model performance.
    """
    
    def __init__(
        self,
        output_dir: str = "results/evaluation",
        style: str = "dark"
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Plot style ('dark' or 'light')
        """
        self.output_dir = output_dir
        self.style = style
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Set plot style
        if style == "dark":
            plt.style.use('dark_background')
        
    def plot_portfolio_performance(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        title: str = "Portfolio Performance"
    ) -> None:
        """
        Plot portfolio performance over time.
        
        Args:
            portfolio_values: Series of portfolio values
            benchmark_values: Optional benchmark values
            title: Plot title
        """
        fig = go.Figure()
        
        # Add portfolio line
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                name="Portfolio",
                line=dict(color='#00ff00')
            )
        )
        
        # Add benchmark if provided
        if benchmark_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_values.index,
                    y=benchmark_values.values,
                    name="Benchmark",
                    line=dict(color='#888888')
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark" if self.style == "dark" else "plotly_white"
        )
        
        # Save plot
        filename = f"portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(os.path.join(self.output_dir, filename))
        
    def plot_trading_signals(
        self,
        prices: pd.Series,
        signals: pd.Series,
        title: str = "Trading Signals"
    ) -> None:
        """
        Plot trading signals with price data.
        
        Args:
            prices: Series of price data
            signals: Series of trading signals
            title: Plot title
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, "Trading Signals")
        )
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                name="Price",
                line=dict(color='#00ff00')
            ),
            row=1, col=1
        )
        
        # Add signals
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals.values,
                name="Signal",
                line=dict(color='#ff0000')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_dark" if self.style == "dark" else "plotly_white"
        )
        
        # Save plot
        filename = f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(os.path.join(self.output_dir, filename))
        
    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        timestamps: List[str],
        title: str = "Attention Weights"
    ) -> None:
        """
        Plot attention weights heatmap.
        
        Args:
            attention_weights: 2D array of attention weights
            timestamps: List of timestamp labels
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            attention_weights,
            xticklabels=timestamps,
            yticklabels=timestamps,
            cmap='viridis',
            center=0
        )
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        filename = f"attention_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(
            os.path.join(self.output_dir, filename),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
    def create_performance_dashboard(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        positions: pd.Series,
        metrics: Dict[str, float]
    ) -> None:
        """
        Create comprehensive performance dashboard.
        
        Args:
            portfolio_values: Series of portfolio values
            returns: Series of returns
            positions: Series of position sizes
            metrics: Dictionary of performance metrics
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value',
                'Returns Distribution',
                'Position Sizes',
                'Rolling Metrics',
                'Drawdown',
                'Performance Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # Portfolio Value
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                name="Portfolio Value"
            ),
            row=1, col=1
        )
        
        # Returns Distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name="Returns",
                nbinsx=50
            ),
            row=1, col=2
        )
        
        # Position Sizes
        fig.add_trace(
            go.Scatter(
                x=positions.index,
                y=positions.values,
                name="Position Size"
            ),
            row=2, col=1
        )
        
        # Rolling Metrics
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std()
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="Rolling Sharpe"
            ),
            row=2, col=2
        )
        
        # Drawdown
        drawdown = (portfolio_values / portfolio_values.cummax() - 1)
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        # Metrics Table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=[
                        list(metrics.keys()),
                        list(metrics.values())
                    ],
                    align="left"
                )
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            template="plotly_dark" if self.style == "dark" else "plotly_white"
        )
        
        # Save dashboard
        filename = f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(os.path.join(self.output_dir, filename))
        
    def plot_risk_metrics(
        self,
        returns: pd.Series,
        var_confidence_levels: List[float] = [0.95, 0.99],
        title: str = "Risk Metrics"
    ) -> None:
        """
        Plot risk metrics including VaR and Expected Shortfall.
        
        Args:
            returns: Series of returns
            var_confidence_levels: List of VaR confidence levels
            title: Plot title
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Returns Distribution',
                'Value at Risk',
                'Rolling Volatility',
                'Rolling VaR'
            )
        )
        
        # Returns Distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name="Returns",
                nbinsx=50
            ),
            row=1, col=1
        )
        
        # VaR Lines
        for conf in var_confidence_levels:
            var = np.percentile(returns, (1 - conf) * 100)
            fig.add_vline(
                x=var,
                line_dash="dash",
                annotation_text=f"VaR {conf:.0%}",
                row=1, col=1
            )
        
        # Rolling Volatility
        rolling_vol = returns.rolling(window=252).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name="Annual Volatility"
            ),
            row=2, col=1
        )
        
        # Rolling VaR
        rolling_var = returns.rolling(window=252).quantile(0.05)
        fig.add_trace(
            go.Scatter(
                x=rolling_var.index,
                y=rolling_var.values,
                name="Rolling VaR (95%)"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=True,
            template="plotly_dark" if self.style == "dark" else "plotly_white"
        )
        
        # Save plot
        filename = f"risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(os.path.join(self.output_dir, filename))

if __name__ == "__main__":
    # Example usage
    visualizer = TradingVisualizer()
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    portfolio_values = pd.Series(
        np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
        index=dates
    )
    returns = portfolio_values.pct_change().dropna()
    positions = pd.Series(
        np.random.uniform(-1, 1, len(dates)),
        index=dates
    )
    
    # Create visualizations
    visualizer.plot_portfolio_performance(portfolio_values)
    visualizer.plot_trading_signals(portfolio_values, positions)
    visualizer.create_performance_dashboard(
        portfolio_values,
        returns,
        positions,
        {
            'Sharpe Ratio': 1.5,
            'Max Drawdown': -0.2,
            'Annual Return': 0.15,
            'Win Rate': 0.55
        }
    )
    visualizer.plot_risk_metrics(returns)
