from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseReward(ABC):
    """Base class for reward calculation strategies."""

    @abstractmethod
    def calculate(self, action: float, position: float,
                 pnl: float, **kwargs) -> float:
        """Calculate reward based on action and state.

        Args:
            action: Taken action (position size)
            position: Current position
            pnl: Profit and loss
            **kwargs: Additional state information

        Returns:
            Calculated reward value
        """
        pass


class PnLReward(BaseReward):
    """Simple PnL-based reward strategy."""

    def calculate(self, action: float, position: float,
                 pnl: float, **kwargs) -> float:
        """Calculate reward as change in PnL."""
        return pnl


class SharpeReward(BaseReward):
    """Sharpe ratio based reward strategy."""

    def __init__(self, window: int = 100, risk_free_rate: float = 0.0):
        """Initialize Sharpe reward calculator.

        Args:
            window: Rolling window for Sharpe calculation
            risk_free_rate: Risk-free rate (annualized)
        """
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.returns_history = []

    def calculate(self, action: float, position: float,
                 pnl: float, **kwargs) -> float:
        """Calculate reward using Sharpe ratio.

        Args:
            action: Taken action (position size)
            position: Current position
            pnl: Profit and loss
            **kwargs: Additional state information

        Returns:
            Sharpe ratio based reward
        """
        self.returns_history.append(pnl)
        if len(self.returns_history) < self.window:
            return 0.0

        # Keep only last window returns
        self.returns_history = self.returns_history[-self.window:]

        returns = np.array(self.returns_history)
        excess_returns = returns - self.risk_free_rate / 252  # Daily adjustment

        if len(returns) < 2:
            return 0.0

        sharpe = np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))
        return sharpe


class RiskAdjustedReward(BaseReward):
    """Risk-adjusted reward strategy combining PnL and position penalties."""

    def __init__(self, position_penalty: float = 0.01):
        """Initialize risk-adjusted reward calculator.

        Args:
            position_penalty: Penalty factor for position size
        """
        self.position_penalty = position_penalty

    def calculate(self, action: float, position: float,
                 pnl: float, **kwargs) -> float:
        """Calculate risk-adjusted reward.

        Args:
            action: Taken action (position size)
            position: Current position
            pnl: Profit and loss
            **kwargs: Additional state information

        Returns:
            Risk-adjusted reward value
        """
        # Penalize large positions and round to avoid floating-point precision issues
        position_cost = round(self.position_penalty * abs(position), 8)
        return round(pnl - position_cost, 8)
