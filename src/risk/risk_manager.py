"""
Dynamic Risk Management Module for Bitcoin Trading RL.
Implements adaptive risk management strategies based on market conditions and model confidence.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class MarketRegime(Enum):
    """Market regime classification."""
    LOW_VOLATILITY = 'low_volatility'
    MEDIUM_VOLATILITY = 'medium_volatility'
    HIGH_VOLATILITY = 'high_volatility'
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    BREAKOUT = 'breakout'

@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    var: float  # Value at Risk
    es: float   # Expected Shortfall
    volatility: float
    drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    model_confidence: float
    market_regime: MarketRegime

class DynamicRiskManager:
    """
    Dynamic risk management system that adjusts position sizes and risk limits
    based on market conditions, model confidence, and portfolio performance.
    """
    
    def __init__(
        self,
        config: Dict,
        initial_capital: float,
        max_position_size: float = 1.0,
        max_leverage: float = 3.0,
        confidence_level: float = 0.95
    ):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
            initial_capital: Initial capital
            max_position_size: Maximum position size as fraction of capital
            max_leverage: Maximum allowed leverage
            confidence_level: Confidence level for VaR calculation
        """
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.confidence_level = confidence_level
        
        # Risk limits
        self.var_limit = config.get('var_limit', 0.02)
        self.drawdown_limit = config.get('drawdown_limit', 0.20)
        self.volatility_limit = config.get('volatility_limit', 0.03)
        
        # Market state
        self.market_regime = MarketRegime.MEDIUM_VOLATILITY
        self.volatility_window = config.get('volatility_window', 20)
        self.trend_window = config.get('trend_window', 50)
        
        # Position tracking
        self.current_position = 0.0
        self.position_history = []
        self.pnl_history = []
        
        logger.info("Initialized dynamic risk manager")
    
    def calculate_risk_metrics(
        self,
        returns: np.ndarray,
        model_confidence: float
    ) -> RiskMetrics:
        """
        Calculate current risk metrics.
        
        Args:
            returns: Historical returns
            model_confidence: Current model confidence score
            
        Returns:
            RiskMetrics object
        """
        # Calculate Value at Risk
        var = -np.percentile(returns, (1 - self.confidence_level) * 100)
        
        # Calculate Expected Shortfall
        es = -np.mean(returns[returns < -var])
        
        # Calculate volatility
        volatility = np.std(returns)
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + returns)
        drawdown = 1 - cumulative_returns / np.maximum.accumulate(cumulative_returns)
        max_drawdown = np.max(drawdown)
        
        # Calculate ratios
        excess_returns = returns - self.config.get('risk_free_rate', 0.0)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        sortino_ratio = np.mean(excess_returns) / np.std(excess_returns[excess_returns < 0])
        calmar_ratio = np.mean(excess_returns) / max_drawdown if max_drawdown > 0 else np.inf
        
        return RiskMetrics(
            var=var,
            es=es,
            volatility=volatility,
            drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            model_confidence=model_confidence,
            market_regime=self.market_regime
        )
    
    def detect_market_regime(
        self,
        prices: np.ndarray,
        returns: np.ndarray
    ) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            prices: Historical prices
            returns: Historical returns
            
        Returns:
            Current market regime
        """
        # Calculate volatility
        volatility = np.std(returns[-self.volatility_window:])
        volatility_percentile = stats.percentileofscore(
            np.std(returns.reshape(-1, self.volatility_window), axis=1),
            volatility
        )
        
        # Calculate trend
        trend = (prices[-1] - prices[-self.trend_window]) / prices[-self.trend_window]
        trend_strength = abs(trend)
        
        # Detect breakout
        recent_high = np.max(prices[-self.trend_window:-1])
        recent_low = np.min(prices[-self.trend_window:-1])
        is_breakout = (prices[-1] > recent_high * 1.02 or
                      prices[-1] < recent_low * 0.98)
        
        # Determine regime
        if is_breakout:
            return MarketRegime.BREAKOUT
        elif volatility_percentile > 80:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility_percentile < 20:
            return MarketRegime.LOW_VOLATILITY
        elif trend > 0.02 and trend_strength > 0.05:
            return MarketRegime.TRENDING_UP
        elif trend < -0.02 and trend_strength > 0.05:
            return MarketRegime.TRENDING_DOWN
        elif trend_strength < 0.02:
            return MarketRegime.RANGING
        else:
            return MarketRegime.MEDIUM_VOLATILITY
    
    def calculate_position_size(
        self,
        prediction: float,
        confidence: float,
        risk_metrics: RiskMetrics
    ) -> float:
        """
        Calculate dynamic position size based on prediction and risk metrics.
        
        Args:
            prediction: Model's price movement prediction
            confidence: Model's confidence in prediction
            risk_metrics: Current risk metrics
            
        Returns:
            Position size as fraction of capital (-1 to 1)
        """
        # Base position size from prediction
        position_size = prediction * self.max_position_size
        
        # Adjust for model confidence
        position_size *= confidence
        
        # Adjust for market regime
        regime_factors = {
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.MEDIUM_VOLATILITY: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.RANGING: 0.7,
            MarketRegime.BREAKOUT: 0.6
        }
        position_size *= regime_factors[risk_metrics.market_regime]
        
        # Risk-based adjustments
        if risk_metrics.var > self.var_limit:
            position_size *= self.var_limit / risk_metrics.var
        
        if risk_metrics.drawdown > self.drawdown_limit:
            position_size *= (1 - risk_metrics.drawdown / self.drawdown_limit)
        
        if risk_metrics.volatility > self.volatility_limit:
            position_size *= self.volatility_limit / risk_metrics.volatility
        
        # Apply leverage limit
        position_size = np.clip(position_size, -self.max_leverage, self.max_leverage)
        
        return float(position_size)
    
    def update_position(
        self,
        new_position: float,
        current_price: float
    ) -> Dict[str, float]:
        """
        Update current position and track P&L.
        
        Args:
            new_position: New position size
            current_price: Current asset price
            
        Returns:
            Dictionary of position metrics
        """
        # Calculate P&L if position changes
        if self.current_position != 0:
            pnl = (current_price - self.position_history[-1]['price'])
            pnl *= self.current_position
            self.pnl_history.append(pnl)
            self.current_capital += pnl
        
        # Update position
        self.current_position = new_position
        self.position_history.append({
            'size': new_position,
            'price': current_price,
            'capital': self.current_capital
        })
        
        return {
            'position_size': new_position,
            'capital': self.current_capital,
            'total_pnl': sum(self.pnl_history),
            'return': (self.current_capital - self.initial_capital) / self.initial_capital
        }
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return {
            'var_limit': self.var_limit,
            'drawdown_limit': self.drawdown_limit,
            'volatility_limit': self.volatility_limit,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage
        }
    
    def adjust_risk_limits(
        self,
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Dynamically adjust risk limits based on performance.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Adjust VaR limit based on Sharpe ratio
        if performance_metrics['sharpe_ratio'] > 2.0:
            self.var_limit *= 1.1
        elif performance_metrics['sharpe_ratio'] < 0.5:
            self.var_limit *= 0.9
        
        # Adjust position size based on Calmar ratio
        if performance_metrics['calmar_ratio'] > 1.5:
            self.max_position_size = min(self.max_position_size * 1.1, 1.0)
        elif performance_metrics['calmar_ratio'] < 0.5:
            self.max_position_size *= 0.9
        
        # Adjust leverage based on volatility
        if performance_metrics['volatility'] < self.volatility_limit * 0.5:
            self.max_leverage = min(self.max_leverage * 1.1, 3.0)
        elif performance_metrics['volatility'] > self.volatility_limit:
            self.max_leverage *= 0.9
        
        logger.info("Adjusted risk limits based on performance")
