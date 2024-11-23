"""
Statistical Arbitrage Module for Bitcoin Trading RL.
Implements pairs trading, mean reversion, and cointegration analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

from src.utils.helpers import setup_logging
from src.data.binance_fetcher import BinanceFetcher

logger = setup_logging(__name__)

class StatArbType(Enum):
    """Types of statistical arbitrage strategies."""
    PAIRS_TRADING = 'pairs_trading'  # Classic pairs trading
    MEAN_REVERSION = 'mean_reversion'  # Single asset mean reversion
    COINTEGRATION = 'cointegration'  # Multi-asset cointegration
    FACTOR_MODEL = 'factor_model'  # Factor-based arbitrage

@dataclass
class PairAnalysis:
    """Container for pair analysis results."""
    asset_a: str
    asset_b: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    half_life: float
    zscore: float
    spread: pd.Series
    residuals: pd.Series
    metrics: Dict[str, float]

@dataclass
class TradingSignal:
    """Container for trading signals."""
    type: StatArbType
    assets: List[str]
    direction: List[float]  # Position sizes for each asset
    entry_zscore: float
    exit_zscore: float
    confidence: float
    timestamp: float
    metrics: Dict[str, float]

class StatisticalArbitrage:
    """
    Statistical arbitrage strategy that implements pairs trading,
    mean reversion, and cointegration analysis.
    """
    
    def __init__(
        self,
        config: Dict,
        exchanges: List[str],
        lookback_window: int = 100,
        zscore_threshold: float = 2.0,
        min_half_life: float = 1.0,
        max_half_life: float = 100.0,
        min_correlation: float = 0.5,
        max_position: float = 1.0,
        risk_limit: float = 0.02
    ):
        """
        Initialize statistical arbitrage strategy.
        
        Args:
            config: Strategy configuration
            exchanges: List of exchanges to trade on
            lookback_window: Window size for analysis
            zscore_threshold: Z-score threshold for trading
            min_half_life: Minimum mean reversion half-life
            max_half_life: Maximum mean reversion half-life
            min_correlation: Minimum correlation threshold
            max_position: Maximum position size
            risk_limit: Maximum risk limit as fraction of capital
        """
        self.config = config
        self.exchanges = exchanges
        self.lookback_window = lookback_window
        self.zscore_threshold = zscore_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_correlation = min_correlation
        self.max_position = max_position
        self.risk_limit = risk_limit
        
        # Initialize exchange clients
        self.exchange_clients = {
            exchange: BinanceFetcher(config['data'][exchange])
            for exchange in exchanges
        }
        
        # Initialize state tracking
        self.pairs = {}  # Pair analysis results
        self.positions = {}  # Current positions
        self.trades = []  # Trade history
        self.signals = []  # Signal history
        
        logger.info(
            f"Initialized statistical arbitrage with "
            f"{len(exchanges)} exchanges"
        )
    
    async def fetch_price_history(
        self,
        symbols: List[str],
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical prices for multiple assets.
        
        Args:
            symbols: List of trading symbols
            lookback: Optional lookback period
            
        Returns:
            DataFrame of prices
        """
        lookback = lookback or self.lookback_window
        prices = {}
        
        for symbol in symbols:
            # Fetch from primary exchange
            data = await self.exchange_clients[self.exchanges[0]].fetch_historical_data(
                symbol=symbol,
                limit=lookback
            )
            prices[symbol] = data['close']
        
        return pd.DataFrame(prices)
    
    def analyze_pair(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> Optional[PairAnalysis]:
        """
        Analyze potential trading pair.
        
        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            
        Returns:
            Pair analysis results or None
        """
        # Calculate correlation
        correlation = prices_a.corr(prices_b)
        if abs(correlation) < self.min_correlation:
            return None
        
        # Test cointegration
        _, pvalue, _ = coint(prices_a, prices_b)
        if pvalue > 0.05:  # Not cointegrated
            return None
        
        # Calculate hedge ratio
        model = LinearRegression()
        model.fit(prices_b.values.reshape(-1, 1), prices_a.values)
        hedge_ratio = model.coef_[0]
        
        # Calculate spread
        spread = prices_a - hedge_ratio * prices_b
        
        # Calculate half-life
        half_life = self.calculate_half_life(spread)
        if not (self.min_half_life <= half_life <= self.max_half_life):
            return None
        
        # Calculate z-score
        zscore = (spread - spread.mean()) / spread.std()
        
        # Calculate residuals
        residuals = prices_a - (model.intercept_ + hedge_ratio * prices_b)
        
        # Calculate additional metrics
        metrics = {
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'residual_std': residuals.std(),
            'hurst_exponent': self.calculate_hurst_exponent(spread),
            'adf_pvalue': adfuller(spread)[1]
        }
        
        return PairAnalysis(
            asset_a=prices_a.name,
            asset_b=prices_b.name,
            correlation=correlation,
            cointegration_pvalue=pvalue,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            zscore=zscore.iloc[-1],
            spread=spread,
            residuals=residuals,
            metrics=metrics
        )
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life.
        
        Args:
            spread: Price spread series
            
        Returns:
            Half-life in periods
        """
        # Calculate lag-1 regression
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()
        
        model = LinearRegression()
        model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
        
        # Calculate half-life
        lambda_param = -model.coef_[0]
        if lambda_param <= 0:
            return np.inf
        
        return np.log(2) / lambda_param
    
    def calculate_hurst_exponent(
        self,
        series: pd.Series,
        lags: Optional[List[int]] = None
    ) -> float:
        """
        Calculate Hurst exponent for time series.
        
        Args:
            series: Time series
            lags: Optional list of lag periods
            
        Returns:
            Hurst exponent
        """
        lags = lags or [2, 4, 8, 16, 32, 64]
        
        # Calculate range over standard deviation
        rs_values = []
        for lag in lags:
            # Calculate rolling means and standard deviations
            rolling_mean = series.rolling(window=lag).mean()
            rolling_std = series.rolling(window=lag).std()
            
            # Calculate range
            series_norm = (series - rolling_mean) / rolling_std
            ranges = pd.Series([
                max(series_norm[i:i+lag]) - min(series_norm[i:i+lag])
                for i in range(0, len(series)-lag+1)
            ])
            
            rs_values.append(np.log(ranges.mean()))
        
        # Calculate Hurst exponent from slope
        x = np.log(lags)
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), rs_values)
        
        return model.coef_[0]
    
    def generate_pairs_signal(
        self,
        pair: PairAnalysis
    ) -> Optional[TradingSignal]:
        """
        Generate pairs trading signal.
        
        Args:
            pair: Pair analysis results
            
        Returns:
            Trading signal or None
        """
        # Check z-score thresholds
        if abs(pair.zscore) > self.zscore_threshold:
            # Calculate position sizes
            size_a = np.sign(-pair.zscore) * self.max_position
            size_b = -size_a * pair.hedge_ratio
            
            # Adjust for risk limit
            risk = abs(size_a) + abs(size_b)
            if risk > self.risk_limit:
                size_a *= self.risk_limit / risk
                size_b *= self.risk_limit / risk
            
            return TradingSignal(
                type=StatArbType.PAIRS_TRADING,
                assets=[pair.asset_a, pair.asset_b],
                direction=[size_a, size_b],
                entry_zscore=pair.zscore,
                exit_zscore=0.0,
                confidence=min(abs(pair.zscore) / self.zscore_threshold, 1.0),
                timestamp=time.time(),
                metrics={
                    'correlation': pair.correlation,
                    'half_life': pair.half_life,
                    'hedge_ratio': pair.hedge_ratio
                }
            )
        
        return None
    
    def generate_mean_reversion_signal(
        self,
        prices: pd.Series
    ) -> Optional[TradingSignal]:
        """
        Generate mean reversion signal.
        
        Args:
            prices: Price series
            
        Returns:
            Trading signal or None
        """
        # Calculate z-score
        rolling_mean = prices.rolling(window=self.lookback_window).mean()
        rolling_std = prices.rolling(window=self.lookback_window).std()
        zscore = (prices - rolling_mean) / rolling_std
        
        # Calculate half-life
        half_life = self.calculate_half_life(prices)
        if not (self.min_half_life <= half_life <= self.max_half_life):
            return None
        
        # Check z-score threshold
        current_zscore = zscore.iloc[-1]
        if abs(current_zscore) > self.zscore_threshold:
            # Calculate position size
            size = np.sign(-current_zscore) * self.max_position
            
            # Adjust for risk limit
            if abs(size) > self.risk_limit:
                size *= self.risk_limit / abs(size)
            
            return TradingSignal(
                type=StatArbType.MEAN_REVERSION,
                assets=[prices.name],
                direction=[size],
                entry_zscore=current_zscore,
                exit_zscore=0.0,
                confidence=min(abs(current_zscore) / self.zscore_threshold, 1.0),
                timestamp=time.time(),
                metrics={
                    'half_life': half_life,
                    'hurst_exponent': self.calculate_hurst_exponent(prices),
                    'adf_pvalue': adfuller(prices)[1]
                }
            )
        
        return None
    
    def generate_cointegration_signal(
        self,
        prices: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """
        Generate cointegration-based signal.
        
        Args:
            prices: Price DataFrame
            
        Returns:
            Trading signal or None
        """
        # Calculate correlation matrix
        correlations = prices.corr()
        
        # Find highly correlated pairs
        pairs = []
        for i in range(len(prices.columns)):
            for j in range(i+1, len(prices.columns)):
                if abs(correlations.iloc[i, j]) > self.min_correlation:
                    pairs.append((
                        prices.columns[i],
                        prices.columns[j]
                    ))
        
        # Analyze pairs
        signals = []
        for asset_a, asset_b in pairs:
            pair = self.analyze_pair(
                prices[asset_a],
                prices[asset_b]
            )
            if pair:
                signal = self.generate_pairs_signal(pair)
                if signal:
                    signals.append(signal)
        
        if signals:
            # Combine signals
            total_risk = sum(
                sum(abs(s) for s in signal.direction)
                for signal in signals
            )
            
            if total_risk > self.risk_limit:
                # Scale signals to meet risk limit
                scale = self.risk_limit / total_risk
                for signal in signals:
                    signal.direction = [
                        d * scale for d in signal.direction
                    ]
            
            # Return strongest signal
            return max(
                signals,
                key=lambda s: s.confidence
            )
        
        return None
    
    async def execute_trades(
        self,
        signal: TradingSignal
    ) -> bool:
        """
        Execute trades based on signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Whether execution was successful
        """
        try:
            # Execute trades on primary exchange
            exchange = self.exchange_clients[self.exchanges[0]]
            
            for asset, size in zip(signal.assets, signal.direction):
                if size != 0:
                    # Determine order side
                    side = 'buy' if size > 0 else 'sell'
                    
                    # Place order
                    order = await exchange.create_order(
                        symbol=asset,
                        side=side,
                        amount=abs(size)
                    )
                    
                    # Update positions
                    self.positions[asset] = self.positions.get(asset, 0) + size
            
            # Record signal
            self.signals.append(signal)
            
            return True
        
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        if not self.trades:
            return {}
        
        returns = [t['pnl'] for t in self.trades]
        
        return {
            'total_pnl': sum(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 else 0,
            'win_rate': np.mean([r > 0 for r in returns]),
            'num_trades': len(self.trades),
            'avg_trade_pnl': np.mean(returns),
            'max_drawdown': min(0, min(np.cumsum(returns))),
            'current_positions': self.positions
        }
    
    async def run(
        self,
        symbols: List[str],
        interval: float = 1.0
    ):
        """
        Run statistical arbitrage strategy.
        
        Args:
            symbols: List of trading symbols
            interval: Update interval in seconds
        """
        while True:
            try:
                # Fetch price history
                prices = await self.fetch_price_history(symbols)
                
                # Generate signals
                signals = []
                
                # Pairs trading signals
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        pair = self.analyze_pair(
                            prices[symbols[i]],
                            prices[symbols[j]]
                        )
                        if pair:
                            signal = self.generate_pairs_signal(pair)
                            if signal:
                                signals.append(signal)
                
                # Mean reversion signals
                for symbol in symbols:
                    signal = self.generate_mean_reversion_signal(
                        prices[symbol]
                    )
                    if signal:
                        signals.append(signal)
                
                # Cointegration signals
                signal = self.generate_cointegration_signal(prices)
                if signal:
                    signals.append(signal)
                
                # Execute strongest signal
                if signals:
                    best_signal = max(
                        signals,
                        key=lambda s: s.confidence
                    )
                    await self.execute_trades(best_signal)
                
                # Log performance
                metrics = self.get_performance_metrics()
                logger.info(f"Performance metrics: {metrics}")
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(interval)
