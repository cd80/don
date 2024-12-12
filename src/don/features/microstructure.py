"""Market microstructure features for the Don trading framework.

This module implements various market microstructure features including:
- Order book imbalance
- Trade flow analysis
- Realized volatility
- Liquidation impact

These features provide deeper insight into market dynamics and liquidity.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import BaseFeatureCalculator


class MarketMicrostructureFeatures(BaseFeatureCalculator):
    """Calculate market microstructure features.

    This class implements various market microstructure features that provide
    insight into market dynamics, liquidity, and trading behavior.
    """

    def _calculate_order_imbalance(self, bids: pd.DataFrame, asks: pd.DataFrame) -> float:
        """Calculate order book imbalance using price-weighted volumes.

        Args:
            bids: DataFrame containing bid orders with 'price' and 'quantity' columns
            asks: DataFrame containing ask orders with 'price' and 'quantity' columns

        Returns:
            float: Order book imbalance ratio in range [-1, 1]
                  Positive values indicate more buying pressure
                  Negative values indicate more selling pressure
        """
        bid_volume = (bids['price'] * bids['quantity']).sum()
        ask_volume = (asks['price'] * asks['quantity']).sum()

        # Handle edge case where both volumes are 0
        if bid_volume == 0 and ask_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / (bid_volume + ask_volume)

    def _calculate_trade_flow(self, trades: pd.DataFrame, window: int = 100) -> pd.Series:
        """Calculate trade flow imbalance using rolling window.

        Args:
            trades: DataFrame containing trade data with columns:
                   - quantity: Trade size
                   - is_buyer_maker: Boolean indicating if buyer was maker
            window: Rolling window size for volume aggregation

        Returns:
            pd.Series: Trade flow imbalance in range [-1, 1]
                      Positive values indicate more aggressive buying
                      Negative values indicate more aggressive selling
        """
        # Handle empty trades DataFrame
        if trades.empty:
            return pd.Series(index=trades.index, dtype=float)

        # Calculate taker buy and sell volumes
        buy_volume = trades[~trades['is_buyer_maker']]['quantity'].rolling(window, min_periods=1).sum()
        sell_volume = trades[trades['is_buyer_maker']]['quantity'].rolling(window, min_periods=1).sum()

        # Handle edge case where both volumes are 0
        total_volume = buy_volume + sell_volume
        return pd.Series(
            np.where(total_volume > 0, (buy_volume - sell_volume) / total_volume, 0),
            index=trades.index
        )

    def _calculate_realized_volatility(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """Calculate realized volatility using rolling window of log returns.

        Args:
            prices: Series of asset prices
            window: Rolling window size for volatility calculation

        Returns:
            pd.Series: Annualized realized volatility
        """
        # Handle empty or single price point
        if len(prices) < 2:
            return pd.Series(index=prices.index, dtype=float)

        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        # Calculate annualized volatility (sqrt of window to annualize)
        return np.sqrt(log_returns.rolling(window, min_periods=2).var() * window)

    def _calculate_liquidation_impact(self, liquidations: pd.DataFrame, window: int = 100) -> Dict[str, pd.Series]:
        """Calculate liquidation impact features using rolling window.

        Args:
            liquidations: DataFrame containing liquidation data with columns:
                         - quantity: Liquidation size
                         - side: 'long' or 'short' indicating liquidation type
            window: Rolling window size for volume aggregation

        Returns:
            Dict[str, pd.Series]: Dictionary containing:
                - long_liquidation_volume: Rolling sum of long liquidations
                - short_liquidation_volume: Rolling sum of short liquidations
                - liquidation_imbalance: Normalized difference between long and short
        """
        # Handle empty liquidations DataFrame
        if liquidations.empty:
            empty_series = pd.Series(index=liquidations.index, dtype=float)
            return {
                'long_liquidation_volume': empty_series,
                'short_liquidation_volume': empty_series,
                'liquidation_imbalance': empty_series
            }

        # Calculate long and short liquidation volumes
        long_liq = liquidations[liquidations['side'] == 'long']['quantity'].rolling(
            window, min_periods=1).sum()
        short_liq = liquidations[liquidations['side'] == 'short']['quantity'].rolling(
            window, min_periods=1).sum()

        # Calculate imbalance
        total_liq = long_liq + short_liq
        imbalance = pd.Series(
            np.where(total_liq > 0, (long_liq - short_liq) / total_liq, 0),
            index=liquidations.index
        )

        return {
            'long_liquidation_volume': long_liq,
            'short_liquidation_volume': short_liq,
            'liquidation_imbalance': imbalance
        }

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features from input data.

        Args:
            data: DataFrame containing market data including:
                - orderbook snapshots (with 'side', 'price', 'quantity' columns)
                - trade data
                - liquidation events

        Returns:
            DataFrame with calculated market microstructure features
        """
        df = data.copy()

        # Initialize feature columns
        df['order_imbalance'] = np.nan
        df['trade_flow_imbalance'] = np.nan
        df['realized_volatility'] = np.nan
        df['long_liquidation_volume'] = np.nan
        df['short_liquidation_volume'] = np.nan
        df['liquidation_imbalance'] = np.nan

        # Calculate order book imbalance
        if all(col in df.columns for col in ['side', 'price', 'quantity']):
            grouped = df.groupby(df.index)
            for timestamp, group in grouped:
                bids = group[group['side'] == 'bid']
                asks = group[group['side'] == 'ask']
                df.at[timestamp, 'order_imbalance'] = self._calculate_order_imbalance(bids, asks)

        # Calculate trade flow imbalance
        if all(col in df.columns for col in ['quantity', 'is_buyer_maker']):
            df['trade_flow_imbalance'] = self._calculate_trade_flow(df)

        # Calculate realized volatility
        if 'close' in df.columns:
            df['realized_volatility'] = self._calculate_realized_volatility(df['close'])

        # Calculate liquidation impact features
        if all(col in df.columns for col in ['quantity', 'side']):
            liquidation_features = self._calculate_liquidation_impact(df)
            df['long_liquidation_volume'] = liquidation_features['long_liquidation_volume']
            df['short_liquidation_volume'] = liquidation_features['short_liquidation_volume']
            df['liquidation_imbalance'] = liquidation_features['liquidation_imbalance']

        return df
