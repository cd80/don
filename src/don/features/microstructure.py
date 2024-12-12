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

    def _calculate_order_imbalance(self, bids: pd.DataFrame, asks: pd.DataFrame) -> pd.Series:
        """Calculate order book imbalance using price-weighted volumes.

        Args:
            bids: DataFrame containing bid orders with 'price' and 'quantity' columns
            asks: DataFrame containing ask orders with 'price' and 'quantity' columns

        Returns:
            pd.Series: Order book imbalance ratio in range [-1, 1]
                      Positive values indicate more buying pressure
                      Negative values indicate more selling pressure
        """
        if bids.empty and asks.empty:
            return pd.Series(0.0, index=bids.index.get_level_values(0).unique() if not bids.empty else asks.index.get_level_values(0).unique())

        # Get unique timestamps
        timestamps = pd.Index(set(bids.index.get_level_values(0).union(asks.index.get_level_values(0))))

        # Initialize result series
        imbalance = pd.Series(0.0, index=timestamps)

        # Calculate price-weighted volumes for each timestamp
        for ts in timestamps:
            ts_bids = bids.loc[ts] if ts in bids.index.get_level_values(0) else pd.DataFrame()
            ts_asks = asks.loc[ts] if ts in asks.index.get_level_values(0) else pd.DataFrame()

            bid_volume = (ts_bids['price'] * ts_bids['quantity']).sum() if not ts_bids.empty else 0
            ask_volume = (ts_asks['price'] * ts_asks['quantity']).sum() if not ts_asks.empty else 0

            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                imbalance[ts] = (bid_volume - ask_volume) / total_volume

        return imbalance

    def _calculate_trade_flow(self, df: pd.DataFrame, window: int = 100) -> pd.Series:
        """Calculate trade flow imbalance using rolling window.

        Args:
            df: DataFrame containing trade data with columns:
                - quantity: Trade size
                - is_buyer_maker: True if buyer was maker
            window: Rolling window size for volume aggregation

        Returns:
            pd.Series: Trade flow imbalance ratio in range [-1, 1]
                      Positive values indicate more aggressive buying
                      Negative values indicate more aggressive selling
        """
        # Get unique timestamps
        timestamps = df.index.get_level_values(0).unique()

        # Initialize result series
        imbalance = pd.Series(0.0, index=timestamps)

        # Calculate maker/taker volumes for each timestamp
        for ts in timestamps:
            ts_data = df.loc[ts]

            # Calculate volumes for this timestamp
            maker_mask = ts_data['is_buyer_maker']
            taker_mask = ~ts_data['is_buyer_maker']

            maker_volume = ts_data.loc[maker_mask, 'quantity'].sum() if any(maker_mask) else 0
            taker_volume = ts_data.loc[taker_mask, 'quantity'].sum() if any(taker_mask) else 0

            total_volume = maker_volume + taker_volume
            if total_volume > 0:
                imbalance[ts] = (maker_volume - taker_volume) / total_volume

        # Apply rolling window
        return imbalance.rolling(window=window, min_periods=1).mean()

    def _calculate_realized_volatility(self, prices: pd.Series, window: int = 100) -> pd.Series:
        """Calculate realized volatility using rolling window of log returns.

        Args:
            prices: Series of close prices
            window: Rolling window size for volatility calculation

        Returns:
            pd.Series: Realized volatility values
        """
        # Get unique timestamps and last price for each timestamp
        if isinstance(prices.index, pd.MultiIndex):
            # For multi-index, get the last price for each timestamp
            timestamps = prices.index.get_level_values(0).unique()
            unique_prices = prices.groupby(level=0).last()
        else:
            timestamps = prices.index
            unique_prices = prices

        # Handle empty or insufficient data
        if len(unique_prices) < 2:
            return pd.Series(0.0, index=timestamps)

        # Calculate log returns
        log_returns = np.log(unique_prices / unique_prices.shift(1))
        log_returns = log_returns.fillna(0)  # Fill NaN from first observation

        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=window, min_periods=1).std()
        volatility = volatility.fillna(0)  # Fill any remaining NaN values

        return volatility

    def _calculate_liquidation_impact(
        self, liquidations: pd.DataFrame, window: int = 100
    ) -> dict[str, pd.Series]:
        """Calculate liquidation impact features.

        Args:
            liquidations: DataFrame with liquidation data containing 'quantity' and 'side' columns
            window: Rolling window size for volume calculations

        Returns:
            dict: Dictionary containing:
                - long_liquidation_volume: Rolling sum of long liquidation volumes
                - short_liquidation_volume: Rolling sum of short liquidation volumes
                - liquidation_imbalance: Normalized difference between long and short volumes
        """
        # Get unique timestamps
        timestamps = liquidations.index.get_level_values(0).unique()

        # Initialize result series
        long_volumes = pd.Series(0.0, index=timestamps)
        short_volumes = pd.Series(0.0, index=timestamps)

        # Calculate volumes for each timestamp
        for ts in timestamps:
            ts_data = liquidations.loc[ts]

            # Sum liquidation volumes by side
            long_mask = ts_data['side'] == 'long'
            short_mask = ts_data['side'] == 'short'

            long_volumes[ts] = ts_data.loc[long_mask, 'quantity'].sum() if any(long_mask) else 0
            short_volumes[ts] = ts_data.loc[short_mask, 'quantity'].sum() if any(short_mask) else 0

        # Calculate rolling sums
        long_rolling = long_volumes.rolling(window=window, min_periods=1).sum()
        short_rolling = short_volumes.rolling(window=window, min_periods=1).sum()

        # Calculate imbalance
        total_volume = long_rolling + short_rolling
        liquidation_imbalance = pd.Series(0.0, index=timestamps)
        nonzero_mask = total_volume > 0
        liquidation_imbalance[nonzero_mask] = (long_rolling - short_rolling) / total_volume

        return {
            'long_liquidation_volume': long_rolling,
            'short_liquidation_volume': short_rolling,
            'liquidation_imbalance': liquidation_imbalance
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all market microstructure features.

        Args:
            df: DataFrame containing market data with columns:
                - price, quantity: For order book data
                - is_buyer_maker: For trade flow analysis
                - close: For volatility calculation
                - side: For liquidation analysis

        Returns:
            pd.DataFrame: DataFrame with calculated features
        """
        # Get unique timestamps for result index
        timestamps = df.index.get_level_values(0).unique()
        result = pd.DataFrame(index=timestamps)

        # Calculate order book imbalance if we have order book data
        if all(col in df.columns for col in ['price', 'quantity', 'side']):
            bids = df[df['side'] == 'bid'][['price', 'quantity']]
            asks = df[df['side'] == 'ask'][['price', 'quantity']]
            result['order_imbalance'] = self._calculate_order_imbalance(bids, asks)

        # Calculate trade flow if we have trade data
        if 'is_buyer_maker' in df.columns and 'quantity' in df.columns:
            result['trade_flow_imbalance'] = self._calculate_trade_flow(df)

        # Calculate volatility if we have price data
        if 'close' in df.columns:
            result['realized_volatility'] = self._calculate_realized_volatility(df['close'])

        # Calculate liquidation features if we have liquidation data
        if 'side' in df.columns and df['side'].isin(['long', 'short']).any():
            liq_features = self._calculate_liquidation_impact(df)
            result['long_liquidation_volume'] = liq_features['long_liquidation_volume']
            result['short_liquidation_volume'] = liq_features['short_liquidation_volume']
            result['liquidation_imbalance'] = liq_features['liquidation_imbalance']
        else:
            # Add empty liquidation features with proper index
            result['long_liquidation_volume'] = pd.Series(0.0, index=result.index)
            result['short_liquidation_volume'] = pd.Series(0.0, index=result.index)
            result['liquidation_imbalance'] = pd.Series(0.0, index=result.index)

        return result
