from typing import Optional
import pandas as pd
import numpy as np
from .base import BaseFeatureCalculator

class TechnicalIndicators(BaseFeatureCalculator):
    """Technical indicator calculator for financial data."""

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators.

        Args:
            data: DataFrame with OHLCV data (open, high, low, close, volume columns)

        Returns:
            DataFrame with additional columns for technical indicators
        """
        df = data.copy()

        # Basic indicators
        df['sma_20'] = self._calculate_sma(df['close'], window=20)
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        macd_data = self._calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        bollinger = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bollinger['upper']
        df['bb_middle'] = bollinger['middle']
        df['bb_lower'] = bollinger['lower']

        return df

    def _calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=window).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> dict:
        """Calculate Moving Average Convergence Divergence."""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }

    def _calculate_bollinger_bands(self, prices: pd.Series,
                                 window: int = 20,
                                 num_std: float = 2.0) -> dict:
        """Calculate Bollinger Bands."""
        middle = self._calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
