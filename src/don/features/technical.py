from typing import Optional
import pandas as pd
import numpy as np
from .base import BaseFeatureCalculator

class TechnicalIndicators(BaseFeatureCalculator):
    """Technical indicator calculator for financial data."""

    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and store in database.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with all calculated indicators
        """
        return self.calculate(data)

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
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        macd_data = self._calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']
        bollinger = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bollinger['upper']
        df['bb_middle'] = bollinger['middle']
        df['bb_lower'] = bollinger['lower']

        # Volume indicators
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        df['vwap'] = self._calculate_vwap(df['high'], df['low'], df['close'], df['volume'])

        # Momentum indicators
        stoch = self._calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['k']
        df['stoch_d'] = stoch['d']
        adx_data = self._calculate_adx(df['high'], df['low'], df['close'])
        df['adx'] = adx_data['adx']
        df['plus_di'] = adx_data['plus_di']
        df['minus_di'] = adx_data['minus_di']

        return df

    def _calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=window).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index using Wilder's smoothing method."""
        if len(prices) < period + 1:
            return pd.Series(np.nan, index=prices.index)

        # Calculate price changes
        delta = prices.diff()
        gains = delta.copy()
        losses = delta.copy()

        # Handle small price changes to avoid numerical instability
        epsilon = 1e-10  # Small threshold for price changes
        gains = gains.where(gains > epsilon, 0)
        losses = losses.where(losses < -epsilon, 0)
        losses = -losses  # Make losses positive

        # Calculate initial averages
        avg_gain = gains.iloc[1:period+1].mean()
        avg_loss = losses.iloc[1:period+1].mean()

        # Initialize RSI series with NaN
        rsi = pd.Series(np.nan, index=prices.index)

        # Calculate initial RSI
        if avg_loss < epsilon:
            rsi.iloc[period] = 100  # All gains, no losses
        elif avg_gain < epsilon:
            rsi.iloc[period] = 0    # All losses, no gains
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[period] = 100 - (100 / (1 + rs))

        # Calculate RSI for remaining periods using Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain = ((avg_gain * (period - 1)) + gains.iloc[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses.iloc[i]) / period

            if avg_loss < epsilon:
                rsi.iloc[i] = 100  # All gains, no losses
            elif avg_gain < epsilon:
                rsi.iloc[i] = 0    # All losses, no gains
            else:
                rs = avg_gain / avg_loss
                rsi.iloc[i] = 100 - (100 / (1 + rs))

        return rsi

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

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (OBV)."""
        close_diff = close.diff()
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calculate_vwap(self, high: pd.Series, low: pd.Series,
                       close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP).

        VWAP is calculated as the cumulative sum of price * volume divided by cumulative volume.
        This is the standard VWAP calculation method used in financial markets.
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series,
                            close: pd.Series, k_period: int = 14,
                            d_period: int = 3) -> dict:
        """Calculate Stochastic Oscillator."""
        if len(high) < k_period:
            empty = pd.Series(np.nan, index=high.index)
            return {'k': empty, 'd': empty}

        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()

        k = pd.Series(np.nan, index=high.index)
        valid_range = (highest_high - lowest_low) > 0
        k[valid_range] = 100 * ((close - lowest_low) / (highest_high - lowest_low))[valid_range]
        d = k.rolling(window=d_period, min_periods=d_period).mean()

        return {
            'k': k,
            'd': d
        }

    def _calculate_adx(self, high: pd.Series, low: pd.Series,
                      close: pd.Series, period: int = 14) -> dict:
        """Calculate Average Directional Index (ADX)."""
        if len(high) < period + 1:
            empty = pd.Series(np.nan, index=high.index)
            return {'adx': empty, 'plus_di': empty, 'minus_di': empty}

        # Calculate True Range
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period).mean()

        # Calculate DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = pd.Series(0, index=high.index)
        minus_dm = pd.Series(0, index=high.index)

        plus_dm[((up_move > down_move) & (up_move > 0))] = up_move
        minus_dm[((down_move > up_move) & (down_move > 0))] = down_move

        # Calculate DI
        plus_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, min_periods=period).mean()

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
