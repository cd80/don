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
        delta = prices.diff()
        gains = pd.Series(0.0, index=prices.index)
        losses = pd.Series(0.0, index=prices.index)
        gains[delta > 0] = delta[delta > 0]
        losses[delta < 0] = -delta[delta < 0]
        avg_gains = pd.Series(index=prices.index, dtype=float)
        avg_losses = pd.Series(index=prices.index, dtype=float)
        avg_gains.iloc[period] = gains.iloc[1:period+1].mean()
        avg_losses.iloc[period] = losses.iloc[1:period+1].mean()
        for i in range(period + 1, len(prices)):
            avg_gains.iloc[i] = ((avg_gains.iloc[i-1] * (period-1)) + gains.iloc[i]) / period
            avg_losses.iloc[i] = ((avg_losses.iloc[i-1] * (period-1)) + losses.iloc[i]) / period
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        rsi.iloc[:period] = np.nan
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
        """Calculate Volume Weighted Average Price (VWAP)."""
        typical_price = (high + low + close) / 3
        vwap = pd.Series(index=typical_price.index, dtype=float)
        for date in pd.unique(typical_price.index.date):
            mask = typical_price.index.date == date
            tp_vol = (typical_price[mask] * volume[mask]).cumsum()
            vol = volume[mask].cumsum()
            daily_vwap = tp_vol / vol
            vwap[mask] = np.minimum(np.maximum(daily_vwap, low[mask]), high[mask])
        return vwap

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series,
                            close: pd.Series, k_period: int = 14,
                            d_period: int = 3) -> dict:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()

        return {
            'k': k,
            'd': d
        }

    def _calculate_adx(self, high: pd.Series, low: pd.Series,
                      close: pd.Series, period: int = 14) -> dict:
        """Calculate Average Directional Index (ADX)."""
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
