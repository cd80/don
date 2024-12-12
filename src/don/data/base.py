from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class DataCollector(ABC):
    """Base class for data collection from cryptocurrency exchanges."""

    @abstractmethod
    def collect_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Collect trade data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of trades to collect

        Returns:
            DataFrame with columns: timestamp, price, quantity, is_buyer_maker
        """
        pass

    @abstractmethod
    def collect_orderbook(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Collect orderbook data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Depth of orderbook to collect

        Returns:
            DataFrame with columns: price, quantity, side (bid/ask)
        """
        pass

    @abstractmethod
    def collect_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Collect liquidation data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of liquidations to collect

        Returns:
            DataFrame with columns: timestamp, price, quantity, side (long/short)
        """
        pass

    @abstractmethod
    def collect_volume(self, symbol: str, interval: str = '1h') -> pd.DataFrame:
        """Collect volume data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '4h', '1d')

        Returns:
            DataFrame with columns: timestamp, volume, quote_volume
        """
        pass
