from typing import Dict, Any, List, Tuple
import pandas as pd
from binance.client import Client
from binance.enums import *
from .base import DataCollector

class BinanceDataCollector(DataCollector):
    """Data collector implementation for Binance Futures."""

    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize Binance client.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.client = Client(api_key, api_secret)

    def collect_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Collect recent trades from Binance Futures."""
        trades = self.client.futures_recent_trades(symbol=symbol, limit=limit)
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        return df[['timestamp', 'price', 'qty', 'isBuyerMaker']].rename(
            columns={'qty': 'quantity', 'isBuyerMaker': 'is_buyer_maker'}
        )

    def collect_orderbook(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Collect orderbook data from Binance Futures."""
        depth = self.client.futures_order_book(symbol=symbol, limit=limit)
        bids = pd.DataFrame(depth['bids'], columns=['price', 'quantity'])
        asks = pd.DataFrame(depth['asks'], columns=['price', 'quantity'])
        bids['side'] = 'bid'
        asks['side'] = 'ask'
        return pd.concat([bids, asks], ignore_index=True)

    def collect_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Collect forced liquidations from Binance Futures."""
        liquidations = self.client.futures_liquidation_orders(symbol=symbol)
        if not liquidations:
            return pd.DataFrame(columns=['timestamp', 'price', 'quantity', 'side'])
        df = pd.DataFrame(liquidations)
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['side'] = df['side'].map({'BUY': 'long', 'SELL': 'short'})
        return df[['timestamp', 'price', 'quantity', 'side']]

    def collect_volume(self, symbol: str, interval: str = '1h') -> pd.DataFrame:
        """Collect volume data from Binance Futures."""
        klines = self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=500  # Default to last 500 intervals
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_base', 'taker_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp', 'volume', 'quote_volume']]
