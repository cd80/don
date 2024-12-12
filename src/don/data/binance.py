from typing import Dict, Any, List, Tuple, Callable
import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.streams import BinanceSocketManager
from .base import DataCollector

class BinanceDataCollector(DataCollector):
    """Data collector implementation for Binance Futures."""

    def __init__(self, symbol: str, api_key: str = None, api_secret: str = None):
        """Initialize Binance client.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.client = Client(api_key, api_secret)
        self.symbol = symbol.upper()
        self.bsm = BinanceSocketManager(self.client)
        self._trade_callbacks: List[Callable] = []

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

    def start_trade_stream(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Start streaming trade data.

        Args:
            callback: Function to handle incoming trade data
        """
        self._trade_callbacks.append(callback)
        self.bsm.start_trade_socket(self.symbol, self._handle_trade_socket)

    def _handle_trade_socket(self, msg: Dict[str, Any]) -> None:
        """Handle incoming trade socket messages.

        Args:
            msg: Trade message from websocket
        """
        for callback in self._trade_callbacks:
            callback(msg)
