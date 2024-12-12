"""Mock implementations for Binance API testing."""
from typing import Optional

class MockBinanceDataCollector:
    """Mock implementation of BinanceDataCollector for testing."""

    def __init__(self, symbol: str, api_key: str, api_secret: str):
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_running = False

    def start(self) -> None:
        """Start data collection."""
        self.is_running = True

    def stop(self) -> None:
        """Stop data collection."""
        self.is_running = False

    def resume(self) -> None:
        """Resume data collection."""
        self.is_running = True
