import os
import sys
import unittest
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import websockets
import tempfile
import shutil
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.stream_fetcher import BinanceStreamFetcher

class TestBinanceStreamFetcher(unittest.TestCase):
    """Test cases for BinanceStreamFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize fetcher with test directory
        self.fetcher = BinanceStreamFetcher(
            symbol="BTCUSDT",
            output_dir=self.test_dir
        )
        
        # Sample data for testing
        self.sample_kline = {
            "e": "kline",
            "E": 1639483200000,
            "s": "BTCUSDT",
            "k": {
                "t": 1639483200000,
                "T": 1639483499999,
                "s": "BTCUSDT",
                "i": "5m",
                "f": 100,
                "L": 200,
                "o": "50000.00",
                "c": "51000.00",
                "h": "51500.00",
                "l": "49800.00",
                "v": "100.5",
                "n": 500,
                "x": False,
                "q": "5100000.00",
                "V": "60.5",
                "Q": "3060000.00",
                "B": "0"
            }
        }
        
        self.sample_trade = {
            "e": "trade",
            "E": 1639483200000,
            "s": "BTCUSDT",
            "t": 12345,
            "p": "50000.00",
            "q": "1.5",
            "b": 100,
            "a": 200,
            "T": 1639483200000,
            "m": True,
            "M": True
        }
        
        self.sample_depth = {
            "e": "depthUpdate",
            "E": 1639483200000,
            "s": "BTCUSDT",
            "U": 100,
            "u": 200,
            "b": [
                ["50000.00", "1.5"],
                ["49900.00", "2.0"]
            ],
            "a": [
                ["50100.00", "1.0"],
                ["50200.00", "2.5"]
            ],
            "T": 1639483200000
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test fetcher initialization."""
        self.assertEqual(self.fetcher.symbol, "btcusdt")
        self.assertEqual(self.fetcher.output_dir, self.test_dir)
        self.assertIsNotNone(self.fetcher.kline_buffer)
        self.assertIsNotNone(self.fetcher.trade_buffer)
        self.assertIsNotNone(self.fetcher.depth_buffer)
    
    async def test_connection(self):
        """Test WebSocket connection."""
        # Setup mock
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps(self.sample_kline)
        
        # Start fetcher
        self.fetcher.running = True
        try:
            # Run connection for a short time
            await asyncio.wait_for(self.fetcher._connect(), timeout=0.1)
        except asyncio.TimeoutError:
            pass
        
        # Verify connection was attempted
        self.assertTrue(mock_ws.recv.called)
    
    async def test_handle_kline_message(self):
        """Test kline message handling."""
        # Process sample kline
        await self.fetcher._handle_message(self.sample_kline)
        
        # Verify data was queued
        self.assertFalse(self.fetcher.kline_buffer.empty())
        
        # Process queued data
        data = self.fetcher.kline_buffer.get()
        self.fetcher._process_kline(data)
        
        # Verify processed data
        self.assertIsNotNone(self.fetcher.current_kline)
        self.assertEqual(
            self.fetcher.current_kline['close'].iloc[0],
            float(self.sample_kline['k']['c'])
        )
    
    async def test_handle_trade_message(self):
        """Test trade message handling."""
        # Process sample trade
        await self.fetcher._handle_message(self.sample_trade)
        
        # Verify data was queued
        self.assertFalse(self.fetcher.trade_buffer.empty())
        
        # Process queued data
        data = self.fetcher.trade_buffer.get()
        self.fetcher._process_trade(data)
        
        # Verify processed data
        self.assertFalse(self.fetcher.recent_trades.empty())
        self.assertEqual(
            self.fetcher.recent_trades['price'].iloc[-1],
            float(self.sample_trade['p'])
        )
    
    async def test_handle_depth_message(self):
        """Test depth message handling."""
        # Process sample depth
        await self.fetcher._handle_message(self.sample_depth)
        
        # Verify data was queued
        self.assertFalse(self.fetcher.depth_buffer.empty())
        
        # Process queued data
        data = self.fetcher.depth_buffer.get()
        self.fetcher._process_depth(data)
        
        # Verify processed data
        self.assertFalse(self.fetcher.order_book.empty())
        self.assertTrue(
            float(self.sample_depth['b'][0][0]) in self.fetcher.order_book.index
        )
    
    def test_get_current_data(self):
        """Test getting current market data."""
        # Get current data
        data = self.fetcher.get_current_data()
        
        # Verify data structure
        self.assertIn('kline', data)
        self.assertIn('order_book', data)
        self.assertIn('recent_trades', data)
    
    async def test_start_stop(self):
        """Test starting and stopping the fetcher."""
        # Setup mock
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps(self.sample_kline)
        
        # Start fetcher
        start_task = asyncio.create_task(self.fetcher.start())
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Stop fetcher
        await self.fetcher.stop()
        
        # Verify state
        self.assertFalse(self.fetcher.running)
        
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    def test_file_output(self):
        """Test data file output."""
        # Set current time to match sample data timestamp
        current_time = datetime.fromtimestamp(self.sample_depth['T'] / 1000)
        
        # Process some data
        self.fetcher._process_kline(self.sample_kline)
        self.fetcher._process_trade(self.sample_trade)
        
        # Process depth data with forced snapshot
        self.fetcher._process_depth(self.sample_depth, force_snapshot=True)
        
        # Wait briefly to ensure file writes complete
        time.sleep(0.1)
        
        # Check if files were created
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('btcusdt_klines_') for f in files))
        self.assertTrue(any(f.startswith('btcusdt_trades_') for f in files))
        self.assertTrue(any(f.startswith('btcusdt_orderbook_') for f in files))

def async_test(coro):
    """Decorator for async test methods."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

if __name__ == '__main__':
    unittest.main()
