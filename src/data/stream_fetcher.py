import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
import websockets
import pandas as pd
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import signal

class BinanceStreamFetcher:
    """
    Real-time data streaming from Binance WebSocket API with parallel processing
    and automatic reconnection capabilities.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        channels: List[str] = None,
        output_dir: str = "data/raw",
        buffer_size: int = 1000,
        max_workers: int = 4
    ):
        """
        Initialize the stream fetcher.
        
        Args:
            symbol: Trading pair symbol
            channels: List of channels to subscribe to (default: ['kline_5m', 'trade', 'depth'])
            output_dir: Directory to save streaming data
            buffer_size: Size of the data buffer
            max_workers: Maximum number of worker threads
        """
        self.symbol = symbol.lower()
        self.channels = channels or ['kline_5m', 'trade', 'depth']
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize data buffers
        self.kline_buffer = Queue(maxsize=buffer_size)
        self.trade_buffer = Queue(maxsize=buffer_size)
        self.depth_buffer = Queue(maxsize=buffer_size)
        
        # Initialize processing threads
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.ws = None
        
        # Setup data processors
        self.processors = {
            'kline': self._process_kline,
            'trade': self._process_trade,
            'depth': self._process_depth
        }
        
        # Initialize data storage
        self.current_kline = None
        self.order_book = pd.DataFrame(columns=['bid_quantity', 'ask_quantity'])
        self.recent_trades = pd.DataFrame()
        
        # Initialize last snapshot time
        self.last_snapshot_time = datetime.now()
        self.snapshot_interval = 10  # seconds
        
    async def _connect(self) -> None:
        """
        Establish WebSocket connection with automatic reconnection.
        """
        while self.running:
            try:
                # Construct WebSocket URL
                streams = [f"{self.symbol}@{channel}" for channel in self.channels]
                url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
                
                async with websockets.connect(url) as websocket:
                    self.ws = websocket
                    self.logger.info(f"Connected to Binance WebSocket: {url}")
                    
                    while self.running:
                        try:
                            message = await websocket.recv()
                            await self._handle_message(json.loads(message))
                        except websockets.ConnectionClosed:
                            self.logger.warning("WebSocket connection closed")
                            break
                        
            except Exception as e:
                self.logger.error(f"WebSocket error: {str(e)}")
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _handle_message(self, message: Dict) -> None:
        """
        Handle incoming WebSocket messages.
        
        Args:
            message: WebSocket message
        """
        try:
            # Extract message type
            event_type = message.get('e', '')
            
            # Route message to appropriate processor
            if event_type == 'kline':
                await self._queue_data(self.kline_buffer, message)
            elif event_type == 'trade':
                await self._queue_data(self.trade_buffer, message)
            elif event_type == 'depthUpdate':
                await self._queue_data(self.depth_buffer, message)
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
    
    async def _queue_data(
        self,
        queue: Queue,
        data: Dict
    ) -> None:
        """
        Queue data for processing.
        
        Args:
            queue: Target queue
            data: Data to queue
        """
        try:
            if not queue.full():
                queue.put(data)
            else:
                self.logger.warning(f"Queue full, dropping data: {data['e']}")
        except Exception as e:
            self.logger.error(f"Error queuing data: {str(e)}")
    
    def _process_kline(self, data: Dict) -> None:
        """
        Process kline data.
        
        Args:
            data: Kline data
        """
        try:
            kline = data['k']
            
            # Create DataFrame row
            row = pd.DataFrame([{
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'close_time': pd.to_datetime(kline['T'], unit='ms'),
                'quote_volume': float(kline['q']),
                'trades': int(kline['n']),
                'taker_buy_volume': float(kline['V']),
                'taker_buy_quote_volume': float(kline['Q'])
            }])
            
            # Update current kline
            self.current_kline = row
            
            # Save to file
            filename = f"{self.symbol}_klines_{datetime.now().strftime('%Y%m%d')}.parquet"
            filepath = os.path.join(self.output_dir, filename)
            
            if os.path.exists(filepath):
                row.to_parquet(filepath, append=True)
            else:
                row.to_parquet(filepath)
                
        except Exception as e:
            self.logger.error(f"Error processing kline: {str(e)}")
    
    def _process_trade(self, data: Dict) -> None:
        """
        Process trade data.
        
        Args:
            data: Trade data
        """
        try:
            # Create DataFrame row
            row = pd.DataFrame([{
                'timestamp': pd.to_datetime(data['T'], unit='ms'),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'buyer_maker': bool(data['m']),
                'trade_id': int(data['t'])
            }])
            
            # Update recent trades
            self.recent_trades = pd.concat([self.recent_trades, row]).tail(1000)
            
            # Save to file
            filename = f"{self.symbol}_trades_{datetime.now().strftime('%Y%m%d')}.parquet"
            filepath = os.path.join(self.output_dir, filename)
            
            if os.path.exists(filepath):
                row.to_parquet(filepath, append=True)
            else:
                row.to_parquet(filepath)
                
        except Exception as e:
            self.logger.error(f"Error processing trade: {str(e)}")
    
    def _process_depth(self, data: Dict, force_snapshot: bool = False) -> None:
        """
        Process order book data.
        
        Args:
            data: Order book data
            force_snapshot: Force saving a snapshot regardless of time
        """
        try:
            timestamp = pd.to_datetime(data['T'], unit='ms')
            
            # Update order book
            for bid in data.get('b', []):
                price, quantity = float(bid[0]), float(bid[1])
                if quantity > 0:
                    self.order_book.loc[price, 'bid_quantity'] = quantity
                else:
                    self.order_book.drop(price, inplace=True, errors='ignore')
                    
            for ask in data.get('a', []):
                price, quantity = float(ask[0]), float(ask[1])
                if quantity > 0:
                    self.order_book.loc[price, 'ask_quantity'] = quantity
                else:
                    self.order_book.drop(price, inplace=True, errors='ignore')
            
            # Sort order book
            self.order_book.sort_index(inplace=True)
            
            # Save snapshot if enough time has passed or forced
            current_time = datetime.now()
            if force_snapshot or (current_time - self.last_snapshot_time).total_seconds() >= self.snapshot_interval:
                filename = f"{self.symbol}_orderbook_{current_time.strftime('%Y%m%d')}.parquet"
                filepath = os.path.join(self.output_dir, filename)
                
                snapshot = self.order_book.copy()
                snapshot['timestamp'] = timestamp
                
                if os.path.exists(filepath):
                    snapshot.to_parquet(filepath, append=True)
                else:
                    snapshot.to_parquet(filepath)
                
                self.last_snapshot_time = current_time
                    
        except Exception as e:
            self.logger.error(f"Error processing depth: {str(e)}")
    
    def _process_queues(self) -> None:
        """
        Process queued data in separate threads.
        """
        while self.running:
            try:
                # Process kline data
                if not self.kline_buffer.empty():
                    data = self.kline_buffer.get()
                    self.executor.submit(self._process_kline, data)
                
                # Process trade data
                if not self.trade_buffer.empty():
                    data = self.trade_buffer.get()
                    self.executor.submit(self._process_trade, data)
                
                # Process depth data
                if not self.depth_buffer.empty():
                    data = self.depth_buffer.get()
                    self.executor.submit(self._process_depth, data)
                    
            except Exception as e:
                self.logger.error(f"Error processing queues: {str(e)}")
            
            # Small delay to prevent CPU overload
            time.sleep(0.001)
    
    async def start(self) -> None:
        """
        Start the data streaming process.
        """
        self.running = True
        
        # Start queue processor thread
        processor_thread = threading.Thread(target=self._process_queues)
        processor_thread.start()
        
        # Start WebSocket connection
        await self._connect()
    
    async def stop(self) -> None:
        """
        Stop the data streaming process.
        """
        self.running = False
        
        if self.ws:
            await self.ws.close()
        
        self.executor.shutdown(wait=True)
        
        self.logger.info("Streaming stopped")
    
    def get_current_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get current market data.
        
        Returns:
            Dictionary containing current kline, order book, and recent trades
        """
        return {
            'kline': self.current_kline,
            'order_book': self.order_book,
            'recent_trades': self.recent_trades
        }

async def main():
    """
    Example usage of BinanceStreamFetcher.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create fetcher instance
    fetcher = BinanceStreamFetcher(symbol="BTCUSDT")
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        asyncio.create_task(fetcher.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start streaming
        await fetcher.start()
    except KeyboardInterrupt:
        await fetcher.stop()

if __name__ == "__main__":
    asyncio.run(main())
