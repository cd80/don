"""
Market Making Strategy Module for Bitcoin Trading RL.
Implements market making strategies to provide liquidity and profit from spreads.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.helpers import setup_logging
from src.data.binance_fetcher import BinanceFetcher

logger = setup_logging(__name__)

class MarketMakingStyle(Enum):
    """Market making strategy styles."""
    BASIC = 'basic'  # Simple spread-based market making
    ADAPTIVE = 'adaptive'  # Adjusts spreads based on volatility
    INVENTORY = 'inventory'  # Manages inventory risk
    PREDICTIVE = 'predictive'  # Uses price predictions

@dataclass
class OrderBook:
    """Container for orderbook state."""
    bids: Dict[float, float]  # price -> volume
    asks: Dict[float, float]  # price -> volume
    timestamp: float
    mid_price: float
    spread: float
    depth: Dict[str, float]

@dataclass
class MarketState:
    """Container for market state."""
    price: float
    volatility: float
    volume: float
    trend: float
    imbalance: float
    timestamp: float

class MarketMaker:
    """
    Market making strategy that provides liquidity and profits
    from bid-ask spreads.
    """
    
    def __init__(
        self,
        config: Dict,
        exchange: str,
        base_spread: float = 0.001,
        min_spread: float = 0.0005,
        max_spread: float = 0.01,
        order_size: float = 0.01,
        max_inventory: float = 1.0,
        risk_aversion: float = 1.0
    ):
        """
        Initialize market maker.
        
        Args:
            config: Strategy configuration
            exchange: Exchange to make markets on
            base_spread: Base spread as fraction of price
            min_spread: Minimum spread as fraction of price
            max_spread: Maximum spread as fraction of price
            order_size: Base order size
            max_inventory: Maximum allowed inventory
            risk_aversion: Risk aversion parameter
        """
        self.config = config
        self.exchange = exchange
        self.base_spread = base_spread
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.risk_aversion = risk_aversion
        
        # Initialize exchange client
        self.client = BinanceFetcher(config['data'][exchange])
        
        # Initialize state tracking
        self.current_orders = {}
        self.inventory = 0.0
        self.position_value = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Initialize market state tracking
        self.market_states = []
        self.order_books = []
        self.trades = []
        
        logger.info(
            f"Initialized market maker on {exchange} "
            f"with base spread {base_spread:.4f}"
        )
    
    async def fetch_market_state(
        self,
        symbol: str
    ) -> Tuple[OrderBook, MarketState]:
        """
        Fetch current market state.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (OrderBook, MarketState)
        """
        # Fetch orderbook
        book = await self.client.fetch_orderbook(symbol)
        
        # Calculate orderbook metrics
        bids = book['bids']
        asks = book['asks']
        
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid_price
        
        depth = {
            'bid_depth': sum(bids.values()),
            'ask_depth': sum(asks.values())
        }
        
        order_book = OrderBook(
            bids=bids,
            asks=asks,
            timestamp=time.time(),
            mid_price=mid_price,
            spread=spread,
            depth=depth
        )
        
        # Calculate market state
        recent_trades = await self.client.fetch_recent_trades(symbol)
        prices = [trade['price'] for trade in recent_trades]
        volumes = [trade['amount'] for trade in recent_trades]
        
        market_state = MarketState(
            price=mid_price,
            volatility=np.std(prices) / np.mean(prices),
            volume=np.mean(volumes),
            trend=self.calculate_trend(prices),
            imbalance=self.calculate_imbalance(order_book),
            timestamp=time.time()
        )
        
        return order_book, market_state
    
    def calculate_trend(self, prices: List[float]) -> float:
        """
        Calculate price trend.
        
        Args:
            prices: List of recent prices
            
        Returns:
            Trend indicator (-1 to 1)
        """
        if len(prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate exponentially weighted trend
        weights = np.exp(np.linspace(-1, 0, len(returns)))
        weights /= weights.sum()
        
        trend = np.sum(returns * weights)
        return np.clip(trend / 0.001, -1, 1)  # Normalize
    
    def calculate_imbalance(self, order_book: OrderBook) -> float:
        """
        Calculate order book imbalance.
        
        Args:
            order_book: Current order book
            
        Returns:
            Imbalance indicator (-1 to 1)
        """
        bid_depth = order_book.depth['bid_depth']
        ask_depth = order_book.depth['ask_depth']
        
        total_depth = bid_depth + ask_depth
        if total_depth == 0:
            return 0.0
        
        return (bid_depth - ask_depth) / total_depth
    
    def calculate_optimal_spread(
        self,
        market_state: MarketState
    ) -> float:
        """
        Calculate optimal spread based on market conditions.
        
        Args:
            market_state: Current market state
            
        Returns:
            Optimal spread as fraction of price
        """
        # Base spread
        spread = self.base_spread
        
        # Adjust for volatility
        spread *= (1 + market_state.volatility * 10)
        
        # Adjust for volume
        volume_factor = market_state.volume / self.order_size
        spread *= np.clip(1 / np.sqrt(volume_factor), 0.5, 2.0)
        
        # Adjust for trend
        trend_impact = abs(market_state.trend) * 0.5
        spread *= (1 + trend_impact)
        
        # Adjust for inventory
        inventory_factor = abs(self.inventory) / self.max_inventory
        spread *= (1 + inventory_factor * self.risk_aversion)
        
        return np.clip(spread, self.min_spread, self.max_spread)
    
    def calculate_order_prices(
        self,
        market_state: MarketState
    ) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask prices.
        
        Args:
            market_state: Current market state
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Calculate spread
        spread = self.calculate_optimal_spread(market_state)
        
        # Calculate base prices
        mid_price = market_state.price
        half_spread = spread / 2
        
        bid_price = mid_price * (1 - half_spread)
        ask_price = mid_price * (1 + half_spread)
        
        # Adjust for inventory
        inventory_skew = self.inventory / self.max_inventory
        price_skew = mid_price * inventory_skew * 0.1
        
        bid_price -= price_skew
        ask_price -= price_skew
        
        return bid_price, ask_price
    
    def calculate_order_sizes(
        self,
        market_state: MarketState
    ) -> Tuple[float, float]:
        """
        Calculate optimal order sizes.
        
        Args:
            market_state: Current market state
            
        Returns:
            Tuple of (bid_size, ask_size)
        """
        # Base size
        base_size = self.order_size
        
        # Adjust for inventory
        inventory_factor = self.inventory / self.max_inventory
        
        # Increase opposite side
        bid_size = base_size * (1 + max(0, inventory_factor))
        ask_size = base_size * (1 - min(0, inventory_factor))
        
        # Adjust for volume
        volume_factor = market_state.volume / base_size
        bid_size *= np.clip(volume_factor, 0.5, 2.0)
        ask_size *= np.clip(volume_factor, 0.5, 2.0)
        
        return bid_size, ask_size
    
    async def place_orders(
        self,
        symbol: str,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float
    ) -> bool:
        """
        Place market making orders.
        
        Args:
            symbol: Trading symbol
            bid_price: Bid price
            ask_price: Ask price
            bid_size: Bid size
            ask_size: Ask size
            
        Returns:
            Whether orders were placed successfully
        """
        try:
            # Cancel existing orders
            await self.cancel_all_orders(symbol)
            
            # Place new orders
            bid_order = await self.client.create_order(
                symbol=symbol,
                side='buy',
                type='limit',
                price=bid_price,
                amount=bid_size
            )
            
            ask_order = await self.client.create_order(
                symbol=symbol,
                side='sell',
                type='limit',
                price=ask_price,
                amount=ask_size
            )
            
            # Track orders
            self.current_orders = {
                'bid': bid_order,
                'ask': ask_order
            }
            
            logger.info(
                f"Placed orders: Bid {bid_size:.4f} @ {bid_price:.2f}, "
                f"Ask {ask_size:.4f} @ {ask_price:.2f}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to place orders: {str(e)}")
            return False
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all open orders.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Whether cancellation was successful
        """
        try:
            for order in self.current_orders.values():
                await self.client.cancel_order(
                    symbol=symbol,
                    order_id=order['id']
                )
            
            self.current_orders = {}
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel orders: {str(e)}")
            return False
    
    def update_inventory(self, trade: Dict) -> None:
        """
        Update inventory after trade.
        
        Args:
            trade: Trade information
        """
        # Update inventory
        if trade['side'] == 'buy':
            self.inventory += trade['amount']
        else:
            self.inventory -= trade['amount']
        
        # Update position value
        self.position_value = self.inventory * trade['price']
        
        # Calculate P&L
        if trade['order_id'] in [o['id'] for o in self.current_orders.values()]:
            self.realized_pnl += trade['realized_pnl']
        
        self.unrealized_pnl = (
            self.position_value - self.inventory * trade['price']
        )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        return {
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'inventory': self.inventory,
            'position_value': self.position_value,
            'num_trades': len(self.trades),
            'avg_spread': np.mean([
                ob.spread for ob in self.order_books[-100:]
            ]) if self.order_books else 0.0,
            'avg_volume': np.mean([
                ms.volume for ms in self.market_states[-100:]
            ]) if self.market_states else 0.0
        }
    
    async def run(
        self,
        symbol: str,
        interval: float = 1.0
    ):
        """
        Run market making strategy.
        
        Args:
            symbol: Trading symbol
            interval: Update interval in seconds
        """
        while True:
            try:
                # Fetch market state
                order_book, market_state = await self.fetch_market_state(
                    symbol
                )
                
                # Store state
                self.order_books.append(order_book)
                self.market_states.append(market_state)
                
                # Calculate orders
                bid_price, ask_price = self.calculate_order_prices(
                    market_state
                )
                bid_size, ask_size = self.calculate_order_sizes(
                    market_state
                )
                
                # Place orders
                await self.place_orders(
                    symbol,
                    bid_price,
                    ask_price,
                    bid_size,
                    ask_size
                )
                
                # Log performance
                metrics = self.get_performance_metrics()
                logger.info(f"Performance metrics: {metrics}")
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in market making loop: {str(e)}")
                await asyncio.sleep(interval)
