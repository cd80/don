"""
High-Frequency Trading Module for Bitcoin Trading RL.
Implements low-latency trading strategies with efficient signal generation and order execution.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import numpy as np
import pandas as pd
from collections import deque

from src.utils.helpers import setup_logging
from src.data.binance_fetcher import BinanceFetcher

logger = setup_logging(__name__)

class SignalType(Enum):
    """Types of trading signals."""
    MOMENTUM = 'momentum'  # Price momentum signals
    MEAN_REVERSION = 'mean_reversion'  # Mean reversion signals
    ORDER_FLOW = 'order_flow'  # Order flow imbalance signals
    TECHNICAL = 'technical'  # Technical analysis signals
    MICROSTRUCTURE = 'microstructure'  # Market microstructure signals

@dataclass
class MarketSignal:
    """Container for market signals."""
    type: SignalType
    direction: float  # -1 to 1
    strength: float  # 0 to 1
    timestamp: float
    features: Dict[str, float]
    confidence: float
    horizon: float

@dataclass
class ExecutionMetrics:
    """Container for execution metrics."""
    latency: float
    slippage: float
    fill_rate: float
    cost: float
    impact: float
    timing_cost: float
    opportunity_cost: float

class HighFrequencyTrader:
    """
    High-frequency trading strategy that implements low-latency
    trading with efficient signal generation and order execution.
    """
    
    def __init__(
        self,
        config: Dict,
        exchange: str,
        max_position: float = 1.0,
        risk_limit: float = 0.02,
        signal_threshold: float = 0.5,
        execution_timeout: float = 0.1,
        buffer_size: int = 1000
    ):
        """
        Initialize high-frequency trader.
        
        Args:
            config: Strategy configuration
            exchange: Exchange to trade on
            max_position: Maximum position size
            risk_limit: Maximum risk limit as fraction of capital
            signal_threshold: Minimum signal strength threshold
            execution_timeout: Maximum execution timeout in seconds
            buffer_size: Size of market data buffer
        """
        self.config = config
        self.exchange = exchange
        self.max_position = max_position
        self.risk_limit = risk_limit
        self.signal_threshold = signal_threshold
        self.execution_timeout = execution_timeout
        
        # Initialize exchange client
        self.client = BinanceFetcher(config['data'][exchange])
        
        # Initialize data buffers
        self.price_buffer = deque(maxlen=buffer_size)
        self.volume_buffer = deque(maxlen=buffer_size)
        self.order_flow_buffer = deque(maxlen=buffer_size)
        self.signal_buffer = deque(maxlen=buffer_size)
        
        # Initialize state tracking
        self.position = 0.0
        self.pending_orders = {}
        self.filled_orders = []
        self.signals = []
        self.metrics = []
        
        logger.info(
            f"Initialized high-frequency trader on {exchange} "
            f"with {buffer_size} buffer size"
        )
    
    async def update_market_data(
        self,
        symbol: str
    ) -> Tuple[float, float, Dict]:
        """
        Update market data buffers.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (price, volume, order_flow)
        """
        # Fetch latest data
        trades = await self.client.fetch_recent_trades(symbol)
        order_book = await self.client.fetch_orderbook(symbol)
        
        # Update price and volume buffers
        price = float(trades[-1]['price'])
        volume = float(trades[-1]['amount'])
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        # Calculate order flow imbalance
        bid_volume = sum(order_book['bids'].values())
        ask_volume = sum(order_book['asks'].values())
        order_flow = {
            'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume),
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'spread': min(order_book['asks'].keys()) - max(order_book['bids'].keys())
        }
        self.order_flow_buffer.append(order_flow)
        
        return price, volume, order_flow
    
    def generate_momentum_signal(self) -> Optional[MarketSignal]:
        """Generate momentum-based signal."""
        if len(self.price_buffer) < 2:
            return None
        
        # Calculate returns
        returns = np.diff(list(self.price_buffer)[-20:]) / list(self.price_buffer)[-21:-1]
        
        # Calculate momentum features
        features = {
            'short_momentum': returns[-5:].mean(),
            'medium_momentum': returns[-10:].mean(),
            'long_momentum': returns.mean(),
            'volatility': returns.std()
        }
        
        # Calculate signal
        signal = np.sign(features['short_momentum']) * min(
            abs(features['short_momentum'] / features['volatility']),
            1.0
        )
        
        # Calculate confidence
        confidence = min(
            1.0,
            abs(features['short_momentum']) / (features['volatility'] * 2)
        )
        
        if abs(signal) > self.signal_threshold:
            return MarketSignal(
                type=SignalType.MOMENTUM,
                direction=signal,
                strength=abs(signal),
                timestamp=time.time(),
                features=features,
                confidence=confidence,
                horizon=5.0  # 5 second horizon
            )
        
        return None
    
    def generate_mean_reversion_signal(self) -> Optional[MarketSignal]:
        """Generate mean reversion signal."""
        if len(self.price_buffer) < 50:
            return None
        
        prices = np.array(list(self.price_buffer))
        
        # Calculate features
        moving_avg = prices[-20:].mean()
        std_dev = prices[-20:].std()
        z_score = (prices[-1] - moving_avg) / std_dev
        
        features = {
            'z_score': z_score,
            'moving_avg': moving_avg,
            'std_dev': std_dev,
            'price_gap': (prices[-1] - moving_avg) / moving_avg
        }
        
        # Calculate signal
        signal = -np.sign(z_score) * min(abs(z_score) / 2, 1.0)
        
        # Calculate confidence
        confidence = min(abs(z_score) / 3, 1.0)
        
        if abs(signal) > self.signal_threshold:
            return MarketSignal(
                type=SignalType.MEAN_REVERSION,
                direction=signal,
                strength=abs(signal),
                timestamp=time.time(),
                features=features,
                confidence=confidence,
                horizon=10.0  # 10 second horizon
            )
        
        return None
    
    def generate_order_flow_signal(self) -> Optional[MarketSignal]:
        """Generate order flow based signal."""
        if len(self.order_flow_buffer) < 10:
            return None
        
        # Calculate features
        imbalances = [flow['imbalance'] for flow in self.order_flow_buffer]
        spreads = [flow['spread'] for flow in self.order_flow_buffer]
        
        features = {
            'imbalance': imbalances[-1],
            'imbalance_ma': np.mean(imbalances[-10:]),
            'spread': spreads[-1],
            'spread_ma': np.mean(spreads[-10:])
        }
        
        # Calculate signal
        signal = np.sign(features['imbalance']) * min(
            abs(features['imbalance']) * 2,
            1.0
        )
        
        # Adjust for spread
        spread_factor = min(
            features['spread_ma'] / features['spread'],
            1.0
        )
        signal *= spread_factor
        
        # Calculate confidence
        confidence = min(
            abs(features['imbalance']) * spread_factor,
            1.0
        )
        
        if abs(signal) > self.signal_threshold:
            return MarketSignal(
                type=SignalType.ORDER_FLOW,
                direction=signal,
                strength=abs(signal),
                timestamp=time.time(),
                features=features,
                confidence=confidence,
                horizon=1.0  # 1 second horizon
            )
        
        return None
    
    def combine_signals(
        self,
        signals: List[MarketSignal]
    ) -> Optional[MarketSignal]:
        """
        Combine multiple signals into single trading signal.
        
        Args:
            signals: List of trading signals
            
        Returns:
            Combined signal or None
        """
        if not signals:
            return None
        
        # Weight signals by confidence and recency
        total_weight = 0
        weighted_direction = 0
        
        for signal in signals:
            # Calculate weight based on confidence and time decay
            time_factor = np.exp(
                -(time.time() - signal.timestamp) / signal.horizon
            )
            weight = signal.confidence * time_factor
            
            weighted_direction += signal.direction * weight
            total_weight += weight
        
        if total_weight > 0:
            # Calculate combined signal
            direction = weighted_direction / total_weight
            strength = min(abs(direction), 1.0)
            
            if strength > self.signal_threshold:
                return MarketSignal(
                    type=SignalType.TECHNICAL,
                    direction=np.sign(direction),
                    strength=strength,
                    timestamp=time.time(),
                    features={
                        'num_signals': len(signals),
                        'total_weight': total_weight
                    },
                    confidence=strength,
                    horizon=min(s.horizon for s in signals)
                )
        
        return None
    
    def calculate_position_size(
        self,
        signal: MarketSignal,
        price: float
    ) -> float:
        """
        Calculate optimal position size.
        
        Args:
            signal: Trading signal
            price: Current price
            
        Returns:
            Position size (positive for long, negative for short)
        """
        # Base size on signal strength
        size = self.max_position * signal.strength
        
        # Adjust for current position
        size -= self.position
        
        # Apply risk limit
        risk = abs(size) * price * signal.confidence
        if risk > self.risk_limit:
            size *= self.risk_limit / risk
        
        return size
    
    async def execute_order(
        self,
        symbol: str,
        size: float,
        price: float
    ) -> Tuple[bool, ExecutionMetrics]:
        """
        Execute order with smart order routing.
        
        Args:
            symbol: Trading symbol
            size: Order size (positive for buy, negative for sell)
            price: Current price
            
        Returns:
            Tuple of (success, metrics)
        """
        start_time = time.time()
        
        try:
            # Place order
            side = 'buy' if size > 0 else 'sell'
            order = await self.client.create_order(
                symbol=symbol,
                side=side,
                amount=abs(size),
                price=price
            )
            
            # Track order
            self.pending_orders[order['id']] = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'timestamp': start_time
            }
            
            # Wait for fill
            filled = await self.wait_for_fill(order['id'])
            
            if filled:
                # Update position
                self.position += size
                
                # Calculate metrics
                latency = time.time() - start_time
                executed_price = float(order['price'])
                slippage = (executed_price - price) / price
                
                metrics = ExecutionMetrics(
                    latency=latency,
                    slippage=slippage,
                    fill_rate=1.0,
                    cost=abs(size) * executed_price * 0.001,  # 0.1% fee
                    impact=slippage * abs(size),
                    timing_cost=latency * abs(size) * 0.0001,  # Estimated timing cost
                    opportunity_cost=0.0  # Calculate based on price movement
                )
                
                return True, metrics
            
            return False, ExecutionMetrics(
                latency=self.execution_timeout,
                slippage=0.0,
                fill_rate=0.0,
                cost=0.0,
                impact=0.0,
                timing_cost=self.execution_timeout * abs(size) * 0.0001,
                opportunity_cost=0.0
            )
        
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}")
            return False, ExecutionMetrics(
                latency=time.time() - start_time,
                slippage=0.0,
                fill_rate=0.0,
                cost=0.0,
                impact=0.0,
                timing_cost=0.0,
                opportunity_cost=0.0
            )
    
    async def wait_for_fill(self, order_id: str) -> bool:
        """
        Wait for order fill with timeout.
        
        Args:
            order_id: Order ID to wait for
            
        Returns:
            Whether order was filled
        """
        start_time = time.time()
        
        while time.time() - start_time < self.execution_timeout:
            try:
                order = await self.client.get_order(
                    symbol=self.pending_orders[order_id]['symbol'],
                    order_id=order_id
                )
                
                if order['status'] == 'filled':
                    self.filled_orders.append(order)
                    del self.pending_orders[order_id]
                    return True
                
                elif order['status'] == 'canceled':
                    del self.pending_orders[order_id]
                    return False
                
                await asyncio.sleep(0.01)  # 10ms check interval
            
            except Exception as e:
                logger.error(f"Error checking order: {str(e)}")
                return False
        
        # Cancel order on timeout
        try:
            await self.client.cancel_order(
                symbol=self.pending_orders[order_id]['symbol'],
                order_id=order_id
            )
        except:
            pass
        
        del self.pending_orders[order_id]
        return False
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        if not self.metrics:
            return {}
        
        latencies = [m.latency for m in self.metrics]
        slippages = [m.slippage for m in self.metrics]
        fill_rates = [m.fill_rate for m in self.metrics]
        costs = [m.cost for m in self.metrics]
        
        return {
            'avg_latency': np.mean(latencies),
            'max_latency': max(latencies),
            'avg_slippage': np.mean(slippages),
            'avg_fill_rate': np.mean(fill_rates),
            'total_cost': sum(costs),
            'num_trades': len(self.metrics),
            'position': self.position
        }
    
    async def run(
        self,
        symbol: str,
        interval: float = 0.001  # 1ms interval
    ):
        """
        Run high-frequency trading strategy.
        
        Args:
            symbol: Trading symbol
            interval: Update interval in seconds
        """
        while True:
            try:
                # Update market data
                price, volume, order_flow = await self.update_market_data(
                    symbol
                )
                
                # Generate signals
                signals = []
                
                momentum_signal = self.generate_momentum_signal()
                if momentum_signal:
                    signals.append(momentum_signal)
                
                mean_rev_signal = self.generate_mean_reversion_signal()
                if mean_rev_signal:
                    signals.append(mean_rev_signal)
                
                flow_signal = self.generate_order_flow_signal()
                if flow_signal:
                    signals.append(flow_signal)
                
                # Combine signals
                signal = self.combine_signals(signals)
                if signal:
                    self.signal_buffer.append(signal)
                    
                    # Calculate position size
                    size = self.calculate_position_size(signal, price)
                    
                    if abs(size) > 0:
                        # Execute order
                        success, metrics = await self.execute_order(
                            symbol,
                            size,
                            price
                        )
                        
                        if success:
                            self.metrics.append(metrics)
                
                # Log performance
                if len(self.metrics) % 100 == 0:
                    metrics = self.get_performance_metrics()
                    logger.info(f"Performance metrics: {metrics}")
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(interval)
