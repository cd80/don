"""
Cross-Exchange Arbitrage Strategy Module for Bitcoin Trading RL.
Implements arbitrage strategies to profit from price differences across exchanges.
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

class ArbitrageType(Enum):
    """Types of arbitrage strategies."""
    SIMPLE = 'simple'  # Direct price difference between two exchanges
    TRIANGULAR = 'triangular'  # Price differences across three currency pairs
    STATISTICAL = 'statistical'  # Mean-reversion based arbitrage
    CROSS_BORDER = 'cross_border'  # Arbitrage across different regions

@dataclass
class ArbitrageOpportunity:
    """Container for arbitrage opportunity details."""
    exchange_a: str
    exchange_b: str
    symbol: str
    price_a: float
    price_b: float
    spread: float
    timestamp: float
    volume_a: float
    volume_b: float
    estimated_profit: float
    transaction_costs: float
    net_profit: float
    risk_metrics: Dict[str, float]

class ArbitrageStrategy:
    """
    Cross-exchange arbitrage strategy that identifies and executes
    profitable arbitrage opportunities.
    """
    
    def __init__(
        self,
        config: Dict,
        exchanges: List[str],
        min_profit_threshold: float = 0.001,
        max_position_size: float = 1.0,
        transaction_costs: Dict[str, float] = None
    ):
        """
        Initialize arbitrage strategy.
        
        Args:
            config: Strategy configuration
            exchanges: List of exchanges to monitor
            min_profit_threshold: Minimum profit threshold for execution
            max_position_size: Maximum position size as fraction of capital
            transaction_costs: Dictionary of transaction costs by exchange
        """
        self.config = config
        self.exchanges = exchanges
        self.min_profit_threshold = min_profit_threshold
        self.max_position_size = max_position_size
        self.transaction_costs = transaction_costs or {
            exchange: 0.001 for exchange in exchanges
        }
        
        # Initialize exchange connections
        self.exchange_clients = {}
        for exchange in exchanges:
            self.exchange_clients[exchange] = BinanceFetcher(
                config['data'][exchange]
            )
        
        # Initialize opportunity tracking
        self.opportunities = []
        self.active_positions = {}
        
        logger.info(
            f"Initialized arbitrage strategy with {len(exchanges)} exchanges"
        )
    
    async def fetch_orderbooks(
        self,
        symbol: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch orderbooks from all exchanges.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of orderbooks by exchange
        """
        orderbooks = {}
        tasks = []
        
        for exchange, client in self.exchange_clients.items():
            tasks.append(client.fetch_orderbook(symbol))
        
        results = await asyncio.gather(*tasks)
        
        for exchange, orderbook in zip(self.exchanges, results):
            orderbooks[exchange] = orderbook
        
        return orderbooks
    
    def calculate_effective_prices(
        self,
        orderbook: pd.DataFrame,
        volume: float
    ) -> Tuple[float, float]:
        """
        Calculate effective prices for given volume.
        
        Args:
            orderbook: Exchange orderbook
            volume: Trade volume
            
        Returns:
            Tuple of (bid price, ask price)
        """
        # Calculate volume-weighted average prices
        bid_price = 0
        ask_price = 0
        remaining_volume = volume
        
        # Calculate bid price
        for price, size in orderbook['bids'].items():
            if remaining_volume <= 0:
                break
            executed = min(size, remaining_volume)
            bid_price += price * executed
            remaining_volume -= executed
        
        if remaining_volume > 0:
            return 0, 0  # Not enough liquidity
        
        bid_price /= volume
        
        # Calculate ask price
        remaining_volume = volume
        for price, size in orderbook['asks'].items():
            if remaining_volume <= 0:
                break
            executed = min(size, remaining_volume)
            ask_price += price * executed
            remaining_volume -= executed
        
        if remaining_volume > 0:
            return 0, 0  # Not enough liquidity
        
        ask_price /= volume
        
        return bid_price, ask_price
    
    def identify_opportunities(
        self,
        orderbooks: Dict[str, pd.DataFrame],
        volume: float
    ) -> List[ArbitrageOpportunity]:
        """
        Identify arbitrage opportunities across exchanges.
        
        Args:
            orderbooks: Dictionary of orderbooks by exchange
            volume: Trade volume to consider
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        for i, exchange_a in enumerate(self.exchanges):
            for exchange_b in self.exchanges[i+1:]:
                # Calculate effective prices
                bid_a, ask_a = self.calculate_effective_prices(
                    orderbooks[exchange_a],
                    volume
                )
                bid_b, ask_b = self.calculate_effective_prices(
                    orderbooks[exchange_b],
                    volume
                )
                
                if bid_a == 0 or ask_a == 0 or bid_b == 0 or ask_b == 0:
                    continue  # Skip if insufficient liquidity
                
                # Check opportunities
                # Buy on A, sell on B
                spread_ab = bid_b - ask_a
                costs_ab = (self.transaction_costs[exchange_a] +
                          self.transaction_costs[exchange_b])
                profit_ab = spread_ab - costs_ab
                
                if profit_ab > self.min_profit_threshold:
                    opportunities.append(
                        ArbitrageOpportunity(
                            exchange_a=exchange_a,
                            exchange_b=exchange_b,
                            symbol=orderbooks[exchange_a].name,
                            price_a=ask_a,
                            price_b=bid_b,
                            spread=spread_ab,
                            timestamp=time.time(),
                            volume_a=volume,
                            volume_b=volume,
                            estimated_profit=profit_ab * volume,
                            transaction_costs=costs_ab * volume,
                            net_profit=(profit_ab - costs_ab) * volume,
                            risk_metrics=self.calculate_risk_metrics(
                                ask_a, bid_b, volume
                            )
                        )
                    )
                
                # Buy on B, sell on A
                spread_ba = bid_a - ask_b
                costs_ba = (self.transaction_costs[exchange_b] +
                          self.transaction_costs[exchange_a])
                profit_ba = spread_ba - costs_ba
                
                if profit_ba > self.min_profit_threshold:
                    opportunities.append(
                        ArbitrageOpportunity(
                            exchange_a=exchange_b,
                            exchange_b=exchange_a,
                            symbol=orderbooks[exchange_a].name,
                            price_a=ask_b,
                            price_b=bid_a,
                            spread=spread_ba,
                            timestamp=time.time(),
                            volume_a=volume,
                            volume_b=volume,
                            estimated_profit=profit_ba * volume,
                            transaction_costs=costs_ba * volume,
                            net_profit=(profit_ba - costs_ba) * volume,
                            risk_metrics=self.calculate_risk_metrics(
                                ask_b, bid_a, volume
                            )
                        )
                    )
        
        return opportunities
    
    def calculate_risk_metrics(
        self,
        entry_price: float,
        exit_price: float,
        volume: float
    ) -> Dict[str, float]:
        """
        Calculate risk metrics for arbitrage opportunity.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            volume: Trade volume
            
        Returns:
            Dictionary of risk metrics
        """
        return {
            'price_ratio': exit_price / entry_price,
            'execution_time_risk': 0.1,  # Estimated execution time in seconds
            'slippage_risk': 0.001 * volume,
            'liquidity_risk': 0.002 * volume,
            'counterparty_risk': 0.001
        }
    
    def filter_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity]
    ) -> List[ArbitrageOpportunity]:
        """
        Filter arbitrage opportunities based on criteria.
        
        Args:
            opportunities: List of opportunities
            
        Returns:
            Filtered list of opportunities
        """
        filtered = []
        
        for opp in opportunities:
            # Check minimum profit
            if opp.net_profit < self.min_profit_threshold:
                continue
            
            # Check risk metrics
            if (opp.risk_metrics['execution_time_risk'] > 0.5 or
                opp.risk_metrics['slippage_risk'] > opp.net_profit * 0.1 or
                opp.risk_metrics['liquidity_risk'] > opp.net_profit * 0.1):
                continue
            
            # Check active positions
            if (opp.exchange_a in self.active_positions or
                opp.exchange_b in self.active_positions):
                continue
            
            filtered.append(opp)
        
        return filtered
    
    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity
    ) -> bool:
        """
        Execute arbitrage opportunity.
        
        Args:
            opportunity: Arbitrage opportunity to execute
            
        Returns:
            Whether execution was successful
        """
        try:
            # Place entry order
            entry_order = await self.exchange_clients[
                opportunity.exchange_a
            ].create_order(
                symbol=opportunity.symbol,
                side='buy',
                quantity=opportunity.volume_a,
                price=opportunity.price_a
            )
            
            # Place exit order
            exit_order = await self.exchange_clients[
                opportunity.exchange_b
            ].create_order(
                symbol=opportunity.symbol,
                side='sell',
                quantity=opportunity.volume_b,
                price=opportunity.price_b
            )
            
            # Update active positions
            self.active_positions[opportunity.exchange_a] = entry_order
            self.active_positions[opportunity.exchange_b] = exit_order
            
            # Record opportunity
            self.opportunities.append(opportunity)
            
            logger.info(
                f"Executed arbitrage: {opportunity.exchange_a} -> "
                f"{opportunity.exchange_b}, profit: {opportunity.net_profit:.4f}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to execute arbitrage: {str(e)}")
            return False
    
    async def monitor_positions(self):
        """Monitor and manage active arbitrage positions."""
        for exchange, order in list(self.active_positions.items()):
            try:
                # Check order status
                status = await self.exchange_clients[exchange].get_order(
                    symbol=order['symbol'],
                    order_id=order['id']
                )
                
                if status['status'] == 'filled':
                    del self.active_positions[exchange]
                elif status['status'] == 'canceled':
                    del self.active_positions[exchange]
                elif (time.time() - status['timestamp'] >
                      self.config['max_order_time']):
                    # Cancel stale orders
                    await self.exchange_clients[exchange].cancel_order(
                        symbol=order['symbol'],
                        order_id=order['id']
                    )
                    del self.active_positions[exchange]
            
            except Exception as e:
                logger.error(f"Error monitoring position: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        if not self.opportunities:
            return {}
        
        profits = [opp.net_profit for opp in self.opportunities]
        
        return {
            'total_profit': sum(profits),
            'avg_profit': np.mean(profits),
            'profit_std': np.std(profits),
            'num_trades': len(profits),
            'success_rate': len([p for p in profits if p > 0]) / len(profits),
            'profit_factor': (sum([p for p in profits if p > 0]) /
                            abs(sum([p for p in profits if p < 0]))
                            if any(p < 0 for p in profits) else float('inf'))
        }
    
    async def run(
        self,
        symbol: str,
        volume: float,
        interval: float = 1.0
    ):
        """
        Run arbitrage strategy.
        
        Args:
            symbol: Trading symbol
            volume: Trade volume
            interval: Update interval in seconds
        """
        while True:
            try:
                # Fetch orderbooks
                orderbooks = await self.fetch_orderbooks(symbol)
                
                # Identify opportunities
                opportunities = self.identify_opportunities(
                    orderbooks,
                    volume
                )
                
                # Filter opportunities
                filtered_opportunities = self.filter_opportunities(
                    opportunities
                )
                
                # Execute opportunities
                for opportunity in filtered_opportunities:
                    await self.execute_arbitrage(opportunity)
                
                # Monitor positions
                await self.monitor_positions()
                
                # Log performance
                metrics = self.get_performance_metrics()
                if metrics:
                    logger.info(f"Performance metrics: {metrics}")
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in arbitrage loop: {str(e)}")
                await asyncio.sleep(interval)
