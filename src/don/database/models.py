"""Database models for Don trading framework.

This module defines SQLAlchemy models for storing trading data, technical indicators,
and training information. Models include:
- Trade: Store individual trades
- OrderBook: Store order book snapshots
- Liquidation: Store liquidation events
- Volume: Store volume data
- TechnicalIndicator: Store calculated technical indicators
- Model: Store model metadata
- TrainingRun: Store training run information
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Trade(Base):
    """Store individual trades."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    is_buyer_maker = Column(Boolean, nullable=False)

    def __repr__(self):
        return f"<Trade(symbol={self.symbol}, timestamp={self.timestamp}, price={self.price})>"

class OrderBook(Base):
    """Store order book snapshots."""
    __tablename__ = 'orderbook'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    side = Column(String, nullable=False)  # 'buy' or 'sell'

    def __repr__(self):
        return f"<OrderBook(symbol={self.symbol}, timestamp={self.timestamp}, side={self.side})>"

class Liquidation(Base):
    """Store liquidation events."""
    __tablename__ = 'liquidations'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    side = Column(String, nullable=False)  # 'long' or 'short'

    def __repr__(self):
        return f"<Liquidation(symbol={self.symbol}, timestamp={self.timestamp}, side={self.side})>"

class Volume(Base):
    """Store volume data."""
    __tablename__ = 'volume'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float, nullable=False)

    def __repr__(self):
        return f"<Volume(symbol={self.symbol}, timestamp={self.timestamp})>"

class TechnicalIndicator(Base):
    """Store calculated technical indicators."""
    __tablename__ = 'technical_indicators'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    # Common technical indicators
    sma = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    obv = Column(Float)
    vwap = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    adx = Column(Float)

    def __repr__(self):
        return f"<TechnicalIndicator(symbol={self.symbol}, timestamp={self.timestamp})>"

class Model(Base):
    """Store model metadata."""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    parameters = Column(JSON, nullable=False)  # Store model hyperparameters
    description = Column(String)
    performance_metrics = Column(JSON)  # Store various performance metrics

    # Relationship with training runs
    training_runs = relationship("TrainingRun", back_populates="model")

    def __repr__(self):
        return f"<Model(name={self.name}, created_at={self.created_at})>"

class TrainingRun(Base):
    """Store training run information."""
    __tablename__ = 'training_runs'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String, nullable=False)  # 'running', 'completed', 'failed'
    metrics = Column(JSON)  # Store training metrics

    # Relationship with model
    model = relationship("Model", back_populates="training_runs")

    def __repr__(self):
        return f"<TrainingRun(model_id={self.model_id}, status={self.status})>"
