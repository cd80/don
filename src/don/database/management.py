"""Database management utilities for Don trading framework.

This module provides utilities for:
- Table partitioning
- Index management
- Data archival
- Query performance monitoring
"""

from datetime import datetime, timedelta
import logging
from typing import List, Optional, Tuple

from sqlalchemy import create_engine, text, Index, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy_utils import create_partitioned_table, drop_partitioned_table

from .models import (
    Base, MarketData, Trade, OrderBook, Liquidation, Volume,
    TechnicalFeatures, MarketMicrostructureFeatures
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage database operations including partitioning and maintenance."""

    def __init__(self, engine: Engine):
        """Initialize database manager.

        Args:
            engine: SQLAlchemy engine instance
        """
        self.engine = engine
        self.metadata = MetaData()

    def setup_partitions(self):
        """Set up partitioned tables for time-series data."""
        # Tables that need partitioning
        time_series_tables = [
            MarketData,
            Trade,
            OrderBook,
            Liquidation,
            Volume,
            TechnicalFeatures,
            MarketMicrostructureFeatures
        ]

        with self.engine.begin() as conn:
            for table_class in time_series_tables:
                # Create partitioned table
                create_partitioned_table(
                    table_class,
                    'timestamp',
                    'RANGE',
                    partition_by='month'
                )

                # Create initial partitions for the next 12 months
                self._create_monthly_partitions(conn, table_class.__tablename__)

    def setup_indexes(self):
        """Create optimized indexes for common query patterns."""
        indexes = [
            # Composite indexes for time-series queries
            Index('idx_market_data_symbol_ts',
                  MarketData.symbol, MarketData.timestamp),
            Index('idx_trades_symbol_ts',
                  Trade.symbol, Trade.timestamp),
            Index('idx_orderbook_symbol_ts',
                  OrderBook.symbol, OrderBook.timestamp),
            Index('idx_liquidations_symbol_ts',
                  Liquidation.symbol, Liquidation.timestamp),
            Index('idx_volume_symbol_ts',
                  Volume.symbol, Volume.timestamp),
            Index('idx_tech_features_symbol_ts',
                  TechnicalFeatures.symbol, TechnicalFeatures.timestamp),
            Index('idx_micro_features_symbol_ts',
                  MarketMicrostructureFeatures.symbol,
                  MarketMicrostructureFeatures.timestamp),

            # Additional indexes for specific queries
            Index('idx_market_data_symbol_ts_close',
                  MarketData.symbol, MarketData.timestamp, MarketData.close),
            Index('idx_trades_symbol_ts_price',
                  Trade.symbol, Trade.timestamp, Trade.price),
        ]

        # Create all indexes
        for idx in indexes:
            try:
                idx.create(self.engine)
                logger.info(f"Created index: {idx.name}")
            except Exception as e:
                logger.error(f"Failed to create index {idx.name}: {str(e)}")

    def _create_monthly_partitions(self, conn, table_name: str):
        """Create monthly partitions for the next 12 months."""
        current_date = datetime.utcnow().replace(day=1, hour=0, minute=0,
                                               second=0, microsecond=0)

        for i in range(12):
            partition_date = current_date + timedelta(days=32 * i)
            next_month = partition_date + timedelta(days=32)

            partition_name = f"{table_name}_{partition_date.strftime('%Y_%m')}"

            # Create partition for the month
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF {table_name}
                FOR VALUES FROM
                ('{partition_date.strftime('%Y-%m-%d')}')
                TO ('{next_month.strftime('%Y-%m-%d')}')
            """))

    def archive_old_data(self, months_to_keep: int = 6):
        """Archive data older than specified months.

        Args:
            months_to_keep: Number of recent months to keep in active tables
        """
        cutoff_date = datetime.utcnow() - timedelta(days=30 * months_to_keep)

        with self.engine.begin() as conn:
            # Archive data from each time-series table
            for table in [MarketData, Trade, OrderBook, Liquidation, Volume,
                         TechnicalFeatures, MarketMicrostructureFeatures]:
                table_name = table.__tablename__
                archive_table = f"{table_name}_archive"

                # Create archive table if it doesn't exist
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {archive_table}
                    (LIKE {table_name} INCLUDING ALL)
                """))

                # Move old data to archive
                conn.execute(text(f"""
                    WITH moved_rows AS (
                        DELETE FROM {table_name}
                        WHERE timestamp < :cutoff_date
                        RETURNING *
                    )
                    INSERT INTO {archive_table}
                    SELECT * FROM moved_rows
                """), {"cutoff_date": cutoff_date})

    def monitor_query_performance(self, query: str) -> Tuple[float, dict]:
        """Monitor query performance and return execution time and stats.

        Args:
            query: SQL query to monitor

        Returns:
            Tuple of (execution_time, stats_dict)
        """
        start_time = datetime.utcnow()

        with self.engine.begin() as conn:
            # Enable query statistics
            conn.execute(text("SET track_io_timing = ON"))

            # Execute query with EXPLAIN ANALYZE
            result = conn.execute(text(f"EXPLAIN ANALYZE {query}"))
            execution_plan = result.fetchall()

            # Get query statistics
            stats = conn.execute(text("""
                SELECT * FROM pg_stat_statements
                WHERE query = :query
                ORDER BY calls DESC
                LIMIT 1
            """), {"query": query}).fetchone()

        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()

        stats_dict = {
            'execution_time': execution_time,
            'execution_plan': execution_plan,
            'calls': stats.calls if stats else 0,
            'total_time': stats.total_time if stats else 0,
            'rows': stats.rows if stats else 0,
        }

        return execution_time, stats_dict
