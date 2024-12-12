"""Test suite for Don trading framework database optimizations."""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from don.database import (
    Base, MarketData, Trade, OrderBook,
    DatabaseManager
)

@pytest.fixture
def engine():
    """Create test database engine."""
    return create_engine('postgresql://localhost/don_test')

@pytest.fixture
def db_manager(engine):
    """Create database manager for testing."""
    manager = DatabaseManager(engine)
    Base.metadata.create_all(engine)
    yield manager
    Base.metadata.drop_all(engine)


def test_table_partitioning(db_manager):
    """Test database partitioning functionality."""
    # Set up partitions
    db_manager.setup_partitions()

    # Verify partition creation
    with db_manager.engine.begin() as conn:
        result = conn.execute(text("""
            SELECT inhrelid::regclass AS partition
            FROM pg_inherits
            WHERE inhparent = 'market_data'::regclass
        """))
        partitions = [row[0] for row in result]
        assert len(partitions) >= 12  # At least 12 monthly partitions

def test_index_creation(db_manager):
    """Test index creation and effectiveness."""
    # Create indexes
    db_manager.setup_indexes()

    # Verify index creation
    with db_manager.engine.begin() as conn:
        result = conn.execute(text("""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'market_data'
        """))
        indexes = [row[0] for row in result]
        assert 'idx_market_data_symbol_ts' in indexes
        assert 'idx_market_data_symbol_ts_close' in indexes

def test_query_performance(db_manager):
    """Test query performance with indexes."""
    # Insert test data
    with Session(db_manager.engine) as session:
        for i in range(1000):
            data = MarketData(
                timestamp=datetime.utcnow() + timedelta(minutes=i),
                symbol='BTC-USD',
                open=40000,
                high=41000,
                low=39000,
                close=40500,
                volume=100
            )
            session.add(data)
        session.commit()

    # Test query performance
    query = """
        SELECT *
        FROM market_data
        WHERE symbol = 'BTC-USD'
        AND timestamp >= NOW() - INTERVAL '1 day'
    """
    execution_time, stats = db_manager.monitor_query_performance(query)

    # Verify query performance
    assert execution_time < 0.1  # Less than 100ms
    assert 'Index Scan' in str(stats['execution_plan'])

def test_data_archival(db_manager):
    """Test data archival functionality."""
    # Insert old data
    old_date = datetime.utcnow() - timedelta(days=200)
    with Session(db_manager.engine) as session:
        data = MarketData(
            timestamp=old_date,
            symbol='BTC-USD',
            open=40000,
            high=41000,
            low=39000,
            close=40500,
            volume=100
        )
        session.add(data)
        session.commit()

    # Archive old data
    db_manager.archive_old_data(months_to_keep=6)

    # Verify data moved to archive
    with db_manager.engine.begin() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*)
            FROM market_data_archive
            WHERE timestamp < NOW() - INTERVAL '6 months'
        """))
        archive_count = result.scalar()
        assert archive_count > 0

        result = conn.execute(text("""
            SELECT COUNT(*)
            FROM market_data
            WHERE timestamp < NOW() - INTERVAL '6 months'
        """))
        active_count = result.scalar()
        assert active_count == 0
