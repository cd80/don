"""Add partitioning and indexes.

Revision ID: 001
Revises:
Create Date: 2024-01-13

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timedelta

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Enable partitioning extension
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_partman')

    # Convert existing tables to partitioned tables
    tables = [
        'market_data',
        'trades',
        'orderbook',
        'liquidations',
        'volume',
        'technical_features',
        'market_microstructure_features'
    ]

    for table in tables:
        # Create temporary table
        op.execute(f'CREATE TABLE {table}_temp (LIKE {table} INCLUDING ALL)')

        # Copy data to temp table
        op.execute(f'INSERT INTO {table}_temp SELECT * FROM {table}')

        # Drop original table
        op.execute(f'DROP TABLE {table}')

        # Create partitioned table
        op.execute(f'''
            CREATE TABLE {table} (LIKE {table}_temp INCLUDING ALL)
            PARTITION BY RANGE (timestamp)
        ''')

        # Create partitions
        current_date = datetime.utcnow().replace(day=1, hour=0, minute=0,
                                               second=0, microsecond=0)
        for i in range(12):
            partition_date = current_date + timedelta(days=32 * i)
            next_month = partition_date + timedelta(days=32)
            partition_name = f"{table}_{partition_date.strftime('%Y_%m')}"

            op.execute(f'''
                CREATE TABLE {partition_name}
                PARTITION OF {table}
                FOR VALUES FROM
                ('{partition_date.strftime('%Y-%m-%d')}')
                TO ('{next_month.strftime('%Y-%m-%d')}')
            ''')

        # Copy data back
        op.execute(f'INSERT INTO {table} SELECT * FROM {table}_temp')

        # Drop temp table
        op.execute(f'DROP TABLE {table}_temp')

    # Create composite indexes
    op.create_index('idx_market_data_symbol_ts',
                    'market_data', ['symbol', 'timestamp'])
    op.create_index('idx_trades_symbol_ts',
                    'trades', ['symbol', 'timestamp'])
    op.create_index('idx_orderbook_symbol_ts',
                    'orderbook', ['symbol', 'timestamp'])
    op.create_index('idx_liquidations_symbol_ts',
                    'liquidations', ['symbol', 'timestamp'])
    op.create_index('idx_volume_symbol_ts',
                    'volume', ['symbol', 'timestamp'])
    op.create_index('idx_tech_features_symbol_ts',
                    'technical_features', ['symbol', 'timestamp'])
    op.create_index('idx_micro_features_symbol_ts',
                    'market_microstructure_features', ['symbol', 'timestamp'])

    # Create additional indexes for common queries
    op.create_index('idx_market_data_symbol_ts_close',
                    'market_data', ['symbol', 'timestamp', 'close'])
    op.create_index('idx_trades_symbol_ts_price',
                    'trades', ['symbol', 'timestamp', 'price'])

def downgrade():
    # Drop indexes
    op.drop_index('idx_market_data_symbol_ts')
    op.drop_index('idx_trades_symbol_ts')
    op.drop_index('idx_orderbook_symbol_ts')
    op.drop_index('idx_liquidations_symbol_ts')
    op.drop_index('idx_volume_symbol_ts')
    op.drop_index('idx_tech_features_symbol_ts')
    op.drop_index('idx_micro_features_symbol_ts')
    op.drop_index('idx_market_data_symbol_ts_close')
    op.drop_index('idx_trades_symbol_ts_price')


    # Convert partitioned tables back to regular tables
    tables = [
        'market_data',
        'trades',
        'orderbook',
        'liquidations',
        'volume',
        'technical_features',
        'market_microstructure_features'
    ]

    for table in tables:
        # Create temporary table
        op.execute(f'CREATE TABLE {table}_temp (LIKE {table} INCLUDING ALL)')

        # Copy data to temp table
        op.execute(f'INSERT INTO {table}_temp SELECT * FROM {table}')

        # Drop partitioned table and its partitions
        op.execute(f'DROP TABLE {table} CASCADE')

        # Create regular table
        op.execute(f'''
            CREATE TABLE {table} (LIKE {table}_temp INCLUDING ALL)
        ''')

        # Copy data back
        op.execute(f'INSERT INTO {table} SELECT * FROM {table}_temp')

        # Drop temp table
        op.execute(f'DROP TABLE {table}_temp')
