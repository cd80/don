"""Verification script for market microstructure features.

This script verifies:
1. Historical data processing
2. Real-time calculation performance
3. Database integration
"""

import pandas as pd
import numpy as np
from don.features.microstructure import MarketMicrostructureFeatures
from don.database.models import MarketMicrostructureFeatures as MicrostructureDB
from sqlalchemy import create_engine
import time
from sqlalchemy.orm import Session

def generate_test_data(size: int = 1000) -> pd.DataFrame:
    """Generate realistic market data for testing."""
    timestamps = pd.date_range('2024-01-01', periods=size, freq='1min')
    np.random.seed(42)

    # Generate base price series with random walk
    base_price = 100 + np.random.randn(size).cumsum() * 0.1

    # Generate order book data with realistic bid-ask spread
    data = []
    order_id = 0  # Unique identifier for each order
    for i, ts in enumerate(timestamps):
        # Generate multiple orders at each timestamp
        for _ in range(5):  # 5 orders per side
            # Bids slightly below base price, asks slightly above
            if np.random.random() < 0.5:
                price = base_price[i] * (1 - np.random.uniform(0.001, 0.005))
                side = 'bid'
            else:
                price = base_price[i] * (1 + np.random.uniform(0.001, 0.005))
                side = 'ask'

            data.append({
                'timestamp': ts,
                'order_id': order_id,  # Add unique order ID
                'side': side,
                'price': price,
                'quantity': abs(np.random.normal(1, 0.5)),
                'is_buyer_maker': np.random.choice([True, False]),
                'close': base_price[i]
            })
            order_id += 1

    # Create DataFrame with multi-index
    df = pd.DataFrame(data)
    df = df.set_index(['timestamp', 'order_id'])

    # Add liquidation events (with realistic frequency)
    liq_timestamps = pd.date_range('2024-01-01', periods=size//20, freq='20min')  # One liquidation every 20 minutes
    liquidations = []
    for ts in liq_timestamps:
        # Add both long and short liquidations with different probabilities
        if np.random.random() < 0.7:  # 70% chance of liquidation
            side = 'long' if np.random.random() < 0.5 else 'short'
            liquidations.append({
                'timestamp': ts,
                'order_id': -len(liquidations) - 1,  # Negative IDs for liquidations
                'side': side,
                'quantity': abs(np.random.normal(5, 2)),
                'price': base_price[timestamps.get_loc(ts)],
                'is_buyer_maker': False,
                'close': base_price[timestamps.get_loc(ts)]
            })

    if liquidations:
        liq_df = pd.DataFrame(liquidations).set_index(['timestamp', 'order_id'])
        df = pd.concat([df, liq_df])

    return df

def verify_historical_processing():
    """Test processing of historical data."""
    print("\n=== Historical Data Processing Test ===")

    # Generate test data
    data = generate_test_data(1000)
    print(f"Generated {len(data)} test records")
    print("\nData Structure:")
    print(f"Index levels: {data.index.names}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Unique timestamps: {len(data.index.get_level_values(0).unique())}")

    # Create calculator instance
    calculator = MarketMicrostructureFeatures()

    try:
        # Calculate features
        features = calculator.calculate(data)

        # Print feature statistics
        print("\nFeature Statistics:")
        for column in features.columns:
            stats = features[column].describe()
            print(f"\n{column}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")

        return features
    except Exception as e:
        print(f"\nError calculating features: {str(e)}")
        print("\nData sample:")
        print(data.head())
        raise

def verify_realtime_performance():
    """Test real-time calculation performance."""
    print("\n=== Real-time Performance Test ===")
    calculator = MarketMicrostructureFeatures()

    # Generate small batches of data
    batch_sizes = [10, 100, 1000]
    for size in batch_sizes:
        data = generate_test_data(size)

        start_time = time.time()
        features = calculator.calculate(data)
        end_time = time.time()

        time_per_record = (end_time - start_time) * 1000 / size  # ms per record
        print(f"Batch size {size}: {time_per_record:.2f} ms per record")

def verify_database_integration(features: pd.DataFrame):
    """Test database integration."""
    print("\n=== Database Integration Test ===")

    # Create in-memory test database
    engine = create_engine('sqlite:///:memory:')
    MicrostructureDB.metadata.create_all(engine)

    try:
        # Create database session
        with Session(engine) as session:
            # Convert features to database records
            records = []
            for timestamp, row in features.iterrows():
                record = MicrostructureDB(
                    timestamp=timestamp,
                    symbol='BTCUSDT',
                    order_imbalance=float(row['order_imbalance']),
                    trade_flow_imbalance=float(row['trade_flow_imbalance']),
                    realized_volatility=float(row['realized_volatility']),
                    long_liquidation_volume=float(row['long_liquidation_volume']),
                    short_liquidation_volume=float(row['short_liquidation_volume']),
                    liquidation_imbalance=float(row['liquidation_imbalance'])
                )
                records.append(record)

            # Insert records
            session.add_all(records[:10])
            session.commit()
            print("Successfully inserted test records into database")

            # Verify retrieval
            result = session.query(MicrostructureDB).first()
            if result:
                print("\nSample record from database:")
                print(f"Timestamp: {result.timestamp}")
                print(f"Symbol: {result.symbol}")
                print(f"Order Imbalance: {result.order_imbalance:.6f}")
                print(f"Trade Flow Imbalance: {result.trade_flow_imbalance:.6f}")
                print(f"Realized Volatility: {result.realized_volatility:.6f}")
                print(f"Long Liquidation Volume: {result.long_liquidation_volume:.6f}")
                print(f"Short Liquidation Volume: {result.short_liquidation_volume:.6f}")
                print(f"Liquidation Imbalance: {result.liquidation_imbalance:.6f}")
            else:
                print("\nNo records found in database")

    except Exception as e:
        print(f"Database test failed: {str(e)}")
        raise

def main():
    """Run all verification tests."""
    print("Starting verification tests...")

    # Run historical processing test
    features = verify_historical_processing()

    # Run performance test
    verify_realtime_performance()

    # Run database integration test
    verify_database_integration(features)

    print("\nVerification complete!")

if __name__ == "__main__":
    main()
