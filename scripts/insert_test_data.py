import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from don.database.models import MarketData

# Create test data with enough points for all technical indicators
periods = 100  # Enough data for all indicators (SMA-20, RSI-14, etc.)
dates = pd.date_range(start='2024-01-01', periods=periods, freq='h')

# Generate realistic price movement
np.random.seed(42)  # For reproducibility
price = 100.0
prices = []
for _ in range(periods):
    change = np.random.normal(0, 1)  # Random price change
    price += change
    prices.append(price)

# Create DataFrame with OHLCV data
data = pd.DataFrame({
    'timestamp': dates,
    'symbol': ['BTCUSDT'] * periods,
    'open': prices,
    'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],  # Slightly higher than price
    'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],   # Slightly lower than price
    'close': [p + np.random.normal(0, 0.2) for p in prices],      # Close near the price
    'volume': np.random.uniform(1000, 2000, periods)              # Random volume
})

# Insert into database
engine = create_engine('postgresql://test:test@localhost/test')
data.to_sql('market_data', engine, if_exists='append', index=False)
print("Test market data inserted successfully")
