import pandas as pd
import numpy as np

# Recreate test data
dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
data = pd.DataFrame({
    'close': np.concatenate([
        np.linspace(100, 120, 50),
        np.linspace(120, 100, 50)
    ]),
    'volume': np.random.uniform(1000, 2000, 100),
    'high': np.concatenate([
        np.linspace(102, 122, 50),
        np.linspace(122, 102, 50)
    ]),
    'low': np.concatenate([
        np.linspace(98, 118, 50),
        np.linspace(118, 98, 50)
    ])
}, index=dates)

# Calculate VWAP components
typical_price = (data['high'] + data['low'] + data['close']) / 3
tp_vol = (typical_price * data['volume']).cumsum()
vol = data['volume'].cumsum()
vwap = tp_vol / vol

# Print analysis
print('\nFirst 5 rows:')
print(pd.DataFrame({
    'high': data['high'],
    'typical_price': typical_price,
    'vwap': vwap
}).head())

print('\nLast 5 rows:')
print(pd.DataFrame({
    'high': data['high'],
    'typical_price': typical_price,
    'vwap': vwap
}).tail())

print('\nVWAP exceeds high price at:')
print((vwap > data['high']).value_counts())
