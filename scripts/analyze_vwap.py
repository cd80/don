import pandas as pd
import numpy as np

# Create sample data like the test
dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
data = pd.DataFrame({
    'close': np.concatenate([np.linspace(100, 120, 50), np.linspace(120, 100, 50)]),
    'volume': np.random.uniform(1000, 2000, 100),
    'high': np.concatenate([np.linspace(102, 122, 50), np.linspace(122, 102, 50)]),
    'low': np.concatenate([np.linspace(98, 118, 50), np.linspace(118, 98, 50)])
}, index=dates)

# Calculate VWAP
typical_price = (data['high'] + data['low'] + data['close']) / 3
vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

# Print analysis
print('\nSample points where VWAP exceeds high:')
exceeds = vwap > data['high']
if exceeds.any():
    print(pd.DataFrame({
        'timestamp': data.index[exceeds],
        'vwap': vwap[exceeds],
        'high': data['high'][exceeds],
        'typical_price': typical_price[exceeds]
    }).to_string())

# Print detailed analysis of price movements
print('\nPrice movement analysis:')
print(f'Initial typical price: {typical_price.iloc[0]:.2f}')
print(f'Final typical price: {typical_price.iloc[-1]:.2f}')
print(f'Max VWAP: {vwap.max():.2f}')
print(f'Min VWAP: {vwap.min():.2f}')
print(f'Max High: {data["high"].max():.2f}')
print(f'Min High: {data["high"].min():.2f}')
