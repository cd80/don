import pandas as pd
import numpy as np

# Create sample data like the test
dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
data = pd.DataFrame({
    'close': np.concatenate([np.linspace(100, 120, 50), np.linspace(120, 100, 50)]),
    'volume': np.ones(100) * 1000,  # Use constant volume to simplify analysis
    'high': np.concatenate([np.linspace(102, 122, 50), np.linspace(122, 102, 50)]),
    'low': np.concatenate([np.linspace(98, 118, 50), np.linspace(118, 98, 50)])
}, index=dates)

# Calculate components
typical_price = (data['high'] + data['low'] + data['close']) / 3
cumsum_typical_price_volume = (typical_price * data['volume']).cumsum()
cumsum_volume = data['volume'].cumsum()
vwap = cumsum_typical_price_volume / cumsum_volume

# Print analysis at key points
print("\nFirst 5 periods:")
for i in range(5):
    print(f"Period {i}:")
    print(f"  Typical Price: {typical_price.iloc[i]:.2f}")
    print(f"  Cumsum(TP*V): {cumsum_typical_price_volume.iloc[i]:.2f}")
    print(f"  Cumsum(V): {cumsum_volume.iloc[i]:.2f}")
    print(f"  VWAP: {vwap.iloc[i]:.2f}")
    print(f"  High: {data['high'].iloc[i]:.2f}")

print("\nAround trend reversal (periods 48-52):")
for i in range(48, 53):
    print(f"Period {i}:")
    print(f"  Typical Price: {typical_price.iloc[i]:.2f}")
    print(f"  Cumsum(TP*V): {cumsum_typical_price_volume.iloc[i]:.2f}")
    print(f"  Cumsum(V): {cumsum_volume.iloc[i]:.2f}")
    print(f"  VWAP: {vwap.iloc[i]:.2f}")
    print(f"  High: {data['high'].iloc[i]:.2f}")

print("\nLast 5 periods:")
for i in range(-5, 0):
    print(f"Period {i}:")
    print(f"  Typical Price: {typical_price.iloc[i]:.2f}")
    print(f"  Cumsum(TP*V): {cumsum_typical_price_volume.iloc[i]:.2f}")
    print(f"  Cumsum(V): {cumsum_volume.iloc[i]:.2f}")
    print(f"  VWAP: {vwap.iloc[i]:.2f}")
    print(f"  High: {data['high'].iloc[i]:.2f}")

# Check if VWAP exceeds bounds
exceeds_high = vwap > data['high']
if exceeds_high.any():
    print("\nPeriods where VWAP exceeds high:")
    for idx in data.index[exceeds_high]:
        print(f"At {idx}:")
        print(f"  VWAP: {vwap[idx]:.2f}")
        print(f"  High: {data['high'][idx]:.2f}")
        print(f"  Difference: {(vwap[idx] - data['high'][idx]):.2f}")
