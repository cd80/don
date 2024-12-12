import pandas as pd
import numpy as np
from don.features.technical import TechnicalIndicators

# Create sample data exactly like the test
dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
data = pd.DataFrame({
    'close': np.concatenate([
        np.linspace(100, 120, 50),
        np.linspace(120, 100, 50)
    ]),
    'volume': np.ones(100) * 1000,  # Use constant volume for easier debugging
    'high': np.concatenate([
        np.linspace(102, 122, 50),
        np.linspace(122, 102, 50)
    ]),
    'low': np.concatenate([
        np.linspace(98, 118, 50),
        np.linspace(118, 98, 50)
    ])
}, index=dates)

# Calculate using our implementation
indicators = TechnicalIndicators()
result = indicators.calculate(data)
vwap = result['vwap']

# Calculate using test's method
typical_price = (data['high'] + data['low'] + data['close']) / 3
expected_vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

# Compare results
print("\nFirst 5 periods:")
for i in range(5):
    print(f"Period {i}:")
    print(f"  Our VWAP:      {vwap.iloc[i]:.6f}")
    print(f"  Expected VWAP: {expected_vwap.iloc[i]:.6f}")
    print(f"  Difference:    {(vwap.iloc[i] - expected_vwap.iloc[i]):.6f}")
    print(f"  High Price:    {data['high'].iloc[i]:.6f}")

print("\nAround trend reversal (periods 48-52):")
for i in range(48, 53):
    print(f"Period {i}:")
    print(f"  Our VWAP:      {vwap.iloc[i]:.6f}")
    print(f"  Expected VWAP: {expected_vwap.iloc[i]:.6f}")
    print(f"  Difference:    {(vwap.iloc[i] - expected_vwap.iloc[i]):.6f}")
    print(f"  High Price:    {data['high'].iloc[i]:.6f}")

print("\nLast 5 periods:")
for i in range(-5, 0):
    print(f"Period {i}:")
    print(f"  Our VWAP:      {vwap.iloc[i]:.6f}")
    print(f"  Expected VWAP: {expected_vwap.iloc[i]:.6f}")
    print(f"  Difference:    {(vwap.iloc[i] - expected_vwap.iloc[i]):.6f}")
    print(f"  High Price:    {data['high'].iloc[i]:.6f}")

# Find where VWAP exceeds high price
exceeds_high = vwap > data['high']
if exceeds_high.any():
    print("\nPeriods where our VWAP exceeds high:")
    for idx in data.index[exceeds_high]:
        print(f"At {idx}:")
        print(f"  Our VWAP:      {vwap[idx]:.6f}")
        print(f"  Expected VWAP: {expected_vwap[idx]:.6f}")
        print(f"  High Price:    {data['high'][idx]:.6f}")
        print(f"  Difference from high: {(vwap[idx] - data['high'][idx]):.6f}")
