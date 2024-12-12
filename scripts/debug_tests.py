import pandas as pd
import numpy as np
from don.features.technical import TechnicalIndicators
from don.rl.actions import DiscreteActionSpace

# Debug VWAP calculation
def debug_vwap():
    print("\n=== VWAP Debug ===")
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

    indicators = TechnicalIndicators()
    result = indicators.calculate(data)
    vwap = result['vwap']

    # Print detailed comparison at points where VWAP exceeds bounds
    exceeds_high = vwap > data['high']
    below_low = vwap < data['low']

    if exceeds_high.any():
        print("\nPoints where VWAP exceeds high:")
        for idx in data.index[exceeds_high]:
            print(f"\nTimestamp: {idx}")
            print(f"VWAP: {vwap[idx]:.6f}")
            print(f"High: {data['high'][idx]:.6f}")
            print(f"Low: {data['low'][idx]:.6f}")
            print(f"Close: {data['close'][idx]:.6f}")
            print(f"Volume: {data['volume'][idx]:.6f}")

    if below_low.any():
        print("\nPoints where VWAP below low:")
        for idx in data.index[below_low]:
            print(f"\nTimestamp: {idx}")
            print(f"VWAP: {vwap[idx]:.6f}")
            print(f"High: {data['high'][idx]:.6f}")
            print(f"Low: {data['low'][idx]:.6f}")
            print(f"Close: {data['close'][idx]:.6f}")
            print(f"Volume: {data['volume'][idx]:.6f}")

# Debug action space conversion
def debug_action_space():
    print("\n=== Action Space Debug ===")
    positions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    space = DiscreteActionSpace(positions)

    test_positions = [-0.7, 0.3]
    for pos in test_positions:
        print(f"\nTesting position: {pos}")
        positions = np.array(space.positions)
        distances = np.abs(positions - pos)
        weights = np.ones_like(distances)
        weights[0] = 2.0
        weights[-1] = 2.0
        weighted_distances = distances / weights

        print("Positions:", positions)
        print("Raw distances:", distances)
        print("Weights:", weights)
        print("Weighted distances:", weighted_distances)
        print("Selected index:", np.argmin(weighted_distances))
        print("Expected index for -0.7:", 0)
        print("Expected index for 0.3:", 2)

if __name__ == "__main__":
    debug_vwap()
    debug_action_space()
