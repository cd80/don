import pytest
import numpy as np
import pandas as pd
from don.features.technical import TechnicalIndicators

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
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
    return data

def test_sma_calculation(sample_data):
    indicators = TechnicalIndicators()
    result = indicators.calculate(sample_data)

    sma_20 = result['sma_20']
    assert len(sma_20) == len(sample_data)
    assert pd.isna(sma_20[:19]).all()
    assert not pd.isna(sma_20[19:]).any()

    expected_sma = sample_data['close'].rolling(window=20).mean()
    np.testing.assert_array_almost_equal(sma_20, expected_sma)

def test_rsi_calculation(sample_data):
    indicators = TechnicalIndicators()
    result = indicators.calculate(sample_data)

    rsi = result['rsi']
    assert len(rsi) == len(sample_data)
    assert pd.isna(rsi[:14]).all()
    assert not pd.isna(rsi[14:]).any()

    valid_rsi = rsi[14:]
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    uptrend_rsi = rsi[20:40].mean()
    downtrend_rsi = rsi[70:90].mean()
    assert uptrend_rsi > downtrend_rsi

def test_macd_calculation(sample_data):
    indicators = TechnicalIndicators()
    result = indicators.calculate(sample_data)

    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns
    assert 'macd_hist' in result.columns

    assert not pd.isna(result['macd'][33:]).any()
    assert not pd.isna(result['macd_signal'][33:]).any()
    assert not pd.isna(result['macd_hist'][33:]).any()

def test_bollinger_bands(sample_data):
    indicators = TechnicalIndicators()
    result = indicators.calculate(sample_data)

    assert 'bb_upper' in result.columns
    assert 'bb_middle' in result.columns
    assert 'bb_lower' in result.columns

    valid_idx = 19
    assert (result['bb_upper'][valid_idx:] > result['bb_middle'][valid_idx:]).all()
    assert (result['bb_middle'][valid_idx:] > result['bb_lower'][valid_idx:]).all()

    np.testing.assert_array_almost_equal(
        result['bb_middle'],
        result['sma_20']
    )

def test_obv_calculation(sample_data):
    indicators = TechnicalIndicators()
    result = indicators.calculate(sample_data)

    obv = result['obv']
    assert len(obv) == len(sample_data)
    assert not pd.isna(obv).any()

    uptrend_obv_change = obv[49] - obv[0]
    downtrend_obv_change = obv[99] - obv[50]
    assert uptrend_obv_change > 0
    assert downtrend_obv_change < 0

def test_vwap_calculation(sample_data):
    indicators = TechnicalIndicators()
    result = indicators.calculate(sample_data)

    vwap = result['vwap']
    assert len(vwap) == len(sample_data)
    assert not pd.isna(vwap).any()

    assert (vwap <= sample_data['high']).all()
    assert (vwap >= sample_data['low']).all()

    typical_price = (sample_data['high'] + sample_data['low'] + sample_data['close']) / 3
    expected_vwap = (typical_price * sample_data['volume']).cumsum() / sample_data['volume'].cumsum()
    np.testing.assert_array_almost_equal(vwap, expected_vwap)
