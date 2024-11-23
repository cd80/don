"""
Tests for the feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.feature_engineering import FeatureEngineer
import os

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')  # Increased data points
    data = {
        'open_time': dates,
        'open': np.random.uniform(30000, 40000, 1000),
        'high': np.random.uniform(30000, 40000, 1000),
        'low': np.random.uniform(30000, 40000, 1000),
        'close': np.random.uniform(30000, 40000, 1000),
        'volume': np.random.uniform(1, 100, 1000),
        'quote_volume': np.random.uniform(30000, 40000, 1000),
        'trades': np.random.randint(100, 1000, 1000),
        'taker_buy_volume': np.random.uniform(1, 50, 1000),
        'taker_buy_quote_volume': np.random.uniform(30000, 40000, 1000)
    }
    # Ensure high is always >= open, close, low
    data['high'] = np.maximum.reduce([data['high'], data['open'], data['close']])
    # Ensure low is always <= open, close, high
    data['low'] = np.minimum.reduce([data['low'], data['open'], data['close']])
    return pd.DataFrame(data)

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'sentiment': {
            'news': {
                'apis': [
                    {
                        'name': 'newsapi',
                        'key': 'test_key'
                    }
                ]
            },
            'social': {
                'twitter': {
                    'api_key': 'test_key',
                    'api_secret': 'test_secret'
                },
                'reddit': {
                    'client_id': 'test_id',
                    'client_secret': 'test_secret'
                }
            },
            'analysis': {
                'window_size': '1h',
                'update_interval': 300
            }
        }
    }

@pytest.fixture
def feature_engineer(sample_data, test_config, tmp_path):
    """Create a FeatureEngineer instance for testing."""
    # Save sample data to temporary parquet file
    input_file = os.path.join(tmp_path, 'test_data.parquet')
    sample_data.to_parquet(input_file)
    
    return FeatureEngineer(
        input_file=input_file,
        output_dir=str(tmp_path),
        config=test_config
    )

def test_statistical_features(feature_engineer):
    """Test statistical feature calculation."""
    features = feature_engineer._calculate_statistical_features(feature_engineer.data)
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert 'price_mean_5' in features.columns
    assert 'volatility_20' in features.columns
    assert 'price_zscore_50' in features.columns
    
    # Test feature values
    assert not features['price_mean_5'].isna().all()
    assert not features['volatility_20'].isna().all()
    assert not features['price_zscore_50'].isna().all()

def test_orderbook_features(feature_engineer):
    """Test order book feature calculation."""
    features = feature_engineer._calculate_orderbook_features(feature_engineer.data)
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert 'buy_volume_ratio' in features.columns
    assert 'avg_trade_size' in features.columns
    
    # Test value ranges
    assert (features['buy_volume_ratio'] >= 0).all() and (features['buy_volume_ratio'] <= 1).all()
    assert (features['avg_trade_size'] >= 0).all()

def test_time_features(feature_engineer):
    """Test time feature calculation."""
    features = feature_engineer._calculate_time_features(feature_engineer.data)
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert 'hour' in features.columns
    assert 'day_of_week' in features.columns
    assert 'month' in features.columns
    assert 'hour_sin' in features.columns
    
    # Test value ranges
    assert (features['hour'] >= 0).all() and (features['hour'] <= 23).all()
    assert (features['day_of_week'] >= 0).all() and (features['day_of_week'] <= 6).all()
    assert (features['month'] >= 1).all() and (features['month'] <= 12).all()

def test_technical_indicators(feature_engineer):
    """Test technical indicator calculation."""
    features = feature_engineer._calculate_technical_indicators(feature_engineer.data)
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert any('momentum' in col for col in features.columns)
    assert any('trend' in col for col in features.columns)
    assert any('volatility' in col for col in features.columns)
    
    # Test for NaN values
    assert not features.isna().all().all()

@pytest.mark.asyncio
async def test_sentiment_features(feature_engineer):
    """Test sentiment feature calculation."""
    features = await feature_engineer._calculate_sentiment_features(feature_engineer.data)
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert 'composite_sentiment' in features.columns
    assert 'sentiment_momentum' in features.columns
    assert 'sentiment_volatility' in features.columns
    assert 'sentiment_divergence' in features.columns
    
    # Test value ranges
    assert (features['composite_sentiment'] >= -1).all() and (features['composite_sentiment'] <= 1).all()
    assert not features['sentiment_momentum'].isna().all()
    assert (features['sentiment_volatility'] >= 0).all()

@pytest.mark.asyncio
async def test_generate_features(feature_engineer):
    """Test complete feature generation pipeline."""
    await feature_engineer.generate_features()
    
    # Check if output file was created
    output_files = os.listdir(feature_engineer.output_dir)
    assert any(file.startswith('features_') and file.endswith('.parquet') 
              for file in output_files)
    
    # Load and check generated features
    latest_file = max(
        [f for f in output_files if f.startswith('features_')],
        key=lambda x: os.path.getctime(os.path.join(feature_engineer.output_dir, x))
    )
    features_df = pd.read_parquet(os.path.join(feature_engineer.output_dir, latest_file))
    
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty
    assert len(features_df) == len(feature_engineer.data)
    
    # Check for presence of different feature types
    assert any('price_mean' in col for col in features_df.columns)
    assert any('volume' in col for col in features_df.columns)
    assert any('sentiment' in col for col in features_df.columns)
    assert any('momentum' in col for col in features_df.columns)
    assert 'hour' in features_df.columns
    assert 'composite_sentiment' in features_df.columns

def test_parallel_feature_calculation(feature_engineer):
    """Test parallel feature calculation."""
    features = feature_engineer._parallel_feature_calculation(
        feature_engineer._calculate_statistical_features,
        feature_engineer.data
    )
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert len(features) == len(feature_engineer.data)
    
    # Test that parallel calculation produces same results as single-threaded
    single_thread = feature_engineer._calculate_statistical_features(feature_engineer.data)
    pd.testing.assert_frame_equal(features, single_thread)

def test_empty_data_handling(feature_engineer):
    """Test handling of empty data."""
    empty_df = pd.DataFrame()
    
    # Test each feature calculation method with empty data
    assert feature_engineer._calculate_statistical_features(empty_df).empty
    assert feature_engineer._calculate_orderbook_features(empty_df).empty
    assert feature_engineer._calculate_time_features(empty_df).empty
    assert feature_engineer._calculate_technical_indicators(empty_df).empty
    assert feature_engineer._parallel_feature_calculation(
        feature_engineer._calculate_statistical_features,
        empty_df
    ).empty
