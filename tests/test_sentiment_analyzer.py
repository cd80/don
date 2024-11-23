"""
Tests for the sentiment analyzer module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def sentiment_analyzer():
    """Create a SentimentAnalyzer instance for testing."""
    config = {
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
    return SentimentAnalyzer(config)

@pytest.fixture
def sample_news_data():
    """Create sample news sentiment data."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='1h'),
        'source': ['NewsAPI'] * 5,
        'title': [f'Test News {i}' for i in range(5)],
        'sentiment_score': np.random.uniform(-1, 1, 5),
        'impact_score': np.random.uniform(0, 1, 5)
    })

@pytest.fixture
def sample_social_data():
    """Create sample social media sentiment data."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='1h'),
        'platform': ['Twitter'] * 5,
        'content_type': ['tweet'] * 5,
        'sentiment_score': np.random.uniform(-1, 1, 5),
        'engagement_score': np.random.uniform(0, 1000, 5)
    })

@pytest.mark.asyncio
async def test_fetch_news_sentiment(sentiment_analyzer):
    """Test news sentiment fetching."""
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    
    result = await sentiment_analyzer.fetch_news_sentiment(start_time, end_time)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(col in result.columns for col in [
        'timestamp', 'source', 'title', 'sentiment_score', 'impact_score'
    ])
    assert (result['sentiment_score'] >= -1).all() and (result['sentiment_score'] <= 1).all()
    assert (result['impact_score'] >= 0).all() and (result['impact_score'] <= 1).all()

@pytest.mark.asyncio
async def test_fetch_social_sentiment(sentiment_analyzer):
    """Test social media sentiment fetching."""
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    
    result = await sentiment_analyzer.fetch_social_sentiment(start_time, end_time)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(col in result.columns for col in [
        'timestamp', 'platform', 'content_type', 'sentiment_score', 'engagement_score'
    ])
    assert (result['sentiment_score'] >= -1).all() and (result['sentiment_score'] <= 1).all()
    assert (result['engagement_score'] >= 0).all()

def test_aggregate_sentiment(sentiment_analyzer, sample_news_data, sample_social_data):
    """Test sentiment aggregation."""
    result = sentiment_analyzer.aggregate_sentiment(
        sample_news_data,
        sample_social_data,
        window='1h'
    )
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(col in result.columns for col in [
        'timestamp', 'composite_sentiment', 'news_sentiment',
        'social_sentiment', 'confidence_score'
    ])
    assert (result['composite_sentiment'] >= -1).all() and (result['composite_sentiment'] <= 1).all()
    assert (result['confidence_score'] >= 0).all() and (result['confidence_score'] <= 1).all()

def test_calculate_sentiment_features(sentiment_analyzer, sample_news_data, sample_social_data):
    """Test sentiment feature calculation."""
    aggregated = sentiment_analyzer.aggregate_sentiment(
        sample_news_data,
        sample_social_data,
        window='1h'
    )
    
    result = sentiment_analyzer.calculate_sentiment_features(aggregated)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Check for additional calculated features
    expected_features = [
        'sentiment_momentum',
        'sentiment_volatility',
        'extreme_sentiment',
        'sentiment_divergence'
    ]
    assert all(feature in result.columns for feature in expected_features)

def test_initialization_with_invalid_config():
    """Test initialization with invalid configuration."""
    with pytest.raises(ValueError):
        SentimentAnalyzer({'invalid_key': 'value'})

def test_initialization_without_config():
    """Test initialization without configuration."""
    analyzer = SentimentAnalyzer()
    assert analyzer.config == {}
