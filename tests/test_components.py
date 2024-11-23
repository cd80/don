import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import websockets
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.binance_fetcher import BinanceFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.base_model import BaseModel, AttentionLayer
from src.utils.helpers import DataNormalizer, ReplayBuffer
from src.training.trainer import TradingEnvironment

class TestBinanceFetcher(unittest.TestCase):
    """Test cases for BinanceFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = BinanceFetcher(
            symbol="BTCUSDT",
            interval="5m",
            start_date="2023-01-01",
            end_date="2023-01-02"
        )
    
    def test_generate_urls(self):
        """Test URL generation for data download."""
        urls = self.fetcher._generate_urls(
            datetime(2023, 1, 1),
            datetime(2023, 1, 2)
        )
        self.assertTrue(len(urls) > 0)
        self.assertTrue(all('BTCUSDT' in url for url in urls))
        self.assertTrue(all('5m' in url for url in urls))
    
    def test_process_klines(self):
        """Test klines data processing."""
        # Create sample data
        sample_data = b"1609459200000,29000.0,29100.0,28900.0,29050.0,100.0,1609459499999,2900000.0,50,60.0,1740000.0,0"
        
        df = self.fetcher._process_klines(sample_data)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('open_time' in df.columns)
        self.assertTrue('close' in df.columns)
        self.assertEqual(df['close'].dtype, np.float64)

class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='5min')
        self.sample_data = pd.DataFrame({
            'open_time': dates,
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.abs(np.random.randn(len(dates))),
            'close_time': dates + timedelta(minutes=5),
            'quote_volume': np.abs(np.random.randn(len(dates))),
            'trades': np.abs(np.random.randn(len(dates))),
            'taker_buy_volume': np.abs(np.random.randn(len(dates))),
            'taker_buy_quote_volume': np.abs(np.random.randn(len(dates)))
        })
        
        # Save sample data
        os.makedirs('data/raw', exist_ok=True)
        self.sample_data.to_parquet('data/raw/test_data.parquet')
        
        self.engineer = FeatureEngineer(
            input_file='data/raw/test_data.parquet',
            output_dir='data/processed'
        )
    
    def test_calculate_statistical_features(self):
        """Test statistical feature calculation."""
        features = self.engineer._calculate_statistical_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertTrue(any('mean' in col for col in features.columns))
        self.assertTrue(any('std' in col for col in features.columns))
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation."""
        features = self.engineer._calculate_technical_indicators(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertTrue(len(features.columns) > len(self.sample_data.columns))
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists('data/raw/test_data.parquet'):
            os.remove('data/raw/test_data.parquet')

class TestBaseModel(unittest.TestCase):
    """Test cases for BaseModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.model = BaseModel(
            state_dim=100,
            action_dim=1,
            hidden_dim=64,
            num_options=4,
            num_heads=4,
            device=self.device
        )
        self.model.to(self.device)
    
    def test_attention_layer(self):
        """Test attention layer forward pass."""
        attention = AttentionLayer(
            input_dim=64,
            num_heads=4
        ).to(self.device)
        
        x = torch.randn(32, 10, 64, device=self.device)  # (batch_size, seq_length, input_dim)
        output, weights = attention(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(weights.shape[1], 4)  # num_heads
    
    def test_model_forward(self):
        """Test model forward pass."""
        batch_size = 32
        seq_length = 10
        state = torch.randn(batch_size, seq_length, 100, device=self.device)
        
        option_probs, selected_option, action_dist, value = self.model(state)
        
        self.assertEqual(option_probs.shape, (batch_size * seq_length, 4))
        self.assertEqual(value.shape, (batch_size * seq_length, 1))

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_data_normalizer(self):
        """Test data normalization."""
        data = np.random.randn(1000, 10)
        normalizer = DataNormalizer(method='standard')
        
        normalizer.fit(data)
        normalized = normalizer.transform(data)
        recovered = normalizer.inverse_transform(normalized)
        
        np.testing.assert_array_almost_equal(data, recovered, decimal=10)
    
    def test_replay_buffer(self):
        """Test replay buffer operations."""
        buffer = ReplayBuffer(
            capacity=1000,
            state_dim=10,
            action_dim=1
        )
        
        # Add experience
        state = torch.randn(10)
        action = torch.randn(1)
        next_state = torch.randn(10)
        
        buffer.add(state, action, 1.0, next_state, False)
        
        self.assertEqual(buffer.size, 1)
        
        # Test sampling when buffer is not full
        with self.assertRaises(ValueError):
            buffer.sample(32)

class TestTradingEnvironment(unittest.TestCase):
    """Test cases for TradingEnvironment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='5min')
        data = {
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.abs(np.random.randn(len(dates))),
            'quote_volume': np.abs(np.random.randn(len(dates))),
            'trades': np.abs(np.random.randn(len(dates))),
            'taker_buy_volume': np.abs(np.random.randn(len(dates))),
            'taker_buy_quote_volume': np.abs(np.random.randn(len(dates)))
        }
        self.sample_data = pd.DataFrame(data, index=dates)
        
        self.env = TradingEnvironment(
            data=self.sample_data,
            initial_balance=100000,
            transaction_fee=0.001
        )
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        self.assertEqual(self.env.balance, 100000)
        self.assertEqual(self.env.position, 0)
        self.assertIsInstance(state, torch.Tensor)
    
    def test_step(self):
        """Test environment step."""
        self.env.reset()
        action = torch.tensor([0.5])  # 50% long position
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(next_state, torch.Tensor)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertTrue('balance' in info)
        self.assertTrue('position' in info)

if __name__ == '__main__':
    unittest.main()
