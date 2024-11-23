"""
Advanced feature engineering module with parallel processing capabilities.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import os
from src.data.sentiment_analyzer import SentimentAnalyzer

class FeatureEngineer:
    """
    Advanced feature engineering with parallel processing capabilities.
    Generates technical indicators, statistical features, and custom features
    for the trading model.
    """
    
    def __init__(
        self,
        input_file: str,
        output_dir: str = "data/processed",
        n_jobs: int = -1,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the FeatureEngineer.
        
        Args:
            input_file: Path to input parquet file
            output_dir: Directory to save processed features
            n_jobs: Number of parallel jobs (-1 for all cores)
            config: Configuration dictionary for feature parameters
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.config = config or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(config.get('sentiment', {}))
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from parquet file.
        
        Returns:
            DataFrame with raw data
        """
        df = pd.read_parquet(self.input_file)
        # Ensure open_time is datetime
        df['open_time'] = pd.to_datetime(df['open_time'])
        return df
    
    def _calculate_statistical_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 50, 100]
    ) -> pd.DataFrame:
        """
        Calculate statistical features using different window sizes.
        
        Args:
            df: Input DataFrame
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with statistical features
        """
        if df.empty or 'close' not in df.columns:
            return pd.DataFrame(index=df.index)
            
        features = pd.DataFrame(index=df.index)
        
        # Price based features
        for window in windows:
            # Rolling statistics
            features[f'price_mean_{window}'] = df['close'].rolling(window, min_periods=1).mean()
            features[f'price_std_{window}'] = df['close'].rolling(window, min_periods=1).std()
            features[f'price_zscore_{window}'] = ((df['close'] - features[f'price_mean_{window}']) / 
                                                (features[f'price_std_{window}'] + 1e-8))
            
            # Price momentum
            features[f'price_mom_{window}'] = df['close'].pct_change(window)
            
            # Volume features
            features[f'volume_mean_{window}'] = df['volume'].rolling(window, min_periods=1).mean()
            features[f'volume_std_{window}'] = df['volume'].rolling(window, min_periods=1).std()
            features[f'volume_zscore_{window}'] = ((df['volume'] - features[f'volume_mean_{window}']) /
                                                 (features[f'volume_std_{window}'] + 1e-8))
            
            # Volatility
            log_returns = np.log(df['close'] / df['close'].shift(1))
            features[f'volatility_{window}'] = log_returns.rolling(window, min_periods=1).std() * np.sqrt(252 * 288)  # Annualized
            
            # Price range
            features[f'price_range_{window}'] = ((df['high'].rolling(window, min_periods=1).max() - 
                                                df['low'].rolling(window, min_periods=1).min()) /
                                               (df['close'] + 1e-8))
        
        return features
    
    def _calculate_orderbook_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate order book derived features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with order book features
        """
        if df.empty or not all(col in df.columns for col in ['volume', 'taker_buy_volume', 'quote_volume', 'taker_buy_quote_volume', 'trades']):
            return pd.DataFrame(index=df.index)
            
        features = pd.DataFrame(index=df.index)
        
        # Volume-based features
        features['buy_volume_ratio'] = np.clip(
            df['taker_buy_volume'] / (df['volume'] + 1e-8),
            0, 1
        )
        features['buy_quote_ratio'] = np.clip(
            df['taker_buy_quote_volume'] / (df['quote_volume'] + 1e-8),
            0, 1
        )
        
        # Trade size features
        features['avg_trade_size'] = df['volume'] / (df['trades'] + 1e-8)
        features['avg_trade_quote_size'] = df['quote_volume'] / (df['trades'] + 1e-8)
        
        return features
    
    def _calculate_time_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        if df.empty or 'open_time' not in df.columns:
            return pd.DataFrame(index=df.index)
            
        features = pd.DataFrame(index=df.index)
        
        # Extract time components
        timestamps = pd.to_datetime(df['open_time'])
        features['hour'] = timestamps.dt.hour
        features['day_of_week'] = timestamps.dt.dayofweek
        features['day_of_month'] = timestamps.dt.day
        features['week_of_year'] = timestamps.dt.isocalendar().week
        features['month'] = timestamps.dt.month
        features['quarter'] = timestamps.dt.quarter
        
        # Create cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features
    
    def _calculate_technical_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate technical indicators using the TA-Lib library.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return pd.DataFrame(index=df.index)
            
        if len(df) < 100:  # Minimum data points needed for indicators
            return pd.DataFrame(index=df.index)
            
        # Create copy of dataframe to avoid modifying original
        df_indicators = df.copy()
        
        try:
            # Add all technical analysis features
            df_indicators = add_all_ta_features(
                df_indicators,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=True
            )
        except Exception as e:
            self.logger.warning(f"Error calculating technical indicators: {e}")
            return pd.DataFrame(index=df.index)
        
        return df_indicators

    async def _calculate_sentiment_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate sentiment-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with sentiment features
        """
        if df.empty or 'open_time' not in df.columns:
            return pd.DataFrame(index=df.index)
            
        self.logger.info("Calculating sentiment features...")
        
        try:
            # Get timestamps for sentiment analysis
            start_time = df['open_time'].min()
            end_time = df['open_time'].max()
            
            # Fetch sentiment data
            news_sentiment = await self.sentiment_analyzer.fetch_news_sentiment(
                start_time,
                end_time
            )
            
            social_sentiment = await self.sentiment_analyzer.fetch_social_sentiment(
                start_time,
                end_time
            )
            
            # Aggregate sentiment data
            sentiment_df = self.sentiment_analyzer.aggregate_sentiment(
                news_sentiment,
                social_sentiment,
                window=self.config.get('sentiment', {}).get('analysis', {}).get('window_size', '1h')
            )
            
            # Calculate additional sentiment features
            sentiment_features = self.sentiment_analyzer.calculate_sentiment_features(sentiment_df)
            
            # Resample to match price data frequency and forward fill missing values
            resampled_features = sentiment_features.set_index('timestamp').resample('5min').ffill()
            
            # Reset index and rename to match price data
            resampled_features.reset_index(inplace=True)
            resampled_features.rename(columns={'timestamp': 'open_time'}, inplace=True)
            
            # Merge with original data on timestamp
            merged_features = pd.merge(
                df[['open_time']],
                resampled_features,
                on='open_time',
                how='left'
            )
            
            # Forward fill and backward fill missing values
            merged_features = merged_features.ffill().bfill()
            
            return merged_features.drop('open_time', axis=1)
            
        except Exception as e:
            self.logger.warning(f"Error calculating sentiment features: {e}")
            return pd.DataFrame(index=df.index)
    
    def _parallel_feature_calculation(
        self,
        feature_func: callable,
        df: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate features in parallel using specified function.
        
        Args:
            feature_func: Function to calculate features
            df: Input DataFrame
            **kwargs: Additional arguments for feature function
            
        Returns:
            DataFrame with calculated features
        """
        if df.empty:
            return pd.DataFrame()
            
        # Split DataFrame into chunks for parallel processing
        n_splits = min(os.cpu_count() or 1, max(1, len(df) // 1000))
        chunks = np.array_split(df, n_splits)
        
        # Calculate features in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(feature_func)(chunk, **kwargs)
            for chunk in chunks
        )
        
        # Combine results
        return pd.concat(results) if results else pd.DataFrame()
    
    async def generate_features(self) -> None:
        """
        Generate all features and save to parquet file.
        """
        self.logger.info("Starting feature generation...")
        
        try:
            # Calculate different feature sets in parallel
            statistical_features = self._parallel_feature_calculation(
                self._calculate_statistical_features,
                self.data
            )
            
            orderbook_features = self._parallel_feature_calculation(
                self._calculate_orderbook_features,
                self.data
            )
            
            time_features = self._calculate_time_features(self.data)
            
            technical_features = self._parallel_feature_calculation(
                self._calculate_technical_indicators,
                self.data
            )
            
            # Calculate sentiment features (not parallelized due to API rate limits)
            sentiment_features = await self._calculate_sentiment_features(self.data)
            
            # Combine all features
            feature_dfs = [
                self.data,  # Original features
                statistical_features,
                orderbook_features,
                time_features,
                technical_features,
                sentiment_features
            ]
            
            # Filter out empty DataFrames
            valid_dfs = [df for df in feature_dfs if not df.empty]
            
            if not valid_dfs:
                raise ValueError("No valid features generated")
                
            all_features = pd.concat(valid_dfs, axis=1)
            
            # Remove any duplicate columns
            all_features = all_features.loc[:, ~all_features.columns.duplicated()]
            
            # Sort by timestamp
            all_features = all_features.sort_values('open_time')
            
            # Save to parquet
            output_file = os.path.join(
                self.output_dir,
                f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
            
            table = pa.Table.from_pandas(all_features)
            pq.write_table(
                table,
                output_file,
                compression='snappy'
            )
            
            self.logger.info(f"Features successfully saved to {output_file}")
            self.logger.info(f"Generated {len(all_features.columns)} features")
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            raise
