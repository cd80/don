import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq

class BinanceFetcher:
    """
    Parallel data fetcher for Binance historical data with robust error handling
    and efficient storage.
    """
    
    BASE_URL = "https://data.binance.vision/data"
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "5m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_workers: int = 8,
        output_dir: str = "data/raw",
    ):
        """
        Initialize the BinanceFetcher.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval for klines
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            max_workers: Maximum number of parallel workers
            output_dir: Directory to save downloaded data
        """
        self.symbol = symbol.upper()
        self.interval = interval
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        self.max_workers = max_workers
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        retry_count: int = 3,
        delay: int = 1
    ) -> Optional[bytes]:
        """
        Download a single file with retry logic.
        
        Args:
            session: aiohttp client session
            url: URL to download from
            retry_count: Number of retries on failure
            delay: Delay between retries in seconds
            
        Returns:
            Downloaded data as bytes or None if failed
        """
        for attempt in range(retry_count):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 404:
                        self.logger.warning(f"File not found: {url}")
                        return None
                    else:
                        self.logger.warning(
                            f"Failed to download {url}. "
                            f"Status: {response.status}"
                        )
            except Exception as e:
                self.logger.error(f"Error downloading {url}: {str(e)}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(delay * (attempt + 1))
                continue
        return None

    async def _parallel_download(
        self,
        urls: List[str]
    ) -> List[Optional[bytes]]:
        """
        Download multiple files in parallel.
        
        Args:
            urls: List of URLs to download
            
        Returns:
            List of downloaded data as bytes
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._download_file(session, url)
                for url in urls
            ]
            return await asyncio.gather(*tasks)

    def _process_klines(
        self,
        data: bytes
    ) -> pd.DataFrame:
        """
        Process raw klines data into a pandas DataFrame.
        
        Args:
            data: Raw klines data in bytes
            
        Returns:
            Processed DataFrame
        """
        df = pd.read_csv(pd.io.common.BytesIO(data))
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert string columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                         'quote_volume', 'taker_buy_volume',
                         'taker_buy_quote_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df

    def _save_to_parquet(
        self,
        df: pd.DataFrame,
        filename: str
    ) -> None:
        """
        Save DataFrame to parquet format with compression.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            filename,
            compression='snappy'
        )

    def _generate_urls(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[str]:
        """
        Generate URLs for data download based on date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of URLs to download
        """
        urls = []
        current_date = start_date
        
        while current_date <= end_date:
            # Format URL based on interval
            if self.interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h']:
                url = (f"{self.BASE_URL}/spot/monthly/klines/"
                      f"{self.symbol}/{self.interval}/"
                      f"{self.symbol}-{self.interval}-"
                      f"{current_date.strftime('%Y-%m')}.zip")
            else:
                url = (f"{self.BASE_URL}/spot/daily/klines/"
                      f"{self.symbol}/{self.interval}/"
                      f"{self.symbol}-{self.interval}-"
                      f"{current_date.strftime('%Y-%m-%d')}.zip")
            
            urls.append(url)
            
            # Increment date based on interval
            if self.interval in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h']:
                current_date = (current_date.replace(day=1) + 
                              timedelta(days=32)).replace(day=1)
            else:
                current_date += timedelta(days=1)
        
        return urls

    async def fetch_data(self) -> None:
        """
        Main method to fetch and process data in parallel.
        """
        # Generate URLs for the date range
        urls = self._generate_urls(self.start_date, self.end_date)
        
        self.logger.info(f"Starting download of {len(urls)} files...")
        
        # Download files in parallel
        raw_data = await self._parallel_download(urls)
        
        # Process downloaded data
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            dfs = list(executor.map(
                lambda x: self._process_klines(x) if x is not None else None,
                raw_data
            ))
        
        # Combine all DataFrames
        df = pd.concat([df for df in dfs if df is not None])
        
        # Sort by timestamp and remove duplicates
        df = df.sort_values('open_time').drop_duplicates()
        
        # Save to parquet
        output_file = os.path.join(
            self.output_dir,
            f"{self.symbol}_{self.interval}_data.parquet"
        )
        self._save_to_parquet(df, output_file)
        
        self.logger.info(f"Data successfully saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    fetcher = BinanceFetcher(
        symbol="BTCUSDT",
        interval="5m",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Run the async fetcher
    asyncio.run(fetcher.fetch_data())
