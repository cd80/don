import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client, AsyncClient
from binance.exceptions import BinanceAPIException
from don.data.binance import BinanceDataCollector
from don.data.base import DataCollector

@pytest.fixture
def mock_binance_client():
    client = Mock(spec=Client)

    # Mock client attributes
    client.API_KEY = "test_key"
    client.API_SECRET = "test_secret"
    client.testnet = False
    client.tld = 'com'
    client.API_URL = 'https://api.binance.com'
    client.STREAM_URL = 'wss://stream.binance.com:9443'
    client.STREAM_API_URL = 'wss://stream.binance.com:9443/ws'
    client.STREAM_TESTNET_URL = 'wss://testnet.binance.vision/ws'

    # Mock futures API methods
    client.futures_recent_trades.return_value = [
        {
            'id': i,
            'price': '50000.0',
            'qty': '1.0',
            'time': 1609459200000 + i * 1000,
            'isBuyerMaker': False
        } for i in range(10)
    ]

    client.futures_order_book.return_value = {
        'lastUpdateId': 1,
        'bids': [['49900.0', '1.0'] for _ in range(5)],
        'asks': [['50100.0', '1.0'] for _ in range(5)]
    }

    client.futures_klines.return_value = [
        [1609459200000, '50000.0', '50100.0', '49900.0', '50000.0', '100.0',
         1609462800000, '5000000.0', 1000, '50.0', '2500000.0', '0.0']
        for _ in range(10)
    ]

    return client

@pytest.fixture
def mock_async_client():
    """Mock AsyncClient."""
    mock = AsyncMock()
    mock.futures_stream_get_listen_key = AsyncMock(return_value="test_listen_key")
    mock.close_connection = AsyncMock()
    return mock

@patch('don.data.binance.Client')
def test_binance_collector_initialization(mock_client_class, mock_binance_client):
    mock_client_class.return_value = mock_binance_client

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test",
        api_secret="test"
    )
    assert collector.client is not None
    assert collector.client.tld == 'com'

@patch('don.data.binance.Client')
@pytest.mark.asyncio
async def test_historical_data_collection(mock_client_class, mock_binance_client):
    mock_client_class.return_value = mock_binance_client

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test_key",
        api_secret="test_secret"
    )

    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)

    data = await collector.get_historical_data(
        start_time=start_time,
        end_time=end_time,
        interval="1h"
    )

    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in [
        'open_time', 'open', 'high', 'low', 'close',
        'volume', 'close_time', 'quote_volume', 'trades'
    ])

@patch('don.data.binance.Client')
@patch('don.data.binance.AsyncClient')
@pytest.mark.asyncio
async def test_realtime_data_collection(mock_async_client_class, mock_client_class,
                                        mock_binance_client, mock_async_client):
    mock_client_class.return_value = mock_binance_client

    async def mock_create(*args, **kwargs):
        return mock_async_client

    mock_async_client_class.create = mock_create

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test_key",
        api_secret="test_secret"
    )

    await collector.start_realtime_collection()
    mock_async_client.futures_stream_get_listen_key.assert_called_once()
    mock_async_client.close_connection.assert_not_called()

    await collector.stop_realtime_collection()
    mock_async_client.close_connection.assert_called_once()

    trade_data = {
        "e": "trade",
        "E": 1499404907056,
        "s": "BTCUSDT",
        "t": 12345,
        "p": "8100.0",
        "q": "1.0",
        "b": 88,
        "a": 50,
        "T": 1499404907056,
        "m": True,
        "M": True
    }

    await collector._handle_trade_socket(trade_data)

@patch('don.data.binance.Client')
def test_error_handling(mock_client_class):
    error_text = '{"code":-2015,"msg":"Invalid API-key, IP, or permissions for action."}'
    mock_response = Mock(status_code=401)
    mock_client = Mock(spec=Client)
    mock_client.futures_recent_trades = Mock(side_effect=BinanceAPIException(
        mock_response,  # response must be first
        401,  # status_code second
        error_text  # text third
    ))
    mock_client_class.return_value = mock_client

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="invalid_key",
        api_secret="invalid_secret"
    )

    with pytest.raises(ValueError):
        collector.collect_trades(symbol="BTCUSDT")

@patch('don.data.binance.Client')
def test_data_validation(mock_client_class, mock_binance_client):
    mock_client_class.return_value = mock_binance_client

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test_key",
        api_secret="test_secret"
    )

    timestamp = collector._convert_timestamp(1499040000000)
    assert isinstance(timestamp, pd.Timestamp)

    raw_kline = [
        1499040000000,
        "8100.0",
        "8200.0",
        "8000.0",
        "8150.0",
        "10.5",
        1499644799999,
        "85000.0",
        100,
        "5.0",
        "42000.0",
        "0"
    ]
    processed = collector._process_kline(raw_kline)
    assert isinstance(processed['open'], float)
    assert isinstance(processed['volume'], float)
    assert isinstance(processed['trades'], int)
