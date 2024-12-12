import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.websockets import BinanceSocketManager
from don.data.binance import BinanceDataCollector
from don.data.base import DataCollector

@pytest.fixture
def mock_binance_client():
    client = Mock(spec=Client)

    # Mock client attributes
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
def mock_socket_manager():
    mock = AsyncMock()
    mock.start_trade_socket.return_value = "ws_connection"
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
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
@patch('don.data.binance.BinanceSocketManager')
@pytest.mark.asyncio
async def test_realtime_data_collection(mock_socket_manager_class, mock_client_class,
                                        mock_binance_client, mock_socket_manager):
    mock_client_class.return_value = mock_binance_client
    mock_socket_manager_class.return_value = mock_socket_manager

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test_key",
        api_secret="test_secret"
    )

    await collector.start_realtime_collection()
    mock_socket_manager.start_trade_socket.assert_called_once_with(
        "BTCUSDT",
        collector._handle_trade_socket
    )
    mock_socket_manager.start.assert_called_once()

    await collector.stop_realtime_collection()
    mock_socket_manager.stop.assert_called_once()

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
    mock_client.tld = 'com'
    mock_client.API_URL = 'https://api.binance.com'
    mock_client.STREAM_URL = 'wss://stream.binance.com:9443'
    mock_client.STREAM_API_URL = 'wss://stream.binance.com:9443/ws'
    mock_client.STREAM_TESTNET_URL = 'wss://testnet.binance.vision/ws'
    mock_client.ping.side_effect = BinanceAPIException(
        mock_response,  # response must be first
        401,  # status_code second
        error_text  # text third
    )
    mock_client_class.return_value = mock_client

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="invalid_key",
        api_secret="invalid_secret"
    )

    with pytest.raises(ValueError):
        collector.get_historical_data(
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            interval="invalid"
        )

    with pytest.raises(ValueError):
        BinanceDataCollector(
            symbol="INVALID",
            api_key="test_key",
            api_secret="test_secret"
        )

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
