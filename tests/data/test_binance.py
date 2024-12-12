import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance import BinanceSocketManager
from don.data.binance import BinanceDataCollector
from don.data.base import DataCollector

@pytest.fixture
def mock_binance_client():
    client = Mock(spec=Client)

    # Mock futures_recent_trades
    client.futures_recent_trades.return_value = [
        {
            'id': i,
            'price': '50000.0',
            'qty': '1.0',
            'time': 1609459200000 + i * 1000,
            'isBuyerMaker': False
        } for i in range(10)
    ]

    # Mock futures_order_book
    client.futures_order_book.return_value = {
        'bids': [['49900.0', '1.0'] for _ in range(5)],
        'asks': [['50100.0', '1.0'] for _ in range(5)]
    }

    # Mock futures_liquidation_orders
    client.futures_liquidation_orders.return_value = [
        {
            'price': '50000.0',
            'qty': '1.0',
            'time': 1609459200000,
            'side': 'BUY'
        }
    ]

    # Mock futures_klines
    client.futures_klines.return_value = [
        [1609459200000, '50000.0', '50100.0', '49900.0', '50000.0', '100.0',
         1609462800000, '5000000.0', 1000, '50.0', '2500000.0', '0.0']
        for _ in range(10)
    ]

    client.tld = 'com'
    return client

@pytest.fixture
def mock_socket_manager():
    socket_manager = Mock()
    socket_manager.start_trade_socket = Mock(return_value="trade_socket")
    return socket_manager

def test_binance_collector_initialization():
    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test_key",
        api_secret="test_secret"
    )
    assert isinstance(collector, DataCollector)
    assert collector.symbol == "BTCUSDT"

@patch('don.data.binance.Client')
def test_historical_data_collection(mock_client_class, mock_binance_client):
    mock_client_class.return_value = mock_binance_client

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test_key",
        api_secret="test_secret"
    )

    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)

    data = collector.get_historical_data(
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
def test_realtime_data_collection(mock_socket_manager_class, mock_client_class,
                                mock_binance_client, mock_socket_manager):
    mock_client_class.return_value = mock_binance_client
    mock_socket_manager_class.return_value = mock_socket_manager

    collector = BinanceDataCollector(
        symbol="BTCUSDT",
        api_key="test_key",
        api_secret="test_secret"
    )

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

    collector._handle_trade_socket(trade_data)

    assert mock_socket_manager.start_trade_socket.called
    assert mock_socket_manager.start_trade_socket.call_args[0][0] == "BTCUSDT"

def test_error_handling():
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

def test_data_validation(mock_binance_client):
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
