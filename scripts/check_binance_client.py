from binance.client import Client
from binance.streams import BinanceWebSocketApiManager
from binance.exceptions import BinanceAPIException
import inspect
from unittest.mock import Mock, call

def main():
    """Inspect Binance Client and SocketManager implementation."""
    # Print Client attributes
    print('Client attributes:')
    for attr in dir(Client):
        if not attr.startswith('_'):
            print(f'  {attr}')

    # Print BinanceWebSocketApiManager initialization
    print('\nBinanceWebSocketApiManager.__init__:')
    try:
        print(inspect.getsource(BinanceWebSocketApiManager.__init__))
    except (TypeError, OSError):
        print('Source not available, printing signature:')
        print(inspect.signature(BinanceWebSocketApiManager.__init__))

    # Create mock client with basic attributes
    mock_client = Mock(spec=Client)
    mock_client.tld = 'com'
    mock_client.API_URL = 'https://api.binance.com'
    mock_client.STREAM_URL = 'wss://stream.binance.com:9443'
    mock_client.STREAM_API_URL = 'wss://stream.binance.com:9443/ws'
    mock_client.STREAM_TESTNET_URL = 'wss://testnet.binance.vision/ws'

    # Try to initialize socket manager
    try:
        bsm = BinanceWebSocketApiManager(mock_client)
        print('\nSuccessfully created BinanceWebSocketApiManager')
        print('Used attributes:', mock_client.mock_calls)
    except Exception as e:
        print('\nException during initialization:')
        print(f'{type(e).__name__}: {str(e)}')
        print('Mock calls before error:', mock_client.mock_calls)

if __name__ == "__main__":
    main()
