from binance.client import Client
from binance import Client as C2
import binance

print('Available modules:', dir(binance))
print('\nClient attributes:', [attr for attr in dir(Client) if 'socket' in attr.lower()])
