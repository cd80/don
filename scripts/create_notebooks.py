import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_data_collection_notebook():
    nb = new_notebook()

    nb.cells = [
        new_markdown_cell('# Bitcoin Futures Data Collection Example\n\nThis notebook demonstrates how to use the Don framework to collect historical and real-time data from Binance Futures.'),

        new_code_cell('''import pandas as pd
from datetime import datetime, timedelta
from don.data.binance import BinanceDataCollector'''),

        new_markdown_cell('## Initialize Data Collector\n\nFirst, create a BinanceDataCollector instance. You\'ll need your Binance API key and secret.'),

        new_code_cell('''# Initialize collector (replace with your API credentials)
collector = BinanceDataCollector(
    symbol='BTCUSDT',
    api_key='your_api_key',
    api_secret='your_api_secret'
)'''),

        new_markdown_cell('## Fetch Historical Data\n\nGet historical klines data for the past week with 1-hour intervals.'),

        new_code_cell('''# Define time range
end_time = datetime.now()
start_time = end_time - timedelta(days=7)

# Fetch historical data
historical_data = collector.get_historical_data(
    start_time=start_time,
    end_time=end_time,
    interval='1h'
)

# Display first few rows
historical_data.head()'''),

        new_markdown_cell('## Real-time Data Collection\n\nSet up real-time trade data collection using WebSocket connection.'),

        new_code_cell('''# Start real-time data collection
def handle_trade(trade_data):
    print(f"New trade: Price={trade_data['price']}, Volume={trade_data['volume']}")

collector.start_trade_stream(callback=handle_trade)'''),

        new_markdown_cell('## Data Analysis\n\nBasic analysis of the collected historical data.'),

        new_code_cell('''# Calculate basic statistics
print("Data Statistics:")
print(historical_data[['close', 'volume', 'trades']].describe())

# Plot price and volume
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Price plot
ax1.plot(historical_data.index, historical_data['close'])
ax1.set_title('BTC/USDT Price')
ax1.set_ylabel('Price (USDT)')

# Volume plot
ax2.bar(historical_data.index, historical_data['volume'])
ax2.set_title('Trading Volume')
ax2.set_ylabel('Volume (BTC)')

plt.tight_layout()
plt.show()''')
    ]

    return nb

def create_feature_calculation_notebook():
    nb = new_notebook()

    nb.cells = [
        new_markdown_cell('# Technical Indicator Calculation Example\n\nThis notebook demonstrates how to calculate various technical indicators using the Don framework.'),

        new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from don.features.technical import TechnicalIndicators
from don.data.binance import BinanceDataCollector'''),

        new_markdown_cell('## Load Historical Data\n\nFirst, we\'ll load some historical data to calculate indicators.'),

        new_code_cell('''# Load historical data
collector = BinanceDataCollector(
    symbol='BTCUSDT',
    api_key='your_api_key',
    api_secret='your_api_secret'
)

# Get one month of hourly data
end_time = pd.Timestamp.now()
start_time = end_time - pd.Timedelta(days=30)
data = collector.get_historical_data(
    start_time=start_time,
    end_time=end_time,
    interval='1h'
)'''),

        new_markdown_cell('## Calculate Technical Indicators\n\nNow we\'ll calculate various technical indicators.'),

        new_code_cell('''# Initialize technical indicators
indicators = TechnicalIndicators()

# Calculate all indicators
result = indicators.calculate(data)

# Display available indicators
print("Available indicators:", list(result.columns))
result.head()'''),

        new_markdown_cell('## Visualize Indicators\n\nLet\'s plot some common technical indicators.'),

        new_code_cell('''# Plot price with SMA and Bollinger Bands
plt.figure(figsize=(12, 8))
plt.plot(result.index, result['close'], label='Price', alpha=0.7)
plt.plot(result.index, result['sma_20'], label='SMA(20)', alpha=0.7)
plt.plot(result.index, result['bb_upper'], '--', label='BB Upper', alpha=0.5)
plt.plot(result.index, result['bb_lower'], '--', label='BB Lower', alpha=0.5)
plt.title('Price with SMA and Bollinger Bands')
plt.legend()
plt.show()

# Plot RSI
plt.figure(figsize=(12, 4))
plt.plot(result.index, result['rsi'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
plt.title('Relative Strength Index (RSI)')
plt.legend()
plt.show()

# Plot MACD
plt.figure(figsize=(12, 4))
plt.plot(result.index, result['macd'], label='MACD')
plt.plot(result.index, result['macd_signal'], label='Signal')
plt.bar(result.index, result['macd_hist'], label='Histogram', alpha=0.3)
plt.title('MACD')
plt.legend()
plt.show()''')
    ]

    return nb

def create_rl_training_notebook():
    nb = new_notebook()

    nb.cells = [
        new_markdown_cell('# Reinforcement Learning Training Example\n\nThis notebook demonstrates how to set up and train an RL agent for Bitcoin futures trading.'),

        new_code_cell('''import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from don.rl.env import TradingEnvironment
from don.rl.actions import DiscreteActionSpace
from don.rl.rewards import SharpeReward
from don.data.binance import BinanceDataCollector'''),

        new_markdown_cell('## Load Training Data\n\nFirst, we\'ll load historical data for training.'),

        new_code_cell('''# Load training data
collector = BinanceDataCollector(
    symbol='BTCUSDT',
    api_key='your_api_key',
    api_secret='your_api_secret'
)

# Get three months of hourly data
end_time = pd.Timestamp.now()
start_time = end_time - pd.Timedelta(days=90)
training_data = collector.get_historical_data(
    start_time=start_time,
    end_time=end_time,
    interval='1h'
)'''),

        new_markdown_cell('## Set Up Trading Environment\n\nCreate the trading environment with discrete actions and Sharpe ratio reward.'),

        new_code_cell('''# Define action space
action_space = DiscreteActionSpace([-1.0, -0.5, 0.0, 0.5, 1.0])

# Create environment
env = TradingEnvironment(
    data=training_data,
    action_space=action_space,
    reward_calculator=SharpeReward(window=20),
    window_size=10
)

# Test environment
observation, info = env.reset()
print("Observation shape:", observation.shape)
print("Action space:", env.action_space)'''),

        new_markdown_cell('## Implement Simple DQN Agent\n\nCreate a basic DQN agent for training.'),

        new_code_cell('''class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Initialize DQN
input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
output_dim = len(action_space.positions)
dqn = DQN(input_dim, output_dim)
optimizer = torch.optim.Adam(dqn.parameters())'''),

        new_markdown_cell('## Training Loop\n\nTrain the agent for a few episodes.'),

        new_code_cell('''def train_episode():
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0)

        # Get action from network
        with torch.no_grad():
            q_values = dqn(obs_tensor)
            action = q_values.argmax().item()

        # Take action in environment
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Move to next observation
        obs = next_obs

    return total_reward

# Train for 10 episodes
for episode in range(10):
    reward = train_episode()
    print(f"Episode {episode + 1}, Total Reward: {reward:.2f}")''')
    ]

    return nb

if __name__ == '__main__':
    # Create all notebooks
    notebooks = {
        '../examples/data_collection.ipynb': create_data_collection_notebook(),
        '../examples/feature_calculation.ipynb': create_feature_calculation_notebook(),
        '../examples/rl_training.ipynb': create_rl_training_notebook()
    }

    # Write notebooks
    for path, nb in notebooks.items():
        with open(path, 'w') as f:
            nbf.write(nb, f)
