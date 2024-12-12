import pytest
import numpy as np
import pandas as pd
from don.rl.env import TradingEnvironment
from don.rl.actions import DiscreteActionSpace, ContinuousActionSpace
from don.rl.rewards import PnLReward, SharpeReward

@pytest.fixture
def mock_market_data():
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1h')
    data = pd.DataFrame({
        'close': np.linspace(100, 200, 200) + np.random.normal(0, 5, 200),
        'volume': np.random.uniform(1000, 2000, 200),
        'trades': np.random.randint(100, 200, 200)
    }, index=dates)
    return data

@pytest.fixture
def discrete_env(mock_market_data):
    action_space = DiscreteActionSpace([-1.0, -0.5, 0.0, 0.5, 1.0])
    return TradingEnvironment(
        data=mock_market_data,
        action_space=action_space,
        window_size=10
    )

@pytest.fixture
def continuous_env(mock_market_data):
    action_space = ContinuousActionSpace(-0.5, 0.5)
    return TradingEnvironment(
        data=mock_market_data,
        action_space=action_space,
        window_size=10
    )

def test_environment_initialization(mock_market_data):
    discrete_space = DiscreteActionSpace([-1.0, -0.5, 0.0, 0.5, 1.0])
    env = TradingEnvironment(
        data=mock_market_data,
        action_space=discrete_space,
        window_size=10
    )
    assert env.window_size == 10
    assert isinstance(env.action_space_handler, DiscreteActionSpace)
    assert env.commission == 0.001

    continuous_space = ContinuousActionSpace(-0.5, 0.5)
    env = TradingEnvironment(
        data=mock_market_data,
        action_space=continuous_space,
        window_size=20,
        commission=0.002
    )
    assert env.window_size == 20
    assert isinstance(env.action_space_handler, ContinuousActionSpace)
    assert env.commission == 0.002

def test_environment_reset(discrete_env):
    observation, info = discrete_env.reset()

    assert observation.shape == (10, 3)
    assert info['step'] == 10
    assert info['position'] == 0.0
    assert info['pnl'] == 0.0
    assert info['trade_count'] == 0

def test_discrete_environment_step(discrete_env):
    discrete_env.reset()

    obs, reward, done, truncated, info = discrete_env.step(2)
    assert obs.shape == (10, 3)
    assert not done
    assert not truncated
    assert info['position'] == 0.0

    obs, reward, done, truncated, info = discrete_env.step(4)
    assert info['position'] == 1.0
    assert len(info['trades']) == 1

def test_continuous_environment_step(continuous_env):
    continuous_env.reset()

    obs, reward, done, truncated, info = continuous_env.step(0.0)
    assert obs.shape == (10, 3)
    assert not done
    assert info['position'] == 0.0

    obs, reward, done, truncated, info = continuous_env.step(0.3)
    assert info['position'] == 0.3
    assert len(info['trades']) == 1

def test_environment_done_condition(discrete_env):
    discrete_env.reset()

    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = discrete_env.step(2)
        steps += 1

    assert steps == len(discrete_env.data) - discrete_env.window_size - 1

def test_custom_reward_integration(mock_market_data):
    env = TradingEnvironment(
        data=mock_market_data,
        action_space=DiscreteActionSpace([-1.0, 0.0, 1.0]),
        reward_calculator=SharpeReward(window=10),
        window_size=10
    )

    env.reset()
    _, reward, _, _, _ = env.step(2)
    assert isinstance(reward, float)
