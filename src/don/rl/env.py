from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
import numpy as np
import pandas as pd
from .actions import DiscreteActionSpace, ContinuousActionSpace
from .rewards import BaseReward, PnLReward

class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning."""

    def __init__(self,
                 data: pd.DataFrame,
                 action_space: DiscreteActionSpace | ContinuousActionSpace,
                 reward_calculator: Optional[BaseReward] = None,
                 window_size: int = 100,
                 commission: float = 0.001):
        self.data = data
        self.action_space_handler = action_space
        self.reward_calculator = reward_calculator or PnLReward()
        self.window_size = window_size
        self.commission = commission

        if isinstance(action_space, DiscreteActionSpace):
            self.action_space = gym.spaces.Discrete(action_space.get_action_space_size())
        else:
            self.action_space = gym.spaces.Box(
                low=action_space.min_position,
                high=action_space.max_position,
                shape=(1,),
                dtype=np.float32
            )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self.data.columns)),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.position = 0.0
        self.pnl = 0.0
        self.trades = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int | float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(self.action_space_handler, DiscreteActionSpace):
            new_position = self.action_space_handler.get_position_for_action(action)
        else:
            new_position = float(action)
            new_position = self.action_space_handler.clip_position(new_position)

        price_change = self.data['close'].iloc[self.current_step] / \
                      self.data['close'].iloc[self.current_step - 1] - 1

        trade_size = abs(new_position - self.position)
        trading_cost = trade_size * self.commission

        step_pnl = self.position * price_change - trading_cost
        self.pnl += step_pnl

        if trade_size > 0:
            self.trades.append({
                'step': self.current_step,
                'price': self.data['close'].iloc[self.current_step],
                'position': new_position,
                'pnl': step_pnl
            })

        self.position = new_position

        reward = self.reward_calculator.calculate(
            action=new_position,
            position=self.position,
            pnl=step_pnl
        )

        self.current_step += 1
        observation = self._get_observation()

        done = self.current_step >= len(self.data) - 1
        truncated = False

        info = self._get_info()

        return observation, reward, done, truncated, info

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        return self.data.iloc[start:end].values

    def _get_info(self) -> Dict[str, Any]:
        return {
            'step': self.current_step,
            'position': self.position,
            'pnl': self.pnl,
            'trade_count': len(self.trades),
            'trades': self.trades
        }
