import gymnasium as gym
from typing import Dict, Any, Tuple
import numpy as np

class TradingEnvironment(gym.Env):
    """Base trading environment."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return new state."""
        raise NotImplementedError
