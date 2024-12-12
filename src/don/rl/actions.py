from typing import List, Tuple, Optional
import numpy as np

class DiscreteActionSpace:
    """Discrete action space for trading.

    Defines a set of discrete positions that can be taken in the market.
    For example: [-1.0, -0.5, 0.0, 0.5, 1.0] represents short, half-short,
    neutral, half-long, and long positions.
    """
    def __init__(self, positions: List[float]):
        """Initialize discrete action space.

        Args:
            positions: List of possible position sizes (-1.0 to 1.0)
        """
        self.positions = sorted(positions)
        self._validate_positions()

    def _validate_positions(self):
        """Validate position values are within [-1, 1] range."""
        if not all(-1.0 <= p <= 1.0 for p in self.positions):
            raise ValueError("All positions must be between -1.0 and 1.0")

    def get_action_space_size(self) -> int:
        """Get number of possible actions."""
        return len(self.positions)

    def get_position_for_action(self, action_idx: int) -> float:
        """Convert action index to position size.

        Args:
            action_idx: Index of the action

        Returns:
            Position size corresponding to the action
        """
        if not 0 <= action_idx < len(self.positions):
            raise ValueError(f"Action index {action_idx} out of range")
        return self.positions[action_idx]

    def get_action_for_position(self, position: float) -> int:
        """Convert position size to closest valid action index.

        Args:
            position: Target position size

        Returns:
            Index of the closest valid action
        """
        # Handle boundary cases first
        if position <= self.positions[0]:
            return 0
        if position >= self.positions[-1]:
            return len(self.positions) - 1

        # Find distances to all positions
        positions = np.array(self.positions)
        distances = np.abs(positions - position)

        # Find the closest position
        return int(np.argmin(distances))


class ContinuousActionSpace:
    """Continuous action space for trading.

    Allows any position size within a specified range, typically [-1.0, 1.0]
    where -1.0 represents maximum short and 1.0 represents maximum long.
    """
    def __init__(self, min_position: float = -1.0, max_position: float = 1.0):
        """Initialize continuous action space.

        Args:
            min_position: Minimum allowed position (default: -1.0)
            max_position: Maximum allowed position (default: 1.0)
        """
        self.min_position = min_position
        self.max_position = max_position
        self._validate_bounds()

    def _validate_bounds(self):
        """Validate position bounds."""
        if not -1.0 <= self.min_position <= self.max_position <= 1.0:
            raise ValueError("Position bounds must be within [-1.0, 1.0]")

    def clip_position(self, position: float) -> float:
        """Clip position to valid range.

        Args:
            position: Target position size

        Returns:
            Clipped position within valid range
        """
        return np.clip(position, self.min_position, self.max_position)
