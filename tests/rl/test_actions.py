import pytest
import numpy as np
from don.rl.actions import DiscreteActionSpace, ContinuousActionSpace

def test_discrete_action_space_initialization():
    # Valid initialization
    positions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    space = DiscreteActionSpace(positions)
    assert space.positions == positions
    assert space.get_action_space_size() == len(positions)

    # Invalid positions
    with pytest.raises(ValueError):
        DiscreteActionSpace([-2.0, 0.0, 1.0])  # Out of range
    with pytest.raises(ValueError):
        DiscreteActionSpace([1.5, 2.0])  # Out of range

def test_discrete_action_space_conversion():
    positions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    space = DiscreteActionSpace(positions)

    # Test get_position_for_action
    assert space.get_position_for_action(0) == -1.0
    assert space.get_position_for_action(2) == 0.0
    assert space.get_position_for_action(4) == 1.0

    # Test invalid action index
    with pytest.raises(ValueError):
        space.get_position_for_action(5)
    with pytest.raises(ValueError):
        space.get_position_for_action(-1)

    # Test get_action_for_position
    assert space.get_action_for_position(-1.0) == 0
    assert space.get_action_for_position(0.0) == 2
    assert space.get_action_for_position(1.0) == 4

    # Test closest position matching
    assert space.get_action_for_position(-0.7) == 0  # Closest to -1.0
    assert space.get_action_for_position(0.3) == 2  # Closest to 0.0
    assert space.get_action_for_position(0.8) == 4  # Closest to 1.0

def test_continuous_action_space_initialization():
    # Valid initialization
    space = ContinuousActionSpace()
    assert space.min_position == -1.0
    assert space.max_position == 1.0

    space = ContinuousActionSpace(-0.5, 0.5)
    assert space.min_position == -0.5
    assert space.max_position == 0.5

    # Invalid bounds
    with pytest.raises(ValueError):
        ContinuousActionSpace(-2.0, 1.0)  # Min out of range
    with pytest.raises(ValueError):
        ContinuousActionSpace(-1.0, 2.0)  # Max out of range
    with pytest.raises(ValueError):
        ContinuousActionSpace(0.5, -0.5)  # Min > Max

def test_continuous_action_space_clipping():
    space = ContinuousActionSpace(-0.5, 0.5)

    # Test within bounds
    assert space.clip_position(0.0) == 0.0
    assert space.clip_position(-0.5) == -0.5
    assert space.clip_position(0.5) == 0.5

    # Test clipping
    assert space.clip_position(-1.0) == -0.5
    assert space.clip_position(1.0) == 0.5
    assert space.clip_position(2.0) == 0.5
    assert space.clip_position(-2.0) == -0.5

    # Test with numpy arrays
    positions = np.array([-2.0, -0.3, 0.0, 0.3, 2.0])
    clipped = np.array([-0.5, -0.3, 0.0, 0.3, 0.5])
    np.testing.assert_array_almost_equal(
        np.array([space.clip_position(p) for p in positions]),
        clipped
    )
