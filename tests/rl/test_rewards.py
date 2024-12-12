import pytest
import numpy as np
from don.rl.rewards import BaseReward, PnLReward, SharpeReward, RiskAdjustedReward

class MockReward(BaseReward):
    def calculate(self, action: float, position: float, pnl: float, **kwargs) -> float:
        return pnl * position

def test_base_reward_interface():
    with pytest.raises(TypeError):
        BaseReward()

    reward = MockReward()
    assert reward.calculate(0.5, 1.0, 0.1) == 0.1

def test_pnl_reward():
    reward = PnLReward()

    assert reward.calculate(0.5, 1.0, 0.1) == 0.1
    assert reward.calculate(-0.5, -1.0, -0.1) == -0.1
    assert reward.calculate(0.0, 0.0, 0.0) == 0.0

    assert reward.calculate(0.5, 0.5, 0.1) == 0.1
    assert reward.calculate(-0.5, -0.5, 0.1) == 0.1

def test_sharpe_reward():
    reward = SharpeReward(window=3, risk_free_rate=0.0)

    assert reward.calculate(0.5, 1.0, 0.1) == 0.0
    assert reward.calculate(0.5, 1.0, 0.2) == 0.0

    reward.calculate(0.5, 1.0, 0.3)

    reward = SharpeReward(window=3, risk_free_rate=0.0)
    returns = [0.1, 0.1, 0.1]
    for ret in returns:
        result = reward.calculate(0.5, 1.0, ret)
    assert result > 0

    reward = SharpeReward(window=3, risk_free_rate=0.0)
    returns = [-0.1, 0.1, -0.1]
    for ret in returns:
        result = reward.calculate(0.5, 1.0, ret)
    assert result < 0

    reward = SharpeReward(window=3, risk_free_rate=0.05)
    returns = [0.1, 0.1, 0.1]
    for ret in returns:
        result = reward.calculate(0.5, 1.0, ret)
    assert 0 < result < np.inf

def test_risk_adjusted_reward():
    reward = RiskAdjustedReward(position_penalty=0.1)

    assert reward.calculate(0.0, 0.0, 0.1) == 0.1

    assert reward.calculate(1.0, 1.0, 0.1) == 0.0
    assert reward.calculate(-1.0, -1.0, 0.1) == 0.0

    assert reward.calculate(0.5, 0.5, 0.1) == 0.05

    reward = RiskAdjustedReward(position_penalty=0.2)
    assert reward.calculate(1.0, 1.0, 0.1) == -0.1

    assert reward.calculate(1.0, 1.0, -0.1) == -0.3
