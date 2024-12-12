from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BaseFeatureCalculator(ABC):
    """Base class for feature calculations."""

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from input data."""
        pass
