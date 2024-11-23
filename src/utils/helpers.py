import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path

class DataNormalizer:
    """
    Handles data normalization and standardization with support for
    different scaling methods and automatic handling of NaN values.
    """
    
    def __init__(
        self,
        method: str = 'standard',
        feature_range: Tuple[float, float] = (-1, 1)
    ):
        """
        Initialize the normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
            feature_range: Range for scaled features (for minmax scaling)
        """
        self.method = method
        self.feature_range = feature_range
        self.stats = {}
        
    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Calculate normalization parameters.
        
        Args:
            data: Input data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if self.method == 'standard':
            self.stats['mean'] = np.nanmean(data, axis=0)
            self.stats['std'] = np.nanstd(data, axis=0)
            self.stats['std'] = np.where(self.stats['std'] == 0, 1, self.stats['std'])
            
        elif self.method == 'minmax':
            self.stats['min'] = np.nanmin(data, axis=0)
            self.stats['max'] = np.nanmax(data, axis=0)
            self.stats['range'] = self.stats['max'] - self.stats['min']
            self.stats['range'] = np.where(self.stats['range'] == 0, 1, self.stats['range'])
            
        elif self.method == 'robust':
            self.stats['median'] = np.nanmedian(data, axis=0)
            self.stats['q1'] = np.nanpercentile(data, 25, axis=0)
            self.stats['q3'] = np.nanpercentile(data, 75, axis=0)
            self.stats['iqr'] = self.stats['q3'] - self.stats['q1']
            self.stats['iqr'] = np.where(self.stats['iqr'] == 0, 1, self.stats['iqr'])
    
    def transform(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Apply normalization.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if self.method == 'standard':
            normalized = (data - self.stats['mean']) / self.stats['std']
            
        elif self.method == 'minmax':
            normalized = (data - self.stats['min']) / self.stats['range']
            normalized = (normalized * (self.feature_range[1] - self.feature_range[0]) + 
                        self.feature_range[0])
            
        elif self.method == 'robust':
            normalized = (data - self.stats['median']) / self.stats['iqr']
            
        return normalized
    
    def inverse_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Reverse normalization.
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if self.method == 'standard':
            original = data * self.stats['std'] + self.stats['mean']
            
        elif self.method == 'minmax':
            scaled = (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            original = scaled * self.stats['range'] + self.stats['min']
            
        elif self.method == 'robust':
            original = data * self.stats['iqr'] + self.stats['median']
            
        return original

class ReplayBuffer:
    """
    Efficient replay buffer implementation with support for prioritized experience replay.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            action_dim: Action dimension
            device: Computing device
            alpha: Prioritization exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        self.pos = 0
        self.size = 0
        
    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        # Set max priority for new experience
        self.priorities[self.pos] = self.priorities.max().item() if self.size > 0 else 1.0
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights)
        """
        if self.size < batch_size:
            raise ValueError("Not enough experiences in buffer")
        
        # Calculate sampling probabilities
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights
        )
    
    def update_priorities(
        self,
        indices: torch.Tensor,
        priorities: torch.Tensor
    ) -> None:
        """
        Update priorities for experiences.
        
        Args:
            indices: Indices of experiences
            priorities: New priorities
        """
        self.priorities[indices] = priorities + 1e-6  # Prevent zero priority

def create_experiment_dir(base_dir: str = "results") -> str:
    """
    Create timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def save_dict_to_yaml(
    data: Dict,
    filepath: str
) -> None:
    """
    Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_yaml_to_dict(filepath: str) -> Dict:
    """
    Load YAML file to dictionary.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def setup_device(device_str: str = "auto") -> torch.device:
    """
    Setup computing device.
    
    Args:
        device_str: Device specification
        
    Returns:
        torch.device object
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio of returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if len(excess_returns) < 2:
        return 0.0
    
    return (np.mean(excess_returns) / np.std(excess_returns, ddof=1) * 
            np.sqrt(periods_per_year))

def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sortino ratio of returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate
    if len(excess_returns) < 2:
        return 0.0
    
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 1:
        return np.inf
    
    return (np.mean(excess_returns) / np.std(downside_returns, ddof=1) * 
            np.sqrt(periods_per_year))

def calculate_max_drawdown(values: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        values: Array of portfolio values
        
    Returns:
        Maximum drawdown
    """
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return np.min(drawdown)

if __name__ == "__main__":
    # Example usage
    data = np.random.randn(1000, 10)
    
    # Test normalizer
    normalizer = DataNormalizer(method='standard')
    normalizer.fit(data)
    normalized_data = normalizer.transform(data)
    original_data = normalizer.inverse_transform(normalized_data)
    
    print("Data normalization test:")
    print(f"Original mean: {data.mean():.4f}, std: {data.std():.4f}")
    print(f"Normalized mean: {normalized_data.mean():.4f}, std: {normalized_data.std():.4f}")
    print(f"Recovered mean: {original_data.mean():.4f}, std: {original_data.std():.4f}")
    
    # Test replay buffer
    buffer = ReplayBuffer(1000, 10, 2)
    state = torch.randn(10)
    action = torch.randn(2)
    buffer.add(state, action, 1.0, state, False)
    
    print("\nReplay buffer test:")
    print(f"Buffer size: {buffer.size}")
    if buffer.size >= 32:
        batch = buffer.sample(32)
        print(f"Sampled batch shapes: {[b.shape for b in batch]}")
