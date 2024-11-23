import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List
import numpy as np
from torch.distributions import Normal
import logging
import os

class AttentionLayer(nn.Module):
    """
    Multi-head attention layer for temporal data processing.
    """
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Linear projections for Query, Key, Value
        self.q_linear = nn.Linear(input_dim, num_heads * head_dim)
        self.k_linear = nn.Linear(input_dim, num_heads * head_dim)
        self.v_linear = nn.Linear(input_dim, num_heads * head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(num_heads * head_dim, input_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, input_dim = x.size()
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        output = self.output_linear(context)
        
        return output, attention_weights

class TemporalFusionEncoder(nn.Module):
    """
    Temporal fusion encoder combining attention mechanisms with LSTM layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, mask)
        
        # Residual connection and layer normalization
        output = self.layer_norm(lstm_out + attended_out)
        
        return output

class HierarchicalPolicy(nn.Module):
    """
    Hierarchical policy network with high-level and low-level policies.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_options: int = 4
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        
        # High-level policy (option selection)
        self.option_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options)
        )
        
        # Low-level policies (one for each option)
        self.action_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std for each action
            )
            for _ in range(num_options)
        ])
        
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        # Option selection
        option_logits = self.option_policy(state)
        option_probs = F.softmax(option_logits, dim=-1)
        
        # Sample option (one per sequence)
        option_distribution = torch.distributions.Categorical(option_probs)
        selected_option = option_distribution.sample()
        
        # Get action distribution for selected option
        batch_size = state.size(0)
        action_params = torch.stack([
            self.action_policies[selected_option[i]](state[i])
            for i in range(batch_size)
        ])
        
        action_mean, action_log_std = torch.chunk(action_params, 2, dim=-1)
        action_std = torch.exp(action_log_std)
        
        # Create action distribution
        action_distribution = Normal(action_mean, action_std)
        
        return option_probs, selected_option, action_distribution

class BaseModel(nn.Module):
    """
    Base model implementing hierarchical reinforcement learning with attention mechanisms.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_options: int = 4,
        num_heads: int = 8,
        learning_rate: float = 3e-4,
        device: str = "auto"
    ):
        super().__init__()
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else
                                     "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.encoder = TemporalFusionEncoder(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        ).to(self.device)
        
        self.policy = HierarchicalPolicy(
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_options=num_options
        ).to(self.device)
        
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def forward(
        self,
        state: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            state: Input state tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (option_probs, selected_option, action_distribution, value)
        """
        # Encode state
        encoded_state = self.encoder(state, mask)
        
        # Reshape for policy if needed
        if len(encoded_state.shape) == 3:  # (batch_size, seq_length, hidden_dim)
            batch_size, seq_length, hidden_dim = encoded_state.shape
            encoded_state = encoded_state.reshape(-1, hidden_dim)  # Flatten sequence dimension
        
        # Get policy outputs
        option_probs, selected_option, action_distribution = self.policy(encoded_state)
        
        # Get value estimate
        value = self.value(encoded_state)
        
        return option_probs, selected_option, action_distribution, value
    
    def save(
        self,
        path: str
    ) -> None:
        """
        Save model state.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(
        self,
        path: str
    ) -> None:
        """
        Load model state.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    # Example usage
    model = BaseModel(
        state_dim=100,
        action_dim=3,
        hidden_dim=256,
        num_options=4,
        num_heads=8
    )
    
    # Test forward pass
    batch_size = 32
    seq_length = 50
    state = torch.randn(batch_size, seq_length, 100)
    option_probs, selected_option, action_dist, value = model(state)
    
    print(f"Option probabilities shape: {option_probs.shape}")
    print(f"Selected option shape: {selected_option.shape}")
    print(f"Value shape: {value.shape}")
