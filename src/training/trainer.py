import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import wandb
from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm

class TradingEnvironment:
    """
    Trading environment simulation with realistic market dynamics.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        transaction_fee: float = 0.001,
        window_size: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize the environment.
        
        Args:
            data: Historical market data
            initial_balance: Initial account balance
            transaction_fee: Transaction fee as percentage
            window_size: Number of time steps to include in state
            device: Computing device
        """
        self.data = data
        self.initial_balance = float(initial_balance)
        self.transaction_fee = float(transaction_fee)
        self.window_size = window_size
        self.device = device
        
        # Convert datetime index to position index if needed
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data = self.data.reset_index(drop=True)
        
        # Ensure all data is numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'quote_volume', 'trades', 'taker_buy_volume',
                         'taker_buy_quote_volume']
        self.data = self.data[numeric_columns].astype(np.float32)
        
        self.reset()
    
    def reset(self) -> torch.Tensor:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state tensor
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.positions_history = []
        self.returns_history = []
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """
        Get current state representation.
        
        Returns:
            State tensor
        """
        # Get window of data
        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Convert to tensor
        state = torch.tensor(window_data.values, dtype=torch.float32, device=self.device)
        
        # Add position information
        position_info = torch.tensor([self.position, self.balance], 
                                   dtype=torch.float32, device=self.device)
        
        # Combine market data with position info
        state = torch.cat([state.flatten(), position_info])
        
        return state
    
    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action tensor
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get current price
        current_price = float(self.data.iloc[self.current_step]['close'])
        
        # Execute action
        action = action.cpu().numpy()
        new_position = float(np.clip(action[0], -1, 1))  # Position size between -1 and 1
        
        # Calculate transaction cost
        position_change = abs(new_position - self.position)
        transaction_cost = float(position_change * current_price * self.transaction_fee)
        
        # Update position
        self.position = new_position
        self.positions_history.append(self.position)
        
        # Calculate returns
        next_price = float(self.data.iloc[self.current_step + 1]['close'])
        price_change = (next_price - current_price) / current_price
        returns = float(self.position * price_change - transaction_cost / self.balance)
        self.returns_history.append(returns)
        
        # Update balance
        self.balance *= (1.0 + returns)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = bool(self.current_step >= len(self.data) - 1)
        
        # Calculate reward (Sharpe ratio for the last 100 steps)
        if len(self.returns_history) >= 100:
            returns_array = np.array(self.returns_history[-100:])
            sharpe = float(np.sqrt(252) * (returns_array.mean() / (returns_array.std() + 1e-6)))
            reward = sharpe
        else:
            reward = 0.0
        
        info = {
            'balance': float(self.balance),
            'position': float(self.position),
            'returns': float(returns),
            'sharpe': float(reward)
        }
        
        return self._get_state(), reward, done, info

class Trainer:
    """
    Trainer class for the trading model with parallel environment support.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        config: Dict[str, Any],
        experiment_name: str = None,
        checkpoint_dir: str = "results/checkpoints",
        log_dir: str = "results/logs"
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = model.device
        
        # Create directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize tensorboard
        self.writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        
        # Initialize W&B
        if experiment_name:
            wandb.init(
                project="bitcoin_trading_rl",
                name=experiment_name,
                config=config
            )
        
        # Initialize environments
        self.n_envs = config.get('n_envs', 8)
        self.envs = [
            TradingEnvironment(
                train_data,
                initial_balance=config['initial_balance'],
                transaction_fee=config['transaction_fee'],
                device=self.device
            )
            for _ in range(self.n_envs)
        ]
        
        # Initialize validation environment
        self.val_env = TradingEnvironment(
            val_data,
            initial_balance=config['initial_balance'],
            transaction_fee=config['transaction_fee'],
            device=self.device
        )
    
    def _parallel_env_step(
        self,
        env_idx: int,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute step in parallel environments.
        
        Args:
            env_idx: Environment index
            action: Action tensor
            
        Returns:
            Step results
        """
        return self.envs[env_idx].step(action)
    
    def train_episode(
        self
    ) -> Dict[str, float]:
        """
        Train for one episode.
        
        Returns:
            Dictionary of episode metrics
        """
        # Reset environments
        states = [env.reset() for env in self.envs]
        states = torch.stack(states)
        
        episode_rewards = []
        episode_lengths = []
        
        done = [False] * self.n_envs
        
        while not all(done):
            # Get actions from model
            with torch.no_grad():
                option_probs, selected_options, action_dists, values = self.model(states)
                actions = torch.stack([dist.sample() for dist in action_dists])
            
            # Execute actions in parallel
            with ThreadPoolExecutor(max_workers=self.n_envs) as executor:
                results = list(executor.map(
                    lambda x: self._parallel_env_step(x[0], x[1]),
                    enumerate(actions)
                ))
            
            # Process results
            new_states = torch.stack([r[0] for r in results])
            rewards = torch.tensor([r[1] for r in results], device=self.device)
            done = [r[2] for r in results]
            infos = [r[3] for r in results]
            
            # Store rewards
            episode_rewards.extend([float(r) for r in rewards])
            
            # Update states
            states = new_states
        
        # Calculate episode metrics
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'final_balance': float(np.mean([env.balance for env in self.envs]))
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model performance.
        
        Returns:
            Dictionary of validation metrics
        """
        state = self.val_env.reset()
        done = False
        validation_rewards = []
        
        while not done:
            # Get action from model
            with torch.no_grad():
                option_probs, selected_option, action_dist, value = self.model(
                    state.unsqueeze(0)
                )
                action = action_dist.sample()
            
            # Execute action
            state, reward, done, info = self.val_env.step(action)
            validation_rewards.append(float(reward))
        
        metrics = {
            'val_mean_reward': float(np.mean(validation_rewards)),
            'val_std_reward': float(np.std(validation_rewards)),
            'val_final_balance': float(self.val_env.balance),
            'val_sharpe_ratio': float(np.mean(validation_rewards) / (np.std(validation_rewards) + 1e-6))
        }
        
        return metrics
    
    def train(
        self,
        num_episodes: int,
        validate_every: int = 10,
        save_every: int = 100
    ) -> None:
        """
        Train the model.
        
        Args:
            num_episodes: Number of episodes to train
            validate_every: Validate every n episodes
            save_every: Save model every n episodes
        """
        best_val_reward = float('-inf')
        
        for episode in tqdm(range(num_episodes)):
            # Train episode
            train_metrics = self.train_episode()
            
            # Log training metrics
            for key, value in train_metrics.items():
                if self.writer:
                    self.writer.add_scalar(f'train/{key}', value, episode)
                if wandb.run is not None:
                    wandb.log({f'train/{key}': value}, step=episode)
            
            # Validate
            if episode % validate_every == 0:
                val_metrics = self.validate()
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    if self.writer:
                        self.writer.add_scalar(f'val/{key}', value, episode)
                    if wandb.run is not None:
                        wandb.log({f'val/{key}': value}, step=episode)
                
                # Save best model
                if val_metrics['val_mean_reward'] > best_val_reward:
                    best_val_reward = val_metrics['val_mean_reward']
                    self.model.save(os.path.join(self.checkpoint_dir, 'best_model.pt'))
            
            # Save checkpoint
            if episode % save_every == 0:
                self.model.save(
                    os.path.join(self.checkpoint_dir, f'model_episode_{episode}.pt')
                )
            
            # Log to console
            self.logger.info(
                f"Episode {episode}: "
                f"Train Reward: {train_metrics['mean_reward']:.4f} ± "
                f"{train_metrics['std_reward']:.4f}, "
                f"Balance: {train_metrics['final_balance']:.2f}"
            )
        
        # Final save
        self.model.save(os.path.join(self.checkpoint_dir, 'final_model.pt'))
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
