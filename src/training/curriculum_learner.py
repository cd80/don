"""
Curriculum Learning Module for Bitcoin Trading RL.
Implements progressive learning strategies from simple to complex trading tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class TaskDifficulty(Enum):
    """Task difficulty levels."""
    VERY_EASY = 'very_easy'
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'
    VERY_HARD = 'very_hard'

@dataclass
class TaskConfig:
    """Configuration for a curriculum task."""
    difficulty: TaskDifficulty
    market_volatility: Tuple[float, float]  # (min, max) volatility range
    trend_strength: Tuple[float, float]     # (min, max) trend strength
    time_horizon: int                       # Time steps for prediction
    position_constraints: Dict              # Position size/leverage limits
    required_accuracy: float                # Required accuracy to progress
    sampling_weight: float                  # Probability of task selection

class CurriculumTask:
    """
    Individual task in the curriculum.
    Represents a specific trading scenario with defined difficulty.
    """
    
    def __init__(
        self,
        config: TaskConfig,
        data_generator: Callable,
        performance_metrics: List[str]
    ):
        """
        Initialize curriculum task.
        
        Args:
            config: Task configuration
            data_generator: Function to generate task data
            performance_metrics: Metrics to evaluate performance
        """
        self.config = config
        self.data_generator = data_generator
        self.performance_metrics = performance_metrics
        self.completion_history = []
        self.current_performance = None
    
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate task data."""
        return self.data_generator(
            volatility_range=self.config.market_volatility,
            trend_range=self.config.trend_strength,
            time_horizon=self.config.time_horizon
        )
    
    def evaluate_performance(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model performance on task.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        for metric in self.performance_metrics:
            if metric == 'accuracy':
                metrics[metric] = torch.mean(
                    (torch.abs(predictions - targets) < 0.1).float()
                ).item()
            elif metric == 'mse':
                metrics[metric] = F.mse_loss(predictions, targets).item()
            elif metric == 'profit':
                # Calculate trading profit
                positions = torch.sign(predictions)
                returns = targets * positions
                metrics[metric] = torch.mean(returns).item()
        
        self.current_performance = metrics
        self.completion_history.append(metrics)
        
        return metrics
    
    def is_completed(self) -> bool:
        """Check if task completion criteria are met."""
        if not self.current_performance:
            return False
        
        # Check if required accuracy is achieved
        if 'accuracy' in self.current_performance:
            return (self.current_performance['accuracy'] >=
                   self.config.required_accuracy)
        
        # Alternative completion criteria
        if 'profit' in self.current_performance:
            return self.current_performance['profit'] > 0
        
        return False

class CurriculumScheduler:
    """
    Scheduler for curriculum learning.
    Manages task progression and selection.
    """
    
    def __init__(
        self,
        tasks: List[CurriculumTask],
        strategy: str = 'performance_based',
        temperature: float = 1.0
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            tasks: List of curriculum tasks
            strategy: Task selection strategy
            temperature: Temperature for softmax sampling
        """
        self.tasks = tasks
        self.strategy = strategy
        self.temperature = temperature
        self.task_history = []
        self.performance_history = []
    
    def select_task(self) -> CurriculumTask:
        """
        Select next task based on strategy.
        
        Returns:
            Selected task
        """
        if self.strategy == 'sequential':
            # Return first incomplete task
            for task in self.tasks:
                if not task.is_completed():
                    return task
            return self.tasks[-1]
        
        elif self.strategy == 'performance_based':
            # Calculate selection probabilities based on performance
            probs = self._calculate_selection_probabilities()
            return np.random.choice(self.tasks, p=probs)
        
        elif self.strategy == 'adaptive':
            # Adapt based on recent performance
            return self._adaptive_selection()
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _calculate_selection_probabilities(self) -> np.ndarray:
        """Calculate task selection probabilities."""
        # Get task weights
        weights = np.array([
            task.config.sampling_weight for task in self.tasks
        ])
        
        # Adjust weights based on completion
        completion_adj = np.array([
            0.1 if task.is_completed() else 1.0
            for task in self.tasks
        ])
        
        # Apply temperature
        logits = np.log(weights * completion_adj) / self.temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        return probs
    
    def _adaptive_selection(self) -> CurriculumTask:
        """Adaptive task selection based on learning progress."""
        if len(self.performance_history) < 2:
            # Not enough history for adaptation
            return self.tasks[0]
        
        # Calculate learning progress for each task
        progress = []
        for task in self.tasks:
            if len(task.completion_history) < 2:
                progress.append(0)
            else:
                # Calculate improvement
                recent = task.completion_history[-1]['accuracy']
                previous = task.completion_history[-2]['accuracy']
                progress.append(recent - previous)
        
        # Select task with highest progress potential
        progress = np.array(progress)
        task_idx = np.argmax(progress)
        
        return self.tasks[task_idx]
    
    def update_progress(
        self,
        task: CurriculumTask,
        performance: Dict[str, float]
    ) -> None:
        """
        Update learning progress.
        
        Args:
            task: Completed task
            performance: Performance metrics
        """
        self.task_history.append(task.config.difficulty)
        self.performance_history.append(performance)
        
        # Adjust temperature based on overall progress
        if len(self.performance_history) > 10:
            recent_perf = np.mean([
                p['accuracy'] for p in self.performance_history[-10:]
            ])
            if recent_perf > 0.8:
                self.temperature *= 0.9  # Increase exploitation
            elif recent_perf < 0.5:
                self.temperature *= 1.1  # Increase exploration

class CurriculumLearner:
    """
    Curriculum learning system that implements progressive learning
    strategies for trading models.
    """
    
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        tasks: List[CurriculumTask],
        scheduler_type: str = 'performance_based'
    ):
        """
        Initialize curriculum learner.
        
        Args:
            config: Curriculum learning configuration
            model: Model to train
            tasks: List of curriculum tasks
            scheduler_type: Type of curriculum scheduler
        """
        self.config = config
        self.model = model
        self.tasks = tasks
        self.scheduler = CurriculumScheduler(
            tasks=tasks,
            strategy=scheduler_type
        )
        
        logger.info(
            f"Initialized curriculum learner with {len(tasks)} tasks"
        )
    
    def train_step(
        self,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform single curriculum training step.
        
        Args:
            optimizer: Optimizer for model update
            
        Returns:
            Dictionary of metrics
        """
        # Select task
        task = self.scheduler.select_task()
        
        # Generate task data
        features, targets = task.generate_data()
        
        # Forward pass
        predictions = self.model(features)
        loss = F.mse_loss(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate performance
        with torch.no_grad():
            performance = task.evaluate_performance(predictions, targets)
        
        # Update curriculum
        self.scheduler.update_progress(task, performance)
        
        return {
            'task': task.config.difficulty.value,
            'loss': loss.item(),
            **performance
        }
    
    def train(
        self,
        num_steps: int,
        optimizer: torch.optim.Optimizer,
        eval_frequency: int = 100
    ) -> Dict[str, List]:
        """
        Train model using curriculum learning.
        
        Args:
            num_steps: Number of training steps
            optimizer: Optimizer for model update
            eval_frequency: Steps between evaluations
            
        Returns:
            Training history
        """
        history = {
            'loss': [],
            'task_difficulty': [],
            'performance': []
        }
        
        for step in range(num_steps):
            # Training step
            metrics = self.train_step(optimizer)
            
            # Record history
            history['loss'].append(metrics['loss'])
            history['task_difficulty'].append(metrics['task'])
            history['performance'].append(metrics['accuracy'])
            
            # Log progress
            if (step + 1) % eval_frequency == 0:
                avg_metrics = {
                    k: np.mean(v[-eval_frequency:])
                    for k, v in history.items()
                }
                logger.info(
                    f"Step {step+1}/{num_steps}: {avg_metrics}"
                )
        
        return history
    
    def get_curriculum_status(self) -> Dict[str, List]:
        """Get current status of curriculum learning."""
        return {
            'completed_tasks': [
                task.config.difficulty.value
                for task in self.tasks
                if task.is_completed()
            ],
            'task_history': [
                t.value for t in self.scheduler.task_history
            ],
            'performance_history': self.scheduler.performance_history
        }
    
    def save_curriculum_state(self, path: str) -> None:
        """
        Save curriculum learning state.
        
        Args:
            path: Path to save state
        """
        state = {
            'model_state': self.model.state_dict(),
            'task_states': [
                {
                    'config': task.config,
                    'history': task.completion_history
                }
                for task in self.tasks
            ],
            'scheduler_state': {
                'task_history': self.scheduler.task_history,
                'performance_history': self.scheduler.performance_history,
                'temperature': self.scheduler.temperature
            }
        }
        torch.save(state, path)
        logger.info(f"Saved curriculum state to {path}")
    
    def load_curriculum_state(self, path: str) -> None:
        """
        Load curriculum learning state.
        
        Args:
            path: Path to load state from
        """
        state = torch.load(path)
        self.model.load_state_dict(state['model_state'])
        
        # Restore task states
        for task, task_state in zip(self.tasks, state['task_states']):
            task.config = task_state['config']
            task.completion_history = task_state['history']
        
        # Restore scheduler state
        self.scheduler.task_history = state['scheduler_state']['task_history']
        self.scheduler.performance_history = state['scheduler_state']['performance_history']
        self.scheduler.temperature = state['scheduler_state']['temperature']
        
        logger.info(f"Loaded curriculum state from {path}")
