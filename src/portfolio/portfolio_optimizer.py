"""
Portfolio Optimization Module for Bitcoin Trading RL.
Implements various portfolio optimization strategies for multi-asset trading.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import cvxopt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.utils.helpers import setup_logging

logger = setup_logging(__name__)

class OptimizationStrategy(Enum):
    """Portfolio optimization strategies."""
    MEAN_VARIANCE = 'mean_variance'
    MIN_VARIANCE = 'min_variance'
    MAX_SHARPE = 'max_sharpe'
    RISK_PARITY = 'risk_parity'
    BLACK_LITTERMAN = 'black_litterman'
    HIERARCHICAL_RISK_PARITY = 'hierarchical_risk_parity'

@dataclass
class PortfolioMetrics:
    """Container for portfolio metrics."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var: float
    es: float
    diversification_ratio: float
    concentration: float
    turnover: float

class PortfolioOptimizer:
    """
    Portfolio optimization system that implements various optimization strategies
    and handles constraints, transaction costs, and rebalancing.
    """
    
    def __init__(
        self,
        config: Dict,
        risk_free_rate: float = 0.0,
        transaction_costs: float = 0.001,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            config: Portfolio optimization configuration
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            transaction_costs: Transaction costs as fraction of trade value
            min_weight: Minimum asset weight
            max_weight: Maximum asset weight
        """
        self.config = config
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Covariance estimation
        self.covariance_method = config.get('covariance_method', 'ledoit_wolf')
        
        # Optimization parameters
        self.target_return = config.get('target_return', None)
        self.risk_aversion = config.get('risk_aversion', 1.0)
        self.max_turnover = config.get('max_turnover', None)
        
        logger.info("Initialized portfolio optimizer")
    
    def estimate_covariance(
        self,
        returns: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Estimate covariance matrix using specified method.
        
        Args:
            returns: Asset returns matrix
            method: Covariance estimation method
            
        Returns:
            Estimated covariance matrix
        """
        method = method or self.covariance_method
        
        if method == 'sample':
            return np.cov(returns.T)
        elif method == 'ledoit_wolf':
            lw = LedoitWolf()
            return lw.fit(returns).covariance_
        elif method == 'exponential':
            decay = self.config.get('decay_factor', 0.94)
            weights = np.array([(1-decay) * decay**i
                              for i in range(len(returns))])
            weights = weights / weights.sum()
            weighted_returns = returns - returns.mean(axis=0)
            return (weighted_returns.T @ np.diag(weights) @ weighted_returns)
        else:
            raise ValueError(f"Unknown covariance method: {method}")
    
    def mean_variance_optimization(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Perform mean-variance optimization.
        
        Args:
            returns: Asset returns matrix
            covariance: Covariance matrix
            current_weights: Current portfolio weights
            
        Returns:
            Optimal weights
        """
        n_assets = len(returns[0])
        expected_returns = returns.mean(axis=0)
        
        # Objective function: w^T Σ w - λ w^T μ
        P = cvxopt.matrix(covariance)
        q = cvxopt.matrix(-self.risk_aversion * expected_returns)
        
        # Constraints
        # Sum of weights = 1
        A = cvxopt.matrix(np.ones((1, n_assets)))
        b = cvxopt.matrix(np.ones(1))
        
        # Weight bounds
        G = cvxopt.matrix(np.vstack((
            np.eye(n_assets),
            -np.eye(n_assets)
        )))
        h = cvxopt.matrix(np.hstack((
            np.repeat(self.max_weight, n_assets),
            np.repeat(-self.min_weight, n_assets)
        )))
        
        # Turnover constraint if current weights provided
        if current_weights is not None and self.max_turnover:
            G_turnover = np.vstack((
                np.eye(n_assets) - current_weights,
                -(np.eye(n_assets) - current_weights)
            ))
            h_turnover = np.repeat(self.max_turnover, 2*n_assets)
            G = cvxopt.matrix(np.vstack((G, G_turnover)))
            h = cvxopt.matrix(np.hstack((h, h_turnover)))
        
        # Solve optimization problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        if solution['status'] != 'optimal':
            raise ValueError("Optimization failed to converge")
        
        return np.array(solution['x']).flatten()
    
    def risk_parity_optimization(
        self,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Perform risk parity optimization.
        
        Args:
            covariance: Covariance matrix
            
        Returns:
            Risk parity weights
        """
        n_assets = len(covariance)
        
        def risk_parity_objective(weights):
            weights = np.array(weights).reshape(-1, 1)
            portfolio_risk = np.sqrt(weights.T @ covariance @ weights)[0, 0]
            asset_rc = (covariance @ weights) * weights / portfolio_risk
            asset_rc = asset_rc.flatten()
            risk_diffs = asset_rc[:, None] - asset_rc
            return (risk_diffs ** 2).sum()
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda x: x - self.min_weight},  # Min weight
            {'type': 'ineq', 'fun': lambda x: self.max_weight - x}   # Max weight
        ]
        
        # Solve optimization
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-12}
        )
        
        if not result.success:
            raise ValueError("Risk parity optimization failed to converge")
        
        return result.x
    
    def hierarchical_risk_parity(
        self,
        returns: np.ndarray,
        correlation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Perform hierarchical risk parity optimization.
        
        Args:
            returns: Asset returns matrix
            correlation: Optional correlation matrix
            
        Returns:
            HRP weights
        """
        if correlation is None:
            correlation = np.corrcoef(returns.T)
        
        # Distance matrix
        dist = np.sqrt(2 * (1 - correlation))
        
        def get_quasi_diag(link):
            """Return quasi-diagonal matrix."""
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = correlation.shape[0]
            
            for i in range(num_items-2):
                for j in range(2):
                    if sort_ix[j] >= num_items:
                        sort_ix[j] = link[sort_ix[j]-num_items, 0]
                        sort_ix = sort_ix.append(pd.Series(
                            [link[sort_ix[j]-num_items, 1]]
                        ))
            return sort_ix.values
        
        def get_cluster_var(cov, cluster_items):
            """Return cluster variance."""
            cov_slice = cov[cluster_items][:, cluster_items]
            weights = 1/np.diag(cov_slice)
            weights /= weights.sum()
            return np.dot(np.dot(weights, cov_slice), weights)
        
        def get_recursive_bisection(cov, sort_ix):
            """Perform recursive bisection."""
            weights = pd.Series(1, index=sort_ix)
            clusters = [sort_ix]
            
            while len(clusters) > 0:
                clusters.sort(key=lambda x: len(x), reverse=True)
                cluster = clusters.pop(0)
                if len(cluster) == 1:
                    continue
                
                # Divide cluster
                mid = len(cluster) // 2
                cluster0 = cluster[:mid]
                cluster1 = cluster[mid:]
                
                # Calculate cluster variances
                var0 = get_cluster_var(cov, cluster0)
                var1 = get_cluster_var(cov, cluster1)
                
                # Calculate alpha
                alpha = 1 - var0/(var0 + var1)
                weights[cluster0] *= alpha
                weights[cluster1] *= (1-alpha)
                
                clusters.extend([cluster0, cluster1])
            
            return weights.values
        
        # Hierarchical clustering
        link = pd.DataFrame(
            dist
        ).corr(method='single').values  # Single linkage
        sort_ix = get_quasi_diag(link)
        weights = get_recursive_bisection(
            self.estimate_covariance(returns),
            sort_ix
        )
        
        return weights
    
    def black_litterman_optimization(
        self,
        returns: np.ndarray,
        market_caps: np.ndarray,
        views: Dict[Tuple[int, int], float],
        view_confidences: Dict[Tuple[int, int], float]
    ) -> np.ndarray:
        """
        Perform Black-Litterman optimization.
        
        Args:
            returns: Asset returns matrix
            market_caps: Market capitalizations
            views: Dictionary of views (asset pairs to relative performance)
            view_confidences: Confidence in views
            
        Returns:
            Optimal weights
        """
        n_assets = len(returns[0])
        
        # Prior (market equilibrium)
        market_weights = market_caps / market_caps.sum()
        covariance = self.estimate_covariance(returns)
        pi = self.risk_aversion * covariance @ market_weights
        
        # Views matrix
        P = np.zeros((len(views), n_assets))
        q = np.zeros(len(views))
        omega = np.zeros((len(views), len(views)))
        
        for i, ((asset1, asset2), view) in enumerate(views.items()):
            P[i, asset1] = 1
            P[i, asset2] = -1
            q[i] = view
            omega[i, i] = 1 / view_confidences[(asset1, asset2)]
        
        # Posterior distribution
        tau = self.config.get('bl_tau', 0.05)
        sigma_post = np.linalg.inv(
            np.linalg.inv(tau * covariance) +
            P.T @ np.linalg.inv(omega) @ P
        )
        mu_post = sigma_post @ (
            np.linalg.inv(tau * covariance) @ pi +
            P.T @ np.linalg.inv(omega) @ q
        )
        
        # Optimize with posterior estimates
        return self.mean_variance_optimization(
            returns,
            sigma_post,
            current_weights=market_weights
        )
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        strategy: OptimizationStrategy,
        current_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> PortfolioMetrics:
        """
        Optimize portfolio using specified strategy.
        
        Args:
            returns: Asset returns matrix
            strategy: Optimization strategy to use
            current_weights: Current portfolio weights
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Portfolio metrics
        """
        covariance = self.estimate_covariance(returns)
        
        # Optimize based on strategy
        if strategy == OptimizationStrategy.MEAN_VARIANCE:
            weights = self.mean_variance_optimization(
                returns, covariance, current_weights
            )
        elif strategy == OptimizationStrategy.MIN_VARIANCE:
            # Set risk aversion to very high value for minimum variance
            old_risk_aversion = self.risk_aversion
            self.risk_aversion = 1e6
            weights = self.mean_variance_optimization(
                returns, covariance, current_weights
            )
            self.risk_aversion = old_risk_aversion
        elif strategy == OptimizationStrategy.MAX_SHARPE:
            # Maximize Sharpe ratio through mean-variance optimization
            expected_returns = returns.mean(axis=0)
            self.risk_aversion = 1.0
            weights = self.mean_variance_optimization(
                returns, covariance, current_weights
            )
        elif strategy == OptimizationStrategy.RISK_PARITY:
            weights = self.risk_parity_optimization(covariance)
        elif strategy == OptimizationStrategy.HIERARCHICAL_RISK_PARITY:
            weights = self.hierarchical_risk_parity(returns)
        elif strategy == OptimizationStrategy.BLACK_LITTERMAN:
            weights = self.black_litterman_optimization(
                returns,
                kwargs['market_caps'],
                kwargs['views'],
                kwargs['view_confidences']
            )
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics(
            weights, returns, covariance, current_weights
        )
        
        return metrics
    
    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: Optional[np.ndarray] = None
    ) -> PortfolioMetrics:
        """
        Calculate portfolio metrics.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns matrix
            covariance: Covariance matrix
            current_weights: Current portfolio weights
            
        Returns:
            Portfolio metrics
        """
        # Expected return and risk
        expected_returns = returns.mean(axis=0)
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ covariance @ weights)
        
        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Value at Risk and Expected Shortfall
        portfolio_returns = returns @ weights
        var = -np.percentile(portfolio_returns, 5)
        es = -portfolio_returns[portfolio_returns < -var].mean()
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(covariance))
        div_ratio = portfolio_vol / (weights @ asset_vols)
        
        # Concentration
        concentration = (weights ** 2).sum()
        
        # Turnover
        turnover = 0.0 if current_weights is None else np.abs(
            weights - current_weights
        ).sum()
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var=var,
            es=es,
            diversification_ratio=div_ratio,
            concentration=concentration,
            turnover=turnover
        )
