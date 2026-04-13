# =============================================================================
# Bayesian Adaptation Engine - Beta Distribution Processing
# =============================================================================
"""
Mathematical core for updating model confidence based on live trade outcomes.
Uses a Beta Distribution (Alpha/Beta) for probabilistic win-rate estimation.
"""

import logging
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class BayesianEngine:
    """
    Beta Distribution Engine for Trading Performance.
    Win Rate ~ Beta(alpha, beta)
    - alpha: Successes + Prior
    - beta: Failures + Prior
    """

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        """
        Initialize Bayesian Engine.
        Default (2,2) prior is a weak multi-modal belief centered at 50%.
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def update(self, alpha: float, beta: float, outcome: str) -> Tuple[float, float]:
        """
        Update alpha/beta parameters based on a single trade outcome.
        
        Args:
            alpha: Current alpha
            beta: Current beta
            outcome: 'SUCCESS' or 'FAIL'
        """
        if outcome == 'SUCCESS':
            return alpha + 1, beta
        elif outcome == 'FAIL':
            return alpha, beta + 1
        return alpha, beta

    def get_posterior_mean(self, alpha: float, beta: float) -> float:
        """Calculate the expected win rate (Mean of Beta Distribution)."""
        if (alpha + beta) == 0:
            return 0.5
        return alpha / (alpha + beta)

    def get_probability_above(self, alpha: float, beta: float, threshold: float = 0.60) -> float:
        """
        Calculate the probability that the true win rate is above a certain threshold.
        Example: P(WinRate > 60%)
        """
        # Beta Survival Function (1 - CDF)
        return 1.0 - beta_dist.cdf(threshold, alpha, beta)

    def get_credible_interval(self, alpha: float, beta: float, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate the 95% Credible Interval for the win rate."""
        lower = (1.0 - confidence) / 2.0
        upper = 1.0 - lower
        return beta_dist.ppf(lower, alpha, beta), beta_dist.ppf(upper, alpha, beta)

    def calculate_confidence_scaling(self, alpha: float, beta: float, model_win_rate: float) -> float:
        """
        Calculate a scaling factor (0.0 to 1.0) to penalize model confidence 
        if historical performance is poor or uncertain.
        """
        # If no history, scaling is neutral (1.0)
        total_trades = alpha + beta - (self.prior_alpha + self.prior_beta)
        if total_trades < 3:
            return 1.0
            
        posterior_mean = self.get_posterior_mean(alpha, beta)
        
        # Scaling logic: If posterior mean is 10% lower than expected, scale down.
        # But we also reward over-performance.
        if posterior_mean < model_win_rate:
            # Linear penalty based on relative under-performance
            penalty = posterior_mean / model_win_rate
            return max(0.20, penalty)
            
        return 1.10 # Slight boost for proven out-performers (max 10%)

# Singleton Access
_engine = None
def get_bayesian_engine() -> BayesianEngine:
    global _engine
    if _engine is None:
        _engine = BayesianEngine()
    return _engine
