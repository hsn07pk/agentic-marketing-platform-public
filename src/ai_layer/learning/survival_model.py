"""
Survival Model for Reward Estimation
Research Plan Section 2.3 (Optional Feature)

This module implements a survival analysis model for refining reward estimates
during the delay window between user actions and final conversions.

The survival curve estimates the probability of conversion over time,
allowing for more accurate surrogate reward calculations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SurvivalEstimate:
    """Represents a survival probability estimate at a given time"""
    time_hours: float
    survival_probability: float
    hazard_rate: float
    cumulative_hazard: float


class SurvivalModel:
    """
    Kaplan-Meier inspired survival model for conversion prediction.
    
    This model estimates the probability that a lead will convert within
    a given time window, enabling more accurate reward estimation during
    the delay period before a booking is confirmed.
    
    Research Plan Reference:
    Section 2.3: "Optionally, a simple survival model can be used to refine
    this estimate."
    """
    
    def __init__(
        self,
        max_conversion_days: int = 7,
        baseline_conversion_rate: float = 0.15,
        time_decay_factor: float = 0.1
    ):
        self.max_conversion_days = max_conversion_days
        self.max_hours = max_conversion_days * 24
        self.baseline_conversion_rate = baseline_conversion_rate
        self.time_decay_factor = time_decay_factor
        
        self._shape_parameter = 1.5  # Weibull shape (k > 1 = increasing hazard)
        self._scale_parameter = 48.0  # Weibull scale (typical hours to conversion)
        
        self._conversion_times: List[float] = []
        self._censored_times: List[float] = []
        
        logger.info(f"SurvivalModel initialized: max_days={max_conversion_days}, "
                   f"baseline_rate={baseline_conversion_rate}")
    
    def survival_function(self, time_hours: float) -> float:
        """Calculate S(t) = P(T > t) using Weibull distribution."""
        if time_hours <= 0:
            return 1.0
            
        # Weibull survival function: S(t) = exp(-(t/λ)^k)
        normalized_time = time_hours / self._scale_parameter
        survival_prob = np.exp(-np.power(normalized_time, self._shape_parameter))
        
        return max(0.0, min(1.0, survival_prob))
    
    def hazard_function(self, time_hours: float) -> float:
        """Calculate instantaneous hazard rate h(t) at time t."""
        if time_hours <= 0:
            return 0.0
            
        # Weibull hazard function: h(t) = (k/λ) * (t/λ)^(k-1)
        k = self._shape_parameter
        scale = self._scale_parameter
        
        hazard = (k / scale) * np.power(time_hours / scale, k - 1)
        return max(0.0, hazard)
    
    def cumulative_hazard(self, time_hours: float) -> float:
        """Calculate H(t) = -log(S(t))."""
        s_t = self.survival_function(time_hours)
        if s_t <= 0:
            return float('inf')
        return -np.log(s_t)
    
    def conversion_probability(
        self,
        time_since_action_hours: float,
        window_hours: Optional[float] = None
    ) -> float:
        """
        Estimate probability of conversion within a time window.
        Key method for refining surrogate rewards.
        """
        if window_hours is None:
            window_hours = self.max_hours - time_since_action_hours
        
        # P(convert in window | survived until now)
        # = [S(now) - S(now + window)] / S(now)
        
        s_now = self.survival_function(time_since_action_hours)
        s_future = self.survival_function(time_since_action_hours + window_hours)
        
        if s_now <= 0:
            return 0.0
            
        conversion_prob = (s_now - s_future) / s_now
        
        final_prob = conversion_prob * self.baseline_conversion_rate
        
        return max(0.0, min(1.0, final_prob))
    
    def estimate_survival_curve(
        self,
        time_points_hours: Optional[List[float]] = None,
        num_points: int = 50
    ) -> List[SurvivalEstimate]:
        if time_points_hours is None:
            time_points_hours = np.linspace(0, self.max_hours, num_points).tolist()
        
        curve = []
        for t in time_points_hours:
            estimate = SurvivalEstimate(
                time_hours=t,
                survival_probability=self.survival_function(t),
                hazard_rate=self.hazard_function(t),
                cumulative_hazard=self.cumulative_hazard(t)
            )
            curve.append(estimate)
        
        return curve
    
    def refine_surrogate_reward(
        self,
        base_ctr: float,
        estimated_conversion_rate: float,
        time_since_action_hours: float = 0
    ) -> float:
        """
        Refine surrogate reward using survival analysis.
        Improves upon simple CTR × CVR by accounting for time-dependent conversion probability.
        """
        survival_adjusted_prob = self.conversion_probability(
            time_since_action_hours
        )
        
        # Weight for survival model contribution in blended estimate
        alpha = 0.3
        
        simple_estimate = base_ctr * estimated_conversion_rate
        survival_estimate = base_ctr * survival_adjusted_prob
        
        refined_reward = (1 - alpha) * simple_estimate + alpha * survival_estimate
        
        logger.debug(
            f"Refined reward: simple={simple_estimate:.4f}, "
            f"survival={survival_estimate:.4f}, combined={refined_reward:.4f}"
        )
        
        return refined_reward
    
    def calibrate_from_history(
        self,
        conversion_times: List[float],
        censored_times: List[float]
    ) -> Dict:
        """Calibrate Weibull parameters from historical data using moment-based estimation."""
        self._conversion_times = conversion_times.copy()
        self._censored_times = censored_times.copy()
        
        if len(conversion_times) < 5:
            logger.warning("Insufficient data for calibration, using defaults")
            return {
                "calibrated": False,
                "reason": "insufficient_data",
                "n_conversions": len(conversion_times)
            }
        
        conv_array = np.array(conversion_times)
        
        self._scale_parameter = float(np.median(conv_array))
        
        # Estimate shape from coefficient of variation
        cv = np.std(conv_array) / np.mean(conv_array)
        # For Weibull: CV ≈ sqrt(Γ(1+2/k)/Γ(1+1/k)² - 1)
        # Simplified approximation: k ≈ 1.2 / CV for CV > 0.5
        if cv > 0.5:
            self._shape_parameter = max(0.5, 1.2 / cv)
        else:
            self._shape_parameter = 2.5
        
        total_leads = len(conversion_times) + len(censored_times)
        if total_leads > 0:
            self.baseline_conversion_rate = len(conversion_times) / total_leads
        
        logger.info(
            f"Calibrated survival model: shape={self._shape_parameter:.2f}, "
            f"scale={self._scale_parameter:.1f}h, "
            f"baseline_rate={self.baseline_conversion_rate:.2%}"
        )
        
        return {
            "calibrated": True,
            "shape_parameter": self._shape_parameter,
            "scale_parameter": self._scale_parameter,
            "baseline_conversion_rate": self.baseline_conversion_rate,
            "n_conversions": len(conversion_times),
            "n_censored": len(censored_times)
        }
    
    def get_model_summary(self) -> Dict:
        return {
            "model_type": "WeibullSurvival",
            "shape_parameter": self._shape_parameter,
            "scale_parameter_hours": self._scale_parameter,
            "baseline_conversion_rate": self.baseline_conversion_rate,
            "max_conversion_days": self.max_conversion_days,
            "median_conversion_hours": self._scale_parameter * np.power(np.log(2), 1/self._shape_parameter),
            "n_calibration_conversions": len(self._conversion_times),
            "n_calibration_censored": len(self._censored_times)
        }


_survival_model_instance: Optional[SurvivalModel] = None


def get_survival_model() -> SurvivalModel:
    global _survival_model_instance
    if _survival_model_instance is None:
        _survival_model_instance = SurvivalModel()
    return _survival_model_instance


def is_survival_model_enabled() -> bool:
    try:
        from src.config.configuration_service import get_configuration_service
        config = get_configuration_service()
        return config.get_bool("ENABLE_SURVIVAL_MODEL", False)
    except Exception:
        return False
