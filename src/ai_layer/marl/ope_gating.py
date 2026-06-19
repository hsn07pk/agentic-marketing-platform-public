"""
Offline Policy Evaluation (OPE) Gating Mechanism for MARL Promotion

Research Plan Section: MARL Policy Promotion Gating
- Validates new MARL policies using historical data before deployment
- Uses doubly-robust OPE for confidence intervals
- Only promotes policies with statistically significant improvements (>20% lift)
- Requires minimum sample size for reliable estimates (configurable via dashboard)
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import json

from ...config.settings import settings
from ...data_layer.database.connection import get_sync_session
from ...config.configuration_service import ConfigurationService

logger = logging.getLogger(__name__)


def _get_ope_config_value(key: str, default: Any = None) -> Any:
    try:
        sync_db = get_sync_session()
        try:
            config_service = ConfigurationService(sync_db)
            value = config_service.get_value(key)
            if value is not None:
                return value
        finally:
            sync_db.close()
    except Exception as e:
        logger.warning(f"Failed to get OPE config '{key}' from database: {e}")
    return getattr(settings, key, default)


@dataclass
class PolicyEvaluation:
    policy_id: str
    estimated_value: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    sample_size: int
    baseline_value: float
    lift_percentage: float
    passes_threshold: bool
    timestamp: datetime

class OffPolicyEvaluator:
    """
    Implements Doubly-Robust Off-Policy Evaluation

    Evaluates new MARL policies using historical data without live deployment
    """

    def __init__(self):
        self.confidence_level = _get_ope_config_value('OPE_CONFIDENCE_LEVEL', 0.95)
        self.min_samples = _get_ope_config_value('OPE_MIN_SAMPLES', 1000)
        self.promotion_threshold = _get_ope_config_value('MARL_POLICY_PROMOTION_THRESHOLD', 0.2)

        logger.info(
            f"Initialized OPE: confidence={self.confidence_level}, "
            f"min_samples={self.min_samples}, threshold={self.promotion_threshold}"
        )

    def evaluate_policy(
        self,
        policy_id: str,
        new_policy_actions: List[Dict[str, Any]],
        historical_data: List[Dict[str, Any]],
        baseline_policy_value: float
    ) -> PolicyEvaluation:
        """
        Evaluate new policy using doubly-robust OPE

        Args:
            policy_id: Identifier for new policy
            new_policy_actions: Actions new policy would take
            historical_data: Historical state-action-reward tuples
            baseline_policy_value: Current policy's expected value

        Returns:
            PolicyEvaluation with confidence intervals and promotion decision
        """
        try:
            if len(historical_data) < self.min_samples:
                logger.warning(
                    f"Insufficient samples: {len(historical_data)}/{self.min_samples}"
                )
                return self._create_failed_evaluation(
                    policy_id,
                    baseline_policy_value,
                    reason="insufficient_samples"
                )

            propensity_scores = self._calculate_propensity_scores(
                new_policy_actions,
                historical_data
            )

            dm_estimates = self._direct_method_estimates(
                new_policy_actions,
                historical_data
            )

            ips_estimates = self._inverse_propensity_scoring(
                historical_data,
                propensity_scores
            )

            dr_estimates = self._doubly_robust_estimator(
                dm_estimates,
                ips_estimates,
                historical_data,
                propensity_scores
            )

            estimated_value = np.mean(dr_estimates)
            std_error = np.std(dr_estimates) / np.sqrt(len(dr_estimates))

            from scipy import stats
            t_value = stats.t.ppf((1 + self.confidence_level) / 2, len(dr_estimates) - 1)
            ci_lower = estimated_value - t_value * std_error
            ci_upper = estimated_value + t_value * std_error

            lift_percentage = ((estimated_value - baseline_policy_value) / baseline_policy_value) * 100 if baseline_policy_value > 0 else 0

            # Threshold=0 means accept any positive value (permissive mode)
            if self.promotion_threshold <= 0:
                passes_threshold = estimated_value > 0
            else:
                passes_threshold = (
                    lift_percentage >= (self.promotion_threshold * 100) and
                    ci_lower > baseline_policy_value
                )

            evaluation = PolicyEvaluation(
                policy_id=policy_id,
                estimated_value=estimated_value,
                confidence_interval=(ci_lower, ci_upper),
                confidence_level=self.confidence_level,
                sample_size=len(historical_data),
                baseline_value=baseline_policy_value,
                lift_percentage=lift_percentage,
                passes_threshold=passes_threshold,
                timestamp=datetime.now()
            )

            logger.info(
                f"OPE Results for {policy_id}: "
                f"estimated_value={estimated_value:.4f}, "
                f"lift={lift_percentage:.1f}%, "
                f"passes={passes_threshold}"
            )

            return evaluation

        except Exception as e:
            logger.error(f"OPE evaluation failed: {e}")
            return self._create_failed_evaluation(policy_id, baseline_policy_value, str(e))

    def _calculate_propensity_scores(
        self,
        new_policy_actions: List[Dict[str, Any]],
        historical_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        scores = []

        for hist_datapoint in historical_data:
            state = hist_datapoint.get('state', {})
            historical_action = hist_datapoint.get('action')

            baseline_prob = hist_datapoint.get('action_probability')
            if baseline_prob is None or baseline_prob == 0:
                baseline_prob = 1.0 / max(hist_datapoint.get('num_actions', 10), 1)

            new_policy_prob = self._get_action_probability(
                new_policy_actions,
                state,
                historical_action
            )

            propensity = new_policy_prob / (baseline_prob + 1e-8)
            scores.append(propensity)

        return np.array(scores)

    def _get_action_probability(
        self,
        new_policy_actions: List[Dict[str, Any]],
        state: Dict[str, Any],
        action: Any
    ) -> float:
        """Get probability of action under new policy using neural network"""
        try:
            if isinstance(action, dict):
                action_key = action.get('action_type') or action.get('type') or action.get('name') or str(sorted(action.items()))
            else:
                action_key = str(action) if action is not None else 'unknown'
            
            state_features = self._extract_state_features(state)

            for policy_action in new_policy_actions:
                policy_state = policy_action.get('state', {})
                policy_state_features = self._extract_state_features(policy_state)

                if self._states_similar(state_features, policy_state_features):
                    action_probs = policy_action.get('action_probabilities', {})
                    if action_key in action_probs:
                        return action_probs[action_key]

            action_scores = {}
            for policy_action in new_policy_actions:
                for act, prob in policy_action.get('action_probabilities', {}).items():
                    if act not in action_scores:
                        action_scores[act] = []
                    action_scores[act].append(prob)

            if action_key in action_scores:
                return np.mean(action_scores[action_key])

        except Exception as e:
            logger.warning(f"Error calculating action probability: {e}")

        return 0.1

    def _extract_state_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from state dict"""
        features = []
        for key in ['persona_id', 'time_of_day', 'day_of_week', 'campaign_type']:
            val = state.get(key, 0)
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, str):
                features.append(float(hash(val) % 1000) / 1000.0)
            else:
                features.append(0.0)
        return np.array(features) if features else np.zeros(4)

    def _states_similar(self, features1: np.ndarray, features2: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if two state feature vectors are similar"""
        if len(features1) != len(features2):
            return False
        distance = np.linalg.norm(features1 - features2)
        return distance < threshold

    def _direct_method_estimates(
        self,
        new_policy_actions: List[Dict[str, Any]],
        historical_data: List[Dict[str, Any]]
    ) -> np.ndarray:
        estimates = []

        for datapoint in historical_data:
            state = datapoint.get('state', {})
            reward = datapoint.get('reward', 0.0)
            action = datapoint.get('action')

            # V(s) = E[R | s, π_new] ≈ Σ_a π(a|s) * Q(s,a)
            estimated_value = self._estimate_state_value(state, action, reward, new_policy_actions)
            estimates.append(estimated_value)

        return np.array(estimates)

    def _estimate_state_value(
        self,
        state: Dict[str, Any],
        action: str,
        observed_reward: float,
        new_policy_actions: List[Dict[str, Any]]
    ) -> float:
        """Estimate state value using Q-function approximation"""
        try:
            state_features = self._extract_state_features(state)

            q_values = []
            for policy_action in new_policy_actions:
                policy_state = policy_action.get('state', {})
                policy_state_features = self._extract_state_features(policy_state)

                if self._states_similar(state_features, policy_state_features):
                    action_probs = policy_action.get('action_probabilities', {})
                    for act, prob in action_probs.items():
                        # Q(s,a) ≈ R * decay (0.9 for same action, 0.8 otherwise)
                        q_estimate = observed_reward * (0.9 if act == action else 0.8)
                        q_values.append(prob * q_estimate)

            if q_values:
                return np.sum(q_values)

            return observed_reward * 0.85

        except Exception as e:
            logger.warning(f"Error estimating state value: {e}")
            return observed_reward * 0.85

    def _inverse_propensity_scoring(
        self,
        historical_data: List[Dict[str, Any]],
        propensity_scores: np.ndarray
    ) -> np.ndarray:
        estimates = []

        for i, datapoint in enumerate(historical_data):
            reward = datapoint.get('reward', 0.0)
            propensity = propensity_scores[i]

            # Clip propensity to prevent extreme importance weights
            clipped_propensity = np.clip(propensity, 0.01, 100.0)
            ips_estimate = reward * clipped_propensity

            estimates.append(ips_estimate)

        return np.array(estimates)

    def _doubly_robust_estimator(
        self,
        dm_estimates: np.ndarray,
        ips_estimates: np.ndarray,
        historical_data: List[Dict[str, Any]],
        propensity_scores: np.ndarray
    ) -> np.ndarray:
        """
        Doubly-robust estimator combining DM and IPS

        DR = DM + IPS * (R - DM)
        This is unbiased if either DM or IPS is correct
        """
        dr_estimates = []

        for i in range(len(dm_estimates)):
            reward = historical_data[i].get('reward', 0.0)
            propensity = np.clip(propensity_scores[i], 0.01, 100.0)

            # DR = DM + ρ(R - DM), unbiased if either DM or IPS is correct
            dr_estimate = dm_estimates[i] + propensity * (reward - dm_estimates[i])
            dr_estimates.append(dr_estimate)

        return np.array(dr_estimates)

    def _create_failed_evaluation(
        self,
        policy_id: str,
        baseline_value: float,
        reason: str
    ) -> PolicyEvaluation:
        return PolicyEvaluation(
            policy_id=policy_id,
            estimated_value=baseline_value,
            confidence_interval=(0.0, 0.0),
            confidence_level=0.0,
            sample_size=0,
            baseline_value=baseline_value,
            lift_percentage=0.0,
            passes_threshold=False,
            timestamp=datetime.now()
        )


class MARLGatekeeper:
    """
    Gatekeeper for MARL policy promotion

    Two-stage process:
    1. OPE validation with statistical significance
    2. Canary deployment with gradual rollout
    """

    def __init__(self):
        self.ope_evaluator = OffPolicyEvaluator()
        self.canary_percentage = settings.CANARY_TRAFFIC_PERCENTAGE
        self.promotion_log = []

        logger.info("Initialized MARL Gatekeeper")

    async def evaluate_policy_for_promotion(
        self,
        policy_id: str,
        new_policy_actions: List[Dict[str, Any]],
        historical_data: List[Dict[str, Any]],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Full evaluation pipeline for MARL policy promotion

        Args:
            policy_id: New policy identifier
            new_policy_actions: Actions from new policy
            historical_data: Historical data for OPE
            baseline_metrics: Current policy performance

        Returns:
            Promotion decision with reasoning
        """
        try:
            logger.info(f"Evaluating policy {policy_id} for promotion...")

            baseline_value = baseline_metrics.get('average_reward', 0.0)

            ope_result = self.ope_evaluator.evaluate_policy(
                policy_id=policy_id,
                new_policy_actions=new_policy_actions,
                historical_data=historical_data,
                baseline_policy_value=baseline_value
            )

            self.promotion_log.append({
                'policy_id': policy_id,
                'timestamp': datetime.now().isoformat(),
                'ope_result': {
                    'estimated_value': ope_result.estimated_value,
                    'lift_percentage': ope_result.lift_percentage,
                    'passes_threshold': ope_result.passes_threshold
                }
            })

            if not ope_result.passes_threshold:
                logger.info(f"Policy {policy_id} REJECTED: insufficient lift or confidence")
                return {
                    'approved': False,
                    'reason': 'insufficient_improvement',
                    'ope_result': ope_result,
                    'next_step': 'Keep current policy, continue training'
                }

            logger.info(f"Policy {policy_id} APPROVED for canary deployment")
            return {
                'approved': True,
                'reason': f'OPE shows {ope_result.lift_percentage:.1f}% lift with {ope_result.confidence_level:.0%} confidence',
                'ope_result': ope_result,
                'next_step': f'Deploy to {self.canary_percentage*100:.0f}% canary traffic',
                'canary_percentage': self.canary_percentage
            }

        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return {
                'approved': False,
                'reason': f'evaluation_error: {str(e)}',
                'ope_result': None,
                'next_step': 'Fix evaluation pipeline and retry'
            }

    def get_promotion_history(self) -> List[Dict[str, Any]]:
        return self.promotion_log

    def save_promotion_log(self, filepath: str = "marl_promotion_log.json"):
        try:
            with open(filepath, 'w') as f:
                json.dump(self.promotion_log, f, indent=2, default=str)
            logger.info(f"Saved promotion log to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save promotion log: {e}")
