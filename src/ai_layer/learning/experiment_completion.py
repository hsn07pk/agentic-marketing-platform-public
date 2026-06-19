"""
Experiment Completion Logic - Research Plan Section 2.3

Determines when experiments should complete based on:
- Target sample size reached
- Statistical significance achieved
- Duration expired
- Early stopping criteria
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompletionDecision:
    """Decision about whether to complete an experiment"""
    should_complete: bool
    reason: str
    winner: Optional[str]
    confidence: float
    recommendation: str


class ExperimentCompletionChecker:
    """Checks if experiments meet completion criteria per Research Plan Section 2.3."""

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        min_samples_per_arm: int = 30,
        early_stopping_enabled: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.min_samples_per_arm = min_samples_per_arm
        self.early_stopping_enabled = early_stopping_enabled

    def check_completion(
        self,
        experiment: Dict[str, Any],
        arms_data: List[Dict[str, Any]],
        parameters: Dict[str, Any] = None
    ) -> CompletionDecision:
        if parameters is None:
            parameters = experiment.get('parameters', {})

        target_sample_size = parameters.get('target_sample_size', 1000)
        duration_days = parameters.get('duration', 14)
        started_at = datetime.fromisoformat(experiment.get('started_at', datetime.utcnow().isoformat()))

        total_pulls = sum(arm.get('pulls', 0) for arm in arms_data)
        days_running = (datetime.utcnow() - started_at).days

        if total_pulls >= target_sample_size:
            decision = self._check_statistical_significance(experiment, arms_data)
            if decision.should_complete:
                decision.reason = f"Target sample size reached ({total_pulls}/{target_sample_size}) and {decision.reason}"
                return decision
            else:
                # Continue even though sample size reached - need significance
                logger.info(f"Experiment {experiment.get('id')}: Sample size reached but not statistically significant yet")

        if days_running >= duration_days:
            decision = self._check_statistical_significance(experiment, arms_data)
            decision.should_complete = True
            decision.reason = f"Duration expired ({days_running}/{duration_days} days). {decision.reason}"
            return decision

        if self.early_stopping_enabled:
            decision = self._check_statistical_significance(experiment, arms_data)
            if decision.should_complete and decision.confidence >= self.confidence_threshold:
                decision.reason = f"Early stopping: {decision.reason}"
                return decision

        progress = (total_pulls / target_sample_size * 100) if target_sample_size > 0 else 0
        return CompletionDecision(
            should_complete=False,
            reason=f"Continue collecting data ({total_pulls}/{target_sample_size} samples, {progress:.1f}% complete)",
            winner=None,
            confidence=0.0,
            recommendation="keep_running"
        )

    def _check_statistical_significance(
        self,
        experiment: Dict[str, Any],
        arms_data: List[Dict[str, Any]]
    ) -> CompletionDecision:
        """
        Check statistical significance using appropriate method for algorithm

        Thompson Sampling: Posterior probability of being best
        LinUCB: Upper confidence bound separation
        """
        algorithm = experiment.get('algorithm', 'thompson_sampling')

        if algorithm == 'thompson_sampling':
            return self._check_thompson_sampling_significance(arms_data)
        elif algorithm == 'linucb':
            return self._check_linucb_significance(arms_data)
        else:
            # Fallback: simple proportion test
            return self._check_proportion_test(arms_data)

    def _check_thompson_sampling_significance(
        self,
        arms_data: List[Dict[str, Any]]
    ) -> CompletionDecision:
        """Uses Monte Carlo sampling from posterior Beta distributions."""
        if len(arms_data) < 2:
            return CompletionDecision(False, "Need at least 2 arms", None, 0.0, "continue")

        min_pulls = min(arm.get('pulls', 0) for arm in arms_data)
        if min_pulls < self.min_samples_per_arm:
            return CompletionDecision(
                False,
                f"Insufficient samples (min: {min_pulls}/{self.min_samples_per_arm})",
                None,
                0.0,
                "continue"
            )

        # Monte Carlo simulation: sample from posteriors
        n_samples = 10000
        win_counts = {arm['arm_id']: 0 for arm in arms_data}

        for _ in range(n_samples):
            samples = {}
            for arm in arms_data:
                alpha = arm.get('alpha', 1.0)
                beta = arm.get('beta', 1.0)
                samples[arm['arm_id']] = np.random.beta(alpha, beta)

            winner = max(samples.items(), key=lambda x: x[1])[0]
            win_counts[winner] += 1

        probabilities = {arm_id: count / n_samples for arm_id, count in win_counts.items()}

        best_arm = max(probabilities.items(), key=lambda x: x[1])
        winner_id = best_arm[0]
        confidence = best_arm[1]

        winner_arm = next(arm for arm in arms_data if arm['arm_id'] == winner_id)
        winner_ctr = (winner_arm.get('successes', 0) / max(winner_arm.get('pulls', 1), 1)) * 100

        if confidence >= self.confidence_threshold:
            return CompletionDecision(
                should_complete=True,
                reason=f"Statistical significance achieved",
                winner=winner_id,
                confidence=confidence,
                recommendation=f"Promote '{winner_id}' (CTR: {winner_ctr:.2f}%, confidence: {confidence:.1%})"
            )
        else:
            second_best = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[1]
            return CompletionDecision(
                should_complete=False,
                reason=f"Not significant: {winner_id}={confidence:.1%} vs {second_best[0]}={second_best[1]:.1%}",
                winner=winner_id,
                confidence=confidence,
                recommendation="continue"
            )

    def _check_linucb_significance(
        self,
        arms_data: List[Dict[str, Any]]
    ) -> CompletionDecision:
        """Check if upper confidence bounds are separated (no overlap)."""
        if len(arms_data) < 2:
            return CompletionDecision(False, "Need at least 2 arms", None, 0.0, "continue")

        intervals = []
        for arm in arms_data:
            pulls = arm.get('pulls', 0)
            successes = arm.get('successes', 0)

            if pulls < self.min_samples_per_arm:
                return CompletionDecision(
                    False,
                    f"Insufficient samples for {arm['arm_id']}",
                    None,
                    0.0,
                    "continue"
                )

            ctr = successes / pulls if pulls > 0 else 0
            std_error = np.sqrt(ctr * (1 - ctr) / pulls) if pulls > 0 else 1.0

            # 95% confidence interval
            z_score = stats.norm.ppf((1 + self.confidence_threshold) / 2)
            ci_lower = max(0, ctr - z_score * std_error)
            ci_upper = min(1, ctr + z_score * std_error)

            intervals.append({
                'arm_id': arm['arm_id'],
                'ctr': ctr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'pulls': pulls
            })

        intervals.sort(key=lambda x: x['ctr'], reverse=True)

        best = intervals[0]
        second_best = intervals[1]

        if best['ci_lower'] > second_best['ci_upper']:
            confidence = self.confidence_threshold
            return CompletionDecision(
                should_complete=True,
                reason="Confidence intervals separated",
                winner=best['arm_id'],
                confidence=confidence,
                recommendation=f"Promote '{best['arm_id']}' (CTR: {best['ctr']*100:.2f}%)"
            )
        else:
            overlap = min(best['ci_upper'], second_best['ci_upper']) - max(best['ci_lower'], second_best['ci_lower'])
            return CompletionDecision(
                should_complete=False,
                reason=f"Confidence intervals overlap (overlap: {overlap*100:.2f}%)",
                winner=best['arm_id'],
                confidence=0.5,
                recommendation="continue"
            )

    def _check_proportion_test(
        self,
        arms_data: List[Dict[str, Any]]
    ) -> CompletionDecision:
        """Two-proportion z-test for A/B tests."""
        if len(arms_data) != 2:
            return CompletionDecision(False, "Proportion test requires exactly 2 arms", None, 0.0, "continue")

        arm_a, arm_b = arms_data

        n_a = arm_a.get('pulls', 0)
        n_b = arm_b.get('pulls', 0)
        x_a = arm_a.get('successes', 0)
        x_b = arm_b.get('successes', 0)

        if n_a < self.min_samples_per_arm or n_b < self.min_samples_per_arm:
            return CompletionDecision(False, "Insufficient samples", None, 0.0, "continue")

        p_a = x_a / n_a if n_a > 0 else 0
        p_b = x_b / n_b if n_b > 0 else 0
        p_pool = (x_a + x_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0

        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if (n_a + n_b) > 0 else 1.0
        z_score = (p_a - p_b) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        winner = arm_a['arm_id'] if p_a > p_b else arm_b['arm_id']
        confidence = 1 - p_value

        if p_value < (1 - self.confidence_threshold):
            return CompletionDecision(
                should_complete=True,
                reason=f"Statistically significant difference (p={p_value:.4f})",
                winner=winner,
                confidence=confidence,
                recommendation=f"Promote '{winner}'"
            )
        else:
            return CompletionDecision(
                should_complete=False,
                reason=f"No significant difference (p={p_value:.4f})",
                winner=winner,
                confidence=confidence,
                recommendation="continue"
            )


# Convenience function
def should_complete_experiment(
    experiment: Dict[str, Any],
    arms_data: List[Dict[str, Any]],
    parameters: Dict[str, Any] = None
) -> CompletionDecision:
    checker = ExperimentCompletionChecker()
    return checker.check_completion(experiment, arms_data, parameters)
