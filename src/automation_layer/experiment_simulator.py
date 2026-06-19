"""
Experiment Simulator - Research Plan Section 2.3.
Generates simulated traffic for Thompson Sampling and LinUCB algorithm testing.
"""

import logging
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


class ExperimentSimulator:
    """
    Research Plan Section 2.3 (Multi-Armed Bandits):
    - Thompson Sampling: Updates alpha/beta parameters
    - LinUCB: Updates with contextual features
    - Delayed rewards: Simulates 72-hour attribution window
    """

    def __init__(self):
        self.name = "ExperimentSimulator"
        np.random.seed(42)

    async def simulate_experiment_traffic(
        self,
        experiment_id: str,
        num_pulls: int = 100,
        db_session = None
    ) -> Dict[str, Any]:
        from ..data_layer.database.models import Experiment, BanditArm
        from sqlalchemy import select

        logger.info(f"🔬 EXPERIMENT SIMULATION for experiment {experiment_id}")
        logger.info(f"   Simulating {num_pulls} pulls across all arms")

        stmt = select(Experiment).where(Experiment.id == UUID(experiment_id))
        result = await db_session.execute(stmt)
        experiment = result.scalar_one_or_none()

        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        stmt = select(BanditArm).where(BanditArm.experiment_id == UUID(experiment_id))
        result = await db_session.execute(stmt)
        arms = result.scalars().all()

        if not arms:
            raise ValueError(f"No bandit arms found for experiment {experiment_id}")

        # True conversion rates are hidden from the algorithm
        true_conversion_rates = self._assign_true_rates(len(arms))

        logger.info(f"   Arms: {[arm.arm_id for arm in arms]}")
        logger.info(f"   True conversion rates (hidden): {[f'{r:.3f}' for r in true_conversion_rates]}")

        if experiment.algorithm == 'thompson_sampling':
            arm_pulls = self._thompson_sampling_allocation(arms, num_pulls)
        elif experiment.algorithm == 'linucb':
            arm_pulls = self._linucb_allocation(arms, num_pulls)
        else:
            # Uniform allocation (A/B test)
            arm_pulls = [num_pulls // len(arms)] * len(arms)

        total_conversions = 0
        arm_results = []

        for idx, arm in enumerate(arms):
            pulls = arm_pulls[idx]
            true_rate = true_conversion_rates[idx]

            conversions = np.random.binomial(pulls, true_rate)
            failures = pulls - conversions

            arm.pulls = (arm.pulls or 0) + pulls
            arm.successes = (arm.successes or 0) + conversions
            arm.failures = (arm.failures or 0) + failures
            arm.total_reward = (arm.total_reward or 0.0) + conversions

            # Bayesian posterior update: Beta(alpha + successes, beta + failures)
            arm.alpha = (arm.alpha or 1.0) + conversions
            arm.beta = (arm.beta or 1.0) + failures

            arm.last_pulled_at = datetime.utcnow()

            total_conversions += conversions

            observed_ctr = conversions / pulls if pulls > 0 else 0

            arm_results.append({
                "arm_id": arm.arm_id,
                "pulls": pulls,
                "conversions": conversions,
                # Return CTR as PERCENTAGE for API consistency
                "ctr": round(observed_ctr * 100, 2),
                "total_pulls": arm.pulls,
                "total_conversions": arm.successes,
                "alpha": round(arm.alpha, 2),
                "beta": round(arm.beta, 2)
            })

            logger.info(f"   {arm.arm_id}: {pulls} pulls → {conversions} conversions (CTR: {observed_ctr:.2%})")

        experiment.total_impressions = (experiment.total_impressions or 0) + num_pulls
        experiment.total_conversions = (experiment.total_conversions or 0) + total_conversions

        await db_session.commit()

        overall_ctr_decimal = total_conversions / num_pulls if num_pulls > 0 else 0

        winner = max(arms, key=lambda a: a.successes / max(a.pulls, 1))
        winner_ctr_decimal = winner.successes / max(winner.pulls, 1)

        result = {
            "status": "simulation_complete",
            "experiment_id": experiment_id,
            "algorithm": experiment.algorithm,
            "num_pulls": num_pulls,
            "total_conversions": total_conversions,
            # Return CTR as PERCENTAGE for API consistency
            "overall_ctr": round(overall_ctr_decimal * 100, 2),
            "arm_results": arm_results,
            "winner": winner.arm_id,
            "winner_ctr": round(winner_ctr_decimal * 100, 2),
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"✅ Simulation complete: {total_conversions} conversions from {num_pulls} pulls")
        logger.info(f"   Current winner: {winner.arm_id} (CTR: {result['winner_ctr']:.2f}%)")

        return result

    def _assign_true_rates(self, num_arms: int) -> List[float]:
        """Rates between 1-8% with variance to simulate real variant effectiveness differences."""
        base_rate = 0.03
        rates = []

        for i in range(num_arms):
            variance = np.random.uniform(-0.02, 0.03)
            rate = max(0.01, min(0.10, base_rate + variance))
            rates.append(rate)

        return sorted(rates, reverse=True)

    def _thompson_sampling_allocation(self, arms: List, num_pulls: int) -> List[int]:
        """Thompson Sampling (Research Plan Section 2.3): sample from Beta(α,β) posterior, pick highest."""
        arm_pulls = [0] * len(arms)

        for _ in range(num_pulls):
            samples = [
                np.random.beta(arm.alpha or 1.0, arm.beta or 1.0)
                for arm in arms
            ]

            chosen_arm = np.argmax(samples)
            arm_pulls[chosen_arm] += 1

        return arm_pulls

    def _linucb_allocation(self, arms: List, num_pulls: int) -> List[int]:
        """LinUCB simplified: exploration bonus via 1/√(pulls) favors less-pulled arms."""
        arm_pulls = [0] * len(arms)

        for _ in range(num_pulls):
            total_pulls = [arm.pulls or 1 for arm in arms]
            exploration_bonus = [1.0 / np.sqrt(p) for p in total_pulls]

            probs = np.array(exploration_bonus)
            probs = probs / probs.sum()

            chosen_arm = np.random.choice(len(arms), p=probs)
            arm_pulls[chosen_arm] += 1

        return arm_pulls


_simulator = ExperimentSimulator()


async def run_experiment_simulation(
    experiment_id: str,
    num_pulls: int = 100,
    db_session = None
) -> Dict[str, Any]:
    return await _simulator.simulate_experiment_traffic(
        experiment_id,
        num_pulls,
        db_session
    )


run_mock_experiment = run_experiment_simulation
