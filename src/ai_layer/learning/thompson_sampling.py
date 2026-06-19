"""
Thompson Sampling implementation for multi-armed bandits
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
from datetime import datetime, timedelta
import logging
from scipy import stats

from ...data_layer.database.models import BanditArm, Experiment
from ...config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ArmStatistics:
    arm_id: str
    alpha: float = 1.0  # Beta distribution: success count + 1
    beta: float = 1.0   # Beta distribution: failure count + 1
    pulls: int = 0
    successes: int = 0
    total_reward: float = 0.0
    last_pulled: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def estimated_ctr(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        return stats.beta.interval(confidence, self.alpha, self.beta)

class ThompsonSamplingBandit:
    
    def __init__(
        self,
        experiment_id: str,
        arms: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        self.experiment_id = experiment_id
        self.arms = {}
        
        for arm_id in arms:
            self.arms[arm_id] = ArmStatistics(
                arm_id=arm_id,
                alpha=prior_alpha,
                beta=prior_beta
            )
        
        self.pending_rewards = {}
        
        self.cumulative_regret = 0.0
        self.optimal_arm_pulls = 0
        
        logger.info(f"Initialized Thompson Sampling with {len(arms)} arms")
    
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        if not self.arms:
            raise ValueError("No arms available")
        
        # Sample from Beta posterior for each arm
        samples = {}
        for arm_id, stats in self.arms.items():
            sample = np.random.beta(stats.alpha, stats.beta)
            samples[arm_id] = sample
        
        selected_arm = max(samples, key=samples.get)
        selected_value = samples[selected_arm]
        
        self.arms[selected_arm].pulls += 1
        self.arms[selected_arm].last_pulled = datetime.utcnow()
        
        logger.debug(f"Selected arm {selected_arm} with sample value {selected_value:.4f}")
        
        return selected_arm, selected_value
    
    def update_arm(
        self,
        arm_id: str,
        reward: float,
        immediate: bool = True
    ) -> None:
        if arm_id not in self.arms:
            logger.error(f"Unknown arm: {arm_id}")
            return
        
        stats = self.arms[arm_id]
        
        if reward > 0:
            stats.alpha += reward
            stats.successes += 1
        else:
            stats.beta += (1 - reward)
        
        stats.total_reward += reward
        
        self._update_regret(arm_id, reward)
        
        logger.debug(f"Updated arm {arm_id}: α={stats.alpha:.2f}, β={stats.beta:.2f}")
    
    def add_delayed_reward(
        self,
        arm_id: str,
        reward_id: str,
        expected_delay_hours: int = 72
    ) -> None:
        self.pending_rewards[reward_id] = {
            'arm_id': arm_id,
            'created_at': datetime.utcnow(),
            'expected_at': datetime.utcnow() + timedelta(hours=expected_delay_hours),
            'surrogate_applied': False
        }
        
        surrogate_reward = self._calculate_surrogate_reward(arm_id)
        self.update_arm(arm_id, surrogate_reward, immediate=False)
        
        logger.info(f"Added delayed reward {reward_id} for arm {arm_id}")
    
    def process_delayed_reward(
        self,
        reward_id: str,
        actual_reward: float
    ) -> None:
        if reward_id not in self.pending_rewards:
            logger.warning(f"Unknown reward ID: {reward_id}")
            return
        
        pending = self.pending_rewards[reward_id]
        arm_id = pending['arm_id']
        
        # Undo surrogate reward before applying actual
        if pending['surrogate_applied']:
            surrogate = self._calculate_surrogate_reward(arm_id)
            self.arms[arm_id].alpha -= surrogate
            self.arms[arm_id].beta -= (1 - surrogate)
        
        self.update_arm(arm_id, actual_reward)
        
        del self.pending_rewards[reward_id]
        
        logger.info(f"Processed delayed reward {reward_id}: {actual_reward}")
    
    def _calculate_surrogate_reward(self, arm_id: str) -> float:
        """
        Surrogate reward for delayed feedback: CTR × estimated_conversion_rate
        """
        stats = self.arms[arm_id]
        
        if stats.pulls > 0:
            estimated_cvr = 0.01
            
            if stats.successes > 0:
                estimated_cvr = min(0.1, stats.successes / max(100, stats.pulls))
        else:
            estimated_cvr = 0.01
        
        surrogate = stats.estimated_ctr * estimated_cvr
        
        return min(1.0, surrogate)
    
    def _update_regret(self, arm_id: str, reward: float) -> None:
        best_arm = max(self.arms.values(), key=lambda x: x.estimated_ctr)
        
        regret = best_arm.estimated_ctr - reward
        self.cumulative_regret += max(0, regret)
        
        if arm_id == best_arm.arm_id:
            self.optimal_arm_pulls += 1
    
    def get_arm_statistics(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        
        for arm_id, arm_stats in self.arms.items():
            ci_lower, ci_upper = arm_stats.confidence_interval()
            
            stats[arm_id] = {
                'pulls': arm_stats.pulls,
                'successes': arm_stats.successes,
                'estimated_ctr': arm_stats.estimated_ctr,
                'confidence_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper
                },
                'total_reward': arm_stats.total_reward,
                'alpha': arm_stats.alpha,
                'beta': arm_stats.beta
            }
        
        return stats
    
    def get_best_arm(self, confidence_threshold: float = 0.95) -> Optional[str]:
        """Get the best performing arm if confidence is high enough."""
        if len(self.arms) < 2:
            return None
        
        sorted_arms = sorted(
            self.arms.items(),
            key=lambda x: x[1].estimated_ctr,
            reverse=True
        )
        
        if len(sorted_arms) < 2:
            return None
        
        best_arm = sorted_arms[0][1]
        second_best = sorted_arms[1][1]
        
        # Check if confidence intervals don't overlap
        best_ci = best_arm.confidence_interval(confidence_threshold)
        second_ci = second_best.confidence_interval(confidence_threshold)
        
        # Best arm's lower bound > second best's upper bound = clear winner
        if best_ci[0] > second_ci[1]:
            return best_arm.arm_id
        
        return None
    
    def should_stop_experiment(
        self,
        min_samples: int = 100,
        confidence_threshold: float = 0.95
    ) -> bool:
        for arm in self.arms.values():
            if arm.pulls < min_samples:
                return False
        
        best_arm = self.get_best_arm(confidence_threshold)
        return best_arm is not None
    
    def get_recommendation(self) -> Dict[str, Any]:
        stats = self.get_arm_statistics()
        best_arm = self.get_best_arm()
        
        total_samples = sum(arm.pulls for arm in self.arms.values())
        power = min(1.0, total_samples / 1000)
        
        recommendation = {
            'best_arm': best_arm,
            'confidence': 0.0,
            'action': 'continue',
            'rationale': '',
            'statistics': stats,
            'total_samples': total_samples,
            'statistical_power': power
        }
        
        if best_arm:
            best_stats = self.arms[best_arm]
            ci_lower, ci_upper = best_stats.confidence_interval()
            
            confidence = 0.0
            for arm_id, arm in self.arms.items():
                if arm_id != best_arm:
                    other_ci = arm.confidence_interval()
                    if ci_lower > other_ci[1]:
                        confidence = max(confidence, 0.95)
            
            recommendation['confidence'] = confidence
            
            if confidence > 0.9:
                recommendation['action'] = 'stop'
                recommendation['rationale'] = f"Arm {best_arm} is clearly superior with {confidence:.1%} confidence"
            elif confidence > 0.7:
                recommendation['action'] = 'continue_limited'
                recommendation['rationale'] = f"Arm {best_arm} is likely superior but needs more data"
            else:
                recommendation['action'] = 'continue'
                recommendation['rationale'] = "No clear winner yet, continue testing"
        else:
            recommendation['action'] = 'continue'
            recommendation['rationale'] = f"Need more data, only {total_samples} samples collected"
        
        return recommendation
    
    def save_state(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'arms': {
                arm_id: {
                    'alpha': stats.alpha,
                    'beta': stats.beta,
                    'pulls': stats.pulls,
                    'successes': stats.successes,
                    'total_reward': stats.total_reward,
                    'last_pulled': stats.last_pulled.isoformat() if stats.last_pulled else None
                }
                for arm_id, stats in self.arms.items()
            },
            'pending_rewards': self.pending_rewards,
            'cumulative_regret': self.cumulative_regret,
            'optimal_arm_pulls': self.optimal_arm_pulls
        }
    
    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> 'ThompsonSamplingBandit':
        bandit = cls(
            experiment_id=state['experiment_id'],
            arms=list(state['arms'].keys())
        )
        
        for arm_id, arm_data in state['arms'].items():
            stats = bandit.arms[arm_id]
            stats.alpha = arm_data['alpha']
            stats.beta = arm_data['beta']
            stats.pulls = arm_data['pulls']
            stats.successes = arm_data['successes']
            stats.total_reward = arm_data['total_reward']
            if arm_data['last_pulled']:
                stats.last_pulled = datetime.fromisoformat(arm_data['last_pulled'])
        
        bandit.pending_rewards = state.get('pending_rewards', {})
        bandit.cumulative_regret = state.get('cumulative_regret', 0.0)
        bandit.optimal_arm_pulls = state.get('optimal_arm_pulls', 0)
        
        return bandit