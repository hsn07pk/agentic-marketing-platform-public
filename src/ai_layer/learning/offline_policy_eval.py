import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import stats

from ...data_layer.database.connection import get_async_session
from ...data_layer.database.models import Experiment, BanditArm, Metric
from .thompson_sampling import ThompsonSamplingBandit
from .linucb import LinUCBBandit
from ...config.settings import settings

logger = logging.getLogger(__name__)

class OfflinePolicyEvaluator:
    
    def __init__(self):
        self.min_samples = settings.MIN_SAMPLES_FOR_DECISION
    
    async def evaluate_thompson_sampling(
        self,
        historical_data: pd.DataFrame,
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        
        try:
            n_arms = historical_data['action'].nunique()
            
            bandit = ThompsonSamplingBandit(n_arms=n_arms)
            
            rewards = []
            regrets = []
            
            for _ in range(n_simulations):
                bandit = ThompsonSamplingBandit(n_arms=n_arms)
                
                sim_rewards = []
                sim_regrets = []
                
                for idx, row in historical_data.iterrows():
                    selected_arm = bandit.select_arm()
                    
                    actual_reward = row['reward'] if row['action'] == selected_arm else 0
                    
                    bandit.update(selected_arm, actual_reward)
                    
                    sim_rewards.append(actual_reward)
                    
                    optimal_reward = historical_data[
                        historical_data['action'] == row['action']
                    ]['reward'].mean()
                    regret = optimal_reward - actual_reward
                    sim_regrets.append(regret)
                
                rewards.append(np.mean(sim_rewards))
                regrets.append(np.sum(sim_regrets))
            
            return {
                "algorithm": "thompson_sampling",
                "avg_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "avg_cumulative_regret": np.mean(regrets),
                "n_simulations": n_simulations
            }
        
        except Exception as e:
            logger.error(f"Failed to evaluate Thompson Sampling: {e}")
            return {"algorithm": "thompson_sampling", "error": str(e)}
    
    async def evaluate_linucb(
        self,
        historical_data: pd.DataFrame,
        context_dim: int
    ) -> Dict[str, Any]:
        
        try:
            n_arms = historical_data['action'].nunique()
            
            bandit = LinUCBBandit(n_arms=n_arms, context_dim=context_dim)
            
            rewards = []
            regrets = []
            
            for idx, row in historical_data.iterrows():
                context = np.array(row['context'])
                
                selected_arm = bandit.select_arm(context)
                
                actual_reward = row['reward'] if row['action'] == selected_arm else 0
                
                bandit.update(selected_arm, context, actual_reward)
                
                rewards.append(actual_reward)
                
                optimal_reward = historical_data[
                    historical_data['action'] == row['action']
                ]['reward'].mean()
                regret = optimal_reward - actual_reward
                regrets.append(regret)
            
            return {
                "algorithm": "linucb",
                "avg_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "cumulative_regret": np.sum(regrets),
                "total_samples": len(historical_data)
            }
        
        except Exception as e:
            logger.error(f"Failed to evaluate LinUCB: {e}")
            return {"algorithm": "linucb", "error": str(e)}
    
    async def compare_policies(
        self,
        historical_data: pd.DataFrame,
        policies: List[str] = None
    ) -> Dict[str, Any]:
        
        if policies is None:
            policies = ["thompson_sampling", "linucb", "random"]
        
        results = {}
        
        if "thompson_sampling" in policies:
            results["thompson_sampling"] = await self.evaluate_thompson_sampling(
                historical_data
            )
        
        if "linucb" in policies:
            context_dim = len(historical_data.iloc[0]['context'])
            results["linucb"] = await self.evaluate_linucb(
                historical_data,
                context_dim
            )
        
        if "random" in policies:
            results["random"] = {
                "algorithm": "random",
                "avg_reward": historical_data['reward'].mean(),
                "std_reward": historical_data['reward'].std()
            }
        
        best_policy = max(
            results.items(),
            key=lambda x: x[1].get("avg_reward", 0)
        )
        
        return {
            "policies": results,
            "best_policy": best_policy[0],
            "best_reward": best_policy[1].get("avg_reward", 0),
            "data_size": len(historical_data)
        }
    
    async def generate_ope_report(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        
        try:
            historical_data = await self._load_campaign_data(campaign_id)
            
            if len(historical_data) < self.min_samples:
                return {
                    "error": f"Insufficient data ({len(historical_data)} samples, need {self.min_samples})"
                }
            
            comparison = await self.compare_policies(historical_data)
            
            recommendation = self._generate_recommendation(comparison)
            
            report = {
                "campaign_id": campaign_id,
                "generated_at": datetime.utcnow().isoformat(),
                "data_size": len(historical_data),
                "policy_comparison": comparison,
                "recommendation": recommendation
            }
            
            logger.info(f"Generated OPE report for campaign {campaign_id}")
            return report
        
        except Exception as e:
            logger.error(f"Failed to generate OPE report: {e}")
            return {"error": str(e)}
    
    async def _load_campaign_data(
        self,
        campaign_id: str
    ) -> pd.DataFrame:
        """
        DEPRECATED: This internal method is not used by API routes.
        Use load_historical_data() from src/api/routers/ope.py instead.
        
        This stub remains for backward compatibility with any internal tests.
        Returns empty DataFrame to ensure no mock data pollutes real evaluations.
        """
        logger.warning(f"_load_campaign_data called for {campaign_id} - use load_historical_data from ope.py router instead")
        # Return empty DataFrame - forces callers to use the proper loader
        return pd.DataFrame(columns=['context', 'action', 'reward'])
    
    def doubly_robust_estimator(
        self,
        historical_data: pd.DataFrame,
        new_policy_probs: np.ndarray,
        baseline_probs: np.ndarray,
        reward_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Doubly-Robust Off-Policy Evaluation

        DR = (1/n) * Σ[ (π_new(a|x) / π_old(a|x)) * r + (1 - (π_new(a|x) / π_old(a|x))) * r_hat(x,a) ]
        """
        try:
            n_samples = len(historical_data)

            importance_weights = new_policy_probs / (baseline_probs + 1e-10)

            if reward_model is not None:
                predicted_rewards = reward_model.predict(historical_data[['context', 'action']])
            else:
                predicted_rewards = np.zeros(n_samples)
                for i, row in historical_data.iterrows():
                    action_data = historical_data[historical_data['action'] == row['action']]
                    predicted_rewards[i] = action_data['reward'].mean()

            actual_rewards = historical_data['reward'].values
            dr_estimates = (importance_weights * actual_rewards +
                            (1 - importance_weights) * predicted_rewards)

            estimated_value = np.mean(dr_estimates)
            std_error = np.std(dr_estimates) / np.sqrt(n_samples)

            ci_lower = estimated_value - 1.96 * std_error
            ci_upper = estimated_value + 1.96 * std_error

            return {
                "estimated_value": float(estimated_value),
                "std_error": float(std_error),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "n_samples": n_samples
            }
        except Exception as e:
            logger.error(f"Doubly-Robust estimation failed: {e}")
            return {"error": str(e)}

    def marl_promotion_gate(
        self,
        baseline_value: float,
        marl_value: float,
        marl_ci_lower: float,
        marl_ci_upper: float,
        n_samples: int,
        min_lift_threshold: float = 0.2,  # 20% lift requirement
        min_samples: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        MARL Policy Promotion Gating Mechanism (Section 2.3)

        Requirements for MARL promotion to production:
        1. Sufficient data: n_samples >= min_samples (default 1000)
        2. Significant lift: marl_value >= baseline_value * (1 + min_lift_threshold)
        3. Statistical confidence: 95% CI lower bound > baseline_value
        """
        try:
            data_sufficient = n_samples >= min_samples

            required_lift = baseline_value * (1 + min_lift_threshold)
            lift_achieved = marl_value >= required_lift
            actual_lift_pct = ((marl_value - baseline_value) / baseline_value) * 100

            ci_above_baseline = marl_ci_lower > baseline_value

            gate_passed = data_sufficient and lift_achieved and ci_above_baseline

            rationale = []
            if not data_sufficient:
                rationale.append(f"❌ Insufficient data: {n_samples}/{min_samples} samples")
            else:
                rationale.append(f"✅ Sufficient data: {n_samples} samples")

            if not lift_achieved:
                rationale.append(f"❌ Insufficient lift: {actual_lift_pct:.1f}% < {min_lift_threshold*100}% required")
            else:
                rationale.append(f"✅ Lift achieved: {actual_lift_pct:.1f}% >= {min_lift_threshold*100}%")

            if not ci_above_baseline:
                rationale.append(f"❌ Low confidence: 95% CI [{marl_ci_lower:.4f}, {marl_ci_upper:.4f}] overlaps baseline {baseline_value:.4f}")
            else:
                rationale.append(f"✅ High confidence: 95% CI [{marl_ci_lower:.4f}, {marl_ci_upper:.4f}] > baseline")

            if gate_passed:
                recommendation = {
                    "action": "APPROVE_CANARY",
                    "message": "MARL policy approved for 5% canary rollout",
                    "canary_percentage": 0.05,
                    "monitor_duration_days": 14
                }
            elif data_sufficient and actual_lift_pct > 0:
                recommendation = {
                    "action": "COLLECT_MORE_DATA",
                    "message": "Promising lift detected, but confidence too low. Continue collecting data.",
                    "additional_samples_needed": min_samples - n_samples
                }
            else:
                recommendation = {
                    "action": "REJECT",
                    "message": "MARL policy does not meet promotion criteria. Continue with baseline.",
                    "stay_with_baseline": True
                }

            logger.info(f"MARL Promotion Gate: {recommendation['action']} - {recommendation['message']}")

            return {
                "gate_passed": gate_passed,
                "rationale": rationale,
                "recommendation": recommendation,
                "metrics": {
                    "baseline_value": baseline_value,
                    "marl_value": marl_value,
                    "lift_pct": actual_lift_pct,
                    "ci_lower": marl_ci_lower,
                    "ci_upper": marl_ci_upper,
                    "n_samples": n_samples
                }
            }
        except Exception as e:
            logger.error(f"MARL promotion gate failed: {e}")
            return {
                "gate_passed": False,
                "error": str(e),
                "recommendation": {
                    "action": "ERROR",
                    "message": f"Gate evaluation failed: {e}"
                }
            }

    def _generate_recommendation(
        self,
        comparison: Dict[str, Any]
    ) -> Dict[str, str]:

        best_policy = comparison.get("best_policy", "thompson_sampling")
        best_reward = comparison.get("best_reward", 0.0)

        return {
            "recommended_policy": best_policy,
            "expected_reward": f"{best_reward:.4f}",
            "confidence": "high" if best_reward > 0.5 else "medium",
            "reasoning": f"{best_policy} shows best performance on historical data"
        }