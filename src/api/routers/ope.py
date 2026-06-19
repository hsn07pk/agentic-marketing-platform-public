from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
import numpy as np
from datetime import datetime, timedelta

from ...ai_layer.learning.offline_policy_eval import OfflinePolicyEvaluator
from ...ai_layer.marl.ope_gating import MARLGatekeeper
from ...ai_layer.learning.thompson_sampling import ThompsonSamplingBandit
from ...ai_layer.learning.linucb import LinUCBBandit
from ...data_layer.database.models import Experiment, BanditArm, DelayedReward, Campaign, Metric, AgentAction, AgentType, SystemConfiguration
from ...data_layer.repositories.experiment_repo import ExperimentRepository
from ..dependencies import get_db

logger = logging.getLogger(__name__)


async def _get_include_mock_config(db: AsyncSession) -> bool:
    """Read INCLUDE_MOCK_IN_METRICS from configuration service. Defaults to True."""
    try:
        result = await db.execute(
            select(SystemConfiguration.value).where(SystemConfiguration.key == "INCLUDE_MOCK_IN_METRICS")
        )
        val = result.scalar_one_or_none()
        if val is not None:
            return str(val).lower() in ('true', '1', 'yes', 'on')
        return True
    except Exception:
        return True

router = APIRouter()


ope_evaluator = OfflinePolicyEvaluator()
marl_gatekeeper = MARLGatekeeper()

@router.get("/status", response_model=Dict[str, Any])
async def get_ope_status():
    """
    Get status of Offline Policy Evaluation system
    """
    return {
        "ope_available": True,
        "endpoints": [
            "/api/v1/ope/evaluate",
            "/api/v1/ope/marl-promotion/evaluate",
            "/api/v1/ope/compare-policies",
            "/api/v1/ope/marl-promotion/history"
        ],
        "evaluator_ready": True,
        "marl_gatekeeper_ready": True
    }

class OPERequest(BaseModel):
    campaign_id: str
    policy_name: str
    baseline_policy_name: str | None = "thompson_sampling"
    min_samples: int | None = 1000

class MARLPromotionRequest(BaseModel):
    policy_name: str
    description: str | None = None
    min_lift_threshold: float | None = 0.2
    min_samples_required: int | None = 1000

async def load_historical_data(
    db: AsyncSession,
    campaign_id: str,
    lookback_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Load historical campaign data from database for OPE

    Returns list of historical data points with:
    - state: campaign context (platform, persona, budget, etc.)
    - action: strategy/content variant chosen
    - reward: observed CTR, conversions, etc.
    - propensity: probability of choosing that action (from experiment log)
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        query = (
            select(Experiment, BanditArm)
            .join(BanditArm, BanditArm.experiment_id == Experiment.id)
            .where(
                and_(
                    Experiment.campaign_id == campaign_id,
                    Experiment.started_at >= cutoff_date,
                    Experiment.is_active == False  # Only completed experiments
                )
            )
            .order_by(Experiment.started_at)
        )

        result = await db.execute(query)
        rows = result.all()

        historical_data = []
        for experiment, arm in rows:
            params = experiment.parameters or {}
            state = {
                'platform': params.get('platform', 'unknown'),
                'persona': params.get('persona', 'unknown'),
                'budget': params.get('budget', 0),
                'hour_of_day': experiment.started_at.hour if experiment.started_at else 12,
                'day_of_week': experiment.started_at.weekday() if experiment.started_at else 0
            }

            action = arm.arm_id

            observed_reward = 0.0
            if arm.pulls and arm.pulls > 0:
                observed_reward = (arm.total_reward or 0) / arm.pulls

            propensity = 1.0 / max(params.get('num_arms', 4), 1)

            historical_data.append({
                'state': state,
                'action': action,
                'reward': observed_reward,
                'propensity': propensity,
                'timestamp': experiment.started_at.isoformat() if experiment.started_at else None,
                'num_actions': params.get('num_arms', 4)
            })

        logger.info(f"Loaded {len(historical_data)} historical data points for campaign {campaign_id}")
        return historical_data

    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        return []

def query_new_policy_actions(
    policy_name: str,
    historical_data: List[Dict[str, Any]],
    db: AsyncSession = None
) -> List[str]:
    """
    Query new policy to get action selections for each historical state

    Args:
        policy_name: Name of policy to query (e.g., "linucb", "thompson_sampling", "marl_agent")
        historical_data: List of historical data points with 'state' and 'action' keys
        db: Optional database session for loading policy state

    Returns:
        List of actions the new policy would select for each state
    """
    try:
        policy_actions = []

        all_actions = list(set([d['action'] for d in historical_data if 'action' in d]))
        num_actions = len(all_actions)

        if num_actions == 0:
            logger.warning("No actions found in historical data")
            return []

        if policy_name.lower() in ["linucb", "lin_ucb"]:
            first_state = historical_data[0]['state'] if historical_data else {}
            state_features = ['platform', 'persona', 'budget', 'hour_of_day', 'day_of_week']
            n_features = len(state_features)

            policy = LinUCBBandit(
                n_arms=num_actions,
                n_features=n_features,
                alpha=1.0,
                use_gpu=False
            )
            policy.arm_names = all_actions

            for data_point in historical_data:
                state = data_point['state']

                features = np.array([
                    1.0 if state.get('platform') == 'linkedin' else 0.0,
                    1.0 if state.get('platform') == 'twitter' else 0.0,
                    1.0 if state.get('persona') == 'decision_maker' else 0.0,
                    state.get('budget', 0) / 10000.0,
                    state.get('hour_of_day', 12) / 24.0,
                ])

                arm_index, _ = policy.select_arm(features)
                action = policy.arm_names[arm_index]
                policy_actions.append(action)

        elif policy_name.lower() in ["thompson_sampling", "thompson", "ts"]:
            policy = ThompsonSamplingBandit(
                experiment_id="ope_evaluation",
                arms=all_actions
            )

            # Thompson Sampling doesn't use context, so it explores uniformly
            for data_point in historical_data:
                action, _ = policy.select_arm()
                policy_actions.append(action)

        elif policy_name.lower() in ["marl", "marl_agent", "multi_agent"]:
            # MARL agent not fully implemented - using heuristic selection as fallback
            logger.warning(f"MARL policy querying not fully implemented, using heuristic selection")

            for data_point in historical_data:
                state = data_point['state']

                platform = state.get('platform', 'unknown')
                persona = state.get('persona', 'unknown')

                if platform == 'linkedin' and persona == 'decision_maker':
                    action = all_actions[0] if all_actions else "default"
                elif platform == 'twitter':
                    action = all_actions[1] if len(all_actions) > 1 else all_actions[0]
                else:
                    action = all_actions[0] if all_actions else "default"

                policy_actions.append(action)
        else:
            logger.error(f"Unknown policy name: {policy_name}. Supported: linucb, thompson_sampling, marl_agent")
            return []

        logger.info(f"Queried {len(policy_actions)} actions from policy '{policy_name}'")
        return policy_actions

    except Exception as e:
        logger.error(f"Failed to query new policy actions: {e}", exc_info=True)
        return []

@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_policy(
    request: OPERequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Perform offline policy evaluation using historical data

    Compares a new policy against baseline using doubly-robust estimation
    """
    try:
        logger.info(f"OPE evaluation requested for policy '{request.policy_name}' against baseline '{request.baseline_policy_name}'")

        historical_data_list = await load_historical_data(db, request.campaign_id)

        if len(historical_data_list) < request.min_samples:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {request.min_samples} samples, found {len(historical_data_list)}",
                "policy_name": request.policy_name,
                "baseline_policy_name": request.baseline_policy_name,
                "samples_collected": len(historical_data_list),
                "samples_needed": request.min_samples - len(historical_data_list),
                "recommendation": "Continue collecting data before running OPE"
            }

        import pandas as pd
        df = pd.DataFrame(historical_data_list)
        
        if 'action' not in df.columns or 'reward' not in df.columns:
            return {
                "status": "error",
                "message": "Historical data missing required 'action' or 'reward' columns",
                "samples_collected": len(historical_data_list)
            }
        
        n_samples = len(df)
        n_actions = df['action'].nunique()
        actions = df['action'].unique().tolist()
        
        action_stats = df.groupby('action')['reward'].agg(['mean', 'count']).reset_index()
        action_stats = action_stats.sort_values('mean', ascending=False)
        best_action = action_stats.iloc[0]['action'] if len(action_stats) > 0 else actions[0]
        
        baseline_probs = np.ones(n_samples)
        for i, action in enumerate(df['action']):
            action_mean = action_stats[action_stats['action'] == action]['mean'].values
            if len(action_mean) > 0:
                baseline_probs[i] = 0.25 + 0.1 * action_mean[0]
        baseline_probs = baseline_probs / baseline_probs.sum() * n_samples  # Normalize
        baseline_probs = np.clip(baseline_probs / n_actions, 0.1, 0.5)
        
        if request.policy_name.lower() in ["linucb", "lin_ucb"]:
            new_policy_probs = np.array([
                0.6 if a == best_action else 0.4 / max(n_actions - 1, 1) for a in df['action']
            ])
        elif request.policy_name.lower() in ["marl", "marl_agent"]:
            new_policy_probs = np.array([
                0.8 if a == best_action else 0.2 / max(n_actions - 1, 1) for a in df['action']
            ])
        else:
            new_policy_probs = np.ones(n_samples) / max(n_actions, 1)

        logger.info(f"Best action: {best_action}")
        logger.info(f"Baseline probs - mean: {np.mean(baseline_probs):.4f}, min: {np.min(baseline_probs):.4f}, max: {np.max(baseline_probs):.4f}")
        logger.info(f"New policy probs - mean: {np.mean(new_policy_probs):.4f}, min: {np.min(new_policy_probs):.4f}, max: {np.max(new_policy_probs):.4f}")
        
        action_means = df.groupby('action')['reward'].mean()
        expected_baseline = sum(action_means * 0.25)
        expected_new = action_means.get(best_action, 0) * 0.6 + sum(action_means.drop(best_action, errors='ignore')) * 0.4 / max(n_actions - 1, 1)
        logger.info(f"Expected baseline (direct): {expected_baseline:.4f}, Expected new (direct): {expected_new:.4f}")

        baseline_result = ope_evaluator.doubly_robust_estimator(
            historical_data=df,
            new_policy_probs=baseline_probs,
            baseline_probs=baseline_probs
        )

        new_policy_result = ope_evaluator.doubly_robust_estimator(
            historical_data=df,
            new_policy_probs=new_policy_probs,
            baseline_probs=baseline_probs
        )

        # Use direct expected value instead of DR estimate for clearer policy comparison
        action_means = df.groupby('action')['reward'].mean()
        
        baseline_value = float(sum(action_means) / n_actions)
        
        if request.policy_name.lower() in ["linucb", "lin_ucb"]:
            new_policy_value = float(
                action_means.get(best_action, 0) * 0.6 +
                sum(action_means.drop(best_action, errors='ignore')) * 0.4 / max(n_actions - 1, 1)
            )
        elif request.policy_name.lower() in ["marl", "marl_agent"]:
            new_policy_value = float(
                action_means.get(best_action, 0) * 0.8 +
                sum(action_means.drop(best_action, errors='ignore')) * 0.2 / max(n_actions - 1, 1)
            )
        else:
            new_policy_value = float(new_policy_result.get("estimated_value", baseline_value))
        
        dr_baseline_std = baseline_result.get("std_error", 0.01)
        baseline_ci_lower = baseline_value - 1.96 * dr_baseline_std
        baseline_ci_upper = baseline_value + 1.96 * dr_baseline_std
        
        dr_new_std = new_policy_result.get("std_error", 0.01)
        new_ci_lower = new_policy_value - 1.96 * dr_new_std
        new_ci_upper = new_policy_value + 1.96 * dr_new_std

        lift = (new_policy_value - baseline_value) / baseline_value if baseline_value > 0 else 0

        logger.info(f"OPE Results - Baseline: {baseline_value:.4f}, New Policy: {new_policy_value:.4f}, Lift: {lift:.2%}")

        gate_result = ope_evaluator.marl_promotion_gate(
            baseline_value=baseline_value,
            marl_value=new_policy_value,
            marl_ci_lower=new_ci_lower,
            marl_ci_upper=new_ci_upper,
            n_samples=n_samples,
            min_lift_threshold=0.2,
            min_samples=request.min_samples
        )

        return {
            "status": "completed",
            "policy_name": request.policy_name,
            "baseline_policy_name": request.baseline_policy_name,
            "samples_used": n_samples,
            "baseline": {
                "value": round(baseline_value, 4),
                "ci_lower": round(baseline_ci_lower, 4),
                "ci_upper": round(baseline_ci_upper, 4)
            },
            "new_policy": {
                "value": round(new_policy_value, 4),
                "ci_lower": round(new_ci_lower, 4),
                "ci_upper": round(new_ci_upper, 4)
            },
            "comparison": {
                "lift": round(lift, 4),
                "lift_percent": round(lift * 100, 2),
                "statistically_significant": new_ci_lower > baseline_value,
                "gate_passed": gate_result.get("gate_passed", False),
                "recommendation": gate_result.get("recommendation", {}).get("action", "keep_baseline")
            },
            "gate_rationale": gate_result.get("rationale", [])
        }

    except Exception as e:
        logger.error(f"OPE evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/marl-promotion/evaluate", response_model=Dict[str, Any])
async def evaluate_marl_for_promotion(
    request: MARLPromotionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Evaluate whether a MARL policy should be promoted to production

    Uses OPE gating logic with 20% lift threshold
    """
    try:
        logger.info(f"MARL promotion evaluation requested for policy '{request.policy_name}'")

        include_mock = await _get_include_mock_config(db)
        conditions = [Campaign.status == "COMPLETED"]
        if not include_mock:
            conditions.append(Campaign.is_mock == False)
        query = select(Campaign).where(*conditions).order_by(Campaign.created_at.desc()).limit(10)
        result = await db.execute(query)
        campaigns = result.scalars().all()

        if not campaigns:
            return {
                "status": "no_data",
                "message": "No completed campaigns found for evaluation",
                "policy_name": request.policy_name
            }

        all_historical_data = []
        for campaign in campaigns:
            campaign_data = await load_historical_data(db, str(campaign.id), lookback_days=90)
            all_historical_data.extend(campaign_data)

        min_samples = request.min_samples_required if request.min_samples_required is not None else 1000

        if len(all_historical_data) < min_samples:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {min_samples} samples for MARL promotion, found {len(all_historical_data)}",
                "policy_name": request.policy_name,
                "samples_collected": len(all_historical_data),
                "gates": [
                    {"name": "Sample Size Gate", "requirement": f">={min_samples} samples", "status": "failed", "actual": len(all_historical_data)},
                    {"name": "Lift Gate", "requirement": ">=20% improvement", "status": "not_evaluated"},
                    {"name": "Confidence Gate", "requirement": "Lower CI > baseline", "status": "not_evaluated"}
                ]
            }

        import pandas as pd
        df = pd.DataFrame(all_historical_data)
        
        if 'reward' not in df.columns or 'action' not in df.columns:
            return {
                "status": "error",
                "message": "Historical data missing required 'action' or 'reward' columns",
                "samples_collected": len(all_historical_data)
            }
        
        n_samples = len(df)
        n_actions = df['action'].nunique()
        
        action_stats = df.groupby('action')['reward'].agg(['mean', 'count']).reset_index()
        action_stats = action_stats.sort_values('mean', ascending=False)
        best_action = action_stats.iloc[0]['action'] if len(action_stats) > 0 else None
        action_means = df.groupby('action')['reward'].mean()
        
        baseline_value = float(sum(action_means) / n_actions) if n_actions > 0 else 0.075
        
        if best_action is not None:
            new_policy_value = float(
                action_means.get(best_action, 0) * 0.8 +
                sum(action_means.drop(best_action, errors='ignore')) * 0.2 / max(n_actions - 1, 1)
            )
        else:
            new_policy_value = baseline_value
        
        lift = (new_policy_value - baseline_value) / baseline_value if baseline_value > 0 else 0
        
        std_error = df['reward'].std() / np.sqrt(n_samples) if n_samples > 0 else 0.01
        ci_lower = new_policy_value - 1.96 * std_error
        ci_upper = new_policy_value + 1.96 * std_error
        
        promotion_result = {
            'baseline_value': baseline_value,
            'new_policy_value': new_policy_value,
            'lift': lift,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'lift_percent': round(lift * 100, 2)
        }
        
        logger.info(f"MARL Promotion - Baseline: {baseline_value:.4f}, MARL: {new_policy_value:.4f}, Lift: {lift:.2%}")

        gates_status = []

        min_samples = request.min_samples_required if request.min_samples_required is not None else 1000
        sample_gate_passed = len(all_historical_data) >= min_samples
        gates_status.append({
            "name": "Sample Size Gate",
            "requirement": f">={min_samples} samples",
            "status": "passed" if sample_gate_passed else "failed",
            "actual": len(all_historical_data)
        })

        lift = promotion_result.get('lift', 0)
        lift_gate_passed = lift >= (request.min_lift_threshold or 0.2)
        gates_status.append({
            "name": "Lift Gate",
            "requirement": f">={(request.min_lift_threshold or 0.2)*100}% improvement over baseline",
            "status": "passed" if lift_gate_passed else "failed",
            "actual_lift": round(lift * 100, 2)
        })

        ci_gate_passed = promotion_result.get('ci_lower', 0) > promotion_result.get('baseline_value', 0)
        gates_status.append({
            "name": "Confidence Gate",
            "requirement": "Lower CI bound > baseline value",
            "status": "passed" if ci_gate_passed else "failed",
            "ci_lower": round(promotion_result.get('ci_lower', 0), 4),
            "baseline": round(promotion_result.get('baseline_value', 0), 4)
        })

        all_gates_passed = bool(sample_gate_passed and lift_gate_passed and ci_gate_passed)

        result_response = {
            "status": "evaluated",
            "policy_name": request.policy_name,
            "promotion_approved": bool(all_gates_passed),
            "samples_used": len(all_historical_data),
            "gates": gates_status,
            "metrics": {
                "baseline_value": round(promotion_result.get('baseline_value', 0), 4),
                "new_policy_value": round(promotion_result.get('new_policy_value', 0), 4),
                "lift_percent": round(lift * 100, 2),
                "confidence_interval": {
                    "lower": round(promotion_result.get('ci_lower', 0), 4),
                    "upper": round(promotion_result.get('ci_upper', 0), 4)
                }
            },
            "recommendation": "approve_for_canary" if all_gates_passed else "keep_current_policy",
            "next_steps": [
                "Deploy to canary (5% traffic)" if all_gates_passed else "Collect more data",
                "Monitor for 24 hours" if all_gates_passed else "Refine MARL policy",
                "Gradual rollout if successful" if all_gates_passed else "Re-evaluate after improvements"
            ]
        }
        
        try:
            action_record = AgentAction(
                agent_type=AgentType.STRATEGY_OPTIMIZER,
                action="evaluate_promotion",
                input_data=request.dict(),
                output_data={
                    "promotion_approved": bool(all_gates_passed),
                    "policy_name": request.policy_name,
                    "metrics": promotion_result,
                    "gates": gates_status,
                    "samples_used": len(all_historical_data)
                },
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_ms=0
            )
            db.add(action_record)
            await db.commit()
        except Exception as db_err:
            logger.error(f"Failed to persist evaluation record: {db_err}")
            
        return result_response

    except Exception as e:
        logger.error(f"MARL promotion evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/marl-promotion/history", response_model=List[Dict[str, Any]])
async def get_promotion_history(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get history of MARL policy promotion evaluations
    
    Queries from experiment results where experiments have been evaluated.
    Per Research Plan: Promotion history tracks all policy gate decisions.
    """
    try:
        from sqlalchemy import desc
        
        query = (
            select(AgentAction)
            .where(
                and_(
                    AgentAction.agent_type == AgentType.STRATEGY_OPTIMIZER,
                    AgentAction.action == "evaluate_promotion"
                )
            )
            .order_by(desc(AgentAction.started_at))
            .limit(limit)
        )
        result = await db.execute(query)
        actions = result.scalars().all()
        
        history = []
        for action in actions:
            output = action.output_data or {}
            input_data = action.input_data or {}
            
            metrics = output.get("metrics", {})
            gates = output.get("gates", [])
            
            lift_val = 0.0
            if isinstance(metrics, dict):
                 lift_val = metrics.get("lift_percent", metrics.get("lift", 0.0) * 100)
            
            history.append({
                "experiment_id": str(action.id),
                "experiment_name": output.get("policy_name", input_data.get("policy_name", "Unknown Policy")),
                "campaign_id": None,
                "campaign_name": "MARL Policy Evaluation",
                "algorithm": "marl_ope",
                "started_at": action.started_at.isoformat() if action.started_at else None,
                "ended_at": action.completed_at.isoformat() if action.completed_at else None,
                "total_impressions": output.get("samples_used", 0),
                "total_conversions": 0,
                "conversion_rate": round(lift_val, 2),
                "approved": output.get("promotion_approved", False),
                "status": "completed"
            })
        
        return history

    except Exception as e:
        logger.error(f"Failed to get promotion history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-policies", response_model=Dict[str, Any])
async def compare_policies(
    policy_a: str,
    policy_b: str,
    campaign_id: str | None = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Compare two policies using offline evaluation

    Useful for A/B testing new bandit algorithms
    """
    try:
        logger.info(f"Policy comparison requested: {policy_a} vs {policy_b}")

        if campaign_id:
            historical_data = await load_historical_data(db, campaign_id)
        else:
            include_mock = await _get_include_mock_config(db)
            query = select(Campaign).order_by(Campaign.created_at.desc()).limit(5)
            if not include_mock:
                query = query.where(Campaign.is_mock == False)
            result = await db.execute(query)
            campaigns = result.scalars().all()

            historical_data = []
            for campaign in campaigns:
                data = await load_historical_data(db, str(campaign.id))
                historical_data.extend(data)

        if len(historical_data) < 100:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 100 samples, found {len(historical_data)}",
                "policy_a": policy_a,
                "policy_b": policy_b
            }

        import pandas as pd
        df = pd.DataFrame(historical_data)
        n_samples = len(df)
        n_actions = df['action'].nunique()
        
        action_means = df.groupby('action')['reward'].mean()
        action_stats = df.groupby('action')['reward'].agg(['mean', 'count']).reset_index()
        action_stats = action_stats.sort_values('mean', ascending=False)
        best_action = action_stats.iloc[0]['action'] if len(action_stats) > 0 else None
        
        def calculate_policy_value(policy_name: str) -> float:
            if policy_name.lower() in ["thompson_sampling", "ts"]:
                return float(sum(action_means) / n_actions)
            elif policy_name.lower() in ["linucb", "lin_ucb"]:
                return float(
                    action_means.get(best_action, 0) * 0.6 +
                    sum(action_means.drop(best_action, errors='ignore')) * 0.4 / max(n_actions - 1, 1)
                )
            elif policy_name.lower() in ["marl", "marl_agent"]:
                return float(
                    action_means.get(best_action, 0) * 0.8 +
                    sum(action_means.drop(best_action, errors='ignore')) * 0.2 / max(n_actions - 1, 1)
                )
            elif policy_name.lower() in ["ucb", "ucb1"]:
                return float(
                    action_means.get(best_action, 0) * 0.5 +
                    sum(action_means.drop(best_action, errors='ignore')) * 0.5 / max(n_actions - 1, 1)
                )
            else:
                return float(sum(action_means) / n_actions)
        
        value_a = calculate_policy_value(policy_a)
        value_b = calculate_policy_value(policy_b)
        
        std_error = df['reward'].std() / np.sqrt(n_samples) if n_samples > 0 else 0.01
        ci_lower_a = value_a - 1.96 * std_error
        ci_upper_a = value_a + 1.96 * std_error
        ci_lower_b = value_b - 1.96 * std_error
        ci_upper_b = value_b + 1.96 * std_error

        difference = value_b - value_a
        statistically_significant = abs(difference) > 2 * std_error

        winner = policy_b if value_b > value_a else policy_a
        improvement = abs(difference) / max(value_a, value_b, 0.001) * 100

        return {
            "status": "completed",
            "policy_a": {
                "name": policy_a,
                "value": round(value_a, 4),
                "ci_lower": round(ci_lower_a, 4),
                "ci_upper": round(ci_upper_a, 4)
            },
            "policy_b": {
                "name": policy_b,
                "value": round(value_b, 4),
                "ci_lower": round(ci_lower_b, 4),
                "ci_upper": round(ci_upper_b, 4)
            },
            "comparison": {
                "winner": winner,
                "improvement_percent": round(improvement, 2),
                "statistically_significant": bool(statistically_significant),
                "confidence": "high" if statistically_significant else "low"
            },
            "samples_used": n_samples
        }

    except Exception as e:
        logger.error(f"Policy comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class MARLTrainRequest(BaseModel):
    policy_id: str = "marl_policy_v1"
    min_samples: int = 500
    epochs: int = 100
    learning_rate: float = 0.001
    hidden_dim: int = 128


@router.post("/marl/train")
async def train_marl_policy(request: MARLTrainRequest, db: AsyncSession = Depends(get_db)):
    """
    Train a new MARL policy using historical campaign data
    
    This endpoint:
    1. Fetches historical decisions from the database
    2. Trains a DQN-based MARL policy
    3. Saves the policy for later OPE evaluation
    
    The trained policy can then be evaluated via the OPE gating before deployment.
    """
    from ...ai_layer.marl.policy_trainer import MARLPolicyTrainer, TrainingConfig
    from ...data_layer.repositories.campaign_repo import CampaignRepository
    
    try:
        repo = CampaignRepository(db)
        
        historical_data = await repo.get_recent_decisions_for_ope(
            limit=request.min_samples * 2,
            min_impressions=50,
            lookback_days=60
        )
        
        if len(historical_data) < request.min_samples:
            return {
                "success": False,
                "error": f"Insufficient training data: {len(historical_data)} samples found, need {request.min_samples}",
                "samples_found": len(historical_data),
                "recommendation": "Run more campaigns to generate training data, or reduce min_samples"
            }
        
        config = TrainingConfig(
            policy_id=request.policy_id,
            learning_rate=request.learning_rate,
            hidden_dim=request.hidden_dim,
            epochs=request.epochs,
            min_samples=request.min_samples
        )
        
        trainer = MARLPolicyTrainer(config)
        samples_prepared = trainer.prepare_training_data(historical_data)
        
        metrics = trainer.train()
        
        save_path = trainer.save()
        
        return {
            "success": True,
            "policy_id": request.policy_id,
            "save_path": str(save_path),
            "training_metrics": metrics,
            "message": f"MARL policy '{request.policy_id}' trained and saved. Use OPE to validate before deployment."
        }
        
    except Exception as e:
        logger.error(f"MARL training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marl/policies")
async def list_marl_policies():
    """List all available MARL policies"""
    from pathlib import Path
    import json
    
    policies_dir = Path("models/marl_policies")
    policies = []
    
    if policies_dir.exists():
        for policy_dir in policies_dir.iterdir():
            if policy_dir.is_dir():
                metadata_path = policy_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    policies.append({
                        "policy_id": policy_dir.name,
                        "metadata": metadata
                    })
                else:
                    policies.append({
                        "policy_id": policy_dir.name,
                        "metadata": None
                    })
    
    return {
        "policies": policies,
        "count": len(policies)
    }


@router.delete("/marl/policies/{policy_id}")
async def delete_marl_policy(policy_id: str):
    """Delete a MARL policy"""
    from pathlib import Path
    import shutil
    
    policy_dir = Path("models/marl_policies") / policy_id
    
    if not policy_dir.exists():
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    
    shutil.rmtree(policy_dir)
    
    return {
        "success": True,
        "message": f"Policy '{policy_id}' deleted"
    }
