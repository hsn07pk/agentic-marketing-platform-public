from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import numpy as np

from ...data_layer.repositories.experiment_repo import ExperimentRepository
from ...ai_layer.learning.offline_policy_eval import OfflinePolicyEvaluator
from ...ai_layer.learning.mlflow_integration import log_experiment_completion
from ..dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("", response_model=List[Dict[str, Any]])
async def list_experiments(
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    try:
        from ...data_layer.database.models import Experiment
        from sqlalchemy import select

        query = select(Experiment).offset(offset).limit(limit).order_by(Experiment.started_at.desc())
        result = await db.execute(query)
        experiments = result.scalars().all()

        return [
            {
                "id": str(exp.id),
                "campaign_id": str(exp.campaign_id) if exp.campaign_id else None,
                "name": exp.name,
                "algorithm": exp.algorithm,
                "is_active": exp.is_active,
                "total_impressions": exp.total_impressions,
                "total_conversions": exp.total_conversions,
                "started_at": exp.started_at.isoformat() if exp.started_at else None,
                "ended_at": exp.ended_at.isoformat() if exp.ended_at else None
            }
            for exp in experiments
        ]
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=Dict[str, Any])
async def create_experiment(
    experiment_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    try:
        repo = ExperimentRepository(db)
        experiment = await repo.create(experiment_data)
        
        return {
            "id": str(experiment.id),
            "campaign_id": str(experiment.campaign_id) if experiment.campaign_id else None,
            "name": experiment.name,
            "algorithm": experiment.algorithm,
            "is_active": experiment.is_active,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None
        }
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/campaign/{campaign_id}", response_model=List[Dict[str, Any]])
async def get_campaign_experiments(
    campaign_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        repo = ExperimentRepository(db)
        experiments = await repo.get_by_campaign(campaign_id)
        
        return [
            {
                "id": str(exp.id),
                "name": exp.name,
                "algorithm": exp.algorithm,
                "is_active": exp.is_active,
                "started_at": exp.started_at.isoformat() if exp.started_at else None
            }
            for exp in experiments
        ]
    except Exception as e:
        logger.error(f"Failed to get experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{experiment_id}", response_model=Dict[str, Any])
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        repo = ExperimentRepository(db)
        experiment = await repo.get_by_id(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "id": str(experiment.id),
            "campaign_id": str(experiment.campaign_id) if experiment.campaign_id else None,
            "name": experiment.name,
            "algorithm": experiment.algorithm,
            "is_active": experiment.is_active,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{experiment_id}/status")
async def get_experiment_status(
    experiment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get current experiment status and recommendations"""
    try:
        repo = ExperimentRepository(db)
        experiment = await repo.get_by_id(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        results = experiment.results or {}
        
        variants = {}
        if 'arm_stats' in results:
            for arm_id, stats in results['arm_stats'].items():
                pulls = stats.get('pulls', 0)
                reward = stats.get('total_reward', 0)

                # Calculate CTR as decimal for internal calculations
                ctr_decimal = reward / pulls if pulls > 0 else 0.0

                confidence = 0.0
                if pulls > 10:
                    std_error = np.sqrt(ctr_decimal * (1 - ctr_decimal) / pulls)
                    z_score = abs(ctr_decimal - results.get('baseline_ctr', 0.02)) / std_error if std_error > 0 else 0
                    confidence = min(0.99, 1 - np.exp(-z_score))

                # Return CTR as PERCENTAGE (0-100 scale) for consistency with other APIs
                variants[arm_id] = {
                    "ctr": round(ctr_decimal * 100, 2),
                    "pulls": pulls,
                    "confidence": round(confidence, 2)
                }
        
        recommendation = _generate_recommendation(variants, "active" if experiment.is_active else "completed")

        return {
            "experiment_id": experiment_id,
            "is_active": experiment.is_active,
            "variants": variants,
            "recommendation": recommendation,
            "total_samples": results.get('total_samples', 0)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_recommendation(
    variants: Dict[str, Dict[str, Any]],
    status: str
) -> Dict[str, Any]:
    """
    Generate experiment recommendation based on real data.
    """
    if not variants or len(variants) < 2:
        return {
            "action": "continue",
            "winner": None,
            "confidence": 0.0,
            "rationale": "Insufficient data for recommendation"
        }
    
    sorted_variants = sorted(
        variants.items(),
        key=lambda x: x[1]['ctr'],
        reverse=True
    )
    
    winner = sorted_variants[0]
    second_best = sorted_variants[1]
    
    winner_confidence = winner[1].get('confidence', 0.0)
    winner_pulls = winner[1].get('pulls', 0)
    
    if winner_confidence >= 0.95 and winner_pulls >= 30:
        action = "promote"
        rationale = f"Winner shows {(winner[1]['ctr'] / second_best[1]['ctr'] - 1) * 100:.1f}% improvement with {winner_confidence:.0%} confidence"
    elif winner_confidence >= 0.8 and winner_pulls >= 20:
        action = "promote_cautiously"
        rationale = f"Winner shows promise but needs more data. Current confidence: {winner_confidence:.0%}"
    elif status == "completed":
        action = "promote"
        rationale = "Experiment completed. Promoting best performing variant."
    else:
        action = "continue"
        rationale = f"Need more data. Current confidence: {winner_confidence:.0%}, samples: {winner_pulls}"
    
    return {
        "action": action,
        "winner": winner[0],
        "confidence": winner_confidence,
        "rationale": rationale
    }

@router.post("/{experiment_id}/results")
async def update_experiment_results(
    experiment_id: str,
    results: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    try:
        repo = ExperimentRepository(db)
        success = await repo.update_results(experiment_id, results)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update results")
        
        return {"message": "Results updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/campaign/{campaign_id}/summary")
async def get_experiment_summary(
    campaign_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        repo = ExperimentRepository(db)
        summary = await repo.get_experiment_summary(campaign_id)
        
        return summary
    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{experiment_id}/simulate")
async def simulate_experiment_traffic(
    experiment_id: str,
    num_pulls: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    Simulate traffic for an experiment (for testing bandit algorithms).

    This is a legitimate testing/research feature per Research Plan Section 2.3 (Multi-Armed Bandits).
    """
    try:
        from ...automation_layer.experiment_simulator import run_experiment_simulation

        result = await run_experiment_simulation(experiment_id, num_pulls, db)
        
        # Tag result as simulation data (distinct from mock campaign data)
        result['is_simulation'] = True
        result['simulation_purpose'] = 'bandit_algorithm_testing'

        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to simulate experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{experiment_id}/check-completion")
async def check_experiment_completion(
    experiment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Check if experiment should complete based on Research Plan Section 2.3 criteria:
    - Target sample size reached
    - Statistical significance achieved
    - Duration expired
    - Early stopping (if enabled)
    """
    try:
        from ...ai_layer.learning.experiment_completion import should_complete_experiment
        from ...data_layer.database.models import Experiment, BanditArm
        from sqlalchemy import select
        from uuid import UUID

        stmt = select(Experiment).where(Experiment.id == UUID(experiment_id))
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        stmt = select(BanditArm).where(BanditArm.experiment_id == UUID(experiment_id))
        result = await db.execute(stmt)
        arms = result.scalars().all()

        experiment_dict = {
            "id": str(experiment.id),
            "name": experiment.name,
            "algorithm": experiment.algorithm,
            "is_active": experiment.is_active,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "parameters": experiment.parameters or {}
        }

        arms_data = [
            {
                "arm_id": arm.arm_id,
                "pulls": arm.pulls,
                "successes": arm.successes,
                "failures": arm.failures,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "total_reward": arm.total_reward
            }
            for arm in arms
        ]

        decision = should_complete_experiment(experiment_dict, arms_data)

        if decision.should_complete and experiment.is_active:
            experiment.is_active = False
            experiment.ended_at = datetime.utcnow()
            experiment.winner_variant = decision.winner
            await db.commit()

            logger.info(f"✅ Experiment {experiment_id} completed: {decision.reason}")

            # Log experiment completion to MLflow (Research Plan Section 8.1)
            try:
                metrics = {
                    "total_pulls": sum(arm.get("pulls", 0) for arm in arms_data),
                    "total_successes": sum(arm.get("successes", 0) for arm in arms_data),
                    "confidence": decision.confidence,
                    "winner_ctr": next(
                        (arm.get("successes", 0) / max(arm.get("pulls", 1), 1)
                         for arm in arms_data if arm.get("arm_id") == decision.winner),
                        0.0
                    )
                }
                run_id = await log_experiment_completion(
                    experiment_id=experiment_id,
                    experiment_name=experiment.name,
                    algorithm=experiment.algorithm,
                    arms=arms_data,
                    winner_arm=decision.winner,
                    metrics=metrics,
                    campaign_id=str(experiment.campaign_id) if experiment.campaign_id else None
                )
                if run_id:
                    logger.info(f"📊 Logged experiment to MLflow: run_id={run_id}")
            except Exception as mlflow_error:
                logger.warning(f"Failed to log experiment to MLflow: {mlflow_error}")

        return {
            "experiment_id": experiment_id,
            "should_complete": decision.should_complete,
            "reason": decision.reason,
            "winner": decision.winner,
            "confidence": decision.confidence,
            "recommendation": decision.recommendation,
            "completed": decision.should_complete and not experiment.is_active
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate/{campaign_id}")
async def evaluate_policies(
    campaign_id: str
):
    try:
        evaluator = OfflinePolicyEvaluator()
        report = await evaluator.generate_ope_report(campaign_id)

        return report
    except Exception as e:
        logger.error(f"Failed to evaluate policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/regret")
async def get_experiment_regret(
    experiment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Calculate cumulative regret for a Thompson Sampling/LinUCB experiment.

    Research Plan Reference: Section 10.2 - "Bandit Regret: Minimize over time"

    Regret = Σ(optimal_reward - actual_reward) over all pulls
    """
    try:
        from ...data_layer.database.models import Experiment, BanditArm, Metric
        from sqlalchemy import select
        from uuid import UUID
        from collections import defaultdict

        stmt = select(Experiment).where(Experiment.id == UUID(experiment_id))
        result = await db.execute(stmt)
        experiment = result.scalar_one_or_none()

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        arms_stmt = select(BanditArm).where(BanditArm.experiment_id == UUID(experiment_id))
        arms_result = await db.execute(arms_stmt)
        arms = arms_result.scalars().all()

        if not arms:
            return {
                "experiment_id": experiment_id,
                "has_data": False,
                "message": "No bandit arms found for this experiment",
                "cumulative_regret": 0.0,
                "regret_trend": [],
                "optimal_arm_id": None,
                "estimated_optimal_ctr": 0.0,
                "regret_per_arm": {}
            }

        # NOTE: CTR is stored internally as DECIMAL for regret calculations
        # but displayed as PERCENTAGE in the response for consistency
        arm_stats = {}
        for arm in arms:
            pulls = arm.pulls or 0
            successes = arm.successes or 0
            ctr_decimal = successes / pulls if pulls > 0 else 0.0

            arm_stats[arm.arm_id] = {
                "pulls": pulls,
                "successes": successes,
                "ctr_decimal": ctr_decimal,
                "alpha": arm.alpha or 1.0,
                "beta": arm.beta or 1.0,
                "name": arm.arm_id
            }

        if not arm_stats:
            return {
                "experiment_id": experiment_id,
                "has_data": False,
                "message": "No arm statistics available",
                "cumulative_regret": 0.0,
                "regret_trend": [],
                "optimal_arm_id": None,
                "estimated_optimal_ctr": 0.0,
                "regret_per_arm": {}
            }

        optimal_arm_id = max(arm_stats.keys(), key=lambda k: arm_stats[k]['ctr_decimal'])
        optimal_ctr_decimal = arm_stats[optimal_arm_id]['ctr_decimal']

        regret_per_arm = {}
        cumulative_regret = 0.0

        for arm_id, stats in arm_stats.items():
            arm_regret = (optimal_ctr_decimal - stats['ctr_decimal']) * stats['pulls']
            regret_per_arm[arm_id] = {
                "name": stats['name'],
                "regret": round(arm_regret, 4),
                "pulls": stats['pulls'],
                "ctr": round(stats['ctr_decimal'] * 100, 2)
            }
            cumulative_regret += arm_regret

        regret_trend = []
        try:
            if experiment.campaign_id:
                metrics_stmt = select(Metric).where(
                    Metric.campaign_id == experiment.campaign_id
                ).order_by(Metric.recorded_at)
                metrics_result = await db.execute(metrics_stmt)
                metrics = metrics_result.scalars().all()

                if metrics:
                    daily_regret = defaultdict(float)
                    running_optimal = 0.0
                    running_actual = 0.0

                    for metric in metrics:
                        date_str = metric.recorded_at.strftime("%Y-%m-%d") if metric.recorded_at else "unknown"
                        impressions = metric.impressions or 0
                        conversions = metric.conversions or 0

                        daily_actual = conversions
                        daily_optimal = impressions * optimal_ctr

                        daily_regret[date_str] += (daily_optimal - daily_actual)

                    running_total = 0.0
                    for date_str in sorted(daily_regret.keys()):
                        running_total += daily_regret[date_str]
                        regret_trend.append({
                            "date": date_str,
                            "daily_regret": round(daily_regret[date_str], 4),
                            "cumulative_regret": round(running_total, 4)
                        })
        except Exception as trend_error:
            logger.warning(f"Could not calculate regret trend: {trend_error}")

        total_pulls = sum(stats['pulls'] for stats in arm_stats.values())
        regret_per_1000 = (cumulative_regret / total_pulls * 1000) if total_pulls > 0 else 0.0

        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "algorithm": experiment.algorithm,
            "has_data": True,
            "cumulative_regret": round(cumulative_regret, 4),
            "regret_per_1000_pulls": round(regret_per_1000, 4),
            "total_pulls": total_pulls,
            "optimal_arm_id": optimal_arm_id,
            "optimal_arm_name": arm_stats[optimal_arm_id]['name'],
            "estimated_optimal_ctr": round(optimal_ctr_decimal * 100, 2),
            "regret_per_arm": regret_per_arm,
            "regret_trend": regret_trend,
            "interpretation": _interpret_regret(cumulative_regret, regret_per_1000, total_pulls)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate experiment regret: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _interpret_regret(cumulative: float, per_1000: float, total_pulls: int) -> str:
    """Generate human-readable interpretation of regret metrics."""
    if total_pulls < 100:
        return "Insufficient data for reliable regret estimation. Need at least 100 pulls."

    if per_1000 < 5:
        return f"Excellent: Very low regret ({per_1000:.1f} per 1000 pulls). The bandit is exploring efficiently."
    elif per_1000 < 20:
        return f"Good: Moderate regret ({per_1000:.1f} per 1000 pulls). Algorithm is learning effectively."
    elif per_1000 < 50:
        return f"Fair: Higher regret ({per_1000:.1f} per 1000 pulls). Consider if exploration is balanced."
    else:
        return f"High regret ({per_1000:.1f} per 1000 pulls). Review arm performance and consider early stopping."


@router.get("/campaign/{campaign_id}/regret")
async def get_campaign_regret_summary(
    campaign_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        from ...data_layer.database.models import Experiment
        from sqlalchemy import select
        from uuid import UUID

        stmt = select(Experiment).where(Experiment.campaign_id == UUID(campaign_id))
        result = await db.execute(stmt)
        experiments = result.scalars().all()

        if not experiments:
            return {
                "campaign_id": campaign_id,
                "experiments": [],
                "total_regret": 0.0,
                "message": "No experiments found for this campaign"
            }

        experiment_regrets = []
        total_regret = 0.0

        for exp in experiments:
            try:
                regret_result = await get_experiment_regret(str(exp.id), db)
                if regret_result.get("has_data"):
                    experiment_regrets.append({
                        "experiment_id": str(exp.id),
                        "experiment_name": exp.name,
                        "algorithm": exp.algorithm,
                        "cumulative_regret": regret_result.get("cumulative_regret", 0),
                        "regret_per_1000": regret_result.get("regret_per_1000_pulls", 0),
                        "total_pulls": regret_result.get("total_pulls", 0),
                        "optimal_arm": regret_result.get("optimal_arm_name", "Unknown")
                    })
                    total_regret += regret_result.get("cumulative_regret", 0)
            except Exception as e:
                logger.warning(f"Could not get regret for experiment {exp.id}: {e}")

        return {
            "campaign_id": campaign_id,
            "experiments": experiment_regrets,
            "total_regret": round(total_regret, 4),
            "experiment_count": len(experiment_regrets)
        }

    except Exception as e:
        logger.error(f"Failed to get campaign regret summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))