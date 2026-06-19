"""Bandit Arms API - Research Plan Section 2.3 (Multi-Armed Bandits)"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any
from uuid import UUID
import logging

from ...data_layer.database.models import BanditArm
from ..dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/experiment/{experiment_id}", response_model=List[Dict[str, Any]])
async def get_experiment_arms(
    experiment_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        stmt = select(BanditArm).where(BanditArm.experiment_id == UUID(experiment_id))
        result = await db.execute(stmt)
        arms = result.scalars().all()

        return [
            {
                "id": str(arm.id),
                "arm_id": arm.arm_id,
                "variant_data": arm.variant_data,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "pulls": arm.pulls,
                "successes": arm.successes,
                "failures": arm.failures,
                "total_reward": arm.total_reward,
                "last_pulled_at": arm.last_pulled_at.isoformat() if arm.last_pulled_at else None
            }
            for arm in arms
        ]

    except Exception as e:
        logger.error(f"Failed to get arms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{arm_id}", response_model=Dict[str, Any])
async def get_arm(
    arm_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        stmt = select(BanditArm).where(BanditArm.id == UUID(arm_id))
        result = await db.execute(stmt)
        arm = result.scalar_one_or_none()

        if not arm:
            raise HTTPException(status_code=404, detail="Arm not found")

        return {
            "id": str(arm.id),
            "experiment_id": str(arm.experiment_id),
            "arm_id": arm.arm_id,
            "variant_data": arm.variant_data,
            "alpha": arm.alpha,
            "beta": arm.beta,
            "pulls": arm.pulls,
            "successes": arm.successes,
            "failures": arm.failures,
            "total_reward": arm.total_reward,
            "context_vector": arm.context_vector,
            "last_pulled_at": arm.last_pulled_at.isoformat() if arm.last_pulled_at else None,
            "created_at": arm.created_at.isoformat() if arm.created_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get arm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{arm_id}/reward", response_model=Dict[str, Any])
async def update_arm_reward(
    arm_id: str,
    reward_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """
    Closed-loop data flow:
    Campaign → Content → Deployment → Metrics → BanditArm → OPE
    """
    try:
        from datetime import datetime
        
        stmt = select(BanditArm).where(BanditArm.id == UUID(arm_id))
        result = await db.execute(stmt)
        arm = result.scalar_one_or_none()

        if not arm:
            raise HTTPException(status_code=404, detail="Arm not found")
        
        reward = reward_data.get("reward", 0)
        is_success = reward_data.get("is_success", reward > 0)
        
        arm.pulls = (arm.pulls or 0) + 1
        arm.total_reward = (arm.total_reward or 0) + reward
        arm.last_pulled_at = datetime.utcnow()
        
        if is_success:
            arm.successes = (arm.successes or 0) + 1
            arm.alpha = (arm.alpha or 1.0) + 1  # Bayesian update
        else:
            arm.failures = (arm.failures or 0) + 1
            arm.beta = (arm.beta or 1.0) + 1
        
        await db.commit()
        
        logger.info(f"Updated arm {arm_id} with reward {reward:.4f} (pulls: {arm.pulls}, successes: {arm.successes})")
        
        # CTR as PERCENTAGE (0-100 scale) for consistency with all other APIs
        ctr_decimal = arm.successes / max(arm.pulls, 1)
        return {
            "arm_id": str(arm.id),
            "updated": True,
            "pulls": arm.pulls,
            "successes": arm.successes,
            "total_reward": arm.total_reward,
            "ctr": round(ctr_decimal * 100, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update arm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiment/{experiment_id}/record-outcome", response_model=Dict[str, Any])
async def record_experiment_outcome(
    experiment_id: str,
    outcome_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    try:
        from datetime import datetime
        from ...data_layer.database.models import Experiment
        
        exp_stmt = select(Experiment).where(Experiment.id == UUID(experiment_id))
        exp_result = await db.execute(exp_stmt)
        experiment = exp_result.scalar_one_or_none()
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        arm_name = outcome_data.get("arm_id", "variant_a")
        stmt = select(BanditArm).where(
            (BanditArm.experiment_id == UUID(experiment_id)) &
            (BanditArm.arm_id == arm_name)
        )
        result = await db.execute(stmt)
        arm = result.scalar_one_or_none()

        if not arm:
            raise HTTPException(status_code=404, detail=f"Arm '{arm_name}' not found in experiment")
        
        reward = outcome_data.get("reward", 0)
        is_success = outcome_data.get("is_success", reward > 0)
        
        arm.pulls = (arm.pulls or 0) + 1
        arm.total_reward = (arm.total_reward or 0) + reward
        arm.last_pulled_at = datetime.utcnow()
        
        if is_success:
            arm.successes = (arm.successes or 0) + 1
            arm.alpha = (arm.alpha or 1.0) + 1
        else:
            arm.failures = (arm.failures or 0) + 1
            arm.beta = (arm.beta or 1.0) + 1
        
        experiment.total_impressions = (experiment.total_impressions or 0) + 1
        if is_success:
            experiment.total_conversions = (experiment.total_conversions or 0) + 1
        
        await db.commit()
        
        logger.info(f"Recorded outcome for experiment {experiment_id}, arm {arm_name}: reward={reward:.4f}")
        
        return {
            "experiment_id": str(experiment_id),
            "arm_id": arm_name,
            "updated": True,
            "experiment_impressions": experiment.total_impressions,
            "experiment_conversions": experiment.total_conversions,
            "arm_pulls": arm.pulls,
            "arm_successes": arm.successes
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))
