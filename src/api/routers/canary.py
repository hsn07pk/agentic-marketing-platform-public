from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
from datetime import datetime, timedelta

from ...automation_layer.deployment.canary_rollout import DeploymentController
from ...data_layer.database.models import Metric, CanaryDeployment
from ...data_layer.database.connection import get_async_session
from ..dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

canary_controller = DeploymentController()

async def persist_deployment_state(deployment):
    try:
        async with get_async_session() as db:
            stmt = select(CanaryDeployment).where(CanaryDeployment.deployment_id == deployment.deployment_id)
            result = await db.execute(stmt)
            db_deployment = result.scalar_one_or_none()
            
            if db_deployment:
                db_deployment.status = deployment.status.value
                db_deployment.current_traffic_percentage = deployment.current_traffic_percentage
                db_deployment.metrics_history = [m.to_dict() for m in deployment.canary_metrics_history]
                
                if deployment.end_time:
                    db_deployment.ended_at = deployment.end_time
                if deployment.rollback_reason:
                    db_deployment.rollback_reason = deployment.rollback_reason
                
                # AUTOMATED PROMOTION: If full rollout, update associated Campaign to Active
                if deployment.status.value == "full_rollout_100_percent":
                    campaign_id = deployment.metadata.get("campaign_id")
                    if campaign_id:
                        try:
                            from ...data_layer.database.models import Campaign
                            from uuid import UUID
                            campaign_uuid = UUID(campaign_id)
                            campaign_conf = await db.get(Campaign, campaign_uuid)
                            if campaign_conf:
                                campaign_conf.status = "active"
                                campaign_conf.is_active = True
                                logger.info(f"🚀 AUTOMATED PROMOTION: Campaign {campaign_id} set to ACTIVE (100% Traffic)")
                        except (ValueError, AttributeError):
                            logger.info(f"Skipping campaign promotion: '{campaign_id}' is not a valid campaign UUID")
                
                await db.commit()
                logger.info(f"Persisted state update for {deployment.deployment_id}: {deployment.status.value} @ {deployment.current_traffic_percentage*100:.0f}%")
    except Exception as e:
        logger.error(f"Failed to persist state callback: {e}")

canary_controller.register_persistence_callback(persist_deployment_state)


async def get_baseline_metrics_from_db(
    db: AsyncSession,
    campaign_id: str = None,
    days: int = 30
) -> Dict[str, float]:
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        query = select(
            func.avg(Metric.clicks / func.nullif(Metric.impressions, 0)).label('ctr'),
            func.avg(Metric.conversions / func.nullif(Metric.clicks, 0)).label('conversion_rate'),
            func.avg(Metric.spend / func.nullif(Metric.conversions, 0)).label('cpl')
        ).where(
            Metric.created_at >= start_date,
            Metric.created_at <= end_date
        )

        if campaign_id:
            query = query.where(Metric.campaign_id == campaign_id)

        result = await db.execute(query)
        row = result.first()

        # Return as decimal ratio (0.03 = 3%) to match canary_rollout.py baseline format
        if row and row.ctr is not None:
            return {
                'ctr': float(row.ctr) if row.ctr else 0.03,
                'conversion_rate': float(row.conversion_rate) if row.conversion_rate else 0.02,
                'cpl': float(row.cpl) if row.cpl else 50.0
            }
        else:
            logger.warning("No historical metrics found, using industry defaults")
            return {
                'ctr': 0.03,
                'conversion_rate': 0.02,
                'cpl': 50.0
            }

    except Exception as e:
        logger.error(f"Error querying baseline metrics: {e}")
        return {
            'ctr': 0.03,
            'conversion_rate': 0.02,
            'cpl': 50.0
        }

class CanaryDeploymentRequest(BaseModel):
    policy_name: str
    policy_version: str = "1.0.0"  # Default version if not specified
    campaign_id: str | None = None
    description: str | None = None
    deployment_name: str | None = None
    deployment_type: str | None = None
    initial_traffic_percent: int | None = None
    auto_rollback_enabled: bool | None = True
    thresholds: Dict[str, Any] | None = None

class CanaryRollbackRequest(BaseModel):
    deployment_id: str
    reason: str

@router.post("/start", response_model=Dict[str, Any])
async def start_canary_deployment(
    request: CanaryDeploymentRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        baseline_metrics = await get_baseline_metrics_from_db(
            db=db,
            campaign_id=request.campaign_id,
            days=30
        )

        is_test_mode = "TEST_MODE_FAST" in (request.description or "")
        
        deployment = await canary_controller.start_canary_deployment(
            policy_id=request.policy_name,
            policy_version=request.policy_version,
            baseline_metrics=baseline_metrics,
            initial_traffic_percent=request.initial_traffic_percent or 5,
            auto_rollback=request.auto_rollback_enabled if request.auto_rollback_enabled is not None else True,
            rollback_thresholds=request.thresholds,
            metadata={"campaign_id": request.campaign_id, "description": request.description, "test_mode": is_test_mode}
        )

        if isinstance(deployment, dict):
            deployment_id = deployment.get("deployment_id")
            status = deployment.get("status")
            traffic_pct = deployment.get("traffic_percent", 5) / 100.0
        else:
            deployment_id = deployment.deployment_id
            status = deployment.status.value if hasattr(deployment.status, 'value') else deployment.status
            traffic_pct = deployment.current_traffic_percentage

        thresholds = request.thresholds or {}
        db_deployment = CanaryDeployment(
            deployment_id=deployment_id,
            policy_id=request.policy_name,
            policy_version=request.policy_version,
            deployment_type=request.deployment_type or "policy",
            status=status,
            current_traffic_percentage=traffic_pct,
            baseline_ctr=baseline_metrics.get('ctr'),
            baseline_conversion_rate=baseline_metrics.get('conversion_rate'),
            baseline_cpl=baseline_metrics.get('cpl'),
            auto_rollback_enabled=request.auto_rollback_enabled if request.auto_rollback_enabled is not None else True,
            rollback_ctr_threshold=thresholds.get('ctr_degradation_percent', 10.0),
            rollback_error_threshold=thresholds.get('error_rate_percent', 5.0),
            rollback_latency_threshold=thresholds.get('latency_p95_ms', 500),
            extra_data={"campaign_id": request.campaign_id, "description": request.description},
            started_at=datetime.utcnow()
        )
        db.add(db_deployment)
        await db.commit()

        logger.info(f"Canary deployment started and persisted: {request.policy_name} v{request.policy_version}")

        return {
            "deployment_id": deployment_id,
            "policy_name": request.policy_name,
            "policy_version": request.policy_version,
            "status": status,
            "current_traffic_percentage": request.initial_traffic_percent or 5,
            "message": "Canary deployment started at 5% traffic"
        }

    except Exception as e:
        logger.error(f"Failed to start canary deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{deployment_id}", response_model=Dict[str, Any])
async def get_deployment_status(
    deployment_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        status = canary_controller.get_deployment_status(deployment_id)

        if not status:
            raise HTTPException(status_code=404, detail="Deployment not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback", response_model=Dict[str, str])
async def rollback_deployment(
    request: CanaryRollbackRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        await canary_controller.rollback_deployment(
            deployment_id=request.deployment_id,
            reason=request.reason
        )

        query = select(CanaryDeployment).where(
            CanaryDeployment.deployment_id == request.deployment_id
        )
        result = await db.execute(query)
        db_deployment = result.scalar_one_or_none()

        if db_deployment:
            db_deployment.status = "rolled_back"
            db_deployment.rollback_reason = request.reason
            db_deployment.current_traffic_percentage = 0.0
            db_deployment.ended_at = datetime.utcnow()
            await db.commit()
            logger.info(f"Canary deployment {request.deployment_id} rollback persisted to DB")

        logger.warning(f"Canary deployment {request.deployment_id} rolled back: {request.reason}")

        return {
            "status": "rolled_back",
            "deployment_id": request.deployment_id,
            "reason": request.reason,
            "message": "Traffic returned to previous stable policy"
        }

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active", response_model=List[Dict[str, Any]])
async def list_active_deployments(
    db: AsyncSession = Depends(get_db)
):
    try:
        active_statuses = [
            'pending', 'canary_5_percent', 'canary_25_percent', 
            'canary_50_percent', 'canary_75_percent'
        ]
        query = (
            select(CanaryDeployment)
            .where(CanaryDeployment.status.in_(active_statuses))
            .order_by(desc(CanaryDeployment.started_at))
        )
        result = await db.execute(query)
        db_deployments = result.scalars().all()

        deployments = []
        for d in db_deployments:
            deployments.append({
                'deployment_id': d.deployment_id,
                'policy_id': d.policy_id,
                'policy_version': d.policy_version,
                'start_time': d.started_at.isoformat() if d.started_at else None,
                'status': d.status,
                'current_traffic_percentage': d.current_traffic_percentage,
                'baseline_metrics': {
                    'ctr': d.baseline_ctr or 3.0,
                    'conversion_rate': d.baseline_conversion_rate or 2.0,
                    'cpl': d.baseline_cpl or 50.0
                },
                'canary_metrics_history': d.metrics_history or [],
                'rollback_reason': d.rollback_reason,
                'end_time': d.ended_at.isoformat() if d.ended_at else None
            })

        memory_deployments = canary_controller.list_active_deployments()
        db_ids = {d['deployment_id'] for d in deployments}
        for mem_d in memory_deployments:
            if mem_d.get('deployment_id') not in db_ids:
                deployments.append(mem_d)

        return deployments

    except Exception as e:
        logger.error(f"Failed to list active deployments from DB, falling back to memory: {e}")
        return canary_controller.list_active_deployments()

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_deployment_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    try:
        query = (
            select(CanaryDeployment)
            .order_by(desc(CanaryDeployment.started_at))
            .limit(limit)
        )
        result = await db.execute(query)
        db_deployments = result.scalars().all()

        history = []
        for d in db_deployments:
            history.append({
                'deployment_id': d.deployment_id,
                'policy_id': d.policy_id,
                'policy_version': d.policy_version,
                'start_time': d.started_at.isoformat() if d.started_at else None,
                'end_time': d.ended_at.isoformat() if d.ended_at else None,
                'status': d.status,
                'current_traffic_percentage': d.current_traffic_percentage,
                'baseline_metrics': {
                    'ctr': d.baseline_ctr or 3.0,
                    'conversion_rate': d.baseline_conversion_rate or 2.0,
                    'cpl': d.baseline_cpl or 50.0
                },
                'canary_metrics_history': d.metrics_history or [],
                'rollback_reason': d.rollback_reason
            })

        memory_history = [d.to_dict() for d in canary_controller.deployment_history[-limit:]]
        db_ids = {d['deployment_id'] for d in history}
        for mem_d in memory_history:
            if mem_d.get('deployment_id') not in db_ids:
                history.append(mem_d)

        history.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return history[:limit]

    except Exception as e:
        logger.error(f"Failed to get deployment history from DB: {e}")
        return [d.to_dict() for d in canary_controller.deployment_history[-limit:]]
