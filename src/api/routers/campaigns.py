"""
Campaign management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, text
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import logging

from ...data_layer.database.models import Campaign, Content, Experiment, CampaignStatus, Platform
from ...ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator
from ..dependencies import get_db, get_current_user, get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("", response_model=List[Dict[str, Any]])
async def list_campaigns(
    status: Optional[CampaignStatus] = None,
    platform: Optional[Platform] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List all campaigns with optional filtering"""
    try:
        query = select(Campaign)
        
        filters = []
        if status:
            filters.append(Campaign.status == status)
        if platform:
            filters.append(Campaign.platform == platform)
        
        if filters:
            query = query.where(and_(*filters))
        
        query = query.offset(offset).limit(limit)
        
        result = await db.execute(query)
        campaigns = result.scalars().all()
        
        return [
            {
                "id": str(c.id),
                "name": c.name,
                "platform": c.platform.value,
                "status": c.status.value,
                "target_persona": c.target_persona,
                "goal": c.goal.value if c.goal else None,
                "budget_total": c.budget_total,
                "budget_spent": c.budget_spent,
                "impressions": c.impressions,
                "clicks": c.clicks,
                "conversions": c.conversions,
                "ctr": c.ctr,
                "cpl": c.cpl,
                "created_at": c.created_at.isoformat(),
                "start_date": c.start_date.isoformat() if c.start_date else None,
                "end_date": c.end_date.isoformat() if c.end_date else None
            }
            for c in campaigns
        ]
    except Exception as e:
        logger.error(f"Failed to list campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Dict[str, Any])
async def create_campaign(
    campaign_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    orchestrator: MarketingOrchestrator = Depends(get_orchestrator)
):
    """Create a new campaign"""
    try:
        campaign_config = campaign_data.get("config", {})

        goal_value = campaign_data.get("goal", "lead_generation")
        campaign_config["goal"] = goal_value

        from ...data_layer.database.models import CampaignGoal
        try:
            goal_enum = CampaignGoal(goal_value) if goal_value else None
        except ValueError:
            goal_enum = CampaignGoal.LEAD_GENERATION  # Default fallback

        campaign = Campaign(
            name=campaign_data.get("name"),
            description=campaign_data.get("description", ""),
            platform=Platform(campaign_data.get("platform", "linkedin")),
            goal=goal_enum,  # ← FIX: Store in goal column
            target_persona=campaign_data.get("target_persona", campaign_data.get("persona", "decision_maker")),  # ← FIX: Accept both keys
            target_keywords=campaign_data.get("keywords", []),
            target_demographics=campaign_data.get("demographics", {}),
            budget_total=campaign_data.get("budget_total", campaign_data.get("budget", 1000)),  # ← FIX: Accept both keys
            budget_daily_limit=campaign_data.get("daily_limit", 100),
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=campaign_data.get("duration", 30)),
            status=CampaignStatus.DRAFT,
            config=campaign_config
        )

        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)

        if campaign_data.get("auto_start", False):
            background_tasks.add_task(
                orchestrator.run_campaign_workflow,
                str(campaign.id)
            )

        return {
            "id": str(campaign.id),
            "name": campaign.name,
            "platform": campaign.platform.value,
            "status": campaign.status.value,
            "goal": campaign_config.get("goal"),
            "budget_total": campaign.budget_total,
            "created_at": campaign.created_at.isoformat(),
            "message": "Campaign created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create campaign: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get campaign details"""
    try:
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return {
            "id": str(campaign.id),
            "name": campaign.name,
            "description": campaign.description,
            "platform": campaign.platform.value,
            "status": campaign.status.value,
            "goal": campaign.goal.value if campaign.goal else None,
            "target_persona": campaign.target_persona,
            "target_keywords": campaign.target_keywords,
            "budget_total": campaign.budget_total,
            "budget_spent": campaign.budget_spent,
            "budget_daily_limit": campaign.budget_daily_limit,
            "impressions": campaign.impressions,
            "clicks": campaign.clicks,
            "conversions": campaign.conversions,
            "ctr": campaign.ctr,
            "cpl": campaign.cpl,
            "config": campaign.config,
            "start_date": campaign.start_date.isoformat() if campaign.start_date else None,
            "end_date": campaign.end_date.isoformat() if campaign.end_date else None,
            "created_at": campaign.created_at.isoformat(),
            "updated_at": campaign.updated_at.isoformat() if campaign.updated_at else None
        }
    except Exception as e:
        logger.error(f"Failed to get campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{campaign_id}", response_model=Dict[str, Any])
async def update_campaign(
    campaign_id: UUID,
    updates: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """Update campaign"""
    try:
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        for key, value in updates.items():
            if hasattr(campaign, key):
                setattr(campaign, key, value)
        
        campaign.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {
            "id": str(campaign.id),
            "status": "updated",
            "message": "Campaign updated successfully"
        }
    except Exception as e:
        logger.error(f"Failed to update campaign: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{campaign_id}")
async def delete_campaign(
    campaign_id: UUID,
    force: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Delete campaign and all related data"""
    try:
        from sqlalchemy import delete as sql_delete
        from ...data_layer.database.models import Content, HITLQueue, CostTracking, Experiment, DelayedReward, WorkflowEvent

        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()

        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        if campaign.status == CampaignStatus.RUNNING and not force:
            campaign.status = CampaignStatus.PAUSED
            await db.commit()
            logger.info(f"Paused campaign {campaign_id} before deletion")

        await db.execute(
            sql_delete(HITLQueue).where(HITLQueue.content_id.in_(
                select(Content.id).where(Content.campaign_id == campaign_id)
            ))
        )

        await db.execute(
            sql_delete(WorkflowEvent).where(WorkflowEvent.campaign_id == campaign_id)
        )

        await db.execute(
            sql_delete(DelayedReward).where(DelayedReward.campaign_id == campaign_id)
        )

        from ...data_layer.database.models import BanditArm
        await db.execute(
            sql_delete(BanditArm).where(BanditArm.experiment_id.in_(
                select(Experiment.id).where(Experiment.campaign_id == campaign_id)
            ))
        )

        await db.execute(
            sql_delete(Experiment).where(Experiment.campaign_id == campaign_id)
        )

        await db.execute(
            sql_delete(CostTracking).where(CostTracking.campaign_id == campaign_id)
        )

        await db.execute(
            text("DELETE FROM metrics WHERE campaign_id = :campaign_id"),
            {"campaign_id": str(campaign_id)}
        )

        await db.execute(
            sql_delete(Content).where(Content.campaign_id == campaign_id)
        )

        await db.execute(
            sql_delete(Campaign).where(Campaign.id == campaign_id)
        )
        await db.commit()

        logger.info(f"Deleted campaign {campaign_id} and all related data")
        return {"status": "deleted", "message": "Campaign and all related data deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete campaign: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{campaign_id}/start")
async def start_campaign(
    campaign_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    orchestrator: MarketingOrchestrator = Depends(get_orchestrator)
):
    """
    Start campaign workflow (Research Plan Section 2.3)
    
    Automatically creates:
    - Experiment record for this campaign
    - BanditArm records for content variants
    
    This ensures OPE has real data for policy evaluation.
    """
    try:
        from ...data_layer.repositories.experiment_repo import ExperimentRepository
        
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        campaign.status = CampaignStatus.RUNNING
        await db.commit()
        
        # Create Experiment record for OPE data (Research Plan Section 2.3)
        # Only create if no active experiment exists for this campaign
        existing_exp = await db.execute(
            select(Experiment).where(
                and_(
                    Experiment.campaign_id == campaign_id,
                    Experiment.is_active == True
                )
            )
        )
        if not existing_exp.scalar_one_or_none():
            experiment_repo = ExperimentRepository(db)
            experiment_data = {
                "name": f"content_experiment_{campaign.name}",
                "campaign_id": str(campaign_id),
                "type": "content_bandit",
                "algorithm": "thompson_sampling",
                "parameters": {
                    "platform": campaign.platform.value if campaign.platform else "linkedin",
                    "persona": campaign.target_persona or "decision_maker",
                    "budget": float(campaign.budget_total or 1000),
                    "num_arms": 4
                },
                "variants": ["variant_a", "variant_b", "variant_c", "variant_d"],
                "is_active": True
            }
            await experiment_repo.create(experiment_data)
            logger.info(f"Created experiment with bandit arms for campaign {campaign_id}")
        
        background_tasks.add_task(
            orchestrator.run_campaign_workflow,
            str(campaign_id)
        )
        
        return {
            "status": "started",
            "campaign_id": str(campaign_id),
            "workflow_initiated": True,
            "experiment_created": True,
            "message": "Campaign workflow started with experiment tracking"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{campaign_id}/pause")
async def pause_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Pause campaign"""
    try:
        await db.execute(
            update(Campaign)
            .where(Campaign.id == campaign_id)
            .values(status=CampaignStatus.PAUSED)
        )
        await db.commit()

        return {"status": "paused", "campaign_id": str(campaign_id)}
    except Exception as e:
        logger.error(f"Failed to pause campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{campaign_id}/check-completion")
async def check_campaign_completion(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Check if campaign should complete based on Research Plan criteria:
    - Budget depletion (Section 7.4: Cost Control)
    - End date reached (Section 5.2: Campaign Duration)
    - Daily budget limit violations

    Automatically completes campaign if criteria met.
    Per Research Plan Section 2.1: Campaigns must complete to measure final ROI.
    """
    try:
        from ...ai_layer.learning.campaign_completion import should_complete_campaign

        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()

        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign_dict = {
            "id": str(campaign.id),
            "name": campaign.name,
            "status": campaign.status.value,
            "budget_total": campaign.budget_total or 0,
            "budget_spent": campaign.budget_spent or 0,
            "budget_daily_limit": campaign.budget_daily_limit or 0,
            "end_date": campaign.end_date,
            "total_impressions": campaign.impressions or 0,
            "total_clicks": campaign.clicks or 0,
            "total_conversions": campaign.conversions or 0
        }

        decision = should_complete_campaign(campaign_dict, budget_threshold=0.98)

        # If should complete and currently running, update campaign AND experiments
        if decision.should_complete and campaign.status == CampaignStatus.RUNNING:
            campaign.status = CampaignStatus.COMPLETED
            campaign.updated_at = datetime.utcnow()
            
            # Also complete all active experiments for this campaign (Research Plan Section 2.3)
            # This is required for OPE - it only queries completed experiments (is_active=False)
            from sqlalchemy import update as sql_update
            await db.execute(
                sql_update(Experiment)
                .where(
                    and_(
                        Experiment.campaign_id == campaign_id,
                        Experiment.is_active == True
                    )
                )
                .values(
                    is_active=False,
                    ended_at=datetime.utcnow()
                )
            )
            
            await db.commit()

            logger.info(f"✅ Campaign {campaign_id} completed: {decision.reason}. Experiments also marked complete.")

        return {
            "campaign_id": str(campaign_id),
            "should_complete": decision.should_complete,
            "reason": decision.reason,
            "completion_type": decision.completion_type,
            "final_metrics": decision.final_metrics,
            "completed": decision.should_complete and campaign.status == CampaignStatus.COMPLETED
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check campaign completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{campaign_id}/metrics", response_model=Dict[str, Any])
async def get_campaign_metrics(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get campaign metrics"""
    try:
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()

        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Calculate ROI using standard formula: ((revenue - cost) / cost) * 100
        # Per Research Plan, default conversion value is €100
        conversion_value = 100  # €100 per conversion
        total_revenue = campaign.conversions * conversion_value
        budget_spent = campaign.budget_spent or 0
        roi = ((total_revenue - budget_spent) / max(budget_spent, 1)) * 100 if budget_spent > 0 else 0

        return {
            "campaign_id": str(campaign_id),
            "impressions": campaign.impressions,
            "clicks": campaign.clicks,
            "conversions": campaign.conversions,
            "ctr": campaign.ctr,  # Already stored as percentage
            "cpl": campaign.cpl,
            "budget_spent": budget_spent,
            "budget_remaining": (campaign.budget_total or 0) - budget_spent,
            "roi": roi  # In percentage (e.g., 31.2 means 31.2% ROI)
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{campaign_id}/simulation", response_model=Dict[str, Any])
async def get_simulation_results(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get simulation results for campaign"""
    try:
        result = await db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        campaign = result.scalar_one_or_none()

        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        config = campaign.config or {}
        simulation_results = config.get('simulation_results')

        if not simulation_results:
            return {
                "campaign_id": str(campaign_id),
                "has_simulation": False,
                "message": "No simulation results available. Simulation may not have run yet."
            }

        return {
            "campaign_id": str(campaign_id),
            "has_simulation": True,
            "predicted_ctr": simulation_results.get('ctr', 0.0),
            "predicted_conversions": simulation_results.get('conversions', 0),
            "predicted_cpl": simulation_results.get('cpl'),
            "predicted_impressions": simulation_results.get('impressions', 0),
            "predicted_clicks": simulation_results.get('clicks', 0),
            "simulation_passed": simulation_results.get('passed', False),
            "simulation_timestamp": simulation_results.get('timestamp'),
            "full_results": simulation_results.get('full_results', {})
        }
    except Exception as e:
        logger.error(f"Failed to get simulation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}/performance", response_model=Dict[str, Any])
async def get_campaign_performance(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get campaign performance tracking - actual vs predicted metrics

    Compares simulation predictions against real campaign performance.
    Automatically calculates accuracy (MAPE) for validation.
    """
    try:
        from ...monitoring.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(db)
        performance = await tracker.track_campaign_performance(str(campaign_id))

        if not performance:
            raise HTTPException(status_code=404, detail="Performance data not available")

        return performance

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get campaign performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/all", response_model=List[Dict[str, Any]])
async def get_all_campaigns_performance(
    db: AsyncSession = Depends(get_db)
):
    """
    Get performance tracking for all active campaigns

    Returns actual vs predicted metrics for all running campaigns.
    """
    try:
        from ...monitoring.performance_tracker import PerformanceTracker

        tracker = PerformanceTracker(db)
        performance_list = await tracker.get_all_campaigns_performance()

        return performance_list

    except Exception as e:
        logger.error(f"Failed to get all campaigns performance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-all-completions")
async def check_all_campaign_completions():
    """
    Manually trigger campaign completion check for all running campaigns

    Useful for testing automatic completion logic.
    Background monitor runs every 5 minutes automatically.
    """
    try:
        from ...monitoring.campaign_monitor import trigger_completion_check

        summary = await trigger_completion_check()

        return {
            "status": "success",
            "campaigns": summary.get("campaigns", {}),
            "experiments": summary.get("experiments", {}),
            "total_checked": summary.get("total_checked", 0),
            "total_completed": summary.get("total_completed", 0),
            "message": f"Checked {summary.get('total_checked', 0)} items, completed {summary.get('total_completed', 0)}"
        }

    except Exception as e:
        logger.error(f"Failed to check campaign completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reconcile-statuses")
async def reconcile_campaign_statuses(
    db: AsyncSession = Depends(get_db)
):
    """
    Reconcile campaign statuses by checking workflow events.
    Fixes campaigns stuck at RUNNING that have actually completed deployment.
    """
    try:
        from ...data_layer.database.models import WorkflowEvent

        running = await db.execute(
            select(Campaign).where(Campaign.status == CampaignStatus.RUNNING)
        )
        running_campaigns = running.scalars().all()

        fixed = []
        for campaign in running_campaigns:
            last_event = await db.execute(
                select(WorkflowEvent)
                .where(WorkflowEvent.campaign_id == campaign.id)
                .order_by(WorkflowEvent.created_at.desc())
                .limit(1)
            )
            event = last_event.scalar_one_or_none()

            if not event:
                continue

            node = event.workflow_node or ""
            title = event.title or ""

            if node == "deployment" and "Complete" in title:
                campaign.status = CampaignStatus.COMPLETED
                campaign.updated_at = datetime.utcnow()
                fixed.append({"id": str(campaign.id), "name": campaign.name, "new_status": "COMPLETED"})
            elif node == "canary_deployment" and "Complete" in title:
                campaign.status = CampaignStatus.COMPLETED
                campaign.updated_at = datetime.utcnow()
                fixed.append({"id": str(campaign.id), "name": campaign.name, "new_status": "COMPLETED"})
            elif "Error" in title or "Failed" in title or "Rejected" in title:
                campaign.status = CampaignStatus.FAILED
                campaign.updated_at = datetime.utcnow()
                fixed.append({"id": str(campaign.id), "name": campaign.name, "new_status": "FAILED"})

        await db.commit()

        return {
            "status": "success",
            "total_running": len(running_campaigns),
            "fixed": len(fixed),
            "campaigns": fixed
        }

    except Exception as e:
        logger.error(f"Failed to reconcile campaign statuses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))