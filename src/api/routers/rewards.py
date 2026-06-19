from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging

from ...ai_layer.learning.reward_tracker import RewardTracker

logger = logging.getLogger(__name__)

router = APIRouter()

reward_tracker = RewardTracker()

@router.post("/register")
async def register_conversion(
    campaign_id: str,
    lead_email: str,
    lead_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    
    try:
        success = await reward_tracker.register_conversion(
            campaign_id,
            lead_email,
            lead_data
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to register conversion")
        
        return {
            "message": "Conversion registered",
            "campaign_id": campaign_id,
            "lead_email": lead_email
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/campaign/{campaign_id}")
async def get_campaign_rewards(
    campaign_id: str
):
    
    try:
        rewards = await reward_tracker.get_campaign_delayed_rewards(campaign_id)
        return rewards
    except Exception as e:
        logger.error(f"Failed to get rewards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def process_pending_rewards():
    
    try:
        stats = await reward_tracker.process_pending_rewards()
        return stats
    except Exception as e:
        logger.error(f"Failed to process rewards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/booking/{lead_email}")
async def check_booking(
    lead_email: str
):
    
    try:
        booking = await reward_tracker.check_for_booking(lead_email)
        
        if booking:
            return {
                "found": True,
                "booking": booking
            }
        else:
            return {
                "found": False
            }
    except Exception as e:
        logger.error(f"Failed to check booking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending")
async def get_pending_rewards():
    try:
        from ...data_layer.database.connection import get_async_session
        from ...data_layer.database.models import DelayedReward, Campaign
        from sqlalchemy import select
        from datetime import timedelta

        async with get_async_session() as session:
            query = select(DelayedReward).where(DelayedReward.status == "pending").limit(100)
            result = await session.execute(query)
            rewards = result.scalars().all()

            campaign_ids = {r.campaign_id for r in rewards}
            campaign_names = {}
            if campaign_ids:
                cq = select(Campaign).where(Campaign.id.in_(campaign_ids))
                cr = await session.execute(cq)
                for c in cr.scalars().all():
                    campaign_names[c.id] = c.name

            return {
                "total": len(rewards),
                "pending_rewards": [
                    {
                        "id": str(r.id),
                        "campaign_id": str(r.campaign_id),
                        "campaign_name": campaign_names.get(r.campaign_id, "Unknown"),
                        "lead_email": r.lead_email,
                        "action": "conversion" if not r.meeting_scheduled else "booking",
                        "initial_reward": r.initial_reward,
                        "current_reward": r.current_reward,
                        "created_at": r.registered_at.isoformat() if r.registered_at else "",
                        "expires_at": (r.registered_at + timedelta(days=30)).isoformat() if r.registered_at else "",
                        "registered_at": r.registered_at.isoformat() if r.registered_at else "",
                        "meeting_scheduled": r.meeting_scheduled,
                        "status": r.status
                    }
                    for r in rewards
                ]
            }
    except Exception as e:
        logger.error(f"Failed to get pending rewards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_reward_stats():
    try:
        from ...data_layer.database.connection import get_async_session
        from ...data_layer.database.models import DelayedReward
        from sqlalchemy import select, func
        from datetime import datetime, timedelta
        
        async with get_async_session() as session:
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            attributed_query = select(func.count()).select_from(DelayedReward).where(
                DelayedReward.status == "booked",
                DelayedReward.resolved_at >= week_ago
            )
            attributed_result = await session.execute(attributed_query)
            attributed_count = attributed_result.scalar() or 0
            
            pending_query = select(func.count()).select_from(DelayedReward).where(
                DelayedReward.status == "pending"
            )
            pending_result = await session.execute(pending_query)
            pending_count = pending_result.scalar() or 0
            
            converted_query = select(func.count()).select_from(DelayedReward).where(
                DelayedReward.status == "booked"
            )
            converted_result = await session.execute(converted_query)
            converted_count = converted_result.scalar() or 0
            
            lost_query = select(func.count()).select_from(DelayedReward).where(
                DelayedReward.status == "lost"
            )
            lost_result = await session.execute(lost_query)
            lost_count = lost_result.scalar() or 0
            
            return {
                "attributed_this_week": attributed_count,
                "total_pending": pending_count,
                "total_converted": converted_count,
                "total_lost": lost_count,
                "conversion_rate": (converted_count / (converted_count + lost_count) * 100) if (converted_count + lost_count) > 0 else 0.0
            }
    except Exception as e:
        logger.error(f"Failed to get reward stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))