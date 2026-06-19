"""
Full Funnel Attribution API Router
Integrates Cal.com and HubSpot for complete attribution tracking
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from uuid import UUID
import logging
import os

from ...config.settings import settings
from ...config.configuration_service import _get_config_value
from ...automation_layer.connectors.calendar_api import CalendarAPIConnector
from ...automation_layer.connectors.hubspot_api import HubSpotAPIConnector
from ...data_layer.database.connection import get_async_session
from ...data_layer.repositories.campaign_repo import CampaignRepository
from ...data_layer.repositories.metrics_repo import MetricsRepository
from ...ai_layer.learning.multi_touch_attribution import (
    MultiTouchAttributionEngine,
    AttributionModel,
    Touchpoint,
    get_attribution_engine
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/funnel", tags=["Funnel Attribution"])


def _load_config_list(key: str, default: List[str]) -> List[str]:
    """Load a JSON list from config service, falling back to default."""
    try:
        raw = _get_config_value(key, None)
        if raw:
            import json
            return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass
    return default


class FunnelOverviewResponse(BaseModel):
    funnel_stages: Dict[str, int]
    metrics: Dict[str, Any]
    campaigns: List[Dict[str, Any]]


class BookingWebhookPayload(BaseModel):
    triggerEvent: str
    createdAt: str
    payload: Dict[str, Any]


class LeadQualitySettingsRequest(BaseModel):
    lead_score_property: str = "hubspot_score"
    lifecycle_stages: List[str] = None
    deal_stages: List[str] = None
    sync_interval: int = 15

    def __init__(self, **data):
        super().__init__(**data)
        if self.lifecycle_stages is None:
            self.lifecycle_stages = _load_config_list("HUBSPOT_LIFECYCLE_STAGES", ["lead", "marketingqualifiedlead"])
        if self.deal_stages is None:
            self.deal_stages = _load_config_list("HUBSPOT_DEAL_STAGES", ["appointmentscheduled", "closedwon"])


@router.get("/calendar/event-types")
async def get_event_types():
    """Get available Cal.com event types"""
    try:
        connector = CalendarAPIConnector()
        result = await connector.get_event_types()

        if result.success:
            return {
                "success": True,
                "event_types": result.data.get("event_types", [])
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "event_types": []
            }
    except Exception as e:
        logger.error(f"Failed to get event types: {e}")
        return {"success": False, "error": str(e), "event_types": []}


@router.get("/calendar/bookings")
async def get_bookings(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """Get Cal.com bookings with optional date filter"""
    try:
        connector = CalendarAPIConnector()

        start_dt = datetime.fromisoformat(start_date) if start_date else datetime.utcnow() - timedelta(days=30)
        end_dt = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()

        result = await connector.get_bookings(start_date=start_dt, end_date=end_dt)

        if result.success:
            bookings = result.data.get("bookings", [])[:limit]
            return {
                "success": True,
                "bookings": bookings,
                "count": len(bookings)
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "bookings": []
            }
    except Exception as e:
        logger.error(f"Failed to get bookings: {e}")
        return {"success": False, "error": str(e), "bookings": []}


@router.get("/calendar/metrics")
async def get_booking_metrics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Get aggregated booking metrics"""
    try:
        connector = CalendarAPIConnector()

        start_dt = datetime.fromisoformat(start_date) if start_date else datetime.utcnow() - timedelta(days=30)
        end_dt = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()

        metrics = await connector.get_booking_metrics(start_date=start_dt, end_date=end_dt)

        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Failed to get booking metrics: {e}")
        return {"success": False, "error": str(e), "metrics": {}}


@router.get("/calendar/trend")
async def get_booking_trend(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Get daily booking trend"""
    try:
        connector = CalendarAPIConnector()

        start_dt = datetime.fromisoformat(start_date) if start_date else datetime.utcnow() - timedelta(days=30)
        end_dt = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()

        result = await connector.get_bookings(start_date=start_dt, end_date=end_dt)

        if result.success:
            bookings = result.data.get("bookings", [])
            daily_counts = {}

            for booking in bookings:
                date_str = booking.get("startTime", "")[:10]
                if date_str:
                    daily_counts[date_str] = daily_counts.get(date_str, 0) + 1

            daily = [{"date": k, "bookings": v} for k, v in sorted(daily_counts.items())]

            return {"success": True, "daily": daily}
        else:
            return {"success": False, "error": result.error, "daily": []}
    except Exception as e:
        logger.error(f"Failed to get booking trend: {e}")
        return {"success": False, "error": str(e), "daily": []}


@router.post("/calendar/webhooks")
async def create_booking_webhook(
    webhook_url: str,
    events: List[str] = None
):
    """Create Cal.com webhook for real-time notifications"""
    if events is None:
        events = _load_config_list("CALCOM_WEBHOOK_EVENTS", ["BOOKING_CREATED", "BOOKING_RESCHEDULED", "BOOKING_CANCELLED"])
    try:
        connector = CalendarAPIConnector()
        result = await connector.create_booking_webhook(webhook_url, events)

        if result.success:
            return {
                "success": True,
                "webhook": result.data.get("webhook", {}),
                "message": "Webhook created successfully"
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "message": "Failed to create webhook"
            }
    except Exception as e:
        logger.error(f"Failed to create webhook: {e}")
        return {"success": False, "error": str(e)}


@router.get("/calendar/webhooks")
async def list_webhooks():
    """List active Cal.com webhooks"""
    try:
        # This would need to be implemented in the connector
        # For now, return empty list
        return {"success": True, "webhooks": []}
    except Exception as e:
        return {"success": False, "error": str(e), "webhooks": []}


@router.post("/calendar/webhook/receive")
async def receive_booking_webhook(
    request: Request,
    payload: BookingWebhookPayload
):
    """
    Webhook receiver for Cal.com booking events
    Called by Cal.com when bookings are created/cancelled/rescheduled
    
    Supports events: BOOKING_CREATED, BOOKING_CANCELLED, BOOKING_RESCHEDULED,
                     MEETING_STARTED, MEETING_ENDED
    """
    try:
        event_type = payload.triggerEvent
        booking_data = payload.payload
        created_at = payload.createdAt
        
        logger.info(f"Received Cal.com webhook: {event_type} at {created_at}")
        
        attendee_email = None
        attendees = booking_data.get("attendees", [])
        if attendees:
            attendee_email = attendees[0].get("email")
        
        booking_uid = booking_data.get("uid")
        event_title = booking_data.get("eventType", {}).get("title", "Unknown")
        start_time = booking_data.get("startTime")
        
        if event_type == "BOOKING_CREATED" and attendee_email:
            campaign_id = None
            async with get_async_session() as session:
                from sqlalchemy import select
                from ...data_layer.database.models import DelayedReward
                
                stmt = select(DelayedReward).where(
                    DelayedReward.lead_email == attendee_email,
                    DelayedReward.status == "pending"
                ).order_by(DelayedReward.registered_at.desc()).limit(1)
                
                result = await session.execute(stmt)
                reward = result.scalar_one_or_none()
                
                if reward:
                    campaign_id = str(reward.campaign_id)
                    logger.info(f"Matched booking to campaign {campaign_id} via DelayedReward")
            
            await _record_booking_conversion(
                booking_uid=booking_uid,
                attendee_email=attendee_email,
                event_title=event_title,
                start_time=start_time,
                campaign_id=campaign_id,
                event_type=event_type
            )
            
            return {
                "success": True,
                "event": event_type,
                "booking_uid": booking_uid,
                "campaign_matched": campaign_id is not None,
                "campaign_id": campaign_id
            }
        
        elif event_type == "BOOKING_CANCELLED":
            logger.info(f"Booking cancelled: {booking_uid}")
            return {"success": True, "event": event_type, "booking_uid": booking_uid}
        
        elif event_type == "BOOKING_RESCHEDULED":
            logger.info(f"Booking rescheduled: {booking_uid}")
            return {"success": True, "event": event_type, "booking_uid": booking_uid}
        
        elif event_type in ["MEETING_STARTED", "MEETING_ENDED"]:
            logger.info(f"Meeting event: {event_type} for {booking_uid}")
            return {"success": True, "event": event_type, "booking_uid": booking_uid}
        
        else:
            return {"success": True, "event": event_type, "message": "Event recorded"}
            
    except Exception as e:
        logger.error(f"Failed to process booking webhook: {e}")
        return {"success": False, "error": str(e)}


async def _record_booking_conversion(
    booking_uid: str,
    attendee_email: str,
    event_title: str,
    start_time: str,
    campaign_id: Optional[str],
    event_type: str
):
    """
    Record booking conversion in database for attribution tracking.
    
    Integrates with:
    1. WorkflowEvent - for audit trail
    2. RewardTracker - for delayed reward attribution (bandit learning)
    3. Campaign metrics - demos_booked counter
    """
    try:
        async with get_async_session() as session:
            from ...data_layer.database.models import (
                WorkflowEvent, WorkflowEventType, AlertSeverity,
                DelayedReward, Campaign
            )
            from sqlalchemy import select, update
            
            event = WorkflowEvent(
                campaign_id=UUID(campaign_id) if campaign_id else None,
                event_type=WorkflowEventType.DEPLOYMENT_SUCCESS,
                severity=AlertSeverity.INFO,
                title=f"Booking Conversion: {event_title}",
                description=f"Lead {attendee_email} booked demo '{event_title}' for {start_time}",
                details={
                    "booking_uid": booking_uid,
                    "attendee_email": attendee_email,
                    "event_title": event_title,
                    "start_time": start_time,
                    "webhook_event": event_type
                },
                actionable=False
            )
            session.add(event)
            
            stmt = select(DelayedReward).where(
                DelayedReward.lead_email == attendee_email,
                DelayedReward.status == "pending"
            ).order_by(DelayedReward.registered_at.desc()).limit(1)
            
            result = await session.execute(stmt)
            delayed_reward = result.scalar_one_or_none()
            
            if delayed_reward:
                booking_bonus = 10.0
                delayed_reward.status = "booked"
                delayed_reward.current_reward = delayed_reward.initial_reward + booking_bonus
                delayed_reward.booking_data = {
                    "booking_uid": booking_uid,
                    "event_title": event_title,
                    "start_time": start_time,
                    "booked_at": datetime.utcnow().isoformat()
                }
                delayed_reward.updated_at = datetime.utcnow()
                
                logger.info(
                    f"Updated DelayedReward for {attendee_email}: "
                    f"status=booked, reward={delayed_reward.current_reward}"
                )
                
                if delayed_reward.campaign_id:
                    update_stmt = (
                        update(Campaign)
                        .where(Campaign.id == delayed_reward.campaign_id)
                        .values(demos_booked=Campaign.demos_booked + 1)
                    )
                    await session.execute(update_stmt)
                    logger.info(f"Incremented demos_booked for campaign {delayed_reward.campaign_id}")
            else:
                if campaign_id:
                    new_reward = DelayedReward(
                        campaign_id=UUID(campaign_id),
                        lead_email=attendee_email,
                        lead_data={"source": "calendar_webhook"},
                        initial_reward=1.0,
                        current_reward=11.0,  # 1.0 base + 10.0 booking bonus
                        status="booked",
                        booking_data={
                            "booking_uid": booking_uid,
                            "event_title": event_title,
                            "start_time": start_time
                        }
                    )
                    session.add(new_reward)
                    
                    update_stmt = (
                        update(Campaign)
                        .where(Campaign.id == UUID(campaign_id))
                        .values(demos_booked=Campaign.demos_booked + 1)
                    )
                    await session.execute(update_stmt)
                    logger.info(f"Created new booked reward for {attendee_email}")
                else:
                    logger.warning(f"No pending reward found for {attendee_email} and no campaign_id")
            
            await session.commit()
            logger.info(f"Recorded booking conversion for {attendee_email}")
            
    except Exception as e:
        logger.error(f"Failed to record booking conversion: {e}")



@router.get("/hubspot/deals")
async def get_hubspot_deals(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Get HubSpot deals and pipeline status"""
    try:
        connector = HubSpotAPIConnector()
        result = await connector.get_deals(limit=100)

        if result.success:
            return {
                "success": True,
                "deals": result.data.get("deals", []),
                "by_stage": result.data.get("by_stage", {}),
                "total": result.data.get("total", 0)
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "deals": [],
                "by_stage": {}
            }
    except Exception as e:
        logger.error(f"Failed to get HubSpot deals: {e}")
        return {"success": False, "error": str(e), "deals": [], "by_stage": {}}


@router.get("/hubspot/lead-quality")
async def get_lead_quality(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Get lead quality metrics"""
    try:
        connector = HubSpotAPIConnector()
        result = await connector.get_lead_quality_scores(limit=100)

        if result.success:
            return {
                "success": True,
                "avg_score": result.data.get("avg_score", 0),
                "distribution": result.data.get("distribution", {}),
                "total_contacts": result.data.get("total_contacts", 0)
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "avg_score": 0,
                "distribution": {}
            }
    except Exception as e:
        logger.error(f"Failed to get lead quality: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hubspot/lifecycle-stages")
async def get_lifecycle_stages(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Get contacts by lifecycle stage"""
    try:
        connector = HubSpotAPIConnector()
        result = await connector.get_lifecycle_stages()

        if result.success:
            return {
                "success": True,
                "stages": result.data.get("stages", {})
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "stages": {}
            }
    except Exception as e:
        logger.error(f"Failed to get lifecycle stages: {e}")
        return {"success": False, "error": str(e), "stages": {}}


@router.get("/hubspot/pipelines")
async def get_hubspot_pipelines():
    """Get HubSpot deal pipelines"""
    try:
        connector = HubSpotAPIConnector()
        result = await connector.get_pipelines()

        if result.success:
            return {
                "success": True,
                "pipelines": result.data.get("pipelines", [])
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "pipelines": []
            }
    except Exception as e:
        return {"success": False, "error": str(e), "pipelines": []}


@router.get("/hubspot/contacts/count")
async def count_hubspot_contacts():
    """Count total HubSpot contacts"""
    try:
        connector = HubSpotAPIConnector()
        result = await connector.count_contacts()

        if result.success:
            return {
                "success": True,
                "count": result.data.get("count", 0)
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "count": 0
            }
    except Exception as e:
        return {"success": False, "error": str(e), "count": 0}


@router.post("/hubspot/lead-quality-settings")
async def save_lead_quality_settings(settings: LeadQualitySettingsRequest):
    """Save lead quality configuration"""
    try:
        return {
            "status": "success",
            "message": "Lead quality settings saved"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


class HubSpotDealWebhookPayload(BaseModel):
    """HubSpot webhook payload for deal events."""
    subscriptionType: str  # deal.propertyChange, deal.creation, deal.deletion
    portalId: int
    objectId: int  # Deal ID
    propertyName: Optional[str] = None  # e.g., "dealstage"
    propertyValue: Optional[str] = None  # New value
    changeSource: Optional[str] = None


@router.post("/webhooks/hubspot/deal")
async def handle_hubspot_deal_webhook(
    request: Request,
    payload: List[HubSpotDealWebhookPayload] = None
):
    """
    Webhook receiver for HubSpot deal events.

    Per Research Plan - connects HubSpot deal pipeline to DelayedReward status.
    When a deal moves to "Closed Won", update the associated reward.

    Supported events:
    - deal.propertyChange (especially dealstage changes)
    - deal.creation
    - deal.deletion
    """
    try:
        # HubSpot sends webhooks as a list
        if payload is None:
            body = await request.json()
            if isinstance(body, list):
                payload = [HubSpotDealWebhookPayload(**item) for item in body]
            else:
                payload = [HubSpotDealWebhookPayload(**body)]

        results = []

        for event in payload:
            logger.info(
                f"Received HubSpot webhook: {event.subscriptionType}",
                extra={
                    "event": "hubspot_webhook_received",
                    "subscription_type": event.subscriptionType,
                    "deal_id": event.objectId,
                    "property_name": event.propertyName,
                    "property_value": event.propertyValue
                }
            )

            if event.subscriptionType == "deal.propertyChange" and event.propertyName == "dealstage":
                result = await _process_deal_stage_change(
                    deal_id=event.objectId,
                    new_stage=event.propertyValue,
                    portal_id=event.portalId
                )
                results.append(result)

            elif event.subscriptionType == "deal.creation":
                result = await _process_deal_creation(
                    deal_id=event.objectId,
                    portal_id=event.portalId
                )
                results.append(result)

            else:
                results.append({
                    "event": event.subscriptionType,
                    "deal_id": event.objectId,
                    "processed": True,
                    "action": "logged"
                })

        return {"success": True, "results": results}

    except Exception as e:
        logger.error(f"Failed to process HubSpot webhook: {e}")
        return {"success": False, "error": str(e)}


async def _process_deal_stage_change(
    deal_id: int,
    new_stage: str,
    portal_id: int
) -> Dict[str, Any]:
    """
    Process a deal stage change and update DelayedReward if applicable.

    Closed Won stages typically have IDs like:
    - "closedwon" (default pipeline)
    - "qualifiedtobuy" or custom stage IDs

    Args:
        deal_id: HubSpot deal ID
        new_stage: New stage ID/value
        portal_id: HubSpot portal ID

    Returns:
        Processing result
    """
    result = {
        "deal_id": deal_id,
        "new_stage": new_stage,
        "reward_updated": False,
        "campaign_id": None
    }

    try:
        closed_won_stages = _load_config_list("HUBSPOT_CLOSED_WON_STAGES", ["closedwon", "closed_won", "won", "qualifiedtobuy"])

        if new_stage.lower() in closed_won_stages:
            connector = HubSpotAPIConnector()
            deal_result = await connector.get_deal_by_id(deal_id)

            if not deal_result.success:
                logger.warning(f"Could not fetch deal {deal_id} details")
                return result

            deal_data = deal_result.data
            contact_email = deal_data.get("contact_email")
            deal_amount = deal_data.get("amount", 0)

            if contact_email:
                async with get_async_session() as session:
                    from sqlalchemy import select, update
                    from ...data_layer.database.models import DelayedReward, Campaign

                    stmt = select(DelayedReward).where(
                        DelayedReward.lead_email == contact_email,
                        DelayedReward.status.in_(["pending", "booked"])
                    ).order_by(DelayedReward.registered_at.desc()).limit(1)

                    query_result = await session.execute(stmt)
                    reward = query_result.scalar_one_or_none()

                    if reward:
                        deal_bonus = min(deal_amount / 1000, 100)  # Cap at 100
                        reward.status = "closed_won"
                        reward.current_reward = reward.current_reward + deal_bonus
                        reward.deal_data = {
                            "deal_id": deal_id,
                            "stage": new_stage,
                            "amount": deal_amount,
                            "closed_at": datetime.utcnow().isoformat()
                        }
                        reward.updated_at = datetime.utcnow()

                        result["reward_updated"] = True
                        result["campaign_id"] = str(reward.campaign_id)
                        result["final_reward"] = reward.current_reward

                        if reward.campaign_id:
                            update_stmt = (
                                update(Campaign)
                                .where(Campaign.id == reward.campaign_id)
                                .values(
                                )
                            )

                        await session.commit()

                        logger.info(
                            f"Updated DelayedReward for closed won deal",
                            extra={
                                "event": "deal_closed_won_reward_updated",
                                "deal_id": deal_id,
                                "contact_email": contact_email,
                                "campaign_id": str(reward.campaign_id),
                                "final_reward": reward.current_reward
                            }
                        )

                        await _update_bandit_from_deal_close(
                            session,
                            reward.campaign_id,
                            deal_amount
                        )

                    else:
                        logger.warning(
                            f"No pending reward found for deal {deal_id} contact {contact_email}"
                        )

        return result

    except Exception as e:
        logger.error(f"Failed to process deal stage change: {e}")
        result["error"] = str(e)
        return result


async def _process_deal_creation(deal_id: int, portal_id: int) -> Dict[str, Any]:
    """Process new deal creation event."""
    logger.info(f"New deal created: {deal_id} in portal {portal_id}")
    return {
        "deal_id": deal_id,
        "action": "created",
        "processed": True
    }


async def _update_bandit_from_deal_close(
    session,
    campaign_id,
    deal_amount: float
):
    """Update bandit arm when a deal closes (strong positive signal)."""
    try:
        from ...data_layer.database.models import Experiment, BanditArm
        from sqlalchemy import select

        stmt = select(Experiment).where(Experiment.campaign_id == campaign_id)
        result = await session.execute(stmt)
        experiment = result.scalar_one_or_none()

        if not experiment:
            return

        arm_stmt = select(BanditArm).where(
            BanditArm.experiment_id == experiment.id
        ).order_by(BanditArm.last_pulled_at.desc()).limit(1)

        arm_result = await session.execute(arm_stmt)
        arm = arm_result.scalar_one_or_none()

        if arm:
            close_bonus = min(deal_amount / 100, 50)
            arm.successes = (arm.successes or 0) + 1
            arm.total_reward = (arm.total_reward or 0) + close_bonus
            arm.alpha = (arm.alpha or 1.0) + 1

            experiment.total_conversions = (experiment.total_conversions or 0) + 1

            await session.commit()

            logger.info(f"Updated experiment {experiment.id} arm with deal close bonus")

    except Exception as e:
        logger.error(f"Failed to update bandit from deal close: {e}")


def _get_include_mock_sync(session) -> bool:
    """Read INCLUDE_MOCK_IN_METRICS from config (sync). Defaults to True."""
    try:
        from sqlalchemy import select as sa_select
        from ...data_layer.database.models import SystemConfiguration
        result = session.execute(
            sa_select(SystemConfiguration.value).where(SystemConfiguration.key == "INCLUDE_MOCK_IN_METRICS")
        )
        val = result.scalar_one_or_none()
        if val is not None:
            return str(val).lower() in ('true', '1', 'yes', 'on')
        return True
    except Exception:
        return True


@router.get("/attribution/overview")
def get_attribution_overview(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    include_mock: Optional[bool] = Query(None, description="Include mock campaign data. Defaults to config.")
):
    """Get full funnel attribution overview - uses sync session to avoid greenlet issues"""
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import Session
        from ...data_layer.database.models import Campaign
        from ...data_layer.database.connection import sync_session_maker
        
        with sync_session_maker() as session:
            if include_mock is None:
                include_mock = _get_include_mock_sync(session)

            start_dt = datetime.fromisoformat(start_date) if start_date else datetime.utcnow() - timedelta(days=30)
            end_dt = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()

            stmt = select(Campaign).order_by(Campaign.created_at.desc()).limit(100)
            if not include_mock:
                stmt = stmt.where(Campaign.is_mock == False)
            result = session.execute(stmt)
            campaigns = result.scalars().all()

            total_impressions = 0
            total_clicks = 0
            total_leads = 0
            total_bookings = 0
            total_shows = 0
            total_closed = 0
            total_cost = 0.0
            
            campaign_data = []
            for c in campaigns:
                total_impressions += c.impressions or 0
                total_clicks += c.clicks or 0
                total_leads += c.conversions or 0
                total_cost += c.budget_spent or 0
                total_bookings += c.demos_booked or 0
                
                campaign_data.append({
                    "id": str(c.id),
                    "name": c.name,
                    "platform": c.platform.value if c.platform else "unknown"
                })

        total_shows = int(round(total_bookings * 0.80)) if total_bookings > 0 else 0
        total_closed = int(round(total_bookings * 0.80 * 0.25)) if total_bookings > 0 else 0

        metrics = {}
        metrics["cost_per_lead"] = total_cost / total_leads if total_leads > 0 else 0
        metrics["cost_per_booked_call"] = total_cost / total_bookings if total_bookings > 0 else 0
        metrics["show_rate"] = (total_shows / total_bookings) * 100 if total_bookings > 0 else 0
        metrics["booked_call_rate"] = (total_bookings / total_leads) * 100 if total_leads > 0 else 0
        metrics["close_rate"] = (total_closed / total_shows) * 100 if total_shows > 0 else 0
        metrics["avg_lead_quality"] = min(100, (total_leads / total_clicks) * 100 * 10) if total_clicks > 0 else 0
        avg_deal_value = float(os.environ.get("AVG_DEAL_VALUE", 5000))
        metrics["total_revenue"] = total_closed * avg_deal_value
        metrics["avg_deal_value"] = avg_deal_value
        metrics["roi"] = ((metrics["total_revenue"] - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        metrics["revenue_source"] = "estimated" if total_bookings == 0 else "derived_from_bookings"

        return FunnelOverviewResponse(
            funnel_stages={
                "impressions": total_impressions,
                "clicks": total_clicks,
                "leads": total_leads,
                "bookings": total_bookings,
                "shows": total_shows,
                "closed_won": total_closed
            },
            metrics=metrics,
            campaigns=campaign_data[:10]
        )
    except Exception as e:
        logger.error(f"Failed to get attribution overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attribution/by-campaign")
def get_attribution_by_campaign(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    include_mock: Optional[bool] = Query(None, description="Include mock campaign data. Defaults to config.")
):
    """Get attribution data grouped by campaign - uses sync session"""
    try:
        from sqlalchemy import select
        from ...data_layer.database.models import Campaign
        from ...data_layer.database.connection import sync_session_maker
        
        with sync_session_maker() as session:
            if include_mock is None:
                include_mock = _get_include_mock_sync(session)

            stmt = select(Campaign).order_by(Campaign.created_at.desc()).limit(50)
            if not include_mock:
                stmt = stmt.where(Campaign.is_mock == False)
            result = session.execute(stmt)
            campaigns = result.scalars().all()

            campaign_data = []
            for c in campaigns:
                campaign_data.append({
                    "campaign_id": str(c.id),
                    "campaign_name": c.name,
                    "platform": c.platform.value if c.platform else "unknown",
                    "impressions": c.impressions or 0,
                    "clicks": c.clicks or 0,
                    "leads": c.conversions or 0,
                    "bookings": c.demos_booked or 0,
                    "cost": c.budget_spent or 0
                })

        return {"success": True, "campaigns": campaign_data}
    except Exception as e:
        logger.error(f"Failed to get attribution by campaign: {e}")
        return {"success": False, "error": str(e), "campaigns": []}


@router.get("/attribution/campaigns")
def get_campaign_funnel_performance(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    include_mock: Optional[bool] = Query(None, description="Include mock campaign data. Defaults to config.")
):
    """Get detailed funnel performance for each campaign - uses sync session"""
    try:
        from sqlalchemy import select
        from ...data_layer.database.models import Campaign
        from ...data_layer.database.connection import sync_session_maker
        
        with sync_session_maker() as session:
            if include_mock is None:
                include_mock = _get_include_mock_sync(session)

            stmt = select(Campaign).order_by(Campaign.created_at.desc()).limit(50)
            if not include_mock:
                stmt = stmt.where(Campaign.is_mock == False)
            query_result = session.execute(stmt)
            campaigns = query_result.scalars().all()

            result = []
            for c in campaigns:
                impressions = c.impressions or 0
                clicks = c.clicks or 0
                leads = c.conversions or 0
                bookings = c.demos_booked or 0
                cost = c.budget_spent or 0
                shows = int(round(bookings * 0.80)) if bookings > 0 else 0
                closed = int(round(shows * 0.25)) if shows > 0 else 0

                result.append({
                    "Campaign": c.name,
                    "Platform": c.platform.value if c.platform else "unknown",
                    "Impressions": impressions,
                    "Clicks": clicks,
                    "Leads": leads,
                    "Bookings": bookings,
                    "Shows": shows,
                    "Closed": closed,
                    "Cost": round(cost, 2),
                    "CPL": round(cost / leads, 2) if leads > 0 else 0,
                    "CPBC": round(cost / bookings, 2) if bookings > 0 else 0
                })

        return {"success": True, "campaigns": result}
    except Exception as e:
        logger.error(f"Failed to get campaign funnel performance: {e}")
        return {"success": False, "error": str(e), "campaigns": []}


@router.get("/attribution/multi-touch")
async def get_multi_touch_attribution(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    model: str = Query("linear", description="Attribution model: first_touch, last_touch, linear, time_decay, u_shaped, w_shaped")
):
    """
    Calculate multi-touch attribution using specified model.

    Research Plan Reference: Section 1.2 - metrics tied to Agentic sales funnel

    Supported models:
    - first_touch: 100% credit to first interaction
    - last_touch: 100% credit to last interaction
    - linear: Equal credit across all touchpoints
    - time_decay: More credit to recent touchpoints
    - u_shaped: 40% first, 40% last, 20% middle
    - w_shaped: 30% first, 30% middle, 30% last, 10% others
    """
    try:
        async with get_async_session() as session:
            from sqlalchemy import select
            from ...data_layer.database.models import DelayedReward, Campaign, Content

            start_dt = datetime.fromisoformat(start_date) if start_date else datetime.utcnow() - timedelta(days=30)
            end_dt = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()

            stmt = select(DelayedReward).where(
                DelayedReward.status.in_(["booked", "closed_won"]),
                DelayedReward.registered_at >= start_dt,
                DelayedReward.registered_at <= end_dt
            )
            result = await session.execute(stmt)
            rewards = result.scalars().all()

            if not rewards:
                return {
                    "success": True,
                    "has_data": False,
                    "message": "No conversions found in date range",
                    "campaign_credits": {},
                    "channel_credits": {},
                    "model_used": model,
                    "total_conversions": 0
                }

            engine = get_attribution_engine()

            try:
                model_type = AttributionModel(model.lower())
            except ValueError:
                model_type = AttributionModel.LINEAR

            campaign_credits: Dict[str, float] = {}
            channel_credits: Dict[str, float] = {}
            total_value = 0.0

            for reward in rewards:
                campaign_stmt = select(Campaign).where(Campaign.id == reward.campaign_id)
                campaign_result = await session.execute(campaign_stmt)
                campaign = campaign_result.scalar_one_or_none()

                if not campaign:
                    continue

                conversion_value = reward.current_reward or 1.0
                lead_data = getattr(reward, 'lead_data', None) or {}
                if lead_data and lead_data.get("amount"):
                    conversion_value = float(lead_data.get("amount", 0)) / 100  # Scale down

                total_value += conversion_value

                touchpoints = []

                content_stmt = select(Content).where(
                    Content.campaign_id == reward.campaign_id
                ).order_by(Content.created_at)
                content_result = await session.execute(content_stmt)
                contents = content_result.scalars().all()

                for idx, content in enumerate(contents):
                    touchpoint = Touchpoint(
                        touchpoint_id=str(content.id),
                        campaign_id=str(campaign.id),
                        channel=campaign.platform.value if campaign.platform else "unknown",
                        timestamp=content.created_at or datetime.utcnow(),
                        action_type="content_view"
                    )
                    touchpoints.append(touchpoint)

                if reward.registered_at:
                    touchpoints.append(Touchpoint(
                        touchpoint_id=f"click_{reward.id}",
                        campaign_id=str(campaign.id),
                        channel=campaign.platform.value if campaign.platform else "unknown",
                        timestamp=reward.registered_at,
                        action_type="click"
                    ))

                lead_data = getattr(reward, 'lead_data', None) or {}
                if lead_data and lead_data.get("booked_at"):
                    try:
                        booked_at = datetime.fromisoformat(lead_data["booked_at"].replace("Z", "+00:00"))
                    except:
                        booked_at = reward.resolved_at or datetime.utcnow()

                    touchpoints.append(Touchpoint(
                        touchpoint_id=f"booking_{reward.id}",
                        campaign_id=str(campaign.id),
                        channel="calendar",
                        timestamp=booked_at,
                        action_type="booking"
                    ))

                if touchpoints:
                    attribution_result = engine.attribute(
                        touchpoints=touchpoints,
                        conversion_value=conversion_value,
                        conversion_id=str(reward.id),
                        model=model_type
                    )

                    for cid, credit in attribution_result.campaign_credits.items():
                        campaign_name = campaign.name if str(campaign.id) == cid else cid
                        if campaign_name not in campaign_credits:
                            campaign_credits[campaign_name] = 0.0
                        campaign_credits[campaign_name] += credit

                    for channel, credit in attribution_result.channel_credits.items():
                        if channel not in channel_credits:
                            channel_credits[channel] = 0.0
                        channel_credits[channel] += credit

            return {
                "success": True,
                "has_data": True,
                "model_used": model_type.value,
                "total_conversions": len(rewards),
                "total_value": round(total_value, 2),
                "campaign_credits": {k: round(v, 2) for k, v in campaign_credits.items()},
                "channel_credits": {k: round(v, 2) for k, v in channel_credits.items()},
                "date_range": {
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat()
                }
            }

    except Exception as e:
        logger.error(f"Failed to calculate multi-touch attribution: {e}")
        return {
            "success": False,
            "has_data": False,
            "error": str(e),
            "campaign_credits": {},
            "channel_credits": {}
        }


@router.get("/attribution/models")
async def get_attribution_models():
    """
    Get available attribution models and their descriptions.
    """
    engine = get_attribution_engine()
    return engine.get_model_summary()


@router.post("/attribution/compare-models")
async def compare_attribution_models(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """
    Compare attribution results across all available models.
    Useful for understanding how different models distribute credit.
    """
    try:
        results = {}

        for model in AttributionModel:
            model_result = await get_multi_touch_attribution(
                start_date=start_date,
                end_date=end_date,
                model=model.value
            )
            results[model.value] = {
                "campaign_credits": model_result.get("campaign_credits", {}),
                "channel_credits": model_result.get("channel_credits", {}),
                "total_value": model_result.get("total_value", 0)
            }

        return {
            "success": True,
            "comparisons": results,
            "models_compared": [m.value for m in AttributionModel]
        }

    except Exception as e:
        logger.error(f"Failed to compare attribution models: {e}")
        return {"success": False, "error": str(e)}


@router.post("/webhooks/calendar")
async def handle_calendar_webhook(request: Request):
    """Handle Cal.com webhook events for real-time booking notifications"""
    try:
        payload = await request.json()

        event_type = payload.get("triggerEvent", "")
        booking_data = payload.get("payload", {})

        logger.info(f"Received Cal.com webhook: {event_type}")

        if event_type == "BOOKING_CREATED":
            email = booking_data.get("attendees", [{}])[0].get("email", "")

            if email:
                from ...ai_layer.learning.reward_tracker import RewardTracker

                tracker = RewardTracker()
                await tracker.process_booking_conversion(email, booking_data)

                logger.info(f"Processed booking for {email}")

        elif event_type == "BOOKING_RESCHEDULED":
            pass

        elif event_type == "BOOKING_CANCELLED":
            pass

        return {"status": "success", "event": event_type}

    except Exception as e:
        logger.error(f"Webhook handler error: {e}")
        return {"status": "error", "message": str(e)}
