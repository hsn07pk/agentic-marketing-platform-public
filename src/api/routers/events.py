import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from datetime import datetime, timedelta

from ...data_layer.database.connection import async_session_maker
from ...data_layer.database.models import WorkflowEvent, AlertSeverity, WorkflowEventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@router.get("/campaign/{campaign_id}")
async def get_campaign_events(
    campaign_id: str,
    limit: int = Query(default=50, le=200),
    severity: Optional[str] = None,
    actionable_only: bool = False,
    include_dismissed: bool = False,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    try:
        query = select(WorkflowEvent).where(
            WorkflowEvent.campaign_id == campaign_id
        )

        if severity:
            query = query.where(WorkflowEvent.severity == severity)

        if actionable_only:
            query = query.where(WorkflowEvent.is_user_actionable == True)

        if not include_dismissed:
            query = query.where(WorkflowEvent.is_dismissed == False)

        query = query.order_by(desc(WorkflowEvent.created_at)).limit(limit)

        result = await db.execute(query)
        events = result.scalars().all()

        return [
            {
                "id": str(event.id),
                "campaign_id": str(event.campaign_id),
                "content_id": str(event.content_id) if event.content_id else None,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "workflow_node": event.workflow_node,
                "workflow_state": event.workflow_state,
                "title": event.title,
                "message": event.message,
                "details": event.details,
                "is_user_actionable": event.is_user_actionable,
                "is_dismissed": event.is_dismissed,
                "created_at": event.created_at.isoformat() if event.created_at else None
            }
            for event in events
        ]

    except Exception as e:
        logger.error(f"Failed to fetch campaign events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_active_alerts(
    campaign_id: Optional[str] = None,
    severity: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        query = select(WorkflowEvent).where(
            and_(
                WorkflowEvent.is_user_actionable == True,
                WorkflowEvent.is_dismissed == False
            )
        )

        if campaign_id:
            query = query.where(WorkflowEvent.campaign_id == campaign_id)

        if severity:
            query = query.where(WorkflowEvent.severity == severity)

        query = query.order_by(
            desc(WorkflowEvent.severity),
            desc(WorkflowEvent.created_at)
        )

        result = await db.execute(query)
        alerts = result.scalars().all()

        grouped = {
            "critical": [],
            "error": [],
            "warning": [],
            "info": []
        }

        for alert in alerts:
            grouped[alert.severity.value].append({
                "id": str(alert.id),
                "campaign_id": str(alert.campaign_id),
                "content_id": str(alert.content_id) if alert.content_id else None,
                "event_type": alert.event_type.value,
                "title": alert.title,
                "message": alert.message,
                "details": alert.details,
                "workflow_node": alert.workflow_node,
                "created_at": alert.created_at.isoformat() if alert.created_at else None
            })

        total_count = sum(len(v) for v in grouped.values())

        return {
            "total_count": total_count,
            "critical_count": len(grouped["critical"]),
            "error_count": len(grouped["error"]),
            "warning_count": len(grouped["warning"]),
            "info_count": len(grouped["info"]),
            "alerts": grouped
        }

    except Exception as e:
        logger.error(f"Failed to fetch active alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{event_id}/dismiss")
async def dismiss_event(
    event_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    try:
        result = await db.execute(
            select(WorkflowEvent).where(WorkflowEvent.id == event_id)
        )
        event = result.scalar_one_or_none()

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        event.is_dismissed = True
        event.dismissed_at = datetime.utcnow()

        await db.commit()

        return {"status": "success", "message": "Event dismissed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to dismiss event: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_events_summary(
    days: int = Query(default=7, le=30),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    try:
        since = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(WorkflowEvent).where(WorkflowEvent.created_at >= since)
        )
        events = result.scalars().all()

        total = len(events)
        by_type = {}
        by_severity = {
            "critical": 0,
            "error": 0,
            "warning": 0,
            "info": 0
        }

        for event in events:
            event_type = event.event_type.value
            by_type[event_type] = by_type.get(event_type, 0) + 1

            by_severity[event.severity.value] += 1

        actionable_count = sum(1 for e in events if e.is_user_actionable and not e.is_dismissed)

        return {
            "period_days": days,
            "total_events": total,
            "actionable_pending": actionable_count,
            "by_severity": by_severity,
            "by_type": by_type,
            "most_recent": {
                "id": str(events[0].id),
                "title": events[0].title,
                "severity": events[0].severity.value,
                "created_at": events[0].created_at.isoformat()
            } if events else None
        }

    except Exception as e:
        logger.error(f"Failed to get events summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-stale-hitl-alerts")
async def cleanup_stale_hitl_alerts(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Cleanup stale HITL alerts - dismiss hitl_queue_added events
    where the corresponding HITL queue item has already been completed.
    
    This is useful for cleaning up existing alerts that weren't
    properly dismissed when the HITL items were reviewed.
    """
    try:
        from sqlalchemy import update
        from ...data_layer.database.models import HITLQueue, Content
        
        result = await db.execute(
            select(WorkflowEvent).where(
                and_(
                    WorkflowEvent.event_type == WorkflowEventType.HITL_QUEUE_ADDED,
                    WorkflowEvent.is_user_actionable == True,
                    WorkflowEvent.is_dismissed == False
                )
            )
        )
        stale_events = result.scalars().all()
        
        dismissed_count = 0
        
        for event in stale_events:
            if event.content_id:
                hitl_result = await db.execute(
                    select(HITLQueue).where(
                        and_(
                            HITLQueue.content_id == event.content_id,
                            HITLQueue.status == "completed"
                        )
                    )
                )
                hitl_item = hitl_result.scalar_one_or_none()
                
                if hitl_item:
                    event.is_dismissed = True
                    event.dismissed_at = datetime.utcnow()
                    dismissed_count += 1
        
        await db.commit()
        
        logger.info(f"Cleaned up {dismissed_count} stale HITL alerts")
        
        return {
            "status": "success",
            "dismissed_count": dismissed_count,
            "total_checked": len(stale_events),
            "message": f"Dismissed {dismissed_count} stale HITL alerts"
        }
    
    except Exception as e:
        logger.error(f"Failed to cleanup stale HITL alerts: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-marl-alerts")
async def cleanup_marl_alerts(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Dismiss MARL gating alerts that are no longer actionable.
    
    MARL policy approvals/rejections are informational events.
    This cleans up all undismissed MARL gating events.
    """
    try:
        result = await db.execute(
            select(WorkflowEvent).where(
                and_(
                    WorkflowEvent.workflow_node == "marl_gating",
                    WorkflowEvent.is_dismissed == False
                )
            )
        )
        stale_events = result.scalars().all()
        
        dismissed_count = 0
        for event in stale_events:
            event.is_dismissed = True
            event.dismissed_at = datetime.utcnow()
            dismissed_count += 1
        
        await db.commit()
        
        logger.info(f"Cleaned up {dismissed_count} MARL alerts")
        
        return {
            "status": "success",
            "dismissed_count": dismissed_count,
            "total_checked": len(stale_events),
            "message": f"Dismissed {dismissed_count} MARL gating alerts"
        }
    
    except Exception as e:
        logger.error(f"Failed to cleanup MARL alerts: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

