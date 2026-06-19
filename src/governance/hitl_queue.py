import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
import logging
from enum import Enum
from dataclasses import dataclass
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_
import redis.asyncio as redis

from ..data_layer.database.models import HITLQueue, Content
from ..config.settings import settings

logger = logging.getLogger(__name__)

class ReviewPriority(int, Enum):
    CRITICAL = 10
    HIGH = 7
    MEDIUM = 5
    LOW = 3
    MINIMAL = 1

@dataclass
class ReviewItem:
    id: str
    content_id: str
    priority: int
    reason: str
    content_data: Dict[str, Any]
    safety_scores: Dict[str, float]
    created_at: datetime
    assigned_to: Optional[str] = None
    deadline: Optional[datetime] = None

class HITLQueueManager:
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.queue_key = "hitl:queue"
        self.processing_key = "hitl:processing"
        self.completed_key = "hitl:completed"
        
        self.new_item_event = asyncio.Event()
        
    async def add_for_review(
        self,
        content_id: str,
        priority: int,
        reason: str,
        content_data: Dict[str, Any] = None,
        safety_scores: Dict[str, float] = None
    ) -> HITLQueue:
        """Add content to review queue."""
        try:
            review_id = str(uuid4())
            
            queue_item = HITLQueue(
                id=UUID(review_id),
                content_id=UUID(content_id),
                priority=priority,
                reason=reason,
                status="pending",
                created_at=datetime.utcnow()
            )
            
            self.db.add(queue_item)
            await self.db.commit()
            
            review_data = {
                "id": review_id,
                "content_id": content_id,
                "priority": priority,
                "reason": reason,
                "content_data": content_data or {},
                "safety_scores": safety_scores or {},
                "created_at": datetime.utcnow().isoformat(),
                "status": "pending"
            }
            
            self.redis.zadd(
                self.queue_key,
                {json.dumps(review_data): -priority}
            )
            
            self.new_item_event.set()
            
            logger.info(f"Added content {content_id} to HITL queue with priority {priority}")

            return queue_item

        except Exception as e:
            logger.error(f"Failed to add to HITL queue: {e}")
            await self.db.rollback()
            raise

    async def get_pending_items(self) -> List[HITLQueue]:
        """Get all pending items in the queue."""
        try:
            result = await self.db.execute(
                select(HITLQueue)
                .where(HITLQueue.status == "pending")
                .order_by(HITLQueue.priority.desc(), HITLQueue.created_at.asc())
            )
            items = result.scalars().all()
            return list(items)
        except Exception as e:
            logger.error(f"Failed to get pending items: {e}")
            return []

    async def get_next_for_review(
        self,
        reviewer_id: Optional[str] = None,
        timeout: int = 0
    ) -> Optional[ReviewItem]:
        """Get next item for review, optionally waiting up to timeout seconds."""
        try:
            start_time = datetime.utcnow()
            
            while True:
                items = self.redis.zrange(self.queue_key, 0, 0)
                
                if items:
                    item_data = json.loads(items[0])
                    
                    self.redis.zrem(self.queue_key, items[0])
                    
                    processing_data = {
                        **item_data,
                        "assigned_to": reviewer_id,
                        "assigned_at": datetime.utcnow().isoformat()
                    }
                    
                    self.redis.hset(
                        self.processing_key,
                        item_data["id"],
                        json.dumps(processing_data)
                    )
                    await self.db.execute(
                        update(HITLQueue)
                        .where(HITLQueue.id == UUID(item_data["id"]))
                        .values(
                            status="reviewing",
                            assigned_to=reviewer_id,
                            assigned_at=datetime.utcnow()
                        )
                    )
                    await self.db.commit()
                    
                    return ReviewItem(
                        id=item_data["id"],
                        content_id=item_data["content_id"],
                        priority=item_data["priority"],
                        reason=item_data["reason"],
                        content_data=item_data.get("content_data", {}),
                        safety_scores=item_data.get("safety_scores", {}),
                        created_at=datetime.fromisoformat(item_data["created_at"]),
                        assigned_to=reviewer_id
                    )
                if timeout == 0:
                    return None
                elapsed = (datetime.utcnow() - start_time).seconds
                remaining_timeout = timeout - elapsed
                
                if remaining_timeout <= 0:
                    return None
                try:
                    await asyncio.wait_for(
                        self.new_item_event.wait(),
                        timeout=min(remaining_timeout, 5)
                    )
                    self.new_item_event.clear()
                except asyncio.TimeoutError:
                    if (datetime.utcnow() - start_time).seconds >= timeout:
                        return None
                    
        except Exception as e:
            logger.error(f"Failed to get next review item: {e}")
            return None
    
    async def submit_review(
        self,
        review_id: str,
        decision: str,
        reviewer_id: str,
        feedback: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Submit review decision (approve/reject/modify)."""
        try:
            processing_data = self.redis.hget(self.processing_key, review_id)
            
            if not processing_data:
                logger.error(f"Review item {review_id} not found in processing")
                return False
            
            item_data = json.loads(processing_data)
            
            await self.db.execute(
                update(HITLQueue)
                .where(HITLQueue.id == UUID(review_id))
                .values(
                    status="completed",
                    decision=decision,
                    feedback=feedback,
                    modifications=modifications,
                    completed_at=datetime.utcnow()
                )
            )
            from ..data_layer.database.models import ContentStatus
            
            if decision == "approve":
                content_status = ContentStatus.APPROVED
            elif decision == "reject":
                content_status = ContentStatus.REJECTED
            else:
                content_status = ContentStatus.GENERATED
            
            await self.db.execute(
                update(Content)
                .where(Content.id == UUID(item_data["content_id"]))
                .values(
                    status=content_status,
                    review_notes=feedback,
                    reviewed_by=reviewer_id,
                    reviewed_at=datetime.utcnow()
                )
            )
            
            await self.db.commit()
            
            self.redis.hdel(self.processing_key, review_id)
            
            completed_data = {
                **item_data,
                "decision": decision,
                "feedback": feedback,
                "reviewer_id": reviewer_id,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            self.redis.hset(
                self.completed_key,
                review_id,
                json.dumps(completed_data)
            )
            
            self.redis.expire(self.completed_key, 7 * 24 * 3600)
            
            logger.info(f"Review {review_id} completed with decision: {decision}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit review: {e}")
            await self.db.rollback()
            return False

    async def approve_content(
        self,
        queue_item_id: str,
        reviewer_email: str,
        feedback: Optional[str] = None
    ) -> bool:
        """Approve content (convenience method for submit_review)."""
        return await self.submit_review(
            review_id=queue_item_id,
            decision="approved",
            reviewer_id=reviewer_email,
            feedback=feedback
        )

    async def reject_content(
        self,
        queue_item_id: str,
        reviewer_email: str,
        feedback: str
    ) -> bool:
        """Reject content (convenience method for submit_review)."""
        return await self.submit_review(
            review_id=queue_item_id,
            decision="rejected",
            reviewer_id=reviewer_email,
            feedback=feedback
        )

    async def wait_for_review(
        self,
        review_id: str,
        timeout_seconds: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """Wait for review completion, returning result or None on timeout."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout_seconds:
            completed_data = self.redis.hget(self.completed_key, review_id)
            
            if completed_data:
                return json.loads(completed_data)
            
            processing_data = self.redis.hget(self.processing_key, review_id)
            
            if not processing_data:
                result = await self.db.execute(
                    select(HITLQueue)
                    .where(HITLQueue.id == UUID(review_id))
                )
                queue_item = result.scalar_one_or_none()
                
                if queue_item and queue_item.status == "completed":
                    return {
                        "decision": queue_item.decision,
                        "feedback": queue_item.feedback,
                        "modifications": queue_item.modifications
                    }
            
            await asyncio.sleep(5)
        
        logger.warning(f"Review {review_id} timed out after {timeout_seconds} seconds")
        return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            queue_size = self.redis.zcard(self.queue_key)
            processing_count = self.redis.hlen(self.processing_key)
            
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
            
            result = await self.db.execute(
                select(HITLQueue)
                .where(
                    and_(
                        HITLQueue.completed_at >= today_start,
                        HITLQueue.status == "completed"
                    )
                )
            )
            completed_today = len(result.scalars().all())
            result = await self.db.execute(
                select(HITLQueue)
                .where(HITLQueue.status == "completed")
                .limit(100)
            )
            completed_items = result.scalars().all()
            
            if completed_items:
                review_times = [
                    (item.completed_at - item.created_at).seconds
                    for item in completed_items
                    if item.completed_at
                ]
                avg_review_time = sum(review_times) / len(review_times) if review_times else 0
            else:
                avg_review_time = 0
            
            return {
                "queue_size": queue_size,
                "processing": processing_count,
                "completed_today": completed_today,
                "average_review_time_seconds": avg_review_time,
                "oldest_item_hours": await self._get_oldest_item_age()
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def _get_oldest_item_age(self) -> float:
        """Get age of oldest item in queue (hours)"""
        try:
            items = self.redis.zrange(self.queue_key, -1, -1)
            
            if items:
                item_data = json.loads(items[0])
                created_at = datetime.fromisoformat(item_data["created_at"])
                age_hours = (datetime.utcnow() - created_at).seconds / 3600
                return age_hours
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get oldest item age: {e}")
            return 0.0
