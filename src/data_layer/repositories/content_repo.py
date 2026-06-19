"""
Content Repository - Data access layer for marketing content
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func, desc
from sqlalchemy.orm import joinedload
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import logging

from ..database.models import Content, Campaign, ContentStatus, Platform
from ...config.settings import settings

logger = logging.getLogger(__name__)

class ContentRepository:

    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, content_data: Dict[str, Any]) -> Content:
        """Create new content."""
        start_time = datetime.now()

        try:
            content = Content(**content_data)
            self.session.add(content)
            await self.session.commit()
            await self.session.refresh(content)

            duration = (datetime.now() - start_time).total_seconds()

            logger.info(
                "Content created successfully",
                extra={
                    "event": "content_created",
                    "content_id": str(content.id),
                    "campaign_id": str(content.campaign_id),
                    "platform": content_data.get('platform'),
                    "status": content.status.value if hasattr(content.status, 'value') else content.status,
                    "duration_seconds": round(duration, 3),
                    "headline": content.headline[:50] if content.headline else None
                }
            )
            return content
        except Exception as e:
            await self.session.rollback()
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Failed to create content",
                extra={
                    "event": "content_create_error",
                    "campaign_id": str(content_data.get('campaign_id', 'unknown')),
                    "platform": content_data.get('platform'),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 3)
                },
                exc_info=True
            )
            raise
    
    async def get_by_id(self, content_id: str) -> Optional[Content]:
        start_time = datetime.now()

        try:
            stmt = (
                select(Content)
                .options(joinedload(Content.campaign))
                .where(Content.id == UUID(content_id))
            )
            result = await self.session.execute(stmt)
            content = result.scalar_one_or_none()

            duration = (datetime.now() - start_time).total_seconds()

            if content:
                logger.info(
                    "Content retrieved successfully",
                    extra={
                        "event": "content_retrieved",
                        "content_id": content_id,
                        "campaign_id": str(content.campaign_id) if content.campaign_id else None,
                        "status": content.status.value if hasattr(content.status, 'value') else content.status,
                        "duration_seconds": round(duration, 3)
                    }
                )
            else:
                logger.warning(
                    "Content not found",
                    extra={
                        "event": "content_not_found",
                        "content_id": content_id,
                        "duration_seconds": round(duration, 3)
                    }
                )

            return content
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Failed to get content",
                extra={
                    "event": "content_get_error",
                    "content_id": content_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 3)
                },
                exc_info=True
            )
            return None
    
    async def get_by_campaign(
        self,
        campaign_id: str,
        status: Optional[ContentStatus] = None
    ) -> List[Content]:
        try:
            stmt = select(Content).where(Content.campaign_id == UUID(campaign_id))
            
            if status:
                stmt = stmt.where(Content.status == status)
            
            stmt = stmt.order_by(desc(Content.created_at))
            
            result = await self.session.execute(stmt)
            contents = result.scalars().all()
            
            return list(contents)
        except Exception as e:
            logger.error(f"Failed to get content for campaign {campaign_id}: {e}")
            return []
    
    async def update(
        self,
        content_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Content]:
        start_time = datetime.now()

        try:
            stmt = (
                update(Content)
                .where(Content.id == UUID(content_id))
                .values(**updates)
                .returning(Content)
            )
            result = await self.session.execute(stmt)
            await self.session.commit()

            content = result.scalar_one_or_none()
            duration = (datetime.now() - start_time).total_seconds()

            if content:
                logger.info(
                    "Content updated successfully",
                    extra={
                        "event": "content_updated",
                        "content_id": content_id,
                        "updates": list(updates.keys()),
                        "duration_seconds": round(duration, 3)
                    }
                )
            else:
                logger.warning(
                    "Content not found for update",
                    extra={
                        "event": "content_update_not_found",
                        "content_id": content_id,
                        "duration_seconds": round(duration, 3)
                    }
                )

            return content
        except Exception as e:
            await self.session.rollback()
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Failed to update content",
                extra={
                    "event": "content_update_error",
                    "content_id": content_id,
                    "updates": list(updates.keys()),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 3)
                },
                exc_info=True
            )
            return None
    
    async def update_status(
        self,
        content_id: str,
        status: ContentStatus,
        reviewed_by: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> bool:
        try:
            updates = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if reviewed_by:
                updates["reviewed_by"] = reviewed_by
                updates["reviewed_at"] = datetime.utcnow()
            
            if feedback:
                updates["review_feedback"] = feedback
            
            result = await self.update(content_id, updates)
            return result is not None
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False
    
    async def get_pending_review(
        self,
        priority_threshold: float = 0.8,
        limit: int = 100
    ) -> List[Content]:
        try:
            stmt = (
                select(Content)
                .options(joinedload(Content.campaign))
                .where(
                    and_(
                        Content.status == ContentStatus.PENDING_REVIEW,
                        Content.safety_score <= priority_threshold
                    )
                )
                .order_by(Content.safety_score.asc(), Content.created_at.asc())
                .limit(limit)
            )
            
            result = await self.session.execute(stmt)
            contents = result.scalars().all()
            
            return list(contents)
        except Exception as e:
            logger.error(f"Failed to get pending review content: {e}")
            return []
    
    async def get_approved_for_deployment(
        self,
        campaign_id: Optional[str] = None,
        platform: Optional[Platform] = None
    ) -> List[Content]:
        try:
            filters = [Content.status == ContentStatus.APPROVED]
            
            if campaign_id:
                filters.append(Content.campaign_id == UUID(campaign_id))
            
            if platform:
                stmt = (
                    select(Content)
                    .join(Campaign)
                    .where(and_(*filters, Campaign.platform == platform))
                    .order_by(desc(Content.created_at))
                )
            else:
                stmt = (
                    select(Content)
                    .where(and_(*filters))
                    .order_by(desc(Content.created_at))
                )
            
            result = await self.session.execute(stmt)
            contents = result.scalars().all()
            
            return list(contents)
        except Exception as e:
            logger.error(f"Failed to get approved content: {e}")
            return []
    
    async def mark_deployed(
        self,
        content_id: str,
        platform_post_id: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark content as deployed with platform details."""
        start_time = datetime.now()

        try:
            platform_data = {
                "platform_post_id": platform_post_id
            }
            if metrics:
                platform_data["deployment_metrics"] = metrics

            updates = {
                "status": ContentStatus.DEPLOYED,
                "deployed_at": datetime.utcnow(),
                "platform_specific": platform_data
            }

            if metrics:
                updates["impressions"] = metrics.get("impressions", 0)
                updates["clicks"] = metrics.get("clicks", 0)
                updates["conversions"] = metrics.get("conversions", 0)
                if updates["impressions"] > 0:
                    updates["engagement_rate"] = updates["clicks"] / updates["impressions"]
                else:
                    updates["engagement_rate"] = 0.0

            result = await self.update(content_id, updates)
            duration = (datetime.now() - start_time).total_seconds()

            if result is not None:
                logger.info(
                    "Content marked as deployed",
                    extra={
                        "event": "content_deployed",
                        "content_id": content_id,
                        "platform_post_id": platform_post_id,
                        "has_metrics": metrics is not None,
                        "duration_seconds": round(duration, 3)
                    }
                )
                return True
            else:
                logger.warning(
                    "Failed to mark content as deployed - content not found",
                    extra={
                        "event": "content_deploy_not_found",
                        "content_id": content_id,
                        "platform_post_id": platform_post_id,
                        "duration_seconds": round(duration, 3)
                    }
                )
                return False
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Failed to mark content as deployed",
                extra={
                    "event": "content_deploy_error",
                    "content_id": content_id,
                    "platform_post_id": platform_post_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 3)
                },
                exc_info=True
            )
            return False
    
    async def get_performance_stats(
        self,
        campaign_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        try:
            filters = [Content.status == ContentStatus.DEPLOYED]
            
            if campaign_id:
                filters.append(Content.campaign_id == UUID(campaign_id))
            
            if start_date:
                filters.append(Content.deployed_at >= start_date)
            
            if end_date:
                filters.append(Content.deployed_at <= end_date)
            
            stmt = (
                select(
                    func.count(Content.id).label('total_content'),
                    func.avg(Content.safety_score).label('avg_safety_score'),
                    func.count(
                        func.distinct(Content.campaign_id)
                    ).label('campaigns_count')
                )
                .where(and_(*filters))
            )
            
            result = await self.session.execute(stmt)
            stats = result.first()
            
            if not stats:
                return {
                    "total_content": 0,
                    "avg_safety_score": 0.0,
                    "campaigns_count": 0
                }
            
            return {
                "total_content": stats.total_content or 0,
                "avg_safety_score": float(stats.avg_safety_score or 0.0),
                "campaigns_count": stats.campaigns_count or 0
            }
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {
                "total_content": 0,
                "avg_safety_score": 0.0,
                "campaigns_count": 0
            }
    
    async def get_safety_distribution(self) -> Dict[str, int]:
        try:
            ranges = [
                ("high", 0.9, 1.0),
                ("medium", 0.7, 0.9),
                ("low", 0.0, 0.7)
            ]
            
            distribution = {}
            
            for label, min_score, max_score in ranges:
                stmt = (
                    select(func.count(Content.id))
                    .where(
                        and_(
                            Content.safety_score >= min_score,
                            Content.safety_score < max_score
                        )
                    )
                )
                result = await self.session.execute(stmt)
                count = result.scalar_one()
                distribution[label] = count
            
            return distribution
        except Exception as e:
            logger.error(f"Failed to get safety distribution: {e}")
            return {"high": 0, "medium": 0, "low": 0}
    
    async def delete(self, content_id: str) -> bool:
        try:
            stmt = delete(Content).where(Content.id == UUID(content_id))
            await self.session.execute(stmt)
            await self.session.commit()
            
            logger.info(f"Deleted content: {content_id}")
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete content: {e}")
            return False
    
    async def search_by_text(
        self,
        query: str,
        limit: int = 50
    ) -> List[Content]:
        try:
            search_pattern = f"%{query}%"
            
            stmt = (
                select(Content)
                .where(
                    or_(
                        Content.headline.ilike(search_pattern),
                        Content.body.ilike(search_pattern)
                    )
                )
                .order_by(desc(Content.created_at))
                .limit(limit)
            )
            
            result = await self.session.execute(stmt)
            contents = result.scalars().all()
            
            return list(contents)
        except Exception as e:
            logger.error(f"Failed to search content: {e}")
            return []
    
    async def get_top_performing(
        self,
        metric: str = "ctr",
        limit: int = 10,
        platform: Optional[Platform] = None
    ) -> List[Content]:
        try:
            metric_paths = {
                "ctr": "$.ctr",
                "engagement": "$.engagement_rate",
                "conversions": "$.conversions"
            }
            
            metric_path = metric_paths.get(metric, "$.ctr")
            
            filters = [Content.status == ContentStatus.DEPLOYED]
            
            if platform:
                stmt = (
                    select(Content)
                    .join(Campaign)
                    .where(and_(*filters, Campaign.platform == platform))
                    .order_by(
                        desc(
                            func.cast(
                                func.json_extract_path_text(
                                    Content.deployment_metrics,
                                    metric
                                ),
                                Float
                            )
                        )
                    )
                    .limit(limit)
                )
            else:
                stmt = (
                    select(Content)
                    .where(and_(*filters))
                    .order_by(desc(Content.created_at))
                    .limit(limit)
                )
            
            result = await self.session.execute(stmt)
            contents = result.scalars().all()
            
            return list(contents)
        except Exception as e:
            logger.error(f"Failed to get top performing content: {e}")
            return []
