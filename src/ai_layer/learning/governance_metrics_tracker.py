"""
Governance Metrics Tracker

Research Plan Section 10.2: Human Override Rate < 5% target

Tracks and measures governance metrics including human override rates,
safety scores, and golden test pass rates.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy import select, func, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from ...data_layer.database.models import (
    GovernanceMetrics, HITLQueue, Content, ContentStatus
)
from ...data_layer.database.connection import get_async_session

logger = logging.getLogger(__name__)


class GovernanceMetricsTracker:
    """
    Tracks governance metrics per Research Plan Section 10.2.

    Key metrics:
    - Human Override Rate: Target < 5%
    - Golden Test Pass Rate: Target 100%
    - Safety Scores: Averages for toxicity, factuality, brand alignment
    """

    OVERRIDE_RATE_TARGET = 5.0  # Target: < 5%
    GOLDEN_TEST_PASS_TARGET = 100.0  # Target: 100%

    async def calculate_current_override_rate(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Override Rate = (rejected + modified) / total_reviews * 100"""
        try:
            async with get_async_session() as session:
                cutoff = datetime.utcnow() - timedelta(days=days)

                query = select(
                    func.count(HITLQueue.id).label('total'),
                    func.sum(
                        case((HITLQueue.decision == 'approve', 1), else_=0)
                    ).label('approved'),
                    func.sum(
                        case((HITLQueue.decision == 'reject', 1), else_=0)
                    ).label('rejected'),
                    func.sum(
                        case((HITLQueue.decision == 'modify', 1), else_=0)
                    ).label('modified')
                ).where(
                    HITLQueue.status == 'completed',
                    HITLQueue.completed_at >= cutoff
                )

                result = await session.execute(query)
                row = result.first()

                total = row.total or 0
                approved = row.approved or 0
                rejected = row.rejected or 0
                modified = row.modified or 0

                override_count = rejected + modified
                override_rate = (override_count / total * 100) if total > 0 else 0.0

                meets_target = override_rate < self.OVERRIDE_RATE_TARGET

                return {
                    "period_days": days,
                    "total_reviews": total,
                    "approved_count": approved,
                    "rejected_count": rejected,
                    "modified_count": modified,
                    "override_count": override_count,
                    "override_rate": round(override_rate, 2),
                    "target": self.OVERRIDE_RATE_TARGET,
                    "meets_target": meets_target,
                    "status": "PASS" if meets_target else "FAIL",
                    "gap_to_target": round(self.OVERRIDE_RATE_TARGET - override_rate, 2)
                }

        except Exception as e:
            logger.error(f"Failed to calculate override rate: {e}")
            return {
                "error": str(e),
                "period_days": days,
                "override_rate": 0.0,
                "meets_target": False
            }

    async def calculate_safety_score_averages(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        try:
            async with get_async_session() as session:
                cutoff = datetime.utcnow() - timedelta(days=days)

                query = select(
                    func.count(Content.id).label('total'),
                    func.avg(Content.safety_score).label('avg_safety'),
                    func.avg(Content.toxicity_score).label('avg_toxicity'),
                    func.avg(Content.factuality_score).label('avg_factuality'),
                    func.avg(Content.brand_alignment_score).label('avg_brand')
                ).where(
                    Content.created_at >= cutoff
                )

                result = await session.execute(query)
                row = result.first()

                return {
                    "period_days": days,
                    "total_content": row.total or 0,
                    "avg_safety_score": round(row.avg_safety or 0.0, 3),
                    "avg_toxicity_score": round(row.avg_toxicity or 0.0, 3),
                    "avg_factuality_score": round(row.avg_factuality or 0.0, 3),
                    "avg_brand_alignment_score": round(row.avg_brand or 0.0, 3),
                    "toxicity_target": 0.1,  # < 0.1 target
                    "meets_toxicity_target": (row.avg_toxicity or 1.0) < 0.1
                }

        except Exception as e:
            logger.error(f"Failed to calculate safety averages: {e}")
            return {"error": str(e)}

    async def save_period_metrics(
        self,
        period_type: str = "daily"
    ) -> Optional[str]:
        try:
            now = datetime.utcnow()
            if period_type == "daily":
                period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                period_end = period_start + timedelta(days=1)
                days = 1
            elif period_type == "weekly":
                period_start = now - timedelta(days=now.weekday())
                period_start = period_start.replace(hour=0, minute=0, second=0, microsecond=0)
                period_end = period_start + timedelta(days=7)
                days = 7
            else:  # monthly
                period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if now.month == 12:
                    period_end = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    period_end = now.replace(month=now.month + 1, day=1)
                days = (period_end - period_start).days

            override_data = await self.calculate_current_override_rate(days)
            safety_data = await self.calculate_safety_score_averages(days)

            async with get_async_session() as session:
                review_time_query = select(
                    func.avg(
                        func.extract('epoch', HITLQueue.completed_at - HITLQueue.created_at) / 60
                    ).label('avg_time')
                ).where(
                    HITLQueue.status == 'completed',
                    HITLQueue.completed_at >= period_start
                )
                review_result = await session.execute(review_time_query)
                avg_review_time = review_result.scalar() or 0.0

                metrics = GovernanceMetrics(
                    period_start=period_start,
                    period_end=period_end,
                    period_type=period_type,
                    total_reviews=override_data.get('total_reviews', 0),
                    approved_count=override_data.get('approved_count', 0),
                    rejected_count=override_data.get('rejected_count', 0),
                    modified_count=override_data.get('modified_count', 0),
                    human_override_rate=override_data.get('override_rate', 0.0),
                    override_rate_target=self.OVERRIDE_RATE_TARGET,
                    meets_override_target=override_data.get('meets_target', True),
                    avg_safety_score=safety_data.get('avg_safety_score', 0.0),
                    avg_toxicity_score=safety_data.get('avg_toxicity_score', 0.0),
                    avg_factuality_score=safety_data.get('avg_factuality_score', 0.0),
                    avg_brand_alignment_score=safety_data.get('avg_brand_alignment_score', 0.0),
                    avg_review_time_minutes=avg_review_time
                )

                session.add(metrics)
                await session.commit()
                await session.refresh(metrics)

                logger.info(
                    f"Saved {period_type} governance metrics: "
                    f"override_rate={metrics.human_override_rate:.2f}%"
                )

                return str(metrics.id)

        except Exception as e:
            logger.error(f"Failed to save period metrics: {e}")
            return None

    async def get_override_rate_trend(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        try:
            async with get_async_session() as session:
                query = select(GovernanceMetrics).where(
                    GovernanceMetrics.period_type == "daily",
                    GovernanceMetrics.created_at >= datetime.utcnow() - timedelta(days=days)
                ).order_by(GovernanceMetrics.period_start)

                result = await session.execute(query)
                records = result.scalars().all()

                return [
                    {
                        "date": r.period_start.date().isoformat(),
                        "override_rate": r.human_override_rate,
                        "total_reviews": r.total_reviews,
                        "meets_target": r.meets_override_target,
                        "target": self.OVERRIDE_RATE_TARGET
                    }
                    for r in records
                ]

        except Exception as e:
            logger.error(f"Failed to get override rate trend: {e}")
            return []

    async def get_dashboard_summary(self) -> Dict[str, Any]:
        override_7d = await self.calculate_current_override_rate(7)
        override_30d = await self.calculate_current_override_rate(30)
        safety_7d = await self.calculate_safety_score_averages(7)

        return {
            "human_override_rate": {
                "current_7d": override_7d.get('override_rate', 0.0),
                "current_30d": override_30d.get('override_rate', 0.0),
                "target": self.OVERRIDE_RATE_TARGET,
                "status_7d": "PASS" if override_7d.get('meets_target') else "FAIL",
                "status_30d": "PASS" if override_30d.get('meets_target') else "FAIL"
            },
            "safety_scores": {
                "avg_safety": safety_7d.get('avg_safety_score', 0.0),
                "avg_toxicity": safety_7d.get('avg_toxicity_score', 0.0),
                "meets_toxicity_target": safety_7d.get('meets_toxicity_target', True)
            },
            "review_volume": {
                "total_7d": override_7d.get('total_reviews', 0),
                "approved_7d": override_7d.get('approved_count', 0),
                "rejected_7d": override_7d.get('rejected_count', 0)
            }
        }


# Convenience function(s)
async def save_daily_governance_metrics() -> Optional[str]:
    tracker = GovernanceMetricsTracker()
    return await tracker.save_period_metrics("daily")


async def get_current_override_rate() -> Dict[str, Any]:
    tracker = GovernanceMetricsTracker()
    return await tracker.calculate_current_override_rate(7)
