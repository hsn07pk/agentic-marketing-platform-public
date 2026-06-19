"""
Budget management and cost control
"""
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
import redis.asyncio as redis

from ..data_layer.database.models import Campaign, CostTracking
from ..config.settings import settings

logger = logging.getLogger(__name__)

class CostCategory(str, Enum):

    API_CALLS = "api_calls"
    INFRASTRUCTURE = "infrastructure"
    PLATFORM_FEES = "platform_fees"
    CONTENT_GENERATION = "content_generation"

@dataclass
class BudgetStatus:

    limit: float
    spent: float
    remaining: float
    exceeded: bool
    percentage_used: float
    projected_overage: Optional[float] = None

class BudgetManager:

    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.daily_limit = settings.MAX_DAILY_API_COST
        self.campaign_limit = settings.MAX_CAMPAIGN_COST
        
    async def check_budget(
        self,
        campaign_id: str,
        estimated_cost: float = 0.0
    ) -> bool:
        """Check if campaign can proceed with estimated cost."""
        try:
            result = await self.db.execute(
                select(Campaign).where(Campaign.id == campaign_id)
            )
            campaign = result.scalar_one_or_none()
            
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            spent = campaign.budget_spent + estimated_cost
            limit = campaign.budget_total
            remaining = limit - spent
            exceeded = spent > limit
            percentage_used = (spent / limit * 100) if limit > 0 else 0

            if percentage_used > 80 and not exceeded:
                logger.warning(f"Campaign {campaign_id} at {percentage_used:.1f}% of budget")
            elif exceeded:
                logger.error(f"Campaign {campaign_id} exceeded budget by €{spent - limit:.2f}")

            return not exceeded
            
        except Exception as e:
            logger.error(f"Budget check failed: {e}")
            # Return False (exceeded) on error to be conservative
            return False
    
    async def track_cost(
        self,
        source_type: str,
        cost_amount: float,
        campaign_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Track cost expenditure."""
        try:
            cost_record = CostTracking(
                source_type=source_type,
                source_id=campaign_id or "system",
                cost_amount=cost_amount,
                campaign_id=campaign_id,
                provider=metadata.get("provider") if metadata else None,
                tokens_prompt=metadata.get("tokens_prompt") if metadata else None,
                tokens_completion=metadata.get("tokens_completion") if metadata else None,
                timestamp=datetime.utcnow()
            )
            
            self.db.add(cost_record)
            
            if campaign_id:
                from sqlalchemy import update
                stmt = (
                    update(Campaign)
                    .where(Campaign.id == campaign_id)
                    .values(budget_spent=Campaign.budget_spent + cost_amount)
                )
                await self.db.execute(stmt)
            
            await self.db.commit()
            
            today_key = f"cost:daily:{datetime.utcnow().date()}"
            await self.redis.incrbyfloat(today_key, cost_amount)
            await self.redis.expire(today_key, 86400 * 7)  # Keep for 7 days
            
            daily_total = float(await self.redis.get(today_key) or 0)
            if daily_total > self.daily_limit:
                logger.error(f"Daily cost limit exceeded: €{daily_total:.2f} / €{self.daily_limit:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cost tracking failed: {e}")
            await self.db.rollback()
            return False
    
    async def get_cost_summary(
        self,
        campaign_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get cost summary."""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            query = select(
                func.sum(CostTracking.cost_amount).label("total_cost"),
                func.count(CostTracking.id).label("total_transactions"),
                CostTracking.source_type,
                func.avg(CostTracking.cost_amount).label("avg_cost")
            ).where(
                CostTracking.timestamp >= since_date
            ).group_by(CostTracking.source_type)
            
            if campaign_id:
                query = query.where(CostTracking.campaign_id == campaign_id)
            
            result = await self.db.execute(query)
            costs_by_type = result.all()
            
            daily_costs = []
            for i in range(days):
                date = datetime.utcnow().date() - timedelta(days=i)
                day_key = f"cost:daily:{date}"
                day_cost = await self.redis.get(day_key)
                if day_cost:
                    daily_costs.append({
                        "date": date.isoformat(),
                        "cost": float(day_cost)
                    })
            
            total_cost = sum(row.total_cost or 0 for row in costs_by_type)
            
            return {
                "period_days": days,
                "total_cost": total_cost,
                "daily_average": total_cost / days if days > 0 else 0,
                "by_category": [
                    {
                        "category": row.source_type,
                        "total": row.total_cost or 0,
                        "count": row.total_transactions,
                        "average": row.avg_cost or 0
                    }
                    for row in costs_by_type
                ],
                "daily_costs": sorted(daily_costs, key=lambda x: x["date"], reverse=True)[:7],
                "daily_limit": self.daily_limit,
                "daily_limit_status": "exceeded" if total_cost / days > self.daily_limit else "ok"
            }
            
        except Exception as e:
            logger.error(f"Failed to get cost summary: {e}")
            return {}

    async def get_campaign_costs(self, campaign_id: str) -> float:
        """Get total costs for a specific campaign."""
        summary = await self.get_cost_summary(campaign_id=campaign_id, days=365)
        return summary.get('total_cost', 0.0)

    async def enforce_limits(self, campaign_id: str) -> bool:
        """Enforce budget limits, pausing campaign if exceeded."""
        within_budget = await self.check_budget(campaign_id)

        if not within_budget:
            from ..data_layer.database.models import CampaignStatus
            from sqlalchemy import update

            stmt = (
                update(Campaign)
                .where(Campaign.id == campaign_id)
                .values(status=CampaignStatus.PAUSED.value)
            )
            await self.db.execute(stmt)
            await self.db.commit()

            logger.warning(f"Campaign {campaign_id} paused due to budget limit")
            return False

        return True
