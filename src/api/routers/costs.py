from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_db
from ...data_layer.database.models import CostTracking, Campaign, Content, SystemConfiguration
from ...data_layer.vector_store.semantic_cache import SemanticCache
from ...config.settings import settings
import redis

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Cost Control"]
)


async def _get_include_mock_config(db: AsyncSession) -> bool:
    """Read INCLUDE_MOCK_IN_METRICS from configuration service. Defaults to True."""
    try:
        result = await db.execute(
            select(SystemConfiguration.value).where(SystemConfiguration.key == "INCLUDE_MOCK_IN_METRICS")
        )
        val = result.scalar_one_or_none()
        if val is not None:
            return str(val).lower() in ('true', '1', 'yes', 'on')
        return True
    except Exception:
        return True


@router.get("/summary", response_model=Dict[str, Any])
async def get_cost_summary(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    include_mock: Optional[bool] = Query(None, description="Include costs from mock campaigns. Defaults to config."),
    db: AsyncSession = Depends(get_db)
):
    """
    Get cost summary for the specified time period.
    
    Returns total costs, cost by provider, and cost by agent type.
    """
    try:
        if include_mock is None:
            include_mock = await _get_include_mock_config(db)

        since_date = datetime.utcnow() - timedelta(days=days)
        
        base_conditions = [CostTracking.timestamp >= since_date]
        if not include_mock:
            mock_campaign_ids = select(Campaign.id).where(Campaign.is_mock == True).scalar_subquery()
            base_conditions.append(
                ~CostTracking.campaign_id.in_(mock_campaign_ids) | CostTracking.campaign_id.is_(None)
            )

        total_result = await db.execute(
            select(
                func.sum(CostTracking.cost_amount).label('total'),
                func.sum(CostTracking.tokens_prompt).label('total_prompt_tokens'),
                func.sum(CostTracking.tokens_completion).label('total_completion_tokens'),
                func.count(CostTracking.id).label('total_calls')
            ).where(and_(*base_conditions))
        )
        total_row = total_result.first()
        
        by_provider_result = await db.execute(
            select(
                CostTracking.provider,
                func.sum(CostTracking.cost_amount).label('cost'),
                func.count(CostTracking.id).label('calls')
            ).where(and_(*base_conditions))
            .group_by(CostTracking.provider)
        )
        by_provider = [
            {"provider": row.provider or "unknown", "cost": row.cost or 0, "calls": row.calls}
            for row in by_provider_result
        ]
        
        by_agent_result = await db.execute(
            select(
                CostTracking.agent_type,
                func.sum(CostTracking.cost_amount).label('cost'),
                func.count(CostTracking.id).label('calls')
            ).where(and_(*base_conditions))
            .group_by(CostTracking.agent_type)
        )
        by_agent = [
            {"agent_type": row.agent_type or "unknown", "cost": row.cost or 0, "calls": row.calls}
            for row in by_agent_result
        ]
        
        return {
            "period_days": days,
            "total_cost": total_row.total or 0,
            "total_prompt_tokens": total_row.total_prompt_tokens or 0,
            "total_completion_tokens": total_row.total_completion_tokens or 0,
            "total_calls": total_row.total_calls or 0,
            "by_provider": by_provider,
            "by_agent": by_agent,
            "since_date": since_date.isoformat(),
            "includes_mock_data": include_mock
        }
    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-model", response_model=Dict[str, Any])
async def get_costs_by_model(
    days: int = Query(30, ge=1, le=365),
    include_mock: Optional[bool] = Query(None, description="Include costs from mock campaigns. Defaults to config."),
    db: AsyncSession = Depends(get_db)
):
    """
    Get cost breakdown by AI model (GPT-4, GPT-3.5, embeddings, Ollama, etc.).
    """
    try:
        if include_mock is None:
            include_mock = await _get_include_mock_config(db)

        since_date = datetime.utcnow() - timedelta(days=days)
        
        conditions = [CostTracking.timestamp >= since_date]
        if not include_mock:
            mock_ids = select(Campaign.id).where(Campaign.is_mock == True).scalar_subquery()
            conditions.append(~CostTracking.campaign_id.in_(mock_ids) | CostTracking.campaign_id.is_(None))

        result = await db.execute(
            select(
                CostTracking.source_id,
                CostTracking.provider,
                func.sum(CostTracking.cost_amount).label('cost'),
                func.sum(CostTracking.tokens_prompt).label('prompt_tokens'),
                func.sum(CostTracking.tokens_completion).label('completion_tokens'),
                func.count(CostTracking.id).label('calls')
            ).where(and_(*conditions))
            .group_by(CostTracking.source_id, CostTracking.provider)
        )
        
        models = []
        total_cost = 0
        
        for row in result:
            cost = row.cost or 0
            total_cost += cost
            
            source = row.source_id or "unknown"
            
            models.append({
                "source": source,
                "provider": row.provider or "unknown",
                "cost": cost,
                "prompt_tokens": row.prompt_tokens or 0,
                "completion_tokens": row.completion_tokens or 0,
                "total_tokens": (row.prompt_tokens or 0) + (row.completion_tokens or 0),
                "calls": row.calls
            })
        
        for model in models:
            model["percentage"] = (model["cost"] / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "models": sorted(models, key=lambda x: x["cost"], reverse=True)
        }
    except Exception as e:
        logger.error(f"Failed to get costs by model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-agent", response_model=Dict[str, Any])
async def get_costs_by_agent(
    days: int = Query(30, ge=1, le=365),
    include_mock: Optional[bool] = Query(None, description="Include costs from mock campaigns. Defaults to config."),
    db: AsyncSession = Depends(get_db)
):
    """
    Get cost breakdown by agent type (content_generator, safety_validator, etc.).
    """
    try:
        if include_mock is None:
            include_mock = await _get_include_mock_config(db)

        since_date = datetime.utcnow() - timedelta(days=days)
        
        conditions = [CostTracking.timestamp >= since_date]
        if not include_mock:
            mock_ids = select(Campaign.id).where(Campaign.is_mock == True).scalar_subquery()
            conditions.append(~CostTracking.campaign_id.in_(mock_ids) | CostTracking.campaign_id.is_(None))

        result = await db.execute(
            select(
                CostTracking.agent_type,
                func.sum(CostTracking.cost_amount).label('cost'),
                func.sum(CostTracking.tokens_prompt + CostTracking.tokens_completion).label('tokens'),
                func.count(CostTracking.id).label('calls')
            ).where(and_(*conditions))
            .group_by(CostTracking.agent_type)
        )
        
        agents = []
        total_cost = 0
        
        for row in result:
            cost = row.cost or 0
            total_cost += cost
            agents.append({
                "agent_type": row.agent_type or "unknown",
                "cost": cost,
                "tokens": row.tokens or 0,
                "calls": row.calls
            })
        
        for agent in agents:
            agent["percentage"] = (agent["cost"] / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "agents": sorted(agents, key=lambda x: x["cost"], reverse=True)
        }
    except Exception as e:
        logger.error(f"Failed to get costs by agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily", response_model=Dict[str, Any])
async def get_daily_costs(
    days: int = Query(30, ge=1, le=365),
    include_mock: Optional[bool] = Query(None, description="Include costs from mock campaigns. Defaults to config."),
    db: AsyncSession = Depends(get_db)
):
    """
    Get daily cost time series for the specified period.
    """
    try:
        if include_mock is None:
            include_mock = await _get_include_mock_config(db)

        since_date = datetime.utcnow() - timedelta(days=days)
        
        conditions = [CostTracking.timestamp >= since_date]
        if not include_mock:
            mock_ids = select(Campaign.id).where(Campaign.is_mock == True).scalar_subquery()
            conditions.append(~CostTracking.campaign_id.in_(mock_ids) | CostTracking.campaign_id.is_(None))

        result = await db.execute(
            select(
                func.date(CostTracking.timestamp).label('date'),
                func.sum(CostTracking.cost_amount).label('cost'),
                func.count(CostTracking.id).label('calls')
            ).where(and_(*conditions))
            .group_by(func.date(CostTracking.timestamp))
            .order_by(func.date(CostTracking.timestamp))
        )
        
        daily_data = []
        total_cost = 0
        
        for row in result:
            cost = row.cost or 0
            total_cost += cost
            daily_data.append({
                "date": row.date.isoformat() if row.date else None,
                "cost": cost,
                "calls": row.calls
            })
        
        all_dates = []
        current_date = since_date.date()
        end_date = datetime.utcnow().date()
        
        existing_dates = {d["date"] for d in daily_data}
        
        while current_date <= end_date:
            date_str = current_date.isoformat()
            if date_str not in existing_dates:
                all_dates.append({
                    "date": date_str,
                    "cost": 0,
                    "calls": 0
                })
            else:
                all_dates.append(next(d for d in daily_data if d["date"] == date_str))
            current_date += timedelta(days=1)
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "daily": all_dates,
            "avg_daily_cost": total_cost / days if days > 0 else 0
        }
    except Exception as e:
        logger.error(f"Failed to get daily costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast", response_model=Dict[str, Any])
async def get_cost_forecast(
    days: int = Query(30, ge=1, le=90, description="Days to forecast"),
    db: AsyncSession = Depends(get_db)
):
    """
    Forecast future costs based on historical burn rate.
    """
    try:
        since_date = datetime.utcnow() - timedelta(days=30)
        
        result = await db.execute(
            select(
                func.sum(CostTracking.cost_amount).label('total'),
                func.count(func.distinct(func.date(CostTracking.timestamp))).label('active_days')
            ).where(CostTracking.timestamp >= since_date)
        )
        row = result.first()
        
        total_past_cost = row.total or 0
        active_days = row.active_days or 1
        
        daily_burn_rate = total_past_cost / active_days if active_days > 0 else 0
        
        forecast = []
        cumulative = 0
        
        for i in range(days):
            forecast_date = (datetime.utcnow() + timedelta(days=i+1)).date()
            cumulative += daily_burn_rate
            forecast.append({
                "date": forecast_date.isoformat(),
                "daily_cost": daily_burn_rate,
                "cumulative_cost": cumulative
            })
        
        return {
            "forecast_days": days,
            "daily_burn_rate": daily_burn_rate,
            "weekly_forecast": daily_burn_rate * 7,
            "monthly_forecast": daily_burn_rate * 30,
            "total_forecast": cumulative,
            "forecast": forecast,
            "based_on_days": active_days,
            "historical_total": total_past_cost
        }
    except Exception as e:
        logger.error(f"Failed to get cost forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache-metrics", response_model=Dict[str, Any])
async def get_semantic_cache_metrics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get semantic cache performance metrics.

    Research Plan Reference: Section 10.2 - "Semantic Cache Hit Rate: Target > 20%"

    Returns cache hits, misses, hit rate, and estimated cost savings.
    """
    try:
        cache = SemanticCache()
        await cache.initialize()
        cache_stats = await cache.get_stats()

        hits = 0
        misses = 0
        try:
            r = redis.from_url(settings.REDIS_URL)
            hits = int(r.get("semantic_cache:hits") or 0)
            misses = int(r.get("semantic_cache:misses") or 0)
        except Exception as redis_error:
            logger.warning(f"Could not get Redis cache counters: {redis_error}")

        total_queries = hits + misses
        hit_rate = (hits / total_queries * 100) if total_queries > 0 else 0.0

        # $0.03 per call estimate
        avg_cost_per_call = 0.03
        estimated_savings = hits * avg_cost_per_call

        avg_cached_latency_ms = 50  # Default estimate for cache hit
        avg_uncached_latency_ms = 2000  # Default estimate for API call

        try:
            r = redis.from_url(settings.REDIS_URL)
            cached_latency = r.get("semantic_cache:avg_cached_latency")
            uncached_latency = r.get("semantic_cache:avg_uncached_latency")
            if cached_latency:
                avg_cached_latency_ms = float(cached_latency)
            if uncached_latency:
                avg_uncached_latency_ms = float(uncached_latency)
        except Exception:
            pass

        time_saved_seconds = hits * (avg_uncached_latency_ms - avg_cached_latency_ms) / 1000

        target_hit_rate = 20.0
        meets_target = hit_rate >= target_hit_rate

        return {
            "enabled": cache_stats.get("enabled", False),
            "total_entries": cache_stats.get("total_entries", 0),
            "similarity_threshold": cache_stats.get("similarity_threshold", 0.95),
            "ttl_hours": cache_stats.get("ttl_hours", 24),
            "cache_hits": hits,
            "cache_misses": misses,
            "total_queries": total_queries,
            "hit_rate": round(hit_rate, 2),
            "hit_rate_target": target_hit_rate,
            "meets_target": meets_target,
            "estimated_cost_savings": round(estimated_savings, 2),
            "avg_cached_latency_ms": avg_cached_latency_ms,
            "avg_uncached_latency_ms": avg_uncached_latency_ms,
            "time_saved_seconds": round(time_saved_seconds, 1),
            "status": "healthy" if meets_target else "below_target",
            "recommendation": None if meets_target else "Consider lowering similarity threshold or increasing TTL to improve hit rate"
        }
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        return {
            "enabled": False,
            "error": str(e),
            "hit_rate": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "meets_target": False
        }


@router.get("/by-campaign", response_model=Dict[str, Any])
async def get_costs_by_campaign(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """
    Get cost breakdown by campaign.

    Research Plan Reference: Section 10.2 - "Cost-Per-Campaign tracking"
    """
    try:
        since_date = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(
                CostTracking.campaign_id,
                func.sum(CostTracking.cost_amount).label('cost'),
                func.sum(CostTracking.tokens_prompt).label('prompt_tokens'),
                func.sum(CostTracking.tokens_completion).label('completion_tokens'),
                func.count(CostTracking.id).label('calls')
            ).where(
                and_(
                    CostTracking.timestamp >= since_date,
                    CostTracking.campaign_id.isnot(None)
                )
            ).group_by(CostTracking.campaign_id)
        )

        campaigns = []
        total_cost = 0

        for row in result:
            cost = row.cost or 0
            total_cost += cost

            campaign_name = "Unknown"
            if row.campaign_id:
                campaign_result = await db.execute(
                    select(Campaign.name).where(Campaign.id == row.campaign_id)
                )
                campaign_row = campaign_result.first()
                if campaign_row:
                    campaign_name = campaign_row.name

            campaigns.append({
                "campaign_id": str(row.campaign_id) if row.campaign_id else None,
                "campaign_name": campaign_name,
                "cost": cost,
                "prompt_tokens": row.prompt_tokens or 0,
                "completion_tokens": row.completion_tokens or 0,
                "total_tokens": (row.prompt_tokens or 0) + (row.completion_tokens or 0),
                "calls": row.calls
            })

        for campaign in campaigns:
            campaign["percentage"] = (campaign["cost"] / total_cost * 100) if total_cost > 0 else 0

        return {
            "period_days": days,
            "total_cost": total_cost,
            "campaigns": sorted(campaigns, key=lambda x: x["cost"], reverse=True),
            "campaign_count": len(campaigns)
        }
    except Exception as e:
        logger.error(f"Failed to get costs by campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
