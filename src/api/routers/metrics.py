from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import numpy as np

from ...data_layer.database.models import Campaign, Metric, Experiment, BanditArm, SystemConfiguration
from ..dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_include_mock_config(db: AsyncSession) -> bool:
    """Read INCLUDE_MOCK_IN_METRICS from configuration service. Defaults to True."""
    try:
        result = await db.execute(
            select(SystemConfiguration.value).where(SystemConfiguration.key == "INCLUDE_MOCK_IN_METRICS")
        )
        val = result.scalar_one_or_none()
        if val is not None:
            return str(val).lower() in ('true', '1', 'yes', 'on')
        return True  # Default
    except Exception:
        return True

@router.get("/overview", response_model=Dict[str, Any])
async def get_metrics_overview(
    days: int = Query(default=30),
    include_mock: Optional[bool] = Query(default=None, description="Include mock campaign data. Defaults to INCLUDE_MOCK_IN_METRICS config."),
    db: AsyncSession = Depends(get_db)
):
    try:
        if include_mock is None:
            include_mock = await _get_include_mock_config(db)

        since_date = datetime.utcnow() - timedelta(days=days)
        
        conditions = [Campaign.created_at >= since_date]
        if not include_mock:
            conditions.append(Campaign.is_mock == False)

        campaign_result = await db.execute(
            select(
                func.count(Campaign.id).label("total_campaigns"),
                func.sum(Campaign.impressions).label("total_impressions"),
                func.sum(Campaign.clicks).label("total_clicks"),
                func.sum(Campaign.conversions).label("total_conversions"),
                func.sum(Campaign.budget_spent).label("total_spent")
            ).where(and_(*conditions))
        )
        
        campaign_metrics = campaign_result.first()
        
        total_campaigns = campaign_metrics.total_campaigns or 0
        total_impressions = campaign_metrics.total_impressions or 0
        total_clicks = campaign_metrics.total_clicks or 0
        total_conversions = campaign_metrics.total_conversions or 0
        total_spent = campaign_metrics.total_spent or 0
        
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        avg_cpl = (total_spent / total_conversions) if total_conversions > 0 else 0
        
        # Calculate ROI using standard formula: ((revenue - cost) / cost) * 100
        # Per Research Plan, default conversion value is €100
        conversion_value = 100  # €100 per conversion
        total_revenue = total_conversions * conversion_value
        roi = ((total_revenue - total_spent) / max(total_spent, 1)) * 100 if total_spent > 0 else 0

        return {
            "period_days": days,
            "total_campaigns": total_campaigns,
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "total_spent": total_spent,
            "average_ctr": avg_ctr,  # Already in percentage (0-100 scale)
            "average_cpl": avg_cpl,
            "roi": roi,
            "includes_mock_data": include_mock
        }
    except Exception as e:
        logger.error(f"Failed to get metrics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/time-series", response_model=List[Dict[str, Any]])
async def get_time_series_metrics(
    metric_name: str = Query(default="impressions"),
    days: int = Query(default=30),
    granularity: str = Query(default="daily"),  # daily, hourly
    db: AsyncSession = Depends(get_db)
):
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        
        result = await db.execute(
            select(Metric)
            .where(
                and_(
                    Metric.metric_name == metric_name,
                    Metric.timestamp >= since_date
                )
            )
            .order_by(Metric.timestamp)
        )
        
        metrics = result.scalars().all()
        
        time_series = []
        for metric in metrics:
            time_series.append({
                "timestamp": metric.timestamp.isoformat(),
                "value": metric.metric_value,
                "platform": metric.platform,
                "tags": metric.tags
            })
        
        return time_series
    except Exception as e:
        logger.error(f"Failed to get time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments", response_model=List[Dict[str, Any]])
async def get_experiment_metrics(
    active_only: bool = Query(default=True),
    db: AsyncSession = Depends(get_db)
):
    try:
        query = select(Experiment)
        
        if active_only:
            query = query.where(Experiment.is_active == True)
        
        result = await db.execute(query)
        experiments = result.scalars().all()
        
        experiment_metrics = []
        for exp in experiments:
            arms_result = await db.execute(
                select(BanditArm).where(BanditArm.experiment_id == exp.id)
            )
            arms = arms_result.scalars().all()
            
            total_pulls = sum(arm.pulls for arm in arms)
            
            if arms:
                best_arm = max(arms, key=lambda a: a.total_reward / max(a.pulls, 1))
                
                experiment_metrics.append({
                    "experiment_id": str(exp.id),
                    "name": exp.name,
                    "type": exp.type,
                    "is_active": exp.is_active,
                    "total_pulls": total_pulls,
                    "total_conversions": exp.total_conversions,
                    "best_arm": best_arm.arm_id,
                    "best_arm_ctr": (best_arm.successes / max(best_arm.pulls, 1)) * 100,
                    "variants": [
                        {
                            "arm_id": arm.arm_id,
                            "pulls": arm.pulls,
                            "successes": arm.successes,
                            "ctr": (arm.successes / max(arm.pulls, 1)) * 100,
                            "confidence": self._calculate_confidence(arm)
                        }
                        for arm in arms
                    ]
                })
        
        return experiment_metrics
    except Exception as e:
        logger.error(f"Failed to get experiment metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _calculate_confidence(arm: BanditArm) -> float:
    if arm.pulls < 10:
        return 0.0
    
    # Simple confidence based on number of pulls and success rate
    success_rate = arm.successes / arm.pulls
    confidence = min(1.0, (arm.pulls / 100) * success_rate)
    
    return confidence