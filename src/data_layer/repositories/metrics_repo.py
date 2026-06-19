"""
Metrics Repository - Real-time and historical metrics tracking
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.orm import joinedload
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import logging
import numpy as np
from collections import defaultdict

from ..database.models import (
    Metric, Campaign, Content, Experiment, 
    Platform, CampaignStatus
)
from ...config.settings import settings

logger = logging.getLogger(__name__)

class MetricsRepository:
    """Repository for metrics tracking and analytics"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def record_metric(self, metric_data: Dict[str, Any]) -> Metric:
        """
        Record a new metric point
        
        Args:
            metric_data: Dictionary with metric fields
        
        Returns:
            Created Metric instance
        """
        try:
            metric = Metric(**metric_data)
            self.session.add(metric)
            await self.session.commit()
            await self.session.refresh(metric)
            
            return metric
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to record metric: {e}")
            raise
    
    async def get_campaign_timeseries(
        self,
        campaign_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "hour"
    ) -> List[Dict[str, Any]]:
        """
        Get time-series metrics for a campaign
        
        Args:
            campaign_id: Campaign UUID
            start_date: Start datetime
            end_date: End datetime
            interval: Time interval ('hour', 'day', 'week')
        
        Returns:
            List of time-series data points
        """
        try:
            # Determine date_trunc format
            trunc_format = {
                "hour": "hour",
                "day": "day",
                "week": "week"
            }.get(interval, "hour")
            
            stmt = (
                select(
                    func.date_trunc(trunc_format, Metric.timestamp).label('period'),
                    func.sum(Metric.impressions).label('impressions'),
                    func.sum(Metric.clicks).label('clicks'),
                    func.sum(Metric.conversions).label('conversions'),
                    func.sum(Metric.spend).label('spend')
                )
                .where(
                    and_(
                        Metric.campaign_id == UUID(campaign_id),
                        Metric.timestamp >= start_date,
                        Metric.timestamp <= end_date
                    )
                )
                .group_by('period')
                .order_by(asc('period'))
            )
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            # Format results
            timeseries = []
            for row in rows:
                impressions = row.impressions or 0
                clicks = row.clicks or 0
                conversions = row.conversions or 0
                spend = float(row.spend or 0.0)
                
                ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
                cvr = (conversions / clicks * 100) if clicks > 0 else 0.0
                cpl = (spend / conversions) if conversions > 0 else 0.0
                
                timeseries.append({
                    "timestamp": row.period.isoformat(),
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": spend,
                    "ctr": round(ctr, 2),
                    "conversion_rate": round(cvr, 2),
                    "cpl": round(cpl, 2)
                })
            
            return timeseries
        except Exception as e:
            logger.error(f"Failed to get timeseries: {e}")
            return []
    
    async def get_platform_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[Platform, Dict[str, Any]]:
        """Get aggregated metrics by platform"""
        try:
            filters = []
            if start_date:
                filters.append(Metric.timestamp >= start_date)
            if end_date:
                filters.append(Metric.timestamp <= end_date)
            
            stmt = (
                select(
                    Campaign.platform,
                    func.count(func.distinct(Metric.campaign_id)).label('campaigns'),
                    func.sum(Metric.impressions).label('impressions'),
                    func.sum(Metric.clicks).label('clicks'),
                    func.sum(Metric.conversions).label('conversions'),
                    func.sum(Metric.spend).label('spend')
                )
                .join(Campaign, Metric.campaign_id == Campaign.id)
                .where(and_(*filters) if filters else True)
                .group_by(Campaign.platform)
            )
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            summary = {}
            for row in rows:
                impressions = row.impressions or 0
                clicks = row.clicks or 0
                conversions = row.conversions or 0
                spend = float(row.spend or 0.0)
                
                ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
                cpl = (spend / conversions) if conversions > 0 else 0.0
                
                summary[row.platform] = {
                    "campaigns": row.campaigns,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": spend,
                    "ctr": round(ctr, 2),
                    "cpl": round(cpl, 2)
                }
            
            return summary
        except Exception as e:
            logger.error(f"Failed to get platform summary: {e}")
            return {}
    
    async def get_persona_breakdown(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get metrics breakdown by target persona"""
        try:
            filters = []
            if start_date:
                filters.append(Metric.timestamp >= start_date)
            if end_date:
                filters.append(Metric.timestamp <= end_date)
            
            stmt = (
                select(
                    Campaign.target_persona,
                    func.count(func.distinct(Metric.campaign_id)).label('campaigns'),
                    func.sum(Metric.impressions).label('impressions'),
                    func.sum(Metric.clicks).label('clicks'),
                    func.sum(Metric.conversions).label('conversions'),
                    func.sum(Metric.spend).label('spend')
                )
                .join(Campaign, Metric.campaign_id == Campaign.id)
                .where(and_(*filters) if filters else True)
                .group_by(Campaign.target_persona)
            )
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            breakdown = {}
            for row in rows:
                impressions = row.impressions or 0
                clicks = row.clicks or 0
                conversions = row.conversions or 0
                spend = float(row.spend or 0.0)
                
                ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
                cvr = (conversions / clicks * 100) if clicks > 0 else 0.0
                cpl = (spend / conversions) if conversions > 0 else 0.0
                
                breakdown[row.target_persona] = {
                    "campaigns": row.campaigns,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": spend,
                    "ctr": round(ctr, 2),
                    "conversion_rate": round(cvr, 2),
                    "cpl": round(cpl, 2)
                }
            
            return breakdown
        except Exception as e:
            logger.error(f"Failed to get persona breakdown: {e}")
            return {}
    
    async def get_funnel_metrics(
        self,
        campaign_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get conversion funnel metrics"""
        try:
            filters = []
            if campaign_id:
                filters.append(Metric.campaign_id == UUID(campaign_id))
            if start_date:
                filters.append(Metric.timestamp >= start_date)
            if end_date:
                filters.append(Metric.timestamp <= end_date)
            
            stmt = (
                select(
                    func.sum(Metric.impressions).label('impressions'),
                    func.sum(Metric.clicks).label('clicks'),
                    func.sum(Metric.conversions).label('leads'),
                    func.sum(Metric.demos_booked).label('demos'),
                    func.sum(Metric.customers).label('customers')
                )
                .where(and_(*filters) if filters else True)
            )
            
            result = await self.session.execute(stmt)
            row = result.first()
            
            if not row:
                return {"stages": []}
            
            impressions = row.impressions or 0
            clicks = row.clicks or 0
            leads = row.leads or 0
            demos = row.demos or 0
            customers = row.customers or 0
            
            # Calculate funnel stages with conversion rates
            stages = [
                {
                    "stage": "Impressions",
                    "count": impressions,
                    "conversion_rate": 100.0,
                    "drop_off": 0
                },
                {
                    "stage": "Clicks",
                    "count": clicks,
                    "conversion_rate": (clicks / impressions * 100) if impressions > 0 else 0.0,
                    "drop_off": impressions - clicks
                },
                {
                    "stage": "Leads",
                    "count": leads,
                    "conversion_rate": (leads / clicks * 100) if clicks > 0 else 0.0,
                    "drop_off": clicks - leads
                },
                {
                    "stage": "Demos Booked",
                    "count": demos,
                    "conversion_rate": (demos / leads * 100) if leads > 0 else 0.0,
                    "drop_off": leads - demos
                },
                {
                    "stage": "Customers",
                    "count": customers,
                    "conversion_rate": (customers / demos * 100) if demos > 0 else 0.0,
                    "drop_off": demos - customers
                }
            ]
            
            return {
                "stages": stages,
                "overall_conversion": (customers / impressions * 100) if impressions > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get funnel metrics: {e}")
            return {"stages": []}
    
    async def calculate_roi(
        self,
        campaign_id: str,
        customer_ltv: float = 5000.0
    ) -> Dict[str, float]:
        """Calculate ROI for a campaign"""
        try:
            stmt = (
                select(
                    func.sum(Metric.spend).label('total_spend'),
                    func.sum(Metric.conversions).label('conversions'),
                    func.sum(Metric.customers).label('customers')
                )
                .where(Metric.campaign_id == UUID(campaign_id))
            )
            
            result = await self.session.execute(stmt)
            row = result.first()
            
            if not row:
                return {"roi": 0.0, "roas": 0.0}
            
            total_spend = float(row.total_spend or 0.0)
            customers = row.customers or 0
            
            revenue = customers * customer_ltv
            roi = ((revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0.0
            roas = (revenue / total_spend) if total_spend > 0 else 0.0
            
            return {
                "total_spend": total_spend,
                "revenue": revenue,
                "roi": round(roi, 2),
                "roas": round(roas, 2),
                "customers": customers
            }
        except Exception as e:
            logger.error(f"Failed to calculate ROI: {e}")
            return {"roi": 0.0, "roas": 0.0}
    
    async def get_cohort_analysis(
        self,
        cohort_by: str = "week",
        metric: str = "retention"
    ) -> List[Dict[str, Any]]:
        """Perform cohort analysis"""
        try:
            # Group campaigns by cohort (week/month they started)
            trunc_format = "week" if cohort_by == "week" else "month"
            
            stmt = (
                select(
                    func.date_trunc(trunc_format, Campaign.created_at).label('cohort'),
                    func.count(Campaign.id).label('campaigns'),
                    func.avg(
                        func.cast(Campaign.conversions, Float) / 
                        func.nullif(func.cast(Campaign.impressions, Float), 0) * 100
                    ).label('avg_cvr')
                )
                .where(Campaign.status.in_([CampaignStatus.RUNNING, CampaignStatus.COMPLETED]))
                .group_by('cohort')
                .order_by(asc('cohort'))
            )
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            cohorts = []
            for row in rows:
                cohorts.append({
                    "cohort": row.cohort.isoformat(),
                    "campaigns": row.campaigns,
                    "avg_conversion_rate": round(float(row.avg_cvr or 0.0), 2)
                })
            
            return cohorts
        except Exception as e:
            logger.error(f"Failed to perform cohort analysis: {e}")
            return []
    
    async def get_realtime_metrics(
        self,
        last_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get real-time metrics from last N minutes"""
        try:
            since_time = datetime.utcnow() - timedelta(minutes=last_minutes)
            
            stmt = (
                select(
                    func.count(func.distinct(Metric.campaign_id)).label('active_campaigns'),
                    func.sum(Metric.impressions).label('impressions'),
                    func.sum(Metric.clicks).label('clicks'),
                    func.sum(Metric.conversions).label('conversions'),
                    func.sum(Metric.spend).label('spend')
                )
                .where(Metric.timestamp >= since_time)
            )
            
            result = await self.session.execute(stmt)
            row = result.first()
            
            if not row:
                return {
                    "active_campaigns": 0,
                    "impressions": 0,
                    "clicks": 0,
                    "conversions": 0,
                    "spend": 0.0,
                    "ctr": 0.0
                }
            
            impressions = row.impressions or 0
            clicks = row.clicks or 0
            conversions = row.conversions or 0
            spend = float(row.spend or 0.0)
            
            ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
            
            return {
                "active_campaigns": row.active_campaigns,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": spend,
                "ctr": round(ctr, 2),
                "period_minutes": last_minutes
            }
        except Exception as e:
            logger.error(f"Failed to get realtime metrics: {e}")
            return {
                "active_campaigns": 0,
                "impressions": 0,
                "clicks": 0,
                "conversions": 0,
                "spend": 0.0,
                "ctr": 0.0
            }
    
    async def get_top_campaigns(
        self,
        metric: str = "conversions",
        limit: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get top performing campaigns by metric"""
        try:
            filters = []
            if start_date:
                filters.append(Metric.timestamp >= start_date)
            if end_date:
                filters.append(Metric.timestamp <= end_date)
            
            # Metric aggregation
            metric_column = {
                "conversions": func.sum(Metric.conversions),
                "clicks": func.sum(Metric.clicks),
                "impressions": func.sum(Metric.impressions),
                "spend": func.sum(Metric.spend)
            }.get(metric, func.sum(Metric.conversions))
            
            stmt = (
                select(
                    Campaign.id,
                    Campaign.name,
                    Campaign.platform,
                    func.sum(Metric.impressions).label('impressions'),
                    func.sum(Metric.clicks).label('clicks'),
                    func.sum(Metric.conversions).label('conversions'),
                    func.sum(Metric.spend).label('spend')
                )
                .join(Campaign, Metric.campaign_id == Campaign.id)
                .where(and_(*filters) if filters else True)
                .group_by(Campaign.id, Campaign.name, Campaign.platform)
                .order_by(desc(metric_column))
                .limit(limit)
            )
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            campaigns = []
            for row in rows:
                impressions = row.impressions or 0
                clicks = row.clicks or 0
                conversions = row.conversions or 0
                spend = float(row.spend or 0.0)
                
                ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
                cpl = (spend / conversions) if conversions > 0 else 0.0
                
                campaigns.append({
                    "id": str(row.id),
                    "name": row.name,
                    "platform": row.platform.value,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": spend,
                    "ctr": round(ctr, 2),
                    "cpl": round(cpl, 2)
                })
            
            return campaigns
        except Exception as e:
            logger.error(f"Failed to get top campaigns: {e}")
            return []
    
    async def get_hourly_patterns(
        self,
        campaign_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[int, Dict[str, float]]:
        """Analyze performance by hour of day"""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            filters = [Metric.timestamp >= since_date]
            if campaign_id:
                filters.append(Metric.campaign_id == UUID(campaign_id))
            
            stmt = (
                select(
                    func.extract('hour', Metric.timestamp).label('hour'),
                    func.avg(
                        func.cast(Metric.clicks, Float) /
                        func.nullif(func.cast(Metric.impressions, Float), 0) * 100
                    ).label('avg_ctr'),
                    func.avg(
                        func.cast(Metric.conversions, Float) /
                        func.nullif(func.cast(Metric.clicks, Float), 0) * 100
                    ).label('avg_cvr')
                )
                .where(and_(*filters))
                .group_by('hour')
                .order_by('hour')
            )
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            patterns = {}
            for row in rows:
                hour = int(row.hour)
                patterns[hour] = {
                    "ctr": round(float(row.avg_ctr or 0.0), 2),
                    "conversion_rate": round(float(row.avg_cvr or 0.0), 2)
                }
            
            return patterns
        except Exception as e:
            logger.error(f"Failed to get hourly patterns: {e}")
            return {}