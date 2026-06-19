"""
Campaign Repository - Data access layer for campaigns
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.orm import joinedload
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import logging

from ..database.models import Campaign, Content, CampaignStatus, Platform
from ...config.settings import settings

logger = logging.getLogger(__name__)

class CampaignRepository:

    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, campaign_data: Dict[str, Any]) -> Campaign:
        """Create new campaign."""
        try:
            campaign = Campaign(**campaign_data)
            self.session.add(campaign)
            await self.session.commit()
            await self.session.refresh(campaign)
            
            logger.info(f"Created campaign: {campaign.id}")
            return campaign
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create campaign: {e}")
            raise
    
    async def get_by_id(self, campaign_id: str) -> Optional[Campaign]:
        try:
            stmt = select(Campaign).where(Campaign.id == UUID(campaign_id))
            result = await self.session.execute(stmt)
            campaign = result.scalar_one_or_none()
            
            return campaign
        except Exception as e:
            logger.error(f"Failed to get campaign {campaign_id}: {e}")
            return None
    
    async def get_with_contents(self, campaign_id: str) -> Optional[Campaign]:
        try:
            stmt = (
                select(Campaign)
                .options(joinedload(Campaign.contents))
                .where(Campaign.id == UUID(campaign_id))
            )
            result = await self.session.execute(stmt)
            campaign = result.unique().scalar_one_or_none()
            
            return campaign
        except Exception as e:
            logger.error(f"Failed to get campaign with contents: {e}")
            return None
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[CampaignStatus] = None,
        platform: Optional[Platform] = None
    ) -> List[Campaign]:
        try:
            stmt = select(Campaign)
            
            if status:
                stmt = stmt.where(Campaign.status == status)
            if platform:
                stmt = stmt.where(Campaign.platform == platform)
            
            stmt = stmt.offset(skip).limit(limit)
            
            stmt = stmt.order_by(Campaign.created_at.desc())
            
            result = await self.session.execute(stmt)
            campaigns = result.scalars().all()
            
            return list(campaigns)
        except Exception as e:
            logger.error(f"Failed to get campaigns: {e}")
            return []
    
    async def list_all(self, limit: int = 100) -> List[Campaign]:
    
        return await self.get_all(limit=limit)

    
    async def get_active(self) -> List[Campaign]:
        return await self.get_all(status=CampaignStatus.RUNNING)
    
    async def get_by_status(
        self,
        status: CampaignStatus,
        limit: Optional[int] = None
    ) -> List[Campaign]:
        return await self.get_all(status=status, limit=limit or 1000)
    
    async def get_by_platform(
        self,
        platform: Platform,
        limit: Optional[int] = None
    ) -> List[Campaign]:
        return await self.get_all(platform=platform, limit=limit or 1000)
    
    async def update(
        self,
        campaign_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Campaign]:
        """Update campaign fields."""
        try:
            stmt = (
                update(Campaign)
                .where(Campaign.id == UUID(campaign_id))
                .values(**updates)
                .returning(Campaign)
            )
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            campaign = result.scalar_one_or_none()
            
            if campaign:
                logger.info(f"Updated campaign: {campaign_id}")
            
            return campaign
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update campaign {campaign_id}: {e}")
            return None
    
    async def update_status(
        self,
        campaign_id: str,
        status: CampaignStatus
    ) -> bool:
        try:
            result = await self.update(campaign_id, {"status": status})
            return result is not None
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False
    
    async def update_metrics(
        self,
        campaign_id: str,
        impressions: int,
        clicks: int,
        conversions: int,
        spend: float
    ) -> bool:
        """Update campaign metrics and calculate derived metrics (CTR, CPL).

        CTR is stored as PERCENTAGE (0-100 scale) for API layer consistency.
        """
        try:
            # Calculate CTR (Click-Through Rate) as PERCENTAGE
            # Per Research Plan Section 10.2: CTR should be displayed consistently
            ctr = (clicks / impressions * 100) if impressions > 0 else 0.0

            cpl = (spend / conversions) if conversions > 0 else 0.0

            updates = {
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "budget_spent": spend,
                "ctr": ctr,  # Stored as percentage (e.g., 1.58 for 1.58%)
                "cpl": cpl,
                "updated_at": datetime.utcnow()
            }

            result = await self.update(campaign_id, updates)

            if result:
                logger.info(
                    "Campaign metrics updated with derived calculations",
                    extra={
                        "campaign_id": campaign_id,
                        "impressions": impressions,
                        "clicks": clicks,
                        "conversions": conversions,
                        "ctr": round(ctr, 2),  # Percentage
                        "cpl": round(cpl, 2),
                        "budget_spent": round(spend, 2)
                    }
                )

            return result is not None
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            return False
    
    async def increment_metrics(
        self,
        campaign_id: str,
        impressions: int = 0,
        clicks: int = 0,
        conversions: int = 0,
        spend: float = 0.0
    ) -> bool:
        try:
            campaign = await self.get_by_id(campaign_id)
            if not campaign:
                return False
            
            new_impressions = campaign.impressions + impressions
            new_clicks = campaign.clicks + clicks
            new_conversions = campaign.conversions + conversions
            new_spend = campaign.spend + spend
            
            return await self.update_metrics(
                campaign_id,
                new_impressions,
                new_clicks,
                new_conversions,
                new_spend
            )
        except Exception as e:
            logger.error(f"Failed to increment metrics: {e}")
            return False
    
    async def delete(self, campaign_id: str) -> bool:
        try:
            stmt = delete(Campaign).where(Campaign.id == UUID(campaign_id))
            await self.session.execute(stmt)
            await self.session.commit()
            
            logger.info(f"Deleted campaign: {campaign_id}")
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete campaign: {e}")
            return False
    
    async def get_campaigns_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Campaign]:
        try:
            stmt = (
                select(Campaign)
                .where(
                    and_(
                        Campaign.created_at >= start_date,
                        Campaign.created_at <= end_date
                    )
                )
                .order_by(Campaign.created_at.desc())
            )
            
            result = await self.session.execute(stmt)
            campaigns = result.scalars().all()
            
            return list(campaigns)
        except Exception as e:
            logger.error(f"Failed to get campaigns by date range: {e}")
            return []
    
    async def get_campaign_performance(
        self,
        campaign_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            campaign = await self.get_by_id(campaign_id)
            if not campaign:
                return None
            
            ctr = (campaign.clicks / campaign.impressions * 100) if campaign.impressions > 0 else 0.0
            conversion_rate = (campaign.conversions / campaign.clicks * 100) if campaign.clicks > 0 else 0.0
            cpl = (campaign.spend / campaign.conversions) if campaign.conversions > 0 else 0.0
            cpc = (campaign.spend / campaign.clicks) if campaign.clicks > 0 else 0.0
            
            return {
                "campaign_id": str(campaign.id),
                "name": campaign.name,
                "status": campaign.status.value,
                "platform": campaign.platform.value,
                "impressions": campaign.impressions,
                "clicks": campaign.clicks,
                "conversions": campaign.conversions,
                "spend": campaign.spend,
                "budget": campaign.budget,
                "budget_used_pct": (campaign.spend / campaign.budget * 100) if campaign.budget > 0 else 0.0,
                "ctr": ctr,
                "conversion_rate": conversion_rate,
                "cpl": cpl,
                "cpc": cpc,
                "roi": ((campaign.conversions * campaign.target_cpl - campaign.spend) / campaign.spend * 100) if campaign.spend > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to calculate performance: {e}")
            return None
    
    async def get_platform_summary(self) -> Dict[Platform, Dict[str, Any]]:
        try:
            stmt = (
                select(
                    Campaign.platform,
                    func.count(Campaign.id).label('count'),
                    func.sum(Campaign.impressions).label('total_impressions'),
                    func.sum(Campaign.clicks).label('total_clicks'),
                    func.sum(Campaign.conversions).label('total_conversions'),
                    func.sum(Campaign.spend).label('total_spend')
                )
                .where(Campaign.status.in_([CampaignStatus.RUNNING, CampaignStatus.COMPLETED]))
                .group_by(Campaign.platform)
            )
            
            result = await self.session.execute(stmt)
            rows = result.all()
            
            summary = {}
            for row in rows:
                summary[row.platform] = {
                    "count": row.count,
                    "impressions": row.total_impressions or 0,
                    "clicks": row.total_clicks or 0,
                    "conversions": row.total_conversions or 0,
                    "spend": float(row.total_spend or 0),
                    "ctr": (row.total_clicks / row.total_impressions * 100) if row.total_impressions else 0.0,
                    "cpl": (float(row.total_spend) / row.total_conversions) if row.total_conversions else 0.0
                }
            
            return summary
        except Exception as e:
            logger.error(f"Failed to get platform summary: {e}")
            return {}
    
    async def check_budget_exceeded(self, campaign_id: str) -> bool:
        try:
            campaign = await self.get_by_id(campaign_id)
            if not campaign:
                return False
            
            return campaign.spend >= campaign.budget
        except Exception as e:
            logger.error(f"Failed to check budget: {e}")
            return False
    
    async def pause_if_budget_exceeded(self, campaign_id: str) -> bool:
        try:
            if await self.check_budget_exceeded(campaign_id):
                await self.update_status(campaign_id, CampaignStatus.PAUSED)
                logger.warning(f"Campaign {campaign_id} paused due to budget exceeded")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to pause campaign: {e}")
            return False

    async def get_recent_decisions_for_ope(
        self,
        limit: int = 1000,
        min_impressions: int = 100,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get recent campaign decisions for Off-Policy Evaluation (MARL gating).

        Returns historical data as (state, action, reward, propensity) tuples
        suitable for OPE analysis.
        """
        try:
            lookback_date = datetime.utcnow() - timedelta(days=lookback_days)

            stmt = (
                select(Campaign)
                .options(joinedload(Campaign.contents))
                .where(
                    and_(
                        Campaign.status == CampaignStatus.COMPLETED,
                        Campaign.impressions >= min_impressions,
                        Campaign.created_at >= lookback_date
                    )
                )
                .order_by(Campaign.created_at.desc())
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            campaigns = result.unique().scalars().all()

            decisions = []
            for campaign in campaigns:
                state = {
                    'persona': campaign.target_persona,
                    'platform': campaign.platform.value if campaign.platform else 'linkedin',
                    'budget': float(campaign.budget_total) if campaign.budget_total else 1000.0,
                    'goal': campaign.goal.value if campaign.goal else 'lead_generation',
                    'industry': campaign.config.get('industry', 'unknown') if campaign.config else 'unknown',
                    'company_size': campaign.config.get('company_size', 'unknown') if campaign.config else 'unknown',
                    'campaign_type': campaign.config.get('type', 'lead_generation') if campaign.config else 'lead_generation'
                }

                action = {
                    'strategy_name': 'default',
                    'hook': 'Transform your business',
                    'cta': 'Learn More',
                    'tone': 'professional',
                    'angle': 'innovation'
                }

                if campaign.metadata and 'strategy' in campaign.metadata:
                    action.update(campaign.metadata['strategy'])
                elif campaign.contents:
                    first_content = campaign.contents[0]
                    if first_content.metadata and 'strategy' in first_content.metadata:
                        action.update(first_content.metadata['strategy'])

                # Calculate CTR and conversion rate as decimals for reward calculation
                ctr_decimal = (campaign.clicks / campaign.impressions) if campaign.impressions > 0 else 0.0
                conversion_rate = (campaign.conversions / campaign.clicks) if campaign.clicks > 0 else 0.0

                # Composite reward: CTR (70%) + Conversion Rate (30%)
                # Uses decimal values for normalized reward calculation
                reward = (ctr_decimal * 0.7) + (conversion_rate * 0.3)

                # Estimate propensity score (logging policy probability)
                # For bandit policies, this would be thompson_sampling probability
                # For now, use uniform distribution as baseline
                propensity = 1.0 / 3.0  # Assuming 3 strategy variants

                if campaign.metadata and 'experiment' in campaign.metadata:
                    exp_data = campaign.metadata['experiment']
                    if 'selection_probability' in exp_data:
                        propensity = exp_data['selection_probability']

                decisions.append({
                    'campaign_id': str(campaign.id),
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'propensity': propensity,
                    'timestamp': campaign.created_at,
                    'impressions': campaign.impressions,
                    'clicks': campaign.clicks,
                    'conversions': campaign.conversions,
                    'ctr': ctr_decimal * 100,  # Return as percentage for consistency
                    'conversion_rate': conversion_rate * 100  # Return as percentage
                })

            logger.info(f"Retrieved {len(decisions)} historical decisions for OPE")
            return decisions

        except Exception as e:
            logger.error(f"Failed to get recent decisions for OPE: {e}", exc_info=True)
            return []
