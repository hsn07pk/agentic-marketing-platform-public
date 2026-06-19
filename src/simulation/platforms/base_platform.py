import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class BasePlatform(ABC):
    """Base class for all platform simulations."""
    
    def __init__(
        self,
        name: str,
        base_cpm: float = 10.0,
        base_cpc: float = 1.0,
        auction_mechanism: str = "second_price"
    ):
        self.name = name
        self.base_cpm = base_cpm
        self.base_cpc = base_cpc
        self.auction_mechanism = auction_mechanism
        self.active_campaigns = {}
        self.historical_performance = []
    
    def run_ad_auction(
        self,
        campaigns: List[Dict[str, Any]],
        available_slots: int = 1
    ) -> List[Dict[str, Any]]:
        """Run ad auction to determine winning campaigns with pricing."""
        if not campaigns:
            return []
        
        scored_campaigns = []
        for campaign in campaigns:
            bid = campaign.get('bid', 0.0)
            quality_score = campaign.get('quality_score', 0.5)
            
            effective_bid = bid * quality_score
            
            scored_campaigns.append({
                'campaign_id': campaign.get('campaign_id'),
                'bid': bid,
                'quality_score': quality_score,
                'effective_bid': effective_bid,
                'campaign_data': campaign
            })
        
        scored_campaigns.sort(key=lambda x: x['effective_bid'], reverse=True)
        
        winners = []
        for i in range(min(available_slots, len(scored_campaigns))):
            winner = scored_campaigns[i]
            
            if self.auction_mechanism == "second_price" and i < len(scored_campaigns) - 1:
                second_price = scored_campaigns[i + 1]['effective_bid']
                actual_cpc = second_price / winner['quality_score'] + 0.01
            else:
                actual_cpc = winner['bid']
            
            winners.append({
                'campaign_id': winner['campaign_id'],
                'position': i + 1,
                'actual_cpc': actual_cpc,
                'quality_score': winner['quality_score'],
                'campaign_data': winner['campaign_data']
            })
        
        return winners
    
    def calculate_organic_reach(
        self,
        content_quality: float,
        follower_count: int,
        engagement_history: float
    ) -> int:
        """Calculate organic reach for non-paid content."""
        base_reach = follower_count * 0.1
        
        quality_multiplier = 1.0 + (content_quality - 0.5) * 2
        engagement_multiplier = 1.0 + engagement_history
        
        viral_factor = 1.0
        if random.random() < content_quality * 0.1:
            viral_factor = random.uniform(1.5, 3.0)
        
        reach = int(
            base_reach * 
            quality_multiplier * 
            engagement_multiplier * 
            viral_factor
        )
        
        return max(reach, int(follower_count * 0.05))
    
    def simulate_engagement(
        self,
        impressions: int,
        content_quality: float,
        target_persona_match: float
    ) -> Dict[str, int]:
        """Simulate user engagement with content."""
        base_ctr = 0.02
        ctr = base_ctr * content_quality * target_persona_match
        ctr = max(0.001, min(0.1, ctr))
        
        clicks = sum(1 for _ in range(impressions) if random.random() < ctr)
        
        like_rate = 0.3 * content_quality
        likes = sum(1 for _ in range(impressions) if random.random() < like_rate)
        
        share_rate = 0.05 * content_quality * target_persona_match
        shares = sum(1 for _ in range(impressions) if random.random() < share_rate)
        
        comment_rate = 0.02 * content_quality
        comments = sum(1 for _ in range(impressions) if random.random() < comment_rate)
        
        return {
            'impressions': impressions,
            'clicks': clicks,
            'likes': likes,
            'shares': shares,
            'comments': comments,
            'ctr': clicks / impressions if impressions > 0 else 0.0,
            'engagement_rate': (likes + shares + comments) / impressions if impressions > 0 else 0.0
        }
    
    def apply_ad_fatigue(
        self,
        base_performance: Dict[str, float],
        exposure_count: int
    ) -> Dict[str, float]:
        """Apply ad fatigue effects based on exposure count."""
        if exposure_count <= 3:
            fatigue_factor = 1.0
        elif exposure_count <= 10:
            fatigue_factor = 1.0 - ((exposure_count - 3) * 0.1)
        else:
            fatigue_factor = 0.3
        
        return {
            key: value * fatigue_factor
            for key, value in base_performance.items()
        }
    
    def calculate_quality_score(
        self,
        relevance: float,
        expected_ctr: float,
        landing_page_quality: float
    ) -> float:
        """Calculate quality score (0-1) for ad."""
        weights = {
            'relevance': 0.4,
            'ctr': 0.4,
            'landing_page': 0.2
        }
        
        normalized_ctr = min(expected_ctr / 0.05, 1.0)
        
        quality_score = (
            relevance * weights['relevance'] +
            normalized_ctr * weights['ctr'] +
            landing_page_quality * weights['landing_page']
        )
        
        return max(0.1, min(1.0, quality_score))
    
    def record_performance(
        self,
        campaign_id: str,
        metrics: Dict[str, Any]
    ):
        self.historical_performance.append({
            'campaign_id': campaign_id,
            'timestamp': datetime.utcnow(),
            'metrics': metrics
        })
        
        if campaign_id in self.active_campaigns:
            if 'total_impressions' not in self.active_campaigns[campaign_id]:
                self.active_campaigns[campaign_id]['total_impressions'] = 0
                self.active_campaigns[campaign_id]['total_clicks'] = 0
            
            self.active_campaigns[campaign_id]['total_impressions'] += metrics.get('impressions', 0)
            self.active_campaigns[campaign_id]['total_clicks'] += metrics.get('clicks', 0)
    
    def get_campaign_stats(self, campaign_id: str) -> Dict[str, Any]:
        """Get aggregated campaign statistics."""
        if campaign_id not in self.active_campaigns:
            return {
                'campaign_id': campaign_id,
                'total_impressions': 0,
                'total_clicks': 0,
                'ctr': 0.0
            }

        stats = self.active_campaigns[campaign_id]
        total_impressions = stats.get('total_impressions', 0)
        total_clicks = stats.get('total_clicks', 0)

        return {
            'campaign_id': campaign_id,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'ctr': total_clicks / total_impressions if total_impressions > 0 else 0.0
        }

    def get_campaign_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all active campaigns."""
        total_impressions = 0
        total_clicks = 0
        total_conversions = 0

        for campaign_id, campaign_data in self.active_campaigns.items():
            total_impressions += campaign_data.get('impressions', 0)
            total_clicks += campaign_data.get('clicks', 0)
            total_conversions += campaign_data.get('conversions', 0)

        return {
            'impressions': total_impressions,
            'clicks': total_clicks,
            'conversions': total_conversions
        }
    
    def run_campaign(self, campaign_config: Dict[str, Any]):
        """Run a campaign simulation (SimPy generator process)."""
        import simpy

        campaign_id = campaign_config.get('campaign_id', f"campaign_{len(self.active_campaigns)}")
        duration_hours = campaign_config.get('duration', 7) * 24
        budget = campaign_config.get('budget', 5000)
        daily_budget = campaign_config.get('daily_budget', budget / (duration_hours / 24))

        # Initialize campaign tracking
        campaign_data = {
            'id': campaign_id,
            'config': campaign_config,
            'start_time': self.env.now if hasattr(self, 'env') else 0,
            'duration': duration_hours,
            'budget': budget,
            'daily_budget': daily_budget,
            'spent': 0.0,
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'status': 'active'
        }

        self.active_campaigns[campaign_id] = campaign_data

        logger.info(f"Started campaign {campaign_id} on {self.name} platform")

        for hour in range(duration_hours):
            if campaign_data['spent'] >= budget:
                logger.info(f"Campaign {campaign_id} reached budget limit")
                break

            # Calculate hourly budget
            hourly_budget = daily_budget / 24
            remaining_budget = min(
                hourly_budget,
                budget - campaign_data['spent']
            )

            if remaining_budget <= 0:
                yield self.env.timeout(1) if hasattr(self, 'env') else None
                continue

            hourly_impressions = int(remaining_budget / self.base_cpm * 1000) if self.base_cpm > 0 else int(remaining_budget * 100)

            engagement = self.simulate_engagement(
                impressions=hourly_impressions,
                content_quality=campaign_config.get('content_quality', 0.7),
                target_persona_match=campaign_config.get('persona_match', 0.8)
            )

            # Update campaign metrics
            campaign_data['impressions'] += engagement['impressions']
            campaign_data['clicks'] += engagement['clicks']

            cost = engagement['impressions'] / 1000 * self.base_cpm
            campaign_data['spent'] += cost

            conversion_rate = campaign_config.get('conversion_rate', 0.02)
            conversions = sum(1 for _ in range(engagement['clicks'])
                            if random.random() < conversion_rate)
            campaign_data['conversions'] += conversions

            self.record_performance(campaign_id, {
                'hour': hour,
                'impressions': engagement['impressions'],
                'clicks': engagement['clicks'],
                'conversions': conversions,
                'cost': cost
            })

            # Wait for next hour
            yield self.env.timeout(1) if hasattr(self, 'env') else None

        campaign_data['status'] = 'completed'

        logger.info(
            f"Campaign {campaign_id} completed. "
            f"Impressions: {campaign_data['impressions']}, "
            f"Clicks: {campaign_data['clicks']}, "
            f"Conversions: {campaign_data['conversions']}, "
            f"Spent: €{campaign_data['spent']:.2f}"
        )

    @abstractmethod
    def get_platform_specific_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement get_platform_specific_metrics")
    
    @abstractmethod
    def simulate_user_behavior(
        self,
        content: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError("Subclass must implement simulate_user_behavior")