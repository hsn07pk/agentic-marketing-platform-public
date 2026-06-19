"""X (Twitter) platform simulation."""
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_platform import BasePlatform

logger = logging.getLogger(__name__)

class XPlatform(BasePlatform):
    """Simulates X (Twitter) advertising platform (faster cycles, lower CPC, viral potential)."""

    def __init__(self, env, market_env):
        super().__init__(
            name="x",
            base_cpm=6.50,
            base_cpc=2.50,
            auction_mechanism="second_price"
        )
        self.env = env
        self.market_env = market_env

    def _get_base_cpc(self) -> float:
        return 2.50
    
    def _get_base_cpm(self) -> float:
        return 6.50
    
    def _get_platform_fee(self) -> float:
        return 0.12
    
    def _get_peak_multiplier(self, hour: int) -> float:
        """X has distributed activity; peaks at lunch (12-3pm) and evening (8-10pm)."""
        if 12 <= hour <= 15:
            return 1.4
        elif 20 <= hour <= 22:
            return 1.3
        elif 8 <= hour <= 11 or 16 <= hour <= 19:
            return 1.0
        elif 23 <= hour or hour <= 2:
            return 0.7
        else:
            return 0.8
    
    def _calculate_trending_boost(self, campaign: Dict) -> float:
        """Returns multiplier [1.0-2.0] based on keyword overlap with trending topics."""
        boost = 1.0
        
        campaign_keywords = campaign['config'].get('keywords', [])
        trending_topics = self.market_env.market_state.trending_topics
        
        for keyword in campaign_keywords:
            if any(keyword.lower() in topic.lower() for topic in trending_topics):
                boost += 0.3
        
        return min(boost, 2.0)
    
    def _calculate_our_bid(self, campaign: Dict, customer: Any) -> float:
        base_bid = super()._calculate_our_bid(campaign, customer)
        
        trending_boost = self._calculate_trending_boost(campaign)
        adjusted_bid = base_bid * trending_boost
        
        if customer.persona.name == 'influencer':
            adjusted_bid *= 1.4
        
        hour = int(self.env.now) % 24
        if 20 <= hour <= 22:
            adjusted_bid *= 1.2
        
        return adjusted_bid
    
    def run_promoted_tweet(self, campaign_config: Dict):
        """Run promoted tweet campaign."""
        campaign_id = f"promoted_{len(self.active_campaigns)}"
        
        campaign = {
            'id': campaign_id,
            'type': 'promoted_tweet',
            'config': campaign_config,
            'start_time': self.env.now,
            'duration': campaign_config.get('duration', 7),
            'budget': campaign_config.get('budget', 2000),
            'spent': 0.0,
            'impressions': 0,
            'engagements': 0,
            'retweets': 0,
            'likes': 0,
            'replies': 0,
            'clicks': 0,
            'status': 'active'
        }
        
        self.active_campaigns.append(campaign)
        
        logger.info(f"Started promoted tweet campaign {campaign_id}")
        
        yield self.env.process(self._deliver_promoted_tweet(campaign))
    
    def _deliver_promoted_tweet(self, campaign: Dict):
        duration_hours = campaign['duration'] * 24
        
        for hour in range(int(duration_hours)):
            if campaign['spent'] >= campaign['budget']:
                break
            
            yield self.env.process(self._promote_tweet_hour(campaign))
            
            yield self.env.timeout(1)
        
        campaign['status'] = 'completed'
        logger.info(
            f"Promoted tweet {campaign['id']} completed. "
            f"Impressions: {campaign['impressions']}, "
            f"Engagements: {campaign['engagements']}"
        )
    
    def _promote_tweet_hour(self, campaign: Dict):
        peak_mult = self._get_peak_multiplier(int(self.env.now) % 24)
        trending_mult = self._calculate_trending_boost(campaign)
        
        base_impressions = 200
        impressions = int(base_impressions * peak_mult * trending_mult)
        
        cost = (impressions / 1000) * self._get_base_cpm()
        
        if campaign['spent'] + cost > campaign['budget']:
            available_budget = campaign['budget'] - campaign['spent']
            impressions = int((available_budget / cost) * impressions)
            cost = available_budget
        
        campaign['impressions'] += impressions
        campaign['spent'] += cost
        
        engagement_rate = 0.02
        engagements = int(impressions * engagement_rate)
        
        campaign['engagements'] += engagements
        
        for _ in range(engagements):
            engagement_type = np.random.choice(
                ['like', 'retweet', 'reply', 'click'],
                p=[0.5, 0.2, 0.1, 0.2]
            )
            
            if engagement_type == 'like':
                campaign['likes'] += 1
            elif engagement_type == 'retweet':
                campaign['retweets'] += 1
                campaign['impressions'] += int(np.random.poisson(50))
            elif engagement_type == 'reply':
                campaign['replies'] += 1
            elif engagement_type == 'click':
                campaign['clicks'] += 1
        
        yield self.env.timeout(0)
    
    def get_viral_potential(self, campaign_id: str) -> float:
        """Calculate viral potential [0-1] based on retweet rate and engagement."""
        campaign = next(
            (c for c in self.active_campaigns if c['id'] == campaign_id),
            None
        )
        
        if not campaign or campaign['impressions'] == 0:
            return 0.0
        
        retweet_rate = campaign.get('retweets', 0) / campaign['impressions']
        engagement_rate = campaign.get('engagements', 0) / campaign['impressions']
        
        viral_score = (retweet_rate * 0.6 + engagement_rate * 0.4) * 100
        
        return min(viral_score, 1.0)
    
    def get_x_specific_metrics(self, campaign_id: str) -> Dict:
        campaign = next(
            (c for c in self.active_campaigns if c['id'] == campaign_id),
            None
        )
        
        if not campaign:
            return {}
        
        impressions = campaign.get('impressions', 1)
        
        return {
            'impressions': impressions,
            'engagements': campaign.get('engagements', 0),
            'engagement_rate': campaign.get('engagements', 0) / impressions,
            'retweets': campaign.get('retweets', 0),
            'likes': campaign.get('likes', 0),
            'replies': campaign.get('replies', 0),
            'clicks': campaign.get('clicks', 0),
            'click_rate': campaign.get('clicks', 0) / impressions,
            'viral_potential': self.get_viral_potential(campaign_id),
            'cost_per_engagement': campaign.get('spent', 0) / max(campaign.get('engagements', 1), 1),
            'amplification_factor': impressions / max(campaign.get('engagements', 1), 1)
        }
    
    def simulate_viral_spread(self, campaign: Dict, initial_engagements: int):
        """Simulate viral spread via network effect of retweets."""
        current_reach = initial_engagements * 200
        total_viral_impressions = 0
        engagement_decay = 0.5

        for level in range(3):
            level_engagements = int(current_reach * 0.02 * (engagement_decay ** level))
            
            if level_engagements == 0:
                break
            
            total_viral_impressions += current_reach
            current_reach = level_engagements * 200

        return total_viral_impressions

    def get_platform_specific_metrics(self) -> Dict[str, Any]:
        return {
            'platform': 'x',
            'base_cpc': self.base_cpc,
            'base_cpm': self.base_cpm,
            'avg_quality_score': 5.5,
            'viral_potential': True,
            'real_time_focused': True,
            'avg_ctr': 0.015,
            'avg_conversion_rate': 0.01,
            'retweet_rate': 0.05
        }

    def simulate_user_behavior(
        self,
        content: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        import random

        base_engagement = 0.015

        content_type = content.get('type', 'tweet')
        if content_type == 'promoted':
            base_engagement *= 0.7
        elif content_type == 'thread':
            base_engagement *= 1.3

        viewed = True
        clicked = random.random() < base_engagement
        engaged = clicked and random.random() < 0.4
        retweeted = engaged and random.random() < 0.15
        converted = engaged and random.random() < 0.03

        return {
            'viewed': viewed,
            'clicked': clicked,
            'engaged': engaged,
            'retweeted': retweeted,
            'converted': converted,
            'time_spent': random.randint(2, 30) if clicked else 0,
            'interaction_type': 'retweet' if retweeted else ('click' if clicked else 'impression')
        }