"""LinkedIn platform simulation."""
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_platform import BasePlatform

logger = logging.getLogger(__name__)

class LinkedInPlatform(BasePlatform):
    """Simulates LinkedIn advertising platform (B2B focused, higher CPC, professional targeting)."""

    def __init__(self, env, market_env):
        super().__init__(
            name="linkedin",
            base_cpm=33.0,
            base_cpc=5.50,
            auction_mechanism="second_price"
        )
        self.env = env
        self.market_env = market_env

    def _get_base_cpc(self) -> float:
        return 5.50
    
    def _get_base_cpm(self) -> float:
        return 33.00
    
    def _get_platform_fee(self) -> float:
        return 0.15
    
    def _get_peak_multiplier(self, hour: int) -> float:
        """LinkedIn activity peaks during business hours (8am-6pm weekdays)."""
        day_of_week = (int(self.env.now / 24) % 7)
        
        if day_of_week >= 5:
            return 0.3
        
        if 8 <= hour <= 18:
            if hour in [9, 10, 14, 15]:
                return 1.5
            else:
                return 1.0
        elif 6 <= hour < 8 or 18 < hour <= 22:
            return 0.6
        else:
            return 0.2
    
    def _calculate_targeting_precision(self, campaign_config: Dict) -> float:
        """Returns precision score [0.5-1.5] based on B2B targeting criteria."""
        precision = 1.0
        
        targeting = campaign_config.get('targeting', {})
        
        if targeting.get('job_titles'):
            precision *= 1.2
        
        if targeting.get('company_sizes'):
            precision *= 1.1
        
        if targeting.get('industries'):
            precision *= 1.15
        
        if targeting.get('seniority_levels'):
            precision *= 1.1
        
        return min(precision, 1.5)
    
    def _calculate_our_bid(self, campaign: Dict, customer: Any) -> float:
        base_bid = super()._calculate_our_bid(campaign, customer)
        
        targeting_precision = self._calculate_targeting_precision(campaign['config'])
        adjusted_bid = base_bid * targeting_precision
        
        if customer.persona.name == 'decision_maker':
            adjusted_bid *= 1.5
        
        return adjusted_bid
    
    def run_sponsored_inmail(self, campaign_config: Dict):
        """Run LinkedIn Sponsored InMail campaign (direct messaging)."""
        campaign_id = f"inmail_{len(self.active_campaigns)}"
        
        campaign = {
            'id': campaign_id,
            'type': 'sponsored_inmail',
            'config': campaign_config,
            'start_time': self.env.now,
            'duration': campaign_config.get('duration', 14),
            'budget': campaign_config.get('budget', 3000),
            'spent': 0.0,
            'messages_sent': 0,
            'opens': 0,
            'clicks': 0,
            'responses': 0,
            'status': 'active'
        }
        
        self.active_campaigns.append(campaign)
        
        logger.info(f"Started InMail campaign {campaign_id}")
        
        # Deliver InMail campaign
        yield self.env.process(self._deliver_inmail_campaign(campaign))
    
    def _deliver_inmail_campaign(self, campaign: Dict):
        duration_days = campaign['duration']
        daily_send_limit = 100
        
        cost_per_send = 0.80
        
        for day in range(duration_days):
            if campaign['spent'] >= campaign['budget']:
                break
            
            # Send messages
            messages_today = min(
                daily_send_limit,
                int((campaign['budget'] - campaign['spent']) / cost_per_send)
            )
            
            for _ in range(messages_today):
                target = self._select_inmail_target(campaign)
                
                if target:
                    campaign['messages_sent'] += 1
                    campaign['spent'] += cost_per_send
                    
                    # LinkedIn InMail benchmark rates
                    open_rate = 0.55
                    if np.random.random() < open_rate:
                        campaign['opens'] += 1
                        
                        click_rate = 0.25
                        if np.random.random() < click_rate:
                            campaign['clicks'] += 1
                            
                            response_rate = 0.10
                            if np.random.random() < response_rate:
                                campaign['responses'] += 1
            
            yield self.env.timeout(24)
        
        campaign['status'] = 'completed'
        logger.info(
            f"InMail campaign {campaign['id']} completed. "
            f"Sent: {campaign['messages_sent']}, "
            f"Opens: {campaign['opens']}, "
            f"Responses: {campaign['responses']}"
        )
    
    def _select_inmail_target(self, campaign: Dict) -> Optional[Any]:
        targeting = campaign['config'].get('targeting', {})
        persona = targeting.get('persona', 'decision_maker')
        
        eligible = [
            agent for agent in self.market_env.customer_agents
            if agent.persona.name == persona and agent.state.is_active
        ]
        
        if not eligible:
            return None
        
        return np.random.choice(eligible)
    
    def get_linkedin_specific_metrics(self, campaign_id: str) -> Dict:
        campaign = next(
            (c for c in self.active_campaigns if c['id'] == campaign_id),
            None
        )
        
        if not campaign:
            return {}
        
        if campaign.get('type') == 'sponsored_inmail':
            return {
                'messages_sent': campaign.get('messages_sent', 0),
                'open_rate': campaign.get('opens', 0) / max(campaign.get('messages_sent', 1), 1),
                'click_rate': campaign.get('clicks', 0) / max(campaign.get('opens', 1), 1),
                'response_rate': campaign.get('responses', 0) / max(campaign.get('messages_sent', 1), 1),
                'cost_per_send': campaign.get('spent', 0) / max(campaign.get('messages_sent', 1), 1)
            }
        else:
            base_metrics = self.get_campaign_metrics(campaign_id)
            base_metrics['quality_score'] = self._calculate_quality_score(campaign)
            base_metrics['relevance_score'] = self._calculate_relevance_score(campaign)
            
            return base_metrics
    
    def _calculate_quality_score(self, campaign: Dict) -> float:
        """Calculate LinkedIn quality score [0-10] based on CTR vs platform avg (~0.45%)."""
        metrics = self.get_campaign_metrics(campaign['id'])
        ctr = metrics['ctr']
        
        if ctr >= 0.0045:
            quality = 10.0
        elif ctr >= 0.003:
            quality = 8.0
        elif ctr >= 0.002:
            quality = 6.0
        elif ctr >= 0.001:
            quality = 4.0
        else:
            quality = 2.0
        
        return quality
    
    def _calculate_relevance_score(self, campaign: Dict) -> float:
        precision = self._calculate_targeting_precision(campaign['config'])
        return min(precision / 1.5, 1.0)

    def get_platform_specific_metrics(self) -> Dict[str, Any]:
        return {
            'platform': 'linkedin',
            'base_cpc': self.base_cpc,
            'base_cpm': self.base_cpm,
            'avg_quality_score': 7.5,
            'professional_audience': True,
            'b2b_focused': True,
            'avg_ctr': 0.003,
            'avg_conversion_rate': 0.02
        }

    def simulate_user_behavior(
        self,
        content: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        import random

        base_engagement = 0.03

        content_type = content.get('type', 'post')
        if content_type == 'sponsored':
            base_engagement *= 0.8
        elif content_type == 'article':
            base_engagement *= 1.2

        viewed = True
        clicked = random.random() < base_engagement
        engaged = clicked and random.random() < 0.3
        converted = engaged and random.random() < 0.05

        return {
            'viewed': viewed,
            'clicked': clicked,
            'engaged': engaged,
            'converted': converted,
            'time_spent': random.randint(5, 60) if clicked else 0,
            'interaction_type': 'click' if clicked else 'impression'
        }