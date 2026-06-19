"""
Customer agent for market simulation
"""
import simpy
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ...data_layer.database.models import Persona

logger = logging.getLogger(__name__)

@dataclass
class CustomerState:
    agent_id: str
    is_active: bool = False
    impressions_seen: int = 0
    ads_clicked: int = 0
    content_engaged: int = 0
    converted: bool = False
    last_interaction: Optional[datetime] = None
    ad_fatigue_level: float = 0.0
    interest_level: float = 0.5
    influenced_by: List[str] = None
    
    def __post_init__(self):
        if self.influenced_by is None:
            self.influenced_by = []

class CustomerAgent:
    
    def __init__(
        self,
        env: simpy.Environment,
        agent_id: str,
        persona: Persona,
        environment: Any
    ):
        self.env = env
        self.agent_id = agent_id
        self.persona = persona
        self.market_env = environment
        
        self.config = self._parse_persona_config()
        
        self.state = CustomerState(agent_id=agent_id)
        
        self.interaction_history: List[Dict] = []
        self.conversion_intent = self._initialize_conversion_intent()
        
        self.connections: List['CustomerAgent'] = []
        
    def _parse_persona_config(self) -> Dict:
        return {
            'daily_active_prob': getattr(self.persona, 'daily_active_prob', 0.4),
            'click_prob': getattr(self.persona, 'click_prob', 0.03),
            'conversion_prob': getattr(self.persona, 'conversion_prob', 0.01),
            'ad_fatigue_threshold': getattr(self.persona, 'ad_fatigue_threshold', 5),
            'ad_fatigue_decay': getattr(self.persona, 'ad_fatigue_decay', 0.15),
            'influence_factor': getattr(self.persona, 'influence_factor', 0.3),
            'peak_hours': getattr(self.persona, 'peak_hours', [9, 10, 11, 14, 15, 16]),
        }
    
    def _initialize_conversion_intent(self) -> float:
        # Use lognormal distribution for realistic buying intent
        intent = np.random.lognormal(mean=-1.0, sigma=0.8)
        return min(max(intent, 0.0), 1.0)
    
    def live(self):
        while True:
            if self._is_active_today():
                self.state.is_active = True
                
                yield self.env.process(self._daily_activity())
                
                self.state.is_active = False
            
            yield self.env.timeout(24)  # 24 hours
    
    def _is_active_today(self) -> bool:
        base_prob = self.config['daily_active_prob']
        
        sentiment_factor = self.market_env.market_state.market_sentiment
        adjusted_prob = base_prob * (0.8 + 0.4 * sentiment_factor)
        
        if self.connections:
            active_peers = sum(1 for c in self.connections if c.state.is_active)
            if active_peers > 0:
                peer_influence = self.config['influence_factor']
                adjusted_prob *= (1 + peer_influence * (active_peers / len(self.connections)))
        
        return np.random.random() < adjusted_prob
    
    def _daily_activity(self):
        peak_hours = self.config['peak_hours']
        
        for hour in peak_hours:
            yield self.env.timeout(1)  # 1 hour
            
            if self._checks_platform_this_hour():
                yield self.env.process(self._platform_session())
    
    def _checks_platform_this_hour(self) -> bool:
        # Higher probability during peak hours
        current_hour = int(self.env.now) % 24
        
        if current_hour in self.config['peak_hours']:
            return np.random.random() < 0.3  # 30% chance during peak
        else:
            return np.random.random() < 0.1  # 10% chance off-peak
    
    def _platform_session(self):
        session_duration = np.random.uniform(5/60, 30/60)  # Convert to hours

        num_impressions = np.random.poisson(3)  # Average 3 impressions per session

        for _ in range(num_impressions):
            self.see_impression()

            if self.decide_click():
                if self.state.ads_clicked >= 2:  # Need at least 2 clicks before considering conversion
                    self.consider_conversion()

            yield self.env.timeout(session_duration / max(num_impressions, 1))
    
    def see_impression(self, ad_data: Optional[Dict] = None):
        self.state.impressions_seen += 1
        self.state.last_interaction = datetime.now()
        
        self.market_env.market_state.total_impressions += 1
        
        self._update_ad_fatigue()
        
        self.interaction_history.append({
            'type': 'impression',
            'timestamp': self.env.now,
            'ad_data': ad_data,
            'fatigue_level': self.state.ad_fatigue_level
        })
    
    def _update_ad_fatigue(self):
        if self.state.impressions_seen > self.config['ad_fatigue_threshold']:
            excess = self.state.impressions_seen - self.config['ad_fatigue_threshold']
            self.state.ad_fatigue_level = 1 - (0.9 ** excess)
        else:
            self.state.ad_fatigue_level = 0.0
    
    def decide_click(self, ad_data: Optional[Dict] = None) -> bool:
        base_click_prob = self.config['click_prob']
        
        fatigue_penalty = self.state.ad_fatigue_level * self.config['ad_fatigue_decay']
        adjusted_prob = base_click_prob * (1 - fatigue_penalty)
        
        if ad_data:
            relevance_score = self._calculate_relevance(ad_data)
            adjusted_prob *= relevance_score
        
        adjusted_prob *= (0.5 + 0.5 * self.state.interest_level)
        
        will_click = np.random.random() < adjusted_prob
        
        if will_click:
            self.state.ads_clicked += 1
            self.market_env.market_state.total_clicks += 1
            
            self.state.interest_level = min(1.0, self.state.interest_level + 0.1)
            
            self.interaction_history.append({
                'type': 'click',
                'timestamp': self.env.now,
                'ad_data': ad_data,
                'probability': adjusted_prob
            })
        
        return will_click
    
    def _calculate_relevance(self, ad_data: Dict) -> float:
        relevance = 1.0

        if ad_data.get('target_persona') == self.persona.name:
            relevance *= 1.5

        content_type = ad_data.get('content_type', '')
        if content_type:
            preferred_content = getattr(self.persona, 'preferred_content_types', [])
            if isinstance(preferred_content, str):
                preferred_content = [preferred_content]

            if content_type in preferred_content:
                relevance *= 1.3

            persona_attrs = getattr(self.persona, 'characteristics', {})
            if isinstance(persona_attrs, dict):
                if content_type == 'technical' and persona_attrs.get('technical_depth', False):
                    relevance *= 1.2
                elif content_type == 'executive' and persona_attrs.get('senior_level', False):
                    relevance *= 1.2

        if 'keywords' in ad_data:
            trending = self.market_env.market_state.trending_topics
            if any(keyword in trending for keyword in ad_data['keywords']):
                relevance *= 1.2

        return min(relevance, 2.0)  # Cap at 2x
    
    def consider_conversion(self, offer_data: Optional[Dict] = None) -> bool:
        if self.state.converted:
            return False  # Already converted
        
        if self.state.ads_clicked == 0:
            return False
        
        base_conv_prob = self.config['conversion_prob']
        
        engagement_factor = min(self.state.ads_clicked / 3.0, 2.0)
        adjusted_prob = base_conv_prob * engagement_factor
        
        adjusted_prob *= self.conversion_intent
        
        # Influence from peer conversions
        if self.connections:
            converted_peers = sum(1 for c in self.connections if c.state.converted)
            if converted_peers > 0:
                peer_boost = self.config['influence_factor']
                adjusted_prob *= (1 + peer_boost * (converted_peers / len(self.connections)))
        
        will_convert = np.random.random() < adjusted_prob
        
        if will_convert:
            self.state.converted = True
            self.market_env.market_state.total_conversions += 1
            
            self.interaction_history.append({
                'type': 'conversion',
                'timestamp': self.env.now,
                'offer_data': offer_data,
                'probability': adjusted_prob
            })
            
            logger.info(f"Agent {self.agent_id} converted!")
        
        return will_convert
    
    def add_connection(self, other_agent: 'CustomerAgent'):
        if other_agent not in self.connections:
            self.connections.append(other_agent)
    
    def get_activity_level(self) -> float:
        if not self.interaction_history:
            return 0.0
        
        recent_interactions = len([
            i for i in self.interaction_history[-10:]
            if i['type'] in ['click', 'conversion']
        ])
        
        return min(recent_interactions / 5.0, 1.0)
    
    def reset(self):
        self.state = CustomerState(agent_id=self.agent_id)
        self.interaction_history = []
        self.conversion_intent = self._initialize_conversion_intent()
    
    def get_summary(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'persona': self.persona.name,
            'impressions': self.state.impressions_seen,
            'clicks': self.state.ads_clicked,
            'converted': self.state.converted,
            'ad_fatigue': self.state.ad_fatigue_level,
            'interest_level': self.state.interest_level,
            'activity_level': self.get_activity_level(),
            'interactions': len(self.interaction_history)
        }