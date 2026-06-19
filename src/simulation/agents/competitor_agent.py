"""
Competitor agent for market simulation
"""
import simpy
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CompetitorStrategy(str, Enum):
    AGGRESSIVE = "aggressive"  # Always bids high
    CONSERVATIVE = "conservative"  # Bids low, saves budget
    ADAPTIVE = "adaptive"  # Adapts to market conditions
    MIRROR = "mirror"  # Mirrors our strategy
    RANDOM = "random"  # Random bidding

class CompetitorAgent:
    
    def __init__(
        self,
        env: simpy.Environment,
        agent_id: str,
        name: str,
        budget: float,
        strategy: str,
        environment: Any
    ):
        self.env = env
        self.agent_id = agent_id
        self.name = name
        self.budget = budget
        self.initial_budget = budget
        self.strategy = CompetitorStrategy(strategy)
        self.market_env = environment
        
        self.spend = 0.0
        self.impressions = 0
        self.clicks = 0
        self.active_campaigns: List[Dict] = []
        self.bid_history: List[Dict] = []
        
        self.base_bid = self._calculate_base_bid()
        self.aggressiveness = self._get_aggressiveness()
        
    def _calculate_base_bid(self) -> float:
        daily_budget = self.budget / 30  # Assume 30-day campaigns
        return daily_budget / 100  # Assume ~100 bids per day
    
    def _get_aggressiveness(self) -> float:
        if self.strategy == CompetitorStrategy.AGGRESSIVE:
            return 0.9
        elif self.strategy == CompetitorStrategy.CONSERVATIVE:
            return 0.3
        elif self.strategy == CompetitorStrategy.ADAPTIVE:
            return 0.6
        elif self.strategy == CompetitorStrategy.MIRROR:
            return 0.5
        else:  # RANDOM
            return np.random.uniform(0.3, 0.9)
    
    def take_action(self):
        if self.budget <= 0:
            return
        
        action_type = self._decide_action()
        
        if action_type == "new_campaign":
            yield self.env.process(self._launch_campaign())
        elif action_type == "adjust_bid":
            yield self.env.process(self._adjust_bidding())
        elif action_type == "pause":
            yield self.env.process(self._pause_campaign())
        
        yield self.env.timeout(np.random.uniform(0.5, 2.0))
    
    def _decide_action(self) -> str:
        if not self.active_campaigns:
            return "new_campaign"
        
        if self.strategy == CompetitorStrategy.ADAPTIVE:
            recent_performance = self._get_recent_performance()
            if recent_performance < 0.5:  # Poor performance
                return np.random.choice(["adjust_bid", "pause"])
        
        actions = ["new_campaign", "adjust_bid", "pause", "none"]
        weights = [0.2, 0.4, 0.1, 0.3]
        
        return np.random.choice(actions, p=weights)
    
    def _launch_campaign(self):
        if self.budget < 1000:  # Minimum campaign budget
            return
        
        campaign_budget = min(
            self.budget * 0.3,  # 30% of remaining budget
            np.random.uniform(5000, 20000)
        )
        
        campaign = {
            'id': f"{self.agent_id}_campaign_{len(self.active_campaigns)}",
            'start_time': self.env.now,
            'budget': campaign_budget,
            'spent': 0.0,
            'platform': np.random.choice(['linkedin', 'twitter']),
            'target_persona': np.random.choice(['decision_maker', 'influencer', 'researcher']),
            'status': 'active'
        }
        
        self.active_campaigns.append(campaign)
        self.budget -= campaign_budget
        
        logger.info(f"Competitor {self.name} launched campaign with €{campaign_budget:.2f}")
        
        yield self.env.process(self._run_campaign(campaign))
    
    def _run_campaign(self, campaign: Dict):
        duration = np.random.uniform(7, 30)  # 7-30 days
        daily_budget = campaign['budget'] / duration
        
        for day in range(int(duration)):
            if campaign['status'] != 'active':
                break
            
            if self.budget <= 0:
                campaign['status'] = 'paused'
                break
            
            daily_spend = daily_budget * np.random.uniform(0.8, 1.2)
            daily_spend = min(daily_spend, campaign['budget'] - campaign['spent'])
            
            campaign['spent'] += daily_spend
            self.spend += daily_spend
            
            cpc = np.random.uniform(2.0, 8.0)  # Cost per click
            clicks = int(daily_spend / cpc)
            impressions = int(clicks / np.random.uniform(0.02, 0.05))  # 2-5% CTR
            
            self.clicks += clicks
            self.impressions += impressions
            
            if self.name not in self.market_env.market_state.competitor_spend:
                self.market_env.market_state.competitor_spend[self.name] = 0.0
            self.market_env.market_state.competitor_spend[self.name] += daily_spend
            
            yield self.env.timeout(1)
        
        campaign['status'] = 'completed'
        logger.info(f"Competitor {self.name} completed campaign, spent €{campaign['spent']:.2f}")
    
    def _adjust_bidding(self):
        if not self.active_campaigns:
            return
        
        active = [c for c in self.active_campaigns if c['status'] == 'active']
        
        if not active:
            return
        
        if self.strategy == CompetitorStrategy.AGGRESSIVE:
            bid_adjustment = 1.2
        elif self.strategy == CompetitorStrategy.CONSERVATIVE:
            bid_adjustment = 0.9
        elif self.strategy == CompetitorStrategy.ADAPTIVE:
            market_sentiment = self.market_env.market_state.market_sentiment
            bid_adjustment = 0.8 + 0.4 * market_sentiment
        else:
            bid_adjustment = np.random.uniform(0.9, 1.1)
        
        self.base_bid *= bid_adjustment
        
        self.bid_history.append({
            'timestamp': self.env.now,
            'old_bid': self.base_bid / bid_adjustment,
            'new_bid': self.base_bid,
            'reason': f'{self.strategy.value}_adjustment'
        })
        
        logger.debug(f"Competitor {self.name} adjusted bid to €{self.base_bid:.2f}")
        
        yield self.env.timeout(0.1)
    
    def _pause_campaign(self):
        active = [c for c in self.active_campaigns if c['status'] == 'active']
        
        if not active:
            return
        
        worst_campaign = min(
            active,
            key=lambda c: self.impressions / max(c['spent'], 1)
        )
        
        worst_campaign['status'] = 'paused'
        
        unspent = worst_campaign['budget'] - worst_campaign['spent']
        self.budget += unspent
        
        logger.info(f"Competitor {self.name} paused campaign, returned €{unspent:.2f}")
        
        yield self.env.timeout(0.1)
    
    def _get_recent_performance(self) -> float:
        if self.spend == 0:
            return 0.5
        
        ctr = self.clicks / max(self.impressions, 1)
        
        performance = min(ctr / 0.05, 1.0)
        
        return performance
    
    def compete_for_impression(self, auction_data: Dict) -> Dict:
        if self.budget <= 0:
            return {'bid': 0.0, 'competitor': self.name}
        
        bid = self._calculate_bid(auction_data)
        
        return {
            'bid': bid,
            'competitor': self.name,
            'quality_score': np.random.uniform(0.7, 1.0)
        }
    
    def _calculate_bid(self, auction_data: Dict) -> float:
        base_bid = self.base_bid
        
        if self.strategy == CompetitorStrategy.AGGRESSIVE:
            bid = base_bid * np.random.uniform(1.5, 2.5)
        
        elif self.strategy == CompetitorStrategy.CONSERVATIVE:
            bid = base_bid * np.random.uniform(0.5, 0.9)
        
        elif self.strategy == CompetitorStrategy.ADAPTIVE:
            market_sentiment = self.market_env.market_state.market_sentiment
            bid = base_bid * (0.8 + 0.6 * market_sentiment)
            
            # Bid more for preferred personas
            target_persona = auction_data.get('target_persona', '')
            if target_persona == 'decision_maker':
                bid *= 1.3
        
        elif self.strategy == CompetitorStrategy.MIRROR:
            # Estimate our bidding patterns from historical market data
            estimated_our_bid = self._estimate_opponent_bid(auction_data, base_bid)
            # Mirror with slight variation to compete
            bid = estimated_our_bid * np.random.uniform(0.95, 1.05)
        
        else:  # RANDOM
            bid = base_bid * np.random.uniform(0.5, 2.0)
        
        # Don't exceed budget
        bid = min(bid, self.budget)

        return bid

    def _estimate_opponent_bid(self, auction_data: Dict, base_bid: float) -> float:
        """
        Estimate opponent's (our) bidding patterns using historical analysis

        Analyzes bid history and market conditions to predict our likely bid
        """
        if hasattr(self.market_env, 'auction_history') and self.market_env.auction_history:
            recent_auctions = self.market_env.auction_history[-50:]  # Last 50 auctions

            target_persona = auction_data.get('target_persona', '')
            similar_auctions = [
                a for a in recent_auctions
                if a.get('target_persona') == target_persona
            ]

            if similar_auctions:
                our_bids = [
                    a.get('winning_bid', base_bid)
                    for a in similar_auctions
                    if a.get('winner') != self.name  # Our bids when we won or lost
                ]

                if our_bids:
                    # Use median of recent bids as estimate (robust to outliers)
                    estimated_bid = np.median(our_bids)

                    market_sentiment = self.market_env.market_state.market_sentiment
                    estimated_bid *= (0.9 + 0.2 * market_sentiment)

                    return estimated_bid

        # Fallback: Use base bid with market-adjusted heuristic
        # Assume opponent bids slightly above base in competitive markets
        competitiveness = getattr(self.market_env.market_state, 'competitiveness', 0.5)
        estimated_bid = base_bid * (1.0 + 0.3 * competitiveness)

        return estimated_bid

    def reset(self):
        self.budget = self.initial_budget
        self.spend = 0.0
        self.impressions = 0
        self.clicks = 0
        self.active_campaigns = []
        self.bid_history = []
        self.base_bid = self._calculate_base_bid()
    
    def get_activity_level(self) -> float:
        active_count = len([c for c in self.active_campaigns if c['status'] == 'active'])
        spend_rate = self.spend / max(self.initial_budget, 1)
        
        activity = (active_count / 3.0 + spend_rate) / 2.0
        return min(activity, 1.0)
    
    def get_summary(self) -> Dict:
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'budget_remaining': self.budget,
            'total_spend': self.spend,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'active_campaigns': len([c for c in self.active_campaigns if c['status'] == 'active']),
            'avg_ctr': self.clicks / max(self.impressions, 1),
            'activity_level': self.get_activity_level()
        }