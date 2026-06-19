"""
SimPy-based marketing environment simulation
"""
import simpy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..data_layer.database.models import Persona, Platform
from .agents.customer_agent import CustomerAgent
from .agents.competitor_agent import CompetitorAgent
from .platforms.linkedin_platform import LinkedInPlatform
from .platforms.x_platform import XPlatform
from .platforms.blog_platform import BlogPlatform
from .platforms.email_platform import EmailPlatform
from ..config.simulation_defaults import DEFAULT_SIMULATION_PERSONAS, DEFAULT_TRENDING_TOPICS

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for simulation run"""
    duration_days: int = 30
    time_step_hours: float = 1.0
    num_customers: int = 1000
    num_competitors: int = 3
    platforms: List[str] = field(default_factory=lambda: ["linkedin", "twitter", "blog", "email"])
    seed: Optional[int] = None
    validation_mode: bool = False
    historical_data_path: Optional[str] = None

@dataclass
class MarketState:
    """Current state of the simulated market"""
    timestamp: datetime
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    competitor_spend: Dict[str, float] = field(default_factory=dict)
    trending_topics: List[str] = field(default_factory=list)
    market_sentiment: float = 0.5  # 0-1 scale

class MarketingEnvironment:
    """
    Main simulation environment for marketing campaigns
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.env = simpy.Environment()
        
        if config.seed:
            np.random.seed(config.seed)
            
        self.customer_agents: List[CustomerAgent] = []
        self.competitor_agents: List[CompetitorAgent] = []
        self.platforms: Dict[str, Any] = {}
        
        self.market_state = MarketState(timestamp=datetime.now())
        self.campaign_results: List[Dict] = []
        self.interaction_log: List[Dict] = []
        
        self.metrics = {
            'impressions': [],
            'clicks': [],
            'conversions': [],
            'ctr': [],
            'cpl': []
        }
        
        self._init_platforms()
        
        if config.validation_mode and config.historical_data_path:
            self.historical_data = self._load_historical_data(config.historical_data_path)
        
    def _init_platforms(self):
        """Initialize marketing platforms"""
        if "linkedin" in self.config.platforms:
            self.platforms["linkedin"] = LinkedInPlatform(self.env, self)
            
        if "twitter" in self.config.platforms:
            self.platforms["twitter"] = XPlatform(self.env, self)
        
        if "blog" in self.config.platforms:
            self.platforms["blog"] = BlogPlatform(self.env, self)
        
        if "email" in self.config.platforms:
            self.platforms["email"] = EmailPlatform(self.env, self)
    
    def _load_historical_data(self, path: str) -> pd.DataFrame:
        """Load historical campaign data for validation"""
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows of historical data")
            return df
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return pd.DataFrame()
    
    def _create_default_personas(self):
        """Create default personas for simulation when none provided.
        Uses centralized defaults from simulation_defaults.py (configurable via dashboard)."""
        from ..data_layer.database.models import Persona

        try:
            from ..config.configuration_service import _get_config_value
            raw = _get_config_value('SIMULATION_DEFAULT_PERSONAS', None)
            if raw:
                import json as _json
                persona_defs = _json.loads(raw) if isinstance(raw, str) else raw
            else:
                persona_defs = DEFAULT_SIMULATION_PERSONAS
        except Exception:
            persona_defs = DEFAULT_SIMULATION_PERSONAS

        default_personas = [
            Persona(
                id=p["id"],
                name=p["name"],
                daily_active_prob=p["daily_active_prob"],
                click_prob=p["click_prob"],
                conversion_prob=p["conversion_prob"],
            )
            for p in persona_defs
        ]

        self.load_personas(default_personas)

    def load_personas(self, personas: List[Persona]):
        """Load customer personas and create agents"""
        for i in range(self.config.num_customers):
            persona = np.random.choice(personas)
            agent = CustomerAgent(
                env=self.env,
                agent_id=f"customer_{i}",
                persona=persona,
                environment=self
            )
            self.customer_agents.append(agent)

        logger.info(f"Created {len(self.customer_agents)} customer agents")
    
    def load_competitors(self, competitor_configs: List[Dict]):
        """Load competitor agents"""
        for i, config in enumerate(competitor_configs[:self.config.num_competitors]):
            agent = CompetitorAgent(
                env=self.env,
                agent_id=f"competitor_{i}",
                name=config.get('name', f'Competitor {i}'),
                budget=config.get('budget', 10000),
                strategy=config.get('strategy', 'aggressive'),
                environment=self
            )
            self.competitor_agents.append(agent)
        
        logger.info(f"Created {len(self.competitor_agents)} competitor agents")
    
    def run_campaign(self, campaign_config: Dict) -> Dict[str, Any]:
        """
        Run a marketing campaign in the simulation
        
        Args:
            campaign_config: Campaign configuration including:
                - platform: Target platform
                - content: Marketing content
                - targeting: Targeting parameters
                - budget: Campaign budget
                - duration: Campaign duration in days
        
        Returns:
            Campaign results with metrics
        """
        platform_name = campaign_config['platform']
        # Normalize: accept both enum objects and strings
        if hasattr(platform_name, 'value'):
            platform_name = platform_name.value
        platform_name = str(platform_name).lower()
        platform = self.platforms.get(platform_name)
        
        if not platform:
            raise ValueError(f"Platform {platform_name} not initialized")
        
        campaign_process = self.env.process(
            platform.run_campaign(campaign_config)
        )
        
        duration_hours = campaign_config.get('duration', 7) * 24
        self.env.run(until=duration_hours)
        
        results = self._collect_campaign_results(campaign_config)
        
        return results

    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the full simulation for the configured duration

        Returns:
            Simulation results with aggregate metrics
        """
        if not self.customer_agents:
            logger.warning("No customer agents found, creating default personas")
            self._create_default_personas()

        self.env.process(self.simulate_market_dynamics())

        for agent in self.customer_agents:
            self.env.process(agent.live())

        for agent in self.competitor_agents:
            if hasattr(agent, 'live'):
                self.env.process(agent.live())

        duration_hours = self.config.duration_days * 24
        self.env.run(until=duration_hours)

        total_impressions = self.market_state.total_impressions
        total_clicks = self.market_state.total_clicks
        total_conversions = self.market_state.total_conversions

        avg_ctr = total_clicks / total_impressions if total_impressions > 0 else 0.0
        conversion_rate = total_conversions / total_clicks if total_clicks > 0 else 0.0

        results = {
            'duration_days': self.config.duration_days,
            'total_interactions': total_impressions,  # Total impressions count as interactions
            'total_clicks': total_clicks,
            'total_conversions': total_conversions,
            # Return CTR and conversion_rate as PERCENTAGES for consistency with API
            'avg_ctr': avg_ctr * 100,  # Convert to percentage (e.g., 1.58 for 1.58%)
            'conversion_rate': conversion_rate * 100,  # Convert to percentage
            'num_customers': len(self.customer_agents),
            'num_competitors': len(self.competitor_agents),
            'platforms': list(self.platforms.keys())
        }

        return results

    def _collect_campaign_results(self, campaign_config: Dict) -> Dict[str, Any]:
        """Collect and aggregate campaign results"""
        platform_name = campaign_config['platform']
        platform = self.platforms[platform_name]
        
        metrics = platform.get_campaign_metrics()
        
        # Calculate derived metrics - CTR as PERCENTAGE for consistency
        if metrics['impressions'] > 0:
            metrics['ctr'] = (metrics['clicks'] / metrics['impressions']) * 100
        else:
            metrics['ctr'] = 0.0
            
        if metrics['conversions'] > 0:
            metrics['cpl'] = campaign_config['budget'] / metrics['conversions']
        else:
            metrics['cpl'] = float('inf')
        
        self.campaign_results.append({
            'config': campaign_config,
            'metrics': metrics,
            'timestamp': self.env.now
        })
        
        return metrics
    
    def simulate_market_dynamics(self):
        """
        Background process simulating market dynamics
        """
        while True:
            sentiment_change = np.random.normal(0, 0.01)
            self.market_state.market_sentiment = np.clip(
                self.market_state.market_sentiment + sentiment_change,
                0, 1
            )
            
            if np.random.random() < 0.1:  # 10% chance of new trend
                new_trend = self._generate_trending_topic()
                self.market_state.trending_topics.append(new_trend)
                self.market_state.trending_topics = self.market_state.trending_topics[-5:]
            
            for competitor in self.competitor_agents:
                if np.random.random() < 0.3:  # 30% chance of competitor action
                    self.env.process(competitor.take_action())
            
            yield self.env.timeout(self.config.time_step_hours)
    
    def _generate_trending_topic(self) -> str:
        """Generate a trending topic from configurable domain-relevant list"""
        try:
            from ..config.configuration_service import _get_config_value
            raw = _get_config_value('SIMULATION_TRENDING_TOPICS', None)
            if raw:
                import json as _json
                topics = _json.loads(raw) if isinstance(raw, str) else raw
            else:
                topics = DEFAULT_TRENDING_TOPICS
        except Exception:
            topics = DEFAULT_TRENDING_TOPICS
        return np.random.choice(topics)
    
    def validate_against_historical(self) -> Dict[str, float]:
        """
        Validate simulation accuracy against historical data

        Returns:
            Validation metrics including MAPE
        """
        if not self.config.validation_mode or self.historical_data.empty:
            logger.warning("Validation requires historical data")
            return {}

        validation_results = []

        for _, row in self.historical_data.iterrows():
            campaign_config = {
                'platform': row['platform'],
                'content': row['content'],
                'targeting': json.loads(row['targeting']),
                'budget': row['budget'],
                'duration': row['duration_days']
            }

            sim_results = self.run_campaign(campaign_config)

            validation_results.append({
                'actual_clicks': row['clicks'],
                'simulated_clicks': sim_results['clicks'],
                'actual_conversions': row['conversions'],
                'simulated_conversions': sim_results['conversions'],
                'actual_ctr': row['ctr'],
                'simulated_ctr': sim_results['ctr']
            })

            self.reset()

        df = pd.DataFrame(validation_results)

        metrics = {}
        for col in ['clicks', 'conversions', 'ctr']:
            actual_col = f'actual_{col}'
            sim_col = f'simulated_{col}'

            mape = np.mean(np.abs((df[actual_col] - df[sim_col]) / df[actual_col])) * 100
            metrics[f'{col}_mape'] = mape

            correlation = df[actual_col].corr(df[sim_col])
            metrics[f'{col}_correlation'] = correlation

        avg_mape = np.mean([metrics['clicks_mape'], metrics['conversions_mape']])
        metrics['overall_accuracy'] = max(0, 100 - avg_mape)

        logger.info(f"Validation complete: {metrics['overall_accuracy']:.2f}% accuracy")

        return metrics

    async def load_calibrated_personas(self, db_session) -> List[Persona]:
        """
        Load calibrated persona parameters from database.
        Uses PersonaCalibration table to get data-driven parameters.

        Args:
            db_session: Async database session

        Returns:
            List of Persona objects with calibrated parameters
        """
        from ..data_layer.database.models import PersonaCalibration
        from sqlalchemy import select

        result = await db_session.execute(
            select(PersonaCalibration).where(PersonaCalibration.is_active == True)
        )
        calibrations = result.scalars().all()

        if not calibrations:
            logger.warning("No active calibrations found, using defaults")
            self._create_default_personas()
            return

        personas = []
        for calib in calibrations:
            persona = Persona(
                name=calib.persona_name,
                title=calib.persona_name.replace('_', ' ').title(),
                description=f"Calibrated {calib.persona_name}",
                role=calib.persona_name,
                daily_active_prob=calib.daily_active_prob,
                click_prob=calib.click_prob,
                conversion_prob=calib.conversion_prob,
                content_engagement_prob=calib.content_engagement_prob,
                share_prob=calib.share_prob,
                active_hours=calib.active_hours,
                attributes={
                    'ad_fatigue_threshold': calib.ad_fatigue_threshold,
                    'ad_fatigue_decay': calib.ad_fatigue_decay,
                    'influence_factor': calib.influence_factor,
                    'calibration_id': str(calib.id),
                    'training_mape': calib.training_mape
                }
            )
            personas.append(persona)
            logger.info(f"Loaded calibrated {calib.persona_name}: "
                       f"click_prob={calib.click_prob:.4f}, "
                       f"conv_prob={calib.conversion_prob:.4f}, "
                       f"MAPE={calib.training_mape:.2f}%")

        self.load_personas(personas)
        return personas
    
    def reset(self):
        """Reset environment for new simulation run"""
        self.env = simpy.Environment()
        self.market_state = MarketState(timestamp=datetime.now())
        self.campaign_results = []
        self.interaction_log = []
        
        for platform in self.platforms.values():
            platform.reset()
        
        for agent in self.customer_agents:
            agent.reset()
        for agent in self.competitor_agents:
            agent.reset()
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state of the simulation"""
        return {
            'timestamp': self.env.now,
            'market_state': {
                'sentiment': self.market_state.market_sentiment,
                'trending_topics': self.market_state.trending_topics,
                'total_impressions': self.market_state.total_impressions,
                'total_clicks': self.market_state.total_clicks,
                'total_conversions': self.market_state.total_conversions
            },
            'active_customers': len([a for a in self.customer_agents if a.is_active]),
            'competitor_activity': {
                c.name: c.get_activity_level() 
                for c in self.competitor_agents
            }
        }
    
    def export_results(self, filepath: str):
        """Export simulation results to file"""
        results = {
            'config': {
                'duration_days': self.config.duration_days,
                'num_customers': self.config.num_customers,
                'num_competitors': self.config.num_competitors
            },
            'campaigns': self.campaign_results,
            'interactions': self.interaction_log,
            'metrics': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")