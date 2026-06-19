"""
Strategy optimization agent with GPU acceleration for learning algorithms
"""
import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

from ..learning.thompson_sampling import ThompsonSamplingBandit
from ..learning.linucb import LinUCBBandit
from ..learning.advanced_experiments import AdvancedExperimentRunner, ExperimentType
from ...data_layer.database.models import Campaign, Experiment, BanditArm, StrategyPerformance
from ...data_layer.database.connection import get_sync_session
from ...config.settings import settings
from ...config.configuration_service import _get_config_value
from ..memory.episodic_memory import EpisodicMemoryStore, AgentMemory, create_memory_from_task
from ..learning.simulation_accuracy_tracker import SimulationAccuracyTracker

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.warning("No GPU detected, using CPU")

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    algorithm: str = "thompson_sampling"
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    use_gpu: bool = True
    batch_size: int = 32
    context_dim: int = 50

class NeuralBandit(nn.Module):
    """Neural network for contextual bandit (GPU optimized)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_arms: int = 4):
        super(NeuralBandit, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_arms)
        )
        
        self.to(DEVICE)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)
    
    def predict_batch(self, contexts: torch.Tensor) -> torch.Tensor:
        """Batch prediction on GPU"""
        with torch.no_grad():
            contexts = contexts.to(DEVICE)
            return self.forward(contexts)

class StrategyOptimizerAgent:
    """
    Agent responsible for optimizing marketing strategy using bandits and RL
    """
    
    def __init__(self):
        self.config = StrategyConfig(
            algorithm=settings.get('BANDIT_ALGORITHM', 'thompson_sampling'),
            exploration_rate=settings.BANDIT_EXPLORATION_RATE,
            learning_rate=settings.BANDIT_LEARNING_RATE,
            use_gpu=settings.get('USE_GPU', torch.cuda.is_available())
        )
        
        self.active_bandits = {}

        self.memory = EpisodicMemoryStore(agent_name="strategy_optimizer")

        self.research_mode = getattr(settings, 'ENABLE_RESEARCH_MODE', True)
        if self.research_mode:
            self.experiment_runner = AdvancedExperimentRunner()
            logger.info(f"Research mode enabled: {self.experiment_runner.config.experiment_type}")
        else:
            self.experiment_runner = None

        if settings.get('ENABLE_NEURAL_BANDIT', False):
            self.neural_bandit = NeuralBandit(
                input_dim=self.config.context_dim,
                num_arms=10  # Max arms
            )
            self.optimizer = torch.optim.Adam(
                self.neural_bandit.parameters(),
                lr=self.config.learning_rate
            )
            self.criterion = nn.CrossEntropyLoss()

        self.performance_history = []
        
        # Simulation Accuracy Tracker (RQ2)
        self.tracker = SimulationAccuracyTracker()
        
    async def get_optimal_strategy(
        self,
        campaign_id: str,
        platform: str,
        target_persona: str,
        budget: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get optimal strategy for campaign

        Args:
            campaign_id: Campaign identifier
            platform: Target platform
            target_persona: Target persona
            budget: Available budget
            context: Additional context

        Returns:
            Optimal strategy configuration
        """
        task_start_time = datetime.now()
        task_id = f"{campaign_id}_{platform}_{target_persona}_{task_start_time.timestamp()}"
        actions_taken = []

        try:
            memory_query = f"Optimize strategy for {platform} targeting {target_persona} with budget {budget}"
            relevant_memories = await self.memory.retrieve_relevant_memories(
                query=memory_query,
                k=3,
                outcome_filter=None
            )

            if relevant_memories:
                actions_taken.append(f"Retrieved {len(relevant_memories)} past optimization experiences")
                logger.info(f"Using {len(relevant_memories)} past memories for strategy optimization")

            context_vector = self._build_context_vector(
                platform=platform,
                persona=target_persona,
                budget=budget,
                additional_context=context
            )
            actions_taken.append("Built context vector")
            
            if self.research_mode and self.experiment_runner is not None:
                logger.info(f"Using research mode: {self.experiment_runner.config.experiment_type}")

                experiment_data = {
                    'campaign_id': campaign_id,
                    'platform': platform,
                    'persona': target_persona,
                    'budget': budget,
                    'context_vector': context_vector.tolist(),
                    'n_arms': 4,  # Standard number of strategy variants
                    'contexts': [f"{platform} campaign for {target_persona}"]  # Text context for transformer bandits
                }

                experiment_result = self.experiment_runner.run_experiment(experiment_data)
                actions_taken.append(f"Ran {self.experiment_runner.config.experiment_type} experiment")

                # For research mode, we use the experiment's recommended action
                action = experiment_result.get('selected_action', 'hook_transform')  # Default fallback
                confidence = experiment_result.get('confidence', 0.7)

                logger.info(f"Research experiment selected action: {action} (confidence: {confidence:.2f})")
            else:
                bandit = self._get_or_create_bandit(campaign_id)
                actions_taken.append(f"Retrieved/created bandit for campaign {campaign_id}")

                if self.config.use_gpu and hasattr(self, 'neural_bandit'):
                    action, confidence = self._neural_bandit_selection(context_vector)
                else:
                    arm_result, confidence = bandit.select_arm(context_vector)

                    # Handle LinUCB (returns int index) vs Thompson Sampling (returns string name)
                    if isinstance(arm_result, int) and hasattr(bandit, 'arm_names'):
                        # LinUCB bandit - convert index to arm name
                        action = bandit.arm_names[arm_result]
                    else:
                        # Thompson Sampling - already returns arm name
                        action = arm_result
            
            strategy = self._build_strategy(
                action=action,
                confidence=confidence,
                platform=platform,
                persona=target_persona,
                budget=budget
            )
            
            # Include experiment_id so deployment feedback loop can update bandit arms
            if campaign_id in self.active_bandits:
                bandit_instance = self.active_bandits[campaign_id]
                if hasattr(bandit_instance, 'experiment_id'):
                    strategy['experiment_id'] = str(bandit_instance.experiment_id)
            
            actions_taken.append(f"Selected strategy: {strategy['strategy_name']} (confidence: {confidence:.2f})")

            logger.info(f"Strategy selected for campaign {campaign_id}: {strategy['strategy_name']}")

            duration = (datetime.now() - task_start_time).total_seconds()
            task_result = {
                'success': True,
                'cost': 0.0,  # No direct LLM cost for bandit selection
                'duration': duration,
                'quality_score': confidence,
                'action': action,
                'confidence': confidence
            }

            memory = create_memory_from_task(
                agent_name="strategy_optimizer",
                task_id=task_id,
                task_description=f"Optimize strategy for {platform} targeting {target_persona} with budget €{budget:.2f}",
                actions=actions_taken,
                result=task_result,
                human_feedback=None  # Will be updated later with performance feedback
            )

            await self.memory.store_memory(memory)

            # RQ2: Record simulation predictions for accuracy tracking
            try:
                estimated_ctr = confidence * 0.05
                estimated_cpl = budget / max(1, confidence * 10)
                estimated_clicks = int((budget * 0.7) / 2.5)  # Assume CPC ~2.5
                estimated_impressions = int(estimated_clicks / max(0.001, estimated_ctr))
                estimated_conversions = int(estimated_clicks * (confidence * 0.02))

                await self.tracker.record_simulation_predictions(
                    campaign_id=campaign_id,
                    simulated_impressions=estimated_impressions,
                    simulated_clicks=estimated_clicks,
                    simulated_conversions=estimated_conversions,
                    simulated_ctr=estimated_ctr,
                    simulated_cpl=estimated_cpl,
                    simulation_timestamp=datetime.utcnow()
                )
                logger.info(f"Recorded simulation predictions for campaign {campaign_id}")
            except Exception as e:
                logger.warning(f"Failed to record simulation predictions: {e}")

            return strategy

        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")

            duration = (datetime.now() - task_start_time).total_seconds()
            actions_taken.append(f"Failed: {str(e)}")

            failure_result = {
                'success': False,
                'cost': 0.0,
                'duration': duration,
                'quality_score': 0.0,
                'error': str(e)
            }

            failure_memory = create_memory_from_task(
                agent_name="strategy_optimizer",
                task_id=task_id,
                task_description=f"Optimize strategy for {platform} targeting {target_persona} with budget €{budget:.2f}",
                actions=actions_taken,
                result=failure_result
            )

            await self.memory.store_memory(failure_memory)

            return self._get_default_strategy(platform, target_persona)

    async def optimize(
        self,
        campaign_id: str,
        historical_data: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy for a campaign (simplified interface for testing)

        Args:
            campaign_id: Campaign identifier
            historical_data: Historical performance data
            constraints: Campaign constraints (budget, duration, etc.)

        Returns:
            Optimized strategy with recommended action and confidence
        """
        platform = constraints.get('platform', 'linkedin') if constraints else 'linkedin'
        target_persona = constraints.get('target_persona', 'decision_maker') if constraints else 'decision_maker'
        budget = constraints.get('max_budget', 1000.0) if constraints else 1000.0

        context = historical_data if historical_data else {}

        return await self.get_optimal_strategy(
            campaign_id=campaign_id,
            platform=platform,
            target_persona=target_persona,
            budget=budget,
            context=context
        )

    def _build_context_vector(
        self,
        platform: str,
        persona: str,
        budget: float,
        additional_context: Optional[Dict] = None
    ) -> np.ndarray:
        """Build context vector for decision making"""
        
        context = np.zeros(self.config.context_dim)
        
        platform_map = {'linkedin': 0, 'twitter': 1, 'email': 2}
        if platform in platform_map:
            context[platform_map[platform]] = 1
        
        persona_map = {
            'decision_maker': 3,
            'influencer': 4,
            'researcher': 5,
            'technical_buyer': 6
        }
        if persona in persona_map:
            context[persona_map[persona]] = 1
        
        context[7] = np.log1p(budget) / 10  # Log-scaled budget
        context[8] = min(1.0, budget / 10000)  # Normalized budget
        
        now = datetime.now()
        context[9] = now.hour / 24  # Hour of day
        context[10] = now.weekday() / 7  # Day of week
        context[11] = now.day / 31  # Day of month
        
        if additional_context:
            if 'past_ctr' in additional_context:
                context[12] = additional_context['past_ctr']
            if 'past_conversions' in additional_context:
                context[13] = np.log1p(additional_context['past_conversions']) / 10
        
        return context
    
    def _neural_bandit_selection(self, context: np.ndarray) -> Tuple[str, float]:
        """Select action using neural bandit on GPU"""
        
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(DEVICE)
        
        q_values = self.neural_bandit.predict_batch(context_tensor)
        
        if np.random.random() < self.config.exploration_rate:
            action_idx = np.random.randint(0, q_values.shape[1])
            confidence = 0.5
        else:
            action_idx = torch.argmax(q_values, dim=1).cpu().item()
            confidence = torch.softmax(q_values, dim=1).max().cpu().item()
        
        action_map = {
            0: "hook_a",
            1: "hook_b",
            2: "hook_c",
            3: "hook_d",
            4: "cta_a",
            5: "cta_b",
            6: "format_short",
            7: "format_long",
            8: "tone_professional",
            9: "tone_casual"
        }
        
        action = action_map.get(action_idx, "hook_a")
        
        return action, confidence
    
    def _get_or_create_bandit(self, campaign_id: str):
        """Get or create bandit for campaign based on configured algorithm"""

        if campaign_id not in self.active_bandits:
            arms = [
                "hook_transform",
                "hook_problem",
                "hook_success",
                "hook_question"
            ]

            if self.config.algorithm == "linucb":
                logger.info(f"Creating LinUCB bandit for campaign {campaign_id}")
                self.active_bandits[campaign_id] = LinUCBBandit(
                    n_arms=len(arms),
                    n_features=self.config.context_dim,
                    alpha=self.config.exploration_rate,
                    use_gpu=self.config.use_gpu
                )
                self.active_bandits[campaign_id].arm_names = arms
            else:
                logger.info(f"Creating Thompson Sampling bandit for campaign {campaign_id}")
                self.active_bandits[campaign_id] = ThompsonSamplingBandit(
                    experiment_id=campaign_id,
                    arms=arms
                )

        return self.active_bandits[campaign_id]
    
    def _build_strategy(
        self,
        action: str,
        confidence: float,
        platform: str,
        persona: str,
        budget: float
    ) -> Dict[str, Any]:
        """Build complete strategy from action"""
        
        strategies = self._load_strategy_templates()
        
        base_strategy = strategies.get(action, strategies.get('hook_transform', list(strategies.values())[0]))
        
        persona_customization = self._load_persona_customizations()
        
        customization = persona_customization.get(persona, persona_customization['decision_maker'])
        
        strategy = {
            'strategy_name': base_strategy['name'],
            'action': action,
            'confidence': confidence,
            'hook': base_strategy['hook'].format(**customization),
            'angle': base_strategy['angle'],
            'cta': base_strategy['cta'],
            'tone': base_strategy['tone'],
            'platform_specific': self._get_platform_optimizations(platform),
            'budget_allocation': self._optimize_budget_allocation(budget, confidence),
            'timing': self._get_optimal_timing(platform, persona),
            'estimated_performance': {
                'ctr': confidence * 0.05,  # Estimated CTR
                'conversion_rate': confidence * 0.02,  # Estimated conversion
                'cpl': budget / max(1, confidence * 10)  # Estimated CPL
            },
            'selected_arm': action,  # Bandit arm selected
            'rationale': f"Selected {base_strategy['name']} strategy for {persona} on {platform} with {confidence:.1%} confidence based on historical performance and context"
        }

        # Add alias for backward compatibility with tests
        strategy['recommended_action'] = strategy['action']

        return strategy
    
    def _get_platform_optimizations(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific optimizations"""
        
        optimizations = {
            'linkedin': {
                'post_type': 'article',
                'hashtags': ['#B2B', '#AI', '#MarketingAutomation'],
                'best_time': '9-10 AM EST',
                'media': 'infographic'
            },
            'twitter': {
                'post_type': 'thread',
                'hashtags': ['#MarTech', '#AI'],
                'best_time': '2-3 PM EST',
                'media': 'video'
            },
            'email': {
                'subject_style': 'question',
                'preview_text': True,
                'best_day': 'Tuesday',
                'send_time': '10 AM'
            }
        }
        
        return optimizations.get(platform, {})
    
    def _optimize_budget_allocation(self, total_budget: float, confidence: float) -> Dict[str, float]:
        """Optimize budget allocation across tactics"""
        
        base_allocation = {
            'content_creation': 0.2,
            'promotion': 0.5,
            'testing': 0.2,
            'analysis': 0.1
        }
        
        if confidence > 0.8:
            base_allocation['promotion'] = 0.6
            base_allocation['testing'] = 0.1
        elif confidence < 0.5:
            base_allocation['testing'] = 0.4
            base_allocation['promotion'] = 0.3
        
        return {
            k: v * total_budget
            for k, v in base_allocation.items()
        }
    
    def _get_optimal_timing(self, platform: str, persona: str) -> Dict[str, Any]:
        """Get optimal timing for campaign. Loads from config service, falls back to built-in heuristics."""
        timing = self._load_timing_defaults()
        return timing.get(platform, {}).get(persona, {
            'days': ['Tuesday'],
            'hours': [10]
        })
    
    def _get_default_strategy(self, platform: str, persona: str) -> Dict[str, Any]:
        """Get default strategy as fallback"""
        
        return {
            'strategy_name': 'Default Safe Strategy',
            'action': 'default',
            'confidence': 0.5,
            'hook': 'Discover how AI can transform your marketing',
            'angle': 'educational',
            'cta': 'Start Free Trial',
            'tone': 'professional',
            'platform_specific': self._get_platform_optimizations(platform),
            'budget_allocation': {
                'content_creation': 0.25,
                'promotion': 0.5,
                'testing': 0.15,
                'analysis': 0.1
            },
            'timing': self._get_optimal_timing(platform, persona),
            'estimated_performance': {
                'ctr': 0.02,
                'conversion_rate': 0.01,
                'cpl': 50.0
            }
        }
    
    async def update_strategy_performance(
        self,
        campaign_id: str,
        action: str,
        reward: float,
        context: Optional[np.ndarray] = None
    ):
        """Update strategy performance with observed reward"""
        
        try:
            if campaign_id in self.active_bandits:
                bandit = self.active_bandits[campaign_id]
                bandit.update_arm(action, reward)
            
            if self.config.use_gpu and hasattr(self, 'neural_bandit') and context is not None:
                self._update_neural_bandit(context, action, reward)
            
            self.performance_history.append({
                'campaign_id': campaign_id,
                'action': action,
                'reward': reward,
                'timestamp': datetime.utcnow()
            })
            
            try:
                db = get_sync_session()
                try:
                    estimated_ctr = round(2.0 + reward * 3.0, 2)  # 2-5% range
                    estimated_conversions = max(1, int(reward * 10))
                    estimated_cpl = round(max(20, 50 - reward * 30), 2)
                    
                    perf_record = StrategyPerformance(
                        campaign_id=campaign_id,
                        action=action,
                        reward=reward,
                        context=context or {},
                        estimated_ctr=estimated_ctr,
                        estimated_conversions=estimated_conversions,
                        estimated_cpl=estimated_cpl
                    )
                    db.add(perf_record)
                    db.commit()
                    logger.debug(f"Persisted strategy performance to database: {campaign_id}/{action}")
                finally:
                    db.close()
            except Exception as db_error:
                logger.warning(f"Failed to persist strategy performance to DB (non-fatal): {db_error}")
            
            logger.info(f"Updated strategy performance for {campaign_id}: action={action}, reward={reward:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")
    
    def _update_neural_bandit(self, context: np.ndarray, action: str, reward: float):
        """Update neural bandit with GPU acceleration"""
        
        action_map = {
            'hook_a': 0, 'hook_b': 1, 'hook_c': 2, 'hook_d': 3,
            'cta_a': 4, 'cta_b': 5,
            'format_short': 6, 'format_long': 7,
            'tone_professional': 8, 'tone_casual': 9
        }
        
        action_idx = action_map.get(action, 0)
        
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(DEVICE)
        action_tensor = torch.LongTensor([action_idx]).to(DEVICE)
        reward_tensor = torch.FloatTensor([reward]).to(DEVICE)
        
        q_values = self.neural_bandit(context_tensor)
        q_value = q_values[0, action_idx]
        
        target = reward_tensor
        loss = self.criterion(q_value.unsqueeze(0), target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get strategy performance report"""
        
        if not self.performance_history:
            self._load_performance_from_db()
        
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        from collections import defaultdict
        action_stats = defaultdict(lambda: {'count': 0, 'total_reward': 0})
        
        for entry in self.performance_history:
            action = entry['action']
            action_stats[action]['count'] += 1
            action_stats[action]['total_reward'] += entry['reward']
        
        for action, stats in action_stats.items():
            stats['average_reward'] = stats['total_reward'] / stats['count']
        
        best_action = max(action_stats.items(), key=lambda x: x[1]['average_reward'])
        
        return {
            'total_decisions': len(self.performance_history),
            'unique_actions': len(action_stats),
            'action_statistics': dict(action_stats),
            'best_action': best_action[0],
            'best_action_reward': best_action[1]['average_reward'],
            'gpu_enabled': torch.cuda.is_available(),
            'device': str(DEVICE)
        }
    
    def _load_performance_from_db(self):
        """Load performance history from database for persistence across restarts"""
        try:
            db = get_sync_session()
            try:
                records = db.query(StrategyPerformance).order_by(
                    StrategyPerformance.created_at.desc()
                ).limit(1000).all()
                
                if records:
                    self.performance_history = [
                        {
                            'campaign_id': r.campaign_id,
                            'action': r.action,
                            'reward': r.reward,
                            'timestamp': r.created_at
                        }
                        for r in reversed(records)  # Oldest first
                    ]
                    logger.info(f"Loaded {len(records)} strategy performance records from database")
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to load performance history from DB: {e}")

    def _load_strategy_templates(self) -> Dict[str, Dict[str, str]]:
        """Load strategy hook templates (RL bandit arm priors) from config or defaults."""
        defaults = {
            'hook_transform': {
                'name': 'Transformation Focus',
                'hook': 'Transform your {industry} with AI',
                'angle': 'innovation',
                'cta': 'Book a Free Demo',
                'tone': 'inspirational'
            },
            'hook_problem': {
                'name': 'Problem-Solution',
                'hook': 'Stop wasting {pain_point}',
                'angle': 'efficiency',
                'cta': 'Start Free Trial',
                'tone': 'direct'
            },
            'hook_success': {
                'name': 'Success Story',
                'hook': 'How we achieved {metric} improvement',
                'angle': 'proof',
                'cta': 'Get Your Results',
                'tone': 'confident'
            },
            'hook_question': {
                'name': 'Engaging Question',
                'hook': 'What if you could {benefit}?',
                'angle': 'curiosity',
                'cta': 'Try It Free',
                'tone': 'conversational'
            }
        }
        try:
            raw = _get_config_value('STRATEGY_HOOK_TEMPLATES', None)
            if raw:
                loaded = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(loaded, dict) and loaded:
                    return loaded
        except Exception:
            pass
        return defaults

    def _load_persona_customizations(self) -> Dict[str, Dict[str, str]]:
        """Load persona customization mappings from config or defaults."""
        defaults = {
            'decision_maker': {
                'pain_point': 'budget on ineffective campaigns',
                'industry': 'marketing operations',
                'metric': '300% ROI',
                'benefit': 'triple your conversions'
            },
            'influencer': {
                'pain_point': 'time on content creation',
                'industry': 'content strategy',
                'metric': '10x content output',
                'benefit': 'automate your workflow'
            },
            'researcher': {
                'pain_point': 'hours on analysis',
                'industry': 'marketing analytics',
                'metric': '80% time savings',
                'benefit': 'predict campaign outcomes'
            },
            'technical_buyer': {
                'pain_point': 'resources on integration',
                'industry': 'martech stack',
                'metric': '99.9% uptime',
                'benefit': 'integrate seamlessly'
            }
        }
        try:
            raw = _get_config_value('STRATEGY_PERSONA_CUSTOMIZATIONS', None)
            if raw:
                loaded = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(loaded, dict) and loaded:
                    return loaded
        except Exception:
            pass
        return defaults

    def _load_timing_defaults(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load optimal timing heuristics (initial priors) from config or defaults."""
        defaults = {
            'linkedin': {
                'decision_maker': {'days': ['Tuesday', 'Thursday'], 'hours': [9, 10, 14]},
                'influencer': {'days': ['Monday', 'Wednesday'], 'hours': [8, 11, 15]},
                'researcher': {'days': ['Wednesday', 'Thursday'], 'hours': [10, 14, 16]},
                'technical_buyer': {'days': ['Tuesday', 'Friday'], 'hours': [11, 15]}
            },
            'twitter': {
                'decision_maker': {'days': ['Monday', 'Thursday'], 'hours': [12, 14, 17]},
                'influencer': {'days': ['Tuesday', 'Friday'], 'hours': [11, 15, 18]},
                'researcher': {'days': ['Wednesday'], 'hours': [13, 16]},
                'technical_buyer': {'days': ['Thursday'], 'hours': [14, 17]}
            },
            'email': {
                'decision_maker': {'days': ['Tuesday'], 'hours': [10]},
                'influencer': {'days': ['Wednesday'], 'hours': [11]},
                'researcher': {'days': ['Thursday'], 'hours': [9]},
                'technical_buyer': {'days': ['Tuesday'], 'hours': [14]}
            }
        }
        try:
            raw = _get_config_value('OPTIMAL_TIMING_DEFAULTS', None)
            if raw:
                loaded = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(loaded, dict) and loaded:
                    return loaded
        except Exception:
            pass
        return defaults