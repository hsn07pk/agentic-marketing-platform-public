import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from langgraph.graph import StateGraph, END
from uuid import UUID
import redis.asyncio as redis

from .workflow_states import (
    WorkflowState,
    WorkflowStep,
    create_initial_state,
    update_state,
    should_review_human,
    can_deploy,
    get_runtime_config
)
from .workflow_events import WorkflowEventLogger
from ...data_layer.database.models import WorkflowEventType, AlertSeverity, CampaignStatus
from ..agents.content_generator import ContentGeneratorAgent
from ..agents.safety_validator import SafetyValidatorAgent
from ..agents.strategy_optimizer import StrategyOptimizerAgent
from ..agents.market_scraper import MarketScraperAgent
from ...automation_layer.deployer import CampaignDeployer
from ...data_layer.repositories.content_repo import ContentRepository
from ...data_layer.repositories.campaign_repo import CampaignRepository
from ...data_layer.database.connection import async_session_maker, get_sync_session
from ...governance.hitl_queue import HITLQueueManager
from ...cost_control.budget_manager import BudgetManager
from ...simulation.environment import MarketingEnvironment, SimulationConfig
from ...simulation.agents.persona_factory import PersonaFactory
from ...ai_layer.marl.ope_gating import MARLGatekeeper
from ..memory.episodic_memory import EpisodicMemoryStore, create_memory_from_task
from ...config.settings import settings
from ...config.configuration_service import ConfigurationService

logger = logging.getLogger(__name__)

class MarketingOrchestrator:
    """
    LangGraph-based supervisor for marketing agent workflow
    Manages state transitions and agent coordination

    Implements OODA-G Loop per Research Plan Section 3:
    - OBSERVE: MarketScraper gathers market intelligence
    - ORIENT: StrategyOptimizer analyzes context
    - DECIDE: ContentGenerator creates content
    - ACT: Deployer publishes to platform
    - GOVERN: SafetyValidator + HITL + BudgetManager + MARL Gating
    """

    def __init__(self):
        self.market_scraper = MarketScraperAgent()

        self.strategy_optimizer = StrategyOptimizerAgent()
        self.content_generator = ContentGeneratorAgent()

        self.safety_validator = SafetyValidatorAgent()

        self.deployer = CampaignDeployer()

        from ..orchestration.workflow_states import get_runtime_config
        enable_marl = get_runtime_config('ENABLE_MARL', False)
        self.marl_gatekeeper = MARLGatekeeper() if enable_marl else None

        self.hitl_queue = None
        self.budget_manager = None

        # Check MLflow availability once at startup
        self._mlflow_available = self._check_mlflow_available()

        self.memory_stores = {
            'content_generator': EpisodicMemoryStore('content_generator'),
            'strategy_optimizer': EpisodicMemoryStore('strategy_optimizer'),
            'safety_validator': EpisodicMemoryStore('safety_validator'),
            'market_scraper': EpisodicMemoryStore('market_scraper'),
        }

        self.app = self._build_graph()
    
    @staticmethod
    def _check_mlflow_available() -> bool:
        """Check once if MLflow tracking server is reachable."""
        try:
            import mlflow
            tracking_uri = mlflow.get_tracking_uri()
            # If it's a remote server, try to connect
            if tracking_uri and tracking_uri.startswith(("http://", "https://")):
                import urllib.request
                urllib.request.urlopen(tracking_uri, timeout=3)
            # If it's a local file store, check if it exists
            elif tracking_uri:
                from pathlib import Path
                mlruns_path = Path(tracking_uri.replace("file://", ""))
                if not mlruns_path.exists():
                    return False
            logger.info(f"MLflow tracking available at: {tracking_uri}")
            return True
        except Exception:
            logger.info("MLflow tracking server not configured — using file-based model loading")
            return False

        self.app = self._build_graph()
    
    def _get_config_value(self, key: str, default: any = None) -> any:
        """
        Get configuration value from the ConfigurationService (database).
        Uses sync session since ConfigurationService is synchronous.
        Falls back to settings.py default if database value not found.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value from database or default
        """
        try:
            sync_db = get_sync_session()
            try:
                config_service = ConfigurationService(sync_db)
                value = config_service.get_value(key)
                if value is not None:
                    return value
            finally:
                sync_db.close()
        except Exception as e:
            logger.warning(f"Failed to get config '{key}' from database: {e}, using fallback")
        
        return settings.get(key, default)
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow with OODA-G loop (Research Plan Section 3)

        OODA-G Workflow:
        - OBSERVE: Market Observation (MarketScraper) - gathers market intelligence
        - ORIENT: Strategy Optimization (StrategyOptimizer) - analyzes context
        - DECIDE: Content Generation (ContentGenerator) - creates content
        - GOVERN: Safety → HITL → Cost → Simulation → [MARL Gating]
        - ACT: Deployment (Deployer) - publishes to platform

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(WorkflowState)


        workflow.add_node("market_observation", self._observe_market_node)

        workflow.add_node("strategy_optimization", self._optimize_strategy_node)

        workflow.add_node("content_generation", self._generate_content_node)

        workflow.add_node("safety_validation", self._validate_safety_node)
        workflow.add_node("cost_check", self._check_cost_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("simulation", self._simulate_campaign_node)
        workflow.add_node("golden_test_gate", self._golden_test_gate_node)

        if self.marl_gatekeeper:
            workflow.add_node("marl_gating", self._marl_gating_node)
            workflow.add_node("canary_deployment", self._canary_deploy_node)

        workflow.add_node("deployment", self._deploy_node)

        workflow.set_entry_point("market_observation")

        workflow.add_edge("market_observation", "strategy_optimization")

        workflow.add_edge("strategy_optimization", "content_generation")
        workflow.add_edge("content_generation", "safety_validation")

        workflow.add_conditional_edges(
            "safety_validation",
            self._route_after_safety,
            {
                "human_review": "human_review",
                "cost_check": "cost_check",
                "regenerate": "content_generation"
            }
        )

        workflow.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "approved": "cost_check",
                "rejected": "content_generation",
                "pending": END
            }
        )

        workflow.add_conditional_edges(
            "cost_check",
            self._route_after_cost_check,
            {
                "simulate": "simulation",
                "budget_exceeded": END
            }
        )

        if self.marl_gatekeeper:
            workflow.add_conditional_edges(
                "simulation",
                self._route_after_simulation,
                {
                    "marl_gating": "marl_gating",
                    "regenerate": "content_generation",
                    "failed": END
                }
            )

            workflow.add_conditional_edges(
                "marl_gating",
                self._route_after_marl_gating,
                {
                    "deploy": "golden_test_gate",
                    "canary_deploy": "golden_test_gate",
                    "rejected": END
                }
            )
        else:
            workflow.add_conditional_edges(
                "simulation",
                self._route_after_simulation,
                {
                    "deploy": "golden_test_gate",
                    "regenerate": "content_generation",
                    "failed": END
                }
            )

        workflow.add_conditional_edges(
            "golden_test_gate",
            self._route_after_golden_test,
            {
                "deploy": "deployment",
                "canary_deploy": "canary_deployment" if self.marl_gatekeeper else "deployment",
                "blocked": END
            }
        )

        workflow.add_edge("deployment", END)
        
        if self.marl_gatekeeper:
            workflow.add_edge("canary_deployment", "deployment")

        return workflow.compile()

    async def _observe_market_node(self, state: WorkflowState) -> WorkflowState:
        """
        OBSERVE Phase - Market Intelligence Gathering (Research Plan Section 3)

        Uses MarketScraper to gather:
        - Competitive landscape analysis
        - Current market trends
        - Differentiation opportunities
        - Content patterns that resonate

        This data informs the Strategy Optimization phase.
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')

        try:
            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("market_observation")

            logger.info(
                "Workflow node: market_observation (OBSERVE phase)",
                extra={
                    "event": "workflow_node_entered",
                    "node": "market_observation",
                    "campaign_id": campaign_id,
                    "workflow_step": state.get('current_step'),
                    "ooda_phase": "OBSERVE"
                }
            )

            metadata = state.get('metadata', {})
            keywords = metadata.get('keywords', metadata.get('target_keywords', []))
            platform = metadata.get('platform', 'linkedin')

            if not keywords:
                persona = metadata.get('target_persona', 'decision_maker')
                keywords = ['employee experience', 'people analytics', 'QWL', 'workplace wellbeing', 'HR technology']

            market_insights = {
                'competitive_analysis': None,
                'content_patterns': None,
                'differentiation_opportunities': [],
                'scraping_enabled': self.market_scraper.enabled
            }

            try:
                competitive_analysis = await self.market_scraper.analyze_competitive_landscape()
                market_insights['competitive_analysis'] = competitive_analysis

                differentiation_opps = self.market_scraper.get_differentiation_opportunities()
                market_insights['differentiation_opportunities'] = differentiation_opps

                if self.market_scraper.enabled and keywords:
                    inspiration = await self.market_scraper.get_inspiration_for_campaign(
                        keywords=keywords[:5],
                        limit=10
                    )
                    if inspiration.get('success'):
                        market_insights['content_patterns'] = inspiration.get('insights')

                logger.info(
                    "Market observation completed",
                    extra={
                        "event": "market_observation_completed",
                        "campaign_id": campaign_id,
                        "competitors_analyzed": competitive_analysis.get('total_competitors', 0) if competitive_analysis else 0,
                        "differentiation_opportunities": len(differentiation_opps),
                        "scraping_enabled": self.market_scraper.enabled
                    }
                )

            except Exception as e:
                logger.warning(f"Market observation partial failure (non-blocking): {e}")
                market_insights['error'] = str(e)


            try:
                memory_store = self.memory_stores.get('market_scraper')
                if memory_store:
                    memory = create_memory_from_task(
                        agent_name='market_scraper',
                        task_id=f"{campaign_id}_observe",
                        task_description=f"Market observation for campaign {campaign_id}",
                        actions=['analyze_competitive_landscape', 'get_differentiation_opportunities'],
                        result={
                            'success': bool(market_insights.get('competitive_analysis')),
                            'cost': 0.0,
                            'quality_score': 1.0 if market_insights.get('differentiation_opportunities') else 0.5
                        }
                    )
                    await memory_store.store_memory(memory)
            except Exception as mem_error:
                logger.warning(f"Failed to store market observation memory: {mem_error}")

            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                comp_analysis = market_insights.get('competitive_analysis') or {}
                diff_opps = market_insights.get('differentiation_opportunities', [])
                event_details = {
                    "competitors_analyzed": comp_analysis.get('total_competitors', 0),
                    "differentiation_opportunities_count": len(diff_opps),
                    "differentiation_opportunities": diff_opps,
                    "duration": round(duration, 2),
                    "scraping_enabled": market_insights.get('scraping_enabled', False),
                }
                if comp_analysis.get('competitor_details'):
                    event_details["competitor_details"] = comp_analysis['competitor_details']
                if comp_analysis.get('market_positions'):
                    event_details["market_positions"] = comp_analysis['market_positions']
                if comp_analysis.get('top_common_strengths'):
                    event_details["top_common_strengths"] = comp_analysis['top_common_strengths']
                if comp_analysis.get('top_common_weaknesses'):
                    event_details["top_common_weaknesses"] = comp_analysis['top_common_weaknesses']
                if comp_analysis.get('aggregated_strengths'):
                    event_details["aggregated_strengths"] = comp_analysis['aggregated_strengths']
                if comp_analysis.get('aggregated_weaknesses'):
                    event_details["aggregated_weaknesses"] = comp_analysis['aggregated_weaknesses']
                content_patterns = market_insights.get('content_patterns')
                if content_patterns:
                    event_details["content_patterns"] = content_patterns

                await event_logger.log_event(
                    event_type=WorkflowEventType.NODE_COMPLETED,
                    title="Market Observation Complete",
                    message=f"Gathered market intelligence: {len(diff_opps)} differentiation opportunities identified across {comp_analysis.get('total_competitors', 0)} competitors. Proceeding to strategy optimization.",
                    severity=AlertSeverity.INFO,
                    workflow_node="market_observation",
                    details=event_details
                )

            return update_state(state, {
                'market_insights': market_insights,
                'current_step': WorkflowStep.STRATEGY_OPTIMIZATION,
                'messages': state['messages'] + [{
                    'role': 'system',
                    'content': f'Market observation complete: {len(market_insights.get("differentiation_opportunities", []))} differentiation opportunities found',
                    'timestamp': str(datetime.now())
                }]
            })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Market observation node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "market_observation",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )

            return update_state(state, {
                'market_insights': {'error': str(e), 'differentiation_opportunities': []},
                'current_step': WorkflowStep.STRATEGY_OPTIMIZATION
            })

    async def _optimize_strategy_node(self, state: WorkflowState) -> WorkflowState:
        """
        ORIENT Phase - Strategy optimization using bandits (Research Plan Section 3)

        Uses market insights from OBSERVE phase to inform strategy selection.
        Retrieves relevant past experiences from episodic memory.
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')

        try:
            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("strategy_optimization")

            logger.info(
                "Workflow node: strategy_optimization (ORIENT phase)",
                extra={
                    "event": "workflow_node_entered",
                    "node": "strategy_optimization",
                    "campaign_id": campaign_id,
                    "workflow_step": state.get('current_step'),
                    "ooda_phase": "ORIENT"
                }
            )

            metadata = state.get('metadata', {})
            market_insights = state.get('market_insights', {})


            memory_context = ""
            try:
                memory_store = self.memory_stores.get('strategy_optimizer')
                if memory_store:
                    task_description = f"Strategy optimization for {metadata.get('platform')} {metadata.get('target_persona')} campaign"
                    relevant_memories = await memory_store.retrieve_relevant_memories(
                        query=task_description,
                        k=3,
                        outcome_filter=None
                    )
                    if relevant_memories:
                        memory_context = await memory_store.format_memories_for_prompt(relevant_memories)
                        logger.info(f"Retrieved {len(relevant_memories)} memories for strategy optimization")
            except Exception as mem_error:
                logger.warning(f"Failed to retrieve strategy memories: {mem_error}")

            context = metadata.get('context', '')
            if market_insights:
                diff_opps = market_insights.get('differentiation_opportunities', [])
                if diff_opps:
                    context += f"\n\nDifferentiation Opportunities from Market Analysis:\n"
                    for i, opp in enumerate(diff_opps[:5], 1):
                        context += f"{i}. {opp}\n"

                if market_insights.get('content_patterns'):
                    patterns = market_insights['content_patterns']
                    if patterns.get('top_hooks'):
                        context += f"\n\nTop performing hooks in market:\n"
                        for hook in patterns['top_hooks'][:3]:
                            context += f"- {hook.get('text', '')}\n"

            if memory_context:
                context += f"\n\n{memory_context}"

            strategy = await self.strategy_optimizer.get_optimal_strategy(
                campaign_id=campaign_id,
                platform=metadata.get('platform', 'linkedin'),
                target_persona=metadata.get('target_persona', 'decision_maker'),
                budget=metadata.get('budget', 1000.0),
                context=context
            )

            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_event(
                    event_type=WorkflowEventType.NODE_COMPLETED,
                    title="Strategy Optimized",
                    message=f"Selected '{strategy.get('strategy_name')}' strategy with {strategy.get('confidence', 0):.0%} confidence. Proceeding to content generation.",
                    severity=AlertSeverity.INFO,
                    workflow_node="strategy_optimization",
                    details={
                        "strategy_name": strategy.get('strategy_name'),
                        "action": strategy.get('action'),
                        "confidence": strategy.get('confidence'),
                        "duration": round(duration, 2)
                    }
                )

            logger.info(
                "Strategy optimization node completed",
                extra={
                    "event": "workflow_node_completed",
                    "node": "strategy_optimization",
                    "campaign_id": campaign_id,
                    "strategy_name": strategy.get('strategy_name'),
                    "action": strategy.get('action'),
                    "confidence": strategy.get('confidence'),
                    "duration_seconds": round(duration, 2),
                    "next_step": "content_generation"
                }
            )


            try:
                memory_store = self.memory_stores.get('strategy_optimizer')
                if memory_store:
                    memory = create_memory_from_task(
                        agent_name='strategy_optimizer',
                        task_id=f"{campaign_id}_strategy",
                        task_description=f"Strategy optimization for {metadata.get('platform')} {metadata.get('target_persona')} campaign",
                        actions=[f"selected_strategy:{strategy.get('strategy_name')}", f"action:{strategy.get('action')}"],
                        result={
                            'success': True,
                            'cost': 0.0,
                            'duration': duration,
                            'quality_score': strategy.get('confidence', 0.5)
                        }
                    )
                    await memory_store.store_memory(memory)
            except Exception as mem_error:
                logger.warning(f"Failed to store strategy memory: {mem_error}")

            return update_state(state, {
                'strategy': strategy,
                'current_step': WorkflowStep.CONTENT_GENERATION,
                'messages': state['messages'] + [{
                    'role': 'system',
                    'content': f'Strategy selected: {strategy.get("strategy_name")} (confidence: {strategy.get("confidence"):.2f})',
                    'timestamp': str(datetime.now())
                }]
            })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_event(
                    event_type=WorkflowEventType.ERROR_OCCURRED,
                    title="Strategy Optimization Error",
                    message=f"Strategy optimization failed: {str(e)}. Falling back to default strategy.",
                    severity=AlertSeverity.WARNING,
                    workflow_node="strategy_optimization",
                    details={"error": str(e), "error_type": type(e).__name__}
                )

            logger.error(
                "Strategy optimization node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "strategy_optimization",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            default_strategy = {
                'strategy_name': 'Default Strategy',
                'action': 'hook_transform',
                'confidence': 0.5,
                'hook': 'Transform your business',
                'cta': 'Learn More',
                'tone': 'professional'
            }
            return update_state(state, {
                'strategy': default_strategy,
                'current_step': WorkflowStep.CONTENT_GENERATION
            })

    async def _generate_content_node(self, state: WorkflowState) -> WorkflowState:
        """
        DECIDE Phase - Content generation (Research Plan Section 3)

        Uses strategy from ORIENT phase and market insights from OBSERVE phase.
        Retrieves relevant past content experiences from episodic memory.
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')
        retry_count = state.get('retry_count', 0)

        try:
            logger.info(
                "Workflow node: content_generation (DECIDE phase)",
                extra={
                    "event": "workflow_node_entered",
                    "node": "content_generation",
                    "campaign_id": campaign_id,
                    "retry_count": retry_count,
                    "workflow_step": state.get('current_step'),
                    "ooda_phase": "DECIDE"
                }
            )

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("content_generation")

            metadata = state.get('metadata', {})
            strategy = state.get('strategy', {})
            market_insights = state.get('market_insights', {})


            memory_context = ""
            try:
                memory_store = self.memory_stores.get('content_generator')
                if memory_store:
                    task_description = f"Generate {metadata.get('platform')} content for {metadata.get('target_persona')} persona"
                    relevant_memories = await memory_store.retrieve_relevant_memories(
                        query=task_description,
                        k=3,
                        outcome_filter=None
                    )
                    if relevant_memories:
                        memory_context = await memory_store.format_memories_for_prompt(relevant_memories)
                        logger.info(f"Retrieved {len(relevant_memories)} memories for content generation")
            except Exception as mem_error:
                logger.warning(f"Failed to retrieve content memories: {mem_error}")

            enhanced_config = {
                'goal': metadata.get('goal', 'lead_generation'),
                'budget': metadata.get('budget', 1000.0),
                'campaign_id': campaign_id,
                'strategy': strategy,
                'market_insights': market_insights,
                'content_patterns': market_insights.get('content_patterns'),
                'memory_context': memory_context
            }

            content, content_metadata = await self.content_generator.generate_content(
                platform=metadata.get('platform', 'linkedin'),
                persona=metadata.get('target_persona', 'decision_maker'),
                campaign_config=enhanced_config,
                previous_feedback=metadata.get('previous_feedback')
            )

            async with async_session_maker() as session:
                content_repo = ContentRepository(session)

                if hasattr(content, 'headline'):
                    content_dict = {
                        "campaign_id": UUID(campaign_id),
                        "headline": content.headline,
                        "body": content.body,
                        "cta": content.cta if hasattr(content, 'cta') else '',
                        "status": "generated",
                        "content_type": content.content_type if hasattr(content, 'content_type') else f"{metadata.get('platform')}_ad",
                        "generated_by": content.generated_by if hasattr(content, 'generated_by') else 'content_generator',
                        "model_used": content.model_used if hasattr(content, 'model_used') else '',
                        "claims_used": content.claims_used if hasattr(content, 'claims_used') else [],
                        "platform_specific": {"platform": metadata.get('platform')}
                    }
                else:
                    content_dict = {
                        "campaign_id": UUID(campaign_id),
                        "headline": content.get('headline'),
                        "body": content.get('body'),
                        "status": "generated",
                        "platform_specific": {"platform": metadata.get('platform')}
                    }

                content_record = await content_repo.create(content_dict)
                content_id = str(content_record.id)

                event_logger = WorkflowEventLogger(session, campaign_id)
                strategy_name = strategy.get('name', 'default')
                await event_logger.log_content_generated(content_id, strategy_name)

            duration = (datetime.now() - start_time).total_seconds()

            headline_log = ""
            if hasattr(content, 'headline'):
                headline_log = (content.headline or '')[:50]
            elif isinstance(content, dict):
                headline_log = content.get('headline', '')[:50]

            logger.info(
                "Content generation node completed",
                extra={
                    "event": "workflow_node_completed",
                    "node": "content_generation",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "duration_seconds": round(duration, 2),
                    "next_step": "safety_validation",
                    "retry_count": retry_count,
                    "headline": headline_log
                }
            )

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_completed("content_generation")


            try:
                memory_store = self.memory_stores.get('content_generator')
                if memory_store:
                    full_headline = ""
                    full_body = ""
                    content_type = ""
                    
                    if hasattr(content, 'headline'):
                        full_headline = content.headline or ''
                        full_body = content.body or '' if hasattr(content, 'body') else ''
                        content_type = content.content_type if hasattr(content, 'content_type') else ''
                    elif isinstance(content, dict):
                        full_headline = content.get('headline', '') or ''
                        full_body = content.get('body', '') or ''
                        content_type = content.get('content_type', '') or ''
                    
                    platform = metadata.get('platform') or content_type or 'linkedin'
                    persona = metadata.get('target_persona') or 'decision_maker'
                    
                    memory = create_memory_from_task(
                        agent_name='content_generator',
                        task_id=f"{campaign_id}_content_{content_id}",
                        task_description=f"Generated {platform}_ad for campaign {campaign_id}",
                        actions=[
                            f"Generated headline: {full_headline or 'N/A'}",
                            f"Body ({len(full_body)} chars): {full_body or 'N/A'}"
                        ],
                        result={
                            'success': True,
                            'cost': content_metadata.get('cost', 0.0) if content_metadata else 0.0,
                            'duration': duration,
                            'quality_score': 0.8
                        }
                    )
                    await memory_store.store_memory(memory)
            except Exception as mem_error:
                logger.warning(f"Failed to store content generation memory: {mem_error}")

            return update_state(state, {
                'content_id': content_id,
                'current_step': WorkflowStep.SAFETY_VALIDATION,
                'messages': state['messages'] + [{
                    'role': 'system',
                    'content': f'Content generated: {content_id}',
                    'timestamp': str(datetime.now())
                }]
            })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Content generation node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "content_generation",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2),
                    "retry_count": retry_count
                },
                exc_info=True
            )


            try:
                memory_store = self.memory_stores.get('content_generator')
                if memory_store:
                    memory = create_memory_from_task(
                        agent_name='content_generator',
                        task_id=f"{campaign_id}_content_failed",
                        task_description=f"Failed to generate {metadata.get('platform')} content for {metadata.get('target_persona')} persona",
                        actions=[f"strategy:{strategy.get('strategy_name', 'default') if strategy else 'none'}"],
                        result={
                            'success': False,
                            'error': str(e),
                            'duration': duration
                        }
                    )
                    await memory_store.store_memory(memory)
            except Exception as mem_error:
                logger.warning(f"Failed to store content generation failure memory: {mem_error}")

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_error(str(e), "content_generation")

            return update_state(state, {'error': str(e)})
    
    async def _validate_safety_node(self, state: WorkflowState) -> WorkflowState:
        """
        GOVERN Phase - Safety validation (Research Plan Section 3)

        Validates content safety using LLM-as-judge with memory of past validation patterns.
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')
        content_id = state.get('content_id')

        if not content_id:
            logger.error(
                "Safety validation skipped - no content_id from previous step",
                extra={
                    "event": "workflow_node_error",
                    "node": "safety_validation",
                    "campaign_id": campaign_id,
                    "content_id": None,
                    "error": "content_id is None"
                }
            )
            return update_state(state, {
                'safety_score': None,
                'error': 'Content generation failed - no content to validate'
            })

        try:
            logger.info(
                "Workflow node: safety_validation (GOVERN phase)",
                extra={
                    "event": "workflow_node_entered",
                    "node": "safety_validation",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "workflow_step": state.get('current_step'),
                    "ooda_phase": "GOVERN"
                }
            )

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("safety_validation")

            metadata = state.get('metadata', {})

            async with async_session_maker() as session:
                content_repo = ContentRepository(session)
                content = await content_repo.get_by_id(content_id)

                if not content:
                    logger.error(
                        "Content not found for safety validation",
                        extra={
                            "event": "workflow_node_error",
                            "node": "safety_validation",
                            "campaign_id": campaign_id,
                            "content_id": content_id,
                            "error": "Content not found in database"
                        }
                    )
                    return update_state(state, {'error': 'Content not found'})

                content_dict = {
                    'headline': content.headline,
                    'body': content.body
                }

                # Concatenate all content fields for safety validation
                # so claim citations in headline/CTA are also detected
                full_content = ' '.join(filter(None, [
                    content.headline, content.body, content.cta
                ]))

                safety_result = await self.safety_validator.validate_content(
                    content_text=full_content,
                    headline=content_dict.get('headline'),
                    claims_used=content.claims_used if hasattr(content, 'claims_used') else [],
                    platform=metadata.get('platform', 'linkedin'),
                    context={
                        'campaign_id': campaign_id,
                        'persona': metadata.get('target_persona', 'decision_maker')
                    }
                )

                await content_repo.update(
                    content_id,
                    {
                        'safety_score': safety_result['overall_score'],
                        'toxicity_score': safety_result.get('toxicity_score', 0.0),
                        'factuality_score': safety_result.get('factuality_score', 1.0),
                        'brand_alignment_score': safety_result.get('brand_score', 1.0)
                    }
                )

            duration = (datetime.now() - start_time).total_seconds()
            safety_score = safety_result['overall_score']

            logger.info(
                "Safety validation node completed",
                extra={
                    "event": "workflow_node_completed",
                    "node": "safety_validation",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "safety_score": round(safety_score, 3),
                    "passed": safety_result.get('passed', True),
                    "requires_review": safety_result.get('requires_review', False),
                    "duration_seconds": round(duration, 2),
                    "toxicity_score": round(safety_result.get('toxicity_score', 0), 3),
                    "factuality_score": round(safety_result.get('factuality_score', 1), 3),
                    "brand_score": round(safety_result.get('brand_score', 1), 3)
                }
            )

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)

                passed = safety_result.get('passed', False)
                requires_review = safety_result.get('requires_review', False)

                extra_scores = {
                    'toxicity_score': round(safety_result.get('toxicity_score', 0), 3),
                    'factuality_score': round(safety_result.get('factuality_score', 1), 3),
                    'brand_alignment_score': round(safety_result.get('brand_score', 1), 3)
                }

                if passed and not requires_review:
                    await event_logger.log_safety_check_passed(content_id, safety_score, **extra_scores)
                else:
                    issues = safety_result.get('issues', ['Safety score below threshold'])
                    await event_logger.log_safety_check_failed(content_id, safety_score, issues, **extra_scores)

                await event_logger.log_node_completed("safety_validation")


            try:
                memory_store = self.memory_stores.get('safety_validator')
                if memory_store:
                    memory = create_memory_from_task(
                        agent_name='safety_validator',
                        task_id=f"{campaign_id}_safety_{content_id}",
                        task_description=f"Safety validation for {metadata.get('platform')} content",
                        actions=[
                            f"toxicity_score:{safety_result.get('toxicity_score', 0):.2f}",
                            f"factuality_score:{safety_result.get('factuality_score', 1):.2f}",
                            f"brand_score:{safety_result.get('brand_score', 1):.2f}"
                        ],
                        result={
                            'success': passed and not requires_review,
                            'cost': 0.0,
                            'duration': duration,
                            'quality_score': safety_score
                        }
                    )
                    await memory_store.store_memory(memory)
            except Exception as mem_error:
                logger.warning(f"Failed to store safety validation memory: {mem_error}")

            return update_state(state, {
                'safety_score': safety_score,
                'safety_passed': passed,
                'current_step': WorkflowStep.HUMAN_REVIEW if not passed or requires_review else WorkflowStep.COST_CHECK,
                'messages': state['messages'] + [{
                    'role': 'system',
                    'content': f'Safety score: {safety_score:.2f}, passed: {passed}',
                    'timestamp': str(datetime.now())
                }]
            })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Safety validation node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "safety_validation",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return update_state(state, {'error': str(e)})
    
    async def _check_cost_node(self, state: WorkflowState) -> WorkflowState:
        """Cost check node"""
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')

        try:
            async with async_session_maker() as db_session:
                event_logger = WorkflowEventLogger(db_session, campaign_id)
                await event_logger.log_node_started("cost_check")
            
            campaign_cost = state.get('cost_accumulated', 0.0)
            budget = state['metadata'].get('budget', float('inf'))
            budget_remaining = budget - campaign_cost
            budget_used_pct = (campaign_cost / budget * 100) if budget != float('inf') else 0

            logger.info(
                "Workflow node: cost_check",
                extra={
                    "event": "workflow_node_entered",
                    "node": "cost_check",
                    "campaign_id": campaign_id,
                    "campaign_cost": round(campaign_cost, 2),
                    "budget": round(budget, 2) if budget != float('inf') else 'unlimited',
                    "budget_remaining": round(budget_remaining, 2),
                    "budget_used_pct": round(budget_used_pct, 1),
                    "workflow_step": state.get('current_step')
                }
            )

            async with async_session_maker() as session:
                redis_client = await redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )

                try:
                    budget_manager = BudgetManager(session, redis_client)

                    can_proceed = await budget_manager.check_budget(
                        campaign_id=campaign_id,
                        estimated_cost=0.0
                    )

                    daily_summary = await budget_manager.get_cost_summary(campaign_id=campaign_id, days=1)
                    daily_cost = daily_summary.get('total_cost', 0.0)
                    daily_limit = settings.MAX_DAILY_API_COST

                    if not can_proceed or daily_cost >= daily_limit:
                        budget_error = 'Campaign budget exceeded' if not can_proceed else 'Daily budget limit exceeded'

                        logger.warning(
                            "Budget exceeded - halting workflow",
                            extra={
                                "event": "workflow_budget_exceeded",
                                "node": "cost_check",
                                "campaign_id": campaign_id,
                                "campaign_cost": round(campaign_cost, 2),
                                "budget": round(budget, 2),
                                "daily_cost": round(daily_cost, 2),
                                "daily_limit": round(daily_limit, 2),
                                "overage": round(campaign_cost - budget, 2) if campaign_cost > budget else 0,
                                "reason": budget_error,
                                "next_step": "END"
                            }
                        )

                        event_logger = WorkflowEventLogger(session, campaign_id)
                        await event_logger.log_budget_exceeded(daily_cost, daily_limit)

                        return update_state(state, {
                            'error': f'Budget exceeded: {budget_error}',
                            'current_step': WorkflowStep.END
                        })

                    budget_utilization = (campaign_cost / budget * 100) if budget > 0 else 0
                    daily_utilization = (daily_cost / daily_limit * 100) if daily_limit > 0 else 0

                    logger.info(
                        "Budget check passed",
                        extra={
                            "event": "budget_check_passed",
                            "campaign_id": campaign_id,
                            "campaign_cost": round(campaign_cost, 2),
                            "campaign_budget": round(budget, 2),
                            "daily_cost": round(daily_cost, 2),
                            "daily_limit": round(daily_limit, 2),
                            "budget_utilization": round(budget_utilization, 1)
                        }
                    )

                    event_logger = WorkflowEventLogger(session, campaign_id)

                    if daily_utilization > 80:
                        await event_logger.log_budget_warning(daily_cost, daily_limit, daily_utilization)

                    await event_logger.log_cost_check_passed(daily_cost, daily_limit)

                finally:
                    await redis_client.close()

            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as db_session:
                event_logger = WorkflowEventLogger(db_session, campaign_id)
                await event_logger.log_node_completed("cost_check", details={
                    "campaign_cost": round(campaign_cost, 2),
                    "budget_remaining": round(budget_remaining, 2),
                    "budget_used_pct": round(budget_used_pct, 1),
                    "duration": round(duration, 2)
                })

            logger.info(
                "Cost check node completed",
                extra={
                    "event": "workflow_node_completed",
                    "node": "cost_check",
                    "campaign_id": campaign_id,
                    "campaign_cost": round(campaign_cost, 2),
                    "budget_remaining": round(budget_remaining, 2),
                    "budget_used_pct": round(budget_used_pct, 1),
                    "duration_seconds": round(duration, 2),
                    "next_step": "deployment",
                    "status": "approved"
                }
            )

            return update_state(state, {
                'cost_accumulated': campaign_cost,
                'current_step': WorkflowStep.DEPLOYMENT
            })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Cost check node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "cost_check",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return update_state(state, {'error': str(e)})
    
    async def _human_review_node(self, state: WorkflowState) -> WorkflowState:
        """Human review node"""
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')
        content_id = state.get('content_id')

        try:
            logger.info(
                "Workflow node: human_review",
                extra={
                    "event": "workflow_node_entered",
                    "node": "human_review",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "safety_score": state.get('safety_score'),
                    "workflow_step": state.get('current_step')
                }
            )

            try:
                from ...data_layer.database.models import HITLQueue
                from uuid import uuid4

                safety_score = state.get('safety_score', 0)

                # Priority tiers by safety score: <0.7=CRITICAL(10), <0.8=HIGH(7), <0.9=MEDIUM(5), ≥0.9=LOW(3)
                if safety_score < 0.7:
                    priority = 10
                    priority_label = "CRITICAL"
                elif safety_score < 0.8:
                    priority = 7
                    priority_label = "HIGH"
                elif safety_score < 0.9:
                    priority = 5
                    priority_label = "MEDIUM"
                else:
                    priority = 3
                    priority_label = "LOW"

                async with async_session_maker() as db_session:
                    queue_item = HITLQueue(
                        id=uuid4(),
                        content_id=UUID(content_id),
                        priority=priority,
                        reason=f"Human review required (safety_score: {safety_score:.2f}, priority: {priority_label})",
                        status="pending",
                        created_at=datetime.utcnow()
                    )

                    db_session.add(queue_item)
                    await db_session.commit()

                    logger.info(f"✅ Added content {content_id} to HITL queue (queue_id: {queue_item.id})")

                    event_logger = WorkflowEventLogger(db_session, campaign_id)
                    safety_score = state.get('safety_score', 0)
                    reason = f"Safety score ({safety_score:.2f}) below threshold"
                    await event_logger.log_hitl_queue_added(content_id, safety_score, reason)

            except Exception as e:
                logger.error(f"Failed to add to HITL queue: {e}")

            review_status = 'pending'

            duration = (datetime.now() - start_time).total_seconds()

            if review_status == 'approved':
                logger.info(
                    "Human review approved",
                    extra={
                        "event": "workflow_node_completed",
                        "node": "human_review",
                        "campaign_id": campaign_id,
                        "content_id": content_id,
                        "review_status": "approved",
                        "duration_seconds": round(duration, 2),
                        "next_step": "cost_check"
                    }
                )
                return update_state(state, {
                    'requires_human_review': False,
                    'human_feedback': 'approved',
                    'current_step': WorkflowStep.COST_CHECK
                })
            elif review_status == 'rejected':
                logger.info(
                    "Human review rejected - regenerating content",
                    extra={
                        "event": "workflow_node_completed",
                        "node": "human_review",
                        "campaign_id": campaign_id,
                        "content_id": content_id,
                        "review_status": "rejected",
                        "duration_seconds": round(duration, 2),
                        "next_step": "content_generation"
                    }
                )
                return update_state(state, {
                    'requires_human_review': False,
                    'human_feedback': 'rejected',
                    'current_step': WorkflowStep.CONTENT_GENERATION
                })
            else:
                logger.info(
                    "Human review pending - added to HITL queue, workflow paused",
                    extra={
                        "event": "workflow_node_pending",
                        "node": "human_review",
                        "campaign_id": campaign_id,
                        "content_id": content_id,
                        "review_status": "pending",
                        "duration_seconds": round(duration, 2),
                        "next_step": "END"
                    }
                )

                return update_state(state, {
                    'requires_human_review': True,
                    'human_feedback': 'pending',
                    'current_step': WorkflowStep.HUMAN_REVIEW
                })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Human review node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "human_review",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return update_state(state, {'error': str(e)})

    async def _simulate_campaign_node(self, state: WorkflowState) -> WorkflowState:
        """
        Simulation node - test campaign in simulated environment before live deployment (RQ2)

        Creates a digital twin of the target market and predicts campaign performance.
        Only proceeds to deployment if simulation metrics meet thresholds.
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')
        content_id = state.get('content_id')

        try:
            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("simulation")

            logger.info(
                "Workflow node: simulation",
                extra={
                    "event": "workflow_node_entered",
                    "node": "simulation",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "workflow_step": state.get('current_step')
                }
            )

            metadata = state.get('metadata', {})
            platform = metadata.get('platform', 'linkedin')
            persona = metadata.get('target_persona', 'decision_maker')
            budget = metadata.get('budget', 1000.0)
            duration_days = metadata.get('duration_days', 7)

            async with async_session_maker() as session:
                content_repo = ContentRepository(session)
                content = await content_repo.get_by_id(content_id)

                if not content:
                    logger.error("Content not found for simulation")
                    return update_state(state, {
                        'error': 'Content not found',
                        'simulation_passed': False
                    })

            sim_config = SimulationConfig(
                duration_days=min(duration_days, 7),
                num_customers=500,
                platforms=[platform],
                seed=42,
                validation_mode=False
            )

            env = MarketingEnvironment(sim_config)

            personas_loaded = False

            try:

                async with async_session_maker() as session:
                    from ...simulation.auto_calibration import ensure_calibration_exists
                    from ...data_layer.database.models import PersonaCalibration
                    from sqlalchemy import select

                    await ensure_calibration_exists(session)

                    result = await session.execute(
                        select(PersonaCalibration).where(PersonaCalibration.is_active == True)
                    )
                    calibrations = result.scalars().all()

                    if calibrations:
                        logger.info(f"✅ Using {len(calibrations)} calibrated personas (data-driven parameters)")
                        personas = await env.load_calibrated_personas(session)
                        personas_loaded = True
                    else:
                        logger.warning("⚠️  No calibrations available - falling back to PersonaFactory")
            except Exception as e:
                logger.warning(f"Auto-calibration failed: {e}")

            if not personas_loaded:
                try:
                    persona_factory = PersonaFactory()
                    if persona in persona_factory.list_available_personas():
                        personas = [persona_factory.create_persona_model(persona)]
                    else:
                        personas = persona_factory.generate_persona_distribution(
                            total_count=1,
                            distribution=None
                        )
                    env.load_personas(personas)
                    personas_loaded = True
                except Exception as e:
                    logger.warning(f"PersonaFactory failed: {e}")

            if not personas_loaded:
                from ...data_layer.database.models import Persona
                fallback_persona = Persona(
                    name=persona,
                    description=f"{persona} persona",
                    engagement_rate=0.05,
                    conversion_rate=0.10
                )
                env.load_personas([fallback_persona])
                logger.warning("Using minimal fallback persona")

            campaign_config = {
                'platform': platform,
                'content': {
                    'headline': content.headline,
                    'body': content.body
                },
                'targeting': {
                    'persona': persona
                },
                'budget': budget,
                'duration': duration_days
            }

            simulation_results = env.run_campaign(campaign_config)

            predicted_ctr = simulation_results.get('ctr', 0.0)
            predicted_conversions = simulation_results.get('conversions', 0)
            predicted_cpl = simulation_results.get('cpl', float('inf'))

            min_ctr = self._get_config_value('SIM_MIN_CTR', 0.01)
            min_conversions = self._get_config_value('SIM_MIN_CONVERSIONS', 1)
            max_cpl = self._get_config_value('SIM_MAX_CPL', budget * 2)

            simulation_passed = (
                predicted_ctr >= min_ctr and
                predicted_conversions >= min_conversions and
                predicted_cpl <= max_cpl
            )

            duration = (datetime.now() - start_time).total_seconds()


            try:
                from ..learning.simulation_accuracy_tracker import record_simulation_for_campaign
                await record_simulation_for_campaign(
                    campaign_id=campaign_id,
                    simulation_results={
                        'impressions': simulation_results.get('impressions', 0),
                        'clicks': simulation_results.get('clicks', 0),
                        'conversions': predicted_conversions,
                        'ctr': predicted_ctr,
                        'cpl': predicted_cpl if predicted_cpl != float('inf') else 0.0
                    }
                )
                logger.info(f"Recorded simulation predictions for RQ2 tracking: campaign {campaign_id}")
            except Exception as rq2_error:
                logger.warning(f"Failed to record RQ2 simulation predictions: {rq2_error}")

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)

                if simulation_passed:
                    await event_logger.log_event(
                        event_type=WorkflowEventType.NODE_COMPLETED,
                        title="Simulation Passed",
                        message=f"Campaign simulation successful! Predicted CTR: {predicted_ctr:.2%}, Conversions: {predicted_conversions}, CPL: €{predicted_cpl:.2f}. Proceeding to deployment.",
                        severity=AlertSeverity.INFO,
                        workflow_node="simulation",
                        content_id=content_id,
                        details={
                            "predicted_ctr": predicted_ctr,
                            "predicted_conversions": predicted_conversions,
                            "predicted_cpl": predicted_cpl if predicted_cpl != float('inf') else None,
                            "simulation_passed": True,
                            "duration": round(duration, 2)
                        }
                    )
                else:
                    await event_logger.log_event(
                        event_type=WorkflowEventType.ERROR_OCCURRED,
                        title="Simulation Failed",
                        message=f"Campaign simulation did not meet thresholds. CTR: {predicted_ctr:.2%} (need ≥{min_ctr:.2%}), Conversions: {predicted_conversions} (need ≥{min_conversions}). Regenerating content.",
                        severity=AlertSeverity.WARNING,
                        workflow_node="simulation",
                        content_id=content_id,
                        details={
                            "predicted_ctr": predicted_ctr,
                            "predicted_conversions": predicted_conversions,
                            "predicted_cpl": predicted_cpl if predicted_cpl != float('inf') else None,
                            "min_ctr": min_ctr,
                            "min_conversions": min_conversions,
                            "max_cpl": max_cpl,
                            "simulation_passed": False
                        }
                    )

            logger.info(
                "Simulation node completed",
                extra={
                    "event": "workflow_node_completed",
                    "node": "simulation",
                    "campaign_id": campaign_id,
                    "simulation_passed": simulation_passed,
                    "predicted_ctr": round(predicted_ctr, 4),
                    "predicted_conversions": predicted_conversions,
                    "predicted_cpl": round(predicted_cpl, 2) if predicted_cpl != float('inf') else 'inf',
                    "duration_seconds": round(duration, 2),
                    "next_step": "deployment" if simulation_passed else "regenerate"
                }
            )

            try:
                from sqlalchemy import update as sql_update
                async with async_session_maker() as db_session:
                    from ...data_layer.database.models import Campaign
                    result = await db_session.execute(
                        select(Campaign).where(Campaign.id == UUID(campaign_id))
                    )
                    campaign = result.scalar_one_or_none()

                    if campaign:
                        campaign_config = campaign.config or {}
                        campaign_config['simulation_results'] = {
                            'ctr': predicted_ctr,
                            'conversions': predicted_conversions,
                            'cpl': predicted_cpl if predicted_cpl != float('inf') else None,
                            'impressions': simulation_results.get('impressions', 0),
                            'clicks': simulation_results.get('clicks', 0),
                            'passed': simulation_passed,
                            'timestamp': datetime.now().isoformat(),
                            'full_results': simulation_results
                        }

                        await db_session.execute(
                            sql_update(Campaign)
                            .where(Campaign.id == UUID(campaign_id))
                            .values(config=campaign_config)
                        )
                        await db_session.commit()
                        logger.info(f"Saved simulation results to campaign {campaign_id}")
            except Exception as save_error:
                logger.error(f"Failed to save simulation results to database: {save_error}")

            return update_state(state, {
                'simulation_results': simulation_results,
                'simulation_passed': simulation_passed,
                'predicted_ctr': predicted_ctr,
                'predicted_conversions': predicted_conversions,
                'predicted_cpl': predicted_cpl,
                'current_step': WorkflowStep.DEPLOYMENT if simulation_passed else WorkflowStep.CONTENT_GENERATION,
                'messages': state['messages'] + [{
                    'role': 'system',
                    'content': f'Simulation {"PASSED" if simulation_passed else "FAILED"}: CTR={predicted_ctr:.2%}, Conv={predicted_conversions}, CPL=€{predicted_cpl:.2f}',
                    'timestamp': str(datetime.now())
                }]
            })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_error(str(e), "simulation", content_id)

            logger.error(
                "Simulation node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "simulation",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return update_state(state, {
                'error': str(e),
                'simulation_passed': False
            })

    def _route_after_simulation(self, state: WorkflowState) -> str:
        """Route after simulation - MARL gating or direct deployment"""
        campaign_id = state.get('campaign_id')
        simulation_passed = state.get('simulation_passed', False)
        predicted_ctr = state.get('predicted_ctr', 0.0)
        predicted_conversions = state.get('predicted_conversions', 0)

        if state.get('error'):
            logger.error(
                "Routing: simulation failed with error",
                extra={
                    "event": "workflow_routing",
                    "router": "after_simulation",
                    "campaign_id": campaign_id,
                    "error": state['error'],
                    "route": "failed"
                }
            )
            return "failed"

        if simulation_passed:
            if self.marl_gatekeeper:
                logger.info(
                    "Routing: simulation passed - proceeding to MARL gating",
                    extra={
                        "event": "workflow_routing",
                        "router": "after_simulation",
                        "campaign_id": campaign_id,
                        "predicted_ctr": round(predicted_ctr, 4),
                        "predicted_conversions": predicted_conversions,
                        "route": "marl_gating"
                    }
                )
                return "marl_gating"
            else:
                logger.info(
                    "Routing: simulation passed - proceeding to deployment",
                    extra={
                        "event": "workflow_routing",
                        "router": "after_simulation",
                        "campaign_id": campaign_id,
                        "predicted_ctr": round(predicted_ctr, 4),
                        "predicted_conversions": predicted_conversions,
                        "route": "deploy"
                    }
                )
                return "deploy"
        else:
            logger.warning(
                "Routing: simulation failed - regenerating content",
                extra={
                    "event": "workflow_routing",
                    "router": "after_simulation",
                    "campaign_id": campaign_id,
                    "predicted_ctr": round(predicted_ctr, 4),
                    "predicted_conversions": predicted_conversions,
                    "route": "regenerate"
                }
            )
            return "regenerate"

    async def _load_and_execute_marl_policy(
        self,
        policy_id: str,
        historical_decisions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Load MARL policy from MLflow and execute it on historical states

        Args:
            policy_id: Policy identifier to load
            historical_decisions: Historical decision data with states

        Returns:
            List of policy actions with probabilities
        """
        try:
            import pickle
            from pathlib import Path

            # Only attempt MLflow registry if tracking server is available
            if self._mlflow_available:
                try:
                    import mlflow
                    model_uri = f"models:/{policy_id}/latest"
                    loaded_model = mlflow.pyfunc.load_model(model_uri)
                    logger.info(f"Loaded MARL policy {policy_id} from MLflow")

                    policy_actions = []
                    for decision in historical_decisions:
                        state = decision['state']
                        state_features = self._state_to_features(state)

                        action_probs = loaded_model.predict(state_features)

                        policy_actions.append({
                            'state': state,
                            'action_probabilities': action_probs if isinstance(action_probs, dict) else {
                                f'action_{i}': float(p) for i, p in enumerate(action_probs)
                            }
                        })

                    return policy_actions

                except Exception as mlflow_error:
                    logger.info(f"MLflow model '{policy_id}' not registered, falling back to file system")

            policy_dir = Path(f"models/marl_policies/{policy_id}")
            policy_file = policy_dir / "policy.pkl"

            if policy_file.exists():
                with open(policy_file, 'rb') as f:
                    policy_model = pickle.load(f)
                logger.info(f"Loaded MARL policy {policy_id} from file system")

                policy_actions = []
                for decision in historical_decisions:
                    state = decision['state']
                    state_features = self._state_to_features(state)

                    if hasattr(policy_model, 'predict_proba'):
                        action_probs = policy_model.predict_proba(state_features)
                    elif hasattr(policy_model, 'predict'):
                        action_probs = policy_model.predict(state_features)
                    else:
                        action_probs = {'default': 1.0}

                    policy_actions.append({
                        'state': state,
                        'action_probabilities': action_probs if isinstance(action_probs, dict) else {
                            f'action_{i}': float(p) for i, p in enumerate(action_probs)
                        }
                    })

                return policy_actions

            logger.info(f"MARL policy {policy_id} not found — using default actions")
            return []

        except Exception as e:
            logger.error(f"Error loading MARL policy: {e}", exc_info=True)
            return []

    def _state_to_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert state dict to feature vector for policy input (12 features).
        
        Must match the trained model's input dimension (12):
        5 numeric + 4 cyclical time encodings + 3 platform one-hot.
        """
        try:
            import numpy as np
            features = []

            for key in ['budget_remaining', 'campaign_day', 'impressions', 'clicks', 'ctr']:
                val = state.get(key, 0)
                if isinstance(val, (int, float)):
                    features.append(float(val))
                elif isinstance(val, str):
                    features.append(float(hash(val) % 10000) / 10000.0)
                else:
                    features.append(0.0)

            hour = state.get('hour', state.get('time_of_day', 12))
            day_of_week = state.get('day_of_week', 0)
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))
            features.append(np.sin(2 * np.pi * day_of_week / 7))
            features.append(np.cos(2 * np.pi * day_of_week / 7))

            # 3 platform one-hot features (matches trained model)
            # Blog maps to email bucket (both are organic/content channels)
            platform = state.get('platform', state.get('campaign_type', 'linkedin'))
            for ptype in ['linkedin', 'twitter', 'email']:
                features.append(1.0 if platform == ptype or (ptype == 'email' and platform == 'blog') else 0.0)

            return np.array(features, dtype=np.float32).reshape(1, -1)

        except Exception as e:
            logger.warning(f"Error converting state to features: {e}")
            import numpy as np
            return np.zeros((1, 12), dtype=np.float32)

    async def _auto_train_marl_policy(
        self,
        policy_id: str,
        historical_decisions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Automatically train a MARL policy from historical campaign decisions.
        
        This enables fully autonomous operation - no manual scripts needed.
        The system trains policies when sufficient data is available.
        
        Args:
            policy_id: Identifier for the policy to train
            historical_decisions: Historical state-action-reward data
            
        Returns:
            Training result with success status and metrics
        """
        try:
            from ..marl.policy_trainer import MARLPolicyTrainer, TrainingConfig
            
            logger.info(f"Auto-training MARL policy '{policy_id}' with {len(historical_decisions)} samples")
            
            config = TrainingConfig(
                policy_id=policy_id,
                epochs=50,
                learning_rate=0.001,
                hidden_dim=128,
                batch_size=min(32, len(historical_decisions)),
                min_samples=20
            )
            
            trainer = MARLPolicyTrainer(config)
            
            training_data = []
            for decision in historical_decisions:
                training_data.append({
                    'state': decision.get('state', {}),
                    'action': decision.get('action', {'selected_hook': 'productivity_boost'}),
                    'reward': float(decision.get('reward', 0.0)),
                    'action_probability': decision.get('action_probability', 0.2),
                    'num_actions': decision.get('num_actions', 5)
                })
            
            samples_prepared = trainer.prepare_training_data(training_data)
            
            if samples_prepared < 10:
                return {
                    'success': False,
                    'error': f'Insufficient samples after preparation: {samples_prepared}'
                }
            
            metrics = trainer.train()
            
            save_path = trainer.save()
            
            logger.info(f"Auto-trained MARL policy saved to {save_path}")
            
            return {
                'success': True,
                'policy_id': policy_id,
                'save_path': str(save_path),
                'training_metrics': metrics,
                'samples_used': samples_prepared
            }
            
        except ImportError as ie:
            logger.warning(f"MARL training dependencies not available: {ie}")
            return {'success': False, 'error': f'Dependencies not installed: {ie}'}
        except Exception as e:
            logger.error(f"Auto-training failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _marl_gating_node(self, state: WorkflowState) -> WorkflowState:
        """
        MARL Gating Node - Offline Policy Evaluation (Research Track)

        Evaluates MARL policy using historical data before deployment.
        Only promotes policy if OPE shows ≥20% lift with 95% confidence.
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')

        try:
            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("marl_gating")

            logger.info(
                "Workflow node: marl_gating",
                extra={
                    "event": "workflow_node_entered",
                    "node": "marl_gating",
                    "campaign_id": campaign_id,
                    "workflow_step": state.get('current_step')
                }
            )

            metadata = state.get('metadata', {})
            policy_id = metadata.get('policy_id', 'marl_policy_v1')

            async with async_session_maker() as session:
                campaign_repo = CampaignRepository(session)

                min_samples = self._get_config_value('OPE_MIN_SAMPLES', 1000)
                historical_decisions = await campaign_repo.get_recent_decisions_for_ope(
                    limit=min_samples,
                    min_impressions=100,
                    lookback_days=30
                )

                logger.info(
                    f"Loaded {len(historical_decisions)} historical decisions for OPE",
                    extra={
                        "event": "ope_data_loaded",
                        "campaign_id": campaign_id,
                        "num_decisions": len(historical_decisions),
                        "min_samples_required": min_samples
                    }
                )

                # Baseline: 25th percentile reward (random/unoptimized strategy performance)
                if historical_decisions:
                    baseline_rewards = sorted([d['reward'] for d in historical_decisions])
                    n = len(baseline_rewards)
                    q25_idx = max(0, n // 4 - 1)
                    random_baseline = baseline_rewards[q25_idx]
                    baseline_metrics = {
                        'average_reward': random_baseline,
                        'mean_reward': sum(baseline_rewards) / n,
                        'min_reward': baseline_rewards[0],
                        'max_reward': baseline_rewards[-1]
                    }
                else:
                    baseline_metrics = {
                        'average_reward': 0.05
                    }
                    logger.warning(
                        "No historical data available for OPE, using default baseline",
                        extra={
                            "event": "ope_no_historical_data",
                            "campaign_id": campaign_id
                        }
                    )

                new_policy_actions = await self._load_and_execute_marl_policy(
                    policy_id, historical_decisions
                )
                

                if not new_policy_actions:
                    logger.info(
                        f"No MARL policy '{policy_id}' found - checking if auto-training is possible",
                        extra={
                            "event": "marl_gating_no_policy",
                            "campaign_id": campaign_id,
                            "historical_samples": len(historical_decisions)
                        }
                    )
                    
                    min_samples_for_training = self._get_config_value('MARL_MIN_TRAINING_SAMPLES', 20)
                    
                    if len(historical_decisions) >= min_samples_for_training:
                        logger.info(
                            f"Auto-training MARL policy '{policy_id}' with {len(historical_decisions)} samples",
                            extra={
                                "event": "marl_auto_training_started",
                                "campaign_id": campaign_id,
                                "samples": len(historical_decisions)
                            }
                        )
                        
                        async with async_session_maker() as session:
                            event_logger = WorkflowEventLogger(session, campaign_id)
                            await event_logger.log_event(
                                event_type=WorkflowEventType.NODE_STARTED,
                                title="Auto-Training MARL Policy",
                                message=f"Automatically training MARL policy '{policy_id}' using {len(historical_decisions)} historical decisions.",
                                severity=AlertSeverity.INFO,
                                workflow_node="marl_gating",
                                details={
                                    "policy_id": policy_id,
                                    "training_samples": len(historical_decisions),
                                    "action": "auto_training"
                                }
                            )
                        
                        try:
                            training_result = await self._auto_train_marl_policy(
                                policy_id=policy_id,
                                historical_decisions=historical_decisions
                            )
                            
                            if training_result.get('success'):
                                logger.info(
                                    f"MARL policy '{policy_id}' auto-trained successfully",
                                    extra={
                                        "event": "marl_auto_training_completed",
                                        "campaign_id": campaign_id,
                                        "final_loss": training_result.get('training_metrics', {}).get('final_loss')
                                    }
                                )
                                
                                new_policy_actions = await self._load_and_execute_marl_policy(
                                    policy_id, historical_decisions
                                )
                            else:
                                logger.warning(f"Auto-training failed: {training_result.get('error')}")
                                
                        except Exception as train_error:
                            logger.warning(f"Auto-training exception: {train_error}")
                    
                    if not new_policy_actions:
                        duration = (datetime.now() - start_time).total_seconds()
                        
                        async with async_session_maker() as session:
                            event_logger = WorkflowEventLogger(session, campaign_id)
                            await event_logger.log_event(
                                event_type=WorkflowEventType.NODE_COMPLETED,
                                title="MARL Gate Passed (No Policy)",
                                message=f"No trained MARL policy available for '{policy_id}'. Need {min_samples_for_training} samples, have {len(historical_decisions)}. Using baseline strategy.",
                                severity=AlertSeverity.INFO,
                                workflow_node="marl_gating",
                                details={
                                    "policy_id": policy_id,
                                    "reason": "insufficient_data_for_training" if len(historical_decisions) < min_samples_for_training else "training_failed",
                                    "samples_available": len(historical_decisions),
                                    "samples_required": min_samples_for_training,
                                    "approved": True,
                                    "action": "passthrough_to_baseline"
                                }
                            )
                        
                        return update_state(state, {
                            'marl_evaluation': {'approved': True, 'reason': 'no_policy_to_evaluate'},
                            'marl_approved': True,
                            'has_trained_policy': False,  # Distinguish from actual policy approval
                            'policy_id': policy_id,
                            'current_step': WorkflowStep.DEPLOYMENT,
                            'messages': state['messages'] + [{
                                'role': 'system',
                                'content': f'MARL Gate PASSED: No trained policy found, using baseline strategy',
                                'timestamp': str(datetime.now())
                            }]
                        })

                evaluation_result = await self.marl_gatekeeper.evaluate_policy_for_promotion(
                    policy_id=policy_id,
                    new_policy_actions=new_policy_actions,
                    historical_data=historical_decisions,
                    baseline_metrics=baseline_metrics
                )

                approved = evaluation_result.get('approved', False)
                ope_result = evaluation_result.get('ope_result')

                duration = (datetime.now() - start_time).total_seconds()

                async with async_session_maker() as session:
                    event_logger = WorkflowEventLogger(session, campaign_id)

                    if approved:
                        lift = ope_result.lift_percentage if ope_result else 0.0
                        await event_logger.log_event(
                            event_type=WorkflowEventType.NODE_COMPLETED,
                            title="MARL Policy Approved",
                            message=f"MARL policy '{policy_id}' approved for deployment. Offline evaluation shows {lift:.1f}% lift over baseline. Proceeding to deployment.",
                            severity=AlertSeverity.INFO,
                            workflow_node="marl_gating",
                            details={
                                "policy_id": policy_id,
                                "lift_percentage": lift,
                                "confidence_level": ope_result.confidence_level if ope_result else None,
                                "baseline_value": ope_result.baseline_value if ope_result else None,
                                "estimated_value": ope_result.estimated_value if ope_result else None,
                                "approved": True
                            }
                        )
                    else:
                        reason = evaluation_result.get('reason', 'Policy did not meet promotion criteria')
                        await event_logger.log_event(
                            event_type=WorkflowEventType.NODE_COMPLETED,
                            title="MARL Policy Rejected",
                            message=f"MARL policy '{policy_id}' did not meet promotion criteria: {reason}. Using baseline strategy (this is expected behavior — policies require ≥20% lift).",
                            severity=AlertSeverity.INFO,
                            workflow_node="marl_gating",
                            details={
                                "policy_id": policy_id,
                                "reason": reason,
                                "lift_percentage": ope_result.lift_percentage if ope_result else 0.0,
                                "approved": False
                            },
                            is_user_actionable=False
                        )

                logger.info(
                    "MARL gating node completed",
                    extra={
                        "event": "workflow_node_completed",
                        "node": "marl_gating",
                        "campaign_id": campaign_id,
                        "policy_id": policy_id,
                        "approved": approved,
                        "lift_percentage": ope_result.lift_percentage if ope_result else 0.0,
                        "duration_seconds": round(duration, 2),
                        "next_step": "deployment" if approved else "rejected"
                    }
                )

                return update_state(state, {
                    'marl_evaluation': evaluation_result,
                    'marl_approved': approved,
                    'has_trained_policy': True,
                    'policy_id': policy_id,
                    'current_step': WorkflowStep.DEPLOYMENT if approved else WorkflowStep.CONTENT_GENERATION,
                    'messages': state['messages'] + [{
                        'role': 'system',
                        'content': f'MARL Policy {"APPROVED" if approved else "REJECTED"}: {evaluation_result.get("reason")}',
                        'timestamp': str(datetime.now())
                    }]
                })

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_error(str(e), "marl_gating")

            logger.error(
                "MARL gating node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "marl_gating",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return update_state(state, {
                'error': str(e),
                'marl_approved': False
            })

    def _route_after_marl_gating(self, state: WorkflowState) -> str:
        """
        Route after MARL gating - start canary deployment if approved, reject otherwise.
        
        CRITICAL FIX: When MARL policy is approved, we now trigger canary deployment
        at 5% traffic instead of direct full deployment. This enables progressive
        rollout with automatic rollback if metrics degrade.
        """
        campaign_id = state.get('campaign_id')
        marl_approved = state.get('marl_approved', False)
        policy_id = state.get('policy_id', 'unknown')

        if state.get('error'):
            logger.error(
                "Routing: MARL gating failed with error",
                extra={
                    "event": "workflow_routing",
                    "router": "after_marl_gating",
                    "campaign_id": campaign_id,
                    "policy_id": policy_id,
                    "error": state['error'],
                    "route": "rejected"
                }
            )
            return "rejected"

        if marl_approved:
            logger.info(
                "Routing: MARL policy approved - starting CANARY deployment at 5%",
                extra={
                    "event": "workflow_routing",
                    "router": "after_marl_gating",
                    "campaign_id": campaign_id,
                    "policy_id": policy_id,
                    "route": "canary_deploy",
                    "initial_traffic_percent": 5
                }
            )
            return "canary_deploy"
        else:
            logger.warning(
                "Routing: MARL policy rejected - insufficient improvement",
                extra={
                    "event": "workflow_routing",
                    "router": "after_marl_gating",
                    "campaign_id": campaign_id,
                    "policy_id": policy_id,
                    "route": "rejected"
                }
            )
            return "rejected"

    async def _golden_test_gate_node(self, state: WorkflowState) -> WorkflowState:
        """
        Golden Test Gate Node - Research Plan Section 7.2

        Runs the golden test suite before deployment. A failure in this test suite
        will block the deployment.

        "A 'golden suite' of 30-50 test items will be created. This suite will be
        run automatically in the CI/CD pipeline for every change to a model or prompt.
        A failure in this test suite will block the deployment."
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')

        try:
            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("golden_test_gate")

            logger.info(
                "Workflow node: golden_test_gate (GOVERN phase - final gate)",
                extra={
                    "event": "workflow_node_entered",
                    "node": "golden_test_gate",
                    "campaign_id": campaign_id,
                    "workflow_step": state.get('current_step'),
                    "ooda_phase": "GOVERN"
                }
            )

            from ...governance.golden_test_suite import GoldenTestSuite

            golden_suite = GoldenTestSuite()
            golden_suite.load_test_cases()

            test_results = await golden_suite.run_all_tests()

            duration = (datetime.now() - start_time).total_seconds()

            approved_for_deployment = test_results.get('approved_for_deployment', False)
            pass_rate = test_results.get('pass_rate', 0.0)

            if approved_for_deployment:
                logger.info(
                    f"Golden test gate PASSED: {test_results['passed']}/{test_results['total']} tests",
                    extra={
                        "event": "workflow_node_completed",
                        "node": "golden_test_gate",
                        "campaign_id": campaign_id,
                        "pass_rate": pass_rate,
                        "tests_passed": test_results['passed'],
                        "tests_total": test_results['total'],
                        "deployment_approved": True,
                        "duration_seconds": duration
                    }
                )
            else:
                logger.error(
                    f"Golden test gate FAILED: {test_results['passed']}/{test_results['total']} tests - DEPLOYMENT BLOCKED",
                    extra={
                        "event": "workflow_node_failed",
                        "node": "golden_test_gate",
                        "campaign_id": campaign_id,
                        "pass_rate": pass_rate,
                        "tests_passed": test_results['passed'],
                        "tests_failed": test_results['failed'],
                        "failures": test_results.get('failures', []),
                        "deployment_approved": False,
                        "duration_seconds": duration
                    }
                )

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_completed(
                    "golden_test_gate",
                    result={
                        "approved_for_deployment": approved_for_deployment,
                        "pass_rate": pass_rate,
                        "tests_passed": test_results['passed'],
                        "tests_total": test_results['total'],
                        "failures": test_results.get('failures', [])
                    }
                )

            return update_state(state, {
                'golden_test_results': test_results,
                'golden_test_approved': approved_for_deployment,
                'golden_test_pass_rate': pass_rate
            })

        except Exception as e:
            logger.error(
                f"Golden test gate error: {e}",
                extra={
                    "event": "workflow_node_error",
                    "node": "golden_test_gate",
                    "campaign_id": campaign_id,
                    "error": str(e)
                }
            )
            return update_state(state, {
                'golden_test_results': {'error': str(e)},
                'golden_test_approved': False,
                'golden_test_pass_rate': 0.0,
                'error': f"Golden test error: {str(e)}"
            })

    def _route_after_golden_test(self, state: WorkflowState) -> str:
        """
        Route after golden test gate

        Returns:
            - "canary_deploy": Golden tests passed and canary deployment is enabled
            - "deploy": Golden tests passed, canary not available
            - "blocked": If any test failed, block deployment
        """
        campaign_id = state.get('campaign_id')
        approved = state.get('golden_test_approved', False)
        pass_rate = state.get('golden_test_pass_rate', 0.0)

        if not approved:
            logger.warning(
                f"Routing: Golden test gate BLOCKED deployment (pass_rate={pass_rate:.1f}%)",
                extra={
                    "event": "workflow_routing",
                    "router": "after_golden_test",
                    "campaign_id": campaign_id,
                    "route": "blocked",
                    "pass_rate": pass_rate
                }
            )
            return "blocked"

        if self.marl_gatekeeper:
            logger.info(
                "Routing: Golden tests passed, proceeding to canary deployment",
                extra={
                    "event": "workflow_routing",
                    "router": "after_golden_test",
                    "campaign_id": campaign_id,
                    "route": "canary_deploy",
                    "pass_rate": pass_rate
                }
            )
            return "canary_deploy"

        logger.info(
            "Routing: Golden tests passed, proceeding to deployment",
            extra={
                "event": "workflow_routing",
                "router": "after_golden_test",
                "campaign_id": campaign_id,
                "route": "deploy",
                "pass_rate": pass_rate
            }
        )
        return "deploy"

    async def _deploy_node(self, state: WorkflowState) -> WorkflowState:
        """Deployment node"""
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')
        content_id = state.get('content_id')

        try:
            platform = state['metadata'].get('platform', 'linkedin')

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("deployment")

            logger.info(
                "Workflow node: deployment",
                extra={
                    "event": "workflow_node_entered",
                    "node": "deployment",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "platform": platform,
                    "workflow_step": state.get('current_step')
                }
            )

            async with async_session_maker() as session:
                content_repo = ContentRepository(session)
                content = await content_repo.get_by_id(content_id)

                if not content:
                    logger.error(
                        "Content not found for deployment",
                        extra={
                            "event": "workflow_node_error",
                            "node": "deployment",
                            "campaign_id": campaign_id,
                            "content_id": content_id,
                            "error": "Content not found in database"
                        }
                    )
                    return update_state(state, {'error': 'Content not found'})

                deployment_result = await self.deployer.deploy(
                    content_id=content_id,
                    platform=platform,
                    content={
                        'headline': content.headline,
                        'body': content.body
                    },
                    campaign_config={
                        'campaign_id': campaign_id
                    }
                )

                if deployment_result['success']:
                    await content_repo.mark_deployed(
                        content_id,
                        platform_post_id=deployment_result.get('post_id', ''),
                        metrics=deployment_result.get('metrics')
                    )

                    metrics = deployment_result.get('metrics', {})
                    if metrics:
                        campaign_repo = CampaignRepository(session)
                        await campaign_repo.update_metrics(
                            campaign_id=campaign_id,
                            impressions=metrics.get('impressions', 0),
                            clicks=metrics.get('clicks', 0),
                            conversions=metrics.get('conversions', 0),
                            spend=metrics.get('budget_spent', 0.0)
                        )

                        logger.info(
                            "Campaign metrics updated",
                            extra={
                                "event": "campaign_metrics_updated",
                                "campaign_id": campaign_id,
                                "impressions": metrics.get('impressions', 0),
                                "clicks": metrics.get('clicks', 0),
                                "conversions": metrics.get('conversions', 0),
                                "budget_spent": metrics.get('budget_spent', 0.0)
                            }
                        )

                    if deployment_result.get('is_mock', False):
                        campaign_repo = CampaignRepository(session)
                        await campaign_repo.update(
                            campaign_id=campaign_id,
                            updates={'is_mock': True}
                        )

            duration = (datetime.now() - start_time).total_seconds()

            if deployment_result['success']:
                async with async_session_maker() as session:
                    event_logger = WorkflowEventLogger(session, campaign_id)
                    await event_logger.log_content_deployed(content_id, platform)
                    await event_logger.log_node_completed("deployment")

                # Closes the deployment → experiment learning feedback loop
                await self._update_bandit_arm_on_deployment(
                    campaign_id=campaign_id,
                    strategy=state.get('strategy', {}),
                    success=True,
                    metrics=deployment_result.get('metrics', {})
                )

                logger.info(
                    "Deployment node completed successfully",
                    extra={
                        "event": "workflow_node_completed",
                        "node": "deployment",
                        "campaign_id": campaign_id,
                        "content_id": content_id,
                        "platform": platform,
                        "post_id": deployment_result.get('post_id', ''),
                        "duration_seconds": round(duration, 2),
                        "next_step": "END",
                        "status": "deployed"
                    }
                )

                return update_state(state, {
                    'deployment_status': 'deployed',
                    'current_step': WorkflowStep.END,
                    'messages': state['messages'] + [{
                        'role': 'system',
                        'content': 'Content deployed successfully',
                        'timestamp': str(datetime.now())
                    }]
                })
            else:
                error_msg = deployment_result.get('error', 'Unknown deployment error')
                async with async_session_maker() as session:
                    event_logger = WorkflowEventLogger(session, campaign_id)
                    await event_logger.log_error(error_msg, "deployment", content_id)

                logger.error(
                    f"Deployment failed: {error_msg}",
                    extra={
                        "event": "workflow_node_error",
                        "node": "deployment",
                        "campaign_id": campaign_id,
                        "content_id": content_id,
                        "platform": platform,
                        "error": error_msg,
                        "duration_seconds": round(duration, 2)
                    }
                )
                return update_state(state, {'error': error_msg})

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_error(str(e), "deployment", content_id)

            logger.error(
                "Deployment node failed with exception",
                extra={
                    "event": "workflow_node_error",
                    "node": "deployment",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "platform": state.get('metadata', {}).get('platform', 'unknown'),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return update_state(state, {'error': str(e)})

    async def _canary_deploy_node(self, state: WorkflowState) -> WorkflowState:
        """
        Canary Deployment Node - Progressive rollout for MARL-approved policies
        
        CRITICAL FIX: Instead of direct 100% deployment for MARL-approved policies,
        this node starts a canary deployment at 5% traffic with automatic
        monitoring and rollback if metrics degrade.
        
        Flow:
        1. Start canary deployment at 5%
        2. Monitor metrics in background
        3. Auto-progress through 5% → 25% → 50% → 75% → 100%
        4. Auto-rollback if CTR drops >10% or error rate >5%
        """
        start_time = datetime.now()
        campaign_id = state.get('campaign_id')
        content_id = state.get('content_id')
        policy_id = state.get('policy_id', 'marl_policy_v1')

        try:
            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_node_started("canary_deployment")

            logger.info(
                "Workflow node: canary_deployment",
                extra={
                    "event": "workflow_node_entered",
                    "node": "canary_deployment",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "policy_id": policy_id,
                    "initial_traffic_percent": 5
                }
            )

            from ...automation_layer.deployment.canary_rollout import DeploymentController

            controller = DeploymentController()

            deployment_result = await controller.start_canary_deployment(
                name=f"MARL_{policy_id}_{campaign_id[:8]}",
                policy_name=policy_id,
                deployment_type="marl_policy",
                initial_traffic_percent=5,
                progression_schedule="conservative",  # 5→25→50→75→100 over time
                auto_rollback=True,
                rollback_thresholds={
                    "ctr_drop_percent": 10,  # Rollback if CTR drops 10%
                    "error_rate_percent": 5,  # Rollback if errors exceed 5%
                    "cpl_increase_percent": 15  # Rollback if CPL increases 15%
                },
                metadata={
                    "campaign_id": str(campaign_id),
                    "content_id": str(content_id),
                    "policy_id": policy_id,
                    "triggered_by": "marl_gating_node"
                }
            )

            duration = (datetime.now() - start_time).total_seconds()

            if deployment_result.get("success"):
                canary_id = deployment_result.get("deployment_id")
                
                try:
                    async with async_session_maker() as db_session:
                        from ...data_layer.database.models import CanaryDeployment
                        canary_record = CanaryDeployment(
                            deployment_id=canary_id,
                            policy_id=policy_id,
                            policy_version="1.0.0",
                            status="canary_5_percent",
                            current_traffic_percentage=0.05,
                            started_at=start_time,
                            extra_data={
                                "campaign_id": str(campaign_id),
                                "content_id": str(content_id),
                                "rollback_thresholds": {
                                    "ctr_drop_percent": 10,
                                    "error_rate_percent": 5,
                                    "cpl_increase_percent": 15
                                }
                            }
                        )
                        db_session.add(canary_record)
                        await db_session.commit()
                        logger.info(f"Canary deployment persisted to DB: {canary_id}")
                except Exception as db_err:
                    logger.warning(f"Failed to persist canary deployment to DB: {db_err}")

                async with async_session_maker() as session:
                    event_logger = WorkflowEventLogger(session, campaign_id)
                    await event_logger.log_event(
                        event_type=WorkflowEventType.CANARY_STARTED,
                        title="Canary Deployment Started",
                        message=(
                            f"MARL policy '{policy_id}' deployed as canary at 5% traffic. "
                            f"Deployment ID: {canary_id}. Progressive rollout will auto-advance "
                            f"if metrics are stable."
                        ),
                        severity=AlertSeverity.INFO,
                        workflow_node="canary_deployment",
                        content_id=content_id,
                        details={
                            "canary_id": canary_id,
                            "policy_id": policy_id,
                            "initial_traffic": 5,
                            "schedule": "conservative",
                            "duration_seconds": round(duration, 2)
                        }
                    )
                    await event_logger.log_node_completed(
                        node="canary_deployment",
                        content_id=content_id,
                        details={
                            "canary_id": canary_id,
                            "policy_id": policy_id,
                            "traffic_percent": 5
                        },
                        duration=duration
                    )

                logger.info(
                    "Canary deployment started successfully",
                    extra={
                        "event": "workflow_node_completed",
                        "node": "canary_deployment",
                        "campaign_id": campaign_id,
                        "canary_id": canary_id,
                        "traffic_percent": 5,
                        "duration_seconds": round(duration, 2)
                    }
                )

                return update_state(state, {
                    'deployment_result': deployment_result,
                    'canary_id': canary_id,
                    'canary_traffic_percent': 5,
                    'deployment_type': 'canary'
                })
            else:
                error_msg = deployment_result.get("error", "Unknown canary deployment error")
                logger.error(f"Canary deployment failed: {error_msg}")
                
                return update_state(state, {
                    'error': error_msg,
                    'deployment_result': deployment_result
                })

        except ImportError as e:
            logger.warning(f"Canary controller not available, falling back to direct deploy: {e}")
            return await self._deploy_node(state)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_error(str(e), "canary_deployment")

            logger.error(
                "Canary deployment node failed",
                extra={
                    "event": "workflow_node_error",
                    "node": "canary_deployment",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return update_state(state, {'error': str(e)})

    def _route_after_safety(self, state: WorkflowState) -> str:
        """
        Route after safety validation.
        
        Per research plan (Section 6.3):
        - Score > 0.9 (high confidence) → skip HITL, proceed to cost_check
        - Score 0.7-0.9 (medium confidence) → goes to HITL review
        - Score < 0.7 (low confidence) → regenerate content
        - REQUIRE_HUMAN_APPROVAL=True → always HITL
        """
        campaign_id = state.get('campaign_id')
        content_id = state.get('content_id')
        safety_score = state.get('safety_score')
        safety_passed = state.get('safety_passed', None)

        if safety_score is None:
            logger.error(
                "Routing: safety_score is None - regenerating content",
                extra={
                    "event": "workflow_routing",
                    "router": "after_safety",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "safety_score": None,
                    "route": "regenerate"
                }
            )
            return "regenerate"

        min_safety_score = get_runtime_config('MIN_SAFETY_SCORE', 0.70)

        # Score below minimum → content is too unsafe, must regenerate
        if safety_score < min_safety_score:
            logger.warning(
                f"Routing: safety score {safety_score:.3f} below minimum {min_safety_score} "
                f"- regenerating content",
                extra={
                    "event": "workflow_routing",
                    "router": "after_safety",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "safety_score": round(safety_score, 3),
                    "route": "regenerate",
                    "reason": "below_minimum_threshold"
                }
            )
            return "regenerate"

        if should_review_human(state):
            logger.info(
                "Routing: sending to human review",
                extra={
                    "event": "workflow_routing",
                    "router": "after_safety",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "safety_score": round(safety_score, 3),
                    "route": "human_review"
                }
            )
            return "human_review"
        else:
            logger.info(
                "Routing: safety check passed - proceeding to cost check",
                extra={
                    "event": "workflow_routing",
                    "router": "after_safety",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "safety_score": round(safety_score, 3),
                    "route": "cost_check"
                }
            )
            return "cost_check"

    def _route_after_human_review(self, state: WorkflowState) -> str:
        """Route after human review"""
        campaign_id = state.get('campaign_id')
        content_id = state.get('content_id')
        feedback = state.get('human_feedback')

        if feedback == 'approved':
            logger.info(
                "Routing: human review approved - proceeding to cost check",
                extra={
                    "event": "workflow_routing",
                    "router": "after_human_review",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "feedback": "approved",
                    "route": "approved"
                }
            )
            return "approved"
        elif feedback == 'rejected':
            logger.info(
                "Routing: human review rejected - regenerating content",
                extra={
                    "event": "workflow_routing",
                    "router": "after_human_review",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "feedback": "rejected",
                    "route": "rejected"
                }
            )
            return "rejected"
        else:
            logger.info(
                "Routing: human review pending - ending workflow (waiting in HITL queue)",
                extra={
                    "event": "workflow_routing",
                    "router": "after_human_review",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "feedback": feedback or "none",
                    "route": "pending"
                }
            )
            return "pending"

    def _route_after_cost_check(self, state: WorkflowState) -> str:
        """Route after cost check - now proceeds to simulation before deployment"""
        campaign_id = state.get('campaign_id')
        campaign_cost = state.get('cost_accumulated', 0.0)
        budget = state['metadata'].get('budget', float('inf'))

        if can_deploy(state):
            logger.info(
                "Routing: cost check passed - proceeding to simulation",
                extra={
                    "event": "workflow_routing",
                    "router": "after_cost_check",
                    "campaign_id": campaign_id,
                    "campaign_cost": round(campaign_cost, 2),
                    "budget": round(budget, 2) if budget != float('inf') else 'unlimited',
                    "route": "simulate"
                }
            )
            return "simulate"
        else:
            logger.warning(
                "Routing: budget exceeded - halting workflow",
                extra={
                    "event": "workflow_routing",
                    "router": "after_cost_check",
                    "campaign_id": campaign_id,
                    "campaign_cost": round(campaign_cost, 2),
                    "budget": round(budget, 2) if budget != float('inf') else 'unlimited',
                    "route": "budget_exceeded"
                }
            )
            return "budget_exceeded"
    
    async def run_campaign_workflow(
        self,
        campaign_id: str,
        config: Optional[Dict] = None,
        previous_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete workflow for campaign

        Args:
            campaign_id: Campaign UUID
            config: Optional configuration
            previous_feedback: Optional feedback from previous rejection/regeneration

        Returns:
            Workflow execution results
        """
        start_time = datetime.now()
        config = config or {}

        async with async_session_maker() as session:
            campaign_repo = CampaignRepository(session)
            campaign = await campaign_repo.get_by_id(campaign_id)
            
            if campaign:
                campaign_config = {
                    'platform': campaign.platform.value if campaign.platform else 'linkedin',
                    'target_persona': campaign.target_persona or 'decision_maker',
                    'goal': campaign.goal or 'lead_generation',
                    'budget': float(campaign.budget_total or 1000.0),
                    'campaign_name': campaign.name,
                    'campaign_id': str(campaign.id),
                }
                for key, value in campaign_config.items():
                    if key not in config or config[key] is None:
                        config[key] = value
            else:
                logger.warning(f"Campaign {campaign_id} not found in database, using default config")

        if previous_feedback:
            config['previous_feedback'] = previous_feedback

        logger.info(
            "Starting campaign workflow",
            extra={
                "event": "workflow_started",
                "campaign_id": campaign_id,
                "platform": config.get('platform', 'linkedin'),
                "persona": config.get('target_persona', 'decision_maker'),
                "goal": config.get('goal', 'lead_generation'),
                "budget": config.get('budget', 1000.0)
            }
        )

        async with async_session_maker() as session:
            event_logger = WorkflowEventLogger(session, campaign_id)
            await event_logger.log_workflow_started()

        initial_state = create_initial_state(campaign_id, config)

        try:
            final_state = await self.app.ainvoke(
                initial_state,
                {"configurable": {"thread_id": campaign_id}}
            )

            duration = (datetime.now() - start_time).total_seconds()
            success = final_state['error'] is None

            is_pending_human_review = (
                final_state.get('requires_human_review') is True and
                final_state.get('human_feedback') == 'pending'
            )

            if success and not is_pending_human_review:
                async with async_session_maker() as session:
                    event_logger = WorkflowEventLogger(session, campaign_id)
                    content_id = final_state.get('content_id')
                    if content_id:
                        await event_logger.log_workflow_completed(content_id)
                    
                    campaign_repo = CampaignRepository(session)
                    deployment_status = final_state.get('deployment_status')
                    if deployment_status == 'deployed':
                        await campaign_repo.update_status(campaign_id, CampaignStatus.COMPLETED)
                        logger.info(f"Campaign {campaign_id} status updated to COMPLETED")
                    elif final_state.get('golden_test_approved') is False:
                        await campaign_repo.update_status(campaign_id, CampaignStatus.PAUSED)
                        logger.info(f"Campaign {campaign_id} status updated to PAUSED (golden test blocked)")

                logger.info(
                    "Campaign workflow completed successfully",
                    extra={
                        "event": "workflow_completed",
                        "campaign_id": campaign_id,
                        "content_id": final_state.get('content_id'),
                        "deployment_status": final_state.get('deployment_status'),
                        "total_cost": round(final_state.get('cost_accumulated', 0.0), 2),
                        "duration_seconds": round(duration, 2),
                        "final_step": final_state.get('current_step'),
                        "num_messages": len(final_state.get('messages', [])),
                        "success": True
                    }
                )
            elif is_pending_human_review:
                logger.info(
                    "Campaign workflow paused - waiting for human approval",
                    extra={
                        "event": "workflow_paused",
                        "campaign_id": campaign_id,
                        "content_id": final_state.get('content_id'),
                        "safety_score": final_state.get('safety_score'),
                        "total_cost": round(final_state.get('cost_accumulated', 0.0), 2),
                        "duration_seconds": round(duration, 2),
                        "final_step": final_state.get('current_step'),
                        "status": "pending_approval"
                    }
                )
            else:
                async with async_session_maker() as session:
                    event_logger = WorkflowEventLogger(session, campaign_id)
                    error_msg = final_state.get('error', 'Unknown error')
                    await event_logger.log_workflow_failed(error_msg)

                logger.error(
                    "Campaign workflow completed with error",
                    extra={
                        "event": "workflow_failed",
                        "campaign_id": campaign_id,
                        "error": final_state.get('error'),
                        "duration_seconds": round(duration, 2),
                        "final_step": final_state.get('current_step'),
                        "success": False
                    }
                )

            return {
                'success': success,
                'content_id': final_state.get('content_id'),
                'deployment_status': final_state.get('deployment_status'),
                'total_cost': final_state.get('cost_accumulated', 0.0),
                'messages': final_state.get('messages', []),
                'metadata': final_state.get('metadata', {}),
                'error': final_state.get('error')
            }

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            async with async_session_maker() as session:
                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_workflow_failed(f"Exception: {str(e)}")

            logger.error(
                "Campaign workflow execution failed with exception",
                extra={
                    "event": "workflow_exception",
                    "campaign_id": campaign_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'messages': initial_state.get('messages', [])
            }

    async def resume_workflow_after_approval(
        self,
        campaign_id: str,
        content_id: str
    ) -> Dict[str, Any]:
        """
        Resume workflow after HITL approval

        Continues from cost_check → simulation → deployment

        Args:
            campaign_id: Campaign UUID
            content_id: Approved content UUID

        Returns:
            Workflow execution results
        """
        start_time = datetime.now()

        logger.info(
            "Resuming workflow after HITL approval",
            extra={
                "event": "workflow_resumed",
                "campaign_id": campaign_id,
                "content_id": content_id
            }
        )

        try:
            async with async_session_maker() as session:
                content_repo = ContentRepository(session)
                campaign_repo = CampaignRepository(session)

                content = await content_repo.get_by_id(content_id)
                campaign = await campaign_repo.get_by_id(campaign_id)

                if not content:
                    raise ValueError(f"Content {content_id} not found")
                if not campaign:
                    raise ValueError(f"Campaign {campaign_id} not found")

            async with async_session_maker() as event_session:
                event_logger = WorkflowEventLogger(event_session, campaign_id)
                await event_logger.log_node_completed("human_review")

            async with async_session_maker() as session:
                content_repo = ContentRepository(session)
                campaign_repo = CampaignRepository(session)
                content = await content_repo.get_by_id(content_id)
                campaign = await campaign_repo.get_by_id(campaign_id)

                resume_state = {
                    'campaign_id': campaign_id,
                    'content_id': content_id,
                    'content': {
                        'headline': content.headline,
                        'body': content.body,
                        'cta': content.cta,
                        'claims_used': content.claims_used or []
                    },
                    'safety_score': content.safety_score or 0.0,
                    'approval_status': 'approved',
                    'current_step': WorkflowStep.COST_CHECK.value,
                    'cost_accumulated': content.generation_cost or 0.0,
                    'messages': [],
                    'metadata': {
                        'platform': campaign.platform,
                        'target_persona': campaign.target_persona,
                        'goal': campaign.config.get('goal', 'lead_generation'),
                        'budget_total': campaign.budget_total,
                        'budget_daily_limit': campaign.budget_daily_limit
                    },
                    'error': None,
                    'requires_human_review': False,
                    'human_feedback': content.review_notes,
                    'deployment_status': None,
                    'strategy': None
                }

                logger.info("Running cost check node")
                state_after_cost = await self._check_cost_node(resume_state)

                cost_route = self._route_after_cost_check(state_after_cost)
                if cost_route == "budget_exceeded":
                    logger.warning("Budget exceeded, stopping workflow")
                    return {
                        'success': False,
                        'content_id': content_id,
                        'error': 'Budget exceeded',
                        'final_step': 'cost_check'
                    }

                logger.info("Running simulation node")
                state_after_sim = await self._simulate_campaign_node(state_after_cost)

                sim_route = self._route_after_simulation(state_after_sim)
                if sim_route == "failed":
                    logger.error("Simulation failed")
                    return {
                        'success': False,
                        'content_id': content_id,
                        'error': 'Simulation failed',
                        'final_step': 'simulation'
                    }
                elif sim_route == "regenerate":
                    logger.warning("Simulation suggests regeneration")
                    return {
                        'success': False,
                        'content_id': content_id,
                        'error': 'Simulation suggests content regeneration',
                        'final_step': 'simulation'
                    }

                if self.marl_gatekeeper and sim_route == "marl_gating":
                    logger.info("Running MARL gating node")
                    state_after_marl = await self._marl_gating_node(state_after_sim)
                    marl_route = self._route_after_marl_gating(state_after_marl)
                    if marl_route == "rejected":
                        logger.warning("MARL gating rejected deployment")
                        return {
                            'success': False,
                            'content_id': content_id,
                            'error': 'MARL gating rejected',
                            'final_step': 'marl_gating'
                        }
                    state_before_golden = state_after_marl
                else:
                    state_before_golden = state_after_sim

                logger.info("Running golden test gate node")
                state_after_golden = await self._golden_test_gate_node(state_before_golden)
                
                golden_route = self._route_after_golden_test(state_after_golden)
                if golden_route == "blocked":
                    logger.warning("Golden test gate blocked deployment")
                    return {
                        'success': False,
                        'content_id': content_id,
                        'error': 'Golden test gate blocked deployment',
                        'golden_test_pass_rate': state_after_golden.get('golden_test_pass_rate', 0),
                        'final_step': 'golden_test_gate'
                    }

                if golden_route == "canary_deploy":
                    logger.info("Running canary deployment node")
                    state_after_canary = await self._canary_deploy_node(state_after_golden)
                    logger.info("Running deployment node after canary")
                    final_state = await self._deploy_node(state_after_canary)
                else:
                    logger.info("Running deployment node")
                    final_state = await self._deploy_node(state_after_golden)

                duration = (datetime.now() - start_time).total_seconds()

                deployment_status = final_state.get('deployment_status')
                if deployment_status == 'deployed':
                    await campaign_repo.update_status(campaign_id, CampaignStatus.COMPLETED)
                    logger.info(f"Campaign {campaign_id} status updated to COMPLETED after HITL resume")
                elif final_state.get('golden_test_approved') is False:
                    await campaign_repo.update_status(campaign_id, CampaignStatus.PAUSED)
                    logger.info(f"Campaign {campaign_id} status updated to PAUSED (golden test blocked)")

                event_logger = WorkflowEventLogger(session, campaign_id)
                await event_logger.log_workflow_completed(content_id)

                logger.info(
                    "Workflow resumed and completed successfully",
                    extra={
                        "event": "workflow_resume_completed",
                        "campaign_id": campaign_id,
                        "content_id": content_id,
                        "deployment_status": deployment_status,
                        "duration_seconds": round(duration, 2)
                    }
                )

                return {
                    'success': True,
                    'content_id': content_id,
                    'deployment_status': deployment_status,
                    'total_cost': final_state.get('cost_accumulated', 0.0),
                    'simulation_results': final_state.get('metadata', {}).get('simulation_results'),
                    'messages': final_state.get('messages', [])
                }

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Workflow resumption failed",
                extra={
                    "event": "workflow_resume_failed",
                    "campaign_id": campaign_id,
                    "content_id": content_id,
                    "error": str(e),
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )
            return {
                'success': False,
                'content_id': content_id,
                'error': str(e)
            }

    async def _update_bandit_arm_on_deployment(
        self,
        campaign_id: str,
        strategy: Dict[str, Any],
        success: bool,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Update bandit arm α/β parameters after deployment.
        
        CRITICAL FIX: This connects the deployment success to the experiment
        learning loop, allowing the bandit to learn from actual outcomes.
        
        Args:
            campaign_id: The campaign ID
            strategy: The selected strategy with arm info
            success: Whether deployment was technically successful
            metrics: Deployment metrics (CTR, impressions, etc.)
        """
        try:
            if not strategy:
                logger.debug(f"No strategy for campaign {campaign_id} — bandit update skipped")
                return
            arm_name = strategy.get('hook', strategy.get('arm_name', 'unknown'))
            experiment_id = strategy.get('experiment_id')
            
            if not experiment_id:
                logger.debug(f"No experiment_id in strategy for campaign {campaign_id} — bandit update skipped")
                return
            
            async with async_session_maker() as session:
                from sqlalchemy import select, update
                from ...data_layer.database.models import BanditArm, Experiment
                
                stmt = select(BanditArm).where(
                    BanditArm.experiment_id == UUID(experiment_id),
                    BanditArm.arm_name == arm_name
                )
                result = await session.execute(stmt)
                arm = result.scalar_one_or_none()
                
                if arm:
                    arm.pulls = (arm.pulls or 0) + 1
                    

                    predicted_ctr = metrics.get('ctr', metrics.get('predicted_ctr', 0.05))
                    
                    is_success = predicted_ctr > 0.03
                    
                    if is_success:
                        arm.successes = (arm.successes or 0) + 1
                        arm.alpha = (arm.alpha or 1.0) + 1.0
                    else:
                        arm.failures = (arm.failures or 0) + 1
                        arm.beta = (arm.beta or 1.0) + 1.0
                    
                    arm.total_reward = (arm.total_reward or 0.0) + predicted_ctr
                    arm.updated_at = datetime.utcnow()
                    
                    await session.commit()
                    
                    logger.info(
                        f"Updated BanditArm '{arm_name}': "
                        f"α={arm.alpha}, β={arm.beta}, pulls={arm.pulls}, "
                        f"success_rate={arm.successes / arm.pulls:.2%}"
                    )
                else:
                    logger.warning(
                        f"BanditArm '{arm_name}' not found for experiment {experiment_id}"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to update bandit arm: {e}", exc_info=True)