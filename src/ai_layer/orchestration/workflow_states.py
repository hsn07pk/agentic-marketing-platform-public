from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum

from ...config.configuration_service import get_runtime_config

class WorkflowStep(str, Enum):
    """Workflow step enumeration - OODA-G Loop per Research Plan Section 3"""
    START = "start"
    MARKET_OBSERVATION = "market_observation"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    CONTENT_GENERATION = "content_generation"
    SAFETY_VALIDATION = "safety_validation"
    HUMAN_REVIEW = "human_review"
    COST_CHECK = "cost_check"
    SIMULATION = "simulation"
    MARL_GATING = "marl_gating"
    DEPLOYMENT = "deployment"
    END = "end"

class WorkflowState(TypedDict):
    """
    State definition for LangGraph workflow
    All fields must be present for proper state management

    OODA-G Loop State per Research Plan Section 3:
    - OBSERVE: market_insights (MarketScraper data)
    - ORIENT: strategy (StrategyOptimizer output)
    - DECIDE: content_id (ContentGenerator output)
    - ACT: deployment_status (Deployer result)
    - GOVERN: safety_score, requires_human_review, cost_accumulated
    """
    campaign_id: str
    content_id: Optional[str]
    messages: List[Dict[str, Any]]
    current_step: WorkflowStep
    market_insights: Optional[Dict[str, Any]]
    strategy: Optional[Dict[str, Any]]
    safety_score: Optional[float]
    cost_accumulated: float
    requires_human_review: bool
    human_feedback: Optional[str]
    simulation_results: Optional[Dict[str, Any]]
    simulation_passed: Optional[bool]
    golden_test_results: Optional[Dict[str, Any]]
    golden_test_approved: Optional[bool]
    golden_test_pass_rate: Optional[float]
    marl_approved: Optional[bool]
    deployment_status: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]
    retry_count: int

class ContentGenerationState(TypedDict):
    """State for content generation step"""
    persona: str
    goal: str
    platform: str
    context: List[str]
    generated_content: Optional[Dict[str, str]]
    generation_timestamp: Optional[str]

class SafetyValidationState(TypedDict):
    """State for safety validation step"""
    content_id: str
    content_text: str
    safety_score: float
    toxicity_score: float
    brand_score: float
    claim_citations: List[str]
    validation_passed: bool
    validation_timestamp: Optional[str]

class HumanReviewState(TypedDict):
    """State for human review step"""
    content_id: str
    review_priority: str
    assigned_reviewer: Optional[str]
    review_decision: Optional[str]
    review_feedback: Optional[str]
    reviewed_at: Optional[str]

class DeploymentState(TypedDict):
    """State for deployment step"""
    content_id: str
    platform: str
    deployment_target: str
    deployment_status: str
    platform_post_id: Optional[str]
    deployed_at: Optional[str]
    deployment_metrics: Optional[Dict[str, Any]]

def create_initial_state(campaign_id: str, config: Dict[str, Any]) -> WorkflowState:
    """
    Create initial workflow state

    Args:
        campaign_id: Campaign UUID
        config: Campaign configuration

    Returns:
        Initial WorkflowState with OODA-G fields initialized
    """
    return WorkflowState(
        campaign_id=campaign_id,
        content_id=None,
        messages=[],
        current_step=WorkflowStep.START,
        market_insights=None,
        strategy=None,
        safety_score=None,
        cost_accumulated=0.0,
        requires_human_review=False,
        human_feedback=None,
        simulation_results=None,
        simulation_passed=None,
        golden_test_results=None,
        golden_test_approved=None,
        golden_test_pass_rate=None,
        marl_approved=None,
        deployment_status=None,
        error=None,
        metadata=config,
        retry_count=0
    )

def update_state(
    current_state: WorkflowState,
    updates: Dict[str, Any]
) -> WorkflowState:
    """
    Update workflow state with new values
    
    Args:
        current_state: Current state
        updates: Dictionary of updates
    
    Returns:
        Updated WorkflowState
    """
    new_state = current_state.copy()
    new_state.update(updates)
    return new_state

def add_message(
    state: WorkflowState,
    role: str,
    content: str
) -> WorkflowState:
    """
    Add message to state
    
    Args:
        state: Current state
        role: Message role (user/assistant/system)
        content: Message content
    
    Returns:
        Updated state with new message
    """
    new_state = state.copy()
    new_state['messages'].append({
        "role": role,
        "content": content,
        "timestamp": str(datetime.now())
    })
    return new_state

def should_review_human(state: WorkflowState) -> bool:
    """
    Determine if content requires human review.
    
    Per research plan (Section 6.3):
    - Score > 0.9 (high confidence) → direct to deployer (skip HITL)
    - Score 0.7-0.9 (medium confidence) → goes to HITL review
    - Score ≤ 0.7 (low confidence) → back to content_generator
    
    Configuration is read from DATABASE (single source of truth),
    NOT from environment variables.

    Args:
        state: Current workflow state

    Returns:
        Boolean indicating if human review is needed
    """
    require_human_approval = get_runtime_config('REQUIRE_HUMAN_APPROVAL', False)
    auto_approve_threshold = get_runtime_config('AUTO_APPROVE_THRESHOLD', 0.90)  # Per research plan: > 0.9
    min_safety_score = get_runtime_config('MIN_SAFETY_SCORE', 0.70)
    
    if require_human_approval:
        return True

    if state['safety_score'] is None:
        return True

    # Below minimum → route to review as safety net (ideally should regenerate)
    if state['safety_score'] < min_safety_score:
        return True
    
    if state['safety_score'] > auto_approve_threshold:
        return False

    # Score in [min, auto_approve] → HITL review per research plan
    return True

def can_deploy(state: WorkflowState) -> bool:
    """
    Determine if content can be deployed.
    
    Configuration is read from DATABASE (single source of truth).
    Human-approved content bypasses safety score threshold.
    
    Args:
        state: Current workflow state
    
    Returns:
        Boolean indicating if deployment is allowed
    """
    if state['error'] is not None:
        return False
    
    # Human approval overrides safety score threshold
    if state.get('approval_status') != 'approved':
        safety_threshold = get_runtime_config('SAFETY_SCORE_THRESHOLD', 0.8)
        
        # Round to 2 decimal places to handle floating point precision (0.79999... should pass 0.8)
        safety_score = state["safety_score"]
        if safety_score is None:
            return False
        if round(safety_score, 2) < safety_threshold:
            return False
    
    if state['requires_human_review'] and state['human_feedback'] is None:
        return False
    
    budget = state['metadata'].get('budget', float('inf'))
    if state['cost_accumulated'] >= budget:
        return False
    
    return True

from datetime import datetime