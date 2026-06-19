"""
SINGLE SOURCE OF TRUTH for all shared constants.

This file defines ALL constants used by both backend and frontend.
Do NOT duplicate these values elsewhere.

Backend: Import directly from this module
Frontend: Fetch via GET /api/v1/config/constants
"""

from enum import Enum
from typing import Dict, Any, List


class Platform(str, Enum):
    """Supported marketing platforms for content deployment."""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    EMAIL = "email"
    BLOG = "blog"


PLATFORMS: List[str] = [p.value for p in Platform]

PLATFORMS_EXTENDED: List[str] = [
    "linkedin",
    "twitter",
    "email",
    "blog",
    "reddit",
    "medium",
]

PLATFORM_DISPLAY_NAMES: Dict[str, str] = {
    "linkedin": "LinkedIn",
    "twitter": "Twitter/X",
    "email": "Email",
    "blog": "Blog",
    "reddit": "Reddit",
    "medium": "Medium",
}

PLATFORM_ICONS: Dict[str, str] = {
    "linkedin": "💼",
    "twitter": "🐦",
    "email": "📧",
    "blog": "📝",
    "reddit": "🔴",
    "medium": "📝",
}


class CampaignGoal(str, Enum):
    """Campaign goal types."""
    LEAD_GENERATION = "lead_generation"
    BRAND_AWARENESS = "brand_awareness"
    CONVERSION = "conversion"


CAMPAIGN_GOALS: List[str] = [g.value for g in CampaignGoal]


class CampaignStatus(str, Enum):
    """Campaign lifecycle states."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


CAMPAIGN_STATUSES: List[str] = [s.value for s in CampaignStatus]
ACTIVE_STATUSES: List[str] = ["running", "active"]
INACTIVE_STATUSES: List[str] = ["paused", "draft", "completed", "failed"]


class ContentStatus(str, Enum):
    """Content review/deployment states."""
    GENERATED = "generated"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


CONTENT_STATUSES: List[str] = [s.value for s in ContentStatus]


class WorkflowEventType(str, Enum):
    """All possible workflow event types for tracking and logging."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    
    CONTENT_GENERATED = "content_generated"
    CONTENT_APPROVED = "content_approved"
    CONTENT_REJECTED = "content_rejected"
    CONTENT_DEPLOYED = "content_deployed"
    
    SAFETY_CHECK_PASSED = "safety_check_passed"
    SAFETY_CHECK_FAILED = "safety_check_failed"
    HITL_QUEUE_ADDED = "hitl_queue_added"
    HITL_REVIEWED = "hitl_reviewed"
    
    DEPLOYMENT_STARTED = "deployment_started"
    DEPLOYMENT_SUCCESS = "deployment_success"
    DEPLOYMENT_FAILED = "deployment_failed"
    CANARY_STARTED = "canary_started"
    CANARY_PROMOTED = "canary_promoted"
    CANARY_ROLLED_BACK = "canary_rolled_back"
    
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    COST_CHECK_PASSED = "cost_check_passed"
    COST_CHECK_FAILED = "cost_check_failed"
    
    ERROR_OCCURRED = "error_occurred"
    RETRY_ATTEMPTED = "retry_attempted"


WORKFLOW_EVENT_TYPES: List[str] = [e.value for e in WorkflowEventType]

WORKFLOW_NODES: List[Dict[str, str]] = [
    {"id": "market_observation", "name": "Market Observation", "icon": "🔍"},
    {"id": "content_generation", "name": "Content Generation", "icon": "✏️"},
    {"id": "safety_validation", "name": "Safety Validation", "icon": "🛡️"},
    {"id": "human_review", "name": "HITL Review", "icon": "👁️"},
    {"id": "cost_check", "name": "Cost Check", "icon": "💰"},
    {"id": "golden_test_gate", "name": "Golden Test Gate", "icon": "🏆"},
    {"id": "strategy_optimization", "name": "Strategy Optimization", "icon": "📈"},
    {"id": "deployment", "name": "Deployment", "icon": "🚀"},
    {"id": "canary_deployment", "name": "Canary Deployment", "icon": "🐤"},
    {"id": "simulation", "name": "Simulation", "icon": "🧪"},
    {"id": "marl_gating", "name": "MARL Gating", "icon": "🤖"},
]

WORKFLOW_NODE_IDS: List[str] = [n["id"] for n in WORKFLOW_NODES]
WORKFLOW_NODE_NAMES: Dict[str, str] = {n["id"]: n["name"] for n in WORKFLOW_NODES}
WORKFLOW_NODE_ICONS: Dict[str, str] = {n["id"]: n["icon"] for n in WORKFLOW_NODES}

EVENT_ICONS: Dict[str, str] = {
    "workflow_started": "▶️",
    "workflow_paused": "⏸️",
    "workflow_resumed": "▶️",
    "workflow_completed": "✅",
    "workflow_failed": "❌",
    "node_started": "🔄",
    "node_completed": "✓",
    "node_failed": "⚠️",
    "content_generated": "✏️",
    "content_approved": "✅",
    "content_rejected": "❌",
    "content_deployed": "🚀",
    "safety_check_passed": "🛡️",
    "safety_check_failed": "⚠️",
    "hitl_queue_added": "📋",
    "hitl_reviewed": "👁️",
    "deployment_started": "🚀",
    "deployment_success": "✅",
    "deployment_failed": "❌",
    "canary_started": "🐤",
    "canary_promoted": "📈",
    "canary_rolled_back": "⏮️",
    "budget_warning": "💰",
    "budget_exceeded": "🚨",
    "cost_check_passed": "✅",
    "cost_check_failed": "❌",
    "error_occurred": "❌",
    "retry_attempted": "🔄",
}

EVENT_TO_NODE_MAPPING: Dict[str, str] = {
    "content_generated": "content_generation",
    "content_approved": "human_review",
    "content_rejected": "human_review",
    "safety_check_passed": "safety_validation",
    "safety_check_failed": "safety_validation",
    "hitl_queue_added": "human_review",
    "hitl_reviewed": "human_review",
    "cost_check_passed": "cost_check",
    "cost_check_failed": "cost_check",
    "deployment_started": "deployment",
    "deployment_success": "deployment",
    "deployment_failed": "deployment",
    "content_deployed": "deployment",
    "canary_started": "canary_deployment",
    "canary_promoted": "canary_deployment",
    "canary_rolled_back": "canary_deployment",
}


class AlertSeverity(str, Enum):
    """Alert severity levels for system notifications."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ENGAGEMENT = "engagement"  # Special: high engagement alert
    TRAFFIC = "traffic"        # Special: traffic anomaly alert


ALERT_SEVERITIES: List[str] = [s.value for s in AlertSeverity]

EVENT_SEVERITY_COLORS: Dict[str, str] = {
    "info": "#3b82f6",      # Blue
    "warning": "#f59e0b",   # Amber
    "error": "#ef4444",     # Red
    "critical": "#dc2626",  # Dark Red
    "engagement": "#10b981", # Green
    "traffic": "#8b5cf6",    # Purple
}


GOVERNANCE_THRESHOLDS: Dict[str, float] = {
    "SAFETY_SCORE_THRESHOLD": 0.80,      # Minimum safety score for auto-approval
    "TOXICITY_THRESHOLD": 0.10,          # Maximum toxicity allowed
    "AUTO_APPROVE_THRESHOLD": 0.95,      # Score for auto-approval (skip HITL)
    "MIN_SAFETY_SCORE": 0.70,            # Absolute minimum safety score
    "BRAND_ALIGNMENT_THRESHOLD": 0.85,   # Brand voice alignment
    "FACTUALITY_THRESHOLD": 0.70,        # Claim verification threshold
}

# Budget control thresholds (percentages)
BUDGET_THRESHOLDS: Dict[str, int] = {
    "WARNING_THRESHOLD": 75,      # Show warning at 75% utilization
    "CRITICAL_THRESHOLD": 90,     # Show critical alert at 90%
    "AUTO_PAUSE_THRESHOLD": 98,   # Auto-pause campaigns at 98%
}

# MARL (Multi-Agent Reinforcement Learning) thresholds
MARL_THRESHOLDS: Dict[str, float] = {
    "POLICY_PROMOTION_THRESHOLD": 0.20,  # 20% improvement required
    "ROLLBACK_THRESHOLD": 0.05,          # 5% degradation triggers rollback
    "MIN_TRAINING_EPISODES": 100,        # Minimum episodes before evaluation
    "EXPLORATION_RATE": 0.15,            # Epsilon for exploration
}

# Simulation accuracy thresholds (from research plan)
SIMULATION_THRESHOLDS: Dict[str, float] = {
    "ACCURACY_TARGET": 0.90,      # 90% simulation accuracy target
    "MAPE_TARGET": 10.0,          # MAPE < 10% for accuracy
    "CONFIDENCE_INTERVAL": 0.95,  # 95% confidence interval
}

ALL_THRESHOLDS: Dict[str, Any] = {
    **GOVERNANCE_THRESHOLDS,
    **{f"BUDGET_{k}": v for k, v in BUDGET_THRESHOLDS.items()},
    **{f"MARL_{k}": v for k, v in MARL_THRESHOLDS.items()},
    **{f"SIMULATION_{k}": v for k, v in SIMULATION_THRESHOLDS.items()},
}


STATUS_COLORS: Dict[str, Dict[str, str]] = {
    "running": {"bg": "#10b981", "text": "white", "label": "RUNNING"},
    "active": {"bg": "#10b981", "text": "white", "label": "ACTIVE"},
    "paused": {"bg": "#f59e0b", "text": "white", "label": "PAUSED"},
    "draft": {"bg": "#6b7280", "text": "white", "label": "DRAFT"},
    "completed": {"bg": "#3b82f6", "text": "white", "label": "COMPLETED"},
    "failed": {"bg": "#ef4444", "text": "white", "label": "FAILED"},
    "pending": {"bg": "#8b5cf6", "text": "white", "label": "PENDING"},
    "pending_approval": {"bg": "#8b5cf6", "text": "white", "label": "PENDING APPROVAL"},
    "approved": {"bg": "#10b981", "text": "white", "label": "APPROVED"},
    "rejected": {"bg": "#ef4444", "text": "white", "label": "REJECTED"},
    "deployed": {"bg": "#059669", "text": "white", "label": "DEPLOYED"},
    "generated": {"bg": "#3b82f6", "text": "white", "label": "GENERATED"},
    "pending_review": {"bg": "#f59e0b", "text": "white", "label": "PENDING REVIEW"},
    "error": {"bg": "#dc2626", "text": "white", "label": "ERROR"},
    "success": {"bg": "#16a34a", "text": "white", "label": "SUCCESS"},
    "warning": {"bg": "#f59e0b", "text": "white", "label": "WARNING"},
    "info": {"bg": "#3b82f6", "text": "white", "label": "INFO"},
    "unknown": {"bg": "#9ca3af", "text": "white", "label": "UNKNOWN"},
}

STATUS_ICONS: Dict[str, str] = {
    "running": "▶️",
    "active": "▶️",
    "paused": "⏸️",
    "draft": "📝",
    "completed": "✅",
    "failed": "❌",
    "pending": "⏳",
    "pending_approval": "⏳",
    "approved": "✅",
    "rejected": "❌",
    "deployed": "🚀",
    "generated": "✏️",
    "pending_review": "👁️",
    "error": "❌",
    "success": "✅",
    "warning": "⚠️",
    "info": "ℹ️",
}

SAFETY_SCORE_TIERS: List[Dict[str, Any]] = [
    {"min": 0.85, "max": 1.0, "label": "Excellent", "color": "#10b981"},
    {"min": 0.70, "max": 0.85, "label": "Good", "color": "#3b82f6"},
    {"min": 0.50, "max": 0.70, "label": "Medium", "color": "#f59e0b"},
    {"min": 0.00, "max": 0.50, "label": "Low", "color": "#ef4444"},
]


def get_all_constants_dict() -> Dict[str, Any]:
    """
    Export all constants as a serializable dictionary.
    Used by API endpoint to serve constants to frontend.
    """
    return {
        # Platform
        "platforms": PLATFORMS,
        "platforms_extended": PLATFORMS_EXTENDED,
        "platform_display_names": PLATFORM_DISPLAY_NAMES,
        "platform_icons": PLATFORM_ICONS,
        
        # Campaign Status
        "campaign_statuses": CAMPAIGN_STATUSES,
        "active_statuses": ACTIVE_STATUSES,
        "inactive_statuses": INACTIVE_STATUSES,
        
        # Content Status
        "content_statuses": CONTENT_STATUSES,
        
        # Workflow
        "workflow_event_types": WORKFLOW_EVENT_TYPES,
        "workflow_nodes": WORKFLOW_NODES,
        "workflow_node_ids": WORKFLOW_NODE_IDS,
        "workflow_node_names": WORKFLOW_NODE_NAMES,
        "workflow_node_icons": WORKFLOW_NODE_ICONS,
        "event_icons": EVENT_ICONS,
        "event_to_node_mapping": EVENT_TO_NODE_MAPPING,
        
        # Alerts
        "alert_severities": ALERT_SEVERITIES,
        "event_severity_colors": EVENT_SEVERITY_COLORS,
        
        # Thresholds
        "governance_thresholds": GOVERNANCE_THRESHOLDS,
        "budget_thresholds": BUDGET_THRESHOLDS,
        "marl_thresholds": MARL_THRESHOLDS,
        "simulation_thresholds": SIMULATION_THRESHOLDS,
        "all_thresholds": ALL_THRESHOLDS,
        
        # Display
        "status_colors": STATUS_COLORS,
        "status_icons": STATUS_ICONS,
        "safety_score_tiers": SAFETY_SCORE_TIERS,
    }
