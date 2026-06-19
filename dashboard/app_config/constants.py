"""
Dashboard Configuration Constants

IMPORTANT: The authoritative constants are defined in src/shared/constants.py
This file should NOT duplicate those values, only fetch them.
"""

import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
import requests
import os

logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")

_cached_constants: Optional[Dict[str, Any]] = None

def _fetch_constants_from_api() -> Dict[str, Any]:
    """Fetch all constants from the backend API."""
    global _cached_constants
    
    if _cached_constants is not None:
        return _cached_constants
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/config/constants",
            timeout=5
        )
        response.raise_for_status()
        _cached_constants = response.json()
        logger.info("Successfully fetched constants from API")
        return _cached_constants
    except Exception as e:
        logger.warning(f"Failed to fetch constants from API: {e}. Using fallbacks.")
        return {}

def clear_constants_cache():
    """Clear the cached constants to force a fresh fetch."""
    global _cached_constants
    _cached_constants = None

def _get_constant(key: str, fallback: Any) -> Any:
    """Get a constant from API cache or use fallback."""
    constants = _fetch_constants_from_api()
    return constants.get(key, fallback)

_FALLBACK_PLATFORMS: List[str] = ["linkedin", "twitter", "email", "blog"]
_FALLBACK_PLATFORMS_EXTENDED: List[str] = ["linkedin", "twitter", "email", "blog", "reddit", "medium"]
_FALLBACK_PLATFORM_DISPLAY_NAMES: Dict[str, str] = {
    "linkedin": "LinkedIn", "twitter": "Twitter/X", "email": "Email",
    "blog": "Blog", "reddit": "Reddit", "medium": "Medium",
}
_FALLBACK_PLATFORM_ICONS: Dict[str, str] = {
    "linkedin": "💼", "twitter": "🐦", "email": "📧",
    "blog": "📝", "reddit": "🔴", "medium": "✍️",
}

def get_platforms() -> List[str]:
    return _get_constant("platforms", _FALLBACK_PLATFORMS)

def get_platforms_extended() -> List[str]:
    return _get_constant("platforms_extended", _FALLBACK_PLATFORMS_EXTENDED)

def get_platform_display_names() -> Dict[str, str]:
    return _get_constant("platform_display_names", _FALLBACK_PLATFORM_DISPLAY_NAMES)

def get_platform_icons() -> Dict[str, str]:
    return _get_constant("platform_icons", _FALLBACK_PLATFORM_ICONS)

PLATFORMS: List[str] = _FALLBACK_PLATFORMS
PLATFORMS_EXTENDED: List[str] = _FALLBACK_PLATFORMS_EXTENDED
PLATFORM_DISPLAY_NAMES: Dict[str, str] = _FALLBACK_PLATFORM_DISPLAY_NAMES
PLATFORM_ICONS: Dict[str, str] = _FALLBACK_PLATFORM_ICONS

_FALLBACK_CAMPAIGN_STATUSES: List[str] = [
    "draft", "pending_approval", "approved", "running", "paused", "completed", "failed"
]
_FALLBACK_ACTIVE_STATUSES: List[str] = ["running", "active"]
_FALLBACK_INACTIVE_STATUSES: List[str] = ["paused", "draft", "completed", "failed"]

CAMPAIGN_STATUSES: List[str] = _FALLBACK_CAMPAIGN_STATUSES
ACTIVE_STATUSES: List[str] = _FALLBACK_ACTIVE_STATUSES
INACTIVE_STATUSES: List[str] = _FALLBACK_INACTIVE_STATUSES

_FALLBACK_CONTENT_STATUSES: List[str] = [
    "generated", "pending_review", "approved", "rejected", "deployed"
]

CONTENT_STATUSES: List[str] = _FALLBACK_CONTENT_STATUSES

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
    "running": "▶️", "active": "▶️", "paused": "⏸️", "draft": "📝",
    "completed": "✅", "failed": "❌", "pending": "⏳", "pending_approval": "⏳",
    "approved": "✅", "rejected": "❌", "deployed": "🚀", "generated": "✏️",
    "pending_review": "👁️", "error": "❌", "success": "✅", "warning": "⚠️", "info": "ℹ️",
}

WORKFLOW_NODES_ALL: List[Dict[str, Any]] = [
    {"id": "market_observation", "name": "Market Observation", "icon": "🔍", "optional": False},
    {"id": "strategy_optimization", "name": "Strategy Optimization", "icon": "📈", "optional": False},
    {"id": "content_generation", "name": "Content Generation", "icon": "✏️", "optional": False},
    {"id": "safety_validation", "name": "Safety Validation", "icon": "🛡️", "optional": False},
    {"id": "human_review", "name": "HITL Review", "icon": "👁️", "optional": True, "config_key": "REQUIRE_HUMAN_APPROVAL"},
    {"id": "cost_check", "name": "Cost Check", "icon": "💰", "optional": False},
    {"id": "simulation", "name": "Simulation", "icon": "🧪", "optional": False},
    {"id": "marl_gating", "name": "MARL Gating", "icon": "🤖", "optional": True, "config_key": "ENABLE_MARL"},
    {"id": "golden_test_gate", "name": "Golden Test Gate", "icon": "🏆", "optional": False},
    {"id": "canary_deployment", "name": "Canary Deployment", "icon": "🐤", "optional": True, "config_key": "ENABLE_CANARY_DEPLOYMENT"},
    {"id": "deployment", "name": "Deployment", "icon": "🚀", "optional": False},
]

WORKFLOW_NODES_CORE: List[Dict[str, str]] = [
    {"id": "market_observation", "name": "Market Observation", "icon": "🔍"},
    {"id": "strategy_optimization", "name": "Strategy Optimization", "icon": "📈"},
    {"id": "content_generation", "name": "Content Generation", "icon": "✏️"},
    {"id": "safety_validation", "name": "Safety Validation", "icon": "🛡️"},
    {"id": "cost_check", "name": "Cost Check", "icon": "💰"},
    {"id": "simulation", "name": "Simulation", "icon": "🧪"},
    {"id": "golden_test_gate", "name": "Golden Test Gate", "icon": "🏆"},
    {"id": "deployment", "name": "Deployment", "icon": "🚀"},
]

WORKFLOW_NODES_OPTIONAL: List[Dict[str, str]] = [
    {"id": "human_review", "name": "HITL Review", "icon": "👁️", "config_key": "REQUIRE_HUMAN_APPROVAL"},
    {"id": "marl_gating", "name": "MARL Gating", "icon": "🤖", "config_key": "ENABLE_MARL"},
    {"id": "canary_deployment", "name": "Canary Deployment", "icon": "🐤", "config_key": "ENABLE_CANARY_DEPLOYMENT"},
]

WORKFLOW_NODES: List[Dict[str, str]] = WORKFLOW_NODES_ALL

WORKFLOW_NODE_IDS: List[str] = [n["id"] for n in WORKFLOW_NODES_ALL]
WORKFLOW_NODE_NAMES: Dict[str, str] = {n["id"]: n["name"] for n in WORKFLOW_NODES_ALL}
WORKFLOW_NODE_ICONS: Dict[str, str] = {n["id"]: n["icon"] for n in WORKFLOW_NODES_ALL}

WORKFLOW_EVENT_TYPES: List[str] = [
    "workflow_started", "workflow_paused", "workflow_resumed",
    "workflow_completed", "workflow_failed",
    "node_started", "node_completed", "node_failed",
    "content_generated", "content_approved", "content_rejected", "content_deployed",
    "safety_check_passed", "safety_check_failed", "hitl_queue_added", "hitl_reviewed",
    "deployment_started", "deployment_success", "deployment_failed",
    "canary_started", "canary_promoted", "canary_rolled_back",
    "budget_warning", "budget_exceeded", "cost_check_passed", "cost_check_failed",
    "error_occurred", "retry_attempted",
]

# Knowledge base document categories (single source — used by Operations page)
KB_CATEGORIES: List[str] = ["general", "strategy", "technical", "competitor", "research"]

# Default persona identifiers (fallback when API unavailable)
_FALLBACK_PERSONAS: List[str] = ["decision_maker", "practitioner", "researcher"]

def get_personas() -> List[str]:
    return _get_constant("personas", _FALLBACK_PERSONAS)

PERSONAS: List[str] = _FALLBACK_PERSONAS

EVENT_ICONS: Dict[str, str] = {
    "workflow_started": "▶️", "workflow_paused": "⏸️", "workflow_resumed": "▶️",
    "workflow_completed": "✅", "workflow_failed": "❌",
    "node_started": "🔄", "node_completed": "✓", "node_failed": "⚠️",
    "content_generated": "✏️", "content_approved": "✅", "content_rejected": "❌",
    "content_deployed": "🚀", "safety_check_passed": "🛡️", "safety_check_failed": "⚠️",
    "hitl_queue_added": "📋", "hitl_reviewed": "👁️",
    "deployment_started": "🚀", "deployment_success": "✅", "deployment_failed": "❌",
    "canary_started": "🐤", "canary_promoted": "📈", "canary_rolled_back": "⏮️",
    "budget_warning": "💰", "budget_exceeded": "🚨",
    "cost_check_passed": "✅", "cost_check_failed": "❌",
    "error_occurred": "❌", "retry_attempted": "🔄",
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

ALERT_SEVERITIES: List[str] = ["info", "warning", "error", "critical"]

EVENT_SEVERITY_COLORS: Dict[str, str] = {
    "info": "#3b82f6", "warning": "#f59e0b",
    "error": "#ef4444", "critical": "#dc2626",
}

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "SAFETY_SCORE_THRESHOLD": 0.80,
    "TOXICITY_THRESHOLD": 0.10,
    "AUTO_APPROVE_THRESHOLD": 0.95,
    "MIN_SAFETY_SCORE": 0.70,
    "BRAND_ALIGNMENT_THRESHOLD": 0.85,
    "FACTUALITY_THRESHOLD": 0.70,
    "BUDGET_WARNING_THRESHOLD": 75,
    "BUDGET_CRITICAL_THRESHOLD": 90,
    "BUDGET_AUTO_PAUSE_THRESHOLD": 98,
    "SIMULATION_ACCURACY_TARGET": 0.90,
    "MAPE_TARGET": 10,
}

SAFETY_SCORE_TIERS: List[Dict[str, Any]] = [
    {"range": "0.85-1.0", "label": "Excellent", "color": "#10b981"},
    {"range": "0.7-0.85", "label": "Good", "color": "#3b82f6"},
    {"range": "0.5-0.7", "label": "Medium", "color": "#f59e0b"},
    {"range": "0.0-0.5", "label": "Low", "color": "#ef4444"},
]

@lru_cache(maxsize=1)
def _fetch_config_category(category: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/config/category/{category}",
            timeout=5
        )
        response.raise_for_status()
        configs = response.json()
        result = {}
        for config in configs:
            key = config.get("key")
            value = config.get("display_value")
            value_type = config.get("value_type", "string")
            if value_type == "float":
                result[key] = float(value)
            elif value_type == "integer":
                result[key] = int(value)
            elif value_type == "boolean":
                result[key] = value.lower() == "true"
            else:
                result[key] = value
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch config category '{category}': {e}")
        return {}

def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value by key, falling back to DEFAULT_THRESHOLDS."""
    if key in DEFAULT_THRESHOLDS:
        return DEFAULT_THRESHOLDS.get(key, default)
    return default

def get_governance_thresholds() -> Dict[str, float]:
    backend_config = _fetch_config_category("governance")
    return {
        "safety_score": backend_config.get("SAFETY_SCORE_THRESHOLD", DEFAULT_THRESHOLDS["SAFETY_SCORE_THRESHOLD"]),
        "toxicity": backend_config.get("TOXICITY_THRESHOLD", DEFAULT_THRESHOLDS["TOXICITY_THRESHOLD"]),
        "auto_approve": backend_config.get("AUTO_APPROVE_THRESHOLD", DEFAULT_THRESHOLDS["AUTO_APPROVE_THRESHOLD"]),
        "min_safety": backend_config.get("MIN_SAFETY_SCORE", DEFAULT_THRESHOLDS["MIN_SAFETY_SCORE"]),
        "brand_alignment": DEFAULT_THRESHOLDS["BRAND_ALIGNMENT_THRESHOLD"],
        "factuality": DEFAULT_THRESHOLDS["FACTUALITY_THRESHOLD"],
    }

def get_budget_thresholds() -> Dict[str, int]:
    return {
        "warning": DEFAULT_THRESHOLDS["BUDGET_WARNING_THRESHOLD"],
        "critical": DEFAULT_THRESHOLDS["BUDGET_CRITICAL_THRESHOLD"],
        "auto_pause": DEFAULT_THRESHOLDS["BUDGET_AUTO_PAUSE_THRESHOLD"],
    }

def clear_config_cache():
    _fetch_config_category.cache_clear()
    clear_constants_cache()

def get_status_color(status: str) -> Dict[str, str]:
    return STATUS_COLORS.get(status.lower(), STATUS_COLORS["unknown"])

def get_status_icon(status: str) -> str:
    return STATUS_ICONS.get(status.lower(), "❓")

def get_event_icon(event_type: str) -> str:
    return EVENT_ICONS.get(event_type, "ℹ️")

def get_workflow_node_from_event(event_type: str) -> Optional[str]:
    return EVENT_TO_NODE_MAPPING.get(event_type)

def get_platform_display(platform: str) -> str:
    return PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform.title())

def get_platform_icon(platform: str) -> str:
    return PLATFORM_ICONS.get(platform.lower(), "🌐")

def _init_constants():
    global PLATFORMS, PLATFORMS_EXTENDED, PLATFORM_DISPLAY_NAMES, PLATFORM_ICONS
    global CAMPAIGN_STATUSES, ACTIVE_STATUSES, INACTIVE_STATUSES, CONTENT_STATUSES
    
    constants = _fetch_constants_from_api()
    if constants:
        PLATFORMS = constants.get("platforms", PLATFORMS)
        PLATFORMS_EXTENDED = constants.get("platforms_extended", PLATFORMS_EXTENDED)
        PLATFORM_DISPLAY_NAMES = constants.get("platform_display_names", PLATFORM_DISPLAY_NAMES)
        PLATFORM_ICONS = constants.get("platform_icons", PLATFORM_ICONS)
        CAMPAIGN_STATUSES = constants.get("campaign_statuses", CAMPAIGN_STATUSES)
        ACTIVE_STATUSES = constants.get("active_statuses", ACTIVE_STATUSES)
        INACTIVE_STATUSES = constants.get("inactive_statuses", INACTIVE_STATUSES)
        CONTENT_STATUSES = constants.get("content_statuses", CONTENT_STATUSES)

_init_constants()
