"""
Dashboard configuration module.

NOTE: Constants are fetched from backend API (SINGLE SOURCE OF TRUTH: src/shared/constants.py)
with local fallbacks for offline resilience.
"""

from .constants import (
    API_BASE_URL,
    
    PLATFORMS,
    PLATFORMS_EXTENDED,
    PLATFORM_DISPLAY_NAMES,
    PLATFORM_ICONS,
    
    CAMPAIGN_STATUSES,
    CONTENT_STATUSES,
    ACTIVE_STATUSES,
    INACTIVE_STATUSES,
    STATUS_COLORS,
    STATUS_ICONS,
    
    WORKFLOW_NODES,
    WORKFLOW_NODES_ALL,
    WORKFLOW_NODES_CORE,
    WORKFLOW_NODES_OPTIONAL,
    WORKFLOW_NODE_IDS,
    WORKFLOW_NODE_NAMES,
    WORKFLOW_NODE_ICONS,
    WORKFLOW_EVENT_TYPES,
    EVENT_ICONS,
    EVENT_TO_NODE_MAPPING,
    EVENT_SEVERITY_COLORS,
    ALERT_SEVERITIES,
    
    DEFAULT_THRESHOLDS,
    SAFETY_SCORE_TIERS,
    
    get_config_value,
    get_governance_thresholds,
    get_budget_thresholds,
    clear_config_cache,
    clear_constants_cache,
    
    get_status_color,
    get_status_icon,
    get_event_icon,
    get_workflow_node_from_event,
    get_platform_display,
    get_platform_icon,
)

__all__ = [
    'API_BASE_URL',
    'PLATFORMS',
    'PLATFORMS_EXTENDED',
    'PLATFORM_DISPLAY_NAMES', 
    'PLATFORM_ICONS',
    'CAMPAIGN_STATUSES',
    'CONTENT_STATUSES',
    'ACTIVE_STATUSES',
    'INACTIVE_STATUSES',
    'STATUS_COLORS',
    'STATUS_ICONS',
    'WORKFLOW_NODES',
    'WORKFLOW_NODES_ALL',
    'WORKFLOW_NODES_CORE',
    'WORKFLOW_NODES_OPTIONAL',
    'WORKFLOW_NODE_IDS',
    'WORKFLOW_NODE_NAMES',
    'WORKFLOW_NODE_ICONS',
    'WORKFLOW_EVENT_TYPES',
    'EVENT_ICONS',
    'EVENT_TO_NODE_MAPPING',
    'EVENT_SEVERITY_COLORS',
    'ALERT_SEVERITIES',
    'DEFAULT_THRESHOLDS',
    'SAFETY_SCORE_TIERS',
    'get_config_value',
    'get_governance_thresholds',
    'get_budget_thresholds',
    'clear_config_cache',
    'clear_constants_cache',
    'get_status_color',
    'get_status_icon',
    'get_event_icon',
    'get_workflow_node_from_event',
    'get_platform_display',
    'get_platform_icon',
]
