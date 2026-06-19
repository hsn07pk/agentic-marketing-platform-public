"""
Shared constants and configuration - SINGLE SOURCE OF TRUTH.

This module contains all constants, enums, and configuration values
that are shared between backend and frontend.

Backend Usage:
    from src.shared.constants import Platform, CampaignStatus, THRESHOLDS

Frontend Usage:
    Fetch via API: GET /api/v1/config/constants
"""

from .constants import (
    Platform,
    PLATFORMS,
    PLATFORMS_EXTENDED,
    PLATFORM_DISPLAY_NAMES,
    PLATFORM_ICONS,
    
    CampaignGoal,
    CAMPAIGN_GOALS,
    
    CampaignStatus,
    CAMPAIGN_STATUSES,
    ACTIVE_STATUSES,
    INACTIVE_STATUSES,
    
    ContentStatus,
    CONTENT_STATUSES,
    
    WorkflowEventType,
    WORKFLOW_EVENT_TYPES,
    WORKFLOW_NODES,
    WORKFLOW_NODE_IDS,
    WORKFLOW_NODE_NAMES,
    WORKFLOW_NODE_ICONS,
    EVENT_ICONS,
    EVENT_TO_NODE_MAPPING,
    
    AlertSeverity,
    ALERT_SEVERITIES,
    EVENT_SEVERITY_COLORS,
    
    GOVERNANCE_THRESHOLDS,
    BUDGET_THRESHOLDS,
    MARL_THRESHOLDS,
    SIMULATION_THRESHOLDS,
    ALL_THRESHOLDS,
    
    STATUS_COLORS,
    STATUS_ICONS,
    SAFETY_SCORE_TIERS,
    
    get_all_constants_dict,
)

__all__ = [
    'Platform',
    'PLATFORMS',
    'PLATFORMS_EXTENDED',
    'PLATFORM_DISPLAY_NAMES',
    'PLATFORM_ICONS',
    'CampaignGoal',
    'CAMPAIGN_GOALS',
    'CampaignStatus',
    'CAMPAIGN_STATUSES',
    'ACTIVE_STATUSES',
    'INACTIVE_STATUSES',
    'ContentStatus',
    'CONTENT_STATUSES',
    'WorkflowEventType',
    'WORKFLOW_EVENT_TYPES',
    'WORKFLOW_NODES',
    'WORKFLOW_NODE_IDS',
    'WORKFLOW_NODE_NAMES',
    'WORKFLOW_NODE_ICONS',
    'EVENT_ICONS',
    'EVENT_TO_NODE_MAPPING',
    'AlertSeverity',
    'ALERT_SEVERITIES',
    'EVENT_SEVERITY_COLORS',
    'GOVERNANCE_THRESHOLDS',
    'BUDGET_THRESHOLDS',
    'MARL_THRESHOLDS',
    'SIMULATION_THRESHOLDS',
    'ALL_THRESHOLDS',
    'STATUS_COLORS',
    'STATUS_ICONS',
    'SAFETY_SCORE_TIERS',
    'get_all_constants_dict',
]
