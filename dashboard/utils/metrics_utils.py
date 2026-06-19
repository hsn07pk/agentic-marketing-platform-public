"""
Centralized Metrics Utility Module

This module provides consistent metric formatting, calculation, and display
across all dashboard pages. Per Research Plan Section 10.2 (Unified KPI Dashboard),
all metrics should be calculated once and displayed consistently.

Key Design Decisions:
- CTR is stored/returned as PERCENTAGE (0-100 scale), e.g., 1.5 means 1.5%
- ROI is stored/returned as PERCENTAGE, e.g., 31.2 means 31.2%
- All formatting functions handle both decimal (0-1) and percentage (0-100) inputs
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class MetricFormat:
    """
    Standard formats for metric display.
    These match the Research Plan Section 10.2 KPI specifications.
    """
    CTR_DECIMALS = 2  # e.g., 1.58%
    ROI_DECIMALS = 1  # e.g., 31.2%
    CPL_DECIMALS = 2  # e.g., €25.50
    CONVERSION_RATE_DECIMALS = 2  # e.g., 2.35%

def normalize_ctr(ctr_value: float, source: str = "unknown") -> float:
    """
    Normalize CTR to percentage scale (0-100).

    After the data consistency fix, all CTR values should be stored as percentages.
    This provides backward compatibility: values <= 0.15 are treated as decimals
    and converted (threshold handles realistic CTR ranges of 0.5-10%).
    """
    if ctr_value is None:
        return 0.0

    # After the data consistency fix, CTR should always be percentage.
    # However, for backward compatibility:
    # - CTR > 0.15 is almost certainly already a percentage (very few CTRs exceed 15%)
    # - CTR <= 0.15 could be decimal (e.g., 0.0158 for 1.58%) - convert if it looks like decimal
    # - CTR that looks like decimal (very small) gets multiplied by 100
    if ctr_value <= 0.15:
        # Check if this is likely a decimal: typical CTR decimals are 0.001 to 0.10
        # If value * 100 gives a realistic CTR range (0.1% to 15%), it was decimal
        potential_percentage = ctr_value * 100
        if 0.1 <= potential_percentage <= 15:
            return potential_percentage

    return ctr_value

def format_ctr(ctr_value: float) -> str:
    """Format CTR for display as percentage string."""
    normalized = normalize_ctr(ctr_value)
    return f"{normalized:.{MetricFormat.CTR_DECIMALS}f}%"

def format_ctr_delta(current: float, previous: float) -> tuple[str, str]:
    """Format CTR change for delta display, returns (delta_value, delta_color)."""
    current_norm = normalize_ctr(current)
    previous_norm = normalize_ctr(previous)

    if previous_norm == 0:
        return None, "normal"

    delta = current_norm - previous_norm
    delta_pct = (delta / previous_norm) * 100

    delta_color = "normal" if delta >= 0 else "inverse"
    return f"{delta_pct:+.1f}%", delta_color

def normalize_roi(roi_value: float) -> float:
    """
    Normalize ROI to percentage scale.

    The API already calculates ROI as a percentage-like value.
    DO NOT multiply by 100 again.
    """
    if roi_value is None:
        return 0.0

    return roi_value

def format_roi(roi_value: float) -> str:
    """Format ROI for display as percentage string."""
    normalized = normalize_roi(roi_value)
    return f"{normalized:.{MetricFormat.ROI_DECIMALS}f}%"

def get_roi_delta_color(roi_value: float) -> str:
    """Return "normal" for positive ROI, "inverse" for negative."""
    return "normal" if roi_value >= 0 else "inverse"

def count_active_campaigns(campaigns: List[Dict[str, Any]]) -> int:
    """Count campaigns with status 'running' or 'active'."""
    active_statuses = ['running', 'active']
    return len([c for c in campaigns if c.get('status') in active_statuses])

def count_total_campaigns(campaigns: List[Dict[str, Any]]) -> int:
    """Count total campaigns regardless of status."""
    return len(campaigns)

def get_campaign_counts(campaigns: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get all campaign counts in a single call."""
    active_statuses = ['running', 'active']
    counts = {
        'active': 0,
        'total': len(campaigns),
        'by_status': {}
    }

    for campaign in campaigns:
        status = campaign.get('status', 'unknown')
        counts['by_status'][status] = counts['by_status'].get(status, 0) + 1
        if status in active_statuses:
            counts['active'] += 1

    return counts

def format_large_number(value: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if value is None:
        return "0"

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:,}"

def format_impressions(value: int) -> str:
    return format_large_number(value)

def format_clicks(value: int) -> str:
    return format_large_number(value)

def calculate_ctr(clicks: int, impressions: int) -> float:
    """Calculate CTR from clicks and impressions, returns percentage."""
    if impressions == 0:
        return 0.0
    return (clicks / impressions) * 100

def calculate_conversion_rate(conversions: int, clicks: int) -> float:
    """Calculate conversion rate from conversions and clicks, returns percentage."""
    if clicks == 0:
        return 0.0
    return (conversions / clicks) * 100

def calculate_roi(conversions: int, spent: float, conversion_value: float = 100.0) -> float:
    """
    Calculate ROI from conversions and spend.

    Formula: ROI = ((revenue - cost) / cost) * 100
    Default conversion value is €100 per Research Plan Section 10.2.
    """
    if spent <= 0:
        return 0.0

    revenue = conversions * conversion_value
    return ((revenue - spent) / spent) * 100

def format_cpl(cpl_value: float) -> str:
    """Format Cost Per Lead for display."""
    if cpl_value is None or cpl_value == 0:
        return "N/A"
    return f"€{cpl_value:.{MetricFormat.CPL_DECIMALS}f}"

def aggregate_campaign_metrics(campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple campaigns."""
    total_impressions = sum(c.get('impressions', 0) for c in campaigns)
    total_clicks = sum(c.get('clicks', 0) for c in campaigns)
    total_conversions = sum(c.get('conversions', 0) for c in campaigns)
    total_spent = sum(c.get('budget_spent', 0) for c in campaigns)

    return {
        'total_campaigns': len(campaigns),
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'total_conversions': total_conversions,
        'total_spent': total_spent,
        'average_ctr': calculate_ctr(total_clicks, total_impressions),
        'average_cpl': (total_spent / total_conversions) if total_conversions > 0 else 0,
        'roi': calculate_roi(total_conversions, total_spent)
    }

def get_period_label(days: int) -> str:
    """Get a human-readable label for a time period."""
    if days == 7:
        return "7d"
    elif days == 14:
        return "14d"
    elif days == 30:
        return "30d"
    elif days == 60:
        return "60d"
    elif days == 90:
        return "90d"
    else:
        return f"{days}d"

def get_comparison_period_dates(days: int) -> tuple[datetime, datetime, datetime, datetime]:
    """Get date ranges for current and previous period comparison."""
    now = datetime.utcnow()
    current_end = now
    current_start = now - timedelta(days=days)
    previous_end = current_start
    previous_start = previous_end - timedelta(days=days)

    return current_start, current_end, previous_start, previous_end

def validate_metrics_response(metrics: Dict[str, Any]) -> bool:
    """Validate that a metrics response contains required fields."""
    required_fields = [
        'total_campaigns',
        'total_impressions',
        'total_clicks',
        'total_conversions',
        'average_ctr',
        'roi'
    ]
    return all(field in metrics for field in required_fields)

def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metrics dictionary: replace None with 0, ensure numeric types."""
    defaults = {
        'total_campaigns': 0,
        'total_impressions': 0,
        'total_clicks': 0,
        'total_conversions': 0,
        'total_spent': 0.0,
        'average_ctr': 0.0,
        'average_cpl': 0.0,
        'roi': 0.0,
        'period_days': 30
    }

    sanitized = {}
    for key, default in defaults.items():
        value = metrics.get(key)
        if value is None:
            sanitized[key] = default
        elif isinstance(default, int):
            sanitized[key] = int(value)
        elif isinstance(default, float):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value

    for key, value in metrics.items():
        if key not in sanitized:
            sanitized[key] = value

    return sanitized
