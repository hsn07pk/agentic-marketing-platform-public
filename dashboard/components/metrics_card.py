"""
Metric card components for displaying KPIs and statistics
"""
import streamlit as st
from typing import Optional, Union


def render_metric_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
    prefix: str = "",
    suffix: str = ""
):
    """
    Render a metric card with optional delta and help text
    
    Args:
        label: Metric label
        value: Metric value
        delta: Change indicator (optional)
        delta_color: Color for delta ("normal", "inverse", "off")
        help_text: Tooltip help text
        prefix: Prefix for value (e.g., "€")
        suffix: Suffix for value (e.g., "%", "K", "M")
    """
    formatted_value = f"{prefix}{value}{suffix}"
    
    if delta is not None and str(delta).strip():
        st.metric(
            label=label,
            value=formatted_value,
            delta=delta,
            delta_color=delta_color,
            help=help_text
        )
    else:
        st.metric(
            label=label,
            value=formatted_value,
            help=help_text
        )


def render_status_card(
    title: str,
    status: str,
    message: str,
    icon: str = "ℹ️",
    show_timestamp: bool = False
):
    """
    Render a status card with icon and message
    
    Args:
        title: Card title
        status: Status level ("success", "warning", "error", "info")
        message: Status message
        icon: Icon to display
        show_timestamp: Whether to show timestamp
    """
    status_colors = {
        "success": "#10b981",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "info": "#3b82f6"
    }
    
    color = status_colors.get(status, "#6b7280")
    
    st.markdown(f"""
    <div style="
        padding: 1rem;
        border-left: 4px solid {color};
        background-color: rgba(59, 130, 246, 0.1);
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <strong style="font-size: 1.1rem;">{title}</strong>
        </div>
        <p style="margin: 0; color: #4b5563;">{message}</p>
    </div>
    """, unsafe_allow_html=True)


def format_large_number(num: Union[int, float]) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(int(num))


def format_currency(amount: float, currency: str = "€") -> str:
    """
    Format currency with proper separators
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{currency}{amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage value
    
    Args:
        value: Value to format (as decimal, e.g., 0.05 for 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"
