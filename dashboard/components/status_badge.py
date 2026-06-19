"""
Status badge component for displaying status indicators
"""
import streamlit as st
from typing import Dict


def render_status_badge(status: str, custom_styles: Dict[str, str] = None) -> None:
    """
    Render a status badge with appropriate styling
    
    Args:
        status: Status string (e.g., "running", "paused", "completed", "failed")
        custom_styles: Optional custom style overrides
    """
    default_styles = {
        "running": {"bg": "#10b981", "text": "white", "label": "RUNNING"},
        "active": {"bg": "#10b981", "text": "white", "label": "ACTIVE"},
        "paused": {"bg": "#f59e0b", "text": "white", "label": "PAUSED"},
        "draft": {"bg": "#6b7280", "text": "white", "label": "DRAFT"},
        "completed": {"bg": "#3b82f6", "text": "white", "label": "COMPLETED"},
        "failed": {"bg": "#ef4444", "text": "white", "label": "FAILED"},
        "pending": {"bg": "#8b5cf6", "text": "white", "label": "PENDING"},
        "approved": {"bg": "#10b981", "text": "white", "label": "APPROVED"},
        "rejected": {"bg": "#ef4444", "text": "white", "label": "REJECTED"},
        "deployed": {"bg": "#059669", "text": "white", "label": "DEPLOYED"},
        "error": {"bg": "#dc2626", "text": "white", "label": "ERROR"},
        "success": {"bg": "#16a34a", "text": "white", "label": "SUCCESS"},
        "warning": {"bg": "#f59e0b", "text": "white", "label": "WARNING"},
        "info": {"bg": "#3b82f6", "text": "white", "label": "INFO"},
    }
    
    # Merge custom styles if provided
    if custom_styles:
        default_styles.update(custom_styles)
    
    status_lower = status.lower() if status else "unknown"
    style = default_styles.get(status_lower, {"bg": "#9ca3af", "text": "white", "label": status.upper()})
    
    st.markdown(f"""
    <span style="
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: {style['bg']};
        color: {style['text']};
        border-radius: 0.375rem;
        font-size: 0.875rem;
        font-weight: 600;
        text-align: center;
        white-space: nowrap;
    ">
        {style['label']}
    </span>
    """, unsafe_allow_html=True)


def render_icon_badge(icon: str, label: str, color: str = "#3b82f6") -> None:
    """
    Render a badge with icon and label
    
    Args:
        icon: Emoji or icon character
        label: Badge label text
        color: Badge background color
    """
    st.markdown(f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.375rem 0.875rem;
        background-color: {color};
        color: white;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        font-weight: 500;
    ">
        <span style="font-size: 1.25rem;">{icon}</span>
        {label}
    </span>
    """, unsafe_allow_html=True)


def render_count_badge(count: int, label: str, color: str = "#3b82f6") -> None:
    """
    Render a count badge (e.g., "5 Pending")
    
    Args:
        count: Count value
        label: Label text
        color: Badge color
    """
    st.markdown(f"""
    <div style="
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
        color: white;
        border-radius: 0.5rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{count}</span>
        <span style="font-size: 0.875rem; opacity: 0.9;">{label}</span>
    </div>
    """, unsafe_allow_html=True)
