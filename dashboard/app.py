"""
Home Dashboard - Executive Overview
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))
from utils.api_client import AgenticAPIClient
from utils.metrics_utils import (
    normalize_ctr, normalize_roi, get_roi_delta_color,
    count_active_campaigns, get_campaign_counts, sanitize_metrics
)
from components import render_metric_card, render_status_badge, create_line_chart

st.set_page_config(page_title="Home - Agentic AI", page_icon="🏠", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

try:
    mock_status = api.get_mock_mode_status()
    if mock_status.get('mock_mode_enabled', False):
        settings = mock_status.get('settings', {})
        enabled_mocks = []
        if settings.get('ENABLE_MOCK_DEPLOYMENT', False):
            enabled_mocks.append("Deployment")
        if settings.get('ENABLE_MOCK_EXPERIMENTS', False):
            enabled_mocks.append("Experiments")
        
        mock_list = ", ".join(enabled_mocks) if enabled_mocks else "All components"
        st.warning(f"🧪 **MOCK MODE ENABLED** | Simulated data in: {mock_list}. KPIs may not reflect real platform performance. Configure in System Transparency → Configuration.")
except Exception:
    pass

st.title("🏠 Home Dashboard")
st.caption("Executive overview of your autonomous marketing platform")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🔋 System Health")

with col2:
    if st.button("🔄 Refresh", type="primary"):
        st.rerun()

health = api.get_detailed_health()

if health.get("status") == "healthy":
    st.success("✅ All systems operational")

    components = health.get("components", {})
    if components:
        cols = st.columns(len(components))

        for idx, (name, status) in enumerate(components.items()):
            with cols[idx]:
                if status.get("status") == "healthy":
                    st.metric(name.capitalize(), "✅ OK")
                else:
                    st.metric(name.capitalize(), "⚠️ Issue")
                    st.caption(status.get("message", "Unknown error"))
else:
    st.error("⚠️ System degraded - check System Monitor page")

st.markdown("---")

st.subheader("📊 Key Performance Indicators (Last 7 Days)")

try:
    metrics_7d = sanitize_metrics(api.get_metrics_overview(days=7))
    campaigns = api.get_campaigns(limit=1000)

    if metrics_7d.get('includes_mock_data', False):
        st.info("🧪 **Includes Mock Data** — These KPIs include simulated campaign metrics. Toggle off via Configuration → INCLUDE_MOCK_IN_METRICS.")

    campaign_counts = get_campaign_counts(campaigns)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        active = campaign_counts['active']
        total = campaign_counts['total']
        render_metric_card("Active Campaigns", active, delta=f"{total} total")

    with col2:
        impressions = metrics_7d.get('total_impressions', 0)
        render_metric_card("Impressions (7d)", f"{impressions:,}", prefix="", suffix="")

    with col3:
        clicks = metrics_7d.get('total_clicks', 0)
        render_metric_card("Clicks (7d)", f"{clicks:,}")

    with col4:
        ctr = normalize_ctr(metrics_7d.get('average_ctr', 0))
        render_metric_card("Avg CTR (7d)", f"{ctr:.2f}", suffix="%")

    with col5:
        conversions = metrics_7d.get('total_conversions', 0)
        render_metric_card("Conversions (7d)", conversions)

    with col6:
        # ROI from API is already in percentage form (e.g., 31.2 means 31.2%)
        # DO NOT multiply by 100
        roi = normalize_roi(metrics_7d.get('roi', 0))
        delta_color = get_roi_delta_color(roi)
        render_metric_card("ROI (7d)", f"{roi:.1f}", suffix="%", delta=f"{roi:.1f}%", delta_color=delta_color)

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🔔 Alerts & Notifications")

        alerts_data = api.get_active_alerts()
        total_alerts = alerts_data.get('total_count', 0)

        if total_alerts > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                critical = alerts_data.get('critical_count', 0)
                if critical > 0:
                    st.metric("🔴 Critical", critical)
            with col2:
                errors = alerts_data.get('error_count', 0)
                if errors > 0:
                    st.metric("⚠️ Errors", errors)
            with col3:
                warnings = alerts_data.get('warning_count', 0)
                if warnings > 0:
                    st.metric("⚡ Warnings", warnings)
            with col4:
                info = alerts_data.get('info_count', 0)
                if info > 0:
                    st.metric("ℹ️ Info", info)

            alerts = alerts_data.get('alerts', {})
            if alerts.get('critical'):
                st.markdown("**🔴 CRITICAL ALERTS:**")
                for alert in alerts['critical'][:3]:  # Show top 3
                    st.error(f"**{alert['title']}** - {alert['message']}")

            if alerts.get('error'):
                st.markdown("**⚠️ ERROR ALERTS:**")
                for alert in alerts['error'][:3]:  # Show top 3
                    st.warning(f"**{alert['title']}** - {alert['message']}")

            st.info(f"👉 View all **{total_alerts} alerts** on the **📡 System Transparency** page")
        else:
            st.success("✅ No active alerts - all systems running smoothly")

        hitl_queue = api.get_hitl_queue(status="pending", limit=100)
        if len(hitl_queue) > 0:
            st.warning(f"⚠️ **{len(hitl_queue)} content items** awaiting review in HITL queue. Go to **👁️ Governance** page to review.")

    with col_right:
        st.subheader("⚡ Quick Actions")

        st.markdown("""
        **Use the sidebar to access:**

        - 📋 **Campaigns** - Create and manage campaigns
        - 📊 **Analytics** - View performance metrics
        - 🔬 **Experiments** - Run A/B tests
        - 👁️ **Governance** - Review content queue
        - 🔍 **System Monitor** - Check system health
        """)

    st.markdown("---")

    st.subheader("📝 Recent Activity")

    recent = sorted(campaigns, key=lambda x: x.get('created_at', ''), reverse=True)[:10]

    activity_data = []
    for c in recent:
        campaign_ctr = normalize_ctr(c.get('ctr', 0))
        activity_data.append({
            "Time": c.get('created_at', 'N/A')[:19] if c.get('created_at') else 'N/A',
            "Campaign": c.get('name', 'Unnamed'),
            "Platform": c.get('platform', 'unknown').capitalize(),
            "Status": c.get('status', 'unknown'),
            "Impressions": c.get('impressions', 0),
            "Clicks": c.get('clicks', 0),
            "CTR": f"{campaign_ctr:.2f}%"
        })

    if activity_data:
        df = pd.DataFrame(activity_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity")

except Exception as e:
    st.error(f"Failed to load dashboard: {str(e)}")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
