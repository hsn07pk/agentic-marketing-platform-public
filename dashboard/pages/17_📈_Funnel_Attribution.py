"""
Full Funnel Attribution - Track Campaign Performance Through Entire Sales Funnel
Integrates Cal.com bookings and HubSpot CRM for complete attribution
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.metrics_utils import normalize_roi
from components import render_metric_card, create_gauge_chart
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Funnel Attribution - Agentic AI", page_icon="📈", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("📈 Full Funnel Attribution")
st.caption("Track campaign performance from impression to closed deal")

with st.expander("ℹ️ Funnel Attribution Guide", expanded=False):
    st.markdown("""
**What this page tracks:** The full marketing funnel — **Impressions → Clicks → Leads → Bookings → Shows → Closed Won** — with multi-touch attribution modeling to understand which campaigns and channels drive real revenue.

**Integration requirements:**
- **Cal.com** — booking tracking (discovery calls, demos)
- **HubSpot** — CRM/deal tracking (pipeline, lifecycle stages, revenue)
- **Mailgun** — email campaign tracking

---

**Funnel stages explained:**
| Stage | Description |
|-------|-------------|
| **Impressions** | Total content views across platforms |
| **Clicks** | Users who clicked on content |
| **Leads** | Users who took a meaningful action (form fill, email reply, etc.) |
| **Bookings** | Users who booked a discovery call via Cal.com |
| **Shows** | Users who actually attended the booked call |
| **Closed Won** | Deals that converted to paying customers |

---

**Key KPIs:**
- **Cost-Per-Booked-Call (CPBC):** Total spend / number of bookings — *lower is better*
- **Show Rate:** Shows / Bookings — *target >70%*
- **Close Rate:** Closed Won / Shows — *target >20%*
- **Campaign ROI:** (Revenue − Cost) / Cost × 100 — *positive = profitable*
- **Lead Quality Score:** Derived from conversion rates (0–100)

---

**Multi-Touch Attribution Models:**
- **Linear:** Equal credit to all touchpoints
- **First Touch:** 100% credit to the first interaction
- **Last Touch:** 100% credit to the last interaction before conversion
- **Time Decay:** More credit to recent touchpoints (exponential decay)
- **U-Shaped:** 40% first, 40% last, 20% split across middle touchpoints
- **W-Shaped:** 30% first, 30% middle, 30% last, 10% split across remaining

---

**When ROI shows N/A:** No revenue data is available yet. Configure HubSpot or Cal.com integrations and close deals to populate revenue.

**Data sources:** Platform metrics (worker-collected), Cal.com API, HubSpot API, Mailgun API.
    """)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90), help="Date range for funnel analysis. Default 90 days. Wider range captures more conversion data.")
with col2:
    end_date = st.date_input("End Date", value=datetime.now(), help="End date for funnel analysis. Defaults to today.")
with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.markdown("---")
_calcom_connected = False
_hubspot_connected = False
try:
    integration_status = api.get_integration_status()

    col_cal, col_hub, col_mail, col_llm = st.columns(4)

    with col_cal:
        calcom = integration_status.get('calcom', {})
        _calcom_connected = calcom.get('connected', False)
        if _calcom_connected:
            st.success("📅 **Cal.com** Connected")
        else:
            st.error("📅 **Cal.com** Disconnected")

    with col_hub:
        hubspot = integration_status.get('hubspot', {})
        _hubspot_connected = hubspot.get('connected', False)
        if _hubspot_connected:
            st.success("🔗 **HubSpot** Connected")
        else:
            st.error("🔗 **HubSpot** Disconnected")

    with col_mail:
        mailgun = integration_status.get('mailgun', {})
        if mailgun.get('connected', False):
            st.success("📧 **Mailgun** Connected")
        elif mailgun.get('configured', False):
            st.warning("📧 **Mailgun** Configured")
        else:
            st.error("📧 **Mailgun** Not Configured")

    with col_llm:
        ollama = integration_status.get('ollama', {})
        if ollama.get('connected', False):
            st.success("🤖 **Ollama** Active")
        else:
            st.info("☁️ **OpenAI** Active")

except Exception as e:
    st.warning(f"Could not check integration status: {e}")


try:
    funnel_data = api.request("GET", "/funnel/attribution/overview", params={
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    })
except Exception as e:
    logger.debug(f"Funnel attribution data unavailable: {e}")
    funnel_data = {
        "funnel_stages": {
            "impressions": 0,
            "clicks": 0,
            "leads": 0,
            "bookings": 0,
            "shows": 0,
            "closed_won": 0
        },
        "metrics": {},
        "campaigns": []
    }

st.markdown("---")
st.subheader("📊 Conversion Funnel")
st.caption("Visualizes drop-off at each stage from impression to closed deal. Percentages show retention relative to initial impressions.")

funnel_stages = funnel_data.get("funnel_stages", {})

stages = ["Impressions", "Clicks", "Leads", "Bookings", "Shows", "Closed Won"]
values = [
    funnel_stages.get("impressions", 0),
    funnel_stages.get("clicks", 0),
    funnel_stages.get("leads", 0),
    funnel_stages.get("bookings", 0),
    funnel_stages.get("shows", 0),
    funnel_stages.get("closed_won", 0)
]

if any(v > 0 for v in values):
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe"]
        ),
        connector=dict(line=dict(color="rgba(102, 126, 234, 0.3)", dash="dot", width=3))
    ))

    fig.update_layout(
        title="Campaign Conversion Funnel",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key="funnel_main")
else:
    _missing = []
    if not _calcom_connected:
        _missing.append("**Cal.com** (`CALENDAR_API_KEY`)")
    if not _hubspot_connected:
        _missing.append("**HubSpot** (`HUBSPOT_API_KEY`)")
    if _missing:
        st.info(f"📊 Funnel data requires integration with {' and '.join(_missing)}. Configure keys in ⚙️ Operations → System Settings.")
    else:
        st.info("📊 No funnel data yet. Run campaigns to populate the conversion funnel.")

st.markdown("---")
st.subheader("📋 Funnel-Specific KPIs")
st.caption("Key performance indicators derived from funnel conversion data. Requires Cal.com and/or HubSpot integration for full data.")

metrics = funnel_data.get("metrics", {})

_has_kpi_data = any(v for v in metrics.values() if isinstance(v, (int, float)) and v != 0)

if _has_kpi_data:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cpbc = metrics.get("cost_per_booked_call", 0)
        cpbc_change = metrics.get("cpbc_change", 0)
        st.metric(
            "Cost-Per-Booked-Call",
            f"€{cpbc:.2f}",
            delta=f"{cpbc_change:+.1f}%" if cpbc_change else None,
            delta_color="inverse",
            help="Total ad spend divided by number of booked calls. Lower is better. Decrease (green) means cheaper bookings."
        )

    with col2:
        booked_rate = metrics.get("booked_call_rate", 0)
        booked_change = metrics.get("booked_rate_change", 0)
        st.metric(
            "Booked Call Rate",
            f"{booked_rate:.1f}%",
            delta=f"{booked_change:+.1f}%" if booked_change else None,
            help="Percentage of leads who booked a discovery call. Higher indicates better lead nurturing."
        )

    with col3:
        show_rate = metrics.get("show_rate", 0)
        show_change = metrics.get("show_rate_change", 0)
        st.metric(
            "Show Rate",
            f"{show_rate:.1f}%",
            delta=f"{show_change:+.1f}%" if show_change else None,
            help="Percentage of booked calls where the lead actually attended. Target >70%. Low rate suggests reminder/follow-up issues."
        )

    with col4:
        avg_lead_quality = metrics.get("avg_lead_quality", 0)
        st.metric(
            "Avg Lead Quality Score",
            f"{avg_lead_quality:.1f}/100",
            help="Average quality score (0–100) derived from lead conversion behavior. Higher means leads are more likely to convert."
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cpl = metrics.get("cost_per_lead", 0)
        st.metric("Cost-Per-Lead (CPL)", f"€{cpl:.2f}", help="Total spend divided by number of leads generated. Lower is better.")

    with col2:
        close_rate = metrics.get("close_rate", 0)
        st.metric("Close Rate", f"{close_rate:.1f}%", help="Percentage of shows that converted to closed-won deals. Target >20%.")

    with col3:
        total_revenue = metrics.get("total_revenue", 0)
        st.metric("Revenue Generated", f"€{total_revenue:,.2f}", help="Total revenue from closed-won deals attributed to campaigns in this period.")
        if metrics.get("revenue_source") == "estimated":
            st.caption("⚠️ No bookings yet — revenue unavailable")

    with col4:
        roi = metrics.get("roi", 0)
        if total_revenue == 0 and roi < 0:
            st.metric("Campaign ROI", "N/A", help="(Revenue − Cost) / Cost × 100. N/A when no revenue data is available.")
            st.caption("No revenue data to calculate ROI")
        else:
            st.metric("Campaign ROI", f"{roi:.1f}%", delta_color="normal", help="(Revenue − Cost) / Cost × 100. Positive means profitable campaigns.")
else:
    _missing = []
    if not _calcom_connected:
        _missing.append("Cal.com")
    if not _hubspot_connected:
        _missing.append("HubSpot")
    if _missing:
        st.info(f"📋 KPI metrics require {' and '.join(_missing)} integration. Configure in ⚙️ Operations → System Settings.")
    else:
        st.info("📋 No KPI data yet. Run campaigns and track conversions to see funnel metrics.")

st.markdown("---")
st.subheader("📅 Cal.com Booking Attribution")
st.caption("Tracks discovery call bookings from Cal.com — links campaigns to booked calls, shows, and no-shows.")

if not _calcom_connected:
    st.info("📅 Cal.com is not connected. Configure `CALENDAR_API_KEY` and `CALENDAR_API_URL` in ⚙️ Operations → System Settings → calendar category to enable booking tracking.")
else:
    tab1, tab2, tab3 = st.tabs(["Recent Bookings", "Booking Metrics", "Attribution Analysis"])

    with tab1:
        try:
            bookings = api.request("GET", "/funnel/calendar/bookings", params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "limit": 50
            })
        except Exception as e:
            logger.debug(f"Funnel attribution data unavailable: {e}")
            bookings = {"bookings": [], "count": 0}

        booking_list = bookings.get("bookings", [])

        if booking_list:
            booking_df = pd.DataFrame([
                {
                    "Booking ID": b.get("id", ""),
                    "Lead Email": (b.get("attendees", [{}])[0].get("email", "") if b.get("attendees") else b.get("attendee_email", "")),
                    "Event Type": b.get("title", b.get("event_type", "Demo")),
                    "Scheduled": (b.get("startTime", "") or b.get("start_time", ""))[:16].replace("T", " "),
                    "Status": b.get("status", "unknown"),
                    "Campaign": b.get("campaign_name", "Direct"),
                    "Source": b.get("source", "Cal.com")
                }
                for b in booking_list
            ])

            st.dataframe(booking_df, use_container_width=True, hide_index=True)
        else:
            st.info("No bookings found in the selected date range")

    with tab2:
        try:
            booking_metrics = api.request("GET", "/funnel/calendar/metrics", params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
        except Exception as e:
            logger.debug(f"Funnel attribution data unavailable: {e}")
            booking_metrics = {"metrics": {}}

        bm = booking_metrics.get("metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Bookings", bm.get("total_bookings", 0), help="Total discovery calls booked via Cal.com in this period.")
        with col2:
            st.metric("Completed", bm.get("completed", 0), help="Bookings where both parties attended the call.")
        with col3:
            st.metric("No-Shows", bm.get("no_shows", 0), help="Bookings where the lead did not attend. High no-shows suggest follow-up/reminder issues.")
        with col4:
            st.metric("Cancelled", bm.get("cancelled", 0), help="Bookings cancelled before the scheduled time.")

        st.markdown("### Booking Trend")

        try:
            booking_trend = api.request("GET", "/funnel/calendar/trend", params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            trend_data = booking_trend.get("daily", [])
        except Exception as e:
            logger.debug(f"Funnel attribution data unavailable: {e}")
            trend_data = []

        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            fig = px.line(
                trend_df,
                x="date",
                y="bookings",
                title="Daily Bookings",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True, key="funnel_channel_perf")
        else:
            st.info("📅 No booking trend data available.")

    with tab3:
        st.markdown("### Campaign → Booking Attribution")

        try:
            attribution = api.request("GET", "/funnel/attribution/by-campaign", params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
        except Exception as e:
            logger.debug(f"Funnel attribution data unavailable: {e}")
            attribution = {"campaigns": []}

        campaigns = attribution.get("campaigns", [])

        if campaigns:
            attr_df = pd.DataFrame(campaigns)
            st.dataframe(attr_df, use_container_width=True, hide_index=True)

            if len(campaigns) > 1:
                fig = px.pie(
                    attr_df,
                    values="bookings",
                    names="campaign_name",
                    title="Bookings by Campaign"
                )
                st.plotly_chart(fig, use_container_width=True, key="funnel_channel_roi")
        else:
            st.info("No attribution data available. Start campaigns to track attribution.")

st.markdown("---")
st.subheader("💼 HubSpot CRM Attribution")
st.caption("CRM pipeline data from HubSpot — deal stages, lead quality scoring, and lifecycle progression.")

if not _hubspot_connected:
    st.info("💼 HubSpot is not connected. Configure `HUBSPOT_API_KEY` in ⚙️ Operations → System Settings → hubspot category to enable CRM tracking.")
else:
    tab1, tab2, tab3 = st.tabs(["Deal Pipeline", "Lead Quality", "Lifecycle Stages"])

    with tab1:
        try:
            pipeline = api.request("GET", "/funnel/hubspot/deals", params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
        except Exception as e:
            logger.debug(f"Funnel attribution data unavailable: {e}")
            pipeline = {"deals": [], "by_stage": {}}

        by_stage = pipeline.get("by_stage", {})

        if by_stage:
            stages = list(by_stage.keys())
            counts = list(by_stage.values())

            fig = go.Figure(data=[
                go.Bar(x=stages, y=counts, marker_color='#667eea')
            ])
            fig.update_layout(
                title="Deals by Pipeline Stage",
                xaxis_title="Stage",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True, key="funnel_time_series")
        else:
            st.info("💼 No deal pipeline data available yet.")

    with tab2:
        st.markdown("### Lead Quality Analysis")

        try:
            lead_quality = api.request("GET", "/funnel/hubspot/lead-quality", params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
        except Exception as e:
            logger.debug(f"Funnel attribution data unavailable: {e}")
            lead_quality = {"distribution": {}, "avg_score": 0}

        distribution = lead_quality.get("distribution", {})
        avg_score = lead_quality.get("avg_score", 0)
        
        if distribution or avg_score > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = create_gauge_chart(
                    value=avg_score,
                    max_value=100,
                    title="Average Lead Score",
                    thresholds={'low': 40, 'medium': 70, 'high': 100}
                )
                st.plotly_chart(fig, use_container_width=True, key="funnel_touchpoint_sankey")

            with col2:
                if distribution:
                    dist_df = pd.DataFrame({
                        'Score Range': list(distribution.keys()),
                        'Count': list(distribution.values())
                    })

                    fig = px.bar(
                        dist_df,
                        x='Score Range',
                        y='Count',
                        title="Lead Score Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="funnel_touchpoint_heatmap")
                else:
                    st.info("📊 No lead score distribution available.")
        else:
            st.info("📈 No lead quality data available yet.")

    with tab3:
        st.markdown("### Lifecycle Stage Progression")

        try:
            lifecycle = api.request("GET", "/funnel/hubspot/lifecycle-stages", params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
        except Exception as e:
            logger.debug(f"Funnel attribution data unavailable: {e}")
            lifecycle = {"stages": {}}

        stages = lifecycle.get("stages", {})

        if stages:
            stage_names = list(stages.keys())
            stage_values = list(stages.values())

            fig = go.Figure(data=[
                go.Bar(
                    x=stage_values,
                    y=[s.replace("_", " ").title() for s in stage_names],
                    orientation='h',
                    marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'][:len(stage_names)]
                )
            ])
            fig.update_layout(
                title="Contacts by Lifecycle Stage",
                xaxis_title="Count",
                yaxis_title="Stage"
            )
            st.plotly_chart(fig, use_container_width=True, key="funnel_conversion_path")
        else:
            st.info("📊 No lifecycle stage data available yet.")

st.markdown("---")
st.subheader("⏱️ Delayed Reward Attribution")
st.caption("Tracks conversions that occur after the initial touchpoint — attributing delayed bookings and deals back to the originating campaign.")

try:
    pending_rewards = api.get_pending_rewards()
except Exception as e:
    logger.debug(f"Funnel attribution data unavailable: {e}")
    pending_rewards = {"total": 0, "pending_rewards": []}

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pending Attributions", pending_rewards.get("total", 0), help="Number of conversion events awaiting attribution to their originating campaign.")

with col2:
    try:
        reward_stats = api.request("GET", "/rewards/stats")
        attributed = reward_stats.get("attributed_this_week", 0)
    except Exception as e:
        logger.debug(f"Funnel attribution data unavailable: {e}")
        attributed = 0
    st.metric("Attributed This Week", attributed, help="Number of delayed rewards successfully attributed to campaigns this week.")

with col3:
    if st.button("🔄 Process Pending Rewards"):
        with st.spinner("Processing..."):
            result = api.process_pending_rewards()
            st.success(f"Processed: {result.get('checked', 0)} checked, {result.get('booked', 0)} attributed")

pending_list = pending_rewards.get("pending_rewards", [])

if pending_list:
    st.markdown("### Pending Reward Entries")

    pending_df = pd.DataFrame([
        {
            "Lead Email": p.get("lead_email", ""),
            "Campaign": p.get("campaign_name", ""),
            "Action": p.get("action", ""),
            "Created": p.get("created_at", ""),
            "Expires": p.get("expires_at", ""),
            "Status": p.get("status", "pending")
        }
        for p in pending_list[:20]
    ])

    st.dataframe(pending_df, use_container_width=True, hide_index=True)
else:
    st.info("No pending delayed rewards. All attributions are up to date.")

st.markdown("---")
st.subheader("🔀 Multi-Touch Attribution Analysis")
st.caption("Distributes conversion credit across multiple campaign touchpoints using different attribution models.")

col1, col2 = st.columns([2, 1])

with col1:
    attribution_model = st.selectbox(
        "Attribution Model",
        ["linear", "first_touch", "last_touch", "time_decay", "u_shaped", "w_shaped"],
        help="How conversion credit is distributed across touchpoints. See the guide at the top of this page for detailed model descriptions."
    )

with col2:
    st.metric("Selected Model", attribution_model.replace("_", " ").title(), help="Currently active attribution model used for credit distribution calculations below.")

model_descriptions = {
    "first_touch": "100% credit to first interaction that initiated the journey",
    "last_touch": "100% credit to last interaction before conversion",
    "linear": "Equal credit distributed across all touchpoints in the journey",
    "time_decay": "More credit to recent touchpoints (exponential decay)",
    "u_shaped": "40% first touch, 40% last touch, 20% shared by middle touchpoints",
    "w_shaped": "30% first, 30% middle, 30% last touchpoints; 10% shared by others"
}

st.info(f"**{attribution_model.replace('_', ' ').title()}:** {model_descriptions.get(attribution_model, 'Custom attribution model')}")

st.markdown("### Attribution Model Comparison")

try:
    attribution_comparison = api.request("GET", "/funnel/attribution/multi-touch", params={
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "model": attribution_model
    })
except Exception as e:
    logger.debug(f"Funnel attribution data unavailable: {e}")
    attribution_comparison = {"has_data": False}

if attribution_comparison.get("has_data"):
    campaign_credits = attribution_comparison.get("campaign_credits", {})
    channel_credits = attribution_comparison.get("channel_credits", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        if campaign_credits:
            st.markdown("#### Credit by Campaign")
            campaigns = list(campaign_credits.keys())
            credits = list(campaign_credits.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=campaigns,
                    y=credits,
                    text=[f"€{c:.2f}" for c in credits],
                    textposition='outside',
                    marker_color='#667eea'
                )
            ])
            fig.update_layout(
                title="Attribution Credit by Campaign",
                xaxis_title="Campaign",
                yaxis_title="Attributed Value (€)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True, key="funnel_segment_compare")
    
    with col2:
        if channel_credits:
            st.markdown("#### Credit by Channel")
            fig = go.Figure(data=[go.Pie(
                labels=list(channel_credits.keys()),
                values=list(channel_credits.values()),
                textinfo='label+percent',
                marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'])
            )])
            fig.update_layout(
                title="Attribution Credit by Channel",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True, key="funnel_segment_detail")

else:
    st.info("💡 Run campaigns with multiple touchpoints to see multi-touch attribution analysis")

st.markdown("### Available Attribution Models")

model_info = pd.DataFrame([
    {"Model": "First Touch", "Description": "100% credit to first interaction", "Best For": "Awareness campaigns"},
    {"Model": "Last Touch", "Description": "100% credit before conversion", "Best For": "Direct response campaigns"},
    {"Model": "Linear", "Description": "Equal credit to all touchpoints", "Best For": "Balanced analysis"},
    {"Model": "Time Decay", "Description": "Recent touchpoints weighted more", "Best For": "Short sales cycles"},
    {"Model": "U-Shaped", "Description": "40-20-40 distribution", "Best For": "Lead generation"},
    {"Model": "W-Shaped", "Description": "30-10-30-10-20 distribution", "Best For": "B2B complex sales"}
])

st.dataframe(model_info, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("📋 Campaign Funnel Performance")
st.caption("Detailed per-campaign breakdown of funnel metrics. Export to CSV for offline analysis.")

try:
    campaign_funnel = api.request("GET", "/funnel/attribution/campaigns", params={
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    })
except Exception as e:
    logger.debug(f"Funnel attribution data unavailable: {e}")
    campaign_funnel = {"campaigns": []}

campaign_list = campaign_funnel.get("campaigns", [])

if campaign_list:
    perf_df = pd.DataFrame(campaign_list)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    csv_data = perf_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Funnel Data",
        data=csv_data,
        file_name=f"funnel_attribution_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("📋 No campaign funnel data available. Run campaigns to see funnel metrics.")

st.markdown("---")
st.caption(f"Full Funnel Attribution | Data from {start_date} to {end_date} | Last updated: {datetime.now().strftime('%H:%M:%S')}")
