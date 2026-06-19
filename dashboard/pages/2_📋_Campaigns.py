"""
Campaign Management - Complete CRUD and Workflow Control
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta


sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.metrics_utils import normalize_ctr, normalize_roi
from utils.data_controls import render_data_controls, render_searchable_select
from utils.llm_checks import check_llm_readiness, render_llm_status_banner
from components import (
    render_metric_card, render_status_badge,
    render_data_table, export_to_csv, confirm_action
)
from app_config import PLATFORMS, CAMPAIGN_STATUSES
from app_config.constants import PERSONAS

st.set_page_config(page_title="Campaigns - Agentic AI", page_icon="📋", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("📋 Campaign Management")
st.caption("Create, monitor, and optimize marketing campaigns")

# ── LLM readiness check — warn before user tries to start campaigns ─────
_llm_status = render_llm_status_banner(api, context="campaign_start")

with st.expander("ℹ️ How Campaign Management Works", expanded=False):
    st.markdown("""
**What are campaigns?**
Campaigns are the core unit of marketing activity. Each campaign targets a specific platform (LinkedIn, Twitter, or Email),
audience persona, and business goal. The AI agent pipeline generates tailored content for each campaign.

**Campaign Lifecycle:**
`DRAFT` → `PENDING_APPROVAL` → `APPROVED` → `RUNNING` → `COMPLETED`
- **Draft**: Initial state after creation. Configure settings before launching.
- **Pending Approval**: Content generated, awaiting human review.
- **Approved**: Ready for deployment.
- **Running**: Actively deployed and collecting performance data.
- **Completed**: Finished — either budget depleted or end date reached.

**Budget & Auto-Completion:**
Daily spend is tracked automatically. When **98% of the budget** is spent, the campaign auto-completes
and final metrics are recorded.

**MARL Policy Gating:**
Before a new MARL-optimized policy is deployed, it must demonstrate **>10% lift** over the current policy
in simulation. This prevents regressions.

**Tips for Creating Effective Campaigns:**
- Use descriptive names that include the quarter, platform, and persona (e.g., "Q1 2025 LinkedIn Decision Makers").
- Start with a moderate budget to test, then scale up based on CTR and conversion data.
- Use 'canary' deployment for high-budget campaigns to catch issues early.

**Content Deployment:**
The system uses LangGraph AI agents to generate platform-specific content, then deploys via API integrations
to LinkedIn, Twitter, or Email. Mock mode simulates deployment for testing.
    """.strip())

tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ All Campaigns",
    "➕ Create Campaign",
    "📊 Campaign Details",
    "⚙️ Bulk Operations"
])

with tab1:
    st.subheader("All Campaigns")
    st.caption("Browse, search, and manage all campaigns. Use filters to narrow results. Expand a campaign card for metrics and actions.")
    
    try:
        campaigns = api.get_campaigns(limit=1000)
        
        if campaigns:
            filtered_campaigns = render_data_controls(
                data=campaigns,
                search_fields=['name', 'id', 'platform', 'target_persona', 'goal'],
                filter_configs=[
                    {'field': 'status', 'label': 'Status', 'type': 'select', 'options': 'auto'},
                    {'field': 'platform', 'label': 'Platform', 'type': 'select', 'options': 'auto'},
                    {'field': 'target_persona', 'label': 'Persona', 'type': 'select', 'options': 'auto'},
                ],
                sort_options=['created_at', 'name', 'ctr', 'conversions', 'impressions', 'budget_spent'],
                key_prefix="campaigns"
            )
            
            st.markdown(f"**Found {len(filtered_campaigns)} campaigns**")
            
            for campaign in filtered_campaigns:
                with st.expander(
                    f"**{campaign.get('name', 'Unnamed')}** | "
                    f"{campaign.get('platform', 'unknown').capitalize()} | "
                    f"{campaign.get('status', 'unknown').upper()}",
                    expanded=False
                ):
                    col_info, col_metrics, col_actions = st.columns([2, 2, 1])
                    
                    with col_info:
                        st.markdown(f"**ID:** `{campaign.get('id')}`")
                        st.markdown(f"**Platform:** {campaign.get('platform', 'unknown').capitalize()}")
                        st.markdown(f"**Persona:** {campaign.get('target_persona', 'N/A')}")
                        st.markdown(f"**Goal:** {campaign.get('goal', 'N/A')}")
                        st.markdown(f"**Created:** {campaign.get('created_at', 'N/A')[:19]}")
                        
                        status = campaign.get('status', 'unknown')
                        render_status_badge(status)
                        
                        config = campaign.get('config', {})
                        is_mock = campaign.get('is_mock', config.get('is_mock', config.get('mock_mode', False)))
                        if is_mock:
                            st.warning("🧪 Mock Mode")
                    
                    with col_metrics:
                        st.metric("Impressions", f"{campaign.get('impressions', 0):,}", help="Total number of times campaign content was shown to users.")
                        st.metric("Clicks", f"{campaign.get('clicks', 0):,}", help="Total clicks on campaign content. Higher clicks indicate engaging copy.")
                        ctr_value = normalize_ctr(campaign.get('ctr', 0))
                        st.metric("CTR", f"{ctr_value:.2f}%", help="Click-Through Rate = Clicks ÷ Impressions × 100. Above 2% is good for LinkedIn, above 1% for Twitter.")
                        st.metric("Conversions", campaign.get('conversions', 0), help="Number of users who completed the desired action (signup, download, purchase).")
                        
                        budget_total = campaign.get('budget_total', 0)
                        budget_spent = campaign.get('budget_spent', 0)
                        if budget_total > 0:
                            pct = (budget_spent / budget_total) * 100
                            st.progress(min(pct / 100, 1.0))
                            st.caption(f"Budget: €{budget_spent:.2f} / €{budget_total:.2f} ({pct:.0f}%)")
                    
                    with col_actions:
                        campaign_id = campaign.get('id')
                        status = campaign.get('status', '')

                        if status in ['draft', 'paused']:
                            if st.button("▶️ Start", key=f"start_{campaign_id}", use_container_width=True):
                                if not _llm_status.get('ready', False):
                                    st.error("Cannot start — no active LLM model. Fix it in 🤖 LLM Management first.")
                                else:
                                    result = api.start_campaign(campaign_id)
                                    if result:
                                        st.toast("Campaign started!", icon="✅")
                                        st.rerun()

                        if status in ['running', 'active']:
                            if st.button("⏸️ Pause", key=f"pause_{campaign_id}", use_container_width=True):
                                result = api.pause_campaign(campaign_id)
                                if result:
                                    st.toast("Campaign paused!", icon="✅")
                                    st.rerun()

                            if st.button("✅ Check Completion", key=f"check_{campaign_id}", use_container_width=True, help="Check if campaign meets completion criteria (budget/end date)"):
                                with st.spinner("Checking completion criteria..."):
                                    try:
                                        result = api.check_campaign_completion(campaign_id)
                                        if result and result.get('completed'):
                                            st.toast(f"✅ Campaign completed! {result.get('reason')}", icon="✅")
                                            st.json(result.get('final_metrics', {}))
                                            st.rerun()
                                        elif result and result.get('should_complete'):
                                            st.warning(f"⚠️ Should complete: {result.get('reason')}")
                                        else:
                                            st.info(f"ℹ️ {result.get('reason', 'Campaign is still running')}")
                                    except Exception as e:
                                        st.error(f"Error checking completion: {str(e)}")

                        if st.button("🗑️ Delete", key=f"delete_{campaign_id}", use_container_width=True, type="secondary"):
                            if st.session_state.get(f'confirm_delete_{campaign_id}'):
                                result = api.delete_campaign(campaign_id)
                                if result:
                                    st.toast("Campaign deleted!", icon="✅")
                                    st.rerun()
                            else:
                                st.session_state[f'confirm_delete_{campaign_id}'] = True
                                st.warning("Click again to confirm deletion")

                        # Clone button (Gap 11 Fix)
                        if st.button("📋 Clone", key=f"clone_{campaign_id}", use_container_width=True):
                            try:
                                new_name = f"{campaign.get('name', 'Campaign')} (Copy)"
                                result = api.clone_campaign(campaign_id, new_name)
                                if result and not result.get('error'):
                                    st.toast(f"✅ Cloned as: {new_name}", icon="✅")
                                    st.rerun()
                                else:
                                    st.error("Clone failed")
                            except Exception as e:
                                st.error(f"Clone error: {e}")

            
            st.markdown("---")
            if st.button("📥 Export All to CSV"):
                df = pd.DataFrame(campaigns)
                export_to_csv(df, "campaigns_export.csv")
        
        else:
            st.info("No campaigns found. Create your first campaign!")
    
    except Exception as e:
        st.error(f"Failed to load campaigns: {str(e)}")

with tab2:
    st.subheader("Create New Campaign")
    st.caption("Define a new campaign. Required fields are marked with *. The AI agent pipeline will generate content after creation.")
    
    with st.form("create_campaign_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Campaign Name*", placeholder="Q1 2025 LinkedIn Gen Campaign", help="A descriptive name for your campaign. Used in reporting and analytics.")
            platform = st.selectbox("Platform*", PLATFORMS, help="Target platform for content distribution. LinkedIn for B2B, Twitter for broad reach, Email for direct nurturing, Blog for SEO content.")

            if platform == "blog":
                try:
                    status_resp = api._get("/config/integrations/status")
                    blog_status = next((s for s in status_resp if s.get("service") == "blog_cms"), None)
                    if blog_status and blog_status.get("status") != "connected":
                        st.warning("⚠️ **WordPress not configured.** Blog deployment requires `BLOG_CMS_URL`, `BLOG_USERNAME`, and `BLOG_APP_PASSWORD`. Go to **⚙️ Operations → Integrations** to set up.")
                except Exception:
                    pass
            elif platform == "email":
                try:
                    status_resp = api._get("/config/integrations/status")
                    email_status = next((s for s in status_resp if s.get("service") == "mailgun"), None)
                    if email_status and email_status.get("status") != "connected":
                        st.warning("⚠️ **Mailgun not configured.** Email deployment requires `MAILGUN_API_KEY` and `MAILGUN_DOMAIN`. Go to **⚙️ Operations → Integrations** to set up.")
                except Exception:
                    pass

            try:
                available_personas = api.get_available_personas()
            except Exception as e:
                st.warning(f"Could not fetch personas from API, using defaults")
                available_personas = PERSONAS

            persona = st.selectbox(
                "Target Persona*",
                available_personas,
                help="Target audience persona. Determines content tone and messaging strategy."
            )
            goal = st.selectbox(
                "Campaign Goal",
                ["lead_generation", "brand_awareness", "engagement", "conversions"],
                help="Primary objective for this campaign. Affects content generation and success metrics."
            )
        
        with col2:
            budget = st.number_input("Total Budget (€)*", min_value=10.0, max_value=1000000.0, value=1000.0, step=100.0, help="Total budget in EUR. Campaign auto-completes when 98% is spent.")
            duration_days = st.number_input("Duration (days)", min_value=1, max_value=365, value=30, help="Campaign duration in days from start date.")
            
            start_date = st.date_input("Start Date", value=datetime.now(), help="Date when the campaign begins deploying content.")
            end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=duration_days), help="Date when the campaign stops. Auto-calculated from duration but can be adjusted.")
        
        auto_start = st.checkbox("Auto-start campaign after creation", value=False, help="If checked, the campaign moves directly to RUNNING after creation, skipping manual approval.")
        
        submit = st.form_submit_button("🚀 Create Campaign", type="primary", use_container_width=True)
        
        if submit:
            if not name:
                st.error("Campaign name is required")
            else:
                try:
                    campaign_data = {
                        "name": name,
                        "platform": platform,
                        "target_persona": persona,
                        "goal": goal,
                        "budget_total": budget,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat()
                    }
                    
                    with st.spinner("Creating campaign..."):
                        result = api.create_campaign(campaign_data)
                    
                    if result:
                        st.success(f"✅ Campaign '{name}' created successfully!")
                        campaign_id = result.get('id')
                        
                        if auto_start and campaign_id:
                            if not _llm_status.get('ready', False):
                                st.warning("⚠️ Campaign created but NOT started — no active LLM model. Fix it in 🤖 LLM Management, then start manually.")
                            else:
                                with st.spinner("Starting campaign..."):
                                    api.start_campaign(campaign_id)
                                st.success("🚀 Campaign started!")
                        
                        st.toast(f"✅ Campaign '{name}' created successfully!", icon="🎉")
                        st.rerun()
                    else:
                        st.error("Failed to create campaign")
                
                except Exception as e:
                    st.error(f"Error creating campaign: {str(e)}")

with tab3:
    st.subheader("Campaign Details")
    st.caption("Deep-dive into a single campaign: performance metrics, simulation predictions, and budget tracking.")
    
    campaigns = api.get_campaigns(limit=500)
    
    if campaigns:
        selected_campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="details_campaign",
            placeholder="Search by name, ID, platform..."
        )
        
        if selected_campaign_id:
            try:
                campaign = api.get_campaign(selected_campaign_id)
                
                if campaign:
                    st.markdown(f"### {campaign.get('name', 'Unnamed Campaign')}")
                    
                    config = campaign.get('config', {})
                    is_mock = campaign.get('is_mock', config.get('is_mock', config.get('mock_mode', False)))
                    if is_mock:
                        st.warning("🧪 **MOCK MODE** | This campaign is using simulated deployment. KPIs are generated by the simulation engine, not real platform data.")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Basic Info**")
                        st.write(f"**ID:** `{campaign.get('id')}`")
                        st.write(f"**Platform:** {campaign.get('platform', 'N/A').capitalize()}")
                        st.write(f"**Persona:** {campaign.get('target_persona', 'N/A')}")
                        st.write(f"**Goal:** {campaign.get('goal', 'N/A')}")
                        st.write(f"**Status:** {campaign.get('status', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Performance**")
                        st.metric("Impressions", f"{campaign.get('impressions', 0):,}", help="Total number of times campaign content was displayed.")
                        st.metric("Clicks", f"{campaign.get('clicks', 0):,}", help="Total user clicks on campaign content.")
                        campaign_ctr = normalize_ctr(campaign.get('ctr', 0))
                        st.metric("CTR", f"{campaign_ctr:.2f}%", help="Click-Through Rate. Above 2% is strong for B2B, above 1% for broad audiences.")
                        st.metric("Conversions", campaign.get('conversions', 0), help="Completed goal actions (leads, signups, purchases).")
                    
                    with col3:
                        st.markdown("**Budget**")
                        st.metric("Total", f"€{campaign.get('budget_total', 0):.2f}", help="Total allocated budget for this campaign.")
                        st.metric("Spent", f"€{campaign.get('budget_spent', 0):.2f}", help="Amount spent so far. Campaign auto-completes at 98% spend.")
                        
                        budget_total = campaign.get('budget_total', 0)
                        budget_spent = campaign.get('budget_spent', 0)
                        remaining = budget_total - budget_spent
                        st.metric("Remaining", f"€{remaining:.2f}", help="Budget left before auto-completion triggers.")
                    
                    st.markdown("---")
                    st.markdown("### 🔮 Simulation Predictions")

                    try:
                        sim_results = api.get_campaign_simulation(selected_campaign_id)

                        if sim_results and sim_results.get('has_simulation'):
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "Predicted CTR",
                                    f"{sim_results.get('predicted_ctr', 0)*100:.2f}%",
                                    help="Simulation-estimated CTR based on content quality, persona, and platform benchmarks."
                                )

                            with col2:
                                st.metric(
                                    "Predicted Conversions",
                                    sim_results.get('predicted_conversions', 0),
                                    help="Estimated conversions from the simulation engine based on budget and predicted engagement."
                                )

                            with col3:
                                cpl = sim_results.get('predicted_cpl')
                                cpl_display = f"€{cpl:.2f}" if cpl else "N/A"
                                st.metric("Predicted CPL", cpl_display, help="Predicted Cost Per Lead. Lower is better — under €10 is strong for B2B.")

                            with col4:
                                st.metric(
                                    "Predicted Impressions",
                                    f"{sim_results.get('predicted_impressions', 0):,}",
                                    help="Estimated total impressions based on budget, platform, and audience size."
                                )

                            if sim_results.get('simulation_passed'):
                                st.success("✅ Simulation passed quality thresholds")
                            else:
                                st.warning("⚠️ Simulation did not meet quality thresholds")

                            st.caption(f"Simulated at: {sim_results.get('simulation_timestamp', 'Unknown')}")
                        else:
                            st.info("ℹ️ No simulation results available. Simulation may not have run yet.")
                    except Exception as e:
                        st.info(f"ℹ️ Could not load simulation results: {str(e)}")

                    st.markdown("---")
                    st.markdown("### 📈 Actual Performance Metrics")

                    try:
                        metrics = api.get_campaign_metrics(selected_campaign_id)
                        if metrics:
                            st.json(metrics)
                        else:
                            st.info("No detailed metrics available yet")
                    except:
                        st.info("Metrics not available")
                else:
                    st.error("Campaign not found")
            
            except Exception as e:
                st.error(f"Error loading campaign: {str(e)}")
    else:
        st.info("No campaigns found. Create a campaign first.")

with tab4:
    st.subheader("Bulk Operations")
    st.caption("Perform actions on multiple campaigns at once. Select campaigns below, then choose an operation.")
    
    st.info("⚠️ Bulk operations affect multiple campaigns - use with caution")
    
    try:
        all_campaigns = api.get_campaigns(limit=1000)
        
        if all_campaigns:
            campaign_options = {
                f"{c.get('name')} ({c.get('id')})": c.get('id')
                for c in all_campaigns
            }
            
            selected = st.multiselect(
                "Select Campaigns",
                options=list(campaign_options.keys()),
                help="Choose one or more campaigns to apply bulk actions to."
            )
            
            selected_ids = [campaign_options[name] for name in selected]
            
            st.write(f"Selected {len(selected_ids)} campaigns")
            
            if selected_ids:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("▶️ Start All", use_container_width=True):
                        if not _llm_status.get('ready', False):
                            st.error("Cannot start — no active LLM model. Fix it in 🤖 LLM Management first.")
                        else:
                            success_count = 0
                            for cid in selected_ids:
                                try:
                                    api.start_campaign(cid)
                                    success_count += 1
                                except:
                                    pass
                            st.toast(f"Started {success_count}/{len(selected_ids)} campaigns", icon="✅")
                        st.rerun()
                
                with col2:
                    if st.button("⏸️ Pause All", use_container_width=True):
                        success_count = 0
                        for cid in selected_ids:
                            try:
                                api.pause_campaign(cid)
                                success_count += 1
                            except:
                                pass
                        st.toast(f"Paused {success_count}/{len(selected_ids)} campaigns", icon="✅")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete All", use_container_width=True, type="secondary"):
                        if st.session_state.get('confirm_bulk_delete'):
                            success_count = 0
                            for cid in selected_ids:
                                try:
                                    api.delete_campaign(cid)
                                    success_count += 1
                                except:
                                    pass
                            st.toast(f"Deleted {success_count}/{len(selected_ids)} campaigns", icon="✅")
                            st.session_state['confirm_bulk_delete'] = False
                            st.rerun()
                        else:
                            st.session_state['confirm_bulk_delete'] = True
                            st.warning("⚠️ Click again to confirm bulk deletion")
        else:
            st.info("No campaigns available for bulk operations")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption(f"Campaign Management | Last updated: {datetime.now().strftime('%H:%M:%S')}")
