"""
Canary Deployments - Progressive Rollout Management
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls
from components import render_metric_card, render_status_badge, create_gauge_chart

st.set_page_config(page_title="Canary Deployments - Agentic AI", page_icon="📦", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("📦 Canary Deployments")
st.caption("Progressive rollout for new policies and MARL agents")

with st.expander("ℹ️ Understanding Canary Deployments", expanded=False):
    st.markdown("""
**What is a Canary Deployment?**
A canary deployment gradually rolls out a new content policy to a small percentage of traffic first,
allowing you to validate performance in production before committing to a full rollout.

**Progressive Rollout Stages:**
Traffic is increased through controlled stages — **5% → 25% → 50% → 100%** — with safety checks at each gate.

**Safety Monitoring:**
At every stage, three key metrics are continuously compared between the baseline and canary:
- **CTR Degradation** — detects drops in click-through rate
- **Error Rate** — catches increased failures or bad responses
- **Latency (P95)** — ensures response times stay within acceptable bounds

**Auto-Rollback:**
If any monitored metric degrades beyond the configured threshold, traffic is **automatically reverted**
to the baseline policy, preventing user-facing impact.

**When to Use:**
After the MARL gating system approves a new policy (via OPE validation), deploy it through a canary
to ensure safe, observable production rollout.

**Traffic Splitting:**
During a canary deployment, traffic is split between the **baseline** (current production policy) and the
**canary** (new policy). Comparison metrics are shown in real-time on the Active Deployments tab.

**Best Practices:**
- Start with **5% traffic** to limit blast radius
- Set **conservative thresholds** initially (e.g., 10% CTR degradation)
- Monitor for at least **24 hours** before advancing to the next stage
- Always enable auto-rollback for unattended deployments
    """)

st.markdown("""
**Canary Deployment Flow:**
1. Start with 5% traffic
2. Monitor metrics for degradation
3. Progressively increase: 5% → 25% → 50% → 100%
4. Automatic rollback if metrics degrade
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Start Deployment",
    "📊 Active Deployments",
    "📜 Deployment History",
    "⚙️ Configuration"
])

with tab1:
    st.subheader("Start New Canary Deployment")
    st.caption("Configure and launch a new canary deployment with traffic splitting and safety thresholds.")
    
    with st.form("canary_deployment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            deployment_name = st.text_input(
                "Deployment Name*",
                placeholder="e.g., marl_policy_v2_canary",
                help="Descriptive name for this deployment. Helps track in history."
            )
            
            policy_name = st.text_input(
                "Policy/Model Name*",
                placeholder="e.g., marl_agent_v2",
                help="Name of the MARL policy or model to deploy as the canary variant."
            )
            
            deployment_type = st.selectbox(
                "Deployment Type",
                ["policy", "model", "agent"],
                index=0,
                help="Type of artifact being deployed: policy (MARL policy), model (ML model), or agent (full agent)."
            )

            description = st.text_area(
                "Description",
                placeholder="Deployment notes (e.g. TEST_MODE_FAST)",
                height=100,
                help="Optional notes about this deployment (e.g., reason for rollout, ticket reference)."
            )
        
        with col2:
            initial_traffic = st.slider(
                "Initial Traffic %",
                min_value=1,
                max_value=20,
                value=5,
                help="Starting percentage of traffic for the canary. Start low (5%) for safety."
            )
            
            progression_schedule = st.selectbox(
                "Progression Schedule",
                ["Conservative (24h intervals)", "Moderate (12h intervals)", "Aggressive (6h intervals)"],
                index=0,
                help="How quickly traffic increases between stages. Conservative is recommended for production."
            )
            
            auto_rollback = st.checkbox(
                "Enable Auto-Rollback",
                value=True,
                help="Automatically revert to baseline if safety thresholds are breached."
            )
        
        st.markdown("#### Rollback Thresholds")
        st.caption("Deployment will rollback if any metric degrades beyond these thresholds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ctr_threshold = st.number_input(
                "CTR Degradation (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Maximum allowed CTR drop (%) before auto-rollback triggers."
            )
        
        with col2:
            error_threshold = st.number_input(
                "Error Rate Threshold (%)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Maximum error rate (%) allowed before auto-rollback."
            )
        
        with col3:
            latency_threshold = st.number_input(
                "Latency Threshold (ms)",
                min_value=0,
                max_value=5000,
                value=500,
                step=50,
                help="Maximum response time (ms) before auto-rollback."
            )
        
        submit = st.form_submit_button("🚀 Start Canary Deployment", type="primary")
        
        if submit:
            if not deployment_name or not policy_name:
                st.error("Deployment name and policy name are required")
            else:
                try:
                    deployment_request = {
                        "deployment_name": deployment_name,
                        "policy_name": policy_name,
                        "deployment_type": deployment_type,
                        "initial_traffic_percent": initial_traffic,
                        "auto_rollback_enabled": auto_rollback,
                        "thresholds": {
                            "ctr_degradation_percent": ctr_threshold,
                            "error_rate_percent": error_threshold,
                            "latency_p95_ms": latency_threshold
                        },
                        "description": description
                    }
                    
                    with st.spinner("Starting canary deployment..."):
                        result = api.start_canary_deployment(deployment_request)
                    
                    if result:
                        st.toast(f"✅ Canary deployment '{deployment_name}' started!", icon="✅")
                        st.info(f"🎯 Initial traffic: {initial_traffic}%")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to start deployment")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.subheader("Active Canary Deployments")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption("Monitor live canary deployments — view traffic split, real-time metrics, and trigger manual rollback if needed.")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    try:
        active_deployments = api.list_active_deployments()
        
        if active_deployments:
            st.success(f"✅ {len(active_deployments)} active deployments")
            
            for deployment in active_deployments:
                deployment_id = deployment.get('deployment_id', deployment.get('id', 'N/A'))
                deployment_name = deployment.get('policy_id', deployment.get('deployment_name', 'Unnamed'))
                policy_name = deployment.get('policy_id', deployment.get('policy_name', 'N/A'))
                policy_version = deployment.get('policy_version', 'N/A')
                current_traffic = deployment.get('current_traffic_percentage', deployment.get('current_traffic_percent', 0))
                if current_traffic < 1:
                    current_traffic = current_traffic * 100
                status = deployment.get('status', 'unknown')
                started_at = deployment.get('start_time', deployment.get('started_at', 'N/A'))
                baseline_metrics = deployment.get('baseline_metrics', {})
                
                with st.expander(
                    f"**{deployment_name} v{policy_version}** | Traffic: {current_traffic:.0f}% | Status: {status.upper()}",
                    expanded=True
                ):
                    col_info, col_metrics, col_actions = st.columns([2, 2, 1])
                    
                    with col_info:
                        st.markdown("**Deployment Info**")
                        st.write(f"**ID:** `{deployment_id}`")
                        st.write(f"**Policy:** {policy_name} v{policy_version}")
                        st.write(f"**Started:** {started_at[:19] if isinstance(started_at, str) and started_at != 'N/A' else 'N/A'}")
                        
                        render_status_badge(status)
                    
                    with col_metrics:
                        st.markdown("**Current Metrics**")
                        
                        st.metric("Traffic", f"{current_traffic:.0f}%", delta=f"Target: 100%", help="Current percentage of total traffic routed to the canary policy.")
                        
                        try:
                            fig = create_gauge_chart(
                                value=current_traffic,
                                max_value=100,
                                title="Traffic %"
                            )
                            fig.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
                            st.plotly_chart(fig, use_container_width=True, key=f"canary_metrics_{deployment_id}")
                        except Exception as e:
                            st.caption(f"Gauge: {current_traffic:.0f}%")
                    
                    with col_actions:
                        st.markdown("**Actions**")
                        
                        if st.button("📊 View Details", key=f"view_{deployment_id}", use_container_width=True):
                            try:
                                details = api.get_deployment_status(deployment_id)
                                if details:
                                    st.json(details)
                            except:
                                st.error("Failed to load details")
                        
                        if st.button("⏮️ Rollback", key=f"rollback_{deployment_id}", use_container_width=True, type="secondary"):
                            if st.session_state.get(f'confirm_rollback_{deployment_id}'):
                                try:
                                    result = api.rollback_deployment(
                                        deployment_id=deployment_id,
                                        reason="Manual rollback from dashboard"
                                    )
                                    if result:
                                        st.toast("Rollback initiated!", icon="✅")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Rollback failed: {str(e)}")
                            else:
                                st.session_state[f'confirm_rollback_{deployment_id}'] = True
                                st.warning("Click again to confirm rollback")
                    
                    st.markdown("---")
                    st.markdown("### Performance Comparison")
                    
                    canary_history = deployment.get('canary_metrics_history', [])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Baseline**")
                        baseline_ctr = baseline_metrics.get('ctr', 0)
                        baseline_cvr = baseline_metrics.get('conversion_rate', 0)
                        baseline_cpl = baseline_metrics.get('cpl', 0)
                        st.metric("CTR", f"{baseline_ctr:.2f}%", help="Baseline click-through rate — the current production policy's performance.")
                        st.metric("Conv. Rate", f"{baseline_cvr:.2f}%", help="Baseline conversion rate for the current production policy.")
                        st.metric("CPL", f"€{baseline_cpl:.2f}", help="Baseline cost per lead for the current production policy.")
                    
                    with col2:
                        st.markdown("**Canary**")
                        if canary_history:
                            latest = canary_history[-1]
                            canary_ctr = latest.get('ctr', 0)
                        else:
                            canary_ctr = baseline_ctr
                        st.metric("CTR", f"{canary_ctr:.2f}%", help="Canary click-through rate — the new policy's observed performance.")
                        st.caption("(Collecting data...)" if not canary_history else "")
                    
                    with col3:
                        st.markdown("**Status**")
                        if canary_history:
                            ctr_diff = canary_ctr - baseline_ctr
                            st.metric("CTR Diff", f"{ctr_diff:+.2f}%", delta=f"{ctr_diff:.2f}%", help="Difference in CTR between canary and baseline. Negative values indicate degradation.")
                        else:
                            st.info("⏳ Collecting metrics...")
                    rollback_reason = deployment.get('rollback_reason')
                    if rollback_reason:
                        st.error(f"⚠️ Rollback: {rollback_reason}")
                    elif status == "full_rollout_100_percent":
                        st.success("✅ Full rollout complete!")
                    elif "canary" in status.lower():
                        st.info(f"🚀 Canary in progress - {current_traffic:.0f}% traffic")
        
        else:
            st.info("📭 No active canary deployments")
            st.info("💡 Start a new deployment from the 'Start Deployment' tab")
    
    except Exception as e:
        st.error(f"Failed to load active deployments: {str(e)}")

with tab3:
    st.subheader("Deployment History")
    st.caption("Review past canary deployments — outcomes, rollbacks, and progression records.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        limit = st.number_input("Records to Load", min_value=5, max_value=100, value=20, help="Number of historical deployment records to fetch.")
    
    with col2:
        if st.button("🔄 Refresh History", use_container_width=True):
            st.rerun()
    
    try:
        history = api.get_deployment_history(limit=limit)
        
        if history:
            st.success(f"✅ Loaded {len(history)} deployment records")
            
            full_rollouts = len([h for h in history if h.get('status') == 'full_rollout_100_percent'])
            rolled_back = len([h for h in history if h.get('status') == 'rolled_back' or h.get('rollback_reason')])
            in_progress = len([h for h in history if 'canary' in h.get('status', '').lower()])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                render_metric_card("Total Deployments", len(history))
            with col2:
                render_metric_card("Full Rollouts", full_rollouts, delta=f"+{full_rollouts}")
            with col3:
                render_metric_card("Rolled Back", rolled_back, delta=f"-{rolled_back}", delta_color="inverse")
            
            st.markdown("---")
            
            history_data = []
            for h in history:
                status = h.get('status', 'unknown')
                rollback = h.get('rollback_reason')
                
                if status == 'full_rollout_100_percent':
                    outcome_display = "✅ Full Rollout"
                elif status == 'rolled_back' or rollback:
                    outcome_display = f"⏮️ Rollback: {rollback or 'Manual'}"
                elif 'canary' in status.lower():
                    outcome_display = f"🚀 {status}"
                else:
                    outcome_display = f"❓ {status}"
                
                traffic = h.get('current_traffic_percentage', 0)
                if traffic < 1:
                    traffic = traffic * 100
                
                history_data.append({
                    'Deployment': h.get('policy_id', h.get('deployment_id', 'N/A')),
                    'Version': h.get('policy_version', 'N/A'),
                    'Started': (h.get('start_time', 'N/A') or 'N/A')[:19],
                    'Ended': (h.get('end_time', '') or 'In Progress')[:19],
                    'Traffic': f"{traffic:.0f}%",
                    'Status': outcome_display
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if st.button("📥 Export History"):
                from components import export_to_csv
                export_to_csv(df, f"canary_history_{datetime.now().strftime('%Y%m%d')}.csv")
        
        else:
            st.info("📭 No deployment history available")
    
    except Exception as e:
        st.error(f"Failed to load deployment history: {str(e)}")

with tab4:
    st.subheader("Canary Deployment Configuration")
    st.caption("Reference guide for default settings, rollback triggers, and deployment best practices.")
    
    st.markdown("""
    ### Default Settings
    
    **Traffic Progression:**
    - Conservative: 5% → 25% → 50% → 100% (24h intervals)
    - Moderate: 5% → 25% → 50% → 100% (12h intervals)
    - Aggressive: 10% → 50% → 100% (6h intervals)
    
    **Automatic Rollback Triggers:**
    - CTR degradation > 10%
    - Error rate > 5%
    - P95 latency > 500ms
    - Any safety violation detected
    
    **Monitoring:**
    - Real-time metric comparison (baseline vs canary)
    - Automated health checks every 5 minutes
    - Alert notifications for degradation
    """)
    
    st.markdown("---")
    
    st.markdown("### Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ **Do:**")
        st.write("- Start with low traffic (5%)")
        st.write("- Monitor metrics continuously")
        st.write("- Enable auto-rollback")
        st.write("- Set conservative thresholds")
        st.write("- Test during low-traffic periods")
    
    with col2:
        st.error("❌ **Don't:**")
        st.write("- Rush progression schedule")
        st.write("- Ignore metric degradation")
        st.write("- Deploy without OPE validation")
        st.write("- Skip MARL promotion gates")
        st.write("- Deploy during peak hours")

st.markdown("---")
st.caption(f"Canary Deployments | Last updated: {datetime.now().strftime('%H:%M:%S')}")
