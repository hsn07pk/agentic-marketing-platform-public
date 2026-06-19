"""
System Transparency - Complete Workflow Visibility & Campaign Insights
NEVER be in a state where user doesn't know what's happening
Enhanced: Campaign Deep Dive, Workflow Progress, Configurations, Recommendations
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls
from utils.llm_checks import check_llm_readiness, render_llm_status_banner
from components import render_status_badge, render_metric_card
from app_config import (
    WORKFLOW_NODES, WORKFLOW_NODES_ALL, WORKFLOW_NODES_CORE, WORKFLOW_NODES_OPTIONAL, WORKFLOW_NODE_IDS,
    WORKFLOW_EVENT_TYPES, EVENT_ICONS, EVENT_TO_NODE_MAPPING,
    get_budget_thresholds, get_governance_thresholds,
    get_event_icon, get_config_value
)

st.set_page_config(page_title="System Transparency - Agentic AI", page_icon="📡", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("📡 System Transparency & Campaign Insights")
st.caption("Complete visibility into all workflow operations, configurations, and recommendations")

with st.expander("ℹ️ System Transparency Guide", expanded=False):
    st.markdown("""
**What this page shows:** Complete visibility into AI agent workflow decisions, events, and reasoning. Every action taken by the system is logged and explainable.

**📋 Campaign Status** — Shows the current state of each campaign, its workflow progress, and *why* a campaign is in its current state (e.g., paused due to budget, blocked by HITL review).

**🔄 Workflow Events** — A detailed log of every workflow step: content generation, safety checks, MARL evaluation, deployment, and more. Each event includes timestamps, severity, and full output details.

**💡 Recommendations** — AI-generated suggestions for improving campaign performance, with root cause analysis, affected workflow nodes, and step-by-step fix instructions.

**🚨 Alerts & Issues** — System-detected problems requiring attention, including MARL rejections, safety validation failures, deployment errors, and budget warnings. Each alert includes a suggested solution.

**Event types you may see:**
- `content_generated` — New content created by the LLM agent
- `safety_check_passed` / `safety_check_failed` — Content safety validation results
- `marl_gated` — Multi-Agent Reinforcement Learning policy evaluation
- `deployed` — Content successfully deployed to platform
- `hitl_review` / `hitl_queue_added` — Human-in-the-loop review events
- `budget_warning` / `budget_exceeded` — Budget threshold alerts
- `workflow_completed` / `workflow_failed` — Overall workflow lifecycle events

**How to use this page:**
1. Select a campaign to trace its complete workflow history
2. Use the **Alerts & Issues** tab to investigate and resolve problems
3. Check **Workflow Progress** to see which pipeline stages completed or failed
4. Review **Recommendations** for actionable improvement steps
5. Inspect **Configurations** to verify campaign settings and thresholds

**Why transparency matters:** Ensures all AI decisions are explainable and auditable, supporting responsible AI governance and regulatory compliance.
    """)


ALL_EVENT_TYPES = WORKFLOW_EVENT_TYPES

def get_status_reason(campaign, events):
    """Determine why campaign is in current state"""
    status = campaign.get('status', 'unknown')
    
    if status == 'draft':
        return "Campaign hasn't been started yet"
    elif status == 'running':
        return "Campaign is actively running"
    elif status == 'paused':
        for event in events:
            if event.get('event_type') == 'workflow_paused':
                return event.get('message', 'Paused by user or system')
        return "Paused by user"
    elif status == 'completed':
        for event in events:
            if event.get('event_type') == 'workflow_completed':
                details = event.get('details', {})
                return details.get('completion_reason', 'Campaign completed successfully')
        return "Campaign completed successfully"
    elif status == 'failed':
        for event in events:
            if event.get('event_type') in ['workflow_failed', 'error_occurred']:
                return event.get('message', 'An error occurred')
        return "An error occurred during execution"
    return f"Status: {status}"

def get_suggested_solution(alert):
    """Generate solution based on alert type"""
    event_type = alert.get('event_type', '')
    title = alert.get('title', '').lower()
    
    solutions = {
        'safety_check_failed': {
            'action': "Content failed safety validation. Review the safety scores and consider regenerating with different parameters.",
            'quick_fix_label': "Regenerate Content",
            'quick_fix': True
        },
        'budget_exceeded': {
            'action': "Budget limit exceeded. Consider increasing the budget or pausing the campaign to prevent overspending.",
            'quick_fix_label': "Pause Campaign",
            'quick_fix': True
        },
        'budget_warning': {
            'action': "Budget is running low. Review spend rate and consider adjusting daily limits.",
            'quick_fix_label': None,
            'quick_fix': False
        },
        'deployment_failed': {
            'action': "Deployment to platform failed. Check API credentials and platform status, then retry.",
            'quick_fix_label': "Retry Deployment",
            'quick_fix': True
        },
        'hitl_queue_added': {
            'action': "Content requires human review before deployment. Go to Governance page to review.",
            'quick_fix_label': "Go to Review Queue",
            'quick_fix': True
        },
        'content_rejected': {
            'action': "Content was rejected during review. Check feedback and regenerate with adjustments.",
            'quick_fix_label': "View Feedback",
            'quick_fix': True
        }
    }
    
    # MARL-specific solutions based on workflow node
    workflow_node = alert.get('workflow_node', '')
    if workflow_node == 'marl_gating':
        details = alert.get('details', {})
        reason = details.get('reason', '')
        if details.get('approved') is False:
            if 'insufficient' in reason:
                return {
                    'action': "MARL policy needs more training data. Run more campaigns to generate historical decisions for OPE evaluation. This is normal during early system operation.",
                    'quick_fix_label': None,
                    'quick_fix': False
                }
            return {
                'action': "MARL policy did not meet the ≥20% lift threshold required for promotion. The system is using the baseline bandit strategy, which is the expected safe behavior. The policy will be re-evaluated as more data is collected.",
                'quick_fix_label': None,
                'quick_fix': False
            }
        if 'error' in title:
            return {
                'action': "MARL gating encountered an error during evaluation. Check system logs and ensure the MARL training pipeline is configured correctly. The baseline strategy is being used as a safe fallback.",
                'quick_fix_label': None,
                'quick_fix': False
            }
    
    if event_type in solutions:
        return solutions[event_type]
    
    return {
        'action': "Review the alert details and take appropriate action based on the error message.",
        'quick_fix_label': None,
        'quick_fix': False
    }

def generate_recommendations(campaign, events):
    """Generate actionable recommendations with detailed root cause analysis and specific fixes"""
    recommendations = []
    status = campaign.get('status', 'unknown')
    
    failure_events = [e for e in events if 'failed' in e.get('event_type', '')]
    
    for failure in failure_events:
        event_type = failure.get('event_type', 'unknown')
        details = failure.get('details', {})
        workflow_node = failure.get('workflow_node', 'unknown')
        message = failure.get('message', 'No message available')
        
        if event_type == 'safety_check_failed':
            safety_score = details.get('safety_score', details.get('overall_score', 'N/A'))
            toxicity = details.get('toxicity_score', 'N/A')
            factuality = details.get('factuality_score', 'N/A')
            threshold = details.get('threshold', 0.95)
            content_id = failure.get('content_id', 'N/A')
            
            recommendations.append({
                'icon': '🛡️',
                'title': 'Safety Validation Failed',
                'priority': 'high',
                'root_cause': f"Content failed safety validation because score ({safety_score}) is below threshold ({threshold})",
                'affected_node': 'Safety Validation',
                'details': {
                    'Content ID': content_id,
                    'Safety Score': safety_score,
                    'Toxicity Score': toxicity,
                    'Factuality Score': factuality,
                    'Required Threshold': threshold,
                    'Error Message': message
                },
                'fix_steps': [
                    "1. Go to **Governance** page → **HITL Review Queue**",
                    "2. Find the flagged content and review the specific issues",
                    "3. Either **approve with modifications** or **reject and regenerate**",
                    "4. If regenerating, consider: adjusting tone, removing bold claims, adding citations",
                    "5. For persistent issues: Review the claim library for problematic claims"
                ]
            })
        
        elif event_type == 'deployment_failed':
            platform = details.get('platform', campaign.get('platform', 'unknown'))
            error_code = details.get('error_code', details.get('status_code', 'N/A'))
            api_error = details.get('api_error', details.get('error', message))
            
            recommendations.append({
                'icon': '🚀',
                'title': 'Platform Deployment Failed',
                'priority': 'high',
                'root_cause': f"Deployment to {platform} failed with error: {api_error}",
                'affected_node': 'Deployment',
                'details': {
                    'Platform': platform,
                    'Error Code': error_code,
                    'API Error': api_error,
                    'Error Message': message
                },
                'fix_steps': [
                    f"1. Check **{platform}** API credentials in Settings → Configurations",
                    "2. Verify the platform API is accessible (check platform status page)",
                    f"3. Review rate limits - you may have exceeded {platform} API quotas",
                    "4. Check content formatting meets platform requirements",
                    "5. After fixing, click **Retry Deployment** or restart campaign"
                ]
            })
        
        elif event_type == 'content_rejected':
            reviewer = details.get('reviewed_by', 'Human reviewer')
            feedback = details.get('feedback', details.get('rejection_reason', message))
            content_id = failure.get('content_id', 'N/A')
            
            recommendations.append({
                'icon': '❌',
                'title': 'Content Rejected by Reviewer',
                'priority': 'high',
                'root_cause': f"Content was rejected during HITL review: {feedback}",
                'affected_node': 'HITL Review',
                'details': {
                    'Content ID': content_id,
                    'Reviewed By': reviewer,
                    'Rejection Feedback': feedback,
                    'Error Message': message
                },
                'fix_steps': [
                    "1. Review the rejection feedback carefully",
                    "2. Go to **Campaigns** page → Select campaign → **Regenerate Content**",
                    "3. Adjust the prompt/parameters based on feedback",
                    "4. Consider updating the claim library if claims are the issue",
                    "5. Submit new content for review"
                ]
            })
        
        elif event_type == 'workflow_failed':
            error_msg = details.get('error', details.get('exception', message))
            failed_node = details.get('failed_node', workflow_node)
            
            recommendations.append({
                'icon': '⚠️',
                'title': 'Workflow Execution Failed',
                'priority': 'high',
                'root_cause': f"Workflow stopped at {failed_node}: {error_msg}",
                'affected_node': failed_node,
                'details': {
                    'Failed Node': failed_node,
                    'Error': error_msg,
                    'Workflow State': failure.get('workflow_state', 'failed'),
                    'Error Message': message
                },
                'fix_steps': [
                    f"1. Check the **{failed_node}** configuration in Settings",
                    "2. Review API keys and external service connections",
                    "3. Check system logs for detailed stack trace",
                    "4. If LLM error: verify OpenAI/Anthropic API keys and quotas",
                    "5. Restart the campaign to retry from the failed step"
                ]
            })
        
        elif event_type == 'node_failed':
            node_name = details.get('node', workflow_node)
            error_msg = details.get('error', message)
            retry_count = details.get('retry_count', 0)
            
            recommendations.append({
                'icon': '🔄',
                'title': f'Node Failed: {node_name.replace("_", " ").title()}',
                'priority': 'high',
                'root_cause': f"The {node_name} step failed after {retry_count} retries: {error_msg}",
                'affected_node': node_name,
                'details': {
                    'Node': node_name,
                    'Error': error_msg,
                    'Retry Count': retry_count,
                    'Error Message': message
                },
                'fix_steps': [
                    f"1. Check dependencies for the **{node_name}** step",
                    "2. Verify all required inputs are available",
                    "3. Check external API connectivity if this node calls APIs",
                    "4. Review system resources (memory, disk space)",
                    "5. Manually trigger the step or restart campaign"
                ]
            })
        
        else:
            # ── Content generation failures: provide LLM-specific guidance ──
            error_msg = details.get('error', message)
            is_content_gen_llm_issue = (
                workflow_node == 'content_generation'
                and ('claims' in error_msg.lower() or 'model' in error_msg.lower() or 'ollama' in error_msg.lower())
            )
            if is_content_gen_llm_issue:
                recommendations.append({
                    'icon': '🤖',
                    'title': 'Content Generation Failed — Likely LLM Model Issue',
                    'priority': 'critical',
                    'root_cause': (
                        f"The content generation LLM produced empty or non-compliant output: {error_msg}. "
                        "This usually means the configured Ollama model is not installed or not responding."
                    ),
                    'affected_node': 'content_generation',
                    'details': {
                        'Event Type': event_type,
                        'Workflow Node': workflow_node,
                        'Error Message': error_msg,
                    },
                    'fix_steps': [
                        "1. Go to **🤖 LLM Management → Model Management** and check that the active model is installed",
                        "2. If the active model shows as *not installed*, click **Set Active** on an installed model (e.g. `llama3:8b`)",
                        "3. Use **🤖 LLM Management → Model Testing** to verify the model responds correctly",
                        "4. After fixing, restart the campaign from the **📋 Campaigns** page"
                    ]
                })
            else:
                recommendations.append({
                    'icon': '⚠️',
                    'title': f'Operation Failed: {event_type.replace("_", " ").title()}',
                    'priority': 'high',
                    'root_cause': message,
                    'affected_node': workflow_node,
                    'details': {
                        'Event Type': event_type,
                        'Workflow Node': workflow_node,
                        'Error Message': message,
                        **{k: v for k, v in details.items() if k not in ['error', 'message']}
                    },
                    'fix_steps': [
                        "1. Review the error details above",
                        "2. Check the Workflow Progress tab for context",
                        "3. Verify system configurations in Settings",
                        "4. Check external service connectivity",
                        "5. Restart campaign or contact support if issue persists"
                    ]
                })
    
    budget_thresholds = get_budget_thresholds()
    critical_threshold = budget_thresholds['critical']
    warning_threshold = budget_thresholds['warning']
    auto_pause_threshold = budget_thresholds['auto_pause']
    
    budget_total = campaign.get('budget_total', 0)
    budget_spent = campaign.get('budget_spent', 0)
    if budget_total > 0:
        utilization = (budget_spent / budget_total) * 100
        remaining = budget_total - budget_spent
        
        if utilization >= critical_threshold:
            recommendations.append({
                'icon': '💰',
                'title': 'Budget Nearly Exhausted',
                'priority': 'high',
                'root_cause': f"Budget utilization is at {utilization:.1f}%, only €{remaining:.2f} remaining",
                'affected_node': 'Budget Control',
                'details': {
                    'Total Budget': f"€{budget_total:.2f}",
                    'Spent': f"€{budget_spent:.2f}",
                    'Remaining': f"€{remaining:.2f}",
                    'Utilization': f"{utilization:.1f}%"
                },
                'fix_steps': [
                    "1. Go to **Campaigns** page → Select this campaign",
                    "2. Click **Edit** and increase the total budget",
                    "3. Or: Pause the campaign to prevent overspend",
                    "4. Consider: Reduce daily limit to extend campaign duration",
                    "5. Review cost efficiency in **Cost Control** page"
                ]
            })
        elif utilization >= warning_threshold:
            recommendations.append({
                'icon': '💵',
                'title': 'Budget Running Low',
                'priority': 'medium',
                'root_cause': f"Budget is at {utilization:.1f}% - campaign will auto-pause at {auto_pause_threshold}%",
                'affected_node': 'Budget Control',
                'details': {
                    'Total Budget': f"€{budget_total:.2f}",
                    'Spent': f"€{budget_spent:.2f}",
                    'Remaining': f"€{remaining:.2f}",
                    'Utilization': f"{utilization:.1f}%"
                },
                'fix_steps': [
                    "1. Monitor spend rate in **Cost Control** page",
                    "2. Consider increasing budget before it depletes",
                    "3. Alternatively: Reduce daily spend limit"
                ]
            })
    
    hitl_pending = [e for e in events if e.get('event_type') == 'hitl_queue_added']
    hitl_reviewed = [e for e in events if e.get('event_type') == 'hitl_reviewed']
    pending_count = len(hitl_pending) - len(hitl_reviewed)
    
    if pending_count > 0:
        content_ids = [e.get('content_id', 'unknown') for e in hitl_pending[:5]]
        recommendations.append({
            'icon': '👁️',
            'title': 'Content Awaiting Human Review',
            'priority': 'medium',
            'root_cause': f"{pending_count} content items are blocked waiting for HITL review",
            'affected_node': 'HITL Review',
            'details': {
                'Pending Reviews': pending_count,
                'Sample Content IDs': ', '.join(str(c) for c in content_ids[:3])
            },
            'fix_steps': [
                "1. Go to **Governance** page → **HITL Review Queue**",
                "2. Review and approve/reject pending content",
                "3. Content will auto-deploy after approval"
            ]
        })
    
    if status == 'paused':
        pause_events = [e for e in events if e.get('event_type') == 'workflow_paused']
        pause_reason = "Paused by user or system"
        if pause_events:
            pause_reason = pause_events[0].get('message', pause_reason)
        
        recommendations.append({
            'icon': '⏸️',
            'title': 'Campaign is Paused',
            'priority': 'medium',
            'root_cause': pause_reason,
            'affected_node': 'Workflow',
            'details': {
                'Status': 'Paused',
                'Reason': pause_reason
            },
            'fix_steps': [
                "1. Review the pause reason above",
                "2. Address any blocking issues",
                "3. Click **Resume Campaign** in Quick Actions below"
            ]
        })
    
    if status == 'draft':
        recommendations.append({
            'icon': '📝',
            'title': 'Campaign Not Started',
            'priority': 'low',
            'root_cause': "Campaign is in draft state - workflow has not been initiated",
            'affected_node': 'Workflow',
            'details': {
                'Status': 'Draft',
                'Action Required': 'Start the campaign'
            },
            'fix_steps': [
                "1. Review campaign configuration",
                "2. Click **Start Campaign** in Quick Actions below"
            ]
        })
    
    return sorted(recommendations, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}.get(x['priority'], 3))

def get_current_workflow_node(events):
    """Determine current workflow node from events - uses centralized mapping"""
    if not events:
        return None
    
    for event in events:
        event_type = event.get('event_type', '')
        if event_type in EVENT_TO_NODE_MAPPING:
            return EVENT_TO_NODE_MAPPING[event_type]
        if event.get('workflow_node'):
            return event['workflow_node']
    
    return 'market_observation'  # Default to first step


col1, col2 = st.columns([4, 1])
with col1:
    pass
with col2:
    auto_refresh = st.toggle("Auto-refresh", value=False, help="Automatically refresh all data every 5 seconds. Useful for monitoring active campaigns in real time.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 Alerts & Issues",
    "📋 Campaign Deep Dive",
    "🔄 Workflow Progress",
    "⚙️ Configurations",
    "💡 Recommendations"
])

with tab1:
    st.subheader("🚨 Active Alerts Dashboard")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.caption("System-detected problems grouped by severity — critical issues, errors, warnings, and informational messages. Each alert includes a suggested solution.")
    with col2:
        if st.button("🧹 Cleanup HITL Alerts", help="Dismiss HITL alerts for already-reviewed content"):
            try:
                if hasattr(api, 'cleanup_stale_hitl_alerts'):
                    result = api.cleanup_stale_hitl_alerts()
                    count = result.get('dismissed_count', 0)
                    if count > 0:
                        st.toast(f"✅ Cleaned up {count} stale HITL alerts")
                        st.rerun()
                    else:
                        st.info("No stale HITL alerts to clean up. Pending reviews are still active.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    with col3:
        if st.button("🧹 Cleanup MARL Alerts", help="Dismiss all MARL gating informational alerts"):
            try:
                result = api.cleanup_marl_alerts()
                count = result.get('dismissed_count', 0)
                if count > 0:
                    st.toast(f"✅ Cleaned up {count} MARL alerts")
                    st.rerun()
                else:
                    st.info("No MARL alerts to clean up.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    alerts_data = api.get_active_alerts()
    
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = alerts_data.get('total_count', 0)
        st.metric("Total Active Alerts", total, delta=None if total == 0 else f"+{total}", help="Total number of unresolved alerts across all campaigns and severity levels.")
    with col2:
        critical = alerts_data.get('critical_count', 0)
        st.metric("🔴 Critical", critical, delta=f"+{critical}" if critical > 0 else None, delta_color="inverse", help="Critical alerts require immediate action — e.g., workflow failures, safety check failures, or budget overruns.")
    with col3:
        errors = alerts_data.get('error_count', 0)
        st.metric("⚠️ Errors", errors, delta=f"+{errors}" if errors > 0 else None, delta_color="inverse", help="Error-level alerts indicate problems that need attention, such as deployment failures or content rejections.")
    with col4:
        warnings = alerts_data.get('warning_count', 0)
        st.metric("⚡ Warnings", warnings, help="Warnings highlight potential issues that may need review, such as budget approaching thresholds.")
    with col5:
        info = alerts_data.get('info_count', 0)
        st.metric("ℹ️ Info", info, help="Informational messages about normal workflow operations, such as MARL gating decisions or completed steps.")
    
    st.markdown("---")
    
    alerts = alerts_data.get('alerts', {})
    
    if alerts.get('critical'):
        st.markdown("### 🔴 CRITICAL ALERTS - Immediate Action Required")
        for alert in alerts['critical']:
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.error(f"**{alert['title']}**")
                    st.caption(f"Campaign: `{alert['campaign_id']}` | {alert['event_type']}")
                    st.write(alert['message'])
                    
                    solution = get_suggested_solution(alert)
                    with st.expander("💡 Suggested Solution"):
                        st.info(solution['action'])
                        if solution.get('quick_fix') and solution.get('quick_fix_label'):
                            st.button(f"🔧 {solution['quick_fix_label']}", key=f"fix_crit_{alert['id']}")
                    
                    with st.expander("📋 Event Details"):
                        if alert.get('content_id'):
                            st.markdown(f"**Content ID:** `{alert['content_id']}`")
                        st.markdown(f"**Campaign ID:** `{alert['campaign_id']}`")
                        st.markdown(f"**Event Type:** `{alert['event_type']}`")
                        if alert.get('workflow_node'):
                            st.markdown(f"**Workflow Node:** {alert['workflow_node'].replace('_', ' ').title()}")
                        if alert.get('created_at'):
                            st.markdown(f"**Created:** {alert['created_at']}")
                        if alert.get('details'):
                            st.markdown("---")
                            for key, value in alert['details'].items():
                                key_display = key.replace('_', ' ').title()
                                if isinstance(value, list):
                                    st.markdown(f"**{key_display}:** {', '.join(str(v) for v in value)}")
                                elif isinstance(value, float):
                                    st.markdown(f"**{key_display}:** {value:.2f}")
                                else:
                                    st.markdown(f"**{key_display}:** {value}")
                
                with col2:
                    if st.button("Dismiss", key=f"dismiss_crit_{alert['id']}"):
                        result = api.dismiss_event(alert['id'])
                        if result.get('status') == 'success':
                            st.success("Dismissed!")
                            st.rerun()
            st.markdown("---")
    
    if alerts.get('error'):
        st.markdown("### ⚠️ ERROR ALERTS - Attention Needed")
        for alert in alerts['error']:
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.warning(f"**{alert['title']}**")
                    st.caption(f"Campaign: `{alert['campaign_id']}` | {alert['event_type']}")
                    st.write(alert['message'])
                    
                    solution = get_suggested_solution(alert)
                    with st.expander("💡 Suggested Solution"):
                        st.info(solution['action'])
                
                with col2:
                    if st.button("Dismiss", key=f"dismiss_err_{alert['id']}"):
                        result = api.dismiss_event(alert['id'])
                        if result.get('status') == 'success':
                            st.rerun()
            st.markdown("---")
    
    if alerts.get('warning'):
        st.markdown("### ⚡ WARNING ALERTS - Review Recommended")
        for alert in alerts['warning']:
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.info(f"**{alert['title']}**")
                    st.caption(f"Campaign: `{alert['campaign_id']}` | {alert['event_type']}")
                    st.write(alert['message'])
                with col2:
                    if st.button("Dismiss", key=f"dismiss_warn_{alert['id']}"):
                        result = api.dismiss_event(alert['id'])
                        if result.get('status') == 'success':
                            st.rerun()
            st.markdown("---")
    
    if alerts.get('info'):
        with st.expander(f"ℹ️ Info Messages ({len(alerts['info'])})", expanded=False):
            for alert in alerts['info']:
                st.markdown(f"**{alert['title']}**")
                st.caption(f"Campaign: `{alert['campaign_id']}` | {alert['event_type']}")
                st.write(alert['message'])
                if st.button("Dismiss", key=f"dismiss_info_{alert['id']}"):
                    result = api.dismiss_event(alert['id'])
                    if result.get('status') == 'success':
                        st.rerun()
                st.markdown("---")
    
    if alerts_data.get('total_count', 0) == 0:
        st.success("✅ No active alerts - all systems operating normally")

with tab2:
    st.subheader("📋 Campaign Deep Dive")
    st.caption("Select a campaign to inspect its current state, performance metrics, budget status, and full event history. Use event type filters to focus on specific workflow stages.")
    
    campaigns = api.get_campaigns(limit=500)
    
    if campaigns:
        from utils.data_controls import render_searchable_select
        
        st.caption("💡 Select a campaign to view its complete workflow history and current status.")
        campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="deep_dive_campaign",
            placeholder="Search by name, ID, platform..."
        )
        
        if campaign_id:
            campaign = api.get_campaign(campaign_id)
            events = api.get_campaign_events(campaign_id, limit=200, include_dismissed=True)
            
            if campaign:
                st.markdown("### 📊 Campaign State")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    status = campaign.get('status', 'unknown')
                    render_status_badge(status)
                    
                with col2:
                    reason = get_status_reason(campaign, events)
                    st.markdown(f"**Reason:** {reason}")
                
                with col3:
                    is_mock = campaign.get('is_mock', campaign.get('config', {}).get('is_mock', False))
                    if is_mock:
                        st.warning("🧪 Mock Mode")
                    else:
                        st.success("🌐 Live Mode")
                
                st.markdown("---")
                
                st.markdown("### 📈 Performance Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Impressions", f"{campaign.get('impressions', 0):,}", help="Total number of times campaign content was displayed to users.")
                with col2:
                    st.metric("Clicks", f"{campaign.get('clicks', 0):,}", help="Total number of user clicks on campaign content.")
                with col3:
                    st.metric("Conversions", campaign.get('conversions', 0), help="Number of completed conversion actions (e.g., sign-ups, purchases) attributed to this campaign.")
                with col4:
                    st.metric("CTR", f"{campaign.get('ctr', 0):.2f}%", help="Click-Through Rate — percentage of impressions that resulted in a click.")
                with col5:
                    st.metric("CPL", f"€{campaign.get('cpl', 0):.2f}", help="Cost Per Lead — average cost to acquire one conversion.")
                
                st.markdown("---")
                
                st.markdown("### 💰 Budget Status")
                budget_total = campaign.get('budget_total', 0) or 1
                budget_spent = campaign.get('budget_spent', 0)
                utilization = (budget_spent / budget_total) * 100 if budget_total > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Budget", f"€{budget_total:,.2f}", help="Total allocated budget for this campaign.")
                with col2:
                    st.metric("Spent", f"€{budget_spent:,.2f}", help="Amount of budget spent so far.")
                with col3:
                    st.metric("Remaining", f"€{budget_total - budget_spent:,.2f}", help="Budget remaining before the campaign reaches its limit.")
                with col4:
                    budget_th = get_budget_thresholds()
                    delta_color = "inverse" if utilization >= budget_th['warning'] else "normal"
                    st.metric("Utilization", f"{utilization:.1f}%", help="Percentage of total budget consumed. Warnings trigger at configured thresholds.")
                
                if utilization >= budget_th['critical']:
                    st.error(f"⚠️ Budget Critical: {utilization:.1f}% used")
                elif utilization >= budget_th['warning']:
                    st.warning(f"💰 Budget Warning: {utilization:.1f}% used")
                else:
                    st.info(f"✅ Budget Healthy: {utilization:.1f}% used")
                
                st.progress(min(utilization / 100, 1.0))
                
                st.markdown("---")
                
                st.markdown("### 📜 Complete Event History")
                st.caption(f"Showing all {len(events)} events (including dismissed)")
                
                event_type_filter = st.multiselect(
                    "Filter by Event Type",
                    options=ALL_EVENT_TYPES,
                    default=None,
                    help="Filter events by type to focus on specific workflow stages. Leave empty to show all event types."
                )
                
                if events:
                    filtered_events = events
                    if event_type_filter:
                        filtered_events = [e for e in events if e.get('event_type') in event_type_filter]
                    
                    event_data = []
                    for event in filtered_events:
                        event_data.append({
                            'Time': event.get('created_at', '')[:19],
                            'Type': f"{get_event_icon(event.get('event_type', ''))} {event.get('event_type', '')}",
                            'Severity': event.get('severity', 'info'),
                            'Node': event.get('workflow_node', '-').replace('_', ' ').title() if event.get('workflow_node') else '-',
                            'Title': event.get('title', ''),
                            'Message': event.get('message', '')
                        })
                    
                    df = pd.DataFrame(event_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    with st.expander("📋 Detailed Event View"):
                        for event in filtered_events[:50]:
                            icon = get_event_icon(event.get('event_type', ''))
                            severity = event.get('severity', 'info')
                            
                            if severity == 'critical':
                                st.error(f"{icon} **{event.get('title', 'Event')}**")
                            elif severity == 'error':
                                st.warning(f"{icon} **{event.get('title', 'Event')}**")
                            else:
                                st.info(f"{icon} **{event.get('title', 'Event')}**")
                            
                            st.markdown(f"Type: `{event.get('event_type')}` | Node: {event.get('workflow_node', '-')} | {event.get('created_at', '')[:19]}")
                            st.markdown(event.get('message', ''))
                            
                            if event.get('details'):
                                st.markdown("**Details:**")
                                st.json(event['details'])
                            
                            st.markdown("---")
                else:
                    st.info("No events recorded for this campaign yet")
    else:
        st.info("No campaigns found. Create a campaign to see deep dive analysis.")

with tab3:
    st.subheader("🔄 Workflow Progress Visualization")
    st.caption("Visual node-by-node pipeline tracking showing all 11 workflow stages. Completed nodes are marked ✅, failed nodes ❌, and the current active node 🔄. Optional stages show whether they are enabled or skipped.")
    
    campaigns = api.get_campaigns(limit=500)
    
    if campaigns:
        from utils.data_controls import render_searchable_select
        
        st.caption("💡 Select a campaign to visualize its workflow pipeline and trace execution through each node.")
        campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="workflow_campaign",
            placeholder="Search by name, ID, platform..."
        )
        
        if campaign_id:
            events = api.get_campaign_events(campaign_id, limit=200, include_dismissed=True)
            campaign = api.get_campaign(campaign_id)
            
            current_node = get_current_workflow_node(events)
            
            is_workflow_completed = any(e.get('event_type', '').upper() == 'WORKFLOW_COMPLETED' for e in events)
            
            # ── Detect content_generation failures and show LLM diagnostic ──
            _cg_errors = [
                e for e in events
                if e.get('workflow_node') == 'content_generation'
                and 'error' in e.get('event_type', '').lower()
            ]
            if _cg_errors:
                _err_detail = (_cg_errors[0].get('details') or {}).get('error', '')
                _llm_status = check_llm_readiness(api)
                if not _llm_status['ready']:
                    st.error(
                        "🚨 **Content generation failed because the LLM model is misconfigured.**\n\n"
                        f"**Diagnosis:** {_llm_status['message']}\n\n"
                        "**How to fix:**\n" +
                        "\n".join(f"{i+1}. {s}" for i, s in enumerate(_llm_status['fix_steps'])) +
                        "\n\nAfter fixing the model, restart this campaign from the Campaigns page."
                    )
                elif 'Insufficient claims' in _err_detail or 'model' in _err_detail.lower():
                    st.warning(
                        f"⚠️ **Content generation failed:** `{_err_detail}`\n\n"
                        "The LLM model is active but produced output that did not include required claim citations. "
                        "This can happen if the model is too small or the prompt template needs adjustment.\n\n"
                        "**Try:** 🤖 LLM Management → Model Testing to verify the active model can follow structured prompts."
                    )
            
            st.markdown("### 📊 Complete Workflow Pipeline")
            st.caption("All 11 workflow stages in execution order (core + optional)")
            
            try:
                config_dict = {}
                for key in ['REQUIRE_HUMAN_APPROVAL', 'ENABLE_MARL', 'ENABLE_CANARY_DEPLOYMENT']:
                    val = api.get_config_value(key)
                    if val is not None:
                        config_dict[key] = val
            except Exception:
                config_dict = {}
            
            row1_nodes = WORKFLOW_NODES_ALL[:6]  # First 6 nodes
            row2_nodes = WORKFLOW_NODES_ALL[6:]  # Last 5 nodes
            
            cols1 = st.columns(len(row1_nodes))
            for i, node in enumerate(row1_nodes):
                node_id = node['id']
                node_events = [e for e in events if e.get('workflow_node') == node_id]
                is_optional = node.get('optional', False)
                config_key = node.get('config_key', '')
                config_val = config_dict.get(config_key)
                is_enabled = bool(config_val) if config_val is not None else (not config_key)
                
                is_completed = len(node_events) > 0 and any(
                    'completed' in e.get('event_type', '').lower() or 
                    'passed' in e.get('event_type', '').lower() or 
                    'deployed' in e.get('event_type', '').lower() or 
                    'generated' in e.get('event_type', '').lower() or
                    'added' in e.get('event_type', '').lower()
                    for e in node_events
                )
                is_failed = any('failed' in e.get('event_type', '').lower() for e in node_events)
                is_current = node_id == current_node
                
                with cols1[i]:
                    if is_optional:
                        status_text = "🟢 ON" if is_enabled else "⚪ OFF"
                        st.caption(f"Optional ({status_text})")
                    else:
                        st.caption("Required")
                    
                    if is_failed:
                        st.markdown(f"### ❌")
                        st.error(f"**{node['name']}**")
                    elif is_completed or (is_workflow_completed and len(node_events) > 0):
                        st.markdown(f"### ✅")
                        st.success(f"**{node['name']}**")
                    elif is_current:
                        st.markdown(f"### 🔄")
                        st.warning(f"**{node['name']}**")
                    elif is_optional and not is_enabled:
                        st.markdown(f"### ⏭️")
                        st.info(f"**{node['name']}**")
                    elif is_workflow_completed and len(node_events) == 0:
                        st.markdown(f"### ⏭️")
                        st.info(f"**{node['name']}**")
                    else:
                        st.markdown(f"### ⏳")
                        st.info(f"**{node['name']}**")
                    
                    st.caption(f"{len(node_events)} events" if node_events else ("Skipped" if is_workflow_completed else "0 events"))
                    if node_events:
                        st.caption(f"{node_events[0].get('event_type', '')}")
            
            st.markdown("##### ↓ Flow continues ↓")
            
            cols2 = st.columns(len(row2_nodes))
            for i, node in enumerate(row2_nodes):
                node_id = node['id']
                node_events = [e for e in events if e.get('workflow_node') == node_id]
                is_optional = node.get('optional', False)
                config_key = node.get('config_key', '')
                config_val = config_dict.get(config_key)
                is_enabled = bool(config_val) if config_val is not None else (not config_key)
                
                is_completed = len(node_events) > 0 and any(
                    'completed' in e.get('event_type', '').lower() or 
                    'passed' in e.get('event_type', '').lower() or 
                    'deployed' in e.get('event_type', '').lower() or
                    'approved' in e.get('event_type', '').lower()
                    for e in node_events
                )
                is_failed = any('failed' in e.get('event_type', '').lower() for e in node_events)
                is_current = node_id == current_node
                
                with cols2[i]:
                    if is_optional:
                        status_text = "🟢 ON" if is_enabled else "⚪ OFF"
                        st.caption(f"Optional ({status_text})")
                    else:
                        st.caption("Required")
                    
                    if is_failed:
                        st.markdown(f"### ❌")
                        st.error(f"**{node['name']}**")
                    elif is_completed or (is_workflow_completed and len(node_events) > 0):
                        st.markdown(f"### ✅")
                        st.success(f"**{node['name']}**")
                    elif is_current:
                        st.markdown(f"### 🔄")
                        st.warning(f"**{node['name']}**")
                    elif is_optional and not is_enabled:
                        st.markdown(f"### ⏭️")
                        st.info(f"**{node['name']}**")
                    elif is_workflow_completed and len(node_events) == 0:
                        st.markdown(f"### ⏭️")
                        st.info(f"**{node['name']}**")
                    else:
                        st.markdown(f"### ⏳")
                        st.info(f"**{node['name']}**")
                    
                    st.caption(f"{len(node_events)} events" if node_events else ("Skipped" if is_workflow_completed else "0 events"))
                    if node_events:
                        st.caption(f"{node_events[0].get('event_type', '')}")
            
            st.markdown("---")
            
            st.markdown("### 📜 Node-by-Node Timeline (All 11 Stages)")
            
            # Prevent Streamlit from truncating text in metrics/labels
            st.markdown("""<style>
                [data-testid="stMetricLabel"] { white-space: normal !important; overflow: visible !important; text-overflow: unset !important; }
                [data-testid="stMetricValue"] { white-space: normal !important; overflow: visible !important; text-overflow: unset !important; }
                [data-testid="stCaptionContainer"] { white-space: normal !important; overflow: visible !important; text-overflow: unset !important; }
            </style>""", unsafe_allow_html=True)
            
            for node in WORKFLOW_NODES_ALL:
                node_id = node['id']
                node_events = [e for e in events if e.get('workflow_node') == node_id]
                is_optional = node.get('optional', False)
                config_key = node.get('config_key', '')
                config_val = config_dict.get(config_key)
                is_enabled = bool(config_val) if config_val is not None else (not config_key)
                
                optional_label = " (Optional - Enabled)" if is_optional and is_enabled else " (Optional - Disabled)" if is_optional else ""
                
                with st.expander(f"{node['icon']} {node['name']} ({len(node_events)} events){optional_label}", expanded=len(node_events) > 0):
                    if node_events:
                        for event in node_events:
                            severity = event.get('severity', 'info')
                            icon = get_event_icon(event.get('event_type', ''))
                            
                            severity_label = ""
                            if severity == 'critical':
                                severity_label = " `🔴 CRITICAL`"
                            elif severity == 'error':
                                severity_label = " `🟠 ERROR`"
                            elif severity == 'warning':
                                severity_label = " `🟡 WARNING`"
                            
                            st.markdown(f"{icon} **{event.get('title', 'Event')}**{severity_label}")
                            st.markdown(f"`{event.get('event_type')}` | {event.get('created_at', '')[:19]}")
                            st.markdown(event.get('message', ''))
                            
                            if event.get('details'):
                                details = event['details']
                                st.markdown("**Output Details:**")
                                
                                # Render all details as a readable table to prevent truncation
                                scalar_details = {k: v for k, v in details.items() if isinstance(v, (int, float, str, bool))}
                                if scalar_details:
                                    detail_rows = []
                                    for dk, dv in scalar_details.items():
                                        label = dk.replace('_', ' ').title()
                                        if isinstance(dv, float):
                                            detail_rows.append(f"| {label} | `{dv:.2f}` |")
                                        elif isinstance(dv, bool):
                                            detail_rows.append(f"| {label} | `{dv}` |")
                                        else:
                                            detail_rows.append(f"| {label} | `{dv}` |")
                                    table_md = "| Detail | Value |\n|---|---|\n" + "\n".join(detail_rows)
                                    st.markdown(table_md)
                                
                                complex_details = {k: v for k, v in details.items() if isinstance(v, (list, dict))}
                                if complex_details:
                                    st.json(complex_details)
                            
                            st.markdown("---")
                    else:
                        if is_optional and not is_enabled:
                            st.caption(f"⏭️ Skipped - {config_key}=False in configuration")
                        else:
                            st.caption("No events for this node yet")
            
            all_node_ids = [n['id'] for n in WORKFLOW_NODES_ALL]
            unmapped_events = [e for e in events if not e.get('workflow_node') or e.get('workflow_node') not in all_node_ids]
            if unmapped_events:
                with st.expander(f"📋 Other Events ({len(unmapped_events)})", expanded=False):
                    for event in unmapped_events:
                        icon = get_event_icon(event.get('event_type', ''))
                        st.markdown(f"{icon} **{event.get('title', 'Event')}**")
                        st.markdown(f"`{event.get('event_type')}` | {event.get('created_at', '')[:19]}")
                        st.markdown(event.get('message', ''))
                        if event.get('details'):
                            details = event['details']
                            scalar_details = {k: v for k, v in details.items() if isinstance(v, (int, float, str, bool))}
                            if scalar_details:
                                detail_rows = []
                                for dk, dv in scalar_details.items():
                                    label = dk.replace('_', ' ').title()
                                    if isinstance(dv, float):
                                        detail_rows.append(f"| {label} | `{dv:.2f}` |")
                                    else:
                                        detail_rows.append(f"| {label} | `{dv}` |")
                                st.markdown("| Detail | Value |\n|---|---|\n" + "\n".join(detail_rows))
                            complex_details = {k: v for k, v in details.items() if isinstance(v, (list, dict))}
                            if complex_details:
                                st.json(complex_details)
                        st.markdown("---")
    else:
        st.info("No campaigns found")

with tab4:
    st.subheader("⚙️ Campaign Configurations")
    st.caption("View deployment mode (mock vs. live), budget settings, timing, and full configuration JSON for each campaign. Platform-wide settings like mock data inclusion are also configurable here.")
    
    st.markdown("### 🌐 Platform-Wide Settings")
    try:
        mock_status = api.get_mock_mode_status()
        include_mock = mock_status.get('include_mock_in_metrics', mock_status.get('settings', {}).get('INCLUDE_MOCK_IN_METRICS', True))
        
        col_toggle, col_desc = st.columns([1, 3])
        with col_toggle:
            new_include_mock = st.toggle(
                "Include Mock Data in Metrics",
                value=bool(include_mock),
                key="include_mock_toggle",
                help="When ON, mock campaign data appears in all KPIs (clearly labeled). When OFF, only real production data is shown."
            )
        with col_desc:
            if new_include_mock:
                st.info("🧪 Mock campaign data is **included** in platform metrics and KPIs. All affected pages show a mock data indicator.")
            else:
                st.success("✅ Only **production** campaign data is shown in metrics and KPIs.")
        
        if new_include_mock != bool(include_mock):
            try:
                save_resp = api.session.put(
                    f"{api.base_url}/api/v1/config/value/INCLUDE_MOCK_IN_METRICS",
                    json={"value": str(new_include_mock)},
                    timeout=5
                )
                if save_resp.status_code == 200:
                    st.success("✅ Setting saved. Refresh pages to see the change.")
                    st.rerun()
            except Exception as save_err:
                st.error(f"Failed to save setting: {save_err}")
    except Exception:
        st.caption("Could not load platform settings")
    
    st.markdown("---")
    st.markdown("### 📋 Campaign-Specific Configurations")
    
    campaigns = api.get_campaigns(limit=500)
    
    if campaigns:
        from utils.data_controls import render_searchable_select
        
        st.caption("💡 Select a campaign to inspect its deployment mode, budget settings, and full configuration.")
        campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="config_campaign",
            placeholder="Search by name, ID, platform..."
        )
        
        if campaign_id:
            campaign = api.get_campaign(campaign_id)
            
            if campaign:
                st.markdown("---")
                
                st.markdown("### 🚀 Deployment Mode")
                col1, col2 = st.columns(2)
                
                with col1:
                    config = campaign.get('config', {})
                    is_mock = campaign.get('is_mock', config.get('is_mock', config.get('mock_mode', False)))
                    
                    if is_mock:
                        st.warning("### 🧪 MOCK MODE")
                        st.write("Campaign is using **simulated deployment**. No real platform API calls are made.")
                        st.write("KPIs are generated by the simulation engine.")
                    else:
                        st.success("### 🌐 LIVE MODE")
                        st.write("Campaign is deployed to **real platforms**.")
                        st.write("KPIs reflect actual platform performance.")
                
                with col2:
                    st.markdown("**Platform:** " + campaign.get('platform', 'N/A'))
                    st.markdown("**Goal:** " + (campaign.get('goal') or 'N/A'))
                    st.markdown("**Persona:** " + (campaign.get('target_persona') or 'N/A'))
                
                st.markdown("---")
                
                st.markdown("### 💰 Budget Configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Budget", f"€{campaign.get('budget_total', 0):,.2f}", help="Total budget allocated for this campaign.")
                with col2:
                    st.metric("Daily Limit", f"€{campaign.get('budget_daily_limit', 0):,.2f}", help="Maximum spend allowed per day. The system auto-pauses when this limit is reached.")
                with col3:
                    utilization = (campaign.get('budget_spent', 0) / max(campaign.get('budget_total', 1), 1)) * 100
                    st.metric("Current Utilization", f"{utilization:.1f}%", help="Percentage of total budget consumed so far.")
                
                budget_th = get_budget_thresholds()
                st.markdown("**Threshold Settings:**")
                st.write(f"- ⚠️ Warning Threshold: **{budget_th['warning']}%** budget utilization")
                st.write(f"- 🔴 Critical Threshold: **{budget_th['critical']}%** budget utilization")
                st.write(f"- 🛑 Auto-Pause Threshold: **{budget_th['auto_pause']}%** budget utilization")
                
                st.markdown("---")
                
                st.markdown("### 📅 Campaign Timing")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    created = campaign.get('created_at', 'N/A')
                    st.markdown(f"**Created:** {created[:19] if created != 'N/A' else 'N/A'}")
                with col2:
                    start = campaign.get('start_date', 'N/A')
                    st.markdown(f"**Start Date:** {start[:10] if start and start != 'N/A' else 'N/A'}")
                with col3:
                    end = campaign.get('end_date', 'N/A')
                    st.markdown(f"**End Date:** {end[:10] if end and end != 'N/A' else 'N/A'}")
                
                st.markdown("---")
                
                st.markdown("### 📋 Full Configuration")
                with st.expander("View Raw Configuration", expanded=False):
                    st.json(campaign.get('config', {}))
                
                with st.expander("View Full Campaign Object", expanded=False):
                    st.json(campaign)
    else:
        st.info("No campaigns found")

with tab5:
    st.subheader("💡 Actionable Recommendations")
    st.caption("AI-generated suggestions with root cause analysis, affected workflow nodes, and step-by-step fix instructions. Recommendations are prioritized by severity (high → medium → low).")
    
    campaigns = api.get_campaigns(limit=500)
    
    if campaigns:
        from utils.data_controls import render_searchable_select
        
        st.caption("💡 Select a campaign to view AI-generated recommendations with root cause analysis and fix steps.")
        campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="recommendations_campaign",
            placeholder="Search by name, ID, platform..."
        )
        
        if campaign_id:
            campaign = api.get_campaign(campaign_id)
            events = api.get_campaign_events(campaign_id, limit=100)
            
            if campaign:
                recommendations = generate_recommendations(campaign, events)
                
                if recommendations:
                    for rec in recommendations:
                        priority = rec.get('priority', 'low')
                        
                        with st.container():
                            if priority == 'high':
                                st.error(f"### {rec['icon']} {rec['title']}")
                            elif priority == 'medium':
                                st.warning(f"### {rec['icon']} {rec['title']}")
                            else:
                                st.info(f"### {rec['icon']} {rec['title']}")
                            
                            st.markdown("**🔍 Root Cause:**")
                            st.write(rec.get('root_cause', 'Unknown'))
                            
                            if rec.get('affected_node'):
                                st.caption(f"📍 Affected: **{rec['affected_node']}**")
                            
                            if rec.get('details'):
                                st.markdown("**📋 Details:**")
                                details = rec['details']
                                detail_md = ""
                                for key, value in details.items():
                                    detail_md += f"- **{key}:** {value}\n"
                                st.markdown(detail_md)
                            
                            if rec.get('fix_steps'):
                                st.markdown("**🔧 How to Fix:**")
                                for step in rec['fix_steps']:
                                    st.markdown(step)
                            
                            st.markdown("---")
                else:
                    st.success("✅ No recommendations - campaign is running optimally!")
                
                st.markdown("### ⚡ Quick Actions")
                col1, col2, col3 = st.columns(3)
                
                status = campaign.get('status', 'unknown')
                
                with col1:
                    if status == 'paused':
                        if st.button("▶️ Resume Campaign", use_container_width=True):
                            result = api.start_campaign(campaign_id)
                            if result:
                                st.success("Campaign resumed!")
                                st.rerun()
                    elif status == 'draft':
                        if st.button("🚀 Start Campaign", use_container_width=True):
                            result = api.start_campaign(campaign_id)
                            if result:
                                st.success("Campaign started!")
                                st.rerun()
                    elif status == 'running':
                        if st.button("⏸️ Pause Campaign", use_container_width=True):
                            result = api.pause_campaign(campaign_id)
                            if result:
                                st.success("Campaign paused!")
                                st.rerun()
                
                with col2:
                    if st.button("🔄 Check Completion", use_container_width=True):
                        result = api.check_campaign_completion(campaign_id)
                        if result:
                            if result.get('should_complete'):
                                st.warning(f"Should complete: {result.get('reason')}")
                            else:
                                st.info("Campaign should continue running")
                
                with col3:
                    if st.button("📊 View Analytics", use_container_width=True):
                        st.info("Navigate to Analytics page for detailed metrics")
    else:
        st.info("No campaigns found")

st.markdown("---")
st.markdown("---")

st.markdown("## 📊 Global Events Summary")

col1, col2 = st.columns(2)
with col1:
    days_summary = st.slider("Summary Period (days)", 1, 30, 7, help="Select the number of past days to include in the global events summary.")
with col2:
    if st.button("Refresh Summary"):
        st.rerun()

summary = api.get_events_summary(days=days_summary)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Events", summary.get('total_events', 0), help="Total number of workflow events recorded across all campaigns in the selected period.")
with col2:
    st.metric("Actionable Pending", summary.get('actionable_pending', 0), help="Number of events that require user action and have not been resolved or dismissed.")
with col3:
    st.metric("Critical", summary.get('by_severity', {}).get('critical', 0), help="Number of critical-severity events in the selected period.")
with col4:
    st.metric("Errors", summary.get('by_severity', {}).get('error', 0), help="Number of error-severity events in the selected period.")

st.markdown("### Events by Type")
by_type = summary.get('by_type', {})
if by_type:
    type_items = list(by_type.items())
    chunks = [type_items[i:i+6] for i in range(0, len(type_items), 6)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for idx, (event_type, count) in enumerate(chunk):
            with cols[idx]:
                icon = get_event_icon(event_type)
                st.metric(f"{icon} {event_type.replace('_', ' ').title()}", count, help=f"Number of '{event_type}' events recorded in the selected period.")
else:
    st.info("No events in this time period")

if auto_refresh:
    # Non-blocking auto-refresh — JS timer triggers rerun without blocking the UI
    st_autorefresh(interval=5000, limit=None, key="transparency_autorefresh")
    st.caption("🔄 Auto-refresh active (every ~5s)")

st.markdown("---")
st.caption(f"System Transparency Dashboard | Complete workflow visibility | Last updated: {datetime.now().strftime('%H:%M:%S')}")
