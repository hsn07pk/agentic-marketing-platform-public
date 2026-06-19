"""
Cost Control - Budget Tracking and Optimization
Uses real cost tracking data from the CostTracking database table.
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
from utils.data_controls import render_data_controls
from components import (
    render_metric_card, create_bar_chart,
    create_line_chart, create_pie_chart, create_gauge_chart
)
from app_config import get_budget_thresholds

st.set_page_config(page_title="Cost Control - Agentic AI", page_icon="💰", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("💰 Cost Control & Budget Management")
st.caption("Track spending, optimize budgets, and control AI costs")

with st.expander("ℹ️ Understanding Cost Control", expanded=False):
    st.markdown("""
    **What Cost Control Monitors**
    - **LLM API call costs** — every call to OpenAI or local Ollama models is tracked with token counts and model-specific rates.
    - **Token consumption** — input and output tokens are recorded per request; cost = `token_count × model_rate`.
    - **Budget allocation** — each campaign has a total budget; this page shows how much has been spent and what remains.

    **How Costs Are Calculated**
    - OpenAI models are billed per token at the rate set by the provider (e.g. GPT-4o is more expensive than GPT-3.5).
    - Ollama (local) models have an estimated cost based on equivalent cloud token rates, but are effectively free to run.
    - Campaign spend is the sum of `budget_spent` across all campaigns.

    **Budget Management**
    - Campaigns are flagged as 🟢 Healthy, 🟡 Warning (≥ 75 % used), or 🔴 Critical (≥ 90 % used).
    - Daily and monthly burn rates are projected to forecast when budgets will be exhausted.

    **Cost Optimization Tips**
    - **Use Ollama (local, free)** for drafts and iterations; reserve OpenAI for final quality checks.
    - **Batch operations** — group similar prompts to reduce per-call overhead.
    - **Semantic caching** — identical or similar prompts return cached responses, avoiding repeat API calls (target > 20 % hit rate).
    - **Prompt optimization** — shorter, focused prompts consume fewer tokens and cost less.

    **How the System Saves Costs**
    - Automatic local-LLM fallback when OpenAI is unavailable or budget thresholds are reached.
    - Response caching via semantic similarity avoids redundant API calls.
    - Prompt templates are pre-optimized to minimize token usage.
    """)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "💳 Budget Tracking",
    "🤖 AI Model Costs",
    "📈 Cost Forecasting",
    "💾 Semantic Cache",
    "🎯 Cost by Campaign"
])


with tab1:
    st.subheader("Cost Overview")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        days = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2, help="Number of past days to include in the cost overview. Longer periods smooth out daily fluctuations.")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, help="Reload all cost data from the database."):
            st.rerun()
    
    try:
        campaigns = api.get_campaigns(limit=1000)
        cost_summary = api.get_cost_summary(days=days)
        daily_costs = api.get_daily_costs(days=days)
        
        if cost_summary.get('includes_mock_data', False):
            st.info("🧪 **Includes Mock Data** — Cost metrics include mock campaign data. Toggle off via Configuration → INCLUDE_MOCK_IN_METRICS.")
        
        campaign_spent = sum(c.get('budget_spent', 0) for c in campaigns)
        campaign_budget = sum(c.get('budget_total', 0) for c in campaigns)
        
        ai_cost = cost_summary.get('total_cost', 0)
        ai_calls = cost_summary.get('total_calls', 0)
        
        total_spent = campaign_spent + ai_cost
        total_budget = campaign_budget
        
        st.caption("High-level spending summary: campaign advertising spend, AI/LLM inference costs, and budget utilization.")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            render_metric_card("Campaign Spend", f"€{campaign_spent:,.2f}")
        
        with col2:
            render_metric_card("AI/LLM Costs", f"€{ai_cost:,.2f}" if ai_cost >= 1 else f"€{ai_cost:,.4f}")
        
        with col3:
            remaining = total_budget - campaign_spent
            render_metric_card("Budget Remaining", f"€{remaining:,.2f}")
        
        with col4:
            pct_spent = (campaign_spent / total_budget * 100) if total_budget > 0 else 0
            render_metric_card("Budget Used", f"{pct_spent:.1f}", suffix="%")
        
        with col5:
            avg_daily = daily_costs.get('avg_daily_cost', 0)
            render_metric_card("Avg Daily AI Cost", f"€{avg_daily:.2f}" if avg_daily >= 1 else f"€{avg_daily:.4f}")
        
        st.markdown("---")
        st.markdown("### Budget Utilization")
        st.caption("The gauge shows how much of the total campaign budget has been consumed. The line chart tracks daily AI inference costs over the selected period.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = create_gauge_chart(
                value=campaign_spent,
                max_value=total_budget if total_budget > 0 else 1000,
                title="Campaign Budget",
                thresholds={
                    'low': total_budget * 0.5,
                    'medium': total_budget * 0.8,
                    'high': total_budget
                } if total_budget > 0 else None
            )
            st.plotly_chart(fig, use_container_width=True, key="cost_budget_gauge")
        
        with col2:
            daily_data = daily_costs.get('daily', [])
            
            if daily_data:
                trend_df = pd.DataFrame(daily_data)
                if 'date' in trend_df.columns and 'cost' in trend_df.columns:
                    fig = px.line(
                        trend_df,
                        x='date',
                        y='cost',
                        title=f"Daily AI Cost Trend ({days} days)",
                        markers=True
                    )
                    fig.update_layout(yaxis_title="AI Cost (€)")
                    st.plotly_chart(fig, use_container_width=True, key="cost_daily_spend")
                else:
                    st.info("No daily cost data available yet. Run some campaigns to generate data.")
            else:
                st.info("No AI cost data tracked yet. Run content generation or safety validation to start tracking.")
        
        st.markdown("---")
        
        st.markdown("### Cost Breakdown")
        st.caption("Platform pie chart shows advertising spend distribution. Agent bar chart shows AI costs grouped by the agent that initiated the call (e.g. content_creator, safety_validator).")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Campaign Spend by Platform")
            
            platform_costs = {}
            for c in campaigns:
                platform = c.get('platform', 'unknown')
                spent = float(c.get('budget_spent', 0))
                platform_costs[platform] = platform_costs.get(platform, 0) + spent
            
            if platform_costs and sum(platform_costs.values()) > 0:
                # Use explicit lists for go.Pie to avoid DataFrame interpretation issues
                platforms = list(platform_costs.keys())
                spends = [float(v) for v in platform_costs.values()]
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=platforms,
                        values=spends,
                        textinfo='label+percent',
                        hovertemplate='%{label}<br>€%{value:,.2f}<br>%{percent}<extra></extra>'
                    )
                ])
                fig.update_layout(title="Campaign Spending by Platform")
                st.plotly_chart(fig, use_container_width=True, key="cost_platform_breakdown")
            else:
                st.info("No platform cost data")
        
        with col2:
            st.markdown("#### AI Costs by Agent")
            
            by_agent = cost_summary.get('by_agent', [])
            
            if by_agent and any(a.get('cost', 0) > 0 for a in by_agent):
                # Extract data as explicit lists to avoid any DataFrame interpretation issues
                agent_types = [a.get('agent_type', 'unknown') for a in by_agent]
                costs = [float(a.get('cost', 0)) for a in by_agent]
                
                # Use go.Figure with explicit trace to ensure correct values
                fig = go.Figure(data=[
                    go.Bar(
                        x=agent_types,
                        y=costs,
                        marker_color='steelblue',
                        text=[f"€{c:.2f}" for c in costs],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="AI Cost by Agent Type",
                    xaxis_title="Agent Type",
                    yaxis_title="Cost (€)"
                )
                st.plotly_chart(fig, use_container_width=True, key="cost_per_content")
            else:
                st.info("No AI agent cost data yet. Run workflows to start tracking.")
        
        st.markdown("---")
        st.markdown("### Top Spending Campaigns")
        st.caption("Campaigns ranked by total advertising spend. Review high-spend campaigns to ensure ROI targets are being met.")
        
        top_campaigns = sorted(campaigns, key=lambda c: c.get('budget_spent', 0), reverse=True)[:10]
        
        if top_campaigns:
            top_data = []
            for c in top_campaigns:
                total = c.get('budget_total', 0)
                spent = c.get('budget_spent', 0)
                pct = (spent / total * 100) if total > 0 else 0
                
                top_data.append({
                    'Campaign': c.get('name', 'Unnamed'),
                    'Platform': c.get('platform', 'N/A').capitalize(),
                    'Spent': f"€{spent:.2f}",
                    'Budget': f"€{total:.2f}",
                    'Used %': f"{pct:.1f}%"
                })
            
            df = pd.DataFrame(top_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No campaign spending data")
    
    except Exception as e:
        st.error(f"Failed to load cost overview: {str(e)}")

with tab2:
    st.subheader("Budget Tracking & Alerts")
    st.caption("Monitor campaign budgets in real time. Campaigns approaching their limits are flagged so you can take action before overspending.")
    
    try:
        campaigns = api.get_campaigns(limit=1000)
        
        if campaigns:
            st.markdown("#### Budget Alerts")
            
            budget_thresholds = get_budget_thresholds()
            critical_threshold = budget_thresholds['critical']
            warning_threshold = budget_thresholds['warning']
            
            critical_campaigns = []
            warning_campaigns = []
            healthy_campaigns = []
            
            for c in campaigns:
                if c.get('status') in ['running', 'active']:
                    budget_total = c.get('budget_total', 0)
                    budget_spent = c.get('budget_spent', 0)
                    
                    if budget_total > 0:
                        pct = (budget_spent / budget_total) * 100
                        
                        if pct >= critical_threshold:
                            critical_campaigns.append((c, pct))
                        elif pct >= warning_threshold:
                            warning_campaigns.append((c, pct))
                        else:
                            healthy_campaigns.append((c, pct))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"🔴 Critical (≥{critical_threshold}%)", len(critical_campaigns), help="Campaigns that have consumed ≥ 90 % of their budget. Immediate action recommended — pause or increase budget.")
                for c, pct in critical_campaigns:
                    st.error(f"**{c.get('name')}**: {pct:.0f}% used")
            
            with col2:
                st.metric(f"🟡 Warning (≥{warning_threshold}%)", len(warning_campaigns), help="Campaigns that have consumed ≥ 75 % of their budget. Review pacing and consider adjustments.")
                for c, pct in warning_campaigns:
                    st.warning(f"**{c.get('name')}**: {pct:.0f}% used")
            
            with col3:
                st.metric(f"🟢 Healthy (<{warning_threshold}%)", len(healthy_campaigns), help="Campaigns spending within safe limits. No action needed.")
                st.success(f"{len(healthy_campaigns)} campaigns within budget")
            
            st.markdown("---")
            
            st.markdown("#### All Campaigns Budget Status")
            
            budget_data = []
            for c in campaigns:
                total = c.get('budget_total', 0)
                spent = c.get('budget_spent', 0)
                remaining = total - spent
                pct = (spent / total * 100) if total > 0 else 0
                
                if pct >= 90:
                    status = "🔴 Critical"
                elif pct >= 75:
                    status = "🟡 Warning"
                else:
                    status = "🟢 Healthy"
                
                budget_data.append({
                    'Campaign': c.get('name', 'Unnamed'),
                    'Platform': c.get('platform', 'N/A').capitalize(),
                    'Total Budget': f"€{total:.2f}",
                    'Spent': f"€{spent:.2f}",
                    'Remaining': f"€{remaining:.2f}",
                    'Used %': f"{pct:.1f}%",
                    'Status': status
                })
            
            df = pd.DataFrame(budget_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if st.button("📥 Export Budget Report", help="Download the budget status table as a CSV file for offline analysis or reporting."):
                from components import export_to_csv
                export_to_csv(df, f"budget_report_{datetime.now().strftime('%Y%m%d')}.csv")
        
        else:
            st.info("No campaigns to track")
    
    except Exception as e:
        st.error(f"Failed to load budget tracking: {str(e)}")

with tab3:
    st.subheader("AI Model Costs")
    st.caption("Real cost breakdown from LLM API calls and local models")
    
    days = st.selectbox("Time Period (AI Costs)", [7, 14, 30, 60, 90], index=2, key="ai_model_days", help="Filter AI cost data to this many past days. Shorter windows show recent trends; longer windows reveal patterns.")
    
    try:
        cost_summary = api.get_cost_summary(days=days)
        by_model = api.get_costs_by_model(days=days)
        by_agent = api.get_costs_by_agent(days=days)
        
        total_cost = cost_summary.get('total_cost', 0)
        total_calls = cost_summary.get('total_calls', 0)
        
        if total_calls == 0:
            st.info("""
            **No AI cost data tracked yet.**
            
            Run content generation or safety validation workflows to start tracking LLM costs.
            Costs are tracked for both OpenAI API calls and local Ollama models (estimated).
            """)
        else:
            st.markdown("### Cost by Provider")
            st.caption("Compares spending between OpenAI (cloud, paid per token) and Ollama (local, effectively free). Shifting work to Ollama reduces costs.")
            by_provider = cost_summary.get('by_provider', [])
            
            col1, col2, col3 = st.columns(3)
            
            openai_cost = sum(p.get('cost', 0) for p in by_provider if p.get('provider') == 'openai')
            ollama_cost = sum(p.get('cost', 0) for p in by_provider if p.get('provider') == 'ollama')
            other_cost = total_cost - openai_cost - ollama_cost
            
            with col1:
                render_metric_card("OpenAI API", f"€{openai_cost:.2f}" if openai_cost >= 1 else f"€{openai_cost:.4f}")
            
            with col2:
                render_metric_card("Ollama (Local)", f"€{ollama_cost:.2f}" if ollama_cost >= 1 else f"€{ollama_cost:.4f}")
                if ollama_cost > 0:
                    st.caption("*Estimated cost based on token usage")
            
            with col3:
                render_metric_card("Total LLM Calls", f"{total_calls}")
            
            st.markdown("---")
            
            st.caption("The pie chart shows cost share per model/source. The bar chart shows which agent types (content creator, safety validator, etc.) are driving costs.")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Cost by Model/Source")
                
                models = by_model.get('models', [])
                if models:
                    model_df = pd.DataFrame(models)
                    
                    if 'source' in model_df.columns and 'cost' in model_df.columns:
                        sources = model_df['source'].tolist()
                        costs = [float(c) for c in model_df['cost'].tolist()]
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=sources,
                            values=costs,
                            textinfo='percent+label',
                            hovertemplate='%{label}: €%{value:.2f}<extra></extra>'
                        )])
                        fig.update_layout(title="AI Model Cost Distribution")
                        st.plotly_chart(fig, use_container_width=True, key="cost_trend_analysis")
                else:
                    st.info("No model breakdown data")
            
            with col2:
                st.markdown("#### Cost by Agent Type")
                
                agents = by_agent.get('agents', [])
                if agents:
                    # Extract data as explicit lists
                    agent_types = [a.get('agent_type', 'unknown') for a in agents]
                    costs = [float(a.get('cost', 0)) for a in agents]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=agent_types,
                            y=costs,
                            marker_color='lightgreen',
                            text=[f"€{c:.2f}" for c in costs],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Cost by Agent Type",
                        xaxis_title="Agent Type",
                        yaxis_title="Cost (€)"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="cost_category_breakdown")
                else:
                    st.info("No agent breakdown data")
            
            st.markdown("---")
            st.markdown("#### Detailed Cost Breakdown")
            st.caption("Per-model breakdown with exact cost and percentage share. Use this to identify which models to optimize or replace.")
            
            models = by_model.get('models', [])
            if models:
                details_df = pd.DataFrame(models)
                if 'cost' in details_df.columns:
                    details_df['cost'] = details_df['cost'].apply(lambda x: f"€{x:.6f}")
                if 'percentage' in details_df.columns:
                    details_df['percentage'] = details_df['percentage'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(details_df, use_container_width=True, hide_index=True)
        

    except Exception as e:
        st.error(f"Failed to load AI model costs: {str(e)}")

with tab4:
    st.subheader("Cost Forecasting")
    st.caption("Predict future spending based on actual historical burn rate. Forecasts assume current usage patterns continue unchanged.")
    
    forecast_days = st.selectbox("Forecast Period", [7, 14, 30, 60, 90], index=2, key="forecast_days", help="How many days into the future to project AI spending. Forecasts are based on your historical daily burn rate.")
    
    try:
        forecast = api.get_cost_forecast(days=forecast_days)
        campaigns = api.get_campaigns(limit=1000)
        
        daily_burn = forecast.get('daily_burn_rate', 0)
        based_on_days = forecast.get('based_on_days', 0)
        historical_total = forecast.get('historical_total', 0)
        
        active_campaigns = [c for c in campaigns if c.get('status') in ['running', 'active']]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_metric_card("Daily AI Burn Rate", f"€{daily_burn:.2f}" if daily_burn >= 1 else f"€{daily_burn:.4f}")
            st.caption(f"Based on {based_on_days} active days")
        
        with col2:
            weekly_forecast = forecast.get('weekly_forecast', daily_burn * 7)
            render_metric_card("7-Day AI Forecast", f"€{weekly_forecast:.2f}" if weekly_forecast >= 1 else f"€{weekly_forecast:.4f}")
        
        with col3:
            monthly_forecast = forecast.get('monthly_forecast', daily_burn * 30)
            render_metric_card("30-Day AI Forecast", f"€{monthly_forecast:.2f}" if monthly_forecast >= 1 else f"€{monthly_forecast:.4f}")
        
        st.markdown("---")
        
        st.markdown("#### AI Spending Forecast (Next {} Days)".format(forecast_days))
        
        forecast_data = forecast.get('forecast', [])
        
        if forecast_data:
            forecast_df = pd.DataFrame(forecast_data)
            
            if 'date' in forecast_df.columns and 'cumulative_cost' in forecast_df.columns:
                fig = px.line(
                    forecast_df,
                    x='date',
                    y='cumulative_cost',
                    title="Projected Cumulative AI Spending",
                    markers=True
                )
                fig.update_layout(yaxis_title="Cumulative AI Cost (€)")
                st.plotly_chart(fig, use_container_width=True, key="cost_forecast")
            else:
                st.info("No forecast data available")
        else:
            st.info("Not enough historical data for forecasting. Run more workflows to generate data.")
        
        st.markdown("---")
        st.markdown("#### Active Campaign Budget Status")
        st.caption("Shows remaining budget and estimated days until exhaustion for each active campaign based on its current daily spend rate.")
        
        if active_campaigns:
            budget_status_data = []
            for c in active_campaigns[:10]:
                budget_total = c.get('budget_total', 0)
                budget_spent = c.get('budget_spent', 0)
                remaining = budget_total - budget_spent
                pct_used = (budget_spent / budget_total * 100) if budget_total > 0 else 0
                
                start_date = c.get('start_date')
                if start_date:
                    try:
                        if isinstance(start_date, str):
                            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        else:
                            start = start_date
                        campaign_days = max(1, (datetime.now(start.tzinfo) - start).days)
                    except:
                        campaign_days = 1
                else:
                    campaign_days = 1
                
                daily_spend_rate = budget_spent / campaign_days if campaign_days > 0 else 0
                days_until_exhaustion = remaining / daily_spend_rate if daily_spend_rate > 0 else float('inf')
                
                if pct_used >= 90:
                    status = "🔴 Critical"
                elif pct_used >= 75:
                    status = "🟡 Warning"  
                else:
                    status = "🟢 Healthy"
                
                budget_status_data.append({
                    'Campaign': c.get('name', 'Unnamed'),
                    'Platform': c.get('platform', 'N/A').capitalize(),
                    'Budget': f"€{budget_total:.2f}",
                    'Spent': f"€{budget_spent:.2f}",
                    'Remaining': f"€{remaining:.2f}",
                    'Daily Rate': f"€{daily_spend_rate:.2f}" if daily_spend_rate > 0 else "N/A",
                    'Days Left': f"{days_until_exhaustion:.0f}" if days_until_exhaustion < float('inf') else "∞",
                    'Status': status
                })
            
            if budget_status_data:
                df = pd.DataFrame(budget_status_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No budget data available for active campaigns")
        else:
            st.info("No active campaigns to analyze")
    
    except Exception as e:
        st.error(f"Failed to load cost forecasting: {str(e)}")

# TAB 5: SEMANTIC CACHE (Research Plan Section 10.2 - Target >20%)
with tab5:
    st.subheader("💾 Semantic Cache Performance")
    st.caption("Monitor LLM response caching efficiency (Target: > 20% hit rate)")

    try:
        cache_metrics = api.get_semantic_cache_metrics()
        
        st.markdown("### Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        hit_rate = cache_metrics.get('hit_rate', 0) / 100 if cache_metrics.get('hit_rate', 0) > 1 else cache_metrics.get('hit_rate', 0)
        target_rate = cache_metrics.get('hit_rate_target', 20) / 100
        total_hits = cache_metrics.get('cache_hits', 0)
        total_misses = cache_metrics.get('cache_misses', 0)
        total_queries = cache_metrics.get('total_queries', 0)
        passes_target = cache_metrics.get('meets_target', False)
        cost_saved = cache_metrics.get('estimated_cost_savings', 0)
        
        with col1:
            delta = (hit_rate - target_rate) * 100
            delta_color = "normal" if passes_target else "inverse"
            st.metric(
                "Cache Hit Rate",
                f"{hit_rate:.1%}",
                delta=f"{delta:+.1f}% from target",
                delta_color=delta_color,
                help="Percentage of LLM queries served from cache instead of making a new API call. Higher is better — above 20 % meets the research target."
            )
        
        with col2:
            st.metric("Target Rate", f"> {target_rate:.0%}", help="The minimum cache hit rate goal from the research plan (Section 10.2). The system should exceed this threshold.")
        
        with col3:
            st.metric("Cache Hits", f"{total_hits:,}", help="Total number of queries successfully served from the semantic cache, avoiding an API call.")
        
        with col4:
            st.metric("Cache Misses", f"{total_misses:,}", help="Queries that did not match any cached response and required a fresh API call. High misses indicate low cache coverage.")
        
        with col5:
            st.metric("Cost Saved", f"€{cost_saved:.2f}", help="Estimated money saved by serving cached responses instead of making paid API calls. Calculated as cache_hits × avg_call_cost.")
        
        st.markdown("---")
        if passes_target:
            st.success(f"✅ **PASSING** - Cache hit rate ({hit_rate:.1%}) exceeds target ({target_rate:.0%})")
        else:
            st.warning(f"⚠️ **BELOW TARGET** - Cache hit rate ({hit_rate:.1%}) is below target ({target_rate:.0%})")
            st.info("""
            **To improve cache hit rate:**
            - Run more similar queries to populate the cache
            - Review semantic similarity thresholds
            - Ensure cache is persisting between restarts
            """)
        
        st.markdown("---")
        st.markdown("### Cache Hit Rate Gauge")
        st.caption("The gauge visualizes current hit rate against the 20 % target. Green zone (≥ 50 %) is excellent, yellow (20–50 %) meets the target, red (< 20 %) needs improvement.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=hit_rate * 100,
                number={'suffix': '%'},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Hit Rate (%)"},
                delta={'reference': target_rate * 100, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': '#ef4444'},
                        {'range': [20, 50], 'color': '#f59e0b'},
                        {'range': [50, 100], 'color': '#22c55e'}
                    ],
                    'threshold': {
                        'line': {'color': "green", 'width': 4},
                        'thickness': 0.75,
                        'value': target_rate * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="cost_roi_scatter")
        
        with col2:
            if total_queries > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Hits (Cached)', 'Misses (API Call)'],
                    values=[total_hits, total_misses],
                    marker=dict(colors=['#22c55e', '#ef4444']),
                    textinfo='percent+label',
                    hovertemplate='%{label}: %{value:,}<extra></extra>'
                )])
                fig.update_layout(
                    title="Cache Hit/Miss Distribution",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True, key="cost_roi_by_platform")
            else:
                st.info("No cache queries recorded yet")
        
        st.markdown("---")
        st.markdown("### Cost Savings")
        st.caption("Quantifies the financial benefit of semantic caching. Each cache hit avoids a paid API call, saving both latency and money.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Estimated Savings from Caching:**
            - **Total Cached Responses:** {total_hits:,}
            - **Avg API Call Cost:** ~€0.002 per call
            - **Total Saved:** €{cost_saved:.2f}
            
            Each cache hit avoids an API call, saving both time and money.
            """)
        
        with col2:
            st.info("""
            **Research Plan Reference:** Section 10.2 - "Semantic Cache Hit Rate: 
            Percentage of LLM calls avoided due to a successful cache hit. Target > 20%"
            
            The semantic cache stores embeddings of previous prompts and retrieves 
            similar cached responses instead of making new API calls.
            """)
        
    except Exception as e:
        st.error(f"Failed to load semantic cache metrics: {str(e)}")

# TAB 6: COST BY CAMPAIGN (Gap 4 Fix)
with tab6:
    st.subheader("🎯 Cost per Campaign")
    st.caption("Track AI and operational costs attributed to each campaign. Use this to compare cost efficiency across campaigns and identify optimization opportunities.")

    try:
        campaign_costs_data = api.get_costs_by_campaign(days=30)
        campaigns = api.get_campaigns(limit=100)
        
        campaign_lookup = {str(c.get('id')): c for c in campaigns}

        if campaign_costs_data and campaign_costs_data.get('campaigns'):
            real_campaign_costs = campaign_costs_data.get('campaigns', [])
            total_cost = campaign_costs_data.get('total_cost', 0)
            
            campaign_costs = []
            for camp_cost in real_campaign_costs:
                campaign_id = camp_cost.get('campaign_id')
                campaign_name = camp_cost.get('campaign_name', 'Unknown')
                cost = camp_cost.get('cost', 0)
                
                camp_details = campaign_lookup.get(campaign_id, {})
                impressions = camp_details.get('impressions', 0)
                conversions = camp_details.get('conversions', 0)
                status = camp_details.get('status', 'unknown')
                platform = camp_details.get('platform', 'unknown')
                
                cpl = cost / max(conversions, 1) if conversions > 0 else 0

                campaign_costs.append({
                    'Campaign': campaign_name,
                    'Status': status.upper() if status else 'UNKNOWN',
                    'Platform': platform,
                    'Impressions': impressions,
                    'Conversions': conversions,
                    'AI Cost': f"€{cost:.2f}" if cost >= 1 else f"€{cost:.4f}",
                    'Cost per Lead': (f"€{cpl:.2f}" if cpl >= 1 else f"€{cpl:.4f}") if cpl > 0 else "N/A"
                })

            if campaign_costs:
                col1, col2, col3 = st.columns(3)

                with col1:
                    render_metric_card("Total Campaigns", len(campaign_costs))

                with col2:
                    avg_cost = total_cost / max(len(campaign_costs), 1)
                    render_metric_card("Avg Cost/Campaign", f"€{avg_cost:.2f}" if avg_cost >= 1 else f"€{avg_cost:.4f}")

                with col3:
                    total_conversions = sum(c.get('conversions', 0) for c in campaigns if c.get('status') in ['running', 'completed'])
                    overall_cpl = total_cost / max(total_conversions, 1) if total_conversions > 0 else 0
                    render_metric_card("Overall CPL", f"€{overall_cpl:.2f}" if overall_cpl >= 1 else f"€{overall_cpl:.4f}")

                st.markdown("---")

                st.markdown("### Campaign Cost Breakdown")
                st.caption("Each row shows the AI inference cost attributed to a campaign plus its conversion metrics.")
                cost_df = pd.DataFrame(campaign_costs)
                st.dataframe(cost_df, use_container_width=True, hide_index=True)

                st.markdown("### Cost Distribution by Campaign")
                st.caption("Visual comparison of AI spending across campaigns. Taller bars indicate higher AI costs for that campaign.")

                fig = go.Figure(go.Bar(
                    x=[c['Campaign'] for c in campaign_costs],
                    y=[float(c['AI Cost'].replace('€', '')) for c in campaign_costs],
                    marker_color='#3b82f6',
                    text=[c['AI Cost'] for c in campaign_costs],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="AI Cost by Campaign",
                    xaxis_title="Campaign",
                    yaxis_title="Cost (€)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key="cost_efficiency_trend")

                st.markdown("### Cost Efficiency (Cost per Lead)")
                st.caption("AI cost divided by conversions. The green dashed line marks the €50 CPL target — bars below it are cost-efficient.")

                valid_cpl = [(c['Campaign'], float(c['Cost per Lead'].replace('€', '')))
                             for c in campaign_costs if c['Cost per Lead'] != "N/A"]

                if valid_cpl:
                    fig_cpl = go.Figure(go.Bar(
                        x=[c[0] for c in valid_cpl],
                        y=[c[1] for c in valid_cpl],
                        marker_color=['#22c55e' if c[1] < 50 else '#f59e0b' if c[1] < 100 else '#ef4444' for c in valid_cpl],
                        text=[f"€{c[1]:.2f}" for c in valid_cpl],
                        textposition='outside'
                    ))
                    fig_cpl.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Target: €50")
                    fig_cpl.update_layout(
                        title="Cost per Lead by Campaign",
                        xaxis_title="Campaign",
                        yaxis_title="CPL (€)",
                        height=400
                    )
                    st.plotly_chart(fig_cpl, use_container_width=True, key="cost_cpl_chart")
                else:
                    st.info("No conversions yet to calculate CPL.")

            else:
                st.info("No campaign cost data tracked yet. Run workflows with campaign context to start tracking.")

        else:
            st.info("No campaign cost data tracked yet. Run content generation or safety validation with campaign context.")

    except Exception as e:
        st.error(f"Failed to load campaign costs: {str(e)}")

st.markdown("---")
st.caption(f"Cost Control & Budget Management | Last updated: {datetime.now().strftime('%H:%M:%S')}")

