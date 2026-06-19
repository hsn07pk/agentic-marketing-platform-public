"""
Market Intelligence & Strategy Optimization - REAL DATA ONLY
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls, render_searchable_select
from utils.metrics_utils import normalize_ctr

st.set_page_config(page_title="Market Intelligence - Agentic AI", page_icon="🔍", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("🔍 Market Intelligence & Strategy Optimization")
st.caption("Observe market trends, scrape competitor content, and optimize campaign strategies")

with st.expander("ℹ️ Understanding Market Intelligence", expanded=False):
    st.markdown("""
**Market Intelligence** monitors competitor content, analyzes market trends, and optimizes your content strategy using AI.

**🔎 Market Scraper** — Collects content from HR tech industry sources and competitor blogs for analysis. Uses the APIFY API for LinkedIn/Twitter when configured; otherwise scrapes public web sources.

**📊 Strategy Optimizer** — AI analyzes scraped content and campaign performance to suggest content themes, optimal posting times, and engagement tactics tailored to your audience.

**📈 Strategy Performance** — Tracks how AI-optimized strategies perform versus baseline metrics over time, so you can measure real improvement.

**🏢 Competitive Intelligence** — Provides comprehensive competitor landscape analysis gathered during each campaign's market observation phase, including strengths, weaknesses, messaging themes, and differentiation opportunities.

**Data Sources:**
- **Web Scraping:** Public HR tech blogs and competitor sites (Personnel Today, HRD, Lattice, Culture Amp, Visier, 15Five)
- **APIFY API:** Real LinkedIn/Twitter engagement data (likes, shares, comments) when configured
- **Sources are configurable** via Operations → System Settings → apify category → MARKET_SCRAPE_SOURCES

**Workflow:** Scrape industry & competitor content → Analyze patterns → Feed insights into content generation → Measure results
""")


_apify_configured = api.is_config_key_set('APIFY_API_TOKEN')

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Market Scraper", "📊 Strategy Optimizer", "📈 Strategy Performance", "🏢 Competitive Intelligence"])

with tab1:
    st.subheader("Market Intelligence Scraper")
    st.caption("Scrape and analyze industry & competitor content")
    
    with st.form("scraper_form"):
        keywords = st.text_area(
            "Keywords (one per line)",
            placeholder="employee experience\npeople analytics\nQWL\nworkplace wellbeing",
            height=100,
            help="Enter industry terms, competitor names, or topics to scrape. One per line."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("Number of posts to scrape", 5, 50, 20, help="Posts to collect per scrape. Sources are configurable in Operations → System Settings → MARKET_SCRAPE_SOURCES.")
        with col2:
            platform_display = st.selectbox("Platform", ["LinkedIn", "Twitter", "Blog", "All"], index=0, help="LinkedIn uses APIFY if configured. 'All' scrapes configured web sources.")

        submitted = st.form_submit_button("🔍 Scrape Content", type="primary")

        if submitted:
            if not keywords:
                st.error("Please enter at least one keyword")
            else:
                keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                platform = platform_display.lower() if platform_display != "All" else "all"

                with st.spinner(f"Scraping {platform_display} content for {len(keyword_list)} keyword(s)..."):
                    result = api.scrape_content(keyword_list, limit=limit, platform=platform)
                
                if result and result.get('status') != 'error':
                    posts = result.get('posts', [])
                    st.session_state['scraped_posts'] = posts
                    st.session_state['scrape_insights'] = result.get('insights')
                    st.session_state['scrape_data_source'] = result.get('data_source', 'unknown')
                    
                    if result.get('data_source') == 'web_scrape':
                        st.info("ℹ️ **Web Scrape Mode**: Content scraped from public industry blogs and competitor sites. Engagement metrics require APIFY API key for LinkedIn/Twitter data.")
                    
                    if posts:
                        st.success(f"✅ Scraped {len(posts)} posts!")
                    else:
                        st.warning("⚠️ No posts found for these keywords. Try broader terms or check source connectivity.")
                else:
                    st.error("Failed to scrape content")
                    st.session_state['scraped_posts'] = []
    
    if 'scraped_posts' in st.session_state and st.session_state['scraped_posts']:
        posts = st.session_state['scraped_posts']
        data_source = st.session_state.get('scrape_data_source', 'unknown')
        is_web_scrape = data_source == 'web_scrape'
        
        st.markdown("### Scraped Content")
        
        for idx, post in enumerate(posts):
            with st.expander(f"Post {idx+1}: {post.get('platform', 'Unknown').upper()} - {post.get('author', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Content")
                    st.write(post.get('text', 'No content'))
                    
                    if post.get('url'):
                        st.markdown(f"**[View Original]({post['url']})**")
                    
                    if post.get('keywords_matched'):
                        st.caption(f"🏷️ Matched: {', '.join(post['keywords_matched'])}")
                
                with col2:
                    st.markdown("#### Metrics")
                    if is_web_scrape or not _apify_configured:
                        st.caption("ℹ️ *Engagement metrics require APIFY API key.*")
                        st.caption("*Configure `APIFY_API_TOKEN` in ⚙️ Operations → System Settings → apify category.*")
                    else:
                        st.metric("Likes", post.get('likes', 0), help="Number of likes/reactions on this post.")
                        st.metric("Shares", post.get('shares', 0), help="Number of times this post was shared or reposted.")
                        st.metric("Comments", post.get('comments', 0), help="Number of comments on this post.")
                        
                        engagement = post.get('likes', 0) + post.get('shares', 0) + post.get('comments', 0)
                        st.metric("Total Engagement", engagement, help="Sum of likes, shares, and comments for this post.")
        
        st.markdown("---")
        if st.button("📊 Analyze Posts", type="primary"):
            with st.spinner("Analyzing content..."):
                analysis = api.analyze_posts(posts)
            
            if analysis and analysis.get('status') != 'error':
                st.success("✅ Analysis complete!")
                
                st.markdown("### Analysis Results")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Posts", analysis.get('total_analyzed', len(posts)), help="Total number of scraped posts included in this analysis.")
                
                with col_b:
                    if is_web_scrape or not _apify_configured:
                        pass
                    else:
                        avg_engagement = sum(p.get('likes',0)+p.get('shares',0)+p.get('comments',0) for p in posts) / len(posts) if posts else 0
                        st.metric("Avg Engagement", f"{avg_engagement:.0f}", help="Average total engagement (likes + shares + comments) per post.")
                
                with col_c:
                    top_platform = max(set(p.get('platform','unknown') for p in posts), key=lambda x: sum(1 for p in posts if p.get('platform')==x)) if posts else 'N/A'
                    st.metric("Top Platform", top_platform.upper(), help="The platform with the most scraped posts in this batch.")
                
                if analysis.get('top_hooks'):
                    st.markdown("### Top Performing Hooks")
                    for hook in analysis['top_hooks'][:5]:
                        st.info(f"📈 {hook.get('text', hook)}")
                
                if analysis.get('top_ctas'):
                    st.markdown("### Top CTAs")
                    df_ctas = pd.DataFrame(analysis['top_ctas'])
                    st.dataframe(df_ctas, use_container_width=True, hide_index=True)
                
                insights = analysis.get('insights', [])
                if insights:
                    st.markdown("### Insights")
                    for insight in insights:
                        st.info(f"💡 {insight}")
                
                topics = analysis.get('trending_topics', [])
                if topics:
                    st.markdown("### Trending Topics")
                    for topic in topics:
                        st.write(f"- 🔥 {topic}")
                
                has_data = analysis.get('top_hooks') or analysis.get('top_ctas') or insights or topics
                if not has_data:
                    if is_web_scrape or not _apify_configured:
                        st.warning("⚠️ Analysis returned limited results. For richer insights (hooks, CTAs, engagement patterns), configure your `APIFY_API_TOKEN` in ⚙️ Operations → System Settings → apify category.")
                    else:
                        st.info("ℹ️ No detailed patterns found. Try scraping more posts or different keywords.")
            else:
                st.error("Failed to analyze posts")


with tab2:
    st.subheader("Strategy Optimizer")
    st.caption("Get AI-powered recommendations for campaign optimization based on market data and current performance")
    
    campaigns = api.get_campaigns(limit=500)
    
    if campaigns:
        selected_campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="strategy_campaign",
            placeholder="Search by name, ID, platform..."
        )
        
        campaign_data = next((c for c in campaigns if c['id'] == selected_campaign_id), None) if selected_campaign_id else None
        
        if campaign_data:
            st.markdown("### Current Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # CTR is already in percentage from API
            ctr_value = normalize_ctr(campaign_data.get('ctr', 0))
            with col1:
                st.metric("CTR", f"{ctr_value:.2f}%", help="Current click-through rate for this campaign. Percentage of impressions that resulted in clicks.")
            with col2:
                st.metric("Conversions", campaign_data.get('conversions', 0), help="Total number of conversions (sign-ups, purchases, etc.) generated by this campaign.")
            with col3:
                st.metric("CPL", f"€{campaign_data.get('cpl', 0):.2f}", help="Cost per lead — average cost to acquire one conversion.")
            with col4:
                st.metric("Spend", f"€{campaign_data.get('budget_spent', 0):.2f}", help="Total budget spent so far on this campaign.")
            
            st.markdown("---")
            
            with st.form("optimize_form"):
                st.markdown("### Optimization Parameters")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    target_ctr = st.number_input(
                        "Target CTR (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=max(ctr_value * 1.5, 2.0),
                        step=0.1,
                        help="Desired click-through rate the optimizer should aim for. Defaults to 1.5× current CTR."
                    )
                    
                    max_budget = st.number_input(
                        "Max Daily Budget (€)",
                        min_value=10,
                        max_value=10000,
                        value=max(10, int(campaign_data.get('budget_total', 100) * 0.1)),
                        step=10,
                        help="Maximum amount to spend per day. The optimizer will allocate within this budget."
                    )
                
                with col_b:
                    target_conversions = st.number_input(
                        "Target Conversions/Day",
                        min_value=1,
                        max_value=1000,
                        value=max(campaign_data.get('conversions', 5) * 2, 5),
                        step=1,
                        help="Desired number of daily conversions. Defaults to 2× current conversion count."
                    )
                    
                    optimization_goal = st.selectbox(
                        "Primary Goal",
                        ["maximize_conversions", "minimize_cpl", "maximize_ctr", "maximize_roi"],
                        help="What the AI should prioritize: more conversions, lower cost per lead, higher click rate, or best ROI."
                    )
                
                optimize_clicked = st.form_submit_button("🚀 Optimize Strategy", type="primary")
                
                if optimize_clicked:
                    strategy_request = {
                        "campaign_id": str(selected_campaign_id),
                        "platform": campaign_data.get('platform', 'linkedin'),
                        "target_persona": campaign_data.get('target_persona', 'decision_maker'),
                        "budget": float(max_budget),  # Required field
                        "context": {
                            "current_performance": {
                                "ctr": campaign_data.get('ctr', 0),
                                "conversions": campaign_data.get('conversions', 0),
                                "cpl": campaign_data.get('cpl', 0)
                            },
                            "constraints": {
                                "target_ctr": target_ctr,
                                "target_conversions": target_conversions
                            },
                            "optimization_goal": optimization_goal
                        }
                    }
                    
                    with st.spinner("Generating optimal strategy..."):
                        optimization = api.get_optimal_strategy(strategy_request)
                    
                    if optimization and optimization.get('status') != 'error' and optimization.get('strategy_name'):
                        st.success("✅ Optimization complete!")
                        
                        st.markdown("### Recommended Strategy")
                        
                        st.markdown(f"**Strategy:** {optimization.get('strategy_name', 'N/A')}")
                        st.markdown(f"**Hook:** {optimization.get('hook', 'N/A')}")
                        st.markdown(f"**CTA:** {optimization.get('cta', 'N/A')}")
                        st.markdown(f"**Confidence:** {optimization.get('confidence', 0)*100:.1f}%")
                        
                        st.markdown("---")
                        
                        if optimization.get('estimated_performance'):
                            st.markdown("### Predicted Performance")
                            
                            pred = optimization['estimated_performance']
                            # Predicted CTR comes as decimal from optimizer, convert to percentage
                            pred_ctr_pct = pred.get('ctr', 0) * 100
                            
                            col_x, col_y, col_z = st.columns(3)
                            
                            with col_x:
                                st.metric(
                                    "Predicted CTR",
                                    f"{pred_ctr_pct:.2f}%",
                                    delta=f"+{(pred_ctr_pct - ctr_value):.2f}%",
                                    help="AI-predicted click-through rate after applying the optimized strategy."
                                )
                            
                            with col_y:
                                st.metric(
                                    "Predicted Conv Rate",
                                    f"{pred.get('conversion_rate', 0)*100:.2f}%",
                                    help="AI-predicted conversion rate (percentage of clicks that convert)."
                                )
                            
                            with col_z:
                                st.metric(
                                    "Predicted CPL",
                                    f"€{pred.get('cpl', 0):.2f}",
                                    delta_color="inverse",
                                    help="AI-predicted cost per lead after optimization. Lower is better."
                                )
                        
                        if optimization.get('budget_allocation'):
                            st.markdown("### Budget Allocation")
                            alloc = optimization['budget_allocation']
                            for key, value in alloc.items():
                                st.write(f"• **{key.replace('_', ' ').title()}**: €{value:.2f}")
                        
                        if optimization.get('rationale'):
                            st.info(f"💡 {optimization['rationale']}")
                    else:
                        st.error("Failed to generate optimization")
    else:
        st.info("No campaigns available. Create a campaign first.")

with tab3:
    st.subheader("Strategy Performance Tracking")
    st.caption("Compare AI-optimized strategies against baseline performance over time")
    
    campaigns = api.get_campaigns(limit=500)
    
    if campaigns:
        selected_campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign to Track",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="perf_campaign",
            placeholder="Search by name, ID, platform..."
        )
        
        if selected_campaign_id:
            performance = api.get_strategy_performance(selected_campaign_id)
        
        if performance and performance.get('status') != 'error':
            st.markdown("### Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Strategies Tested", performance.get('total_strategies', 0), help="Total number of different strategies tested for this campaign.")
            
            with col2:
                st.metric("Best CTR", f"{performance.get('best_ctr', 0):.2f}%", help="Highest click-through rate achieved across all tested strategies.")
            
            with col3:
                st.metric("Best Conversions", performance.get('best_conversions', 0), help="Highest number of conversions achieved by any single strategy.")
            
            with col4:
                st.metric("Lowest CPL", f"€{performance.get('lowest_cpl', 0):.2f}", help="Lowest cost per lead achieved. Lower is better.")
            
            with col5:
                st.metric("Success Rate", f"{performance.get('success_rate', 0)*100:.1f}%", help="Percentage of tested strategies that outperformed the baseline.")
            
            if performance.get('strategy_history'):
                st.markdown("### Strategy History")
                
                df = pd.DataFrame(performance['strategy_history'])
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                if 'ctr' in df.columns and len(df) > 1:
                    fig = px.line(df, x='timestamp' if 'timestamp' in df.columns else df.index, 
                                y='ctr', title="CTR Over Time", markers=True)
                    st.plotly_chart(fig, use_container_width=True, key="mi_sentiment_chart")
        else:
            st.info("No performance data available for this campaign")
    else:
        st.info("No campaigns available")

st.markdown("---")
st.caption("Market Intelligence & Strategy Optimization")

with tab4:
    st.subheader("🏢 Campaign Market Intelligence")
    st.caption("View detailed competitive analysis gathered during each campaign's market observation (OBSERVE) phase")

    campaigns = api.get_campaigns(limit=500)
    if campaigns:
        selected_campaign_id = render_searchable_select(
            items=campaigns,
            display_field='name',
            id_field='id',
            label="Select Campaign",
            search_fields=['name', 'id', 'platform', 'target_persona', 'status'],
            key_prefix="mi_campaign",
            placeholder="Search by name, ID, platform..."
        )

        mi_campaign_data = next((c for c in campaigns if c['id'] == selected_campaign_id), None) if selected_campaign_id else None

        if mi_campaign_data:
            events = api.get_campaign_events(str(selected_campaign_id), limit=200)
            market_events = [
                e for e in (events or [])
                if e.get('workflow_node') == 'market_observation' and e.get('details')
            ]

            completion_event = None
            for e in market_events:
                d = e.get('details', {})
                if d.get('competitors_analyzed') or d.get('competitor_details'):
                    completion_event = e
                    break

            competitor_profiles_resp = api.list_competitors()
            competitor_profiles = competitor_profiles_resp.get('competitors', []) if isinstance(competitor_profiles_resp, dict) else []
            comp_profile_map = {cp.get('name', '').lower(): cp for cp in competitor_profiles}

            if completion_event:
                details = completion_event['details']
                has_rich_data = bool(details.get('competitor_details'))

                st.success(completion_event.get('message', 'Market observation completed.'))
                ctx_cols = st.columns(4)
                with ctx_cols[0]:
                    st.caption(f"**Campaign:** {mi_campaign_data.get('name', 'N/A')}")
                with ctx_cols[1]:
                    st.caption(f"**Platform:** {mi_campaign_data.get('platform', 'N/A').upper()}")
                with ctx_cols[2]:
                    st.caption(f"**Persona:** {mi_campaign_data.get('target_persona', 'N/A')}")
                with ctx_cols[3]:
                    ts = completion_event.get('created_at', '')
                    st.caption(f"**Observed:** {ts[:19].replace('T', ' ') if ts else 'N/A'}")

                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Competitors Analyzed", details.get('competitors_analyzed', 0), help="Number of competitors examined during the market observation phase.")
                with col2:
                    diff_raw = details.get('differentiation_opportunities', [])
                    if isinstance(diff_raw, int):
                        diff_count = diff_raw
                    else:
                        diff_count = details.get('differentiation_opportunities_count', len(diff_raw) if isinstance(diff_raw, list) else 0)
                    st.metric("Differentiation Opps", diff_count, help="Unique positioning opportunities identified where you can stand out from competitors.")
                with col3:
                    st.metric("Duration", f"{details.get('duration', 0):.1f}s", help="Time taken to complete the market observation analysis.")
                with col4:
                    st.metric("Scraping Enabled", "Yes" if details.get('scraping_enabled') else "No", help="Whether live web scraping was used to gather competitor data.")
                with col5:
                    content_patterns = details.get('content_patterns', {})
                    st.metric("Posts Analyzed", content_patterns.get('total_analyzed', 0) if content_patterns else 0, help="Number of competitor social media posts analyzed for content patterns.")

                st.markdown("---")

                diff_opps = details.get('differentiation_opportunities', [])
                if diff_opps and isinstance(diff_opps, list) and len(diff_opps) > 0 and isinstance(diff_opps[0], str):
                    st.markdown("### 💡 Differentiation Opportunities")
                    st.caption("Unique advantages identified against competitors during market observation")
                    for i, opp in enumerate(diff_opps, 1):
                        st.markdown(f"**{i}.** {opp}")

                positions = details.get('market_positions', {})
                if positions:
                    st.markdown("### 📊 Market Positions")
                    pos_df = pd.DataFrame([
                        {"Position": pos, "Count": count}
                        for pos, count in positions.items()
                    ])
                    col_chart, col_table = st.columns([2, 1])
                    with col_chart:
                        fig = px.pie(pos_df, names="Position", values="Count", title="Competitor Market Positions")
                        st.plotly_chart(fig, use_container_width=True, key="mi_trend_chart")
                    with col_table:
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)

                comp_details = details.get('competitor_details', [])

                # Fallback: if event has competitors_analyzed > 0 but no rich details, build from profiles
                if not comp_details and details.get('competitors_analyzed', 0) > 0 and competitor_profiles:
                    st.info("ℹ️ This campaign ran before rich event details were stored. Showing competitor profiles from the current database.")
                    comp_details = []
                    for cp in competitor_profiles:
                        key_features = cp.get('key_features', '')
                        features_list = [f.strip() for f in key_features.split(';') if f.strip()] if isinstance(key_features, str) else (key_features or [])
                        risky = cp.get('risky_topics', '')
                        risky_list = [r.strip() for r in risky.split(';') if r.strip()] if isinstance(risky, str) else (risky or [])
                        comp_details.append({
                            "name": cp.get('name', 'Unknown'),
                            "position": cp.get('category', 'N/A'),
                            "strengths_count": len(features_list),
                            "weaknesses_count": len(risky_list),
                            "messaging_themes": [],
                            "latest_messaging": {},
                        })
                    # Also build fallback positions from categories
                    if not positions:
                        cat_counts = {}
                        for cp in competitor_profiles:
                            cat = cp.get('category', 'Unknown')
                            cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        if cat_counts:
                            st.markdown("### 📊 Market Positions")
                            pos_df = pd.DataFrame([{"Position": p, "Count": c} for p, c in cat_counts.items()])
                            col_chart, col_table = st.columns([2, 1])
                            with col_chart:
                                fig = px.pie(pos_df, names="Position", values="Count", title="Competitor Market Positions")
                                st.plotly_chart(fig, use_container_width=True, key="mi_topic_chart")
                            with col_table:
                                st.dataframe(pos_df, use_container_width=True, hide_index=True)

                if comp_details:
                    st.markdown("### 🏢 Competitor Breakdown")
                    st.caption(f"{len(comp_details)} competitors analyzed during this campaign's OBSERVE phase")

                    comp_summary = pd.DataFrame([
                        {
                            "Competitor": c.get('name', 'Unknown'),
                            "Position": c.get('position', 'N/A'),
                            "Strengths": c.get('strengths_count', 0),
                            "Weaknesses": c.get('weaknesses_count', 0),
                            "Messaging Themes": ', '.join(c.get('messaging_themes', [])[:3]) or 'N/A'
                        }
                        for c in comp_details
                    ])
                    st.dataframe(comp_summary, use_container_width=True, hide_index=True)

                    chart_data = pd.DataFrame([
                        {"Competitor": c.get('name', 'Unknown'), "Strengths": c.get('strengths_count', 0), "Weaknesses": c.get('weaknesses_count', 0)}
                        for c in comp_details
                    ])
                    if len(chart_data) > 0:
                        chart_melted = chart_data.melt(id_vars="Competitor", value_vars=["Strengths", "Weaknesses"],
                                                       var_name="Type", value_name="Count")
                        fig_comp = px.bar(chart_melted, x="Competitor", y="Count", color="Type", barmode="group",
                                          title="Competitor Strengths vs Weaknesses", color_discrete_map={"Strengths": "#2ecc71", "Weaknesses": "#e74c3c"})
                        fig_comp.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_comp, use_container_width=True, key="mi_competitor_chart")

                    for comp in comp_details:
                        comp_name = comp.get('name', 'Unknown')
                        profile = comp_profile_map.get(comp_name.lower(), {})
                        with st.expander(f"**{comp_name}** — {comp.get('position', 'N/A')}"):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Strengths", comp.get('strengths_count', 0), help="Number of identified competitive strengths for this competitor.")
                            with c2:
                                st.metric("Weaknesses", comp.get('weaknesses_count', 0), help="Number of identified competitive weaknesses or vulnerabilities.")
                            with c3:
                                website = profile.get('url', profile.get('website', ''))
                                if website:
                                    st.markdown(f"🌐 [{website}]({website})")
                                else:
                                    st.caption("No website")

                            key_features = profile.get('key_features', '')
                            if key_features:
                                features_list = [f.strip() for f in key_features.split(';') if f.strip()] if isinstance(key_features, str) else key_features
                                if features_list:
                                    st.markdown("**🔑 Key Features:**")
                                    for feat in features_list:
                                        st.markdown(f"- {feat}")

                            typical_claims = profile.get('typical_claims', '')
                            if typical_claims:
                                claims_list = [c.strip() for c in typical_claims.split(';') if c.strip()] if isinstance(typical_claims, str) else typical_claims
                                if claims_list:
                                    st.markdown("**📢 Typical Claims:**")
                                    for claim in claims_list:
                                        st.markdown(f"- {claim}")

                            differentiators = profile.get('differentiators_vs_us', '')
                            if differentiators:
                                st.markdown("**⚡ Our Differentiation:**")
                                st.info(differentiators)

                            risky = profile.get('risky_topics', '')
                            if risky:
                                st.markdown("**⚠️ Risky Topics (avoid in messaging):**")
                                st.warning(risky)

                            themes = comp.get('messaging_themes', [])
                            if themes:
                                st.markdown("**💬 Messaging Themes:**")
                                for theme in themes:
                                    st.markdown(f"- {theme}")

                            latest = comp.get('latest_messaging', {})
                            if latest and latest.get('success'):
                                st.markdown("**🕸️ Latest Scraped Website Messaging:**")
                                ls_col1, ls_col2 = st.columns(2)
                                with ls_col1:
                                    scraped_themes = latest.get('messaging_themes', [])
                                    if scraped_themes:
                                        st.markdown("**Headlines found:**")
                                        for t in scraped_themes[:10]:
                                            st.markdown(f"- {t}")
                                with ls_col2:
                                    meta_desc = latest.get('meta_description', '')
                                    if meta_desc:
                                        st.markdown(f"**Meta description:** {meta_desc}")
                                    scraped_kw = latest.get('keywords', [])
                                    if scraped_kw:
                                        st.markdown(f"**Keywords:** {', '.join(scraped_kw[:15])}")
                                    scraped_at = latest.get('scraped_at', '')
                                    if scraped_at:
                                        st.caption(f"Scraped at: {scraped_at[:19]}")

                strengths = details.get('top_common_strengths', [])
                weaknesses = details.get('top_common_weaknesses', [])
                agg_strengths = details.get('aggregated_strengths', [])
                agg_weaknesses = details.get('aggregated_weaknesses', [])

                if strengths or weaknesses or agg_strengths or agg_weaknesses:
                    st.markdown("### 📋 Aggregated Strengths & Weaknesses")
                    st.caption("Cross-competitor analysis of common capabilities and vulnerabilities")
                    s_col, w_col = st.columns(2)
                    with s_col:
                        st.markdown("**Top Strengths**")
                        if strengths:
                            str_df = pd.DataFrame([
                                {"Strength": s.get('strength', s) if isinstance(s, dict) else s,
                                 "Count": s.get('count', 0) if isinstance(s, dict) else 0}
                                for s in strengths
                            ])
                            fig_s = px.bar(str_df, x="Count", y="Strength", orientation='h',
                                           color_discrete_sequence=["#2ecc71"])
                            fig_s.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=max(250, len(strengths) * 35))
                            st.plotly_chart(fig_s, use_container_width=True, key="mi_strengths_chart")
                        elif agg_strengths:
                            for s in agg_strengths[:10]:
                                st.markdown(f"- {s}")
                        else:
                            st.caption("No data")
                    with w_col:
                        st.markdown("**Top Weaknesses**")
                        if weaknesses:
                            weak_df = pd.DataFrame([
                                {"Weakness": w.get('weakness', w) if isinstance(w, dict) else w,
                                 "Count": w.get('count', 0) if isinstance(w, dict) else 0}
                                for w in weaknesses
                            ])
                            fig_w = px.bar(weak_df, x="Count", y="Weakness", orientation='h',
                                           color_discrete_sequence=["#e74c3c"])
                            fig_w.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=max(250, len(weaknesses) * 35))
                            st.plotly_chart(fig_w, use_container_width=True, key="mi_weaknesses_chart")
                        elif agg_weaknesses:
                            for w in agg_weaknesses[:10]:
                                st.markdown(f"- {w}")
                        else:
                            st.caption("No data")

                    if agg_strengths or agg_weaknesses:
                        with st.expander("📄 Full Strengths & Weaknesses Lists"):
                            fl_s, fl_w = st.columns(2)
                            with fl_s:
                                st.markdown("**All Strengths**")
                                for s in agg_strengths:
                                    st.markdown(f"- {s}")
                            with fl_w:
                                st.markdown("**All Weaknesses**")
                                for w in agg_weaknesses:
                                    st.markdown(f"- {w}")

                content_patterns = details.get('content_patterns', {})
                if content_patterns:
                    st.markdown("---")
                    st.markdown("### 📝 Content Patterns from Market Scraping")
                    st.caption("Hooks, CTAs, themes, and engagement patterns discovered from scraped market content")

                    # Engagement patterns overview (only with APIFY — web scrape has no engagement data)
                    engagement = content_patterns.get('engagement_patterns', {})
                    has_engagement = engagement and engagement.get('avg_total_engagement', 0) > 0
                    if has_engagement and _apify_configured:
                        e1, e2, e3, e4 = st.columns(4)
                        with e1:
                            st.metric("Avg Likes", f"{engagement.get('avg_likes', 0):.0f}", help="Average number of likes per post across scraped competitor content.")
                        with e2:
                            st.metric("Avg Shares", f"{engagement.get('avg_shares', 0):.0f}", help="Average number of shares per post across scraped competitor content.")
                        with e3:
                            st.metric("Avg Comments", f"{engagement.get('avg_comments', 0):.0f}", help="Average number of comments per post across scraped competitor content.")
                        with e4:
                            st.metric("Avg Total Engagement", f"{engagement.get('avg_total_engagement', 0):.0f}", help="Average total engagement (likes + shares + comments) per post.")
                    elif not _apify_configured:
                        st.info("💡 **Engagement metrics (likes, shares, comments) require APIFY integration.** Configure `APIFY_API_TOKEN` in ⚙️ Operations → System Settings → apify category to enable social media engagement data.")

                    cp_col1, cp_col2 = st.columns(2)

                    with cp_col1:
                        hooks = content_patterns.get('top_hooks', [])
                        if hooks:
                            st.markdown("#### 🪝 Top Performing Hooks")
                            for idx, hook in enumerate(hooks[:10], 1):
                                hook_text = hook.get('text', hook) if isinstance(hook, dict) else hook
                                hook_eng = hook.get('engagement', 0) if isinstance(hook, dict) else 0
                                is_engaging = hook.get('is_engaging', False) if isinstance(hook, dict) else False
                                badge = " 🔥" if is_engaging else ""
                                eng_label = f" — engagement: {hook_eng}" if hook_eng and _apify_configured else ""
                                st.markdown(f"**{idx}.** {hook_text}{eng_label}{badge}")

                    with cp_col2:
                        ctas = content_patterns.get('top_ctas', [])
                        if ctas:
                            st.markdown("#### 🎯 Top CTAs")
                            cta_df = pd.DataFrame([
                                {"CTA": c.get('cta', c) if isinstance(c, dict) else c,
                                 "Frequency": c.get('count', 0) if isinstance(c, dict) else 0}
                                for c in ctas
                            ])
                            if len(cta_df) > 0 and cta_df['Frequency'].sum() > 0:
                                fig_cta = px.bar(cta_df, x="Frequency", y="CTA", orientation='h',
                                                  color_discrete_sequence=["#3498db"])
                                fig_cta.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False,
                                                       height=max(200, len(ctas) * 35))
                                st.plotly_chart(fig_cta, use_container_width=True, key="mi_cta_chart")
                            else:
                                st.dataframe(cta_df, use_container_width=True, hide_index=True)

                    themes = content_patterns.get('common_themes', [])
                    if themes:
                        st.markdown("#### 🏷️ Common Themes")
                        theme_df = pd.DataFrame([
                            {"Theme": t.get('theme', t) if isinstance(t, dict) else t,
                             "Frequency": t.get('count', 0) if isinstance(t, dict) else 0}
                            for t in themes
                        ])
                        if len(theme_df) > 0 and theme_df['Frequency'].sum() > 0:
                            fig_theme = px.bar(theme_df, x="Theme", y="Frequency",
                                               color_discrete_sequence=["#9b59b6"])
                            fig_theme.update_layout(xaxis_tickangle=-45, showlegend=False)
                            st.plotly_chart(fig_theme, use_container_width=True, key="mi_theme_chart")
                        else:
                            st.dataframe(theme_df, use_container_width=True, hide_index=True)

                    platforms = content_patterns.get('platforms_analyzed', [])
                    if platforms:
                        st.caption(f"Platforms analyzed: {', '.join(p.upper() for p in platforms)}")

                    total_analyzed = content_patterns.get('total_analyzed', 0)
                    if total_analyzed:
                        st.caption(f"Total posts analyzed: {total_analyzed}")

                if len(market_events) > 1:
                    st.markdown("---")
                    st.markdown("### 🕐 Market Observation Event Timeline")
                    for evt in market_events:
                        evt_type = evt.get('event_type', 'unknown')
                        evt_title = evt.get('title', evt_type)
                        evt_ts = evt.get('created_at', '')[:19].replace('T', ' ')
                        evt_sev = evt.get('severity', 'info')
                        icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(evt_sev, "📌")
                        st.markdown(f"{icon} **{evt_title}** — {evt_ts}")
                        if evt.get('message'):
                            st.caption(evt['message'])

                with st.expander("📄 Raw Market Intelligence Data"):
                    st.json(details)
            else:
                st.info("No detailed market intelligence data available for this campaign. Run a new campaign to collect data — newer campaigns store full competitor analysis details.")
    else:
        st.info("No campaigns available. Create a campaign first.")
