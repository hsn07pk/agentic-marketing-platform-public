"""
Agent Episodic Memory Dashboard - REAL DATA ONLY
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls

st.set_page_config(
    page_title="Agent Memory - Agentic AI",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("🧠 Agent Episodic Memory")
st.caption("Agent learning history and self-improvement tracking")

with st.expander("ℹ️ Understanding Agent Memory & Self-Improvement", expanded=False):
    st.markdown("""
    **What is Episodic Memory?**
    Each AI agent remembers past tasks — content generation, safety checks, deployments — as discrete memory episodes.
    After every task, the outcome (success/failure), duration, and context are stored for future reference.

    **How Self-Improvement Works**
    Before starting a new task, agents query their memory for similar past experiences.
    This lets them avoid repeating past mistakes and reuse strategies that worked well.

    **Key Concepts**
    - **Success Rate** — Percentage of tasks completed without errors. A healthy agent targets **>80%**.
    - **Failure Patterns** — Recurring error categories that reveal systematic issues (e.g., API timeouts, content policy violations).
    - **Agent Leaderboard** — Ranks agents by performance so you can identify which ones need optimization or retraining.

    **How to Use This Page**
    1. Select an agent to review its memory history and KPIs.
    2. Use **Query Memory** to search for similar past experiences by describing a task or scenario.
    3. Review **Failure Patterns** to identify bottlenecks and recurring errors.
    4. Check **Analytics** for performance trends over time.
    5. Compare agents in the **Leaderboard** to spot under-performers.
    """)

agents = api.list_agents_with_memory()

if not agents:
    st.warning("No agents with episodic memory found")
    st.stop()

selected_agent = st.selectbox("Select Agent", agents, index=0, help="Choose an agent to view its memory history. Each agent handles specific tasks.")

if selected_agent:
    stats = api.get_agent_memory_stats(selected_agent)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Memories", stats.get("total_memories", 0), help="Total number of task episodes stored in this agent's memory.")
    with col2:
        st.metric("Success Count", stats.get("success_count", 0), help="Number of tasks this agent completed successfully without errors.")
    with col3:
        st.metric("Failure Count", stats.get("failure_count", 0), help="Number of tasks that resulted in errors or did not meet quality thresholds.")
    with col4:
        st.metric("Success Rate", f"{stats.get('success_rate', 0)*100:.1f}%", help="Percentage of tasks completed without errors. Target: >80%.")
    with col5:
        st.metric("Avg Duration", f"{stats.get('avg_duration', 0):.2f}s", help="Average time this agent takes to complete a task, in seconds.")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Recent Memories",
        "🔍 Query Memory",
        "❌ Failure Patterns",
        "📊 Analytics",
        "🏆 Agent Leaderboard"
    ])

    
    with tab1:
        st.subheader(f"Recent Memories - {selected_agent}")
        st.caption("Browse the latest task episodes stored in this agent's memory, including actions taken and outcomes.")
        
        memories = api.get_recent_memories(selected_agent, limit=20)

        if memories:
            for memory_idx, memory in enumerate(memories):
                task_desc = memory.get('task_description') or 'Unknown task'
                if task_desc == 'None content for None persona' or 'None' in str(task_desc):
                    content_type = memory.get('content_type') or memory.get('metrics', {}).get('content_type') or 'content'
                    persona = memory.get('persona') or memory.get('target_persona') or memory.get('metrics', {}).get('persona')
                    campaign = memory.get('campaign_name') or memory.get('campaign_id', '') if memory.get('campaign_id') else None

                    if persona and persona != 'None':
                        task_desc = f"Generate {content_type} for {persona}"
                    elif campaign:
                        task_desc = f"Generate {content_type} (Campaign: {campaign})"
                    else:
                        task_desc = f"Generate {content_type}"

                outcome = memory.get('outcome', 'unknown')
                with st.expander(
                    f"**{task_desc}** | Outcome: {outcome}"
                ):
                    col_left, col_right = st.columns([2, 1])

                    with col_left:
                        st.markdown("#### Task Description")
                        st.write(task_desc if task_desc != 'Unknown task' else 'No description available')
                        
                        st.markdown("#### Actions Taken")
                        actions = memory.get('actions_taken', [])
                        if actions:
                            for idx, action in enumerate(actions):
                                if action.startswith("Body (") and " chars): " in action:
                                    parts = action.split(" chars): ", 1)
                                    char_info = parts[0] + " chars)"
                                    body_content = parts[1] if len(parts) > 1 else "N/A"
                                    st.markdown(f"**{char_info}**")
                                    st.text_area("Body Content", body_content, height=150, disabled=True, 
                                                key=f"body_mem{memory_idx}_action{idx}")
                                else:
                                    st.write(f"- {action}")
                        else:
                            st.write("No actions recorded")
                        
                        if memory.get('human_feedback'):
                            st.markdown("#### Human Feedback")
                            st.info(memory['human_feedback'])
                    
                    with col_right:
                        st.markdown("#### Metrics")
                        metrics = memory.get('metrics', {})
                        for key, value in metrics.items():
                            if isinstance(value, float):
                                if 'duration' in key or 'time' in key or 'latency' in key:
                                    display_value = f"{value:.1f}s"
                                elif 'score' in key or 'rate' in key or 'ratio' in key:
                                    display_value = f"{value:.2f}"
                                elif 'cost' in key or 'price' in key or 'budget' in key:
                                    display_value = f"€{value:.2f}"
                                elif 'percent' in key or 'pct' in key:
                                    display_value = f"{value:.1f}%"
                                else:
                                    display_value = f"{value:.2f}"
                            else:
                                display_value = value
                            st.metric(key.replace('_', ' ').title(), display_value)
                        
                        st.markdown("#### Outcome")
                        outcome = memory.get('outcome', 'unknown')
                        if outcome == 'success':
                            st.success("✅ Success")
                        elif outcome == 'failure':
                            st.error("❌ Failure")
                        else:
                            st.info(f"ℹ️ {outcome}")
                        
                        if memory.get('timestamp'):
                            st.caption(f"Time: {memory['timestamp']}")
        else:
            st.info(f"No memories found for {selected_agent}")
    
    with tab2:
        st.subheader("Query Agent Memory")
        st.caption("Search the agent's past experiences by describing a task or scenario to find similar memories.")
        
        query = st.text_area(
            "Enter a task description to find similar past experiences",
            placeholder="Generate LinkedIn post for decision makers about AI automation",
            help="Describe a task or scenario to find similar past experiences. The agent uses semantic similarity to match memories."
        )
        
        k = st.slider("Number of results", min_value=1, max_value=20, value=5, help="How many matching memories to retrieve. More results give broader context but may include less relevant matches.")
        
        if st.button("🔍 Search Memory", type="primary"):
            if query:
                with st.spinner(f"Querying {selected_agent}'s memory..."):
                    results = api.query_agent_memory(selected_agent, query, k=k)
                
                if results:
                    st.success(f"Found {len(results)} relevant memories")
                    
                    for idx, result in enumerate(results):
                        task_desc = result.get('task_description', 'Unknown')
                        with st.expander(f"Result {idx+1}: {task_desc}"):
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                st.markdown("#### Task")
                                st.write(result.get('task_description', 'No description'))
                                
                                st.markdown("#### Actions")
                                for action in result.get('actions_taken', []):
                                    st.write(f"- {action}")
                                
                                if result.get('lessons_learned'):
                                    st.markdown("#### Lessons Learned")
                                    st.write(result['lessons_learned'])
                            
                            with col_b:
                                st.metric("Similarity Score", f"{result.get('similarity_score', 0):.2f}", help="How closely this memory matches your query (0–1). Higher is more relevant.")
                                st.metric("Outcome", result.get('outcome', 'unknown'), help="Whether this past task succeeded or failed.")
                                
                                metrics = result.get('metrics', {})
                                for key, value in metrics.items():
                                    if isinstance(value, float):
                                        if 'duration' in key or 'time' in key or 'latency' in key:
                                            display_value = f"{value:.1f}s"
                                        elif 'score' in key or 'rate' in key or 'ratio' in key:
                                            display_value = f"{value:.2f}"
                                        elif 'cost' in key or 'price' in key or 'budget' in key:
                                            display_value = f"€{value:.2f}"
                                        elif 'percent' in key or 'pct' in key:
                                            display_value = f"{value:.1f}%"
                                        else:
                                            display_value = f"{value:.2f}"
                                    else:
                                        display_value = value
                                    st.metric(key.replace('_', ' ').title(), display_value)
                else:
                    st.warning("No relevant memories found")
            else:
                st.warning("Please enter a query")
    
    with tab3:
        st.subheader("Common Failure Patterns")
        st.caption("Recurring error categories that reveal systematic issues. Use these to identify and fix bottlenecks.")
        
        failures = api.get_failure_patterns(selected_agent, limit=10)
        
        if failures:
            for failure in failures:
                with st.expander(f"❌ {failure.get('pattern', 'Unknown pattern')}"):
                    st.metric("Occurrences", failure.get('count', 0), help="Number of times this failure pattern has been observed.")
                    st.write(failure.get('description', 'No description'))
                    
                    if failure.get('examples'):
                        st.markdown("#### Examples")
                        for example in failure['examples']:
                            st.code(example)
        else:
            st.success("✅ No failure patterns detected! Agent is performing well.")
    
    with tab4:
        st.subheader("Memory Analytics")
        st.caption("Visualize outcome distribution and performance trends over time to track agent improvement.")
        
        memories = api.get_recent_memories(selected_agent, limit=100)
        
        if memories:
            df = pd.DataFrame(memories)
            
            if 'outcome' in df.columns:
                outcome_counts = df['outcome'].value_counts()
                
                import plotly.graph_objects as go
                
                fig_outcomes = go.Figure(data=[go.Pie(
                    labels=list(outcome_counts.index),
                    values=list(outcome_counts.values),
                    textinfo='percent+label',
                    marker=dict(colors=['#10b981' if o == 'success' else '#ef4444' for o in outcome_counts.index])
                )])
                fig_outcomes.update_layout(title="Outcome Distribution")
                st.plotly_chart(fig_outcomes, use_container_width=True, key="mem_outcomes")
            
            st.markdown("### Performance Trends")
            
            if len(df) >= 3 and 'timestamp' in df.columns:
                df['parsed_time'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df_sorted = df.dropna(subset=['parsed_time']).sort_values('parsed_time')
                
                if len(df_sorted) >= 3:
                    df_sorted['success_num'] = (df_sorted['outcome'] == 'success').astype(int)
                    df_sorted['rolling_success'] = df_sorted['success_num'].rolling(window=min(5, len(df_sorted)), min_periods=1).mean() * 100
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=df_sorted['parsed_time'],
                        y=df_sorted['rolling_success'],
                        mode='lines+markers',
                        name='Success Rate %',
                        line=dict(color='#10b981', width=2)
                    ))
                    fig_trend.update_layout(
                        title="Rolling Success Rate Over Time",
                        xaxis_title="Time",
                        yaxis_title="Success Rate (%)",
                        yaxis=dict(range=[0, 105])
                    )
                    st.plotly_chart(fig_trend, use_container_width=True, key="mem_trend")
            
            if 'metrics' in df.columns:
                safety_scores = []
                for m in df['metrics']:
                    if isinstance(m, dict) and 'safety_score' in m:
                        safety_scores.append(m['safety_score'])
                
                if safety_scores:
                    avg_safety = sum(safety_scores) / len(safety_scores)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Safety Score", f"{avg_safety:.2%}", help="Average safety compliance score across all analyzed memories. Higher is safer.")
                    with col2:
                        st.metric("Memories Analyzed", len(df), help="Total number of memory episodes included in this analytics view.")
            
            if len(df) < 3:
                st.info("📊 Generate more content to see performance trends over time")
        else:
            st.info("Not enough data for analytics")

    with tab5:
        st.subheader("🏆 Agent Performance Leaderboard")
        st.caption("Cross-agent performance comparison — identify top performers and agents that need optimization.")

        leaderboard_data = []
        for agent in agents:
            stats = api.get_agent_memory_stats(agent)
            leaderboard_data.append({
                'Agent': agent,
                'Total Memories': stats.get('total_memories', 0),
                'Successes': stats.get('success_count', 0),
                'Failures': stats.get('failure_count', 0),
                'Success Rate': stats.get('success_rate', 0) * 100,
                'Avg Duration': round(stats.get('avg_duration', 0), 2)
            })

        if leaderboard_data:
            leaderboard_df = pd.DataFrame(leaderboard_data)
            leaderboard_df = leaderboard_df.sort_values('Success Rate', ascending=False).reset_index(drop=True)

            st.markdown("### 🥇 Top Performers")

            import plotly.graph_objects as go

            fig = go.Figure()
            agents = leaderboard_df['Agent'].tolist()
            success_rates = leaderboard_df['Success Rate'].tolist()
            colors = ['#ffd700' if i == 0 else '#c0c0c0' if i == 1 else '#cd7f32' if i == 2 else '#3b82f6'
                      for i in range(len(agents))]

            fig.add_trace(go.Bar(
                x=agents,
                y=success_rates,
                marker_color=colors,
                text=[f"{r:.1f}%" for r in success_rates],
                textposition='outside'
            ))

            fig.update_layout(
                title="Agent Success Rate Ranking",
                xaxis_title="Agent",
                yaxis_title="Success Rate (%)",
                yaxis=dict(range=[0, 105]),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True, key="mem_network")

            st.markdown("### Detailed Rankings")
            display_df = leaderboard_df.copy()
            display_df['Success Rate'] = display_df['Success Rate'].apply(lambda x: f"{x:.1f}%")
            display_df['Avg Duration'] = display_df['Avg Duration'].apply(lambda x: f"{x:.2f}s")

            display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
            display_df['Rank'] = display_df['Rank'].apply(lambda x: f"🥇 {x}" if x == 1 else f"🥈 {x}" if x == 2 else f"🥉 {x}" if x == 3 else str(x))

            st.dataframe(display_df, use_container_width=True, hide_index=True)

        else:
            st.info("No agent data available for leaderboard")

st.markdown("---")
st.caption("Agent Episodic Memory - Enabling continuous improvement through experience")

