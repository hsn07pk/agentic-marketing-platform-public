"""
Experiments - A/B Testing and Multi-Armed Bandits
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
    render_metric_card, render_status_badge,
    create_bar_chart, create_line_chart
)

st.set_page_config(page_title="Experiments - Agentic AI", page_icon="🔬", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("🔬 Experiments & A/B Testing")
st.caption("Multi-armed bandits, Thompson Sampling, and LinUCB optimization")

with st.expander("ℹ️ Understanding A/B Testing & Multi-Armed Bandits", expanded=False):
    st.markdown("""
    **What are experiments?**
    Experiments compare content variants (headlines, images, CTAs) to find the best performers using real user data.

    **Algorithms:**
    - **Thompson Sampling** — Bayesian approach using Beta distributions. Automatically balances exploration (trying new variants) vs exploitation (using the best-known variant). Converges to the optimal arm over time.
    - **LinUCB** — Contextual bandit that adapts to user features (device, location, time). Uses upper confidence bounds to select arms based on context, ideal when performance varies across audience segments.
    - **Random** — Uniform random selection for baseline comparison.

    **Statistical Significance:**
    Results require sufficient samples before drawing conclusions. Aim for **100+ samples per variant** for reliable results; the power calculator in the Create tab helps determine exact sample sizes.

    **Experiment Lifecycle:**
    `Created → Running → Completed` — Experiments collect data while running and conclude when the target sample size or confidence threshold is reached.

    **When to use experiments:**
    - Testing new content strategies (headlines, formats, topics)
    - Validating MARL policy changes before full rollout
    - Comparing algorithm performance (Thompson Sampling vs LinUCB)

    **Tips:**
    - Use the power calculator to plan sample sizes before starting
    - Monitor cumulative regret in the Bandit Analytics tab to assess algorithm efficiency
    - Higher confidence thresholds (95%+) reduce false positives but require more samples
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 All Experiments",
    "➕ Create Experiment",
    "📈 Experiment Details",
    "🎯 Bandit Analytics"
])

with tab1:
    st.subheader("All Experiments")
    st.caption("Browse, filter, and simulate traffic for all experiments. Click 'View Details' for in-depth analysis.")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption("Active experiments with variant performance tracking")

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    try:
        experiments = api.list_experiments(limit=100)

        if experiments:
            st.success(f"✅ Found {len(experiments)} experiments")

            active_count = len([e for e in experiments if e.get('is_active', False)])
            completed_count = len([e for e in experiments if not e.get('is_active', True)])

            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric_card("Total Experiments", len(experiments), help_text="Total number of experiments created across all algorithms")
            with col2:
                render_metric_card("Active", active_count, delta=f"+{active_count}" if active_count > 0 else None, help_text="Experiments currently collecting data")
            with col3:
                render_metric_card("Completed", completed_count, help_text="Experiments that reached their target sample size or were manually stopped")

            st.markdown("---")
            
            filtered_experiments = render_data_controls(
                data=experiments,
                search_fields=['name', 'id', 'algorithm', 'campaign_id'],
                filter_configs=[
                    {'field': 'algorithm', 'label': 'Algorithm', 'type': 'select', 'options': 'auto'},
                    {'field': 'is_active', 'label': 'Status', 'type': 'select', 'options': ['True', 'False']},
                ],
                sort_options=['started_at', 'name', 'algorithm'],
                key_prefix="experiments"
            )

            for idx, exp in enumerate(filtered_experiments):
                exp_id = exp.get('id', f'exp_{idx}')
                exp_name = exp.get('name', 'Unnamed Experiment')
                is_active = exp.get('is_active', False)
                status = "ACTIVE" if is_active else "COMPLETED"
                algorithm = exp.get('algorithm', 'ab_test')
                started_at = exp.get('started_at', 'N/A')

                try:
                    arms_data = api.get_bandit_arms(exp_id)
                except:
                    arms_data = []

                total_pulls = sum(arm.get('pulls', 0) for arm in arms_data)
                total_conversions = sum(arm.get('successes', 0) for arm in arms_data)
                overall_ctr = (total_conversions / total_pulls * 100) if total_pulls > 0 else 0

                best_arm = max(arms_data, key=lambda a: a.get('successes', 0) / max(a.get('pulls', 1), 1)) if arms_data else None

                with st.expander(
                    f"**{exp_name}** | Type: {(algorithm or 'unknown').upper()} | Status: {status}",
                    expanded=(idx == 0)
                ):
                    col_info, col_metrics, col_simulate = st.columns([2, 2, 1.5])

                    with col_info:
                        st.markdown("**Experiment Details**")
                        st.write(f"**ID:** `{exp_id}`")
                        st.write(f"**Algorithm:** {algorithm}")
                        st.write(f"**Status:** {'🟢 Active' if is_active else '⚪ Completed'}")
                        st.write(f"**Started:** {started_at[:19] if started_at != 'N/A' else 'N/A'}")
                        
                        if exp.get('is_mock', False):
                            st.warning("🧪 Mock Mode")

                        if arms_data:
                            st.markdown("**Variants:**")
                            for arm in arms_data:
                                arm_ctr = (arm.get('successes', 0) / max(arm.get('pulls', 1), 1)) * 100
                                st.markdown(f"- {arm.get('arm_id', 'Unnamed')}: {arm_ctr:.2f}% CTR")

                    with col_metrics:
                        st.markdown("**Performance**")

                        st.metric("Total Pulls", f"{total_pulls:,}", help="Total arm selections across all variants")
                        st.metric("Total Conversions", f"{total_conversions:,}", help="Successful outcomes (clicks, signups, etc.)")
                        st.metric("Overall CTR", f"{overall_ctr:.2f}%", help="Aggregate conversion rate across all variants")

                        if best_arm:
                            best_ctr = (best_arm.get('successes', 0) / max(best_arm.get('pulls', 1), 1)) * 100
                            st.metric(
                                "Best Variant",
                                best_arm.get('arm_id', 'N/A'),
                                delta=f"{best_ctr:.2f}% CTR",
                                help="Variant with the highest observed conversion rate"
                            )

                    with col_simulate:
                        st.markdown("**Algorithm Testing**")
                        st.caption("🔬 *Simulate traffic to test bandit convergence*")

                        num_pulls = st.number_input(
                            "Pulls to simulate",
                            min_value=10,
                            max_value=1000,
                            value=100,
                            step=10,
                            key=f"pulls_{exp_id}",
                            help="Number of simulated arm selections to generate. Higher values produce more data for convergence analysis."
                        )

                        if st.button("🔬 Simulate Traffic", key=f"sim_{exp_id}", use_container_width=True):
                            try:
                                with st.spinner("Running simulation..."):
                                    result = api.simulate_experiment(exp_id, num_pulls=num_pulls)
                                    if result:
                                        st.toast(f"✅ Simulation complete: {num_pulls} pulls", icon="✅")
                                        st.json(result)
                                        st.rerun()
                                    else:
                                        st.error("Simulation failed")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                        if st.button("📊 View Details", key=f"view_{exp_id}", use_container_width=True):
                            st.session_state['selected_experiment'] = exp_id
                            st.rerun()
        else:
            st.info("📭 No experiments found. Create your first experiment!")

    except Exception as e:
        st.error(f"Failed to load experiments: {str(e)}")
        st.exception(e)

with tab2:
    st.subheader("Create New Experiment")
    st.caption("Configure a new A/B test or bandit experiment. Use the power calculator to determine the right sample size.")

    # A/B TEST POWER CALCULATOR (Gap 24 Fix)
    with st.expander("📊 Sample Size Power Calculator", expanded=False):
        st.caption("Calculate required sample size for statistically significant results")

        col_calc1, col_calc2 = st.columns(2)

        with col_calc1:
            baseline_ctr = st.number_input(
                "Baseline CTR (%)",
                min_value=0.1,
                max_value=50.0,
                value=2.5,
                step=0.1,
                help="Your current expected conversion rate"
            )

            min_detectable_effect = st.number_input(
                "Minimum Detectable Effect (%)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Minimum relative improvement you want to detect (e.g., 20% = 2.5% -> 3.0%)"
            )

        with col_calc2:
            statistical_power = st.selectbox(
                "Statistical Power",
                [80, 85, 90, 95],
                index=0,
                help="Probability of detecting a true effect (80% is standard)"
            )

            significance_level = st.selectbox(
                "Significance Level",
                [0.01, 0.05, 0.10],
                index=1,
                format_func=lambda x: f"{x*100:.0f}% (α = {x})",
                help="Probability of a false positive (Type I error). 5% is the standard threshold."
            )

        if st.button("🧮 Calculate Sample Size"):
            # Power calculation using normal approximation
            import math
            from scipy import stats

            p1 = baseline_ctr / 100
            p2 = p1 * (1 + min_detectable_effect / 100)

            # Z-scores
            z_alpha = stats.norm.ppf(1 - significance_level / 2)
            z_beta = stats.norm.ppf(statistical_power / 100)

            # Pooled variance approximation
            p_avg = (p1 + p2) / 2
            effect_size = abs(p2 - p1)

            if effect_size > 0:
                n_per_variant = int(
                    2 * ((z_alpha + z_beta) ** 2) * p_avg * (1 - p_avg) / (effect_size ** 2)
                )
                total_sample = n_per_variant * 2

                st.success(f"**Required Sample Size:**")
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Per Variant", f"{n_per_variant:,}", help="Required samples for each variant arm")
                with col_r2:
                    st.metric("Total (2 variants)", f"{total_sample:,}", help="Total samples needed across both variants")
                with col_r3:
                    daily_traffic = st.number_input("Daily Traffic", min_value=100, value=1000, key="daily_traffic_calc", help="Estimated daily visitors to calculate experiment duration")
                    days_needed = math.ceil(total_sample / daily_traffic)
                    st.metric("Est. Days", f"{days_needed}", help="Estimated days to reach target sample size at the given daily traffic")

                st.info(f"""
                **Parameters:**
                - Baseline: {baseline_ctr:.2f}% → Target: {p2*100:.2f}%
                - Effect Size: {min_detectable_effect:.1f}% relative lift
                - Power: {statistical_power}%, α = {significance_level}
                """)
            else:
                st.error("Effect size must be greater than 0")

        st.markdown("---")

    with st.form("create_experiment_form"):

        col1, col2 = st.columns(2)

        with col1:
            exp_name = st.text_input(
                "Experiment Name*",
                placeholder="Q1 2025 Headline Test",
                help="A descriptive name to identify this experiment in dashboards and reports."
            )

            exp_type = st.selectbox(
                "Experiment Type",
                ["thompson_sampling", "linucb"],
                format_func=lambda x: {
                    "thompson_sampling": "Thompson Sampling (Bayesian)",
                    "linucb": "LinUCB (Contextual Bandit)"
                }.get(x, x),
                help="Thompson Sampling for automatic exploration-exploitation balance. LinUCB for context-aware optimization using user features."
            )

            campaign_id = st.text_input(
                "Campaign ID (optional)",
                placeholder="Leave empty for standalone experiment",
                help="Link this experiment to a specific campaign. Leave empty for standalone experiments."
            )

        with col2:
            target_samples = st.number_input(
                "Target Sample Size",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Minimum data points per variant before statistical significance can be calculated. Higher = more reliable but slower. Use the power calculator above for guidance."
            )

            # Use selectbox to avoid slider visual inconsistency
            confidence_options = [80, 85, 90, 95, 97, 99]
            confidence_threshold = st.selectbox(
                "Confidence Threshold (%)",
                options=confidence_options,
                index=3,
                help="Statistical confidence required to complete experiment (95% = industry standard)"
            )

            duration_days = st.number_input(
                "Duration (days)",
                min_value=1,
                max_value=90,
                value=14,
                help="Maximum experiment runtime. The experiment may complete earlier if the target sample size and confidence threshold are reached."
            )

        st.markdown("#### Define Variants")
        num_variants = st.number_input("Number of Variants", min_value=2, max_value=10, value=2, help="Number of content variants to compare. 2 is standard for A/B tests; use more for multivariate experiments.")

        variants = []
        for i in range(num_variants):
            col_a, col_b = st.columns(2)

            with col_a:
                variant_name = st.text_input(
                    f"Variant {i+1} Name",
                    value=f"Variant {chr(65+i)}",
                    key=f"variant_name_{i}",
                    help="Unique name for this variant (e.g., 'Control', 'Short Headline', 'Emoji CTA')."
                )

            with col_b:
                variant_desc = st.text_input(
                    f"Description",
                    placeholder="Brief description of this variant",
                    key=f"variant_desc_{i}",
                    help="Describe what makes this variant different from others."
                )

            variants.append({
                "name": variant_name,
                "description": variant_desc
            })

        submit = st.form_submit_button("🚀 Create Experiment", type="primary", use_container_width=True)

        if submit:
            if not exp_name:
                st.error("Experiment name is required")
            else:
                try:
                    experiment_data = {
                        "name": exp_name,
                        "experiment_type": exp_type,
                        "campaign_id": campaign_id if campaign_id else None,
                        "target_sample_size": target_samples,
                        "confidence_threshold": confidence_threshold / 100,
                        "duration": duration_days,
                        "variants": variants
                    }

                    with st.spinner("Creating experiment..."):
                        result = api.create_experiment(experiment_data)

                    if result:
                        st.toast(f"✅ Experiment '{exp_name}' created successfully!", icon="✅")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to create experiment")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab3:
    st.subheader("Experiment Details & Analysis")
    st.caption("Deep-dive into a single experiment: variant performance, confidence intervals, and posterior distributions.")

    selected_exp_id = st.session_state.get('selected_experiment', None)

    if not selected_exp_id:
        exp_id_input = st.text_input("Enter Experiment ID", placeholder="UUID or experiment ID", help="Paste an experiment ID from the All Experiments tab, or click 'View Details' on any experiment.")
        if st.button("Load Experiment"):
            if exp_id_input:
                st.session_state['selected_experiment'] = exp_id_input
                st.rerun()
    else:
        try:
            experiment = api.get_experiment(selected_exp_id)
            if not experiment:
                st.error("Experiment not found")
                st.session_state['selected_experiment'] = None
                st.stop()

            arms_data = api.get_bandit_arms(selected_exp_id)

            st.markdown(f"### {experiment.get('name', 'Unnamed Experiment')}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Info**")
                st.write(f"**ID:** `{experiment.get('id')}`")
                st.write(f"**Algorithm:** {experiment.get('algorithm', 'N/A')}")
                st.write(f"**Status:** {'🟢 Active' if experiment.get('is_active') else '⚪ Completed'}")

            with col2:
                st.markdown("**Progress**")
                total_pulls = sum(arm.get('pulls', 0) for arm in arms_data)
                total_conversions = sum(arm.get('successes', 0) for arm in arms_data)

                st.metric("Total Pulls", f"{total_pulls:,}", help="Total arm selections across all variants in this experiment")
                st.metric("Total Conversions", f"{total_conversions:,}", help="Total successful outcomes across all variants")

            with col3:
                st.markdown("**Performance**")
                overall_ctr = (total_conversions / total_pulls * 100) if total_pulls > 0 else 0
                st.metric("Overall CTR", f"{overall_ctr:.2f}%", help="Aggregate conversion rate across all variants")

                if arms_data:
                    best_arm = max(arms_data, key=lambda a: a.get('successes', 0) / max(a.get('pulls', 1), 1))
                    best_ctr = (best_arm.get('successes', 0) / max(best_arm.get('pulls', 1), 1)) * 100
                    st.metric("Best CTR", f"{best_ctr:.2f}%", help="Highest observed conversion rate among all variants")

            st.markdown("---")
            st.markdown("### Variant Performance")

            if arms_data:
                variant_data = []
                for arm in arms_data:
                    pulls = arm.get('pulls', 0)
                    successes = arm.get('successes', 0)
                    ctr = (successes / pulls * 100) if pulls > 0 else 0

                    variant_data.append({
                        'Variant': arm.get('arm_id', 'Unnamed'),
                        'Pulls': pulls,
                        'Conversions': successes,
                        'CTR': f"{ctr:.2f}%",
                        'Alpha': round(arm.get('alpha', 1.0), 1),
                        'Beta': round(arm.get('beta', 1.0), 1)
                    })

                df = pd.DataFrame(variant_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # CONFIDENCE INTERVAL VISUALIZATION (Gap 25 Fix)
                st.markdown("### Conversion Rate with Confidence Intervals")

                fig = go.Figure()

                variant_names = [v['Variant'] for v in variant_data]
                ctrs = [float(v['CTR'].rstrip('%')) for v in variant_data]

                # Calculate confidence intervals using Wilson score interval
                ci_lower = []
                ci_upper = []
                for arm in arms_data:
                    pulls = arm.get('pulls', 0)
                    successes = arm.get('successes', 0)

                    if pulls > 0:
                        p = successes / pulls
                        z = 1.96  # 95% confidence

                        # Wilson score interval
                        denominator = 1 + z**2 / pulls
                        center = (p + z**2 / (2 * pulls)) / denominator
                        margin = z * ((p * (1 - p) / pulls + z**2 / (4 * pulls**2)) ** 0.5) / denominator

                        ci_lower.append(max(0, (center - margin) * 100))
                        ci_upper.append(min(100, (center + margin) * 100))
                    else:
                        ci_lower.append(0)
                        ci_upper.append(0)

                error_minus = [ctrs[i] - ci_lower[i] for i in range(len(ctrs))]
                error_plus = [ci_upper[i] - ctrs[i] for i in range(len(ctrs))]

                fig.add_trace(go.Bar(
                    name='CTR',
                    x=variant_names,
                    y=ctrs,
                    text=[f"{cr:.2f}%" for cr in ctrs],
                    textposition='outside',
                    marker_color=['#3b82f6' if i == ctrs.index(max(ctrs)) else '#94a3b8' for i in range(len(ctrs))],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=error_plus,
                        arrayminus=error_minus,
                        color='#1f2937',
                        thickness=2,
                        width=10
                    )
                ))

                fig.update_layout(
                    title="Conversion Rate by Variant (95% CI)",
                    yaxis_title="CTR (%)",
                    showlegend=False,
                    height=450
                )

                st.plotly_chart(fig, use_container_width=True, key="exp_arm_performance")

                with st.expander("📊 Confidence Interval Details"):
                    ci_df = pd.DataFrame({
                        'Variant': variant_names,
                        'CTR': [f"{c:.2f}%" for c in ctrs],
                        '95% CI Lower': [f"{c:.2f}%" for c in ci_lower],
                        '95% CI Upper': [f"{c:.2f}%" for c in ci_upper],
                        'CI Width': [f"{ci_upper[i] - ci_lower[i]:.2f}%" for i in range(len(ctrs))]
                    })
                    st.dataframe(ci_df, use_container_width=True, hide_index=True)


                # Thompson Sampling posterior visualization
                if experiment.get('algorithm') == 'thompson_sampling' and arms_data:
                    st.markdown("### Thompson Sampling Posterior Distributions")
                    st.caption("Beta distributions showing belief about each variant's true conversion rate")

                    try:
                        import numpy as np
                        from scipy.stats import beta as beta_dist

                        has_data = any(arm.get('pulls', 0) > 0 for arm in arms_data)

                        if has_data:
                            try:
                                import numpy as np
                                from scipy.stats import beta as beta_dist
                                import plotly.graph_objects as go

                                x = np.linspace(0.001, 0.20, 500)

                                traces = []
                                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

                                for idx, arm in enumerate(arms_data):
                                    alpha = arm.get('alpha', 1.0)
                                    beta_param = arm.get('beta', 1.0)

                                    if alpha > 1 or beta_param > 1:
                                        y = beta_dist.pdf(x, alpha, beta_param)

                                        pulls = arm.get('pulls', 0)
                                        successes = arm.get('successes', 0)
                                        observed_ctr = (successes / pulls * 100) if pulls > 0 else 0

                                        trace = go.Scatter(
                                            x=(x * 100).tolist(),
                                            y=y.tolist(),
                                            name=f"{arm.get('arm_id')} (Obs: {observed_ctr:.2f}%)",
                                            mode='lines',
                                            fill='tozeroy',
                                            line=dict(width=2, color=colors[idx % len(colors)]),
                                            opacity=0.7
                                        )
                                        traces.append(trace)

                                fig = go.Figure(data=traces)

                                fig.update_layout(
                                    title={
                                        'text': "Thompson Sampling Posterior Distributions",
                                        'x': 0.5,
                                        'xanchor': 'center'
                                    },
                                    xaxis_title="Conversion Rate (%)",
                                    yaxis_title="Probability Density",
                                    height=500,
                                    showlegend=True,
                                    hovermode='x unified',
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01
                                    )
                                )

                                st.plotly_chart(fig, use_container_width=True, key=f"thompson_{selected_exp_id}")

                                st.caption("💡 **Interpretation**: Higher/narrower peaks indicate more confidence. The variant with the rightmost peak is currently believed to have the highest conversion rate.")

                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                        else:
                            st.info("📊 Posterior distributions will appear after collecting some samples. Run traffic simulation to get started!")
                    except Exception as e:
                        st.error(f"Error creating Thompson Sampling visualization: {str(e)}")

            else:
                st.info("No variant data available yet. Run some traffic simulations!")

            if st.button("← Back to List"):
                st.session_state['selected_experiment'] = None
                st.rerun()

        except Exception as e:
            st.error(f"Error loading experiment: {str(e)}")
            st.exception(e)

with tab4:
    st.subheader("🎯 Multi-Armed Bandit Analytics")
    st.caption("Track algorithm convergence, cumulative regret, and compare Thompson Sampling vs LinUCB performance across experiments.")

    st.markdown("""
    **Bandit Algorithms (Research Plan Section 2.3):**
    - **Thompson Sampling**: Bayesian approach with Beta(α, β) posterior
    - **LinUCB**: Linear contextual bandits with upper confidence bounds
    - **Exploration-Exploitation**: Optimal balance between trying new variants and exploiting best performers
    """)

    # CUMULATIVE REGRET SECTION (Gap 2 Fix)
    st.markdown("---")
    st.markdown("### 📉 Cumulative Regret Analysis")
    st.caption("Regret measures the cost of exploration - lower is better")

    try:
        experiments = api.list_experiments(limit=100)
        active_experiments = [e for e in experiments if e.get('is_active', False)] if experiments else []

        if active_experiments:
            exp_options = {e.get('name', f"Exp {e.get('id')}"): e.get('id') for e in active_experiments}
            selected_exp_name = st.selectbox("Select Experiment", list(exp_options.keys()), key="regret_exp_select", help="Choose an active experiment to analyze its cumulative regret and arm performance.")
            selected_exp_id = exp_options.get(selected_exp_name)

            if selected_exp_id:
                try:
                    arms_data = api.get_bandit_arms(selected_exp_id)

                    if arms_data:
                        arm_ctrs = []
                        for arm in arms_data:
                            pulls = arm.get('pulls', 0)
                            successes = arm.get('successes', 0)
                            ctr = successes / pulls if pulls > 0 else 0
                            arm_ctrs.append({'arm': arm.get('arm_id'), 'ctr': ctr, 'pulls': pulls, 'successes': successes})

                        optimal_arm = max(arm_ctrs, key=lambda x: x['ctr'])
                        optimal_ctr = optimal_arm['ctr']

                        total_pulls = sum(a['pulls'] for a in arm_ctrs)
                        total_successes = sum(a['successes'] for a in arm_ctrs)
                        expected_if_optimal = optimal_ctr * total_pulls
                        cumulative_regret = expected_if_optimal - total_successes

                        col_regret, col_optimal = st.columns(2)

                        with col_regret:
                            fig_regret = go.Figure(go.Indicator(
                                mode="number+delta",
                                value=round(cumulative_regret, 2),
                                title={'text': "Cumulative Regret"},
                                delta={'reference': 0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                                number={'font': {'size': 48}}
                            ))
                            fig_regret.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                            st.plotly_chart(fig_regret, use_container_width=True, key="exp_cumulative_regret")
                            st.caption("Regret = Expected optimal rewards - Actual rewards")

                        with col_optimal:
                            st.markdown("### 🏆 Optimal Arm")
                            st.success(f"**{optimal_arm['arm']}**")
                            st.metric("CTR", f"{optimal_ctr*100:.2f}%", help="Conversion rate of the best-performing arm")
                            st.metric("Pulls", optimal_arm['pulls'], help="Number of times this arm was selected")

                        st.markdown("#### Regret by Variant")
                        regret_data = []
                        for arm in arm_ctrs:
                            arm_regret = (optimal_ctr - arm['ctr']) * arm['pulls']
                            regret_data.append({
                                'Variant': arm['arm'],
                                'CTR': f"{arm['ctr']*100:.2f}%",
                                'Pulls': arm['pulls'],
                                'Regret Contribution': round(arm_regret, 2)
                            })

                        regret_df = pd.DataFrame(regret_data)
                        st.dataframe(regret_df, use_container_width=True, hide_index=True)

                        fig_regret_bar = go.Figure(go.Bar(
                            x=[r['Variant'] for r in regret_data],
                            y=[r['Regret Contribution'] for r in regret_data],
                            marker_color=['#ef4444' if r['Regret Contribution'] > 0 else '#22c55e' for r in regret_data],
                            text=[f"{r['Regret Contribution']:.1f}" for r in regret_data],
                            textposition='outside'
                        ))
                        fig_regret_bar.update_layout(
                            title="Regret Contribution by Variant",
                            xaxis_title="Variant",
                            yaxis_title="Regret",
                            height=300
                        )
                        st.plotly_chart(fig_regret_bar, use_container_width=True, key="exp_regret_bar")

                    else:
                        st.info("No arm data available yet. Run traffic simulation to see regret analysis.")
                except Exception as e:
                    st.warning(f"Could not calculate regret: {e}")
        else:
            st.info("No active experiments. Create an experiment to see regret analysis.")

    except Exception as e:
        st.error(f"Failed to load regret data: {e}")

    st.markdown("---")


    try:
        experiments = api.list_experiments(limit=100)

        if experiments:
            active_experiments = [e for e in experiments if e.get('is_active', False)]

            st.success(f"✅ Loaded {len(active_experiments)} active experiments")

            thompson_exps = [e for e in active_experiments if e.get('algorithm') == 'thompson_sampling']
            linucb_exps = [e for e in active_experiments if e.get('algorithm') == 'linucb']

            col1, col2, col3 = st.columns(3)

            with col1:
                render_metric_card("Active Experiments", len(active_experiments), help_text="Number of experiments currently running and collecting data")

            with col2:
                render_metric_card("Thompson Sampling", len(thompson_exps), help_text="Experiments using Bayesian Thompson Sampling algorithm")

            with col3:
                render_metric_card("LinUCB", len(linucb_exps), help_text="Experiments using Linear Upper Confidence Bound contextual bandits")

            st.markdown("---")

            st.markdown("#### All Active Experiments")

            exp_table_data = []
            for e in active_experiments:
                exp_id = e.get('id')

                try:
                    arms_data = api.get_bandit_arms(exp_id)
                except:
                    arms_data = []

                total_pulls = sum(arm.get('pulls', 0) for arm in arms_data)
                total_conversions = sum(arm.get('successes', 0) for arm in arms_data)
                ctr = (total_conversions / total_pulls * 100) if total_pulls > 0 else 0

                exp_table_data.append({
                    'Name': e.get('name', 'N/A'),
                    'Algorithm': e.get('algorithm', 'N/A'),
                    'Variants': len(arms_data),
                    'Total Pulls': total_pulls,
                    'Conversions': total_conversions,
                    'CTR': f"{ctr:.2f}%"
                })

            if exp_table_data:
                df = pd.DataFrame(exp_table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

            if thompson_exps and linucb_exps:
                st.markdown("---")
                st.markdown("#### Algorithm Performance Comparison")

                def get_algorithm_stats(exps):
                    total_pulls = 0
                    total_conversions = 0

                    for e in exps:
                        try:
                            arms_data = api.get_bandit_arms(e['id'])
                            total_pulls += sum(arm.get('pulls', 0) for arm in arms_data)
                            total_conversions += sum(arm.get('successes', 0) for arm in arms_data)
                        except:
                            pass

                    ctr = (total_conversions / total_pulls * 100) if total_pulls > 0 else 0
                    return total_pulls, total_conversions, ctr

                ts_pulls, ts_conv, ts_ctr = get_algorithm_stats(thompson_exps)
                lu_pulls, lu_conv, lu_ctr = get_algorithm_stats(linucb_exps)

                comparison_df = pd.DataFrame({
                    'Algorithm': ['Thompson Sampling', 'LinUCB'],
                    'Experiments': [len(thompson_exps), len(linucb_exps)],
                    'Total Pulls': [ts_pulls, lu_pulls],
                    'Conversions': [ts_conv, lu_conv],
                    'CTR': [f"{ts_ctr:.2f}%", f"{lu_ctr:.2f}%"]
                })

                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        else:
            st.info("📭 No active experiments found")

    except Exception as e:
        st.error(f"Failed to load bandit analytics: {str(e)}")
        st.exception(e)

st.markdown("---")
st.caption(f"Experiments & A/B Testing | Last updated: {datetime.now().strftime('%H:%M:%S')}")
