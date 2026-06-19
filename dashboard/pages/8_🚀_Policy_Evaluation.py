"""
Policy Evaluation & MARL Gating Dashboard - REAL DATA ONLY
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from utils.data_controls import render_data_controls

st.set_page_config(
    page_title="Policy Evaluation - Agentic AI",
    page_icon="🚀",
    layout="wide"
)

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("🚀 Policy Evaluation & MARL Gating")
st.caption("Offline policy evaluation, MARL promotion gating, and canary validation")

with st.expander("ℹ️ Understanding Policy Evaluation & MARL Gating", expanded=False):
    st.markdown("""
    **Off-Policy Evaluation (OPE)** lets you evaluate new AI policies using historical campaign data
    *without* deploying them live — protecting your campaigns from untested changes.

    **How MARL Gating works:**
    - A new policy must demonstrate **>10% estimated lift** over the 25th-percentile baseline to be promoted.
    - **Baseline** = 25th percentile of historical campaign rewards (represents unoptimized performance).
    - **Policy value** = estimated reward calculated via **Inverse Propensity Scoring (IPS)**, weighted by action probabilities.
    - **Lift calculation:** `lift = ((new_policy_value − baseline_value) / baseline_value) × 100`
    - **Statistical significance:** 95% confidence intervals ensure improvements aren't due to random chance.

    **Promotion flow:**
    1. **Train Policy** → 2. **OPE Evaluation** → 3. **MARL Gate** (>10% lift?) → 4. **Canary Deployment** (5% traffic) → 5. **Full Rollout**

    *Why this matters:* deploying an under-performing policy can waste budget and hurt campaign KPIs.
    OPE + MARL gating adds a safety net so only proven policies reach production.
    """)

ope_status = api.get_ope_status()

if not ope_status.get("ope_available", False):
    st.error("❌ OPE system not available. Check backend configuration.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Policy Evaluation",
    "🔬 MARL Promotion Gate",
    "📜 Promotion History",
    "🆚 Policy Comparison"
])

with tab1:
    st.subheader("Offline Policy Evaluation")
    st.caption("Run OPE on a candidate policy against a baseline using logged campaign interactions.")
    
    with st.form("ope_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_id = st.text_input(
                "Campaign ID",
                placeholder="Enter campaign ID for evaluation",
                help="Select the campaign whose data will be used for policy evaluation."
            )
            
            policy_name = st.selectbox(
                "Policy to Evaluate",
                ["linucb", "thompson_sampling", "marl_agent"],
                index=0,
                help="The candidate policy to evaluate against the baseline using historical data."
            )
        
        with col2:
            baseline_policy = st.selectbox(
                "Baseline Policy",
                ["thompson_sampling", "linucb", "random"],
                index=0,
                help="The existing production policy used as the performance baseline for comparison."
            )
            
            min_samples = st.number_input(
                "Minimum Samples Required",
                min_value=1,
                max_value=10000,
                value=1000,
                step=100,
                help="Minimum data points needed for reliable evaluation. More samples = more statistical confidence."
            )
        
        submitted = st.form_submit_button("🔍 Evaluate Policy", type="primary")
        
        if submitted:
            if not campaign_id:
                st.error("Campaign ID is required")
            else:
                with st.spinner("Running offline policy evaluation..."):
                    request = {
                        "campaign_id": campaign_id,
                        "policy_name": policy_name,
                        "baseline_policy_name": baseline_policy,
                        "min_samples": min_samples
                    }
                    
                    result = api.evaluate_policy(request)
                
                if result:
                    status = result.get('status', 'unknown')
                    
                    if status == 'completed':
                        st.success("✅ Evaluation completed successfully!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.markdown("### Baseline")
                            baseline = result.get('baseline', {})
                            st.metric("Value", f"{baseline.get('value', 0):.3f}",
                                     help="Estimated reward of the current baseline policy via IPS weighting.")
                            st.caption(f"CI: [{baseline.get('ci_lower', 0):.3f}, {baseline.get('ci_upper', 0):.3f}]")
                        
                        with col_b:
                            st.markdown("### New Policy")
                            new_policy = result.get('new_policy', {})
                            st.metric("Value", f"{new_policy.get('value', 0):.3f}",
                                     help="Estimated reward of the candidate policy via IPS weighting on historical data.")
                            st.caption(f"CI: [{new_policy.get('ci_lower', 0):.3f}, {new_policy.get('ci_upper', 0):.3f}]")
                        
                        with col_c:
                            st.markdown("### Comparison")
                            comparison = result.get('comparison', {})
                            lift = comparison.get('lift_percent', 0)
                            
                            if lift >= 0:
                                st.metric("Lift", f"+{lift:.2f}%", delta=f"{lift:.2f}%",
                                          help="Percentage improvement of the new policy over baseline: ((new − baseline) / baseline) × 100.")
                            else:
                                st.metric("Lift", f"{lift:.2f}%", delta=f"{lift:.2f}%", delta_color="inverse",
                                          help="Percentage improvement of the new policy over baseline: ((new − baseline) / baseline) × 100.")
                            
                            if comparison.get('statistically_significant'):
                                st.success("✅ Statistically significant")
                            else:
                                st.warning("⚠️ Not statistically significant")
                        
                        st.markdown("---")
                        recommendation = comparison.get('recommendation', 'keep_baseline')
                        gate_passed = comparison.get('gate_passed', False)
                        
                        if recommendation in ['promote', 'APPROVE_CANARY'] or gate_passed:
                            st.success(f"✅ **RECOMMENDATION**: Promote {policy_name} to canary deployment (5% traffic)")
                        else:
                            st.info(f"ℹ️ **RECOMMENDATION**: Keep current baseline policy")
                        
                        st.caption(f"Samples used: {result.get('samples_used', 0)}")
                    
                    elif status == 'insufficient_data':
                        st.warning(result.get('message', 'Insufficient data'))
                        st.metric("Samples Collected", result.get('samples_collected', 0),
                                 help="Number of logged interactions collected so far for this campaign.")
                        st.metric("Samples Needed", result.get('samples_needed', 0),
                                 help="Total samples required before OPE can produce a reliable estimate.")
                        st.info(result.get('recommendation', 'Continue collecting data'))
                    else:
                        st.error(f"Evaluation failed: {result.get('message', 'Unknown error')}")

with tab2:
    st.subheader("MARL Promotion Gating")
    st.caption("Evaluate a MARL policy against OPE gates — sample size, lift threshold, and confidence — before allowing production deployment.")
    
    with st.form("marl_promotion_form"):
        policy_name = st.text_input(
            "MARL Policy Name",
            placeholder="e.g., marl_policy_v1",
            help="Name of the MARL policy to evaluate against historical baseline."
        )
        
        description = st.text_area(
            "Policy Description (optional)",
            placeholder="Describe the MARL policy being evaluated",
            help="Optional notes about this policy version — e.g., training parameters or changes from previous version."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_samples_gate = st.number_input(
                "Minimum Samples Required",
                min_value=1,
                max_value=10000,
                value=1000,
                step=100,
                help="Minimum data points needed for reliable evaluation. More samples = more statistical confidence."
            )
        with col2:
            min_lift = st.slider(
                "Minimum Lift Threshold",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                format="%.2f",
                help="Minimum improvement (%) required over baseline. Higher = safer but harder to pass."
            )
        
        st.markdown(f"""
        **Promotion Criteria:**
        1. ✅ Sample Size Gate: ≥{min_samples_gate} samples
        2. ✅ Lift Gate: ≥{int(min_lift*100)}% improvement over baseline
        3. ✅ Confidence Gate: Lower CI > baseline value
        """)
        
        submitted = st.form_submit_button("🚀 Evaluate for Promotion", type="primary")
        
        if submitted:
            if not policy_name:
                st.error("Policy name is required")
            else:
                with st.spinner("Evaluating MARL policy for promotion... (this may take a minute)"):
                    request = {
                        "policy_name": policy_name,
                        "description": description,
                        "min_lift_threshold": min_lift,
                        "min_samples_required": min_samples_gate
                    }
                    
                    result = api.evaluate_marl_promotion(request)
                
                if result:
                    status = result.get('status', 'unknown')
                    
                    if status == 'evaluated':
                        promotion_approved = result.get('promotion_approved', False)
                        
                        if promotion_approved:
                            st.success(f"🎉 **PROMOTION APPROVED** for {policy_name}!")
                            st.balloons()
                        else:
                            st.error(f"❌ **PROMOTION REJECTED** for {policy_name}")
                        
                        st.markdown("### Gate Results")
                        
                        gates = result.get('gates', [])
                        
                        for gate in gates:
                            gate_name = gate.get('name', 'Unknown')
                            gate_status = gate.get('status', 'unknown')
                            requirement = gate.get('requirement', '')
                            
                            if gate_status == 'passed':
                                st.success(f"✅ **{gate_name}**: PASSED - {requirement}")
                            else:
                                st.error(f"❌ **{gate_name}**: FAILED - {requirement}")
                            
                            if 'actual' in gate:
                                st.caption(f"Actual: {gate['actual']}")
                            if 'actual_lift' in gate:
                                st.caption(f"Actual lift: {gate['actual_lift']:.2f}%")
                        
                        st.markdown("### Performance Metrics")
                        
                        metrics = result.get('metrics', {})
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Baseline Value", f"{metrics.get('baseline_value', 0):.3f}",
                                     help="25th-percentile historical reward representing unoptimized performance.")
                        with col_b:
                            st.metric("New Policy Value", f"{metrics.get('new_policy_value', 0):.3f}",
                                     help="IPS-estimated reward of the candidate MARL policy on historical data.")
                        with col_c:
                            st.metric("Lift", f"{metrics.get('lift_percent', 0):.2f}%",
                                     help="Percentage lift of new policy over baseline: ((new − baseline) / baseline) × 100.")
                        
                        ci = metrics.get('confidence_interval', {})
                        st.caption(f"95% CI: [{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]")
                        
                        st.markdown("### Next Steps")
                        
                        next_steps = result.get('next_steps', [])
                        for step in next_steps:
                            st.write(f"- {step}")
                        
                        st.caption(f"Samples used: {result.get('samples_used', 0)}")
                    
                    elif status == 'insufficient_data':
                        st.warning(result.get('message', 'Insufficient data'))
                        gates = result.get('gates', [])
                        for gate in gates:
                            st.write(f"- {gate.get('name')}: {gate.get('status')}")
                    else:
                        st.error(f"Evaluation failed: {result.get('message', 'Unknown error')}")

with tab3:
    st.subheader("MARL Promotion History")
    st.caption("Chronological log of all MARL promotion gate evaluations — approved and rejected.")
    
    history = api.get_marl_promotion_history(limit=20)
    
    if history:
        for entry in history:
            exp_name = entry.get('experiment_name', entry.get('policy_name', 'Unknown'))
            campaign_name = entry.get('campaign_name', 'Unknown Campaign')
            is_approved = entry.get('approved', False)
            timestamp = entry.get('ended_at', entry.get('timestamp', 'N/A'))
            
            with st.expander(
                f"**{exp_name}** ({campaign_name}) | "
                f"{'✅ APPROVED' if is_approved else '❌ REJECTED'} | "
                f"{timestamp}"
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Algorithm:** {entry.get('algorithm', 'thompson_sampling')}")
                    st.write(f"**Campaign:** {campaign_name}")
                    st.write(f"**Started:** {entry.get('started_at', 'N/A')}")
                    st.write(f"**Completed:** {entry.get('ended_at', 'N/A')}")
                    
                    impressions = entry.get('total_impressions', 0)
                    conversions = entry.get('total_conversions', 0)
                    ctr = entry.get('conversion_rate', 0)
                    
                    st.write(f"**Impressions:** {impressions:,}")
                    st.write(f"**Conversions:** {conversions:,}")
                    st.write(f"**CTR:** {ctr:.2f}%")
                
                with col2:
                    if is_approved:
                        st.success("✅ Approved for Deployment")
                    else:
                        st.error("❌ Rejected")
                    
                    st.metric("CTR", f"{entry.get('conversion_rate', 0):.2f}%",
                             help="Click-through rate achieved by this policy during the experiment.")
                    st.metric("Impressions", f"{entry.get('total_impressions', 0):,}",
                             help="Total ad impressions served during the experiment period.")
    else:
        st.info("No MARL promotion history available. Run experiments and complete campaigns to see history.")

with tab4:
    st.subheader("Policy Comparison")
    st.caption("Head-to-head OPE comparison of two policies on the same historical data with confidence intervals.")
    
    with st.form("comparison_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            policy_a = st.selectbox(
                "Policy A",
                ["thompson_sampling", "linucb", "marl_agent", "random"],
                index=0,
                help="First policy in the comparison (typically the current production policy)."
            )
        
        with col2:
            policy_b = st.selectbox(
                "Policy B",
                ["linucb", "thompson_sampling", "marl_agent", "random"],
                index=1,
                help="Second policy in the comparison (typically the candidate you want to test)."
            )
        
        campaign_id = st.text_input(
            "Campaign ID (optional)",
            placeholder="Leave empty to use all recent campaigns",
            help="Select the campaign whose data will be used for policy evaluation. Leave empty to aggregate across all recent campaigns."
        )
        
        submitted = st.form_submit_button("🆚 Compare Policies", type="primary")
        
        if submitted:
            with st.spinner(f"Comparing {policy_a} vs {policy_b}..."):
                result = api.compare_policies(
                    policy_a=policy_a,
                    policy_b=policy_b,
                    campaign_id=campaign_id if campaign_id else None
                )
            
            if result:
                status = result.get('status', 'unknown')
                
                if status == 'completed':
                    st.success("✅ Comparison completed!")
                    
                    policy_a_data = result.get('policy_a', {})
                    policy_b_data = result.get('policy_b', {})
                    comparison = result.get('comparison', {})
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown(f"### {policy_a}")
                        st.metric("Value", f"{policy_a_data.get('value', 0):.3f}",
                                 help="IPS-estimated expected reward for Policy A on the evaluation dataset.")
                        st.caption(f"CI: [{policy_a_data.get('ci_lower', 0):.3f}, {policy_a_data.get('ci_upper', 0):.3f}]")
                    
                    with col_b:
                        st.markdown(f"### {policy_b}")
                        st.metric("Value", f"{policy_b_data.get('value', 0):.3f}",
                                 help="IPS-estimated expected reward for Policy B on the evaluation dataset.")
                        st.caption(f"CI: [{policy_b_data.get('ci_lower', 0):.3f}, {policy_b_data.get('ci_upper', 0):.3f}]")
                    
                    with col_c:
                        st.markdown("### Winner")
                        winner = comparison.get('winner', 'N/A')
                        improvement = comparison.get('improvement_percent', comparison.get('percent_improvement', 0))
                        
                        st.metric("Winner", winner,
                                 help="The policy with the higher estimated reward value.")
                        st.metric("Improvement", f"{improvement:.2f}%",
                                 help="Percentage by which the winner outperforms the other policy.")
                        
                        if comparison.get('statistically_significant'):
                            st.success("✅ Significant")
                        else:
                            st.warning("⚠️ Not significant")
                        
                        st.caption(f"Confidence: {comparison.get('confidence', 'low')}")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name=policy_a,
                        x=[policy_a],
                        y=[policy_a_data.get('value', 0)],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[policy_a_data.get('ci_upper', 0) - policy_a_data.get('value', 0)],
                            arrayminus=[policy_a_data.get('value', 0) - policy_a_data.get('ci_lower', 0)]
                        )
                    ))
                    
                    fig.add_trace(go.Bar(
                        name=policy_b,
                        x=[policy_b],
                        y=[policy_b_data.get('value', 0)],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[policy_b_data.get('ci_upper', 0) - policy_b_data.get('value', 0)],
                            arrayminus=[policy_b_data.get('value', 0) - policy_b_data.get('ci_lower', 0)]
                        )
                    ))
                    
                    fig.update_layout(
                        title="Policy Comparison with 95% Confidence Intervals",
                        yaxis_title="Expected Value",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="policy_reward_chart")
                    
                    st.caption(f"Samples used: {result.get('samples_used', 0)}")
                
                elif status == 'insufficient_data':
                    st.warning(result.get('message', 'Insufficient data'))
                else:
                    st.error(f"Comparison failed: {result.get('message', 'Unknown error')}")

st.markdown("---")
st.caption("MARL Promotion & OPE - Rigorous policy evaluation before production deployment")
