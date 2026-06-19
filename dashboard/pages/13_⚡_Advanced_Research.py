"""
Advanced Research - Research Mode Experiments
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_client import AgenticAPIClient
from components import render_metric_card, render_status_card

st.set_page_config(page_title="Advanced Research - Agentic AI", page_icon="⚡", layout="wide")

@st.cache_resource
def get_api():
    return AgenticAPIClient()

api = get_api()

st.title("⚡ Advanced Research Experiments")
st.caption("Research-mode experimentation with cutting-edge algorithms")

with st.expander("ℹ️ Advanced Research Guide", expanded=False):
    st.markdown("""
    **What this page does:** Run controlled experiments comparing different bandit/RL algorithms
    for academic evaluation and pre-deployment benchmarking.

    ---

    **Algorithm Overview:**

    | Algorithm | Description |
    |---|---|
    | **Thompson Sampling** | Bayesian method that balances exploration and exploitation via posterior sampling. Maintains a probability distribution over each arm's reward and samples from it to decide which arm to pull. |
    | **LinUCB** | Linear Upper Confidence Bound — a context-aware bandit that uses feature vectors to model rewards linearly, adding an exploration bonus based on uncertainty. |
    | **Epsilon-Greedy** | Simple exploration strategy that picks the best-known action with probability 1−ε and a random action with probability ε. |
    | **Transformer Bandits** | Uses transformer models for sequential decision making with long context windows. |
    | **Meta-Learning** | Learn-to-learn across multiple tasks for rapid adaptation. |
    | **Gaussian Processes** | Non-parametric Bayesian approach with strong uncertainty quantification. |

    ---

    **Key Metrics:**
    - **Mean Reward** — Average reward obtained across iterations (higher = better).
    - **Cumulative Regret** — Total reward lost by not always choosing the optimal arm. Measures learning efficiency (lower = better).
    - **Convergence Speed** — How quickly the algorithm identifies the best action.
    - **Std Reward** — Variance in rewards; lower indicates more consistent performance.

    ---

    **When to use:** Comparing algorithms before deploying in production, academic evaluation,
    or validating that a new method outperforms the current baseline.

    **Research workflow:** Configure → Run → Compare Methods → Analyze Results → Select Best → Deploy
    """)

try:
    research_status = api.get_research_mode_status()
    research_enabled = research_status.get("research_mode_enabled", False)
except:
    research_enabled = False
    research_status = {}

if not research_enabled:
    st.error("❌ Research mode is not enabled")
    st.info("💡 Set `ENABLE_RESEARCH_MODE=True` in environment variables to enable advanced experiments")
    st.markdown("""
    **Research Mode Features:**
    - Transformer-based contextual bandits
    - Meta-learning algorithms
    - Gaussian process optimization
    - Causal inference experiments
    - Bayesian optimization
    - Ensemble methods
    """)
    st.stop()

st.success("✅ Research mode enabled")

tab1, tab2, tab3 = st.tabs([
    "🔬 Run Experiment", 
    "📊 Compare Methods",
    "📜 Experiment History"
])

with tab1:
    st.subheader("Run Advanced Research Experiment")
    st.caption("Configure and launch a single experiment with custom parameters.")
    
    st.markdown("""
    **Available Experiment Types:**
    - **Transformer Bandits**: Use transformer models for sequential decision making
    - **Meta-Learning**: Learn-to-learn across multiple tasks
    - **Gaussian Processes**: Non-parametric Bayesian approach
    - **Causal Inference**: Estimate causal effects from observational data
    - **Bayesian Optimization**: Efficient hyperparameter tuning
    - **Ensemble Methods**: Combine multiple models for better performance
    """)
    
    with st.form("research_experiment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_type = st.selectbox(
                "Experiment Type",
                [
                    "transformer_bandits",
                    "meta_learning",
                    "gaussian_process",
                    "causal_inference",
                    "bayesian_optimization",
                    "ensemble"
                ],
                format_func=lambda x: x.replace('_', ' ').title(),
                help="The algorithm or method to evaluate. Each type uses a different approach to the exploration-exploitation tradeoff."
            )
            
            experiment_name = st.text_input(
                "Experiment Name",
                placeholder="e.g., transformer_bandit_v1",
                help="A unique identifier for this experiment run. Used to track and compare results in history."
            )
        
        with col2:
            n_iterations = st.number_input(
                "Number of Iterations",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of rounds to simulate. More iterations = more reliable results but slower."
            )
            

        
        with st.expander("⚙️ Advanced Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    step=0.0001,
                    format="%.4f",
                    help="How quickly the algorithm adapts. Higher = faster learning but more variance."
                )
                
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=256,
                    value=32,
                    help="Number of samples processed per update step. Larger batches give more stable gradients but use more memory."
                )
            
            with col2:
                exploration_param = st.slider(
                    "Exploration Parameter",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Controls exploration vs exploitation. Higher = more exploration of unknown options."
                )
                
                random_seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    max_value=9999,
                    value=42,
                    help="Set for reproducible results. Same seed = same random sequence."
                )
        
        submit = st.form_submit_button("🚀 Run Experiment", type="primary")
        
        if submit:
            if not experiment_name:
                st.error("Experiment name is required")
            else:
                try:
                    experiment_request = {
                        "experiment_type": experiment_type,
                        "experiment_name": experiment_name,
                        "n_iterations": n_iterations,
                        "parameters": {
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "exploration_param": exploration_param,
                            "random_seed": random_seed
                        }
                    }
                    
                    with st.spinner(f"Running {experiment_type} experiment... (this may take several minutes)"):
                        result = api.run_advanced_experiment(experiment_request)
                    
                    if result and result.get('success', False):
                        st.success("✅ Experiment completed!")
                        
                        exp_results = result.get('results', result)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            status = exp_results.get('status', 'completed')
                            st.metric("Status", status.upper() if status else 'N/A', help="Final status of the experiment run.")
                        
                        with col2:
                            duration = exp_results.get('duration_seconds', 0)
                            st.metric("Duration", f"{duration:.2f}s", help="Wall-clock time the experiment took to complete.")
                        
                        with col3:
                            final_reward = exp_results.get('final_reward', 0)
                            st.metric("Final Reward", f"{final_reward:.3f}", help="Reward obtained on the last iteration. Indicates convergence quality.")
                        
                        st.markdown("---")
                        st.markdown("### Performance Metrics")
                        
                        metrics = exp_results.get('metrics', {})
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Mean Reward", f"{metrics.get('mean_reward', 0):.3f}", help="Average reward across all iterations. Higher is better.")
                        
                        with col2:
                            st.metric("Std Reward", f"{metrics.get('std_reward', 0):.3f}", help="Standard deviation of rewards. Lower means more consistent performance.")
                        
                        with col3:
                            st.metric("Max Reward", f"{metrics.get('max_reward', 0):.3f}", help="Highest single-iteration reward observed during the experiment.")
                        
                        with col4:
                            regret = metrics.get('cumulative_regret', 0)
                            st.metric("Cumulative Regret", f"{regret:.2f}", help="Total reward lost by not always choosing the best arm. Measures learning efficiency — lower is better.")
                        
                        st.markdown("---")
                        st.markdown("### Learning Curve")
                        
                        rewards = exp_results.get('reward_history', [])
                        if rewards:
                            reward_df = pd.DataFrame({
                                'Iteration': list(range(1, len(rewards) + 1)),
                                'Reward': rewards
                            })
                            st.line_chart(
                                reward_df.set_index('Iteration'),
                                color="#00FF00"
                            )
                        
                        if st.checkbox("Show Full Results"):
                            st.json(result)
                    
                    else:
                        st.error("Experiment failed")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.subheader("Compare Experimental Methods")
    st.caption("Benchmark different algorithms side-by-side to find the best performer for your use case.")
    
    with st.form("compare_methods_form"):
        st.markdown("#### Select Methods to Compare")
        
        methods = st.multiselect(
            "Experiment Types",
            [
                "transformer_bandits",
                "meta_learning",
                "gaussian_process",
                "bayesian_optimization",
                "ensemble"
            ],
            default=["transformer_bandits", "meta_learning"],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select two or more algorithms to run under identical conditions for fair comparison."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_iterations = st.number_input(
                "Iterations per Method",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of rounds to simulate per method. More iterations = more reliable results but slower."
            )
        
        with col2:
            n_runs = st.number_input(
                "Number of Runs",
                min_value=1,
                max_value=10,
                value=3,
                help="Average results over multiple runs"
            )
        
        compare_button = st.form_submit_button("🆚 Compare Methods", type="primary")
        
        if compare_button:
            if len(methods) < 2:
                st.error("Please select at least 2 methods to compare")
            else:
                try:
                    with st.spinner("Running comparison experiment..."):
                        result = api.compare_experiment_methods(
                            experiment_types=methods,
                            n_iterations=n_iterations
                        )
                    
                    if result:
                        st.success("✅ Comparison completed!")
                        
                        comparison_data = []
                        for method_result in result.get('results', []):
                            if 'error' in method_result:
                                comparison_data.append({
                                    'Method': method_result.get('method', 'N/A').replace('_', ' ').title(),
                                    'Mean Reward': '0.0000',
                                    'Std Reward': '0.0000',
                                    'Max Reward': '0.0000',
                                    'Regret': 'N/A',
                                    'Status': '❌ Error'
                                })
                                continue

                            comparison_data.append({
                                'Method': method_result.get('method', 'N/A').replace('_', ' ').title(),
                                'Mean Reward': f"{method_result.get('mean_reward', 0):.3f}",
                                'Std Reward': f"{method_result.get('std_reward', 0):.3f}",
                                'Max Reward': f"{method_result.get('max_reward', 0):.3f}",
                                'Regret': f"{method_result.get('cumulative_regret', 0):.2f}",
                                'Status': '✅ Success'
                            })
                        
                        if comparison_data:
                            df = pd.DataFrame(comparison_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            winner = result.get('best_method', 'N/A')
                            st.success(f"🏆 **Best Method**: {winner.replace('_', ' ').title()}")
                            
                            st.markdown("---")
                            st.markdown("### Performance Comparison")
                            
                            fig = px.bar(
                                df,
                                x='Method',
                                y='Mean Reward',
                                title="Mean Reward by Method",
                                color='Mean Reward',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True, key="research_chart")
                
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")

with tab3:
    st.subheader("Research Experiment History")
    st.caption("Browse past experiment runs, compare results over time, and export data for analysis.")
    
    st.info("📋 Experiment history tracking - store results for comparison and analysis")
    
    try:
        history_response = api.get_experiment_history(limit=50)
        history_data = history_response.get("experiments", [])
    except:
        history_data = []
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if st.button("📥 Export History"):
            from components import export_to_csv
            export_to_csv(df, f"research_experiments_{datetime.now().strftime('%Y%m%d')}.csv")
    else:
        st.info("No experiment history available")
    
    st.markdown("---")
    
    st.markdown("### 💡 Research Insights")
    
    st.markdown("""
    **Algorithm Performance Summary:**
    - **Transformer Bandits**: Best for sequential decision making with long context
    - **Meta-Learning**: Excellent for few-shot learning scenarios
    - **Gaussian Processes**: Strong uncertainty quantification
    - **Bayesian Optimization**: Efficient for expensive function evaluations
    - **Ensemble Methods**: Robust performance across diverse scenarios
    
    **Recommendations:**
    - Use transformer bandits for content sequence optimization
    - Apply meta-learning for quick adaptation to new personas
    - Leverage Bayesian optimization for hyperparameter tuning
    - Combine methods with ensemble for production robustness
    """)

st.markdown("---")
st.caption(f"Advanced Research Experiments | Last updated: {datetime.now().strftime('%H:%M:%S')}")
