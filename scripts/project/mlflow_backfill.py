"""
Comprehensive MLflow enrichment: artifacts, models, visualizations, documentation.
Runs inside the API container.
"""
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import json
import os
import sys
import csv
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# Add project root
sys.path.insert(0, '/app')
from src.data_layer.database.connection import sync_session_maker
from src.data_layer.database.models import (
    Experiment, BanditArm, Campaign, Content, WorkflowEvent
)
from sqlalchemy import select, func

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

EXPERIMENT_NAME = "agentic-production"


def setup_experiment():
    """Create a professional experiment with full metadata."""
    try:
        exp_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "project": "agentic-ai-marketing-platform",
                "institution": "University of Oulu",
                "thesis": "Autonomous AI Agent for B2B SaaS Marketing",
                "framework": "Thompson Sampling with Bayesian Beta Posteriors",
                "description": "Production bandit experiments for content optimization in autonomous marketing pipeline",
                "version": "2.0",
                "environment": "production",
            }
        )
        print(f"Created experiment: {EXPERIMENT_NAME} (id={exp_id})")
    except Exception:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        exp_id = exp.experiment_id
        print(f"Using existing experiment: {EXPERIMENT_NAME} (id={exp_id})")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    return exp_id


def get_experiment_data():
    """Extract rich experiment data from database."""
    with sync_session_maker() as session:
        experiments = session.execute(
            select(Experiment).order_by(Experiment.started_at)
        ).scalars().all()
        
        data = []
        for exp in experiments:
            arms = session.execute(
                select(BanditArm).where(BanditArm.experiment_id == exp.id)
            ).scalars().all()
            
            if not arms:
                continue
            
            campaign = None
            if hasattr(exp, 'campaign_id') and exp.campaign_id:
                campaign = session.execute(
                    select(Campaign).where(Campaign.id == exp.campaign_id)
                ).scalars().first()
            
            arm_data = []
            for a in arms:
                alpha = float(a.alpha or 1)
                beta = float(a.beta or 1)
                pulls = a.pulls or 0
                expected = alpha / (alpha + beta)
                successes = max(0, int(alpha - 1))
                arm_data.append({
                    "arm_id": a.arm_id,
                    "variant_name": getattr(a, 'variant_name', a.arm_id),
                    "alpha": alpha,
                    "beta": beta,
                    "pulls": pulls,
                    "successes": successes,
                    "expected_reward": round(expected, 6),
                    "ctr": round(successes / max(pulls, 1), 6),
                    "confidence_interval_lower": round(max(0, expected - 1.96 * (expected * (1 - expected) / max(pulls, 1)) ** 0.5), 6),
                    "confidence_interval_upper": round(min(1, expected + 1.96 * (expected * (1 - expected) / max(pulls, 1)) ** 0.5), 6),
                })
            
            total_pulls = sum(a["pulls"] for a in arm_data)
            total_successes = sum(a["successes"] for a in arm_data)
            best_arm = max(arm_data, key=lambda x: x["expected_reward"])
            
            # Regret calculation
            optimal_reward = best_arm["expected_reward"]
            if total_pulls > 0:
                weighted_reward = sum(a["expected_reward"] * a["pulls"] for a in arm_data) / total_pulls
                cumulative_regret = (optimal_reward - weighted_reward) * total_pulls
                normalized_regret = cumulative_regret / max(total_pulls, 1)
            else:
                cumulative_regret = 0
                normalized_regret = 0
            
            data.append({
                "experiment": exp,
                "arms": arm_data,
                "campaign": campaign,
                "total_pulls": total_pulls,
                "total_successes": total_successes,
                "overall_ctr": total_successes / max(total_pulls, 1),
                "best_arm": best_arm,
                "cumulative_regret": cumulative_regret,
                "normalized_regret": normalized_regret,
                "num_arms": len(arm_data),
            })
        
        return data


def generate_arm_distribution_chart(arms, exp_name, tmpdir):
    """Generate Beta distribution visualization for each arm."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
    except ImportError:
        return None
    
    fig, axes = plt.subplots(1, min(len(arms), 4), figsize=(4 * min(len(arms), 4), 4))
    if len(arms) == 1:
        axes = [axes]
    
    x = np.linspace(0, 1, 200)
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    
    for i, (arm, ax) in enumerate(zip(arms[:4], axes)):
        alpha, beta_val = arm['alpha'], arm['beta']
        y = stats.beta.pdf(x, alpha, beta_val)
        color = colors[i % len(colors)]
        ax.fill_between(x, y, alpha=0.3, color=color)
        ax.plot(x, y, color=color, linewidth=2)
        ax.axvline(x=arm['expected_reward'], color=color, linestyle='--', alpha=0.7)
        ax.set_title(f"Arm {arm['arm_id'][:15]}", fontsize=9)
        ax.set_xlabel('θ (Success Probability)')
        ax.set_ylabel('Density')
        ax.text(0.95, 0.95, f"α={alpha:.1f}\nβ={beta_val:.1f}\nE[θ]={arm['expected_reward']:.3f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"Beta Posterior Distributions", fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(tmpdir, "beta_posteriors.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def generate_arm_comparison_chart(arms, tmpdir):
    """Generate arm comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    names = [a['arm_id'][:12] for a in arms]
    expected = [a['expected_reward'] for a in arms]
    pulls = [a['pulls'] for a in arms]
    ci_low = [a['confidence_interval_lower'] for a in arms]
    ci_high = [a['confidence_interval_upper'] for a in arms]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(arms)))
    
    yerr_low = [e - l for e, l in zip(expected, ci_low)]
    yerr_high = [h - e for e, h in zip(expected, ci_high)]
    
    ax1.bar(names, expected, yerr=[yerr_low, yerr_high], capsize=5, color=colors)
    ax1.set_title("Expected Reward (95% CI)")
    ax1.set_ylabel("E[θ]")
    ax1.tick_params(axis='x', rotation=30)
    
    ax2.bar(names, pulls, color=colors)
    ax2.set_title("Pulls per Arm")
    ax2.set_ylabel("Pulls")
    ax2.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    path = os.path.join(tmpdir, "arm_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def generate_experiment_report(data, tmpdir):
    """Generate markdown experiment report."""
    exp = data["experiment"]
    arms = data["arms"]
    campaign = data["campaign"]
    best = data["best_arm"]
    
    report = f"""# Experiment Report: {exp.name}

## Summary
| Metric | Value |
|--------|-------|
| Experiment ID | `{exp.id}` |
| Algorithm | Thompson Sampling (Bayesian Beta) |
| Status | {'Active' if getattr(exp, 'is_active', False) else 'Completed'} |
| Started | {exp.started_at} |
| Arms | {data['num_arms']} |
| Total Pulls | {data['total_pulls']:,} |
| Total Successes | {data['total_successes']:,} |
| Overall CTR | {data['overall_ctr']:.4f} ({data['overall_ctr']*100:.2f}%) |
| Cumulative Regret | {data['cumulative_regret']:.2f} |
| Normalized Regret | {data['normalized_regret']:.4f} |
"""
    
    if campaign:
        report += f"""
## Campaign
| Field | Value |
|-------|-------|
| Name | {campaign.name} |
| Platform | {campaign.platform.value if campaign.platform else 'N/A'} |
| Status | {campaign.status.value if campaign.status else 'N/A'} |
| Budget | €{campaign.budget_total or 0:.2f} |
"""
    
    report += f"""
## Winner
**Best Arm**: `{best['arm_id']}`
- Expected Reward: {best['expected_reward']:.4f}
- 95% CI: [{best['confidence_interval_lower']:.4f}, {best['confidence_interval_upper']:.4f}]
- Pulls: {best['pulls']:,}
- CTR: {best['ctr']:.4f}

## Arm Details
| Arm ID | Pulls | Successes | CTR | E[θ] | α | β | 95% CI |
|--------|-------|-----------|-----|------|---|---|--------|
"""
    for a in arms:
        report += f"| {a['arm_id'][:20]} | {a['pulls']:,} | {a['successes']:,} | {a['ctr']:.4f} | {a['expected_reward']:.4f} | {a['alpha']:.1f} | {a['beta']:.1f} | [{a['confidence_interval_lower']:.4f}, {a['confidence_interval_upper']:.4f}] |\n"
    
    report += f"""
## Methodology
- **Algorithm**: Thompson Sampling with Beta-Bernoulli conjugate priors
- **Prior**: Beta(1, 1) (uniform / uninformative)
- **Update Rule**: On success: α += 1, On failure: β += 1
- **Arm Selection**: Sample θ_i ~ Beta(α_i, β_i), select arm with max θ_i
- **Regret**: Σ(θ* - θ_selected) over all pulls

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    path = os.path.join(tmpdir, "experiment_report.md")
    with open(path, 'w') as f:
        f.write(report)
    return path


def generate_arms_csv(arms, tmpdir):
    """Generate CSV with arm details."""
    path = os.path.join(tmpdir, "arm_metrics.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'arm_id', 'pulls', 'successes', 'ctr', 'expected_reward',
            'alpha', 'beta', 'ci_lower', 'ci_upper'
        ])
        writer.writeheader()
        for a in arms:
            writer.writerow({
                'arm_id': a['arm_id'],
                'pulls': a['pulls'],
                'successes': a['successes'],
                'ctr': a['ctr'],
                'expected_reward': a['expected_reward'],
                'alpha': a['alpha'],
                'beta': a['beta'],
                'ci_lower': a['confidence_interval_lower'],
                'ci_upper': a['confidence_interval_upper'],
            })
    return path


def generate_training_summary(data, tmpdir):
    """Generate JSON training summary."""
    exp = data["experiment"]
    summary = {
        "experiment": {
            "name": exp.name,
            "id": str(exp.id),
            "algorithm": getattr(exp, 'algorithm', 'thompson_sampling') or 'thompson_sampling',
            "type": getattr(exp, 'type', 'content_bandit'),
            "started_at": str(exp.started_at),
            "is_active": getattr(exp, 'is_active', False),
        },
        "performance": {
            "total_pulls": data["total_pulls"],
            "total_successes": data["total_successes"],
            "overall_ctr": data["overall_ctr"],
            "cumulative_regret": data["cumulative_regret"],
            "normalized_regret": data["normalized_regret"],
            "num_arms": data["num_arms"],
        },
        "winner": {
            "arm_id": data["best_arm"]["arm_id"],
            "expected_reward": data["best_arm"]["expected_reward"],
            "pulls": data["best_arm"]["pulls"],
            "ctr": data["best_arm"]["ctr"],
        },
        "arms": data["arms"],
        "methodology": {
            "algorithm": "Thompson Sampling",
            "prior": "Beta(1, 1)",
            "update_rule": "Conjugate Bayesian update",
            "exploration": "Posterior sampling",
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    if data["campaign"]:
        c = data["campaign"]
        summary["campaign"] = {
            "name": c.name,
            "platform": c.platform.value if c.platform else None,
            "status": c.status.value if c.status else None,
            "budget": float(c.budget_total) if c.budget_total else None,
        }
    
    path = os.path.join(tmpdir, "training_summary.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    return path


def generate_serving_config(data, tmpdir):
    """Generate model serving configuration."""
    config = {
        "model_type": "thompson_sampling_bandit",
        "version": "1.0",
        "arms": [
            {
                "arm_id": a["arm_id"],
                "alpha": a["alpha"],
                "beta": a["beta"],
                "expected_reward": a["expected_reward"],
            }
            for a in data["arms"]
        ],
        "selection_strategy": "posterior_sampling",
        "fallback_strategy": "uniform_random",
        "metadata": {
            "total_training_pulls": data["total_pulls"],
            "best_arm": data["best_arm"]["arm_id"],
            "overall_ctr": data["overall_ctr"],
        }
    }
    
    path = os.path.join(tmpdir, "serving_config.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    return path


class ThompsonSamplingModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for Thompson Sampling bandit."""
    
    def load_context(self, context):
        import json
        config_path = context.artifacts.get("serving_config")
        if config_path:
            with open(config_path) as f:
                self.config = json.load(f)
            self.arms = self.config["arms"]
        else:
            self.arms = []
    
    def predict(self, context, model_input):
        import numpy as np
        if not self.arms:
            return {"selected_arm": None, "error": "No arms configured"}
        
        samples = []
        for arm in self.arms:
            sample = np.random.beta(arm["alpha"], arm["beta"])
            samples.append({"arm_id": arm["arm_id"], "sample": float(sample), "expected": arm["expected_reward"]})
        
        best = max(samples, key=lambda x: x["sample"])
        return {"selected_arm": best["arm_id"], "sample_value": best["sample"], "all_samples": samples}


def log_experiment_run(data, run_index, total):
    """Log a single experiment run with full artifacts."""
    exp = data["experiment"]
    arms = data["arms"]
    campaign = data["campaign"]
    
    exp_type = getattr(exp, 'type', 'content_bandit') or 'content_bandit'
    algorithm = getattr(exp, 'algorithm', 'thompson_sampling') or 'thompson_sampling'
    
    # Determine run name
    if campaign:
        run_name = f"{campaign.name[:25]}_{algorithm}"
    else:
        run_name = f"{exp.name[:30]}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # ---- Parameters ----
        mlflow.log_param("experiment_name", exp.name[:250])
        mlflow.log_param("experiment_type", exp_type)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("num_arms", data["num_arms"])
        mlflow.log_param("prior", "Beta(1,1)")
        mlflow.log_param("selection_strategy", "posterior_sampling")
        mlflow.log_param("is_active", str(getattr(exp, 'is_active', False)))
        
        if campaign:
            mlflow.log_param("campaign_name", campaign.name[:100])
            mlflow.log_param("campaign_platform", campaign.platform.value if campaign.platform else "unknown")
            mlflow.log_param("campaign_status", campaign.status.value if campaign.status else "unknown")
            if campaign.budget_total:
                mlflow.log_param("campaign_budget", float(campaign.budget_total))
        
        # ---- Metrics ----
        mlflow.log_metric("total_pulls", data["total_pulls"])
        mlflow.log_metric("total_successes", data["total_successes"])
        mlflow.log_metric("overall_ctr", data["overall_ctr"])
        mlflow.log_metric("cumulative_regret", data["cumulative_regret"])
        mlflow.log_metric("normalized_regret", data["normalized_regret"])
        mlflow.log_metric("num_arms", data["num_arms"])
        mlflow.log_metric("best_arm_expected", data["best_arm"]["expected_reward"])
        
        if data["total_pulls"] > 0:
            exploration_ratio = 1.0 - (max(a["pulls"] for a in arms) / data["total_pulls"])
            mlflow.log_metric("exploration_ratio", exploration_ratio)
            mlflow.log_metric("pulls_per_arm_avg", data["total_pulls"] / data["num_arms"])
        
        # Per-arm metrics
        for i, arm in enumerate(arms[:6]):
            mlflow.log_metric(f"arm_{i}_expected", arm["expected_reward"])
            mlflow.log_metric(f"arm_{i}_pulls", arm["pulls"])
            mlflow.log_metric(f"arm_{i}_ctr", arm["ctr"])
            mlflow.log_metric(f"arm_{i}_alpha", arm["alpha"])
            mlflow.log_metric(f"arm_{i}_beta", arm["beta"])
        
        # ---- Tags ----
        mlflow.set_tag("experiment_id", str(exp.id))
        mlflow.set_tag("winner_arm", data["best_arm"]["arm_id"])
        mlflow.set_tag("model_type", "thompson_sampling_bandit")
        mlflow.set_tag("framework", "agentic-ai")
        if campaign:
            mlflow.set_tag("campaign_id", str(campaign.id))
        
        has_data = data["total_pulls"] > 0
        mlflow.set_tag("has_training_data", str(has_data))
        if has_data:
            mlflow.set_tag("model_quality", "trained" if data["total_pulls"] >= 50 else "limited_data")
        
        # ---- Artifacts ----
        tmpdir = tempfile.mkdtemp()
        try:
            # 1. Training summary JSON
            summary_path = generate_training_summary(data, tmpdir)
            mlflow.log_artifact(summary_path)
            
            # 2. Arms CSV
            csv_path = generate_arms_csv(arms, tmpdir)
            mlflow.log_artifact(csv_path)
            
            # 3. Serving config
            config_path = generate_serving_config(data, tmpdir)
            mlflow.log_artifact(config_path)
            
            # 4. Experiment report
            report_path = generate_experiment_report(data, tmpdir)
            mlflow.log_artifact(report_path)
            
            # 5. Visualizations (only for experiments with data)
            if has_data and len(arms) > 1:
                chart1 = generate_arm_distribution_chart(arms, exp.name, tmpdir)
                if chart1:
                    mlflow.log_artifact(chart1, "visualizations")
                
                chart2 = generate_arm_comparison_chart(arms, tmpdir)
                if chart2:
                    mlflow.log_artifact(chart2, "visualizations")
            
            # 6. Register as PyFunc model (for experiments with sufficient data)
            if has_data and data["total_pulls"] >= 10:
                try:
                    artifacts = {"serving_config": config_path}
                    mlflow.pyfunc.log_model(
                        artifact_path="bandit_model",
                        python_model=ThompsonSamplingModel(),
                        artifacts=artifacts,
                        pip_requirements=["numpy"],
                        registered_model_name=f"agentic-bandit-{exp_type}" if data["total_pulls"] >= 50 else None,
                    )
                except Exception as e:
                    pass  # Model logging is best-effort
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    return run.info.run_id


def generate_experiment_summary():
    """Log an aggregate summary run with overview artifacts."""
    with sync_session_maker() as session:
        total_experiments = session.execute(select(func.count()).select_from(Experiment)).scalar()
        total_campaigns = session.execute(select(func.count()).select_from(Campaign)).scalar()
        total_content = session.execute(select(func.count()).select_from(Content)).scalar()
        total_events = session.execute(select(func.count()).select_from(WorkflowEvent)).scalar()
    
    with mlflow.start_run(run_name="system-overview") as run:
        mlflow.log_param("run_type", "system_overview")
        mlflow.log_param("generated_at", datetime.now().isoformat())
        
        mlflow.log_metric("total_experiments_in_db", total_experiments)
        mlflow.log_metric("total_campaigns", total_campaigns)
        mlflow.log_metric("total_content_pieces", total_content)
        mlflow.log_metric("total_workflow_events", total_events)
        
        mlflow.set_tag("run_purpose", "aggregate_system_overview")
        mlflow.set_tag("model_type", "overview")
        
        tmpdir = tempfile.mkdtemp()
        try:
            # System overview document
            overview = f"""# Agentic AI Marketing Platform - MLflow Overview

## System Statistics
| Metric | Value |
|--------|-------|
| Total Experiments | {total_experiments} |
| Total Campaigns | {total_campaigns} |
| Total Content Pieces | {total_content} |
| Total Workflow Events | {total_events} |

## Architecture
- **Algorithm**: Thompson Sampling (Bayesian Beta-Bernoulli)
- **MARL**: Multi-Agent Reinforcement Learning for deployment gating
- **OPE**: Off-Policy Evaluation for safe deployment
- **Safety**: LLM-based content governance with HITL review

## Experiment Pipeline
1. Campaign creates experiment with content variants as arms
2. Thompson Sampling selects arms based on posterior sampling
3. Engagement feedback updates Beta posteriors (α/β)
4. MARL policy evaluates deployment decisions
5. OPE validates policy before production rollout

## Model Registry
- Bandit policies registered after experiment completion
- Auto-promotion: Dev → Staging (≥95% confidence) → Production (≥99%)
- Serving via Thompson Sampling posterior sampling

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            overview_path = os.path.join(tmpdir, "system_overview.md")
            with open(overview_path, 'w') as f:
                f.write(overview)
            mlflow.log_artifact(overview_path)
            
            # Requirements
            reqs = "numpy>=1.21.0\nscipy>=1.7.0\nscikit-learn>=1.0.0\nmlflow>=2.0.0\n"
            reqs_path = os.path.join(tmpdir, "requirements.txt")
            with open(reqs_path, 'w') as f:
                f.write(reqs)
            mlflow.log_artifact(reqs_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    return run.info.run_id


def main():
    print("=" * 60)
    print("MLflow Professional Enrichment")
    print("=" * 60)
    
    exp_id = setup_experiment()
    
    # Get all experiment data
    print("\nExtracting experiment data from database...")
    all_data = get_experiment_data()
    print(f"Found {len(all_data)} experiments with arms")
    
    # Filter to most interesting ones (with data + campaign context)
    with_data = [d for d in all_data if d["total_pulls"] > 0]
    without_data = [d for d in all_data if d["total_pulls"] == 0]
    
    print(f"  With training data: {len(with_data)}")
    print(f"  Without data: {len(without_data)}")
    
    # Log experiments with data first (up to 30)
    logged = 0
    print(f"\nLogging experiments with training data...")
    for i, data in enumerate(sorted(with_data, key=lambda x: -x["total_pulls"])[:30]):
        exp = data["experiment"]
        campaign = data["campaign"]
        name = campaign.name[:25] if campaign else exp.name[:25]
        print(f"  [{i+1}/{min(len(with_data),30)}] {name} (pulls={data['total_pulls']}, ctr={data['overall_ctr']:.4f})", end=" ")
        try:
            run_id = log_experiment_run(data, i, len(with_data))
            print(f"✅ {run_id[:8]}")
            logged += 1
        except Exception as e:
            print(f"❌ {str(e)[:60]}")
    
    # Log a sample of experiments without data (up to 20)
    print(f"\nLogging experiments without data (sample)...")
    for i, data in enumerate(without_data[:20]):
        exp = data["experiment"]
        name = exp.name[:25]
        print(f"  [{i+1}/20] {name}", end=" ")
        try:
            run_id = log_experiment_run(data, i, 20)
            print(f"✅ {run_id[:8]}")
            logged += 1
        except Exception as e:
            print(f"❌ {str(e)[:60]}")
    
    # Log system overview
    print(f"\nLogging system overview...")
    try:
        overview_id = generate_experiment_summary()
        print(f"  ✅ Overview: {overview_id[:8]}")
        logged += 1
    except Exception as e:
        print(f"  ❌ {str(e)[:60]}")
    
    print(f"\n{'=' * 60}")
    print(f"Total runs logged: {logged}")
    
    # Verify
    runs = mlflow.search_runs(experiment_ids=[exp_id], max_results=5)
    print(f"Verification: {len(runs)} runs visible in experiment")
    for _, r in runs.head(3).iterrows():
        artifacts = client.list_artifacts(r['run_id'])
        print(f"  {r['tags.mlflow.runName']:40s} artifacts={[a.path for a in artifacts]}")
    
    # Check model registry
    try:
        models = client.search_registered_models()
        print(f"\nRegistered models: {len(models)}")
        for m in models[:5]:
            versions = client.search_model_versions(f"name='{m.name}'")
            print(f"  {m.name}: {len(versions)} version(s)")
    except Exception as e:
        print(f"Model registry check: {e}")
    
    print(f"\n✅ MLflow enrichment complete!")
    print(f"View at: http://localhost:5000/#/experiments/{exp_id}")


if __name__ == "__main__":
    main()
