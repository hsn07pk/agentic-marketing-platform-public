#!/usr/bin/env python3
"""
Script 09: Advanced Research Experiments Data Extraction
Runs experiments via API, collects results, and generates thesis-ready visualizations.
"""
import json
import os
import sys
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
OUTPUT_DIR = Path(__file__).parent.parent.parent / "agentic" / "thesis-research"
DATA_DIR = OUTPUT_DIR / "data"
VIZ_DIR = OUTPUT_DIR / "visualizations"
TABLE_DIR = OUTPUT_DIR / "tables"

for d in [DATA_DIR, VIZ_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXPERIMENT_TYPES = [
    "transformer_bandits",
    "meta_learning",
    "gaussian_process",
    "bayesian_optimization",
    "ensemble",
]

ITERATIONS = 200
SEEDS = [42, 123, 456]


def run_experiment(exp_type: str, n_iterations: int, seed: int) -> dict:
    """Run a single experiment via API."""
    payload = {
        "experiment_type": exp_type,
        "experiment_name": f"thesis_{exp_type}_seed{seed}",
        "n_iterations": n_iterations,
        "parameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "exploration_param": 0.1,
            "random_seed": seed,
        },
    }
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/advanced-experiments/run",
            json=payload,
            timeout=300,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  Error {resp.status_code}: {resp.text[:200]}")
            return {"error": resp.text[:200], "status_code": resp.status_code}
    except Exception as e:
        print(f"  Exception: {e}")
        return {"error": str(e)}


def run_comparison(methods: list, n_iterations: int) -> dict:
    """Run method comparison via API."""
    payload = {"experiment_types": methods, "n_iterations": n_iterations}
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/advanced-experiments/compare-methods",
            json=payload,
            timeout=600,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": resp.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def generate_visualizations(all_results: dict, comparison: dict):
    """Generate publication-quality visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return

    # 1. Learning curves for each method (averaged over seeds)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, exp_type in enumerate(EXPERIMENT_TYPES):
        ax = axes[idx]
        if exp_type in all_results:
            runs = all_results[exp_type]
            for seed, result in runs.items():
                rewards = result.get("results", result).get("reward_history", [])
                if rewards:
                    ax.plot(rewards, alpha=0.3, linewidth=0.8)
            # Average
            all_rewards = []
            for seed, result in runs.items():
                rh = result.get("results", result).get("reward_history", [])
                if rh:
                    all_rewards.append(rh)
            if all_rewards:
                min_len = min(len(r) for r in all_rewards)
                avg = np.mean([r[:min_len] for r in all_rewards], axis=0)
                ax.plot(avg, color="red", linewidth=2, label="Mean")
                ax.legend()
        ax.set_title(exp_type.replace("_", " ").title())
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
    axes[-1].axis("off")
    fig.suptitle("Learning Curves by Algorithm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "advanced_learning_curves.png")
    plt.close()

    # 2. Comparison bar chart
    if comparison and "results" in comparison:
        methods = []
        mean_rewards = []
        std_rewards = []
        regrets = []
        for r in comparison["results"]:
            if "error" not in r:
                methods.append(r.get("method", "?").replace("_", " ").title())
                mean_rewards.append(float(r.get("mean_reward", 0)))
                std_rewards.append(float(r.get("std_reward", 0)))
                regrets.append(float(r.get("cumulative_regret", 0)))

        if methods:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
            bars = ax1.bar(methods, mean_rewards, yerr=std_rewards, capsize=5, color=colors)
            ax1.set_title("Mean Reward ± Std by Algorithm")
            ax1.set_ylabel("Mean Reward")
            ax1.tick_params(axis="x", rotation=30)
            for bar, val in zip(bars, mean_rewards):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9)

            bars2 = ax2.bar(methods, regrets, color=colors)
            ax2.set_title("Cumulative Regret by Algorithm")
            ax2.set_ylabel("Cumulative Regret")
            ax2.tick_params(axis="x", rotation=30)
            for bar, val in zip(bars2, regrets):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=9)

            plt.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(VIZ_DIR / "advanced_algorithm_comparison.png")
            plt.close()

    # 3. Convergence speed analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_type in EXPERIMENT_TYPES:
        if exp_type in all_results:
            all_rewards = []
            for seed, result in all_results[exp_type].items():
                rh = result.get("results", result).get("reward_history", [])
                if rh:
                    all_rewards.append(rh)
            if all_rewards:
                min_len = min(len(r) for r in all_rewards)
                avg = np.mean([r[:min_len] for r in all_rewards], axis=0)
                cumulative = np.cumsum(avg) / np.arange(1, len(avg) + 1)
                ax.plot(cumulative, label=exp_type.replace("_", " ").title(), linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Average Reward")
    ax.set_title("Convergence Speed Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "advanced_convergence_speed.png")
    plt.close()

    print("Generated advanced experiment visualizations")


def generate_tables(all_results: dict, comparison: dict):
    """Generate CSV and LaTeX tables."""
    rows = []
    for exp_type in EXPERIMENT_TYPES:
        if exp_type not in all_results:
            continue
        seed_metrics = []
        for seed, result in all_results[exp_type].items():
            res = result.get("results", result)
            metrics = res.get("metrics", {})
            seed_metrics.append({
                "mean_reward": float(metrics.get("mean_reward", 0)),
                "std_reward": float(metrics.get("std_reward", 0)),
                "max_reward": float(metrics.get("max_reward", 0)),
                "cumulative_regret": float(metrics.get("cumulative_regret", 0)),
            })
        if seed_metrics:
            avg_mean = np.mean([m["mean_reward"] for m in seed_metrics])
            avg_std = np.mean([m["std_reward"] for m in seed_metrics])
            avg_max = np.mean([m["max_reward"] for m in seed_metrics])
            avg_regret = np.mean([m["cumulative_regret"] for m in seed_metrics])
            rows.append({
                "Algorithm": exp_type.replace("_", " ").title(),
                "Mean Reward": f"{avg_mean:.4f}",
                "Std Reward": f"{avg_std:.4f}",
                "Max Reward": f"{avg_max:.4f}",
                "Cumulative Regret": f"{avg_regret:.2f}",
                "Seeds": len(seed_metrics),
            })

    if rows:
        # CSV
        import csv
        csv_path = TABLE_DIR / "advanced_experiment_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # LaTeX
        tex_path = TABLE_DIR / "advanced_experiment_results.tex"
        with open(tex_path, "w") as f:
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write("\\caption{Advanced Algorithm Comparison Results}\n")
            f.write("\\label{tab:advanced-algorithms}\n")
            f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
            f.write("Algorithm & Mean Reward & Std & Max Reward & Regret \\\\\n\\midrule\n")
            for r in rows:
                f.write(f"{r['Algorithm']} & {r['Mean Reward']} & {r['Std Reward']} & {r['Max Reward']} & {r['Cumulative Regret']} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

        print(f"Generated tables: {csv_path.name}, {tex_path.name}")


def main():
    print("=" * 60)
    print("Advanced Research Experiments - Thesis Data Extraction")
    print("=" * 60)

    # Check research mode
    try:
        status = requests.get(f"{BASE_URL}/api/v1/advanced-experiments/status", timeout=5).json()
        if not status.get("research_mode_enabled"):
            print("ERROR: Research mode is not enabled!")
            sys.exit(1)
        print(f"Research mode: ENABLED")
        print(f"Available types: {status.get('available_experiment_types', [])}")
    except Exception as e:
        print(f"ERROR: Cannot connect to API: {e}")
        sys.exit(1)

    # Run individual experiments
    all_results = {}
    for exp_type in EXPERIMENT_TYPES:
        print(f"\n--- {exp_type.replace('_', ' ').title()} ---")
        all_results[exp_type] = {}
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_experiment(exp_type, ITERATIONS, seed)
            all_results[exp_type][str(seed)] = result
            if "error" in result:
                print(f"FAILED: {result['error'][:80]}")
            else:
                res = result.get("results", result)
                metrics = res.get("metrics", {})
                print(f"mean={metrics.get('mean_reward', 0):.4f}, regret={metrics.get('cumulative_regret', 0):.2f}")
            time.sleep(0.5)

    # Save raw results
    results_path = DATA_DIR / "advanced_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved raw results to {results_path}")

    # Run comparison
    print("\n--- Method Comparison ---")
    comparison = run_comparison(EXPERIMENT_TYPES, ITERATIONS)
    comp_path = DATA_DIR / "advanced_comparison_results.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"Saved comparison to {comp_path}")

    if comparison.get("best_method"):
        print(f"Best method: {comparison['best_method']}")

    # Generate visualizations
    generate_visualizations(all_results, comparison)

    # Generate tables
    generate_tables(all_results, comparison)

    # Get experiment history
    print("\n--- Experiment History ---")
    try:
        history = requests.get(
            f"{BASE_URL}/api/v1/advanced-experiments/experiment-history",
            params={"limit": 100},
            timeout=30,
        ).json()
        hist_path = DATA_DIR / "advanced_experiment_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        print(f"Saved {len(history.get('experiments', []))} experiment records")
    except Exception as e:
        print(f"History fetch failed: {e}")

    print("\n" + "=" * 60)
    print("Advanced experiments complete!")
    print(f"Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
