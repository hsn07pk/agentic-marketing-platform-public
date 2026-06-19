#!/usr/bin/env python3
"""
Regenerate stale visualizations that still show incorrect data.
Fixes:
  - fig_infrastructure.png: was showing hash IDs, wrong container count (17/32)
  - fig_hitl_governance.png: was showing 28.3% override rate, 53 total reviews
"""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research'
DATA_DIR = BASE / 'data'
TABLE_DIR = BASE / 'tables'
VIZ_DIR = BASE / 'visualizations'

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548', '#607D8B']


def regenerate_infrastructure_viz():
    """Regenerate fig_infrastructure.png with corrected data."""
    # Load corrected container metrics
    with open(DATA_DIR / 'container_metrics.json') as f:
        data = json.load(f)
    containers = data['containers']

    # Load corrected container resources table for roles
    roles = {}
    with open(TABLE_DIR / 'container_resources.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            roles[row['Container']] = row['Role']

    # Load API metrics
    with open(DATA_DIR / 'api_metrics.json') as f:
        api_metrics = json.load(f)

    # Load prometheus targets
    try:
        with open(DATA_DIR / 'prometheus_targets.json') as f:
            targets_data = json.load(f)
        targets = targets_data.get('targets', [])
    except FileNotFoundError:
        targets = []

    # Sort by CPU usage descending
    sorted_containers = sorted(containers, key=lambda x: x.get('cpu_mean', 0), reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Container CPU usage (Mean + Max shadow)
    names = [c.get('name', '?') for c in sorted_containers]
    cpu_means = [c.get('cpu_mean', 0) for c in sorted_containers]
    cpu_maxs = [c.get('cpu_max', 0) for c in sorted_containers]

    axes[0, 0].barh(names[::-1], cpu_maxs[::-1], color=COLORS[0], alpha=0.3, label='Max (6h)')
    axes[0, 0].barh(names[::-1], cpu_means[::-1], color=COLORS[0], alpha=0.85, label='Mean (6h)')
    axes[0, 0].set_xlabel('CPU Usage (%)')
    axes[0, 0].set_title('Container CPU Utilization (Mean vs Max)')
    axes[0, 0].legend()

    # Plot 2: Container memory usage (Mean + Max shadow)
    mem_sorted = sorted(containers, key=lambda x: x.get('mem_mean', 0), reverse=True)
    mem_names = [c.get('name', '?') for c in mem_sorted]
    mem_means = [c.get('mem_mean', 0) for c in mem_sorted]
    mem_maxs = [c.get('mem_max', 0) for c in mem_sorted]

    axes[0, 1].barh(mem_names[::-1], mem_maxs[::-1], color=COLORS[1], alpha=0.3, label='Max (6h)')
    axes[0, 1].barh(mem_names[::-1], mem_means[::-1], color=COLORS[1], alpha=0.85, label='Mean (6h)')
    axes[0, 1].set_xlabel('Memory Usage (MB)')
    axes[0, 1].set_title('Container Memory Utilization (Mean vs Max)')
    axes[0, 1].legend()

    # Plot 3: Prometheus targets health
    if targets:
        health_counts = {'up': 0, 'down': 0, 'unknown': 0}
        for t in targets:
            h = t.get('health', 'unknown')
            health_counts[h] = health_counts.get(h, 0) + 1

        labels = [k for k, v in health_counts.items() if v > 0]
        sizes = [health_counts[k] for k in labels]
        colors_map = {'up': '#4CAF50', 'down': '#F44336', 'unknown': '#9E9E9E'}

        axes[1, 0].pie(sizes, labels=[f"{l} ({s})" for l, s in zip(labels, sizes)],
                      colors=[colors_map.get(l, '#607D8B') for l in labels],
                      autopct='%1.0f%%', startangle=90)
        axes[1, 0].set_title('Monitoring Targets Health')
    else:
        axes[1, 0].text(0.5, 0.5, 'No target data available', ha='center', va='center')
        axes[1, 0].set_title('Monitoring Targets Health')

    # Plot 4: System overview text
    axes[1, 1].axis('off')
    n_services = 16  # 15 running + ollama (profiles)
    n_monitored = len(containers)
    n_healthy = sum(1 for t in targets if t.get('health') == 'up')

    info_text = "System Infrastructure Summary\n" + "=" * 35 + "\n\n"
    info_text += f"Services Defined: {n_services}\n"
    info_text += f"Containers Monitored: {n_monitored}\n"
    info_text += f"Prometheus Targets: {len(targets)}\n"
    info_text += f"Targets Healthy: {n_healthy}\n\n"

    if api_metrics:
        rps = api_metrics.get('request_rate_per_sec')
        info_text += f"API Request Rate: {f'{rps:.2f}' if rps is not None else 'N/A'}/s\n"
        p50 = api_metrics.get('latency_p50_ms')
        info_text += f"API Latency P50: {f'{p50:.1f}' if p50 is not None else 'N/A'}ms\n"
        p95 = api_metrics.get('latency_p95_ms')
        info_text += f"API Latency P95: {f'{p95:.1f}' if p95 is not None else 'N/A'}ms\n"
        err = api_metrics.get('error_rate', 0)
        info_text += f"API Error Rate: {f'{err:.4f}' if err is not None else 'N/A'}\n"

    axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    plt.suptitle(f'Infrastructure & Monitoring: Docker Compose Deployment\n({n_services}-Service Architecture, {n_monitored} Containers Monitored)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_infrastructure.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  [OK] fig_infrastructure.png regenerated")


def regenerate_hitl_governance_viz():
    """Regenerate fig_hitl_governance.png with corrected override rate."""
    with open(DATA_DIR / 'hitl_queue.json') as f:
        data = json.load(f)

    items = data.get('items', [])

    # Separate completed vs pending
    completed_items = [i for i in items if i.get('status') == 'completed']
    total_completed = len(completed_items)

    # Normalize "approved" -> "approve" and count decisions among completed only
    decision_counts = {}
    for item in completed_items:
        d = item.get('decision', 'unknown')
        if d == 'approved':
            d = 'approve'
        decision_counts[d] = decision_counts.get(d, 0) + 1

    pending_count = sum(1 for i in items if i.get('status') == 'pending')

    # Calculate override rate: overrides = reject + regenerate
    total_approved = decision_counts.get('approve', 0)
    total_overridden = total_completed - total_approved
    override_rate = (total_overridden / total_completed * 100) if total_completed > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Decision distribution pie chart (completed items only)
    if decision_counts:
        labels = list(decision_counts.keys())
        sizes = list(decision_counts.values())
        colors_map = {
            'approve': '#4CAF50', 'reject': '#F44336',
            'modify': '#FF9800', 'regenerate': '#FF9800', 'pending': '#9E9E9E'
        }
        pie_colors = [colors_map.get(l, '#607D8B') for l in labels]

        axes[0].pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=pie_colors, autopct='%1.0f%%', startangle=90)
        axes[0].set_title(f'HITL Decision Distribution\n(Completed Reviews Only, {pending_count} Pending Excluded)')

    # Plot 2: Approval rate over time (cumulative)
    sorted_items = sorted(completed_items, key=lambda x: x.get('created_at', ''))

    approve_count = 0
    override_count = 0
    approval_rates = []

    for item in sorted_items:
        d = item.get('decision', '')
        if d in ('approve', 'approved'):
            approve_count += 1
        else:
            override_count += 1
        total = approve_count + override_count
        if total > 0:
            approval_rates.append(approve_count / total * 100)

    if approval_rates:
        axes[1].plot(range(len(approval_rates)), approval_rates, color=COLORS[1], linewidth=2)
        axes[1].fill_between(range(len(approval_rates)), approval_rates, alpha=0.1, color=COLORS[1])
        axes[1].axhline(95, color='green', linestyle='--', alpha=0.5, label='Target: 95% approval')
        axes[1].set_xlabel('Review Sequence')
        axes[1].set_ylabel('Cumulative Approval Rate (%)')
        axes[1].set_title('HITL Approval Rate Over Time')
        axes[1].set_ylim(0, 105)
        axes[1].legend()

    plt.suptitle(f'Human-in-the-Loop Governance Analysis\n(Override Rate: {override_rate:.1f}%, Completed Reviews: {total_completed})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_hitl_governance.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  [OK] fig_hitl_governance.png regenerated")


if __name__ == '__main__':
    print("=" * 60)
    print("Regenerating stale visualizations with corrected data")
    print("=" * 60)

    print("\n1. Infrastructure visualization...")
    regenerate_infrastructure_viz()

    print("\n2. HITL Governance visualization...")
    regenerate_hitl_governance_viz()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
