#!/usr/bin/env python3
"""
Generate additional thesis visualizations:
- Cost efficiency analysis
- HITL governance analysis
- Experiment convergence
- Delayed reward funnel
- System event timeline
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'data'
VIZ_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'visualizations'
TABLE_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'tables'

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548', '#607D8B']

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'figure.figsize': (10, 6),
    'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def load_json(filename):
    path = DATA_DIR / filename
    if not path.exists(): return None
    with open(path) as f: return json.load(f)


def viz_cost_efficiency():
    """Figure: Cost Efficiency Analysis"""
    data = load_json('cost_tracking.json')
    if not data: return
    
    records = data.get('records', [])
    if not records:
        print("  No cost records")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cost by provider
    provider_costs = defaultdict(float)
    provider_tokens = defaultdict(int)
    for r in records:
        p = r.get('provider', 'unknown') or 'unknown'
        provider_costs[p] += r.get('amount', 0)
        provider_tokens[p] += (r.get('tokens_prompt', 0) or 0) + (r.get('tokens_completion', 0) or 0)
    
    providers = list(provider_costs.keys())
    costs = [provider_costs[p] for p in providers]
    
    axes[0, 0].bar(providers, costs, color=COLORS[:len(providers)], alpha=0.85)
    axes[0, 0].set_ylabel('Cost (€)')
    axes[0, 0].set_title('Cost by Provider')
    for i, c in enumerate(costs):
        axes[0, 0].text(i, c + max(costs)*0.02, f'€{c:.4f}', ha='center', fontsize=9)
    
    # Token usage by provider
    tokens = [provider_tokens[p] for p in providers]
    axes[0, 1].bar(providers, tokens, color=COLORS[:len(providers)], alpha=0.85)
    axes[0, 1].set_ylabel('Tokens')
    axes[0, 1].set_title('Token Usage by Provider')
    for i, t in enumerate(tokens):
        axes[0, 1].text(i, t + max(tokens)*0.02, f'{t:,}', ha='center', fontsize=9)
    
    # Cost per token
    cpt = [provider_costs[p] / provider_tokens[p] * 1000 if provider_tokens[p] > 0 else 0 for p in providers]
    axes[1, 0].bar(providers, cpt, color=COLORS[:len(providers)], alpha=0.85)
    axes[1, 0].set_ylabel('Cost per 1K Tokens (€)')
    axes[1, 0].set_title('Cost Efficiency')
    for i, c in enumerate(cpt):
        axes[1, 0].text(i, c + max(cpt)*0.02, f'€{c:.6f}', ha='center', fontsize=9)
    
    # TCO Breakdown
    total_llm = data.get('total_llm_cost', sum(costs))
    total_infra = data.get('total_infra_cost_est', 0)
    
    axes[1, 1].bar(['LLM API', 'Infrastructure\n(Est.)'], [total_llm, total_infra], 
                color=[COLORS[0], COLORS[1]], alpha=0.85)
    axes[1, 1].set_ylabel('Total Cost (€) [Log Scale]')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Total Cost of Ownership (TCO)')
    
    # Stacked Total
    bottom_val = 0
    total_val = total_llm + total_infra
    axes[1, 1].text(0, total_llm + total_val*0.02, f'€{total_llm:.2f}', ha='center', fontweight='bold')
    if total_infra > 0:
        axes[1, 1].text(1, total_infra * 1.1, f'€{total_infra:.2f}', ha='center', fontweight='bold')
        
    axes[1, 1].text(0.5, 0.9, f'Total: €{total_val:.2f}', 
                 ha='center', transform=axes[1, 1].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Add OpenAI cost comparison annotation
    if sum(costs) > 0:
        openai_cost_per_1k = 0.03  # GPT-4o approx cost per 1K tokens
        ollama_cost_per_1k = cpt[0] if cpt else 0
        total_tokens_val = sum(tokens)
        openai_equiv = total_tokens_val / 1000 * openai_cost_per_1k
        
        fig.text(0.5, 0.02,
                f'[Note] OpenAI GPT-4o equivalent cost: EUR {openai_equiv:.2f} (at EUR {openai_cost_per_1k}/1K tokens) -- '
                f'Ollama saves {((openai_equiv - sum(costs)) / max(openai_equiv, 0.001) * 100):.0f}%',
                ha='center', fontsize=10, style='italic', color='#444', 
                bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.suptitle(f'Cost Control Facility: LLM Cost Analysis\n(Total: €{sum(costs):.4f}, {sum(tokens):,} tokens)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust rect to accommodate suptitle and footer
    plt.savefig(VIZ_DIR / 'fig_cost_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  ✅ fig_cost_analysis.png")


def viz_hitl_governance():
    """Figure: HITL Governance Analysis"""
    data = load_json('hitl_queue.json')
    if not data: return
    
    items = data.get('items', [])
    decisions = data.get('decision_distribution', {})
    
    # Normalize "approved" → "approve"
    if 'approved' in decisions:
        decisions['approve'] = decisions.get('approve', 0) + decisions.pop('approved')
    for item in items:
        if item.get('decision') == 'approved':
            item['decision'] = 'approve'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Decision distribution pie chart
    if decisions:
        labels = list(decisions.keys())
        sizes = list(decisions.values())
        colors_map = {
            'approve': '#4CAF50', 'reject': '#F44336',
            'modify': '#FF9800', 'regenerate': '#FF9800', 'pending': '#9E9E9E'
        }
        pie_colors = [colors_map.get(l, '#607D8B') for l in labels]
        
        axes[0].pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
                   colors=pie_colors, autopct='%1.0f%%', startangle=90)
        axes[0].set_title('HITL Decision Distribution')
    
    # Approval rate over time (cumulative)
    sorted_items = sorted(items, key=lambda x: x.get('created_at', ''))
    
    approve_count = 0
    reject_count = 0
    approval_rates = []
    
    for item in sorted_items:
        d = item.get('decision', '')
        if d in ('approve', 'approved'):
            approve_count += 1
        elif d in ('reject', 'rejected'):
            reject_count += 1
        total = approve_count + reject_count
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
    
    total_reviewed = sum(decisions.values())
    total_approved = decisions.get('approve', 0) + decisions.get('approved', 0)
    override_rate = (1 - total_approved / total_reviewed) * 100 if total_reviewed > 0 else 0
    
    plt.suptitle(f'Human-in-the-Loop Governance Analysis\n(Override Rate: {override_rate:.1f}%, Total Reviews: {total_reviewed})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_hitl_governance.png', bbox_inches='tight')
    plt.close()
    print("  ✅ fig_hitl_governance.png")


def viz_delayed_reward_funnel():
    """Figure: Delayed Reward Attribution Funnel"""
    data = load_json('delayed_rewards.json')
    if not data: return
    
    rewards = data.get('rewards', [])
    if not rewards:
        print("  No delayed reward data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Status distribution
    status_counts = Counter(r.get('status', 'unknown') for r in rewards)
    
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors_map = {
        'pending': '#FF9800', 'booked': '#2196F3', 'converted': '#4CAF50',
        'expired': '#9E9E9E', 'attributed': '#00BCD4'
    }
    pie_colors = [colors_map.get(l, '#607D8B') for l in labels]
    
    axes[0].pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
               colors=pie_colors, autopct='%1.0f%%', startangle=90)
    axes[0].set_title('Delayed Reward Status Distribution')
    
    # Meeting scheduled vs attended
    scheduled = sum(1 for r in rewards if r.get('meeting_scheduled'))
    not_scheduled = len(rewards) - scheduled
    
    axes[1].bar(['Registered', 'Meeting\nScheduled'],
               [len(rewards), scheduled],
               color=[COLORS[0], COLORS[1]], alpha=0.85)
    
    for i, v in enumerate([len(rewards), scheduled]):
        axes[1].text(i, v + 0.2, str(v), ha='center', fontweight='bold')
    
    axes[1].set_ylabel('Count')
    axes[1].set_title('Lead Progression: Registration → Booking')
    
    plt.suptitle(f'Delayed Reward Attribution Analysis\n(Total Leads: {len(rewards)})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_delayed_rewards.png', bbox_inches='tight')
    plt.close()
    print("  ✅ fig_delayed_rewards.png")


def viz_bandit_arm_performance():
    """Figure: Bandit Arm Performance (Thompson Sampling Parameters)"""
    data = load_json('bandit_decisions.json')
    if not data: return
    
    arms = data.get('decisions', [])
    if not arms:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Expected reward distribution
    expected_rewards = [a.get('expected_reward', 0.5) for a in arms if a.get('pulls', 0) > 0]
    if expected_rewards:
        axes[0].hist(expected_rewards, bins=30, color=COLORS[0], alpha=0.85, edgecolor='white')
        axes[0].axvline(np.mean(expected_rewards), color='red', linestyle='--',
                       label=f'Mean: {np.mean(expected_rewards):.3f}')
        axes[0].set_xlabel('Expected Reward (α/(α+β))')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Thompson Sampling: Expected Reward Distribution')
        axes[0].legend()
    
    # Pulls distribution (exploration vs exploitation)
    pulls = [a.get('pulls', 0) for a in arms if a.get('pulls', 0) > 0]
    if pulls:
        axes[1].hist(pulls, bins=30, color=COLORS[1], alpha=0.85, edgecolor='white')
        axes[1].axvline(np.mean(pulls), color='red', linestyle='--',
                       label=f'Mean: {np.mean(pulls):.0f}')
        axes[1].set_xlabel('Number of Pulls')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Arm Selection Frequency\n(Exploration-Exploitation Trade-off)')
        axes[1].legend()
    
    total_arms = len(arms)
    active_arms = sum(1 for a in arms if a.get('pulls', 0) > 0)
    
    plt.suptitle(f'Contextual Bandit: Arm Performance Analysis\n(Total Arms: {total_arms}, Active: {active_arms})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_bandit_arms.png', bbox_inches='tight')
    plt.close()
    print("  ✅ fig_bandit_arms.png")


def viz_system_event_timeline():
    """Figure: System Event Timeline Analysis"""
    data = load_json('workflow_events.json')
    if not data: return
    
    events = data.get('events', [])
    if not events:
        return
    
    # Group events by date
    daily_counts = defaultdict(int)
    daily_types = defaultdict(lambda: defaultdict(int))
    
    for e in events:
        date = e.get('created_at', '')[:10]
        if date:
            daily_counts[date] += 1
            daily_types[date][e.get('event_type', 'unknown')] += 1
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Event volume over time
    sorted_dates = sorted(daily_counts.keys())
    counts = [daily_counts[d] for d in sorted_dates]
    
    axes[0].bar(range(len(sorted_dates)), counts, color=COLORS[0], alpha=0.85)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Event Count')
    axes[0].set_title('Daily Workflow Event Volume')
    
    # Set x-tick labels (show every Nth date)
    step = max(1, len(sorted_dates) // 10)
    axes[0].set_xticks(range(0, len(sorted_dates), step))
    axes[0].set_xticklabels([sorted_dates[i] for i in range(0, len(sorted_dates), step)], rotation=45, fontsize=8)
    
    # Top event types stacked area
    all_types = Counter()
    for dt in daily_types.values():
        all_types.update(dt)
    top_types = [t for t, _ in all_types.most_common(6)]
    
    # Create stacked data
    x = range(len(sorted_dates))
    bottom = np.zeros(len(sorted_dates))
    
    for i, etype in enumerate(top_types):
        values = [daily_types[d].get(etype, 0) for d in sorted_dates]
        axes[1].bar(x, values, bottom=bottom, label=etype, color=COLORS[i % len(COLORS)], alpha=0.85)
        bottom += np.array(values)
    
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Event Count')
    axes[1].set_title('Event Type Breakdown (Top 6)')
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].set_xticks(range(0, len(sorted_dates), step))
    axes[1].set_xticklabels([sorted_dates[i] for i in range(0, len(sorted_dates), step)], rotation=45, fontsize=8)
    
    plt.suptitle(f'System Transparency: Event Timeline\n(Total: {len(events)} events across {len(sorted_dates)} days)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_event_timeline.png', bbox_inches='tight')
    plt.close()
    print("  ✅ fig_event_timeline.png")


def viz_canary_deployments():
    """Figure: Canary Deployment Analysis"""
    data = load_json('canary_deployments.json')
    if not data or not data.get('deployments'):
        print("  No canary deployment data")
        return
    
    deployments = data['deployments']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    statuses = Counter(d.get('status', 'unknown') for d in deployments)
    labels = list(statuses.keys())
    sizes = list(statuses.values())
    
    colors_map = {
        'promoted': '#4CAF50', 'completed': '#4CAF50', 'active': '#2196F3',
        'rolled_back': '#F44336', 'failed': '#F44336', 'monitoring': '#FF9800',
        'full_rollout_100_percent': '#4CAF50', 'canary_25_percent': '#2196F3',
        'canary_5_percent': '#FF9800', 'canary_50_percent': '#00BCD4',
    }
    pie_colors = [colors_map.get(l, '#607D8B') for l in labels]
    
    ax.bar(labels, sizes, color=pie_colors, alpha=0.85, edgecolor='white', linewidth=2)
    ax.set_ylabel('Count')
    ax.set_title(f'Canary Deployment Outcomes\n(Total: {len(deployments)} deployments)')
    
    for i, v in enumerate(sizes):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_canary_deployments.png', bbox_inches='tight')
    plt.close()
    print("  ✅ fig_canary_deployments.png")


def generate_cost_table():
    """Table: Cost Tracking Summary"""
    import csv
    data = load_json('cost_tracking.json')
    if not data: return
    
    headers = ['Metric', 'Value']
    rows = [
        ['Total Records', data.get('total_records', 0)],
        ['Total Cost (LLM)', f"€{data.get('total_llm_cost', data.get('total_cost', 0)):.4f}"],
        ['Infrastructure (Est)', f"€{data.get('total_infra_cost_est', 0):.4f}"],
        ['Total TCO', f"€{data.get('total_tco', 0):.4f}"],
        ['Total Tokens', f"{data.get('total_tokens', 0):,}"],
        ['Cache Hit Rate', f"{data.get('cache_hit_rate', 0):.1f}%"],
        ['Avg TCO/Record', f"€{data.get('total_tco', 0) / max(data.get('total_records', 1), 1):.6f}"],
        ['LLM Cost per 1K Tokens', f"€{data.get('total_llm_cost', 0) / max(data.get('total_tokens', 1), 1) * 1000:.6f}"],
    ]
    
    with open(TABLE_DIR / 'cost_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  ✅ cost_summary.csv")


def generate_hitl_table():
    """Table: HITL Queue Analysis"""
    import csv
    data = load_json('hitl_queue.json')
    if not data: return
    
    decisions = data.get('decision_distribution', {})
    total = sum(decisions.values())
    approved = decisions.get('approve', 0) + decisions.get('approved', 0)
    rejected = decisions.get('reject', 0) + decisions.get('rejected', 0)
    
    headers = ['Metric', 'Value']
    rows = [
        ['Total Reviews', total],
        ['Approved', approved],
        ['Rejected', rejected],
        ['Regenerated', decisions.get('regenerate', 0)],
        ['Pending', decisions.get('pending', 0)],
        ['Approval Rate', f"{approved / max(total, 1) * 100:.1f}%"],
        ['Override Rate', f"{(total - approved) / max(total, 1) * 100:.1f}%"],
        ['Target Override Rate', '<5%'],
    ]
    
    with open(TABLE_DIR / 'hitl_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  ✅ hitl_summary.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("THESIS RESEARCH: Additional Visualizations & Tables")
    print("=" * 60)
    
    generators = [
        ("Cost Analysis", viz_cost_efficiency),
        ("HITL Governance", viz_hitl_governance),
        ("Delayed Rewards", viz_delayed_reward_funnel),
        ("Bandit Arms", viz_bandit_arm_performance),
        ("Event Timeline", viz_system_event_timeline),
        ("Canary Deployments", viz_canary_deployments),
        ("Cost Table", generate_cost_table),
        ("HITL Table", generate_hitl_table),
    ]
    
    for name, gen in generators:
        try:
            print(f"\n📊 Generating: {name}...")
            gen()
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Additional visualizations complete!")
    print("=" * 60)
