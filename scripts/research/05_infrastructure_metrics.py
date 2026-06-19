#!/usr/bin/env python3
"""
Extract and visualize system infrastructure metrics from Prometheus.
Container performance, API latency, resource utilization.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'data'
VIZ_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'visualizations'
TABLE_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'tables'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PROMETHEUS_URL = "http://localhost:9090"
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548', '#607D8B']

def query_prometheus(query, start=None, end=None, step='5m'):
    """Query Prometheus for metrics."""
    try:
        if start and end:
            r = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params={
                'query': query, 'start': start, 'end': end, 'step': step
            }, timeout=10)
        else:
            r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query}, timeout=10)
        
        if r.status_code == 200:
            return r.json().get('data', {}).get('result', [])
    except Exception as e:
        print(f"  ⚠️  Prometheus query failed: {e}")
    return []

def resolve_container_names():
    """Build container hash → service name mapping from Docker."""
    import subprocess
    mapping = {}
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--no-trunc', '--format', '{{.ID}} {{.Names}}'],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                full_id, name = parts
                short_hash = full_id[:12]
                # Normalize: "agentic-marketing-platform-api-1" → "api"
                clean = name.split('-')
                # Find the service name part (after project prefix, before instance suffix)
                for known in ['api', 'worker', 'dashboard', 'postgres', 'redis', 'grafana',
                              'prometheus', 'loki', 'promtail', 'cadvisor', 'mlflow',
                              'alertmanager', 'node_exporter', 'postgres_exporter',
                              'redis_exporter', 'ollama']:
                    if known.replace('_', '-') in name or known in name:
                        mapping[short_hash] = known
                        break
                else:
                    mapping[short_hash] = name
    except Exception as e:
        print(f"  Warning: docker ps failed ({e}), using hash fallback")
    return mapping


def extract_time_series_metrics():
    """Extract container CPU, memory metrics over time (6h window)."""
    import os
    import numpy as np
    
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=6)
    
    # Format for Prometheus API
    end_ts = end_dt.isoformat("T") + "Z"
    start_ts = start_dt.isoformat("T") + "Z"
    
    # Resolve container hashes to service names via Docker
    KNOWN_CONTAINERS = resolve_container_names()
    print(f"  Resolved {len(KNOWN_CONTAINERS)} container name(s) from Docker")
    
    # Range queries for time-series analysis (step=1m for granularity)
    cpu_query = 'rate(container_cpu_usage_seconds_total{id=~"/system.slice/docker-.*"}[5m])'
    mem_query = 'container_memory_usage_bytes{id=~"/system.slice/docker-.*"}'
    
    cpu_series = query_prometheus(cpu_query, start=start_ts, end=end_ts, step='1m')
    mem_series = query_prometheus(mem_query, start=start_ts, end=end_ts, step='1m')
    
    # Deduplicate: multiple container instances for the same service → keep latest
    containers = {}
    
    # Process CPU Series
    for result in cpu_series:
        full_id = result['metric'].get('id', '')
        container_hash = full_id.split('docker-')[-1].split('.')[0][:12]
        name = KNOWN_CONTAINERS.get(container_hash, container_hash)
        
        values = [float(v[1]) * 100 for v in result['values']]
        
        # Use service name as key (deduplicates old+new containers for same service)
        if name not in containers or sum(values) > sum(
            [containers[name].get('cpu_mean', 0)] * containers[name].get('cpu_samples', 1)
        ):
            containers[name] = {'name': name}
            containers[name].update({
                'cpu_mean': round(np.mean([v for v in values if v > 0.001]) if any(v > 0.001 for v in values) else 0, 2),
                'cpu_max': round(np.max(values), 2),
                'cpu_min': round(np.min(values), 2),
                'cpu_std': round(np.std(values), 2),
                'cpu_current': round(values[-1], 2) if values else 0,
                'cpu_samples': len(values)
            })

    # Process Memory Series
    for result in mem_series:
        full_id = result['metric'].get('id', '')
        container_hash = full_id.split('docker-')[-1].split('.')[0][:12]
        name = KNOWN_CONTAINERS.get(container_hash, container_hash)
        
        values = [float(v[1]) / 1024 / 1024 for v in result['values']] # MB
        
        if name in containers:
            # Filter out 0 memory usage (container down) for mean calculation
            active_values = [v for v in values if v > 1.0]
            
            containers[name].update({
                'mem_mean': round(np.mean(active_values), 1) if active_values else 0,
                'mem_max': round(np.max(values), 1),
                'mem_min': round(np.min(values), 1),
                'mem_std': round(np.std(values), 1),
                'mem_current': round(values[-1], 1) if values else 0
            })
    
    # Fallback for containers with no data (avoid key errors)
    for c in containers.values():
        if 'cpu_mean' not in c: c['cpu_mean'] = 0
        if 'mem_mean' not in c: c['mem_mean'] = 0
        
    metrics = {
        "extracted_at": datetime.utcnow().isoformat(),
        "window_hours": 6,
        "containers": list(containers.values()),
        "total_containers": len(containers),
    }
    
    with open(OUTPUT_DIR / 'container_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ container_metrics.json ({len(containers)} containers, time-series analysis)")
    return containers

def extract_api_metrics():
    """Extract API performance metrics."""
    # Request rate
    request_rate = query_prometheus('sum(rate(http_requests_total[5m]))')
    
    # Request duration (histogram)
    latency_p50 = query_prometheus('histogram_quantile(0.5, rate(http_request_duration_seconds_bucket[5m]))')
    latency_p95 = query_prometheus('histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))')
    latency_p99 = query_prometheus('histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))')
    
    # Error rate
    error_rate = query_prometheus('sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))')
    
    def safe_float(val):
        """Convert to float, returning None for NaN/Inf."""
        try:
            v = float(val)
            return v if not (np.isnan(v) or np.isinf(v)) else None
        except (TypeError, ValueError):
            return None
    
    metrics = {
        "extracted_at": datetime.utcnow().isoformat(),
        "request_rate_per_sec": safe_float(request_rate[0]['value'][1]) if request_rate else None,
        "latency_p50_ms": round(safe_float(latency_p50[0]['value'][1]) * 1000, 2) if latency_p50 and safe_float(latency_p50[0]['value'][1]) is not None else None,
        "latency_p95_ms": round(safe_float(latency_p95[0]['value'][1]) * 1000, 2) if latency_p95 and safe_float(latency_p95[0]['value'][1]) is not None else None,
        "latency_p99_ms": round(safe_float(latency_p99[0]['value'][1]) * 1000, 2) if latency_p99 and safe_float(latency_p99[0]['value'][1]) is not None else None,
        "error_rate": safe_float(error_rate[0]['value'][1]) if error_rate else 0,
    }
    
    with open(OUTPUT_DIR / 'api_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ api_metrics.json")
    return metrics

def extract_prometheus_targets():
    """Extract Prometheus targets status."""
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=10)
        targets = r.json().get('data', {}).get('activeTargets', [])
        
        data = []
        for t in targets:
            data.append({
                "job": t.get('labels', {}).get('job', ''),
                "instance": t.get('labels', {}).get('instance', ''),
                "health": t.get('health', ''),
                "last_scrape": t.get('lastScrape', ''),
                "scrape_duration_ms": round(float(t.get('lastScrapeDuration', 0)) * 1000, 2),
            })
        
        with open(OUTPUT_DIR / 'prometheus_targets.json', 'w') as f:
            json.dump({"targets": data, "total": len(data)}, f, indent=2)
        print(f"  ✅ prometheus_targets.json ({len(data)} targets)")
        return data
    except Exception as e:
        print(f"  ⚠️  {e}")
        return []

def generate_infrastructure_viz(containers, api_metrics, targets):
    """Generate infrastructure visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Container CPU usage
    # Plot 1: Container CPU usage (Mean + Max shadow)
    if containers:
        sorted_containers = sorted(containers.values(), key=lambda x: x.get('cpu_mean', 0), reverse=True)
        names = [c.get('name', '?')[:20] for c in sorted_containers[:15]]
        cpu_means = [c.get('cpu_mean', 0) for c in sorted_containers[:15]]
        cpu_maxs = [c.get('cpu_max', 0) for c in sorted_containers[:15]]
        
        # Plot max as light shadow, mean as main bar
        axes[0, 0].barh(names[::-1], cpu_maxs[::-1], color=COLORS[0], alpha=0.3, label='Max (6h)')
        axes[0, 0].barh(names[::-1], cpu_means[::-1], color=COLORS[0], alpha=0.85, label='Mean (6h)')
        
        axes[0, 0].set_xlabel('CPU Usage (%)')
        axes[0, 0].set_title('Container CPU Utilization (Mean vs Max)')
        axes[0, 0].legend()
    
    # Plot 2: Container memory usage
    # Plot 2: Container memory usage (Mean)
    if containers:
        # Use existing sorted order or re-sort by memory
        names = [c.get('name', '?')[:20] for c in sorted_containers[:15]]
        mem_means = [c.get('mem_mean', 0) for c in sorted_containers[:15]]
        mem_maxs = [c.get('mem_max', 0) for c in sorted_containers[:15]]
        
        axes[0, 1].barh(names[::-1], mem_maxs[::-1], color=COLORS[1], alpha=0.3, label='Max (6h)')
        axes[0, 1].barh(names[::-1], mem_means[::-1], color=COLORS[1], alpha=0.85, label='Mean (6h)')
        
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
    
    # Plot 4: System overview text
    axes[1, 1].axis('off')
    info_text = "System Infrastructure Summary\n" + "=" * 35 + "\n\n"
    info_text += f"Containers Monitored: {len(containers)}\n"
    info_text += f"Prometheus Targets: {len(targets)}\n"
    info_text += f"Targets Healthy: {sum(1 for t in targets if t.get('health') == 'up')}\n\n"
    
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
    
    plt.suptitle(f'Infrastructure & Monitoring: Docker Compose Deployment\n({len(containers)}-Service Architecture)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fig_infrastructure.png', bbox_inches='tight')
    plt.close()
    print("  ✅ fig_infrastructure.png")

def generate_container_table(containers):
    """Generate container resource utilization table."""
def generate_container_table(containers):
    """Generate container resource utilization table."""
    headers = ['Container', 'CPU Mean (%)', 'CPU Max', 'Mem Mean (MB)', 'Mem Max', 'Role']
    
    container_roles = {
        'agentic_api': 'FastAPI Backend',
        'agentic_worker': 'Task Queue Worker',
        'agentic_dashboard': 'Streamlit Frontend',
        'agentic_postgres': 'PostgreSQL Database',
        'agentic_redis': 'Redis Cache + Queue',
        'agentic_ollama': 'Ollama LLM Runtime',
        'agentic_prometheus': 'Metrics Collection',
        'agentic_grafana': 'Metrics Dashboard',
        'agentic_loki': 'Log Aggregation',
        'agentic_promtail': 'Log Shipping',
        'agentic_cadvisor': 'Container Metrics',
        'agentic_mlflow': 'Experiment Tracking',
        'agentic_chromadb': 'Vector Store (RAG)',
        'agentic_nginx': 'Reverse Proxy',
        'agentic_node_exporter': 'Host Metrics',
        'agentic_alertmanager': 'Alert Management',
        'agentic_postgres_exporter': 'DB Metrics Export',
    }
    
    rows = []
    for c in sorted(containers.values(), key=lambda x: x.get('cpu_mean', 0), reverse=True):
        name = c.get('name', '?')
        role = container_roles.get(name, 'Service')
        rows.append([
            name, 
            f"{c.get('cpu_mean', 0):.2f}", 
            f"{c.get('cpu_max', 0):.2f}",
            f"{c.get('mem_mean', 0):.0f}", 
            f"{c.get('mem_max', 0):.0f}",
            role
        ])
    
    with open(TABLE_DIR / 'container_resources.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  ✅ container_resources.csv ({len(rows)} containers)")


if __name__ == "__main__":
    print("=" * 60)
    print("THESIS RESEARCH: Infrastructure Metrics")
    print("=" * 60)
    
    print("\n📊 Extracting container metrics (Time-Series)...")
    containers = extract_time_series_metrics()
    
    print("\n📊 Extracting API metrics...")
    api_metrics = extract_api_metrics()
    
    print("\n📊 Extracting Prometheus targets...")
    targets = extract_prometheus_targets()
    
    print("\n📊 Generating infrastructure visualization...")
    generate_infrastructure_viz(containers, api_metrics, targets)
    
    print("\n📊 Generating container table...")
    generate_container_table(containers)
    
    print("\n" + "=" * 60)
    print("✅ Infrastructure metrics extraction complete!")
    print("=" * 60)
