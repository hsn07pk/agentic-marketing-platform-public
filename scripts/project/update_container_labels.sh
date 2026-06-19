#!/bin/bash
# Regenerates Prometheus cadvisor metric_relabel_configs
# Run this after docker compose up to update container ID mappings
set -e

PROM_CONFIG="monitoring/prometheus/prometheus.yml"
cd "$(dirname "$0")/../.."

python3 -c "
import subprocess, yaml

result = subprocess.run(['docker', 'ps', '--no-trunc', '--format', '{{.ID}}:{{.Names}}'], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

relabel_configs = []
for line in lines:
    cid, name = line.split(':')
    short = cid[:12]
    relabel_configs.append({
        'source_labels': ['id'],
        'regex': f'.*docker-{short}.*',
        'target_label': 'name',
        'replacement': name
    })

with open('$PROM_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

for job in config['scrape_configs']:
    if job['job_name'] == 'cadvisor':
        job['metric_relabel_configs'] = relabel_configs
        break

with open('$PROM_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f'Updated {len(relabel_configs)} container relabel configs')
"

# Reload Prometheus config
curl -s -X POST http://localhost:9090/-/reload && echo "Prometheus config reloaded"
