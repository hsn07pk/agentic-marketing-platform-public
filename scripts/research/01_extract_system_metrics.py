#!/usr/bin/env python3
"""
Extract comprehensive system metrics for thesis research.
Outputs: JSON data files + CSV tables for all KPIs defined in the research plan.
"""
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, '/app' if os.path.exists('/app/src') else str(Path(__file__).parent.parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_db_session():
    from src.data_layer.database.connection import sync_session_maker
    return sync_session_maker()

def extract_campaign_metrics():
    """Extract all campaign data for funnel analysis."""
    from src.data_layer.database.models import Campaign, Content
    from sqlalchemy import select, func
    
    with get_db_session() as session:
        campaigns = session.execute(select(Campaign).order_by(Campaign.created_at)).scalars().all()
        
        data = []
        for c in campaigns:
            # Count content items
            content_count = session.execute(
                select(func.count()).select_from(Content).where(Content.campaign_id == c.id)
            ).scalar() or 0
            
            data.append({
                "id": str(c.id),
                "name": c.name,
                "platform": c.platform.value if c.platform else "unknown",
                "status": c.status.value if c.status else "unknown",
                "persona": c.persona if hasattr(c, 'persona') else None,
                "goal": c.goal if hasattr(c, 'goal') else None,
                "budget_total": float(c.budget_total or 0),
                "budget_spent": float(c.budget_spent or 0),
                "impressions": c.impressions or 0,
                "clicks": c.clicks or 0,
                "conversions": c.conversions or 0,
                "demos_booked": c.demos_booked or 0,
                "ctr": round((c.clicks or 0) / (c.impressions or 1) * 100, 4),
                "conversion_rate": round((c.conversions or 0) / (c.clicks or 1) * 100, 4) if (c.clicks or 0) > 0 else 0,
                "cpl": round((c.budget_spent or 0) / (c.conversions or 1), 2) if (c.conversions or 0) > 0 else None,
                "content_items_generated": content_count,
                "is_mock": c.is_mock if hasattr(c, 'is_mock') else False,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "started_at": c.started_at.isoformat() if hasattr(c, 'started_at') and c.started_at else None,
                "completed_at": c.completed_at.isoformat() if hasattr(c, 'completed_at') and c.completed_at else None,
                "duration_days": c.duration_days if hasattr(c, 'duration_days') else None,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total_campaigns": len(data), "campaigns": data}
        with open(OUTPUT_DIR / 'campaigns.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} campaigns")
        return data

def extract_content_governance():
    """Extract content + governance data for safety analysis."""
    from src.data_layer.database.models import Content
    from sqlalchemy import select
    
    with get_db_session() as session:
        contents = session.execute(select(Content).order_by(Content.created_at)).scalars().all()
        
        data = []
        for c in contents:
            data.append({
                "id": str(c.id),
                "campaign_id": str(c.campaign_id) if c.campaign_id else None,
                "platform": c.platform.value if hasattr(c, 'platform') and c.platform else None,
                "status": c.status.value if hasattr(c, 'status') and c.status else None,
                "safety_score": float(c.safety_score) if hasattr(c, 'safety_score') and c.safety_score is not None else None,
                "toxicity_score": float(c.toxicity_score) if hasattr(c, 'toxicity_score') and c.toxicity_score is not None else None,
                "factuality_score": float(c.factuality_score) if hasattr(c, 'factuality_score') and c.factuality_score is not None else None,
                "brand_alignment_score": float(c.brand_alignment_score) if hasattr(c, 'brand_alignment_score') and c.brand_alignment_score is not None else None,
                "review_status": c.status.value if hasattr(c, 'status') and c.status else None,
                "reviewer": c.reviewed_by if hasattr(c, 'reviewed_by') else None,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "reviewed_at": c.reviewed_at.isoformat() if hasattr(c, 'reviewed_at') and c.reviewed_at else None,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total_content": len(data), "content_items": data}
        with open(OUTPUT_DIR / 'content_governance.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} content items")
        return data

def extract_workflow_events():
    """Extract workflow events for transparency and audit trail analysis."""
    from src.data_layer.database.models import WorkflowEvent
    from sqlalchemy import select
    
    with get_db_session() as session:
        events = session.execute(
            select(WorkflowEvent).order_by(WorkflowEvent.created_at).limit(5000)
        ).scalars().all()
        
        data = []
        for e in events:
            data.append({
                "id": str(e.id),
                "campaign_id": str(e.campaign_id) if e.campaign_id else None,
                "event_type": e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type) if e.event_type else None,
                "workflow_node": e.workflow_node if hasattr(e, 'workflow_node') else None,
                "severity": e.severity.value if hasattr(e, 'severity') and hasattr(e.severity, 'value') else str(e.severity) if hasattr(e, 'severity') else None,
                "title": e.title if hasattr(e, 'title') else None,
                "message": e.message if hasattr(e, 'message') else None,
                "details": e.details if hasattr(e, 'details') else None,
                "is_dismissed": e.is_dismissed if hasattr(e, 'is_dismissed') else None,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total_events": len(data), "events": data}
        with open(OUTPUT_DIR / 'workflow_events.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} workflow events")
        return data

def extract_delayed_rewards():
    """Extract delayed reward / funnel tracking data."""
    from src.data_layer.database.models import DelayedReward, Campaign
    from sqlalchemy import select
    
    with get_db_session() as session:
        rewards = session.execute(select(DelayedReward).order_by(DelayedReward.registered_at)).scalars().all()
        
        # Get campaign names
        campaign_names = {}
        campaigns = session.execute(select(Campaign)).scalars().all()
        for c in campaigns:
            campaign_names[c.id] = c.name
        
        data = []
        for r in rewards:
            data.append({
                "id": str(r.id),
                "campaign_id": str(r.campaign_id),
                "campaign_name": campaign_names.get(r.campaign_id, "Unknown"),
                "lead_email": r.lead_email,
                "status": r.status,
                "initial_reward": float(r.initial_reward) if r.initial_reward else 1.0,
                "current_reward": float(r.current_reward) if r.current_reward else 1.0,
                "meeting_scheduled": r.meeting_scheduled,
                "meeting_attended": r.meeting_attended if hasattr(r, 'meeting_attended') else None,
                "lead_score": r.lead_score if hasattr(r, 'lead_score') else None,
                "registered_at": r.registered_at.isoformat() if r.registered_at else None,
                "resolved_at": r.resolved_at.isoformat() if hasattr(r, 'resolved_at') and r.resolved_at else None,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total_rewards": len(data), "rewards": data}
        with open(OUTPUT_DIR / 'delayed_rewards.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} delayed rewards")
        return data

def extract_experiments():
    """Extract A/B test / bandit experiment data."""
    from src.data_layer.database.models import Experiment, BanditArm
    from sqlalchemy import select
    
    with get_db_session() as session:
        experiments = session.execute(select(Experiment).order_by(Experiment.started_at)).scalars().all()
        
        data = []
        for e in experiments:
            arms = session.execute(
                select(BanditArm).where(BanditArm.experiment_id == e.id)
            ).scalars().all()
            
            arm_data = []
            for a in arms:
                arm_data.append({
                    "id": str(a.id),
                    "arm_id": a.arm_id,
                    "alpha": float(a.alpha) if a.alpha else 1.0,
                    "beta": float(a.beta) if a.beta else 1.0,
                    "pulls": a.pulls or 0,
                    "reward": float(a.reward) if hasattr(a, 'reward') and a.reward else 0,
                })
            
            data.append({
                "id": str(e.id),
                "name": e.name,
                "type": e.type if hasattr(e, 'type') else None,
                "algorithm": e.algorithm if hasattr(e, 'algorithm') else None,
                "is_active": e.is_active if hasattr(e, 'is_active') else None,
                "winner_variant": e.winner_variant if hasattr(e, 'winner_variant') else None,
                "total_impressions": e.total_impressions if hasattr(e, 'total_impressions') else 0,
                "total_conversions": e.total_conversions if hasattr(e, 'total_conversions') else 0,
                "variants": e.variants if hasattr(e, 'variants') else None,
                "results": e.results if hasattr(e, 'results') else None,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "ended_at": e.ended_at.isoformat() if hasattr(e, 'ended_at') and e.ended_at else None,
                "arms": arm_data,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total_experiments": len(data), "experiments": data}
        with open(OUTPUT_DIR / 'experiments.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} experiments")
        return data

def extract_agent_memory():
    """Extract agent action data + supplement with workflow events for richer analysis."""
    from src.data_layer.database.models import AgentAction, WorkflowEvent
    from sqlalchemy import select, func
    
    with get_db_session() as session:
        actions = session.execute(select(AgentAction).order_by(AgentAction.started_at)).scalars().all()
        
        data = []
        for a in actions:
            output = a.output_data if hasattr(a, 'output_data') else {}
            data.append({
                "id": str(a.id),
                "agent_type": a.agent_type.value if hasattr(a.agent_type, 'value') else str(a.agent_type),
                "action": a.action,
                "campaign_id": str(a.campaign_id) if a.campaign_id else None,
                "outcome": output.get('status', 'unknown') if isinstance(output, dict) else 'unknown',

                "duration_seconds": (
                    float(output.get('duration', 0)) 
                    if isinstance(output, dict) and output.get('duration') 
                    else float(a.duration_ms) / 1000.0 if hasattr(a, 'duration_ms') and a.duration_ms 
                    else 0.0
                ),
                "duration_ms": a.duration_ms if hasattr(a, 'duration_ms') else None,
                "tokens_used": a.tokens_used if hasattr(a, 'tokens_used') else 0,
                "api_cost": float(a.api_cost) if hasattr(a, 'api_cost') and a.api_cost else 0,
                "safety_score": float(output.get('safety_score', 0)) if isinstance(output, dict) and output.get('safety_score') else None,
                "created_at": a.started_at.isoformat() if a.started_at else None,
                "source": "agent_action",
            })
        
        # Supplement with workflow events to build richer agent activity picture
        events = session.execute(
            select(WorkflowEvent).order_by(WorkflowEvent.created_at)
        ).scalars().all()
        
        # Map workflow_node to agent types
        node_to_agent = {
            'content_generator': 'content_generator',
            'safety_checker': 'safety_checker',
            'strategy_optimizer': 'strategy_optimizer',
            'campaign_manager': 'campaign_manager',
            'supervisor': 'supervisor',
            'governance': 'governance',
            'deployment': 'deployment',
        }
        
        # Build agent activity from workflow events
        # Also compute durations by pairing node_started/node_completed events
        pending_starts = {}  # key: (campaign_id, workflow_node) -> started_at timestamp
        agent_events = []
        for e in events:
            event_type_str = e.event_type.value if hasattr(e.event_type, 'value') else str(e.event_type)
            node = e.workflow_node if hasattr(e, 'workflow_node') else None
            severity = e.severity.value if hasattr(e.severity, 'value') else str(e.severity)

            # Determine agent type from event
            agent_type = None
            if node:
                for key, agent in node_to_agent.items():
                    if key in str(node).lower():
                        agent_type = agent
                        break
            if not agent_type:
                if 'content' in event_type_str.lower():
                    agent_type = 'content_generator'
                elif 'safety' in event_type_str.lower():
                    agent_type = 'safety_checker'
                elif 'deploy' in event_type_str.lower():
                    agent_type = 'deployment'
                elif 'governance' in event_type_str.lower() or 'hitl' in event_type_str.lower():
                    agent_type = 'governance'
                elif 'workflow' in event_type_str.lower():
                    agent_type = 'supervisor'
                else:
                    agent_type = 'system'

            # Determine outcome from severity
            if severity == 'error':
                outcome = 'failure'
            elif 'completed' in event_type_str.lower() or 'approved' in event_type_str.lower() or 'passed' in event_type_str.lower():
                outcome = 'success'
            elif 'started' in event_type_str.lower() or 'pending' in event_type_str.lower():
                outcome = 'in_progress'
            elif 'rejected' in event_type_str.lower() or 'failed' in event_type_str.lower():
                outcome = 'failure'
            else:
                outcome = 'success'

            # Compute duration by pairing started/completed events
            duration_seconds = 0.0
            campaign_id = str(e.campaign_id) if e.campaign_id else None
            pair_key = (campaign_id, node)

            if 'started' in event_type_str.lower() and e.created_at:
                pending_starts[pair_key] = e.created_at
            elif 'completed' in event_type_str.lower() and e.created_at and pair_key in pending_starts:
                start_time = pending_starts.pop(pair_key)
                duration_seconds = (e.created_at - start_time).total_seconds()

            # Also extract duration from event details if available
            if not duration_seconds and hasattr(e, 'details') and isinstance(e.details, dict):
                duration_seconds = float(e.details.get('duration', 0))

            agent_events.append({
                "agent_type": agent_type,
                "event_type": event_type_str,
                "outcome": outcome,
                "duration_seconds": duration_seconds,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "source": "workflow_event",
            })
        
        # Aggregate stats by agent type (combining both sources)
        agent_stats = {}
        
        # From direct agent actions
        for a in data:
            agent = a.get("agent_type", "unknown")
            if agent not in agent_stats:
                agent_stats[agent] = {"total": 0, "success": 0, "failure": 0, "durations": [], "source": "agent_action"}
            agent_stats[agent]["total"] += 1
            if a.get("outcome") in ("success", "completed"):
                agent_stats[agent]["success"] += 1
            elif a.get("outcome") in ("failure", "error", "failed"):
                agent_stats[agent]["failure"] += 1
            if a.get("duration_seconds"):
                agent_stats[agent]["durations"].append(a["duration_seconds"])
        
        # From workflow events
        for ae in agent_events:
            agent = ae["agent_type"]
            if agent not in agent_stats:
                agent_stats[agent] = {"total": 0, "success": 0, "failure": 0, "durations": [], "source": "workflow_event"}
            agent_stats[agent]["total"] += 1
            if ae["outcome"] == "success":
                agent_stats[agent]["success"] += 1
            elif ae["outcome"] == "failure":
                agent_stats[agent]["failure"] += 1
            if ae.get("duration_seconds", 0) > 0:
                agent_stats[agent]["durations"].append(ae["duration_seconds"])
        
        for agent, stats in agent_stats.items():
            stats["success_rate"] = round(stats["success"] / stats["total"] * 100, 2) if stats["total"] > 0 else 0
            stats["avg_duration"] = round(sum(stats["durations"]) / len(stats["durations"]), 2) if stats["durations"] else 0
            del stats["durations"]
        
        # Build combined memory list for learning curve
        all_memories = data + [
            {"agent_type": ae["agent_type"], "outcome": ae["outcome"], "created_at": ae["created_at"], "source": "workflow_event"}
            for ae in agent_events
        ]
        all_memories.sort(key=lambda x: x.get("created_at", ""))
        
        output = {
            "extracted_at": datetime.utcnow().isoformat(),
            "total_memories": len(all_memories),
            "total_agent_actions": len(data),
            "total_workflow_events_mapped": len(agent_events),
            "agent_stats": agent_stats,
            "memories": all_memories
        }
        with open(OUTPUT_DIR / 'agent_memory.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} agent actions + {len(agent_events)} workflow events → {len(agent_stats)} agent types")

def extract_bandit_decisions():
    """Extract bandit arm data for learning analysis."""
    from src.data_layer.database.models import BanditArm, Experiment
    from sqlalchemy import select
    
    with get_db_session() as session:
        arms = session.execute(
            select(BanditArm).order_by(BanditArm.created_at).limit(10000)
        ).scalars().all()
        
        data = []
        for a in arms:
            data.append({
                "id": str(a.id),
                "experiment_id": str(a.experiment_id) if a.experiment_id else None,
                "arm_id": a.arm_id,
                "alpha": float(a.alpha) if a.alpha else 1.0,
                "beta": float(a.beta) if a.beta else 1.0,
                "pulls": a.pulls or 0,
                "reward": float(a.reward) if hasattr(a, 'reward') and a.reward is not None else 0,
                "expected_reward": float(a.alpha / (a.alpha + a.beta)) if a.alpha and a.beta and (a.alpha + a.beta) > 0 else 0.5,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total_decisions": len(data), "decisions": data}
        with open(OUTPUT_DIR / 'bandit_decisions.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} bandit arms")
        return data

def extract_system_config():
    """Extract system configuration for architecture documentation."""
    from src.data_layer.database.models import SystemConfiguration
    from sqlalchemy import select
    
    with get_db_session() as session:
        configs = session.execute(select(SystemConfiguration)).scalars().all()
        
        data = []
        for c in configs:
            data.append({
                "key": c.key,
                "category": c.category.value if hasattr(c.category, 'value') else str(c.category),
                "description": c.description if hasattr(c, 'description') else None,
                "value_type": c.value_type if hasattr(c, 'value_type') else "string",
                "is_secret": c.is_secret,
                "has_value": bool(c.value),
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total_configs": len(data), "configurations": data}
        with open(OUTPUT_DIR / 'system_config.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} configuration entries")

def extract_canary_deployments():
    """Extract canary deployment history."""
    from src.data_layer.database.models import CanaryDeployment
    from sqlalchemy import select
    
    with get_db_session() as session:
        deployments = session.execute(select(CanaryDeployment).order_by(CanaryDeployment.created_at)).scalars().all()
        
        data = []
        for d in deployments:
            data.append({
                "id": str(d.id) if hasattr(d, 'id') else None,
                "deployment_id": d.deployment_id if hasattr(d, 'deployment_id') else None,
                "policy_id": d.policy_id if hasattr(d, 'policy_id') else None,
                "status": d.status if hasattr(d, 'status') else None,
                "current_traffic_percentage": float(d.current_traffic_percentage) if hasattr(d, 'current_traffic_percentage') and d.current_traffic_percentage else None,
                "created_at": d.created_at.isoformat() if hasattr(d, 'created_at') and d.created_at else None,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total": len(data), "deployments": data}
        with open(OUTPUT_DIR / 'canary_deployments.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} canary deployments")

def extract_cost_tracking():
    """Extract cost tracking data for cost efficiency analysis."""
    from src.data_layer.database.models import CostTracking
    from sqlalchemy import select
    
    with get_db_session() as session:
        costs = session.execute(select(CostTracking).order_by(CostTracking.timestamp)).scalars().all()
        
        data = []
        for c in costs:
            data.append({
                "id": str(c.id),
                "campaign_id": str(c.campaign_id) if c.campaign_id else None,
                "source_type": c.source_type if hasattr(c, 'source_type') else None,
                "provider": c.provider if hasattr(c, 'provider') else None,
                "agent_type": c.agent_type if hasattr(c, 'agent_type') else None,
                "amount": float(c.cost_amount) if hasattr(c, 'cost_amount') and c.cost_amount else 0,
                "currency": c.cost_currency if hasattr(c, 'cost_currency') else "EUR",
                "tokens_prompt": c.tokens_prompt if hasattr(c, 'tokens_prompt') else 0,
                "tokens_completion": c.tokens_completion if hasattr(c, 'tokens_completion') else 0,
                "timestamp": c.timestamp.isoformat() if c.timestamp else None,
            })
        
        total_cost = sum(c['amount'] for c in data)
        total_tokens = sum((c.get('tokens_prompt', 0) or 0) + (c.get('tokens_completion', 0) or 0) for c in data)
        cached_count = sum(1 for c in data if c.get('cached'))
        
        # TCO Calculation
        # Assumption: Fixed infrastructure cost of €50/month (VPS 16GB RAM)
        # Period covers ~60 days (Dec-Jan) based on data, but we'll normalize to monthly or daily if needed.
        # For this summary, we'll estimate the relevant period from the timestamps.
        
        if data:
            timestamps = [datetime.fromisoformat(c['timestamp']) for c in data if c['timestamp']]
            if timestamps:
                min_ts = min(timestamps)
                max_ts = max(timestamps)
                days_active = max(1, (max_ts - min_ts).days)
            else:
                days_active = 30
        else:
            days_active = 30
            
        daily_infra_cost = 50.0 / 30.0
        total_infra_cost = daily_infra_cost * days_active
        total_tco = total_cost + total_infra_cost

        output = {
            "extracted_at": datetime.utcnow().isoformat(),
            "period_days": days_active,
            "total_records": len(data),
            "total_llm_cost": round(total_cost, 4),
            "total_infra_cost_est": round(total_infra_cost, 4),
            "total_tco": round(total_tco, 4),
            "total_tokens": total_tokens,
            "cache_hit_count": cached_count,
            "cache_hit_rate": round(cached_count / len(data) * 100, 2) if data else 0,
            "records": data
        }
        with open(OUTPUT_DIR / 'cost_tracking.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} cost records (total: ${total_cost:.4f}, {total_tokens} tokens)")

def extract_hitl_queue():
    """Extract HITL queue data for governance analysis."""
    from src.data_layer.database.models import HITLQueue
    from sqlalchemy import select
    
    with get_db_session() as session:
        items = session.execute(select(HITLQueue).order_by(HITLQueue.created_at)).scalars().all()
        
        data = []
        for h in items:
            data.append({
                "id": str(h.id),
                "content_id": str(h.content_id) if h.content_id else None,
                "priority": h.priority if hasattr(h, 'priority') else 0,
                "reason": h.reason if hasattr(h, 'reason') else None,
                "status": h.status if hasattr(h, 'status') else None,
                "decision": h.decision if hasattr(h, 'decision') else None,
                "assigned_to": h.assigned_to if hasattr(h, 'assigned_to') else None,
                "feedback": h.feedback if hasattr(h, 'feedback') else None,
                "created_at": h.created_at.isoformat() if h.created_at else None,
                "completed_at": h.completed_at.isoformat() if hasattr(h, 'completed_at') and h.completed_at else None,
            })
        
        # Decision distribution
        decisions = {}
        for h in data:
            d = h.get('decision', 'unknown') or 'pending'
            decisions[d] = decisions.get(d, 0) + 1
        
        output = {
            "extracted_at": datetime.utcnow().isoformat(),
            "total_items": len(data),
            "decision_distribution": decisions,
            "items": data
        }
        with open(OUTPUT_DIR / 'hitl_queue.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} HITL items (decisions: {decisions})")

def extract_governance_metrics():
    """Extract governance metrics for compliance analysis."""
    from src.data_layer.database.models import GovernanceMetrics
    from sqlalchemy import select
    
    with get_db_session() as session:
        metrics = session.execute(select(GovernanceMetrics).order_by(GovernanceMetrics.created_at)).scalars().all()
        
        data = []
        for m in metrics:
            data.append({
                "id": str(m.id),
                "metric_type": m.metric_type if hasattr(m, 'metric_type') else None,
                "value": float(m.value) if hasattr(m, 'value') and m.value else 0,
                "details": m.details if hasattr(m, 'details') else None,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            })
        
        output = {"extracted_at": datetime.utcnow().isoformat(), "total": len(data), "metrics": data}
        with open(OUTPUT_DIR / 'governance_metrics.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Extracted {len(data)} governance metric records")

if __name__ == "__main__":
    print("=" * 60)
    print("THESIS RESEARCH: System Metrics Extraction")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    extractors = [
        ("Campaign Metrics", extract_campaign_metrics),
        ("Content & Governance", extract_content_governance),
        ("Workflow Events", extract_workflow_events),
        ("Delayed Rewards", extract_delayed_rewards),
        ("Experiments", extract_experiments),
        ("Agent Actions", extract_agent_memory),
        ("Bandit Arms", extract_bandit_decisions),
        ("System Config", extract_system_config),
        ("Canary Deployments", extract_canary_deployments),
        ("Cost Tracking", extract_cost_tracking),
        ("HITL Queue", extract_hitl_queue),
        ("Governance Metrics", extract_governance_metrics),
    ]
    
    for name, extractor in extractors:
        try:
            print(f"\n📊 Extracting: {name}...")
            extractor()
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Extraction complete!")
    print(f"Files written to: {OUTPUT_DIR}")
    print("=" * 60)
