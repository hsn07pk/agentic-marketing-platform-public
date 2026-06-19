#!/usr/bin/env python3
"""
Comprehensive seed data script for dashboard testing.

Creates:
1. Simulation Live Accuracy data (MAPE gauge)
2. Workflow Events (System Transparency)
3. Governance Metrics (Override rate tracking)
4. Weekly Learning Reports

Run with: python scripts/seed_data.py
Or via Docker: docker-compose run --rm api python scripts/seed_data.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import uuid
import random
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database URL from environment or default
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://agentic:agentic_dev@localhost:5432/agentic_dev'
)


def get_session():
    """Get database session"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()


def get_campaign_ids(session) -> list:
    """Get existing campaign IDs"""
    result = session.execute(text("SELECT id FROM campaigns LIMIT 20"))
    return [str(row[0]) for row in result.fetchall()]


def seed_simulation_accuracy(session, campaign_ids: list):
    """
    Seed simulation_live_accuracy table for MAPE gauge display.

    Creates accuracy records with realistic simulation vs actual metrics.
    Target: MAPE < 10% (>90% accuracy) per research plan RQ2.
    """
    logger.info("Seeding simulation_live_accuracy data...")

    # Check if table exists
    try:
        session.execute(text("SELECT 1 FROM simulation_live_accuracy LIMIT 1"))
    except Exception as e:
        logger.warning(f"simulation_live_accuracy table doesn't exist. Run: alembic upgrade head")
        logger.warning(f"Error: {e}")
        return 0

    count = 0
    now = datetime.utcnow()

    for campaign_id in campaign_ids[:10]:  # Use first 10 campaigns
        # Create 5 accuracy measurements per campaign over past 30 days
        for i in range(5):
            days_ago = random.randint(1, 30)
            measurement_time = now - timedelta(days=days_ago)

            # Generate realistic simulated values
            sim_impressions = random.randint(500, 5000)
            sim_clicks = int(sim_impressions * random.uniform(0.01, 0.05))
            sim_conversions = int(sim_clicks * random.uniform(0.05, 0.15))
            sim_ctr = (sim_clicks / sim_impressions * 100) if sim_impressions > 0 else 0
            sim_cpl = random.uniform(5, 25)

            # Generate actual values with some variance (target MAPE < 10%)
            # Variance factor between 0.92 and 1.08 gives roughly 8% MAPE
            variance = random.uniform(0.92, 1.08)

            actual_impressions = int(sim_impressions * variance)
            actual_clicks = int(sim_clicks * variance)
            actual_conversions = int(sim_conversions * variance)
            actual_ctr = (actual_clicks / actual_impressions * 100) if actual_impressions > 0 else 0
            actual_cpl = sim_cpl * variance

            # Calculate MAPE for each metric
            def calc_mape(simulated, actual):
                if actual == 0:
                    return 0
                return abs(simulated - actual) / actual * 100

            mape_impressions = calc_mape(sim_impressions, actual_impressions)
            mape_clicks = calc_mape(sim_clicks, actual_clicks)
            mape_conversions = calc_mape(sim_conversions, actual_conversions)
            mape_ctr = calc_mape(sim_ctr, actual_ctr)
            mape_cpl = calc_mape(sim_cpl, actual_cpl)

            # Overall MAPE (average of all metrics)
            overall_mape = (mape_impressions + mape_clicks + mape_conversions + mape_ctr + mape_cpl) / 5
            overall_accuracy = 100 - overall_mape
            passes_threshold = overall_mape < 10  # Target: <10% MAPE

            record_id = str(uuid.uuid4())

            session.execute(text("""
                INSERT INTO simulation_live_accuracy (
                    id, campaign_id,
                    simulated_impressions, simulated_clicks, simulated_conversions,
                    simulated_ctr, simulated_cpl,
                    actual_impressions, actual_clicks, actual_conversions,
                    actual_ctr, actual_cpl,
                    mape_impressions, mape_clicks, mape_conversions, mape_ctr, mape_cpl,
                    overall_mape, overall_accuracy, passes_threshold,
                    rq2_target, rq2_gap,
                    simulation_timestamp, measurement_timestamp, measurement_type,
                    created_at
                ) VALUES (
                    :id, :campaign_id,
                    :sim_imp, :sim_clicks, :sim_conv, :sim_ctr, :sim_cpl,
                    :act_imp, :act_clicks, :act_conv, :act_ctr, :act_cpl,
                    :mape_imp, :mape_clicks, :mape_conv, :mape_ctr, :mape_cpl,
                    :overall_mape, :overall_acc, :passes,
                    90.0, :rq2_gap,
                    :sim_ts, :meas_ts, 'daily',
                    :created_at
                )
                ON CONFLICT DO NOTHING
            """), {
                'id': record_id,
                'campaign_id': campaign_id,
                'sim_imp': sim_impressions,
                'sim_clicks': sim_clicks,
                'sim_conv': sim_conversions,
                'sim_ctr': sim_ctr,
                'sim_cpl': sim_cpl,
                'act_imp': actual_impressions,
                'act_clicks': actual_clicks,
                'act_conv': actual_conversions,
                'act_ctr': actual_ctr,
                'act_cpl': actual_cpl,
                'mape_imp': mape_impressions,
                'mape_clicks': mape_clicks,
                'mape_conv': mape_conversions,
                'mape_ctr': mape_ctr,
                'mape_cpl': mape_cpl,
                'overall_mape': overall_mape,
                'overall_acc': overall_accuracy,
                'passes': passes_threshold,
                'rq2_gap': overall_accuracy - 90,
                'sim_ts': measurement_time - timedelta(days=1),
                'meas_ts': measurement_time,
                'created_at': measurement_time
            })
            count += 1

    session.commit()
    logger.info(f"Created {count} simulation accuracy records")
    return count


def seed_workflow_events(session, campaign_ids: list):
    """
    Seed workflow_events table for System Transparency page.

    Creates realistic workflow events for campaigns.
    """
    logger.info("Seeding workflow_events data...")

    count = 0
    now = datetime.utcnow()

    event_templates = [
        {
            'event_type': 'WORKFLOW_STARTED',
            'severity': 'INFO',
            'title': 'Campaign workflow started',
            'message': 'Initiating content generation and optimization workflow',
            'workflow_node': 'orchestrator',
            'workflow_state': 'running',
            'is_user_actionable': False
        },
        {
            'event_type': 'CONTENT_GENERATED',
            'severity': 'INFO',
            'title': 'Content generated successfully',
            'message': 'AI agent generated marketing content ready for review',
            'workflow_node': 'content_generator',
            'workflow_state': 'running',
            'is_user_actionable': False
        },
        {
            'event_type': 'SAFETY_CHECK_PASSED',
            'severity': 'INFO',
            'title': 'Safety validation passed',
            'message': 'Content passed all safety checks (toxicity, factuality, brand alignment)',
            'workflow_node': 'safety_validator',
            'workflow_state': 'running',
            'is_user_actionable': False
        },
        {
            'event_type': 'HITL_QUEUE_ADDED',
            'severity': 'WARNING',
            'title': 'Content awaiting review',
            'message': 'Content has been added to the human review queue',
            'workflow_node': 'governance',
            'workflow_state': 'paused',
            'is_user_actionable': True
        },
        {
            'event_type': 'CONTENT_APPROVED',
            'severity': 'INFO',
            'title': 'Content approved for deployment',
            'message': 'Human reviewer approved the content',
            'workflow_node': 'governance',
            'workflow_state': 'running',
            'is_user_actionable': False
        },
        {
            'event_type': 'DEPLOYMENT_SUCCESS',
            'severity': 'INFO',
            'title': 'Content deployed successfully',
            'message': 'Content has been published to the target platform',
            'workflow_node': 'deployer',
            'workflow_state': 'completed',
            'is_user_actionable': False
        },
        {
            'event_type': 'BUDGET_WARNING',
            'severity': 'WARNING',
            'title': 'Budget threshold approaching',
            'message': 'Campaign is approaching 80% of allocated budget',
            'workflow_node': 'cost_control',
            'workflow_state': 'running',
            'is_user_actionable': True
        },
        {
            'event_type': 'SAFETY_CHECK_FAILED',
            'severity': 'ERROR',
            'title': 'Safety validation failed',
            'message': 'Content failed toxicity check - requires revision',
            'workflow_node': 'safety_validator',
            'workflow_state': 'paused',
            'is_user_actionable': True
        }
    ]

    for campaign_id in campaign_ids[:8]:  # Use first 8 campaigns
        # Create a sequence of events for each campaign
        event_time = now - timedelta(days=random.randint(1, 14))

        for i, template in enumerate(event_templates[:random.randint(4, 8)]):
            event_id = str(uuid.uuid4())
            event_time += timedelta(minutes=random.randint(5, 60))

            # Add some variation
            details = {
                'campaign_name': f'Campaign {campaign_id[:8]}',
                'step': i + 1,
                'total_steps': len(event_templates)
            }

            if 'SAFETY' in template['event_type']:
                details['safety_score'] = round(random.uniform(0.7, 0.98), 2)
                details['toxicity_score'] = round(random.uniform(0.01, 0.15), 3)

            if 'BUDGET' in template['event_type']:
                details['budget_used_pct'] = round(random.uniform(75, 95), 1)
                details['remaining_budget'] = round(random.uniform(100, 500), 2)

            session.execute(text("""
                INSERT INTO workflow_events (
                    id, campaign_id, event_type, severity,
                    workflow_node, workflow_state,
                    title, message, details,
                    is_user_actionable, is_dismissed, created_at
                ) VALUES (
                    :id, :campaign_id, :event_type, :severity,
                    :workflow_node, :workflow_state,
                    :title, :message, :details::jsonb,
                    :is_user_actionable, false, :created_at
                )
                ON CONFLICT DO NOTHING
            """), {
                'id': event_id,
                'campaign_id': campaign_id,
                'event_type': template['event_type'],
                'severity': template['severity'],
                'workflow_node': template['workflow_node'],
                'workflow_state': template['workflow_state'],
                'title': template['title'],
                'message': template['message'],
                'details': str(details).replace("'", '"'),
                'is_user_actionable': template['is_user_actionable'],
                'created_at': event_time
            })
            count += 1

    session.commit()
    logger.info(f"Created {count} workflow events")
    return count


def seed_governance_metrics(session):
    """
    Seed governance_metrics table for override rate tracking.
    """
    logger.info("Seeding governance_metrics data...")

    # Check if table exists
    try:
        session.execute(text("SELECT 1 FROM governance_metrics LIMIT 1"))
    except Exception:
        logger.warning("governance_metrics table doesn't exist. Run: alembic upgrade head")
        return 0

    count = 0
    now = datetime.utcnow()

    # Create weekly metrics for past 8 weeks
    for week in range(8):
        week_start = now - timedelta(weeks=week+1)
        week_end = week_start + timedelta(days=7)

        total_reviews = random.randint(20, 80)
        approved = int(total_reviews * random.uniform(0.85, 0.98))
        rejected = total_reviews - approved - random.randint(0, 3)
        modified = total_reviews - approved - rejected

        override_rate = (rejected + modified) / total_reviews * 100 if total_reviews > 0 else 0

        record_id = str(uuid.uuid4())

        session.execute(text("""
            INSERT INTO governance_metrics (
                id, period_start, period_end, period_type,
                total_reviews, approved_count, rejected_count, modified_count,
                human_override_rate, override_rate_target, meets_override_target,
                avg_safety_score, avg_toxicity_score, avg_factuality_score, avg_brand_alignment_score,
                golden_test_pass_rate, golden_tests_run, golden_tests_passed,
                auto_approved_count, auto_approval_rate,
                avg_review_time_minutes, median_review_time_minutes,
                created_at
            ) VALUES (
                :id, :period_start, :period_end, 'weekly',
                :total_reviews, :approved, :rejected, :modified,
                :override_rate, 5.0, :meets_target,
                :avg_safety, :avg_toxicity, :avg_factuality, :avg_brand,
                :golden_pass_rate, :golden_run, :golden_passed,
                :auto_approved, :auto_rate,
                :avg_time, :median_time,
                :created_at
            )
            ON CONFLICT DO NOTHING
        """), {
            'id': record_id,
            'period_start': week_start,
            'period_end': week_end,
            'total_reviews': total_reviews,
            'approved': approved,
            'rejected': rejected,
            'modified': modified,
            'override_rate': override_rate,
            'meets_target': override_rate < 5.0,
            'avg_safety': round(random.uniform(0.85, 0.95), 3),
            'avg_toxicity': round(random.uniform(0.02, 0.08), 3),
            'avg_factuality': round(random.uniform(0.88, 0.98), 3),
            'avg_brand': round(random.uniform(0.82, 0.95), 3),
            'golden_pass_rate': round(random.uniform(95, 100), 1),
            'golden_run': 40,
            'golden_passed': random.randint(38, 40),
            'auto_approved': int(approved * random.uniform(0.3, 0.5)),
            'auto_rate': round(random.uniform(30, 50), 1),
            'avg_time': round(random.uniform(2, 8), 1),
            'median_time': round(random.uniform(1.5, 5), 1),
            'created_at': week_end
        })
        count += 1

    session.commit()
    logger.info(f"Created {count} governance metrics records")
    return count


def seed_weekly_reports(session):
    """
    Seed weekly_learning_reports table.
    """
    logger.info("Seeding weekly_learning_reports data...")

    # Check if table exists
    try:
        session.execute(text("SELECT 1 FROM weekly_learning_reports LIMIT 1"))
    except Exception:
        logger.warning("weekly_learning_reports table doesn't exist. Run: alembic upgrade head")
        return 0

    count = 0
    now = datetime.utcnow()

    # Create reports for past 4 weeks
    for week in range(4):
        week_start = now - timedelta(weeks=week+1)
        week_end = week_start + timedelta(days=7)
        week_num = week_start.isocalendar()[1]
        year = week_start.year

        # Check if report already exists
        existing = session.execute(text(
            "SELECT id FROM weekly_learning_reports WHERE week_number = :wn AND year = :y"
        ), {'wn': week_num, 'y': year}).fetchone()

        if existing:
            continue

        record_id = str(uuid.uuid4())

        best_hooks = [
            {"hook": "🚀 AI-powered automation", "ctr": round(random.uniform(2.5, 4.5), 2), "impressions": random.randint(1000, 5000)},
            {"hook": "Stop wasting time on...", "ctr": round(random.uniform(2.0, 3.8), 2), "impressions": random.randint(800, 4000)},
            {"hook": "The secret to scaling", "ctr": round(random.uniform(1.8, 3.5), 2), "impressions": random.randint(600, 3000)}
        ]

        worst_hooks = [
            {"hook": "Generic value prop", "ctr": round(random.uniform(0.3, 0.8), 2), "impressions": random.randint(500, 2000)},
            {"hook": "Learn more about...", "ctr": round(random.uniform(0.2, 0.6), 2), "impressions": random.randint(400, 1500)}
        ]

        platform_performance = {
            "linkedin": {"ctr": round(random.uniform(1.5, 3.0), 2), "conversions": random.randint(10, 50)},
            "twitter": {"ctr": round(random.uniform(0.8, 2.0), 2), "conversions": random.randint(5, 30)}
        }

        persona_performance = {
            "decision_maker": {"ctr": round(random.uniform(2.0, 4.0), 2), "conversion_rate": round(random.uniform(8, 15), 1)},
            "engineer": {"ctr": round(random.uniform(1.5, 3.0), 2), "conversion_rate": round(random.uniform(5, 12), 1)},
            "marketer": {"ctr": round(random.uniform(1.8, 3.5), 2), "conversion_rate": round(random.uniform(6, 14), 1)}
        }

        recommendations = [
            "Focus on urgency-based hooks - they showed 40% higher CTR",
            "LinkedIn outperforms Twitter for decision_maker persona",
            "Consider A/B testing CTA variations on top performers"
        ]

        import json

        session.execute(text("""
            INSERT INTO weekly_learning_reports (
                id, week_start, week_end, week_number, year,
                best_hooks, worst_hooks,
                platform_performance, persona_performance,
                bandit_insights, regret_cumulative, exploration_exploitation_ratio,
                ctr_this_week, ctr_last_week, ctr_change_pct,
                conversions_this_week, conversions_last_week, conversions_change_pct,
                cpl_this_week, cpl_last_week, cpl_change_pct,
                recommendations, generated_at, generated_by
            ) VALUES (
                :id, :week_start, :week_end, :week_number, :year,
                :best_hooks::jsonb, :worst_hooks::jsonb,
                :platform_perf::jsonb, :persona_perf::jsonb,
                :bandit_insights::jsonb, :regret, :exploration_ratio,
                :ctr_this, :ctr_last, :ctr_change,
                :conv_this, :conv_last, :conv_change,
                :cpl_this, :cpl_last, :cpl_change,
                :recommendations::jsonb, :generated_at, 'system'
            )
            ON CONFLICT DO NOTHING
        """), {
            'id': record_id,
            'week_start': week_start,
            'week_end': week_end,
            'week_number': week_num,
            'year': year,
            'best_hooks': json.dumps(best_hooks),
            'worst_hooks': json.dumps(worst_hooks),
            'platform_perf': json.dumps(platform_performance),
            'persona_perf': json.dumps(persona_performance),
            'bandit_insights': json.dumps({"algorithm": "thompson_sampling", "arms_explored": 12}),
            'regret': round(random.uniform(50, 200), 2),
            'exploration_ratio': round(random.uniform(0.15, 0.35), 2),
            'ctr_this': round(random.uniform(1.5, 3.0), 2),
            'ctr_last': round(random.uniform(1.3, 2.8), 2),
            'ctr_change': round(random.uniform(-10, 20), 1),
            'conv_this': random.randint(30, 80),
            'conv_last': random.randint(25, 70),
            'conv_change': round(random.uniform(-15, 25), 1),
            'cpl_this': round(random.uniform(8, 20), 2),
            'cpl_last': round(random.uniform(9, 22), 2),
            'cpl_change': round(random.uniform(-20, 10), 1),
            'recommendations': json.dumps(recommendations),
            'generated_at': week_end
        })
        count += 1

    session.commit()
    logger.info(f"Created {count} weekly reports")
    return count


def update_config_research_mode(session):
    """Enable research mode in configuration"""
    logger.info("Enabling research mode in configuration...")

    try:
        session.execute(text("""
            UPDATE system_configurations
            SET value = 'True', updated_at = NOW()
            WHERE key = 'ENABLE_RESEARCH_MODE'
        """))

        # If no rows updated, insert the config
        result = session.execute(text("""
            INSERT INTO system_configurations (id, key, value, category, is_secret, value_type, description, created_at, updated_at)
            VALUES (
                gen_random_uuid(),
                'ENABLE_RESEARCH_MODE',
                'True',
                'feature_flags',
                false,
                'boolean',
                'Enable advanced research experiments',
                NOW(),
                NOW()
            )
            ON CONFLICT (key) DO UPDATE SET value = 'True', updated_at = NOW()
        """))

        session.commit()
        logger.info("Research mode enabled")
        return True
    except Exception as e:
        logger.warning(f"Could not update research mode config: {e}")
        return False


def main():
    """Main seed function"""
    print("=" * 70)
    print("🌱 SEEDING DATABASE WITH TEST DATA")
    print("=" * 70)

    try:
        session = get_session()

        # Get campaign IDs
        campaign_ids = get_campaign_ids(session)

        if not campaign_ids:
            logger.warning("No campaigns found. Please create campaigns first.")
            logger.info("Run: python scripts/populate_test_data.py")
            return

        logger.info(f"Found {len(campaign_ids)} campaigns")

        # Seed data
        results = {
            'simulation_accuracy': seed_simulation_accuracy(session, campaign_ids),
            'workflow_events': seed_workflow_events(session, campaign_ids),
            'governance_metrics': seed_governance_metrics(session),
            'weekly_reports': seed_weekly_reports(session),
            'research_mode': update_config_research_mode(session)
        }

        print("\n" + "=" * 70)
        print("✅ SEED DATA COMPLETE")
        print("=" * 70)
        print("\n📊 Summary:")
        for key, value in results.items():
            status = "✅" if value else "⚠️"
            print(f"   {status} {key}: {value}")

        print("\n🔧 Next steps:")
        print("   1. Run migrations if tables don't exist: alembic upgrade head")
        print("   2. Restart the API: make restart")
        print("   3. View dashboard: http://localhost:8501")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Seed failed: {e}")
        raise


if __name__ == "__main__":
    main()
