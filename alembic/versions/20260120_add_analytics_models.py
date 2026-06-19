"""Add analytics models for RQ2 tracking, governance metrics, and weekly reports

Revision ID: 20260120_analytics
Revises:
Create Date: 2026-01-20

Research Plan Requirements:
- RQ2: Simulation-to-Live Accuracy >90%
- Section 10.2: Human Override Rate <5%
- Section 10.2: Weekly Learning Reports
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '20260120_analytics'
down_revision = '002_add_demos_booked'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SimulationLiveAccuracy table for RQ2 tracking
    op.create_table(
        'simulation_live_accuracy',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id'), nullable=False),

        # Simulated metrics
        sa.Column('simulated_impressions', sa.Integer(), default=0),
        sa.Column('simulated_clicks', sa.Integer(), default=0),
        sa.Column('simulated_conversions', sa.Integer(), default=0),
        sa.Column('simulated_ctr', sa.Float(), default=0.0),
        sa.Column('simulated_cpl', sa.Float(), default=0.0),

        # Actual metrics
        sa.Column('actual_impressions', sa.Integer(), default=0),
        sa.Column('actual_clicks', sa.Integer(), default=0),
        sa.Column('actual_conversions', sa.Integer(), default=0),
        sa.Column('actual_ctr', sa.Float(), default=0.0),
        sa.Column('actual_cpl', sa.Float(), default=0.0),

        # MAPE per metric
        sa.Column('mape_impressions', sa.Float()),
        sa.Column('mape_clicks', sa.Float()),
        sa.Column('mape_conversions', sa.Float()),
        sa.Column('mape_ctr', sa.Float()),
        sa.Column('mape_cpl', sa.Float()),

        # Overall accuracy
        sa.Column('overall_mape', sa.Float()),
        sa.Column('overall_accuracy', sa.Float()),
        sa.Column('passes_threshold', sa.Boolean(), default=False),

        # RQ2 tracking
        sa.Column('rq2_target', sa.Float(), default=90.0),
        sa.Column('rq2_gap', sa.Float()),

        # Metadata
        sa.Column('simulation_timestamp', sa.DateTime()),
        sa.Column('measurement_timestamp', sa.DateTime()),
        sa.Column('measurement_type', sa.String(50)),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_index('idx_sim_accuracy_campaign', 'simulation_live_accuracy', ['campaign_id'])
    op.create_index('idx_sim_accuracy_passes', 'simulation_live_accuracy', ['passes_threshold'])
    op.create_index('idx_sim_accuracy_created', 'simulation_live_accuracy', ['created_at'])

    # GovernanceMetrics table for override rate tracking
    op.create_table(
        'governance_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),

        # Time period
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('period_type', sa.String(20)),

        # HITL metrics
        sa.Column('total_reviews', sa.Integer(), default=0),
        sa.Column('approved_count', sa.Integer(), default=0),
        sa.Column('rejected_count', sa.Integer(), default=0),
        sa.Column('modified_count', sa.Integer(), default=0),

        # Override rate
        sa.Column('human_override_rate', sa.Float(), default=0.0),
        sa.Column('override_rate_target', sa.Float(), default=5.0),
        sa.Column('meets_override_target', sa.Boolean(), default=True),

        # Safety metrics
        sa.Column('avg_safety_score', sa.Float()),
        sa.Column('avg_toxicity_score', sa.Float()),
        sa.Column('avg_factuality_score', sa.Float()),
        sa.Column('avg_brand_alignment_score', sa.Float()),

        # Golden test metrics
        sa.Column('golden_test_pass_rate', sa.Float()),
        sa.Column('golden_tests_run', sa.Integer(), default=0),
        sa.Column('golden_tests_passed', sa.Integer(), default=0),

        # Auto-approval
        sa.Column('auto_approved_count', sa.Integer(), default=0),
        sa.Column('auto_approval_rate', sa.Float(), default=0.0),

        # Response time
        sa.Column('avg_review_time_minutes', sa.Float()),
        sa.Column('median_review_time_minutes', sa.Float()),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    op.create_index('idx_gov_metrics_period', 'governance_metrics', ['period_start', 'period_end'])
    op.create_index('idx_gov_metrics_type', 'governance_metrics', ['period_type'])

    # WeeklyLearningReport table
    op.create_table(
        'weekly_learning_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),

        # Report period
        sa.Column('week_start', sa.DateTime(), nullable=False),
        sa.Column('week_end', sa.DateTime(), nullable=False),
        sa.Column('week_number', sa.Integer()),
        sa.Column('year', sa.Integer()),

        # Best/worst hooks
        sa.Column('best_hooks', postgresql.JSONB(), default=[]),
        sa.Column('worst_hooks', postgresql.JSONB(), default=[]),

        # Performance by platform/persona
        sa.Column('platform_performance', postgresql.JSONB(), default={}),
        sa.Column('persona_performance', postgresql.JSONB(), default={}),

        # Bandit insights
        sa.Column('bandit_insights', postgresql.JSONB(), default={}),
        sa.Column('regret_cumulative', sa.Float()),
        sa.Column('exploration_exploitation_ratio', sa.Float()),

        # CTR comparison
        sa.Column('ctr_this_week', sa.Float()),
        sa.Column('ctr_last_week', sa.Float()),
        sa.Column('ctr_change_pct', sa.Float()),

        # Conversions comparison
        sa.Column('conversions_this_week', sa.Integer()),
        sa.Column('conversions_last_week', sa.Integer()),
        sa.Column('conversions_change_pct', sa.Float()),

        # CPL comparison
        sa.Column('cpl_this_week', sa.Float()),
        sa.Column('cpl_last_week', sa.Float()),
        sa.Column('cpl_change_pct', sa.Float()),

        # Recommendations
        sa.Column('recommendations', postgresql.JSONB(), default=[]),

        # Metadata
        sa.Column('generated_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('generated_by', sa.String(100), default='system'),
    )

    op.create_index('idx_weekly_report_period', 'weekly_learning_reports', ['week_start', 'week_end'])
    op.create_unique_constraint('uq_weekly_report_week_year', 'weekly_learning_reports', ['week_number', 'year'])


def downgrade() -> None:
    op.drop_table('weekly_learning_reports')
    op.drop_table('governance_metrics')
    op.drop_table('simulation_live_accuracy')
