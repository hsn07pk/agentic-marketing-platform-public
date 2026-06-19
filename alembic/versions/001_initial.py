"""Initial migration - baseline schema matching models.py

Revision ID: 001_initial
Revises:
Create Date: 2026-01-07

This migration represents the complete schema from src/data_layer/database/models.py.
For existing databases: alembic stamp head
For new databases: alembic upgrade head
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables matching src/data_layer/database/models.py"""

    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # 1. system_configurations
    op.create_table('system_configurations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('key', sa.String(100), unique=True, nullable=False),
        sa.Column('value', sa.Text),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('is_secret', sa.Boolean, default=False),
        sa.Column('description', sa.Text),
        sa.Column('default_value', sa.Text),
        sa.Column('value_type', sa.String(20), default='string'),
        sa.Column('validation', postgresql.JSONB, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )
    op.create_index('idx_config_category', 'system_configurations', ['category'])
    op.create_index('idx_config_key', 'system_configurations', ['key'])

    # 2. personas
    op.create_table('personas',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), unique=True, nullable=False),
        sa.Column('title', sa.String(200)),
        sa.Column('description', sa.Text),
        sa.Column('role', sa.String(100)),
        sa.Column('seniority', sa.String(50)),
        sa.Column('industry', sa.String(100)),
        sa.Column('daily_active_prob', sa.Float, default=0.5),
        sa.Column('content_engagement_prob', sa.Float, default=0.1),
        sa.Column('click_prob', sa.Float, default=0.05),
        sa.Column('conversion_prob', sa.Float, default=0.01),
        sa.Column('share_prob', sa.Float, default=0.02),
        sa.Column('preferred_content_types', postgresql.ARRAY(sa.String)),
        sa.Column('preferred_channels', postgresql.ARRAY(sa.String)),
        sa.Column('active_hours', postgresql.ARRAY(sa.Integer)),
        sa.Column('company_size', sa.String(50)),
        sa.Column('budget_range', sa.String(50)),
        sa.Column('pain_points', postgresql.ARRAY(sa.String)),
        sa.Column('goals', postgresql.ARRAY(sa.String)),
        sa.Column('attributes', postgresql.JSONB, default={}),
        sa.Column('preferences', postgresql.JSONB, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )

    # 3. campaigns
    op.create_table('campaigns',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('platform', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('goal', sa.String(50)),
        sa.Column('target_persona', sa.String(100)),
        sa.Column('target_keywords', postgresql.ARRAY(sa.String)),
        sa.Column('target_demographics', postgresql.JSONB),
        sa.Column('budget_total', sa.Float, default=0.0),
        sa.Column('budget_spent', sa.Float, default=0.0),
        sa.Column('budget_daily_limit', sa.Float),
        sa.Column('start_date', sa.DateTime),
        sa.Column('end_date', sa.DateTime),
        sa.Column('impressions', sa.Integer, default=0),
        sa.Column('clicks', sa.Integer, default=0),
        sa.Column('conversions', sa.Integer, default=0),
        sa.Column('ctr', sa.Float, default=0.0),
        sa.Column('cpl', sa.Float, default=0.0),
        sa.Column('config', postgresql.JSONB, default={}),
        sa.Column('is_mock', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )
    op.create_index('idx_campaign_status', 'campaigns', ['status'])
    op.create_index('idx_campaign_platform', 'campaigns', ['platform'])

    # 4. contents
    op.create_table('contents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id'), nullable=False),
        sa.Column('content_type', sa.String(50)),
        sa.Column('headline', sa.Text),
        sa.Column('body', sa.Text, nullable=False),
        sa.Column('cta', sa.String(255)),
        sa.Column('image_url', sa.String(500)),
        sa.Column('platform_specific', postgresql.JSONB, default={}),
        sa.Column('generated_by', sa.String(50), default='content_generator'),
        sa.Column('prompt_used', sa.Text),
        sa.Column('model_used', sa.String(50)),
        sa.Column('claims_used', postgresql.ARRAY(sa.String)),
        sa.Column('status', sa.String(50), default='generated'),
        sa.Column('safety_score', sa.Float),
        sa.Column('toxicity_score', sa.Float),
        sa.Column('factuality_score', sa.Float),
        sa.Column('brand_alignment_score', sa.Float),
        sa.Column('review_notes', sa.Text),
        sa.Column('reviewed_by', sa.String(100)),
        sa.Column('reviewed_at', sa.DateTime),
        sa.Column('impressions', sa.Integer, default=0),
        sa.Column('clicks', sa.Integer, default=0),
        sa.Column('conversions', sa.Integer, default=0),
        sa.Column('engagement_rate', sa.Float, default=0.0),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('deployed_at', sa.DateTime),
        sa.Column('generation_cost', sa.Float, default=0.0),
        sa.Column('is_mock', sa.Boolean, default=False)
    )
    op.create_index('idx_content_status', 'contents', ['status'])
    op.create_index('idx_content_safety_score', 'contents', ['safety_score'])

    # 5. experiments
    op.create_table('experiments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('type', sa.String(50)),
        sa.Column('variants', postgresql.JSONB),
        sa.Column('algorithm', sa.String(50)),
        sa.Column('parameters', postgresql.JSONB),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('winner_variant', sa.String(100)),
        sa.Column('total_impressions', sa.Integer, default=0),
        sa.Column('total_conversions', sa.Integer, default=0),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime)
    )

    # 6. bandit_arms
    op.create_table('bandit_arms',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('experiments.id')),
        sa.Column('arm_id', sa.String(100), nullable=False),
        sa.Column('variant_data', postgresql.JSONB),
        sa.Column('alpha', sa.Float, default=1.0),
        sa.Column('beta', sa.Float, default=1.0),
        sa.Column('pulls', sa.Integer, default=0),
        sa.Column('successes', sa.Integer, default=0),
        sa.Column('failures', sa.Integer, default=0),
        sa.Column('total_reward', sa.Float, default=0.0),
        sa.Column('context_vector', Vector(50)),
        sa.Column('last_pulled_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('experiment_id', 'arm_id')
    )

    # 7. metrics
    op.create_table('metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id')),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('metric_type', sa.String(50)),
        sa.Column('platform', sa.String(50)),
        sa.Column('agent_type', sa.String(50)),
        sa.Column('tags', postgresql.JSONB, default={}),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now())
    )
    op.create_index('idx_metrics_timestamp', 'metrics', ['timestamp'])
    op.create_index('idx_metrics_name', 'metrics', ['metric_name'])

    # 8. agent_actions
    op.create_table('agent_actions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('agent_type', sa.String(50), nullable=False),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id')),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('contents.id')),
        sa.Column('input_data', postgresql.JSONB),
        sa.Column('output_data', postgresql.JSONB),
        sa.Column('error', sa.Text),
        sa.Column('tokens_used', sa.Integer, default=0),
        sa.Column('api_cost', sa.Float, default=0.0),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('duration_ms', sa.Integer)
    )
    op.create_index('idx_agent_actions_timestamp', 'agent_actions', ['started_at'])
    op.create_index('idx_agent_actions_agent', 'agent_actions', ['agent_type'])

    # 9. hitl_queue
    op.create_table('hitl_queue',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('contents.id'), nullable=False),
        sa.Column('priority', sa.Integer, default=0),
        sa.Column('reason', sa.Text),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('assigned_to', sa.String(100)),
        sa.Column('decision', sa.String(50)),
        sa.Column('feedback', sa.Text),
        sa.Column('modifications', postgresql.JSONB),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('assigned_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime)
    )
    op.create_index('idx_hitl_status', 'hitl_queue', ['status'])
    op.create_index('idx_hitl_priority', 'hitl_queue', ['priority'])

    # 10. vector_store
    op.create_table('vector_store',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('doc_type', sa.String(50), nullable=False),
        sa.Column('doc_id', sa.String(255)),
        sa.Column('doc_text', sa.Text, nullable=False),
        sa.Column('embedding', Vector(1536)),
        sa.Column('meta_data', postgresql.JSONB, default={}),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )
    op.create_index('idx_vector_doc_type', 'vector_store', ['doc_type'])

    # 11. cost_tracking
    op.create_table('cost_tracking',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('source_type', sa.String(50)),
        sa.Column('source_id', sa.String(255)),
        sa.Column('cost_amount', sa.Float, nullable=False),
        sa.Column('cost_currency', sa.String(3), default='EUR'),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id')),
        sa.Column('agent_type', sa.String(50)),
        sa.Column('provider', sa.String(50)),
        sa.Column('tokens_prompt', sa.Integer),
        sa.Column('tokens_completion', sa.Integer),
        sa.Column('timestamp', sa.DateTime, server_default=sa.func.now())
    )
    op.create_index('idx_cost_timestamp', 'cost_tracking', ['timestamp'])
    op.create_index('idx_cost_campaign', 'cost_tracking', ['campaign_id'])

    # 12. delayed_rewards
    op.create_table('delayed_rewards',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id'), nullable=False),
        sa.Column('lead_email', sa.String(255), nullable=False),
        sa.Column('lead_data', postgresql.JSON),
        sa.Column('initial_reward', sa.Float, default=1.0),
        sa.Column('current_reward', sa.Float, default=1.0),
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('registered_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime),
        sa.Column('meeting_scheduled', sa.Boolean, default=False),
        sa.Column('meeting_date', sa.DateTime),
        sa.Column('meeting_attended', sa.Boolean, default=False),
        sa.Column('hubspot_contact_id', sa.String(50)),
        sa.Column('lead_score', sa.Integer)
    )
    op.create_index('idx_delayed_reward_campaign', 'delayed_rewards', ['campaign_id'])
    op.create_index('idx_delayed_reward_status', 'delayed_rewards', ['status'])
    op.create_index('idx_delayed_reward_registered', 'delayed_rewards', ['registered_at'])

    # 13. scraped_content
    op.create_table('scraped_content',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('source_url', sa.String(500)),
        sa.Column('keywords', sa.Text),
        sa.Column('raw_content', sa.Text),
        sa.Column('insights', postgresql.JSONB, default={}),
        sa.Column('scraped_at', sa.DateTime, server_default=sa.func.now())
    )
    op.create_index('idx_scraped_content_source', 'scraped_content', ['source'])
    op.create_index('idx_scraped_content_scraped_at', 'scraped_content', ['scraped_at'])

    # 14. calibration_runs
    op.create_table('calibration_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('historical_data_source', sa.String(255)),
        sa.Column('num_training_campaigns', sa.Integer),
        sa.Column('num_validation_campaigns', sa.Integer),
        sa.Column('validation_mape', sa.Float),
        sa.Column('validation_accuracy', sa.Float),
        sa.Column('passes_threshold', sa.Boolean, default=False),
        sa.Column('metrics', postgresql.JSONB, default={}),
        sa.Column('optimization_method', sa.String(50)),
        sa.Column('random_seed', sa.Integer, default=42),
        sa.Column('started_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime),
        sa.Column('duration_seconds', sa.Float),
        sa.Column('status', sa.String(20), default='running'),
        sa.Column('error_message', sa.Text)
    )
    op.create_index('idx_calibration_status', 'calibration_runs', ['status'])
    op.create_index('idx_calibration_started', 'calibration_runs', ['started_at'])

    # 15. workflow_events
    op.create_table('workflow_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('campaigns.id'), nullable=False),
        sa.Column('content_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('contents.id')),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), default='info'),
        sa.Column('workflow_node', sa.String(100)),
        sa.Column('workflow_state', sa.String(50)),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('details', postgresql.JSONB, default={}),
        sa.Column('is_user_actionable', sa.Boolean, default=False),
        sa.Column('is_dismissed', sa.Boolean, default=False),
        sa.Column('dismissed_at', sa.DateTime),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )
    op.create_index('idx_workflow_event_campaign', 'workflow_events', ['campaign_id'])
    op.create_index('idx_workflow_event_type', 'workflow_events', ['event_type'])
    op.create_index('idx_workflow_event_severity', 'workflow_events', ['severity'])
    op.create_index('idx_workflow_event_created', 'workflow_events', ['created_at'])
    op.create_index('idx_workflow_event_actionable', 'workflow_events', ['is_user_actionable'])
    op.create_index('idx_workflow_event_dismissed', 'workflow_events', ['is_dismissed'])

    # 16. data_file_ingestion
    op.create_table('data_file_ingestion',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('file_path', sa.String(500), unique=True, nullable=False),
        sa.Column('file_name', sa.String(255), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=False),
        sa.Column('file_category', sa.String(100)),
        sa.Column('file_size_bytes', sa.Integer),
        sa.Column('file_hash', sa.String(64)),
        sa.Column('last_modified_at', sa.DateTime),
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('ingestion_started_at', sa.DateTime),
        sa.Column('ingestion_completed_at', sa.DateTime),
        sa.Column('ingestion_duration_seconds', sa.Float),
        sa.Column('error_message', sa.Text),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('document_ids', postgresql.ARRAY(sa.Integer)),
        sa.Column('num_chunks', sa.Integer),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, onupdate=sa.func.now())
    )
    op.create_index('idx_file_ingestion_status', 'data_file_ingestion', ['status'])
    op.create_index('idx_file_ingestion_category', 'data_file_ingestion', ['file_category'])
    op.create_index('idx_file_ingestion_path', 'data_file_ingestion', ['file_path'])

    # 17. persona_calibrations
    op.create_table('persona_calibrations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('calibration_run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('calibration_runs.id'), nullable=False),
        sa.Column('persona_name', sa.String(100), nullable=False),
        sa.Column('platform', sa.String(50)),
        sa.Column('daily_active_prob', sa.Float, nullable=False),
        sa.Column('click_prob', sa.Float, nullable=False),
        sa.Column('conversion_prob', sa.Float, nullable=False),
        sa.Column('content_engagement_prob', sa.Float, default=0.1),
        sa.Column('share_prob', sa.Float, default=0.02),
        sa.Column('ad_fatigue_threshold', sa.Integer, default=5),
        sa.Column('ad_fatigue_decay', sa.Float, default=0.15),
        sa.Column('influence_factor', sa.Float, default=0.3),
        sa.Column('active_hours', postgresql.JSONB, default=[9, 10, 11, 14, 15, 16]),
        sa.Column('training_mape', sa.Float),
        sa.Column('num_training_samples', sa.Integer),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint('persona_name', 'platform', 'version', name='uq_persona_platform_version')
    )
    op.create_index('idx_persona_calibration_name', 'persona_calibrations', ['persona_name'])
    op.create_index('idx_persona_calibration_active', 'persona_calibrations', ['is_active'])


def downgrade() -> None:
    """Drop all tables in reverse order"""
    op.drop_table('persona_calibrations')
    op.drop_table('data_file_ingestion')
    op.drop_table('workflow_events')
    op.drop_table('calibration_runs')
    op.drop_table('scraped_content')
    op.drop_table('delayed_rewards')
    op.drop_table('cost_tracking')
    op.drop_table('vector_store')
    op.drop_table('hitl_queue')
    op.drop_table('agent_actions')
    op.drop_table('metrics')
    op.drop_table('bandit_arms')
    op.drop_table('experiments')
    op.drop_table('contents')
    op.drop_table('campaigns')
    op.drop_table('personas')
    op.drop_table('system_configurations')
