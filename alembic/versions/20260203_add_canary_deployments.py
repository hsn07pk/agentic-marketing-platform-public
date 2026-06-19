"""Add canary_deployments table for production persistence

Revision ID: add_canary_deployments_20260203
Revises: fix_ctr_percentage_20260129
Create Date: 2026-02-03

Per Research Plan Section 2.3: Canary Rollout requires persistent tracking of
deployment state for durability across API restarts. This migration adds the
canary_deployments table to store deployment history and active deployments.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = 'add_canary_deployments_20260203'
down_revision = 'fix_ctr_percentage_20260129'
branch_labels = None
depends_on = None


def upgrade():
    """Create canary_deployments table"""
    op.create_table(
        'canary_deployments',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('deployment_id', sa.String(255), unique=True, nullable=False, index=True),
        
        # Policy information
        sa.Column('policy_id', sa.String(255), nullable=False, index=True),
        sa.Column('policy_version', sa.String(50), server_default='1.0.0'),
        sa.Column('deployment_type', sa.String(50), server_default='policy'),
        
        # Status tracking
        sa.Column('status', sa.String(50), nullable=False, server_default='pending', index=True),
        
        # Traffic control
        sa.Column('current_traffic_percentage', sa.Float, server_default='0.05'),
        
        # Baseline metrics for comparison (as percentages)
        sa.Column('baseline_ctr', sa.Float),
        sa.Column('baseline_conversion_rate', sa.Float),
        sa.Column('baseline_cpl', sa.Float),
        
        # Rollback configuration
        sa.Column('auto_rollback_enabled', sa.Boolean, server_default='true'),
        sa.Column('rollback_ctr_threshold', sa.Float, server_default='10.0'),
        sa.Column('rollback_error_threshold', sa.Float, server_default='5.0'),
        sa.Column('rollback_latency_threshold', sa.Integer, server_default='500'),
        
        # Rollback info
        sa.Column('rollback_reason', sa.Text, nullable=True),
        
        # Extra data (campaign_id, description, etc.)
        sa.Column('extra_data', JSONB, server_default='{}'),
        
        # Canary metrics history
        sa.Column('metrics_history', JSONB, server_default='[]'),
        
        # Timestamps
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        
        # Decision and results (keeping for historical compatibility)
        sa.Column('decision', sa.String(50), nullable=True),
        sa.Column('decision_reason', sa.Text, nullable=True),
        sa.Column('final_metrics', JSONB, nullable=True),
        
        # Audit columns
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    )
    
    # Create indexes for common queries
    op.create_index('ix_canary_deployments_status_started', 'canary_deployments', ['status', 'started_at'])


def downgrade():
    """Drop canary_deployments table"""
    op.drop_index('ix_canary_deployments_status_started', table_name='canary_deployments')
    op.drop_table('canary_deployments')
