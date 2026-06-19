"""Add strategy_performance table for bandit optimizer persistence

Revision ID: strategy_perf_20260203
Revises: add_canary_deployments_20260203
Create Date: 2026-02-03

Per Research Plan Section 4.1: Multi-Armed Bandit strategy selection requires
persistent tracking of strategy decisions and rewards for learning continuity
across API restarts.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = 'strategy_perf_20260203'
down_revision = 'add_canary_deployments_20260203'
branch_labels = None
depends_on = None


def upgrade():
    """Create strategy_performance table"""
    op.create_table(
        'strategy_performance',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        
        # Campaign and action info
        sa.Column('campaign_id', sa.String(255), nullable=False, index=True),
        sa.Column('action', sa.String(100), nullable=False, index=True),
        
        # Performance metrics
        sa.Column('reward', sa.Float(), nullable=False),
        
        # Context when decision was made
        sa.Column('context', JSONB, server_default='{}'),
        
        # Derived metrics
        sa.Column('estimated_ctr', sa.Float(), nullable=True),
        sa.Column('estimated_conversions', sa.Integer(), nullable=True),
        sa.Column('estimated_cpl', sa.Float(), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('NOW()'), nullable=False),
    )
    
    # Create indexes
    op.create_index('idx_strategy_campaign', 'strategy_performance', ['campaign_id'])
    op.create_index('idx_strategy_action', 'strategy_performance', ['action'])
    op.create_index('idx_strategy_created', 'strategy_performance', ['created_at'])


def downgrade():
    """Drop strategy_performance table"""
    op.drop_index('idx_strategy_created', table_name='strategy_performance')
    op.drop_index('idx_strategy_action', table_name='strategy_performance')
    op.drop_index('idx_strategy_campaign', table_name='strategy_performance')
    op.drop_table('strategy_performance')
