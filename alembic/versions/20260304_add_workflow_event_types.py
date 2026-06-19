"""Add missing workflow event type enum values

Revision ID: add_event_types_20260304
Revises: alter_cta_20260304
Create Date: 2026-03-04 09:30:00.000000
"""
from alembic import op

revision = 'add_event_types_20260304'
down_revision = 'alter_cta_text_20260304'
branch_labels = None
depends_on = None

# Values that exist in Python enum but not yet in PostgreSQL enum
NEW_VALUES = [
    'DEPLOYMENT_STARTED',
    'DEPLOYMENT_SUCCESS',
    'DEPLOYMENT_FAILED',
    'CANARY_STARTED',
    'CANARY_PROMOTED',
    'CANARY_ROLLED_BACK',
]


def upgrade() -> None:
    for value in NEW_VALUES:
        op.execute(
            f"ALTER TYPE workfloweventtype ADD VALUE IF NOT EXISTS '{value}'"
        )


def downgrade() -> None:
    pass  # PostgreSQL does not support removing enum values
