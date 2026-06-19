"""Add demos_booked column to campaigns table

Revision ID: 002_add_demos_booked
Revises: 001_initial
Create Date: 2026-01-12

Per Audit v0.1 - Cal.com webhook integration requires tracking demo bookings
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_demos_booked'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade():
    """Add demos_booked column to campaigns table for Cal.com booking tracking."""
    op.add_column(
        'campaigns',
        sa.Column('demos_booked', sa.Integer(), nullable=True, server_default='0')
    )


def downgrade():
    """Remove demos_booked column from campaigns table."""
    op.drop_column('campaigns', 'demos_booked')
