"""Alter contents.cta from varchar(255) to text

Revision ID: alter_cta_text_20260304
Revises: strategy_perf_20260203
Create Date: 2026-03-04

LLM-generated CTAs frequently exceed 255 characters, causing
StringDataRightTruncationError on content insertion.
"""

from alembic import op
import sqlalchemy as sa

revision = 'alter_cta_text_20260304'
down_revision = 'strategy_perf_20260203'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        'contents',
        'cta',
        existing_type=sa.String(255),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade():
    op.alter_column(
        'contents',
        'cta',
        existing_type=sa.Text(),
        type_=sa.String(255),
        existing_nullable=True,
    )
