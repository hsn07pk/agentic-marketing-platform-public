"""Fix CTR values to percentage format

Revision ID: fix_ctr_percentage_20260129
Revises: 20260120_add_analytics_models
Create Date: 2026-01-29

Per Research Plan Section 10.2, CTR should be displayed as percentage (0-100 scale).
This migration converts any CTR values stored as decimals to percentages.

Detection logic:
- CTR values < 1.0 are likely decimals (e.g., 0.0158 should be 1.58%)
- CTR values >= 1.0 are likely already percentages
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'fix_ctr_percentage_20260129'
down_revision = '20260120_analytics'
branch_labels = None
depends_on = None


def upgrade():
    """
    Convert CTR values from decimal to percentage where needed.
    
    Only converts values < 1.0 (which are decimal format)
    Values >= 1.0 are assumed to already be in percentage format
    """
    # Update campaigns table
    # Only convert CTR values that appear to be in decimal format (< 1.0)
    # Values like 0.0158 become 1.58
    op.execute("""
        UPDATE campaigns 
        SET ctr = ctr * 100 
        WHERE ctr > 0 AND ctr < 1
    """)
    
    # Log the update
    print("Converted CTR values < 1.0 to percentage format in campaigns table")


def downgrade():
    """
    Revert CTR values from percentage back to decimal.
    
    Only converts values between 1 and 100 back to decimal
    This is a lossy operation - we assume all values in that range were converted
    """
    op.execute("""
        UPDATE campaigns 
        SET ctr = ctr / 100 
        WHERE ctr >= 1 AND ctr < 100
    """)
    
    print("Reverted CTR values to decimal format in campaigns table")
