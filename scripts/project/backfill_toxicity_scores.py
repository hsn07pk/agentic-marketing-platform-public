#!/usr/bin/env python3
"""
Backfill toxicity_score for old content records
Sets toxicity_score = 0.0 where it's currently NULL (assumes old content was safe)
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, update
from src.data_layer.database.connection import get_async_session, async_session_maker
from src.data_layer.database.models import Content


async def backfill_toxicity_scores():
    """Backfill toxicity scores for old content"""
    print("🔄 Starting toxicity score backfill...")

    async with async_session_maker() as session:
        # Check how many records need updating
        result = await session.execute(
            text("SELECT COUNT(*) FROM contents WHERE toxicity_score IS NULL")
        )
        count = result.scalar()

        print(f"📊 Found {count} records with NULL toxicity_score")

        if count == 0:
            print("✅ No records need updating")
            return

        # Ask for confirmation
        response = input(f"\n⚠️  This will set toxicity_score = 0.0 for {count} old records.\n"
                        f"   This assumes old content was safe (0.0 = no toxicity).\n"
                        f"   Continue? (yes/no): ")

        if response.lower() != 'yes':
            print("❌ Aborted")
            return

        # Update records
        print("🔄 Updating records...")
        await session.execute(
            text("""
                UPDATE contents
                SET toxicity_score = 0.0
                WHERE toxicity_score IS NULL
            """)
        )
        await session.commit()

        # Verify
        result = await session.execute(
            text("SELECT COUNT(*) FROM contents WHERE toxicity_score IS NULL")
        )
        remaining = result.scalar()

        print(f"✅ Backfill complete!")
        print(f"   Updated: {count - remaining} records")
        print(f"   Remaining NULL: {remaining}")


if __name__ == "__main__":
    asyncio.run(backfill_toxicity_scores())
