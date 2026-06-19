#!/usr/bin/env python3
"""
Reset database - Clear all data from all tables
USE WITH CAUTION: This will delete ALL campaign, content, and analytics data
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.data_layer.database.connection import get_async_session


async def reset_database():
    """Clear all data from database tables"""
    print("="*70)
    print("DATABASE RESET - Clearing all data")
    print("="*70)
    print("\nThis will delete ALL data from the following tables:")
    print("  - campaigns")
    print("  - contents")
    print("  - hitl_queue")
    print("  - metrics")
    print("  - experiments")
    print("  - delayed_rewards")
    print("  - cost_tracking")
    print("  - scraped_content")
    print("  - agent_memory (episodic)")
    print("  - documents (RAG)")
    print("  - semantic_cache")
    print("\n" + "="*70)

    # Confirm
    confirm = input("Type 'RESET' to confirm deletion of all data: ")
    if confirm != "RESET":
        print("❌ Reset cancelled")
        return False

    print("\n🔄 Resetting database...")

    try:
        async with get_async_session() as session:
            # Disable foreign key checks temporarily
            await session.execute(text("SET session_replication_role = 'replica';"))

            # Clear tables in correct order
            tables_to_clear = [
                'hitl_queue',
                'delayed_rewards',
                'experiments',
                'cost_tracking',
                'metrics',
                'contents',
                'campaigns',
                'scraped_content',
                'documents',
                'semantic_cache'
            ]

            for table in tables_to_clear:
                try:
                    result = await session.execute(text(f"DELETE FROM {table};"))
                    count = result.rowcount
                    print(f"  ✓ Cleared {table}: {count} rows deleted")
                except Exception as e:
                    print(f"  ⚠ Skipped {table}: {e}")

            # Re-enable foreign key checks
            await session.execute(text("SET session_replication_role = 'origin';"))

            # Reset sequences (auto-increment IDs)
            try:
                await session.execute(text("""
                    SELECT setval(pg_get_serial_sequence('campaigns', 'id'), 1, false);
                """))
                print("  ✓ Reset sequences")
            except:
                pass

            await session.commit()

            print("\n✅ Database reset complete!")
            print("="*70)
            return True

    except Exception as e:
        print(f"\n❌ Error during reset: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_reset():
    """Verify that tables are empty"""
    print("\n🔍 Verifying reset...")

    async with get_async_session() as session:
        tables = ['campaigns', 'contents', 'metrics', 'hitl_queue']

        all_empty = True
        for table in tables:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table};"))
            count = result.scalar()
            status = "✓" if count == 0 else "✗"
            print(f"  {status} {table}: {count} rows")
            if count > 0:
                all_empty = False

        if all_empty:
            print("\n✅ All tables verified empty")
        else:
            print("\n⚠ Some tables still contain data")

        return all_empty


if __name__ == "__main__":
    print("\n⚠️  WARNING: This script will DELETE ALL DATA from the database")
    print("This action cannot be undone!\n")

    # Run reset
    success = asyncio.run(reset_database())

    if success:
        # Verify
        asyncio.run(verify_reset())

        print("\n📝 Next steps:")
        print("  1. Restart services: docker-compose restart api dashboard")
        print("  2. Create new campaigns via dashboard")
        print("  3. Or seed demo data: docker-compose exec api python scripts/seed_demo_data.py")
    else:
        print("\n❌ Reset failed or cancelled")
        sys.exit(1)
