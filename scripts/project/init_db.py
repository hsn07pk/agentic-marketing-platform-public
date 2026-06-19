#!/usr/bin/env python3
"""
Database Initialization Script

Creates all necessary database tables, extensions, and indexes.
Run this after creating the database but before starting the application.
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.data_layer.database.connection import get_engine, get_async_session
from src.data_layer.database.models import Base
from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def init_database():
    """Initialize database with tables and extensions"""
    try:
        logger.info("=" * 60)
        logger.info("AGENTIC DATABASE INITIALIZATION")
        logger.info("=" * 60)

        # Get database engine
        logger.info("Connecting to database...")
        logger.info(f"Database URL: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'configured'}")

        engine = get_engine()

        # Test connection
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"✓ Connected to PostgreSQL: {version.split(',')[0]}")

        # Install pgvector extension
        logger.info("\nInstalling pgvector extension...")
        async with engine.begin() as conn:
            # Check if extension exists
            result = await conn.execute(
                text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            extension_exists = result.scalar()

            if not extension_exists:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("✓ pgvector extension installed")
            else:
                logger.info("✓ pgvector extension already installed")

        # Create all tables
        logger.info("\nCreating database tables...")
        async with engine.begin() as conn:
            # Drop all tables if RESET_DB environment variable is set
            if settings.ENVIRONMENT == "development":
                logger.warning("Development mode: Dropping existing tables...")
                await conn.run_sync(Base.metadata.drop_all)

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("✓ Database tables created")

        # Create indexes for performance
        logger.info("\nCreating performance indexes...")
        async with engine.begin() as conn:
            # Index on campaign status for filtering
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_campaigns_status
                ON campaigns(status)
            """))

            # Index on content safety_score for filtering
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_content_safety_score
                ON contents(safety_score)
            """))

            # Index on metrics timestamp for time-series queries
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON metrics(timestamp DESC)
            """))

            # Index on campaign_id for joins
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_content_campaign_id
                ON contents(campaign_id)
            """))

            logger.info("✓ Performance indexes created")

        # Verify tables were created
        logger.info("\nVerifying tables...")
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]

            expected_tables = [
                'campaigns', 'content', 'metrics', 'personas',
                'hitl_queue', 'claim_library', 'agent_memory',
                'policy_versions', 'canary_deployments'
            ]

            logger.info(f"Found {len(tables)} tables:")
            for table in tables:
                status = "✓" if table in expected_tables else "?"
                logger.info(f"  {status} {table}")

        # Create vector store collections table
        logger.info("\nCreating vector store collections...")
        async with engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) UNIQUE NOT NULL,
                    cmetadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                    embedding vector(1536),
                    document TEXT,
                    cmetadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Create HNSW index for fast vector similarity search
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
                ON langchain_pg_embedding
                USING hnsw (embedding vector_cosine_ops)
            """))

            logger.info("✓ Vector store tables created")

        logger.info("\n" + "=" * 60)
        logger.info("DATABASE INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Run: python scripts/seed_demo_data.py (to load sample data)")
        logger.info("2. Start API: uvicorn src.api.main:app --reload")
        logger.info("3. Start Dashboard: streamlit run dashboard/app.py")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"\n❌ Database initialization failed: {e}")
        logger.exception("Full error details:")
        return False


async def test_connection():
    """Test database connection and extensions"""
    try:
        logger.info("Testing database connection...")

        engine = get_engine()
        async with engine.begin() as conn:
            # Test basic query
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

            # Test pgvector extension
            result = await conn.execute(
                text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            has_vector = result.scalar()

            if has_vector:
                logger.info("✓ Database connection successful")
                logger.info("✓ pgvector extension available")
                return True
            else:
                logger.error("✗ pgvector extension not installed")
                logger.info("Run: CREATE EXTENSION vector; in PostgreSQL")
                return False

    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("1. Check PostgreSQL is running")
        logger.info("2. Verify DATABASE_URL in .env")
        logger.info("3. Ensure database 'agentic' exists")
        logger.info("4. Check credentials are correct")
        return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Initialize Agentic database')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test connection only, do not initialize'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reset database (WARNING: deletes all data)'
    )

    args = parser.parse_args()

    if args.test:
        success = asyncio.run(test_connection())
    else:
        if args.force:
            logger.warning("⚠️  FORCE RESET ENABLED - All data will be deleted!")
            response = input("Are you sure? Type 'yes' to continue: ")
            if response.lower() != 'yes':
                logger.info("Aborted.")
                sys.exit(0)

        success = asyncio.run(init_database())

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
