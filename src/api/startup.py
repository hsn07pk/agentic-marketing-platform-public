"""
Startup tasks for FastAPI application
Auto-initializes database, runs migrations, and sets up the system
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from ..data_layer.database.connection import get_async_session, engine
from ..data_layer.database.models import Base
from ..config.settings import settings

logger = logging.getLogger(__name__)

async def check_database_connection() -> bool:
    """
    Check if database connection is working
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

async def check_tables_exist() -> bool:
    """
    Check if database tables exist
    """
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'campaigns'
                );
            """))
            exists = result.scalar()
            return bool(exists)
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return False

async def ensure_pgvector_extension() -> bool:
    """
    Ensure pgvector extension is installed
    """
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                );
            """))
            exists = result.scalar()

            if not exists:
                logger.info("Installing pgvector extension...")
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("✅ pgvector extension installed")
            else:
                logger.info("✅ pgvector extension already installed")

            return True
    except Exception as e:
        logger.error(f"Failed to install pgvector extension: {e}")
        return False

async def initialize_database() -> bool:
    """Initialize database schema if not exists."""
    try:
        logger.info("Initializing database schema...")

        if not await ensure_pgvector_extension():
            logger.error("Failed to install pgvector extension")
            return False

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ Database schema initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        return False

async def seed_initial_data() -> bool:
    """Seed initial data if database is empty."""
    try:
        from ..data_layer.repositories.campaign_repo import CampaignRepository

        async with get_async_session() as session:
            repo = CampaignRepository(session)

            campaigns = await repo.list_campaigns(limit=1)

            if not campaigns:
                logger.info("Database is empty - no campaigns found")
                logger.info("To seed data, use API endpoints: /api/knowledge-base/seed or /api/campaigns")
                return True
            else:
                logger.info("Database already contains data, skipping seeding")
                return True

    except Exception as e:
        logger.error(f"Error seeding initial data: {e}", exc_info=True)
        return False


async def _recover_canary_deployments():
    """Resume monitoring for canary deployments that were in-progress when API restarted."""
    from sqlalchemy import select
    from ..data_layer.database.models import CanaryDeployment
    from ..data_layer.database.connection import get_async_session
    
    active_statuses = ['canary_5_percent', 'canary_25_percent', 'canary_50_percent', 'canary_75_percent']
    
    async with get_async_session() as db:
        result = await db.execute(
            select(CanaryDeployment).where(CanaryDeployment.status.in_(active_statuses))
        )
        stuck_deployments = result.scalars().all()
    
    if not stuck_deployments:
        logger.info("✅ No in-progress canary deployments to recover")
        return
    
    from .routers.canary import canary_controller
    recovered = 0
    for dep in stuck_deployments:
        if dep.deployment_id not in canary_controller.active_deployments:
            from ..automation_layer.deployment.canary_rollout import CanaryDeployment as CD, DeploymentStatus
            status_map = {
                'canary_5_percent': DeploymentStatus.CANARY_5,
                'canary_25_percent': DeploymentStatus.CANARY_25,
                'canary_50_percent': DeploymentStatus.CANARY_50,
                'canary_75_percent': DeploymentStatus.CANARY_75,
            }
            in_mem = CD(
                deployment_id=dep.deployment_id,
                policy_id=dep.policy_id,
                policy_version=dep.policy_version,
                start_time=dep.started_at,
                status=status_map.get(dep.status, DeploymentStatus.CANARY_5),
                current_traffic_percentage=dep.current_traffic_percentage or 0.05,
                baseline_metrics={
                    'ctr': dep.baseline_ctr or 3.0,
                    'conversion_rate': dep.baseline_conversion_rate or 2.0,
                    'cpl': dep.baseline_cpl or 50.0
                }
            )
            in_mem.auto_rollback = dep.auto_rollback_enabled
            in_mem.metadata = dep.extra_data or {}
            canary_controller.active_deployments[dep.deployment_id] = in_mem
            asyncio.create_task(canary_controller._monitor_deployment(dep.deployment_id))
            recovered += 1
    
    logger.info(f"✅ Recovered {recovered} canary deployment(s) — monitoring resumed")


async def startup_sequence() -> None:
    """Complete startup sequence for FastAPI application."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Agentic AI Marketing Platform")
    logger.info("=" * 60)

    logger.info("Step 1: Checking database connection...")
    max_retries = getattr(settings, 'DB_MAX_RETRIES', 5)
    retry_delay = getattr(settings, 'DB_RETRY_DELAY', 2)

    for attempt in range(max_retries):
        if await check_database_connection():
            break
        if attempt < max_retries - 1:
            logger.warning(f"Database not ready, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(retry_delay)
        else:
            logger.error("❌ Failed to connect to database after multiple retries")
            raise Exception("Database connection failed")

    logger.info("Step 2: Checking database schema...")
    tables_exist = await check_tables_exist()

    if not tables_exist:
        logger.info("Database schema not found, initializing...")
        if not await initialize_database():
            logger.error("❌ Failed to initialize database")
            raise Exception("Database initialization failed")
    else:
        logger.info("✅ Database schema exists")

    if settings.AUTO_SEED_DATA:
        logger.info("Step 3: Checking initial data...")
        await seed_initial_data()
    else:
        logger.info("Step 3: Skipping initial data seeding (AUTO_SEED_DATA=False)")

    logger.info("Step 4: Verifying system components...")

    claim_library_path = Path("config/prompts/claim_library.yaml")
    if claim_library_path.exists():
        logger.info("✅ Claim library found")
    else:
        logger.warning("⚠️  Claim library not found at config/prompts/claim_library.yaml")

    brand_voice_path = Path("data/company/brand_voice.json")
    if brand_voice_path.exists():
        logger.info("✅ Brand voice configuration found")
    else:
        logger.warning("⚠️  Brand voice not found at data/company/brand_voice.json")

    logger.info("Step 5: Starting data file monitoring system...")
    try:
        from ..monitoring.data_file_monitor import start_monitoring
        monitor = start_monitoring()
        logger.info("✅ Data file monitoring started - auto-ingestion enabled")
    except Exception as e:
        logger.error(f"⚠️  Failed to start data file monitor: {e}", exc_info=True)
        logger.warning("Data files will need to be ingested manually via API")

    logger.info("Step 6: Starting background task scheduler...")
    try:
        from ..worker.scheduler import setup_scheduler
        scheduler = setup_scheduler()
        logger.info(f"✅ RQ Scheduler started with {len(list(scheduler.get_jobs()))} recurring tasks")
        logger.info("   - Hourly: Platform metrics polling, delayed rewards processing")
        logger.info("   - Daily: Agent memory cleanup (90-day retention)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to start scheduler: {e}")
        logger.warning("Background tasks will not run automatically. Use worker container instead.")

    logger.info("Step 7: Recovering in-progress canary deployments...")
    try:
        await _recover_canary_deployments()
    except Exception as e:
        logger.warning(f"⚠️  Canary recovery failed: {e}")

    logger.info("=" * 60)
    logger.info("✅ Startup sequence completed successfully")
    logger.info(f"📊 API Docs: http://localhost:{settings.PORT}/docs")
    logger.info(f"📈 Dashboard: http://localhost:{getattr(settings, 'DASHBOARD_PORT', 8501)}")
    logger.info("=" * 60)

async def shutdown_sequence() -> None:
    """Shutdown sequence for FastAPI application."""
    logger.info("=" * 60)
    logger.info("🛑 Shutting down Agentic AI Marketing Platform")
    logger.info("=" * 60)

    try:
        from ..monitoring.data_file_monitor import stop_monitoring
        stop_monitoring()
        logger.info("✅ Data file monitor stopped")
    except Exception as e:
        logger.warning(f"⚠️  Error stopping data file monitor: {e}")

    await engine.dispose()
    logger.info("✅ Database connections closed")

    logger.info("=" * 60)
    logger.info("✅ Shutdown completed")
    logger.info("=" * 60)
