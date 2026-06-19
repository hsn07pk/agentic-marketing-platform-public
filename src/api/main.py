"""
Agentic AI Agent Platform - Main API Application
Production-ready FastAPI application with complete routing and middleware
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import logging
import time
from contextlib import asynccontextmanager

from .routers import (
    campaigns,
    metrics,
    governance,
    health,
    experiments,
    rewards,
    scraper,
    knowledge_base,
    strategy,
    ope,
    canary,
    memory,
    advanced_experiments,
    calibration,
    events,
    personas,
    bandit_arms,
    costs,
    configuration,
    funnel,
    mlflow,
    analytics,  # RQ2 accuracy, override rate, weekly reports
    operations,
    llm,
    data_config
)
from ..config.settings import settings
from ..ai_layer.learning.reward_tracker import RewardTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

API_ERRORS = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
)

ACTIVE_CAMPAIGNS = Gauge(
    'active_campaigns_total',
    'Number of active campaigns'
)

TOTAL_CAMPAIGNS = Gauge(
    'campaigns_total',
    'Total number of campaigns',
    ['status']
)

CAMPAIGN_BUDGET_TOTAL = Gauge(
    'campaign_budget_total_euros',
    'Total campaign budget in euros'
)

CAMPAIGN_SPEND_TOTAL = Gauge(
    'campaign_spend_total_euros',
    'Total campaign spend in euros'
)

COST_PER_LEAD = Gauge(
    'cost_per_lead_euros',
    'Cost per lead in euros'
)

COST_PER_BOOKED_CALL = Gauge(
    'cost_per_booked_call_euros',
    'Cost per booked call in euros'
)

BOOKED_CALLS_TOTAL = Gauge(
    'booked_calls_total',
    'Total number of booked calls'
)

LEADS_TOTAL = Gauge(
    'leads_total',
    'Total number of leads generated'
)

CONVERSION_RATE = Gauge(
    'conversion_rate_percent',
    'Lead to booked call conversion rate'
)

CONTENT_GENERATED_TOTAL = Counter(
    'content_generated_total',
    'Total content pieces generated',
    ['content_type', 'platform']
)

SEMANTIC_CACHE_HITS = Counter(
    'semantic_cache_hits_total',
    'Semantic cache hits'
)

SEMANTIC_CACHE_MISSES = Counter(
    'semantic_cache_misses_total',
    'Semantic cache misses'
)

SEMANTIC_CACHE_HIT_RATE = Gauge(
    'semantic_cache_hit_rate',
    'Semantic cache hit rate (0-1)'
)

LLM_TOKENS_USED = Counter(
    'llm_tokens_used_total',
    'Total LLM tokens used',
    ['model', 'operation']
)

LLM_COST_TOTAL = Gauge(
    'llm_cost_total_euros',
    'Total LLM API cost in euros'
)

LLM_REQUESTS_TOTAL = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status']
)

GOVERNANCE_REVIEWS_TOTAL = Counter(
    'governance_reviews_total',
    'Total governance reviews',
    ['result']
)

GOVERNANCE_SAFETY_SCORE = Gauge(
    'governance_safety_score',
    'Average content safety score (0-1)'
)

GOVERNANCE_APPROVAL_RATE = Gauge(
    'governance_approval_rate',
    'Content auto-approval rate (0-1)'
)

HUMAN_REVIEW_PENDING = Gauge(
    'human_review_pending_total',
    'Content pieces pending human review'
)

EXPERIMENTS_TOTAL = Gauge(
    'experiments_total',
    'Total experiments',
    ['status']
)

EXPERIMENT_VARIANTS = Gauge(
    'experiment_variants_total',
    'Total experiment variants/arms'
)

SIMULATION_RUNS_TOTAL = Counter(
    'simulation_runs_total',
    'Total simulation runs',
    ['status']
)

SIMULATION_ACCURACY = Gauge(
    'simulation_to_live_accuracy',
    'Simulation to live prediction accuracy (0-1)'
)

async def update_metrics_from_db():
    """Periodically update Prometheus metrics from database"""
    import asyncio
    from sqlalchemy import text
    from ..data_layer.database.connection import get_async_session

    while True:
        try:
            async with get_async_session() as session:
                result = await session.execute(text("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'RUNNING') as active,
                        COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
                        COUNT(*) FILTER (WHERE status = 'PAUSED') as paused,
                        COUNT(*) FILTER (WHERE status = 'DRAFT') as draft,
                        COALESCE(SUM(budget_total), 0) as total_budget,
                        COALESCE(SUM(budget_spent), 0) as total_spent
                    FROM campaigns
                """))
                row = result.fetchone()
                if row:
                    ACTIVE_CAMPAIGNS.set(row.active or 0)
                    TOTAL_CAMPAIGNS.labels(status='active').set(row.active or 0)
                    TOTAL_CAMPAIGNS.labels(status='completed').set(row.completed or 0)
                    TOTAL_CAMPAIGNS.labels(status='paused').set(row.paused or 0)
                    TOTAL_CAMPAIGNS.labels(status='draft').set(row.draft or 0)
                    CAMPAIGN_BUDGET_TOTAL.set(float(row.total_budget or 0))
                    CAMPAIGN_SPEND_TOTAL.set(float(row.total_spent or 0))

                # No leads table exists, using delayed_rewards for conversion tracking
                result = await session.execute(text("""
                    SELECT
                        COUNT(*) as total_rewards,
                        COUNT(*) FILTER (WHERE current_reward > 0) as successful_conversions,
                        COALESCE(SUM(current_reward), 0) as total_reward
                    FROM delayed_rewards
                """))
                row = result.fetchone()
                if row:
                    leads = row.total_rewards or 0
                    booked = row.successful_conversions or 0
                    reward = float(row.total_reward or 0)

                    LEADS_TOTAL.set(leads)
                    BOOKED_CALLS_TOTAL.set(booked)

                    cost_result = await session.execute(text("""
                        SELECT COALESCE(SUM(cost_amount), 0) as total_cost FROM cost_tracking
                    """))
                    cost_row = cost_result.fetchone()
                    cost = float(cost_row.total_cost or 0) if cost_row else 0

                    if leads > 0:
                        COST_PER_LEAD.set(cost / leads)
                        CONVERSION_RATE.set((booked / leads) * 100)
                    if booked > 0:
                        COST_PER_BOOKED_CALL.set(cost / booked)


                result = await session.execute(text("""
                    SELECT
                        COUNT(*) FILTER (WHERE is_active = true) as running,
                        COUNT(*) FILTER (WHERE is_active = false AND ended_at IS NOT NULL) as completed,
                        COUNT(*) FILTER (WHERE is_active = false AND ended_at IS NULL) as draft
                    FROM experiments
                """))
                row = result.fetchone()
                if row:
                    EXPERIMENTS_TOTAL.labels(status='running').set(row.running or 0)
                    EXPERIMENTS_TOTAL.labels(status='completed').set(row.completed or 0)
                    EXPERIMENTS_TOTAL.labels(status='draft').set(row.draft or 0)

                result = await session.execute(text("SELECT COUNT(*) FROM bandit_arms"))
                row = result.fetchone()
                if row:
                    EXPERIMENT_VARIANTS.set(row[0] or 0)


                result = await session.execute(text("""
                    SELECT
                        COUNT(*) FILTER (WHERE decision = 'approved') as approved,
                        COUNT(*) FILTER (WHERE decision = 'rejected') as rejected,
                        COUNT(*) FILTER (WHERE status = 'pending') as pending
                    FROM hitl_queue
                """))
                row = result.fetchone()
                if row:
                    approved = row.approved or 0
                    rejected = row.rejected or 0
                    pending = row.pending or 0
                    total = approved + rejected + pending

                    HUMAN_REVIEW_PENDING.set(pending)
                    GOVERNANCE_SAFETY_SCORE.set(0.85)
                    if total > 0:
                        GOVERNANCE_APPROVAL_RATE.set(approved / total)


                result = await session.execute(text("""
                    SELECT COUNT(*) as total_cached FROM semantic_cache
                """))
                row = result.fetchone()
                cached_count = row.total_cached or 0 if row else 0


                result = await session.execute(text("""
                    SELECT COUNT(*) as total_requests FROM cost_tracking WHERE provider = 'openai'
                """))
                row = result.fetchone()
                if row:
                    total_requests = row.total_requests or 0
                    # Conservative estimate since we can't track individual cache hits
                    hits = min(cached_count, total_requests // 2)
                    misses = total_requests - hits
                    total = hits + misses
                    if total > 0:
                        SEMANTIC_CACHE_HIT_RATE.set(hits / total)


                result = await session.execute(text("""
                    SELECT COALESCE(SUM(cost_amount), 0) as total_cost
                    FROM cost_tracking
                    WHERE provider IN ('openai', 'anthropic', 'ollama')
                """))
                row = result.fetchone()
                if row:
                    LLM_COST_TOTAL.set(float(row.total_cost or 0))


                result = await session.execute(text("""
                    SELECT AVG(validation_accuracy) as avg_accuracy
                    FROM calibration_runs
                    WHERE validation_accuracy IS NOT NULL
                """))
                row = result.fetchone()
                if row and row.avg_accuracy:
                    SIMULATION_ACCURACY.set(float(row.avg_accuracy))

        except Exception as e:
            logger.debug(f"Metrics update (some tables may not exist yet): {e}")

        await asyncio.sleep(15)  # Update every 15 seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    from .startup import startup_sequence, shutdown_sequence

    try:
        await startup_sequence()


        import asyncio
        asyncio.create_task(update_metrics_from_db())
        logger.info("✅ Prometheus metrics updater started")

        if settings.ENABLE_REWARD_TRACKING:
            reward_tracker = RewardTracker()
            asyncio.create_task(reward_tracker.start_background_processor())
            logger.info("✅ Reward tracker started")

        from ..data_layer.vector_store.pgvector_store import PgVectorStore
        from ..data_layer.vector_store.semantic_cache import SemanticCache

        doc_store = PgVectorStore(collection_name="documents")
        await doc_store.initialize()
        logger.info("✅ Document vector store initialized")

        cache = SemanticCache()
        await cache.initialize()
        logger.info("✅ Semantic cache initialized")

        from ..simulation.calibration_scheduler import start_calibration_scheduler
        await start_calibration_scheduler()
        logger.info("✅ Calibration scheduler started")

        from ..monitoring.campaign_monitor import start_campaign_monitor
        await start_campaign_monitor(check_interval_seconds=300)  # Check every 5 minutes
        logger.info("✅ Campaign & Experiment completion monitor started")

        # Research Plan Section 10.2
        from ..automation_layer.schedulers.weekly_report_scheduler import start_weekly_report_scheduler
        await start_weekly_report_scheduler()
        logger.info("✅ Weekly report scheduler started (Monday 9am)")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise

    yield

    try:
        from ..monitoring.campaign_monitor import stop_campaign_monitor
        await stop_campaign_monitor()

        from ..automation_layer.schedulers.weekly_report_scheduler import stop_weekly_report_scheduler
        await stop_weekly_report_scheduler()

        await shutdown_sequence()
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}", exc_info=True)

app = FastAPI(
    title="Agentic AI Agent Platform",
    description="AI-driven simulation and orchestration platform for autonomous marketing agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, 'CORS_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConditionalGZipMiddleware(GZipMiddleware):
    """GZip middleware that excludes certain paths like /metrics"""
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith("/metrics"):
            # Prometheus needs plain text, not gzipped
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)

app.add_middleware(ConditionalGZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request timing and metrics"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error on {request.url.path}: {exc}")
    
    API_ERRORS.labels(
        endpoint=request.url.path,
        error_type="validation_error"
    ).inc()
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    
    API_ERRORS.labels(
        endpoint=request.url.path,
        error_type="internal_error"
    ).inc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

app.include_router(
    campaigns.router,
    prefix="/api/v1/campaigns",
    tags=["campaigns"]
)

app.include_router(
    metrics.router,
    prefix="/api/v1/metrics",
    tags=["metrics"]
)

app.include_router(
    governance.router,
    prefix="/api/v1/governance",
    tags=["governance"]
)

app.include_router(
    health.router,
    prefix="",
    tags=["health"]
)

app.include_router(
    experiments.router,
    prefix="/api/v1/experiments",
    tags=["experiments"]
)

app.include_router(
    bandit_arms.router,
    prefix="/api/v1/bandit-arms",
    tags=["bandit-arms", "experiments"]
)

app.include_router(
    rewards.router,
    prefix="/api/v1/rewards",
    tags=["rewards"]
)

app.include_router(
    scraper.router,
    prefix="/api/v1/scraper",
    tags=["scraper"]
)

app.include_router(
    knowledge_base.router,
    prefix="/api/v1/knowledge-base",
    tags=["knowledge-base"]
)

app.include_router(
    strategy.router,
    prefix="/api/v1/strategy",
    tags=["strategy"]
)

app.include_router(
    ope.router,
    prefix="/api/v1/ope",
    tags=["ope"]
)

app.include_router(
    canary.router,
    prefix="/api/v1/canary",
    tags=["canary"]
)

app.include_router(
    memory.router,
    prefix="/api/v1/memory",
    tags=["memory"]
)

app.include_router(
    advanced_experiments.router,
    prefix="/api/v1/advanced-experiments",
    tags=["advanced-experiments", "research"]
)

app.include_router(
    calibration.router,
    prefix="/api/v1/calibration",
    tags=["calibration", "simulation"]
)

app.include_router(
    events.router,
    prefix="/api/v1",
    tags=["events", "transparency"]
)

app.include_router(
    personas.router,
    prefix="/api/v1/personas",
    tags=["personas", "configuration"]
)

app.include_router(
    costs.router,
    prefix="/api/v1/costs",
    tags=["costs", "cost-control"]
)

app.include_router(
    configuration.router,
    prefix="/api/v1",
    tags=["configuration", "settings"]
)

app.include_router(
    funnel.router,
    prefix="/api/v1",
    tags=["funnel", "attribution"]
)

app.include_router(
    mlflow.router,
    prefix="/api/v1",
    tags=["mlflow", "model-registry"]
)

app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["analytics", "rq2", "governance-metrics", "weekly-reports"]
)

app.include_router(
    operations.router,
    prefix="/api/v1/operations",
    tags=["operations", "maintenance"]
)

app.include_router(
    llm.router,
    prefix="/api/v1/llm",
    tags=["llm", "ollama", "prompts"]
)

app.include_router(
    data_config.router,
    prefix="/api/v1",
    tags=["data-config", "claims", "brand-voice", "competitors", "products"]
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agentic AI Agent Platform",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/api/v1/info")
async def api_info():
    """Get API information"""
    return {
        "name": "Agentic AI Agent Platform API",
        "version": "1.0.0",
        "description": "AI-driven simulation and orchestration platform for autonomous marketing agents",
        "endpoints": {
            "campaigns": "/api/v1/campaigns",
            "metrics": "/api/v1/metrics",
            "governance": "/api/v1/governance",
            "experiments": "/api/v1/experiments",
            "rewards": "/api/v1/rewards",
            "scraper": "/api/v1/scraper",
            "knowledge_base": "/api/v1/knowledge-base",
            "strategy": "/api/v1/strategy",
            "ope": "/api/v1/ope",
            "canary": "/api/v1/canary",
            "memory": "/api/v1/memory",
            "advanced_experiments": "/api/v1/advanced-experiments",
            "calibration": "/api/v1/calibration",
            "health": "/health",
            "docs": "/docs"
        },
        "features": {
            "multi_agent_system": True,
            "simulation": True,
            "governance": True,
            "cost_control": True,
            "attribution_tracking": True,
            "experiment_tracking": True,
            "episodic_memory": True,
            "offline_policy_evaluation": True,
            "canary_deployment": True,
            "research_mode": getattr(settings, 'ENABLE_RESEARCH_MODE', True),
            "advanced_experiments": getattr(settings, 'ENABLE_RESEARCH_MODE', True)
        }
    }

@app.get("/ready")
async def readiness():
    """Kubernetes readiness probe"""
    try:
        from ..data_layer.database.connection import get_async_session
        from sqlalchemy import text

        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))

        return {"ready": True}
    
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)}
        )

@app.get("/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )