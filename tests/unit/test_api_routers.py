# tests/unit/test_api_routers.py
"""
Unit tests for API-related services and integrations.

Since FastAPI is not installed in the test environment, we test the
underlying service functions directly rather than through HTTP endpoints.

Covers:
  - MLflowModelRegistry (learning module integration)
  - Cost control BudgetManager creation
  - ConfigurationService init and defaults
  - QueueService enqueuing
  - Scheduler info

All external dependencies (database, Redis, MLflow) are mocked.
"""
import sys
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# ---------------------------------------------------------------------------
# Pre-mock heavy dependencies
# ---------------------------------------------------------------------------
_STUB_MODULES = {}


def _ensure_stub(name):
    if name not in sys.modules:
        _STUB_MODULES[name] = MagicMock()
        sys.modules[name] = _STUB_MODULES[name]


# FastAPI + dependencies
_ensure_stub("fastapi")
_ensure_stub("fastapi.testclient")
_ensure_stub("fastapi.responses")
_ensure_stub("starlette")
_ensure_stub("starlette.responses")
_ensure_stub("uvicorn")
_ensure_stub("pydantic")

_ensure_stub("redis")
_ensure_stub("redis.asyncio")
_ensure_stub("pgvector")
_ensure_stub("pgvector.sqlalchemy")
_ensure_stub("mlflow")
_ensure_stub("mlflow.pyfunc")
_ensure_stub("rq")
_ensure_stub("rq.job")
_ensure_stub("rq_scheduler")
_ensure_stub("httpx")
_ensure_stub("aiohttp")
_ensure_stub("sendgrid")
_ensure_stub("sendgrid.helpers")
_ensure_stub("sendgrid.helpers.mail")

for _m in [
    "langchain", "langchain.chat_models", "langchain.prompts",
    "langchain.schema", "langchain.callbacks",
    "langchain_openai", "langchain_community",
    "langchain_community.callbacks",
    "sentence_transformers",
    "torch", "torch.nn", "torch.cuda",
    "apify_client",
    "watchdog", "watchdog.events", "watchdog.observers",
]:
    _ensure_stub(_m)


# ===========================================================================
# 1. MLflow Registry Tests
# ===========================================================================

class TestMLflowRegistry:
    """Tests for src/ai_layer/learning/mlflow_integration.py

    MLflowModelRegistry manages experiment tracking and model versioning.
    The module uses `import mlflow` at the top level (mocked via sys.modules)
    and stores it as `self._mlflow` after `_ensure_initialized`.
    """

    def _make_registry(self):
        from src.ai_layer.learning.mlflow_integration import MLflowModelRegistry
        registry = object.__new__(MLflowModelRegistry)
        registry.tracking_uri = "http://mlflow:5000"
        registry.experiment_name = "test-experiment"
        registry._initialized = False
        registry._mlflow = None
        registry.client = MagicMock()
        return registry

    def test_init(self):
        registry = self._make_registry()
        assert registry is not None
        assert registry.tracking_uri == "http://mlflow:5000"
        assert registry._initialized is False

    def test_ensure_initialized_sets_flag(self):
        """After _ensure_initialized, _initialized should be True."""
        registry = self._make_registry()
        # The mlflow mock is already in sys.modules from _ensure_stub
        registry._ensure_initialized()
        assert registry._initialized is True
        assert registry._mlflow is not None

    def test_get_mlflow_registry_singleton(self):
        """get_mlflow_registry returns a singleton instance."""
        import src.ai_layer.learning.mlflow_integration as mod
        original = mod._registry
        try:
            mod._registry = None
            reg = mod.get_mlflow_registry()
            assert reg is not None
            assert mod._registry is reg
        finally:
            mod._registry = original

    def test_log_bandit_policy_basic(self):
        """log_bandit_policy should call _mlflow methods."""
        registry = self._make_registry()
        registry._initialized = True

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        registry._mlflow = mock_mlflow

        result = registry.log_bandit_policy(
            policy_name="test_policy",
            policy_type="thompson_sampling",
            arms_data=[
                {"arm_id": "arm-1", "pulls": 100, "reward": 0.8},
                {"arm_id": "arm-2", "pulls": 80, "reward": 0.6},
            ],
            metrics={"total_reward": 0.75, "regret": 0.1},
        )
        # Should have called start_run
        mock_mlflow.start_run.assert_called()


# ===========================================================================
# 2. Cost Control BudgetManager Tests
# ===========================================================================

class TestCostControlAPI:
    """Tests for BudgetManager used by cost control API."""

    def test_budget_manager_creation(self):
        """Verify BudgetManager can be created with correct constructor."""
        from src.cost_control.budget_manager import BudgetManager
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("src.cost_control.budget_manager.settings") as mock_settings:
            mock_settings.MAX_DAILY_API_COST = 100.0
            mock_settings.MAX_CAMPAIGN_COST = 500.0
            manager = BudgetManager(db_session=mock_db, redis_client=mock_redis)

        assert manager.db is mock_db
        assert manager.redis is mock_redis

    @pytest.mark.asyncio
    async def test_budget_check_returns_bool(self):
        """Verify check_budget returns boolean."""
        from src.cost_control.budget_manager import BudgetManager
        mgr = object.__new__(BudgetManager)
        mgr.db = AsyncMock()
        mgr.redis = AsyncMock()
        mgr.daily_limit = 100.0
        mgr.campaign_limit = 500.0

        mock_campaign = MagicMock()
        mock_campaign.budget_spent = 10.0
        mock_campaign.budget_total = 100.0

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_campaign
        mgr.db.execute = AsyncMock(return_value=mock_result)

        result = await mgr.check_budget("camp-1", estimated_cost=5.0)
        assert isinstance(result, bool)
        assert result is True


# ===========================================================================
# 3. Configuration Service Tests
# ===========================================================================

class TestConfigServiceAPI:
    """Tests for ConfigurationService used by config API."""

    def test_default_configs_exist(self):
        from src.config.configuration_service import DEFAULT_CONFIGURATIONS
        assert isinstance(DEFAULT_CONFIGURATIONS, dict)
        assert len(DEFAULT_CONFIGURATIONS) > 10
        assert "USE_LOCAL_LLM" in DEFAULT_CONFIGURATIONS
        assert "MAX_DAILY_API_COST" in DEFAULT_CONFIGURATIONS

    def test_configs_have_required_fields(self):
        from src.config.configuration_service import DEFAULT_CONFIGURATIONS
        for key, config in DEFAULT_CONFIGURATIONS.items():
            assert "category" in config, f"Config {key} missing category"
            assert "default" in config, f"Config {key} missing default"
            assert "description" in config, f"Config {key} missing description"

    def test_secret_configs_flagged(self):
        from src.config.configuration_service import DEFAULT_CONFIGURATIONS
        secret_keys = [
            k for k, v in DEFAULT_CONFIGURATIONS.items() if v.get("is_secret")
        ]
        assert len(secret_keys) > 0
        assert "OPENAI_API_KEY" in secret_keys

    def test_configuration_service_init(self):
        from src.config.configuration_service import ConfigurationService
        mock_db = MagicMock()
        service = ConfigurationService(db_session=mock_db)
        assert service.db is mock_db
        assert service._cache == {}
        assert service._cache_loaded is False


# ===========================================================================
# 4. QueueService Tests (API-facing)
# ===========================================================================

class TestQueueServiceAPI:
    """Tests for QueueService used by worker API."""

    def test_queue_service_init(self):
        with patch("src.worker.queue_service.settings") as mock_settings:
            mock_settings.REDIS_URL = "redis://localhost"
            from src.worker.queue_service import QueueService
            service = QueueService(redis_url="redis://localhost")
            assert service.redis_url == "redis://localhost"

    def test_queue_service_enqueue(self):
        with patch("src.worker.queue_service.settings") as mock_settings:
            mock_settings.REDIS_URL = "redis://localhost"
            from src.worker.queue_service import QueueService
            service = QueueService(redis_url="redis://localhost")

            mock_queue = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "job-123"
            mock_queue.enqueue.return_value = mock_job

            with patch.object(service, 'get_queue', return_value=mock_queue):
                result = service.enqueue_delayed_rewards()
                assert result.id == "job-123"


# ===========================================================================
# 5. Scheduler Info Tests (API-facing)
# ===========================================================================

class TestSchedulerAPI:
    """Tests for get_scheduler_info used by status API."""

    def test_get_scheduler_info_success(self):
        with patch("src.worker.scheduler.Scheduler") as MockScheduler, \
             patch("src.worker.scheduler.Redis") as MockRedis, \
             patch("src.worker.scheduler.settings") as mock_settings:
            mock_job = MagicMock()
            mock_job.id = "job-1"
            mock_job.func_name = "process_delayed_rewards"
            mock_job.created_at = datetime.utcnow()
            mock_job.meta = {"description": "Hourly reward processing"}

            mock_scheduler = MagicMock()
            mock_scheduler.get_jobs.return_value = [mock_job]
            MockScheduler.return_value = mock_scheduler
            MockRedis.from_url = MagicMock(return_value=MagicMock())
            mock_settings.REDIS_URL = "redis://localhost"

            from src.worker.scheduler import get_scheduler_info
            info = get_scheduler_info()
            assert info["scheduled_jobs"] == 1
            assert len(info["jobs"]) == 1

    def test_get_scheduler_info_error(self):
        with patch("src.worker.scheduler.Scheduler") as MockScheduler, \
             patch("src.worker.scheduler.Redis") as MockRedis, \
             patch("src.worker.scheduler.settings") as mock_settings:
            MockRedis.from_url = MagicMock(side_effect=Exception("Connection refused"))
            mock_settings.REDIS_URL = "redis://localhost"

            from src.worker.scheduler import get_scheduler_info
            info = get_scheduler_info()
            assert info["scheduled_jobs"] == 0
            assert "error" in info
