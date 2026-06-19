# tests/unit/test_remaining_modules.py
"""
Unit tests for remaining modules not covered by other test files.

Covers:
  - BudgetManager (cost control, budget enforcement)
  - TokenTracker (LLM token/cost tracking)
  - ConfigurationService (dynamic config)

All external dependencies are mocked.
"""
import sys
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Pre-mock heavy dependencies
# ---------------------------------------------------------------------------
_STUB_MODULES = {}


def _ensure_stub(name):
    if name not in sys.modules:
        _STUB_MODULES[name] = MagicMock()
        sys.modules[name] = _STUB_MODULES[name]


_ensure_stub("redis")
_ensure_stub("redis.asyncio")
_ensure_stub("pgvector")
_ensure_stub("pgvector.sqlalchemy")
_ensure_stub("mlflow")

for _m in [
    "langchain", "langchain.chat_models", "langchain.prompts",
    "langchain.schema", "langchain.callbacks",
    "langchain_openai", "langchain_community",
    "langchain_community.callbacks",
    "sentence_transformers",
]:
    _ensure_stub(_m)


# ===========================================================================
# 1. BudgetManager Tests  (async, db-backed)
# ===========================================================================

class TestBudgetManager:
    """Tests for src/cost_control/budget_manager.py

    BudgetManager.__init__(db_session, redis_client) — requires both.
    All public methods (check_budget, track_cost, get_cost_summary,
    enforce_limits) are async and hit the DB so we bypass __init__
    via object.__new__ and mock the db / redis attributes.
    """

    def _make_manager(self, daily_limit=100.0, campaign_limit=500.0):
        from src.cost_control.budget_manager import BudgetManager
        mgr = object.__new__(BudgetManager)
        mgr.db = AsyncMock()
        mgr.redis = AsyncMock()
        mgr.daily_limit = daily_limit
        mgr.campaign_limit = campaign_limit
        return mgr

    def test_init(self):
        mgr = self._make_manager()
        assert mgr is not None
        assert mgr.daily_limit == 100.0

    # --- check_budget ---

    @pytest.mark.asyncio
    async def test_check_budget_within_limit(self):
        mgr = self._make_manager()
        # Mock campaign row with budget_spent = 10, budget_total = 100
        mock_campaign = MagicMock()
        mock_campaign.budget_spent = 10.0
        mock_campaign.budget_total = 100.0

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_campaign
        mgr.db.execute = AsyncMock(return_value=mock_result)

        result = await mgr.check_budget("camp-1", estimated_cost=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_budget_exceeds(self):
        mgr = self._make_manager()
        mock_campaign = MagicMock()
        mock_campaign.budget_spent = 95.0
        mock_campaign.budget_total = 100.0

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_campaign
        mgr.db.execute = AsyncMock(return_value=mock_result)

        result = await mgr.check_budget("camp-1", estimated_cost=10.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_budget_campaign_not_found(self):
        mgr = self._make_manager()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mgr.db.execute = AsyncMock(return_value=mock_result)

        # Campaign not found → returns False (conservative)
        result = await mgr.check_budget("nonexistent", estimated_cost=5.0)
        assert result is False

    # --- track_cost ---

    @pytest.mark.asyncio
    async def test_track_cost_success(self):
        mgr = self._make_manager()
        mgr.db.add = MagicMock()
        mgr.db.commit = AsyncMock()
        mgr.redis.incrbyfloat = AsyncMock()
        mgr.redis.expire = AsyncMock()
        mgr.redis.get = AsyncMock(return_value="15.0")

        with patch("src.cost_control.budget_manager.CostTracking"):
            result = await mgr.track_cost(
                source_type="content_generation",
                cost_amount=5.0,
                campaign_id="camp-1",
                metadata={"provider": "openai", "tokens_prompt": 100, "tokens_completion": 50},
            )
        assert result is True
        mgr.db.add.assert_called_once()
        mgr.db.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_track_cost_updates_redis_daily(self):
        mgr = self._make_manager()
        mgr.db.add = MagicMock()
        mgr.db.commit = AsyncMock()
        mgr.redis.incrbyfloat = AsyncMock()
        mgr.redis.expire = AsyncMock()
        mgr.redis.get = AsyncMock(return_value="50.0")

        with patch("src.cost_control.budget_manager.CostTracking"):
            await mgr.track_cost(
                source_type="api_calls",
                cost_amount=10.0,
            )
        mgr.redis.incrbyfloat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_track_cost_failure_returns_false(self):
        mgr = self._make_manager()
        mgr.db.add = MagicMock(side_effect=Exception("DB down"))
        mgr.db.rollback = AsyncMock()

        with patch("src.cost_control.budget_manager.CostTracking"):
            result = await mgr.track_cost(
                source_type="api_calls",
                cost_amount=5.0,
            )
        assert result is False

    # --- get_cost_summary ---

    @pytest.mark.asyncio
    async def test_get_cost_summary_returns_dict(self):
        mgr = self._make_manager()
        mock_row = MagicMock()
        mock_row.total_cost = 50.0
        mock_row.total_transactions = 10
        mock_row.source_type = "content_generation"
        mock_row.avg_cost = 5.0

        mock_result = MagicMock()
        mock_result.all.return_value = [mock_row]
        mgr.db.execute = AsyncMock(return_value=mock_result)
        mgr.redis.get = AsyncMock(return_value=None)

        summary = await mgr.get_cost_summary(days=7)
        assert isinstance(summary, dict)
        assert summary["total_cost"] == 50.0
        assert summary["period_days"] == 7

    @pytest.mark.asyncio
    async def test_get_cost_summary_empty(self):
        mgr = self._make_manager()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mgr.db.execute = AsyncMock(return_value=mock_result)
        mgr.redis.get = AsyncMock(return_value=None)

        summary = await mgr.get_cost_summary()
        assert summary["total_cost"] == 0

    # --- enforce_limits ---

    @pytest.mark.asyncio
    async def test_enforce_limits_within_budget(self):
        mgr = self._make_manager()
        # Mock check_budget to return True
        with patch.object(mgr, "check_budget", new_callable=AsyncMock, return_value=True):
            result = await mgr.enforce_limits("camp-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_enforce_limits_exceeded_pauses_campaign(self):
        mgr = self._make_manager()
        mgr.db.commit = AsyncMock()
        with patch.object(mgr, "check_budget", new_callable=AsyncMock, return_value=False):
            result = await mgr.enforce_limits("camp-1")
        assert result is False
        mgr.db.commit.assert_awaited()


# ===========================================================================
# 2. TokenTracker Tests
# ===========================================================================

class TestTokenTracker:
    """Tests for src/cost_control/token_tracker.py"""

    def _make_tracker(self):
        from src.cost_control.token_tracker import TokenTracker
        return TokenTracker()

    def test_init(self):
        tracker = self._make_tracker()
        assert tracker is not None
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0

    def test_track_usage(self):
        tracker = self._make_tracker()
        result = tracker.track_usage(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert tracker.total_tokens == 150
        assert isinstance(result, dict)
        assert result["total_tokens"] == 150

    def test_track_usage_accumulates(self):
        tracker = self._make_tracker()
        tracker.track_usage(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        tracker.track_usage(model="gpt-4", prompt_tokens=200, completion_tokens=100)
        assert tracker.total_tokens == 450

    def test_track_usage_multiple_models(self):
        tracker = self._make_tracker()
        tracker.track_usage(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        tracker.track_usage(model="gpt-3.5-turbo", prompt_tokens=200, completion_tokens=100)
        assert tracker.total_tokens == 450

    def test_track_usage_cost_calculated(self):
        tracker = self._make_tracker()
        result = tracker.track_usage(
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        # GPT-4: prompt 0.03/1K, completion 0.06/1K
        expected = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert abs(result["total_cost"] - expected) < 0.001

    def test_track_usage_unknown_model_uses_default(self):
        tracker = self._make_tracker()
        result = tracker.track_usage(
            model="unknown-model-xyz",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        # Default costs: prompt 0.01/1K, completion 0.03/1K
        expected = (1000 / 1000) * 0.01 + (500 / 1000) * 0.03
        assert abs(result["total_cost"] - expected) < 0.001

    def test_get_summary(self):
        tracker = self._make_tracker()
        tracker.track_usage(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        tracker.track_usage(model="gpt-3.5-turbo", prompt_tokens=500, completion_tokens=200)

        summary = tracker.get_summary()
        assert isinstance(summary, dict)
        assert summary["total_tokens"] == 850
        assert summary["total_cost"] > 0
        assert "by_model" in summary
        assert "gpt-4" in summary["by_model"]
        assert "gpt-3.5-turbo" in summary["by_model"]

    def test_get_summary_empty(self):
        tracker = self._make_tracker()
        summary = tracker.get_summary()
        assert summary["total_tokens"] == 0
        assert summary["total_cost"] == 0.0

    def test_usage_history(self):
        tracker = self._make_tracker()
        tracker.track_usage(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        assert len(tracker.usage_history) == 1
        entry = tracker.usage_history[0]
        assert entry["model"] == "gpt-4"
        assert entry["prompt_tokens"] == 100
        assert entry["completion_tokens"] == 50

    def test_get_summary_recent_calls(self):
        tracker = self._make_tracker()
        for i in range(15):
            tracker.track_usage(model="gpt-4", prompt_tokens=10, completion_tokens=5)
        summary = tracker.get_summary()
        # recent_calls should only have last 10
        assert len(summary["recent_calls"]) == 10


# ===========================================================================
# 3. ConfigurationService Tests
# ===========================================================================

class TestConfigurationService:
    """Tests for src/config/configuration_service.py"""

    def _make_service(self):
        from src.config.configuration_service import ConfigurationService
        service = object.__new__(ConfigurationService)
        service.db = MagicMock()
        service._cache = {}
        service._cache_loaded = False
        return service

    def test_init(self):
        service = self._make_service()
        assert service is not None
        assert service._cache == {}

    def test_cache_starts_empty(self):
        service = self._make_service()
        assert service._cache_loaded is False
        assert len(service._cache) == 0
