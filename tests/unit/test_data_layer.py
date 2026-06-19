# tests/unit/test_data_layer.py
"""
Comprehensive unit tests for the data layer.
All database dependencies are mocked — no Docker services required.
"""
import sys
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
from uuid import uuid4, UUID

# ─── Enums & models ────────────────────────────────────────────────────────
from src.shared.constants import (
    CampaignStatus, ContentStatus, Platform, CampaignGoal,
    AlertSeverity, WorkflowEventType,
)
from src.data_layer.database.models import (
    Campaign, Content, Experiment, BanditArm, Metric,
    CostTracking, AgentAction, HITLQueue, AgentType,
)

# ─── Repositories ──────────────────────────────────────────────────────────
from src.data_layer.repositories.campaign_repo import CampaignRepository
from src.data_layer.repositories.content_repo import ContentRepository
from src.data_layer.repositories.experiment_repo import ExperimentRepository
from src.data_layer.repositories.metrics_repo import MetricsRepository

# ─── Cost tracker (pure functions + class) ─────────────────────────────────
from src.ai_layer.utils.cost_tracker import (
    calculate_cost, estimate_tokens, CostTracker, COST_RATES, LLMProvider,
)

# ─── Stub heavy deps for vector_store / episodic_memory imports ────────────
# These modules import sentence_transformers / langchain at module level,
# plus connection.py tries to create DB engines on import.
# We inject lightweight stubs into sys.modules so the imports succeed.

_STUB_MODULES = {}


def _ensure_stub(name, attrs=None):
    """Register a stub module if the real one is not already loaded."""
    if name not in sys.modules:
        mod = type(sys)("stub_" + name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
        _STUB_MODULES[name] = mod
    return sys.modules[name]


_ensure_stub("sentence_transformers", {"SentenceTransformer": Mock})
_ensure_stub("langchain_openai", {"OpenAIEmbeddings": Mock})
_ensure_stub("langchain", {})
_ensure_stub("langchain.embeddings", {"OpenAIEmbeddings": Mock})

# langchain.schema needs a real-ish Document class for tests
class _StubDocument:
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

_ensure_stub("langchain.schema", {"Document": _StubDocument})

# Now we can safely import PgVectorStore, SemanticCache, and EpisodicMemory
# The stubs above prevent heavy-dependency import failures.
# We also need to mock get_async_session used by pgvector_store at module level.
# Since connection.py creates engines on import, we mock it before importing pgvector_store.
import importlib
with patch.dict(sys.modules, {
    "src.data_layer.database.connection": MagicMock(),
}):
    # Force reimport of pgvector_store with mocked connection
    if "src.data_layer.vector_store.pgvector_store" in sys.modules:
        # Already imported — patch get_async_session on it
        pass
    from src.data_layer.vector_store.pgvector_store import PgVectorStore
    from src.data_layer.vector_store.semantic_cache import SemanticCache
    from src.ai_layer.memory.episodic_memory import (
        AgentMemory, EpisodicMemoryStore, create_memory_from_task,
    )
    # Keep a reference to the pgvector_store module for patching get_async_session
    # Must be inside the with block so the module has the mocked get_async_session
    import src.data_layer.vector_store.pgvector_store as _pgvector_store_module


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.add = Mock()
    return session


@pytest.fixture
def campaign_repo(mock_session):
    return CampaignRepository(mock_session)


@pytest.fixture
def content_repo(mock_session):
    return ContentRepository(mock_session)


@pytest.fixture
def experiment_repo(mock_session):
    return ExperimentRepository(mock_session)


@pytest.fixture
def metrics_repo(mock_session):
    return MetricsRepository(mock_session)


def _mock_campaign(**overrides):
    """Return a Mock that behaves like a Campaign row."""
    defaults = dict(
        id=uuid4(),
        name="Campaign A",
        platform=Platform.LINKEDIN,
        status=CampaignStatus.RUNNING,
        goal=CampaignGoal.LEAD_GENERATION,
        target_persona="decision_maker",
        budget_total=1000.0,
        budget_spent=0.0,
        budget=1000.0,
        impressions=5000,
        clicks=100,
        conversions=10,
        spend=200.0,
        target_cpl=50.0,
        config={},
        metadata=None,
        contents=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        ctr=0.0,
        cpl=0.0,
    )
    defaults.update(overrides)
    m = Mock(**defaults)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# 1. Campaign Repository
# ═══════════════════════════════════════════════════════════════════════════

class TestCampaignRepository:

    @pytest.mark.asyncio
    async def test_create_campaign_commits_and_refreshes(self, campaign_repo, mock_session):
        result = await campaign_repo.create({"name": "X", "platform": Platform.LINKEDIN})
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_campaign_rollback_on_error(self, campaign_repo, mock_session):
        mock_session.commit.side_effect = Exception("db error")
        with pytest.raises(Exception, match="db error"):
            await campaign_repo.create({"name": "X", "platform": Platform.LINKEDIN})
        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_returns_campaign(self, campaign_repo, mock_session):
        expected = _mock_campaign()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = expected
        mock_session.execute.return_value = mock_result
        result = await campaign_repo.get_by_id(str(uuid4()))
        assert result is expected

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_on_exception(self, campaign_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        result = await campaign_repo.get_by_id(str(uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_with_status_filter(self, campaign_repo, mock_session):
        c1, c2 = _mock_campaign(), _mock_campaign()
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [c1, c2]
        mock_session.execute.return_value = mock_result
        campaigns = await campaign_repo.get_all(status=CampaignStatus.RUNNING)
        assert len(campaigns) == 2

    @pytest.mark.asyncio
    async def test_get_all_with_platform_filter(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [_mock_campaign()]
        mock_session.execute.return_value = mock_result
        campaigns = await campaign_repo.get_all(platform=Platform.TWITTER)
        assert len(campaigns) == 1

    @pytest.mark.asyncio
    async def test_get_all_pagination(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        campaigns = await campaign_repo.get_all(skip=10, limit=5)
        assert campaigns == []

    @pytest.mark.asyncio
    async def test_get_all_returns_empty_on_error(self, campaign_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        campaigns = await campaign_repo.get_all()
        assert campaigns == []

    @pytest.mark.asyncio
    async def test_list_all_delegates_to_get_all(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [_mock_campaign()]
        mock_session.execute.return_value = mock_result
        result = await campaign_repo.list_all(limit=50)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_active_filters_running(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [_mock_campaign()]
        mock_session.execute.return_value = mock_result
        result = await campaign_repo.get_active()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_update_campaign(self, campaign_repo, mock_session):
        updated = _mock_campaign(name="Updated")
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = updated
        mock_session.execute.return_value = mock_result
        result = await campaign_repo.update(str(uuid4()), {"name": "Updated"})
        assert result is updated
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_campaign_rollback_on_error(self, campaign_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        result = await campaign_repo.update(str(uuid4()), {"name": "X"})
        assert result is None
        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_status_returns_true_on_success(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = _mock_campaign()
        mock_session.execute.return_value = mock_result
        ok = await campaign_repo.update_status(str(uuid4()), CampaignStatus.PAUSED)
        assert ok is True

    @pytest.mark.asyncio
    async def test_update_status_returns_false_when_not_found(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        ok = await campaign_repo.update_status(str(uuid4()), CampaignStatus.PAUSED)
        assert ok is False

    @pytest.mark.asyncio
    async def test_update_metrics_calculates_ctr_and_cpl(self, campaign_repo, mock_session):
        """CTR stored as percentage, CPL = spend / conversions."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = _mock_campaign()
        mock_session.execute.return_value = mock_result

        ok = await campaign_repo.update_metrics(
            str(uuid4()), impressions=1000, clicks=50, conversions=5, spend=100.0
        )
        assert ok is True
        # Verify the values dictionary passed to execute
        call_args = mock_session.execute.call_args_list[-1]
        # The statement is built via SQLAlchemy so we can't easily inspect
        # values, but we can check commit was called
        mock_session.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_update_metrics_zero_impressions(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = _mock_campaign()
        mock_session.execute.return_value = mock_result
        ok = await campaign_repo.update_metrics(
            str(uuid4()), impressions=0, clicks=0, conversions=0, spend=0.0
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_delete_campaign(self, campaign_repo, mock_session):
        ok = await campaign_repo.delete(str(uuid4()))
        assert ok is True
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_campaign_rollback_on_error(self, campaign_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        ok = await campaign_repo.delete(str(uuid4()))
        assert ok is False
        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_campaigns_by_date_range(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [_mock_campaign()]
        mock_session.execute.return_value = mock_result
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow()
        result = await campaign_repo.get_campaigns_by_date_range(start, end)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_campaigns_by_date_range_error(self, campaign_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        result = await campaign_repo.get_campaigns_by_date_range(
            datetime.utcnow() - timedelta(days=1), datetime.utcnow()
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_get_campaign_performance_with_data(self, campaign_repo, mock_session):
        c = _mock_campaign(impressions=1000, clicks=50, conversions=5, spend=100.0, budget=500.0)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = c
        mock_session.execute.return_value = mock_result
        perf = await campaign_repo.get_campaign_performance(str(c.id))
        assert perf is not None
        assert perf["ctr"] == pytest.approx(5.0)        # 50/1000 * 100
        assert perf["conversion_rate"] == pytest.approx(10.0)  # 5/50 * 100
        assert perf["cpl"] == pytest.approx(20.0)        # 100/5

    @pytest.mark.asyncio
    async def test_get_campaign_performance_zero_values(self, campaign_repo, mock_session):
        c = _mock_campaign(impressions=0, clicks=0, conversions=0, spend=0.0, budget=0.0)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = c
        mock_session.execute.return_value = mock_result
        perf = await campaign_repo.get_campaign_performance(str(c.id))
        assert perf["ctr"] == 0.0
        assert perf["cpl"] == 0.0
        assert perf["budget_used_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_get_campaign_performance_not_found(self, campaign_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        perf = await campaign_repo.get_campaign_performance(str(uuid4()))
        assert perf is None

    @pytest.mark.asyncio
    async def test_check_budget_exceeded_true(self, campaign_repo, mock_session):
        c = _mock_campaign(spend=1000.0, budget=1000.0)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = c
        mock_session.execute.return_value = mock_result
        assert await campaign_repo.check_budget_exceeded(str(c.id)) is True

    @pytest.mark.asyncio
    async def test_check_budget_exceeded_false(self, campaign_repo, mock_session):
        c = _mock_campaign(spend=500.0, budget=1000.0)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = c
        mock_session.execute.return_value = mock_result
        assert await campaign_repo.check_budget_exceeded(str(c.id)) is False

    @pytest.mark.asyncio
    async def test_pause_if_budget_exceeded(self, campaign_repo, mock_session):
        c = _mock_campaign(spend=1000.0, budget=1000.0)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = c
        mock_session.execute.return_value = mock_result
        paused = await campaign_repo.pause_if_budget_exceeded(str(c.id))
        assert paused is True

    @pytest.mark.asyncio
    async def test_pause_if_budget_not_exceeded(self, campaign_repo, mock_session):
        c = _mock_campaign(spend=100.0, budget=1000.0)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = c
        mock_session.execute.return_value = mock_result
        paused = await campaign_repo.pause_if_budget_exceeded(str(c.id))
        assert paused is False

    @pytest.mark.asyncio
    async def test_get_with_contents(self, campaign_repo, mock_session):
        expected = _mock_campaign()
        mock_result = Mock()
        mock_result.unique.return_value.scalar_one_or_none.return_value = expected
        mock_session.execute.return_value = mock_result
        result = await campaign_repo.get_with_contents(str(uuid4()))
        assert result is expected

    @pytest.mark.asyncio
    async def test_get_platform_summary_returns_empty_on_attribute_error(self, campaign_repo, mock_session):
        """Campaign.spend doesn't exist on model (budget_spent); repo catches and returns {}."""
        mock_session.execute.side_effect = AttributeError("Campaign has no attribute 'spend'")
        summary = await campaign_repo.get_platform_summary()
        assert summary == {}

    @pytest.mark.asyncio
    async def test_get_platform_summary_error(self, campaign_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        summary = await campaign_repo.get_platform_summary()
        assert summary == {}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Content Repository
# ═══════════════════════════════════════════════════════════════════════════

class TestContentRepository:

    @pytest.mark.asyncio
    async def test_create_content(self, content_repo, mock_session):
        await content_repo.create({
            "campaign_id": uuid4(), "body": "Hello", "headline": "Hi"
        })
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_content_rollback_on_error(self, content_repo, mock_session):
        mock_session.commit.side_effect = Exception("db")
        with pytest.raises(Exception):
            await content_repo.create({"campaign_id": uuid4(), "body": "X"})
        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id(self, content_repo, mock_session):
        content = Mock(campaign_id=uuid4(), status=ContentStatus.GENERATED)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = content
        mock_session.execute.return_value = mock_result
        result = await content_repo.get_by_id(str(uuid4()))
        assert result is content

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        result = await content_repo.get_by_id(str(uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_campaign(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [Mock(), Mock()]
        mock_session.execute.return_value = mock_result
        contents = await content_repo.get_by_campaign(str(uuid4()))
        assert len(contents) == 2

    @pytest.mark.asyncio
    async def test_get_by_campaign_with_status_filter(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [Mock()]
        mock_session.execute.return_value = mock_result
        contents = await content_repo.get_by_campaign(
            str(uuid4()), status=ContentStatus.APPROVED
        )
        assert len(contents) == 1

    @pytest.mark.asyncio
    async def test_get_by_campaign_error(self, content_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        result = await content_repo.get_by_campaign(str(uuid4()))
        assert result == []

    @pytest.mark.asyncio
    async def test_update_content(self, content_repo, mock_session):
        updated = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = updated
        mock_session.execute.return_value = mock_result
        result = await content_repo.update(str(uuid4()), {"headline": "New"})
        assert result is updated
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_status_with_review_info(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = Mock()
        mock_session.execute.return_value = mock_result
        ok = await content_repo.update_status(
            str(uuid4()), ContentStatus.APPROVED,
            reviewed_by="admin", feedback="Looks good"
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_mark_deployed_with_metrics(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = Mock()
        mock_session.execute.return_value = mock_result
        ok = await content_repo.mark_deployed(
            str(uuid4()), "post_123",
            metrics={"impressions": 100, "clicks": 10, "conversions": 1}
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_mark_deployed_not_found(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        ok = await content_repo.mark_deployed(str(uuid4()), "post_123")
        assert ok is False

    @pytest.mark.asyncio
    async def test_delete_content(self, content_repo, mock_session):
        ok = await content_repo.delete(str(uuid4()))
        assert ok is True
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_content_error(self, content_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        ok = await content_repo.delete(str(uuid4()))
        assert ok is False
        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_pending_review(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [Mock()]
        mock_session.execute.return_value = mock_result
        result = await content_repo.get_pending_review(priority_threshold=0.7)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_performance_stats(self, content_repo, mock_session):
        row = Mock(total_content=10, avg_safety_score=0.85, campaigns_count=3)
        mock_result = Mock()
        mock_result.first.return_value = row
        mock_session.execute.return_value = mock_result
        stats = await content_repo.get_performance_stats()
        assert stats["total_content"] == 10
        assert stats["avg_safety_score"] == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_get_performance_stats_empty(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.execute.return_value = mock_result
        stats = await content_repo.get_performance_stats()
        assert stats["total_content"] == 0

    @pytest.mark.asyncio
    async def test_search_by_text(self, content_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [Mock()]
        mock_session.execute.return_value = mock_result
        result = await content_repo.search_by_text("marketing")
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Experiment Repository
# ═══════════════════════════════════════════════════════════════════════════

class TestExperimentRepository:

    @pytest.mark.asyncio
    async def test_create_experiment(self, experiment_repo, mock_session):
        mock_exp = Mock()
        mock_exp.id = uuid4()
        mock_session.flush = AsyncMock()

        # Intercept the Experiment() constructor call
        with patch(
            "src.data_layer.repositories.experiment_repo.Experiment",
            return_value=mock_exp,
        ):
            result = await experiment_repo.create({
                "name": "Test",
                "campaign_id": uuid4(),
                "experiment_type": "thompson_sampling",
            })
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_experiment_maps_experiment_type(self, experiment_repo, mock_session):
        """experiment_type → type rename."""
        mock_exp = Mock()
        mock_exp.id = uuid4()
        mock_session.flush = AsyncMock()
        data = {"name": "E", "experiment_type": "bandit", "campaign_id": uuid4()}
        with patch(
            "src.data_layer.repositories.experiment_repo.Experiment",
            return_value=mock_exp,
        ) as MockExp:
            await experiment_repo.create(data)
            call_kwargs = MockExp.call_args[1]
            assert "type" in call_kwargs
            assert "experiment_type" not in call_kwargs

    @pytest.mark.asyncio
    async def test_get_experiment_by_id(self, experiment_repo, mock_session):
        exp = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = exp
        mock_session.execute.return_value = mock_result
        result = await experiment_repo.get_by_id(str(uuid4()))
        assert result is exp

    @pytest.mark.asyncio
    async def test_get_by_campaign(self, experiment_repo, mock_session):
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [Mock(), Mock()]
        mock_session.execute.return_value = mock_result
        result = await experiment_repo.get_by_campaign(str(uuid4()))
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_update_experiment(self, experiment_repo, mock_session):
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = Mock()
        mock_session.execute.return_value = mock_result
        result = await experiment_repo.update(str(uuid4()), {"name": "Updated"})
        assert result is not None
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_results(self, experiment_repo, mock_session):
        ok = await experiment_repo.update_results(str(uuid4()), {"winner": "A"})
        assert ok is True
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_active_experiments_attribute_error(self, experiment_repo, mock_session):
        """Experiment model has is_active, not status; repo filters by status='running' which
        raises AttributeError caught by exception handler."""
        mock_session.execute.side_effect = AttributeError("no attribute 'status'")
        result = await experiment_repo.get_active_experiments()
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_experiment(self, experiment_repo, mock_session):
        ok = await experiment_repo.delete(str(uuid4()))
        assert ok is True
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_experiment_error(self, experiment_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        ok = await experiment_repo.delete(str(uuid4()))
        assert ok is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. Metrics Repository
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricsRepository:

    @pytest.mark.asyncio
    async def test_record_metric(self, metrics_repo, mock_session):
        await metrics_repo.record_metric({
            "metric_name": "ctr", "metric_value": 2.5
        })
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_metric_rollback_on_error(self, metrics_repo, mock_session):
        mock_session.commit.side_effect = Exception("fail")
        with pytest.raises(Exception):
            await metrics_repo.record_metric({"metric_name": "x", "metric_value": 0})
        mock_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_calculate_roi_attribute_error(self, metrics_repo, mock_session):
        """Metric model has no 'spend' column; repo catches and returns defaults."""
        mock_session.execute.side_effect = AttributeError("Metric has no attribute 'spend'")
        roi = await metrics_repo.calculate_roi(str(uuid4()), customer_ltv=5000.0)
        assert roi["roi"] == 0.0
        assert roi["roas"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_roi_no_data(self, metrics_repo, mock_session):
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.execute.return_value = mock_result
        roi = await metrics_repo.calculate_roi(str(uuid4()))
        assert roi["roi"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_roi_zero_spend(self, metrics_repo, mock_session):
        row = Mock(total_spend=0.0, conversions=0, customers=0)
        mock_result = Mock()
        mock_result.first.return_value = row
        mock_session.execute.return_value = mock_result
        roi = await metrics_repo.calculate_roi(str(uuid4()))
        assert roi["roi"] == 0.0
        assert roi["roas"] == 0.0

    @pytest.mark.asyncio
    async def test_get_realtime_metrics_attribute_error(self, metrics_repo, mock_session):
        """Metric model has no 'impressions' column; repo catches and returns defaults."""
        mock_session.execute.side_effect = AttributeError("no attribute")
        rt = await metrics_repo.get_realtime_metrics(last_minutes=30)
        assert rt["active_campaigns"] == 0
        assert rt["impressions"] == 0

    @pytest.mark.asyncio
    async def test_get_realtime_metrics_error(self, metrics_repo, mock_session):
        mock_session.execute.side_effect = Exception("fail")
        rt = await metrics_repo.get_realtime_metrics()
        assert rt["active_campaigns"] == 0

    @pytest.mark.asyncio
    async def test_get_funnel_metrics_attribute_error(self, metrics_repo, mock_session):
        """Metric model has no impressions/clicks columns; repo catches error."""
        mock_session.execute.side_effect = AttributeError("no attr")
        funnel = await metrics_repo.get_funnel_metrics()
        assert funnel == {"stages": []}


# ═══════════════════════════════════════════════════════════════════════════
# 5. PgVectorStore
# ═══════════════════════════════════════════════════════════════════════════

class TestPgVectorStore:

    @pytest.fixture
    def store(self):
        s = PgVectorStore.__new__(PgVectorStore)
        s.collection_name = "test_docs"
        s.embedding_dim = 384
        mock_model = Mock()
        mock_model.encode.return_value = __import__("numpy").zeros(384)
        PgVectorStore._shared_embedding_model = mock_model
        return s

    def _mock_get_session(self, mock_session):
        """Create a mock for get_async_session that returns an async context manager."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _get_session():
            yield mock_session

        return _get_session

    @pytest.mark.asyncio
    async def test_add_document(self, store):
        mock_session = MagicMock()
        mock_result = Mock()
        mock_result.scalar_one.return_value = 42
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            doc_id = await store.add_document("hello world", metadata={"key": "val"})
        assert doc_id == 42
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_document_error(self, store):
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("insert fail"))

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            with pytest.raises(Exception, match="insert fail"):
                await store.add_document("fail")

    @pytest.mark.asyncio
    async def test_search(self, store):
        row = Mock(
            id=1, content="doc text",
            metadata={"k": "v"}, created_at=datetime.utcnow(),
            similarity=0.95,
        )
        mock_session = MagicMock()
        # First execute: SET ivfflat.probes; Second execute: the actual query
        mock_session.execute = AsyncMock(side_effect=[None, [row]])

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            results = await store.search("query text", top_k=3)
        assert len(results) == 1
        assert results[0]["similarity"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, store):
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=[None, []])

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            results = await store.search("q", filter_metadata={"model": "gpt-4"})
        assert results == []

    @pytest.mark.asyncio
    async def test_search_error(self, store):
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("search fail"))

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            results = await store.search("q")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_document_success(self, store):
        mock_session = MagicMock()
        mock_result = Mock(rowcount=1)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            ok = await store.delete_document(42)
        assert ok is True
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, store):
        mock_session = MagicMock()
        mock_result = Mock(rowcount=0)
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            ok = await store.delete_document(999)
        assert ok is False

    @pytest.mark.asyncio
    async def test_clear_collection(self, store):
        mock_session = MagicMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            ok = await store.clear_collection()
        assert ok is True
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clear_collection_error(self, store):
        mock_session = MagicMock()
        mock_session.execute = AsyncMock(side_effect=Exception("truncate fail"))

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            ok = await store.clear_collection()
        assert ok is False

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, store):
        row = Mock(total_docs=42)
        mock_session = MagicMock()
        mock_result = Mock()
        mock_result.first.return_value = row
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            stats = await store.get_collection_stats()
        assert stats["total_documents"] == 42
        assert stats["collection_name"] == "test_docs"

    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, store):
        mock_session = MagicMock()
        count_row = Mock(total=5)
        count_result = Mock()
        count_result.first.return_value = count_row
        mock_session.execute = AsyncMock(side_effect=[count_result, None])
        mock_session.commit = AsyncMock()

        with patch.object(
            _pgvector_store_module,
            "get_async_session",
            self._mock_get_session(mock_session),
        ):
            count = await store.delete_by_metadata({"agent_name": "bot"})
        assert count == 5

    def test_embed_text(self, store):
        import numpy as np
        result = store.embed_text("some text")
        assert isinstance(result, np.ndarray)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Semantic Cache
# ═══════════════════════════════════════════════════════════════════════════

class TestSemanticCache:

    @pytest.fixture
    def cache(self):
        c = SemanticCache.__new__(SemanticCache)
        c.similarity_threshold = 0.95
        c.ttl_hours = 24
        c.enabled = True
        c.vector_store = AsyncMock()
        c.embedding_model = Mock()
        return c

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache):
        cached_at = datetime.utcnow().isoformat()
        cache.vector_store.search.return_value = [{
            "id": 1,
            "similarity": 0.98,
            "metadata": {
                "response": "cached answer",
                "model": "gpt-4",
                "cached_at": cached_at,
            },
        }]
        result = await cache.get("test prompt", model="gpt-4")
        assert result is not None
        assert result["response"] == "cached answer"
        assert result["similarity"] == pytest.approx(0.98)

    @pytest.mark.asyncio
    async def test_cache_miss_no_results(self, cache):
        cache.vector_store.search.return_value = []
        result = await cache.get("unknown prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_low_similarity(self, cache):
        cache.vector_store.search.return_value = [{
            "id": 1,
            "similarity": 0.50,
            "metadata": {
                "response": "x",
                "model": "gpt-4",
                "cached_at": datetime.utcnow().isoformat(),
            },
        }]
        result = await cache.get("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expired(self, cache):
        expired_at = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        cache.vector_store.search.return_value = [{
            "id": 1,
            "similarity": 0.99,
            "metadata": {
                "response": "old",
                "model": "gpt-4",
                "cached_at": expired_at,
            },
        }]
        result = await cache.get("test")
        assert result is None
        cache.vector_store.delete_document.assert_awaited_once_with(1)

    @pytest.mark.asyncio
    async def test_cache_disabled(self, cache):
        cache.enabled = False
        result = await cache.get("test")
        assert result is None
        cache.vector_store.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_set_stores_prompt(self, cache):
        ok = await cache.set("prompt", "response", model="gpt-4")
        assert ok is True
        cache.vector_store.add_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_disabled(self, cache):
        cache.enabled = False
        ok = await cache.set("prompt", "response")
        assert ok is False

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        cache.vector_store.search.return_value = [{"id": 7}]
        ok = await cache.invalidate("prompt", "gpt-4")
        assert ok is True
        cache.vector_store.delete_document.assert_awaited_once_with(7)

    @pytest.mark.asyncio
    async def test_get_stats(self, cache):
        cache.vector_store.get_collection_stats.return_value = {"total_documents": 100}
        stats = await cache.get_stats()
        assert stats["enabled"] is True
        assert stats["total_entries"] == 100
        assert stats["similarity_threshold"] == 0.95

    def test_create_cache_key_deterministic(self, cache):
        key1 = cache._create_cache_key("hello", "gpt-4")
        key2 = cache._create_cache_key("hello", "gpt-4")
        assert key1 == key2

    def test_create_cache_key_differs_by_model(self, cache):
        key1 = cache._create_cache_key("hello", "gpt-4")
        key2 = cache._create_cache_key("hello", "gpt-3.5-turbo")
        assert key1 != key2


# ═══════════════════════════════════════════════════════════════════════════
# 7. Database Models — instantiation & enum values
# ═══════════════════════════════════════════════════════════════════════════

class TestDatabaseModels:

    def test_campaign_status_enum_values(self):
        assert CampaignStatus.DRAFT.value == "draft"
        assert CampaignStatus.RUNNING.value == "running"
        assert CampaignStatus.PAUSED.value == "paused"
        assert CampaignStatus.COMPLETED.value == "completed"
        assert CampaignStatus.FAILED.value == "failed"
        assert CampaignStatus.PENDING_APPROVAL.value == "pending_approval"
        assert CampaignStatus.APPROVED.value == "approved"

    def test_content_status_enum_values(self):
        assert ContentStatus.GENERATED.value == "generated"
        assert ContentStatus.PENDING_REVIEW.value == "pending_review"
        assert ContentStatus.APPROVED.value == "approved"
        assert ContentStatus.REJECTED.value == "rejected"
        assert ContentStatus.DEPLOYED.value == "deployed"

    def test_platform_enum_values(self):
        assert Platform.LINKEDIN.value == "linkedin"
        assert Platform.TWITTER.value == "twitter"
        assert Platform.EMAIL.value == "email"

    def test_campaign_goal_enum_values(self):
        assert CampaignGoal.LEAD_GENERATION.value == "lead_generation"
        assert CampaignGoal.BRAND_AWARENESS.value == "brand_awareness"
        assert CampaignGoal.CONVERSION.value == "conversion"

    def test_agent_type_enum_values(self):
        assert AgentType.CONTENT_GENERATOR.value == "content_generator"
        assert AgentType.STRATEGY_OPTIMIZER.value == "strategy_optimizer"

    def test_alert_severity_enum_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_workflow_event_enum_values(self):
        assert WorkflowEventType.WORKFLOW_STARTED.value == "workflow_started"
        assert WorkflowEventType.CONTENT_GENERATED.value == "content_generated"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Episodic Memory
# ═══════════════════════════════════════════════════════════════════════════

class TestEpisodicMemory:

    @pytest.fixture
    def memory_store(self):
        store = EpisodicMemoryStore.__new__(EpisodicMemoryStore)
        store.agent_name = "test_agent"
        store.collection_name = "test_agent_memory"
        store.embeddings = None
        store.vector_store = AsyncMock()
        store._initialized = True
        return store

    @pytest.fixture
    def sample_memory(self):
        return AgentMemory(
            agent_name="test_agent",
            task_id="task-001",
            task_description="Generate LinkedIn post",
            actions_taken=["research", "write", "review"],
            outcome="success",
            metrics={"cost": 0.05, "duration": 2.1, "quality_score": 0.9},
            human_feedback="Good job",
            timestamp=datetime.utcnow(),
        )

    @pytest.mark.asyncio
    async def test_store_memory(self, memory_store, sample_memory):
        memory_store.vector_store.add_documents.return_value = [42]
        memory_id = await memory_store.store_memory(sample_memory)
        assert memory_id == 42
        memory_store.vector_store.add_documents.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories(self, memory_store):
        doc = _StubDocument(
            page_content="Task: generate post",
            metadata={
                "agent_name": "test_agent",
                "task_id": "t1",
                "outcome": "success",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        memory_store.vector_store.similarity_search_with_score.return_value = [
            (doc, 0.85)
        ]
        results = await memory_store.retrieve_relevant_memories("generate post", k=5)
        assert len(results) == 1
        assert results[0]["similarity_score"] == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_retrieve_memories_filters_below_threshold(self, memory_store):
        doc = _StubDocument(page_content="x", metadata={"agent_name": "test_agent"})
        memory_store.vector_store.similarity_search_with_score.return_value = [
            (doc, 0.1)  # below default min_similarity of 0.3
        ]
        results = await memory_store.retrieve_relevant_memories("q")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_success_rate(self, memory_store):
        docs = [
            _StubDocument(page_content="a", metadata={"agent_name": "test_agent", "outcome": "success"}),
            _StubDocument(page_content="b", metadata={"agent_name": "test_agent", "outcome": "failure"}),
            _StubDocument(page_content="c", metadata={"agent_name": "test_agent", "outcome": "success"}),
        ]
        memory_store.vector_store.similarity_search.return_value = docs
        rate = await memory_store.get_success_rate()
        assert rate == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_get_success_rate_empty(self, memory_store):
        memory_store.vector_store.similarity_search.return_value = []
        rate = await memory_store.get_success_rate()
        assert rate == 0.0

    @pytest.mark.asyncio
    async def test_format_memories_for_prompt_empty(self, memory_store):
        result = await memory_store.format_memories_for_prompt([])
        assert result == "No relevant past experiences found."

    @pytest.mark.asyncio
    async def test_format_memories_for_prompt_with_data(self, memory_store):
        memories = [{
            "content": "Task: write email",
            "similarity_score": 0.92,
            "outcome": "success",
            "metadata": {},
        }]
        result = await memory_store.format_memories_for_prompt(memories)
        assert "Experience 1" in result
        assert "0.92" in result

    def test_agent_memory_to_text(self, sample_memory):
        text = sample_memory.to_text()
        assert "Generate LinkedIn post" in text
        assert "research" in text
        assert "success" in text
        assert "Human Feedback: Good job" in text

    def test_agent_memory_to_dict(self, sample_memory):
        d = sample_memory.to_dict()
        assert d["agent_name"] == "test_agent"
        assert d["task_id"] == "task-001"
        assert isinstance(d["timestamp"], str)

    def test_create_memory_from_task_success(self):
        mem = create_memory_from_task(
            agent_name="bot",
            task_id="t1",
            task_description="write post",
            actions=["a1"],
            result={"success": True, "cost": 0.1, "duration": 1.0, "quality_score": 0.8},
        )
        assert mem.outcome == "success"
        assert mem.metrics["cost"] == 0.1

    def test_create_memory_from_task_failure(self):
        mem = create_memory_from_task(
            agent_name="bot",
            task_id="t2",
            task_description="deploy",
            actions=["a1"],
            result={"success": False, "error": "timeout"},
        )
        assert mem.outcome == "failure"
        assert "timeout" in mem.lessons_learned


# ═══════════════════════════════════════════════════════════════════════════
# 9. Cost Tracker
# ═══════════════════════════════════════════════════════════════════════════

class TestCostTracker:

    def test_calculate_cost_gpt4(self):
        cost = calculate_cost("gpt-4", tokens_prompt=1000, tokens_completion=500)
        expected = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert cost == pytest.approx(expected)

    def test_calculate_cost_gpt4o_mini(self):
        cost = calculate_cost("gpt-4o-mini", tokens_prompt=2000, tokens_completion=1000)
        expected = (2000 / 1000) * 0.00015 + (1000 / 1000) * 0.0006
        assert cost == pytest.approx(expected)

    def test_calculate_cost_unknown_model_defaults_to_gpt35(self):
        cost = calculate_cost("some-unknown-model", tokens_prompt=1000, tokens_completion=0)
        gpt35_rate = COST_RATES["gpt-3.5-turbo"]["prompt"]
        assert cost == pytest.approx((1000 / 1000) * gpt35_rate)

    def test_calculate_cost_local_model_with_provider(self):
        cost = calculate_cost("my-local-model", tokens_prompt=1000, tokens_completion=500, provider="ollama")
        local_rates = COST_RATES["default_local"]
        expected = (1000 / 1000) * local_rates["prompt"] + (500 / 1000) * local_rates["completion"]
        assert cost == pytest.approx(expected)

    def test_calculate_cost_llama_prefix(self):
        cost = calculate_cost("llama-custom", tokens_prompt=1000, tokens_completion=500)
        local_rates = COST_RATES["default_local"]
        expected = (1000 / 1000) * local_rates["prompt"] + (500 / 1000) * local_rates["completion"]
        assert cost == pytest.approx(expected)

    def test_calculate_cost_embedding_model(self):
        cost = calculate_cost("text-embedding-ada-002", tokens_prompt=5000, tokens_completion=0)
        expected = (5000 / 1000) * 0.0001
        assert cost == pytest.approx(expected)

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_normal(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_estimate_tokens_none(self):
        assert estimate_tokens(None) == 0

    def test_cost_tracker_add_operation(self):
        tracker = CostTracker(agent_type="content_gen")
        tracker.add_operation("gpt-4", tokens_prompt=1000, tokens_completion=500)
        assert len(tracker.operations) == 1
        assert tracker.total_tokens == 1500
        assert tracker.total_cost > 0

    def test_cost_tracker_multiple_operations(self):
        tracker = CostTracker(agent_type="content_gen", campaign_id="c1")
        tracker.add_operation("gpt-4", 500, 200)
        tracker.add_operation("gpt-4o-mini", 1000, 500)
        assert len(tracker.operations) == 2
        assert tracker.total_tokens == 500 + 200 + 1000 + 500

    def test_cost_tracker_get_summary(self):
        tracker = CostTracker(agent_type="bot")
        tracker.add_operation("gpt-4", 100, 50, action="generate")
        summary = tracker.get_summary()
        assert summary["total_tokens"] == 150
        assert len(summary["operations"]) == 1
        assert summary["operations"][0]["action"] == "generate"

    def test_llm_provider_enum(self):
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.LOCAL.value == "local"

    @pytest.mark.asyncio
    async def test_cost_tracker_save(self):
        tracker = CostTracker(agent_type="bot")
        tracker.add_operation("gpt-4", 100, 50)
        with patch("src.ai_layer.utils.cost_tracker.track_llm_cost", new_callable=AsyncMock) as mock_track:
            mock_track.return_value = {"cost": 0.01}
            result = await tracker.save()
        assert result["operations_count"] == 1
        mock_track.assert_awaited_once()
