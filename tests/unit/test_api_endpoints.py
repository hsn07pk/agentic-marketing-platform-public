# tests/unit/test_api_endpoints.py
"""
Comprehensive pytest tests for FastAPI API endpoints.

All database dependencies are mocked — no Docker services required.
Testing router functions DIRECTLY because FastAPI and dependencies are not fully installed in test env.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime, timedelta
import sys

# ---------------------------------------------------------------------------
# Pre-mock dependencies to allow imports
# ---------------------------------------------------------------------------
for mod in [
    "fastapi", "fastapi.security", "fastapi.middleware", "fastapi.responses",
    "prometheus_client", "tenacity", "frontmatter", "sendgrid", "apify_client",
    "bs4", "selenium", "watchdog", "watchdog.observers", "rq", "schedule", "psutil",
    "langgraph", "langgraph.graph", "langchain", "langchain_openai", "langchain.schema",
    "langchain_core", "langchain_core.messages", "langchain_core.prompts",
    "langchain_community", "langchain_community.chat_models"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Configure mocked APIRouter to not swallow functions (return passthrough decorator)
# This is crucial because @router.get() on a MagicMock returns a MagicMock, 
# replacing the decorated function with a Mock. We want the original function.
if "fastapi" in sys.modules:
    fastapi_mock = sys.modules["fastapi"]
    
    def passthrough_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    # Configure all HTTP methods to return the passthrough decorator
    for method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
        getattr(fastapi_mock.APIRouter.return_value, method).side_effect = passthrough_decorator

# Fix HTTPException to be a real exception class that accepts kwargs
class MockHTTPException(Exception):
    def __init__(self, status_code: int, detail: str = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

if "fastapi.exceptions" in sys.modules:
    sys.modules["fastapi.exceptions"].HTTPException = MockHTTPException
else:
    m = MagicMock()
    m.HTTPException = MockHTTPException
    sys.modules["fastapi.exceptions"] = m

# Import router functions
from src.api.routers.campaigns import (
    list_campaigns, create_campaign, get_campaign, update_campaign, delete_campaign,
    start_campaign, pause_campaign, check_campaign_completion, get_campaign_metrics
)
from src.api.routers.metrics import get_metrics_overview
from src.api.routers.governance import get_hitl_queue, submit_review
from src.data_layer.database.models import (
    Campaign, CampaignStatus, Platform, CampaignGoal,
    Content, ContentStatus, HITLQueue, WorkflowEvent,
    CostTracking, SystemConfiguration, Experiment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _campaign_mock(**overrides):
    """Return a mock Campaign ORM object with sensible defaults."""
    c = Mock(spec=Campaign)
    c.id = overrides.get("id", uuid4())
    c.name = overrides.get("name", "Test Campaign")
    c.description = overrides.get("description", "A test campaign")
    c.platform = overrides.get("platform", Platform.LINKEDIN)
    c.status = overrides.get("status", CampaignStatus.DRAFT)
    c.goal = overrides.get("goal", CampaignGoal.LEAD_GENERATION)
    c.target_persona = overrides.get("target_persona", "decision_maker")
    c.budget_total = overrides.get("budget_total", 1000.0)
    c.budget_spent = overrides.get("budget_spent", 200.0)
    c.budget_daily_limit = overrides.get("budget_daily_limit", 50.0)
    c.impressions = overrides.get("impressions", 5000)
    c.clicks = overrides.get("clicks", 250)
    c.conversions = overrides.get("conversions", 10)
    c.ctr = overrides.get("ctr", 5.0)
    c.cpl = overrides.get("cpl", 20.0)
    c.config = overrides.get("config", {})
    c.start_date = overrides.get("start_date", datetime.utcnow())
    end_date_val = overrides.get("end_date")
    if end_date_val is None:
        c.end_date = datetime.utcnow() + timedelta(days=30)
    else:
        c.end_date = end_date_val
    
    c.created_at = overrides.get("created_at", datetime.utcnow())
    return c

def _hitl_mock(**overrides):
    """Return a mock HITLQueue ORM object."""
    h = Mock(spec=HITLQueue)
    h.id = overrides.get("id", uuid4())
    h.content_id = overrides.get("content_id", uuid4())
    h.status = overrides.get("status", "pending")
    h.priority = overrides.get("priority", 5)
    h.reason = overrides.get("reason", "safety_review")
    h.decision = overrides.get("decision", None)
    h.feedback = overrides.get("feedback", None)
    h.assigned_to = overrides.get("assigned_to", None)
    h.completed_at = overrides.get("completed_at", None)
    h.created_at = overrides.get("created_at", datetime.utcnow())
    return h

def _content_mock(**overrides):
    """Return a mock Content ORM object."""
    c = Mock(spec=Content)
    c.id = overrides.get("id", uuid4())
    c.campaign_id = overrides.get("campaign_id", uuid4())
    c.headline = overrides.get("headline", "Test Headline")
    c.body = overrides.get("body", "Test body content")
    c.status = overrides.get("status", ContentStatus.GENERATED)
    c.reviewed_by = overrides.get("reviewed_by", None)
    c.reviewed_at = overrides.get("reviewed_at", None)
    c.safety_score = overrides.get("safety_score", 0.95)
    return c

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    """Mock AsyncSession for database operations"""
    session = AsyncMock()
    # Explicitly set async methods to AsyncMock to be sure
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.get = AsyncMock()
    session.close = AsyncMock()
    
    # session.add is synchronous in SQLAlchemy, so use MagicMock
    session.add = MagicMock()
    
    # Mock result of execute()
    # When await db.execute() finishes, it returns mock_result
    # mock_result.scalars().all() should return a list (not a mock)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalars.return_value.first.return_value = None
    mock_result.scalar_one_or_none.return_value = None
    
    session.execute.return_value = mock_result
    return session

@pytest.fixture
def mock_bg_tasks():
    return MagicMock()

@pytest.fixture
def mock_orchestrator():
    orch = AsyncMock()
    orch.run_campaign_workflow.return_value = {"success": True}
    orch.resume_workflow_after_approval.return_value = {"success": True}
    return orch

# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestCampaignEndpoints:
    
    @pytest.mark.asyncio
    async def test_list_campaigns(self, mock_db):
        """Test list_campaigns endpoint."""
        campaign = _campaign_mock()
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [campaign]
        mock_db.execute.return_value = mock_result
        
        result = await list_campaigns(limit=10, offset=0, db=mock_db)
        
        assert len(result) == 1
        assert result[0]["id"] == str(campaign.id)
        assert result[0]["name"] == "Test Campaign"
        mock_db.execute.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_create_campaign(self, mock_db, mock_bg_tasks, mock_orchestrator):
        """Test create_campaign endpoint."""
        campaign_data = {
            "name": "New Campaign",
            "platform": "linkedin",
            "budget_total": 5000.0,
            "goal": "brand_awareness",
            "auto_start": True
        }
        
        # Mock db.refresh to simulate ID assignment
        async def fake_refresh(obj):
            obj.id = uuid4()
            obj.created_at = datetime.utcnow()
            
        mock_db.refresh = AsyncMock(side_effect=fake_refresh)
        
        result = await create_campaign(
            campaign_data=campaign_data, 
            background_tasks=mock_bg_tasks, 
            db=mock_db, 
            orchestrator=mock_orchestrator
        )
        
        assert result["name"] == "New Campaign"
        assert "id" in result
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()
        mock_bg_tasks.add_task.assert_called_once()  # Because auto_start=True
        
    @pytest.mark.asyncio
    async def test_get_campaign(self, mock_db):
        """Test get_campaign endpoint."""
        campaign = _campaign_mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = campaign
        mock_db.execute.return_value = mock_result
        
        result = await get_campaign(campaign_id=campaign.id, db=mock_db)
        
        assert result["id"] == str(campaign.id)
        assert result["name"] == "Test Campaign"

    @pytest.mark.asyncio
    async def test_start_campaign(self, mock_db, mock_bg_tasks, mock_orchestrator):
        """Test start_campaign endpoint."""
        campaign = _campaign_mock(status=CampaignStatus.DRAFT)
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = campaign
        
        # Mock finding existing experiment (none found)
        exp_result = Mock()
        exp_result.scalar_one_or_none.return_value = None
        
        mock_db.execute.side_effect = [mock_result, exp_result]
        
        # Mock Repository
        with patch("src.data_layer.repositories.experiment_repo.ExperimentRepository") as MockRepo:
            repo_instance = AsyncMock()
            MockRepo.return_value = repo_instance
            
            result = await start_campaign(
                campaign_id=campaign.id,
                background_tasks=mock_bg_tasks,
                db=mock_db,
                orchestrator=mock_orchestrator
            )
            
            assert result["status"] == "started"
            assert campaign.status == CampaignStatus.RUNNING
            mock_db.commit.assert_called()
            mock_bg_tasks.add_task.assert_called()
            repo_instance.create.assert_called()

    @pytest.mark.asyncio
    async def test_check_campaign_completion(self, mock_db):
        """Test check_campaign_completion endpoint."""
        campaign = _campaign_mock(
            status=CampaignStatus.RUNNING,
            budget_total=1000.0,
            budget_spent=990.0  # > 98% spent
        )
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = campaign
        mock_db.execute.return_value = mock_result
        
        # Mock internal function logic by patching the imported function
        with patch("src.ai_layer.learning.campaign_completion.should_complete_campaign") as mock_should:
            mock_decision = Mock()
            mock_decision.should_complete = True
            mock_decision.reason = "Budget depleted"
            mock_decision.completion_type = "budget"
            mock_decision.final_metrics = {}
            mock_should.return_value = mock_decision
            
            result = await check_campaign_completion(campaign_id=campaign.id, db=mock_db)
            
            assert result["should_complete"] is True
            assert campaign.status == CampaignStatus.COMPLETED
            mock_db.commit.assert_called()

class TestMetricsEndpoints:
    
    @pytest.mark.asyncio
    async def test_metrics_overview(self, mock_db):
        """Test metrics overview."""
        # Mock aggregate metrics row (must support attribute access)
        metrics_row = Mock()
        metrics_row.total_campaigns = 10
        metrics_row.total_impressions = 10000
        metrics_row.total_clicks = 500
        metrics_row.total_conversions = 50
        metrics_row.total_spent = 500.0
        
        agg_result = Mock()
        agg_result.first.return_value = metrics_row
        
        # When include_mock=True, only one DB call is made (metrics query)
        mock_db.execute.side_effect = [agg_result]
        
        result = await get_metrics_overview(days=30, include_mock=True, db=mock_db)
        
        assert result["total_impressions"] == 10000
        assert result["total_clicks"] == 500
        assert result["average_ctr"] == 5.0
        assert result["total_conversions"] == 50
        assert result["total_spent"] == 500.0

class TestGovernanceEndpoints:
    
    @pytest.mark.asyncio
    async def test_get_hitl_queue(self, mock_db):
        """Test get_hitl_queue endpoint."""
        hitl_item = _hitl_mock()
        content = _content_mock(id=hitl_item.content_id)
        campaign = _campaign_mock(id=content.campaign_id)
        
        # Order: Queue query -> Content query -> Campaign query
        q_result = Mock()
        q_result.scalars.return_value.all.return_value = [hitl_item]
        
        c_result = Mock()
        c_result.scalar_one_or_none.return_value = content
        
        cmp_result = Mock()
        cmp_result.scalar_one_or_none.return_value = campaign
        
        mock_db.execute.side_effect = [q_result, c_result, cmp_result]
        
        items = await get_hitl_queue(status="pending", limit=10, db=mock_db)
        
        assert len(items) == 1
        assert items[0]["id"] == str(hitl_item.id)
        assert items[0]["headline"] == "Test Headline"
        
    @pytest.mark.asyncio
    async def test_submit_review_approve(self, mock_db):
        """Test submit_review endpoint (approve)."""
        content = _content_mock()
        
        # Content query
        c_result = Mock()
        c_result.scalar_one_or_none.return_value = content
        mock_db.execute.return_value = c_result
        
        review_data = {
            "content_id": str(content.id),
            "decision": "approve",
            "feedback": "Looks good"
        }
        
        # Patch internal modules
        with patch("src.ai_layer.learning.governance_metrics_tracker.GovernanceMetricsTracker"), \
             patch("src.ai_layer.memory.episodic_memory.EpisodicMemoryStore"), \
             patch("src.ai_layer.orchestration.langgraph_supervisor.MarketingOrchestrator") as MockOrch:
            
            mock_orch_instance = AsyncMock()
            mock_orch_instance.resume_workflow_after_approval.return_value = {
                "success": True, "deployment_status": "completed"
            }
            MockOrch.return_value = mock_orch_instance
            
            result = await submit_review(review_data=review_data, db=mock_db)
            
            assert result["status"] == "success"
            assert result["decision"] == "approve"
            # Verify status update calls
            # We expect at least updates to HITLQueue, Content, and WorkflowEvent
            assert mock_db.execute.call_count >= 2
            mock_db.commit.assert_called()

