# tests/integration/test_workflow_integration.py
import pytest
import asyncio
from datetime import datetime

from src.ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator
from src.ai_layer.agents.content_generator import ContentGeneratorAgent as ContentGenerator
from src.ai_layer.agents.safety_validator import SafetyValidatorAgent as SafetyValidator
from src.data_layer.database.connection import get_async_session

@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_campaign_workflow():
    orchestrator = MarketingOrchestrator()
    
    campaign_config = {
        "name": "Integration Test Campaign",
        "platform": "linkedin",
        "persona": "decision_maker",
        "goal": "lead_generation",
        "budget": 1000.0
    }
    
    result = await orchestrator.run_campaign_workflow(
        campaign_id="test_campaign_123",
        config=campaign_config
    )
    
    assert result['success'] == True
    assert result['content_id'] is not None
    assert result['deployment_status'] is not None

@pytest.mark.integration
@pytest.mark.asyncio
async def test_content_generation_pipeline():
    generator = ContentGenerator()
    validator = SafetyValidator()
    
    content = await generator.generate(
        persona="decision_maker",
        goal="lead_generation",
        platform="linkedin"
    )
    
    assert content is not None
    assert "headline" in content
    assert "body" in content
    
    safety_result = await validator.validate(content)
    
    assert safety_result["safety_score"] > 0.0
    assert safety_result["safety_score"] <= 1.0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_operations():
    async with get_async_session() as session:
        from src.data_layer.repositories.campaign_repo import CampaignRepository
        
        repo = CampaignRepository(session)
        
        campaign_data = {
            "name": "Test Campaign",
            "platform": "linkedin",
            "status": "draft",
            "budget": 1000.0,
            "target_persona": "decision_maker"
        }
        
        campaign = await repo.create(campaign_data)
        
        assert campaign.id is not None
        
        retrieved = await repo.get_by_id(str(campaign.id))
        assert retrieved is not None
        assert retrieved.name == "Test Campaign"

@pytest.mark.integration
@pytest.mark.asyncio
async def test_bandit_learning_flow():
    from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit
    
    bandit = ThompsonSamplingBandit(n_arms=3)
    
    for _ in range(100):
        arm = bandit.select_arm()
        reward = 1 if arm == 0 else 0
        bandit.update(arm, reward)
    
    stats = bandit.get_stats()
    
    assert stats['total_pulls'] == 100
    assert stats['arm_pulls'][0] > stats['arm_pulls'][1]
    assert stats['arm_pulls'][0] > stats['arm_pulls'][2]