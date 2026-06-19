import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.automation_layer.deployer import CampaignDeployer

async def test():
    print("Initializing Deployer...")
    deployer = CampaignDeployer()
    
    # deploy() calls initialize() internally now
    # This should trigger DB config fetch
    print("Calling deploy()...")
    result = await deployer.deploy(
        content_id="test_content_123",
        platform="linkedin",
        content={"headline": "Test Headline", "body": "Test Body"},
        campaign_config={"campaign_id": "test_camp_123"}
    )
    
    print(f"Deployment Result: {result}")
    
    # Check if MockDeployer was used
    # The default DB config has MOCK_MODE_ENABLED=True
    if result.get('is_mock'):
        print("✅ SUCCESS: Mock Mode Active via DB Config")
        # Verify it didn't crash trying to use real API key
    elif result.get('success') is False and "not configured" in result.get('error', ''):
        print("❌ FAILURE: It failed to fallback to Mock properly (or Mock disabled)")
    else:
        print(f"❓ UNEXPECTED: {result}")

if __name__ == "__main__":
    asyncio.run(test())
