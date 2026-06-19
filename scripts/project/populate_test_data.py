#!/usr/bin/env python3
"""
Populate test data for dashboard testing
Creates campaigns with multiple personas and platforms
"""
import asyncio
import httpx
import json
from typing import List, Dict

API_BASE = "http://localhost:8000/api/v1"

# Define test campaigns with different personas and platforms
TEST_CAMPAIGNS = [
    {
        "name": "LinkedIn - Decision Makers",
        "platform": "linkedin",
        "target_persona": "decision_maker",
        "goal": "lead_generation",
        "budget": 2000,
        "description": "Target C-level executives"
    },
    {
        "name": "LinkedIn - Engineers",
        "platform": "linkedin",
        "target_persona": "engineer",
        "goal": "lead_generation",
        "budget": 1500,
        "description": "Target technical decision makers"
    },
    {
        "name": "LinkedIn - Managers",
        "platform": "linkedin",
        "target_persona": "manager",
        "goal": "brand_awareness",
        "budget": 1800,
        "description": "Target middle management"
    },
    {
        "name": "Twitter - Marketers",
        "platform": "twitter",
        "target_persona": "marketer",
        "goal": "engagement",
        "budget": 1200,
        "description": "Engage marketing professionals"
    },
    {
        "name": "Twitter - HR",
        "platform": "twitter",
        "target_persona": "HR",
        "goal": "brand_awareness",
        "budget": 1000,
        "description": "Build brand with HR community"
    }
]

async def create_campaign(client: httpx.AsyncClient, campaign_data: Dict) -> Dict:
    """Create a campaign"""
    print(f"\n📝 Creating campaign: {campaign_data['name']}")
    response = await client.post(
        f"{API_BASE}/campaigns/",
        json=campaign_data,
        timeout=120.0
    )
    response.raise_for_status()
    campaign = response.json()
    print(f"✅ Campaign created: {campaign['id']}")
    return campaign

async def wait_for_approval(client: httpx.AsyncClient, campaign_id: str, max_wait: int = 60) -> bool:
    """Wait for content to appear in HITL queue"""
    print(f"⏳ Waiting for content to be generated...")

    for i in range(max_wait):
        await asyncio.sleep(2)

        # Check HITL queue
        response = await client.get(f"{API_BASE}/governance/hitl-queue?status=pending")
        queue_items = response.json()

        # Find items for this campaign
        campaign_items = [item for item in queue_items if campaign_id in str(item.get('campaign_name', ''))]

        if campaign_items:
            print(f"✅ Content ready for review: {len(campaign_items)} items")
            return True

        if i % 5 == 0:
            print(f"   Still waiting... ({i*2}s)")

    print(f"⚠️  Timeout waiting for content")
    return False

async def approve_content(client: httpx.AsyncClient, content_id: str, reviewer: str = "test@example.com") -> bool:
    """Approve content"""
    print(f"✅ Approving content: {content_id[:8]}...")

    response = await client.post(
        f"{API_BASE}/governance/review",
        json={
            "content_id": content_id,
            "decision": "approve",
            "feedback": "Auto-approved for testing",
            "reviewer_email": reviewer
        },
        timeout=120.0
    )
    response.raise_for_status()
    result = response.json()

    if result.get('workflow_status') == 'deployed':
        print(f"✅ Content deployed successfully!")
        return True
    else:
        print(f"⚠️  Deployment status: {result.get('workflow_status')}")
        return False

async def reject_content(client: httpx.AsyncClient, content_id: str, reviewer: str = "test@example.com") -> bool:
    """Reject content"""
    print(f"❌ Rejecting content: {content_id[:8]}...")

    response = await client.post(
        f"{API_BASE}/governance/review",
        json={
            "content_id": content_id,
            "decision": "reject",
            "feedback": "Test rejection - content needs revision",
            "reviewer_email": reviewer
        },
        timeout=120.0
    )
    response.raise_for_status()
    result = response.json()
    print(f"✅ Content rejected: {result.get('decision')}")
    return True

async def get_pending_content(client: httpx.AsyncClient) -> List[Dict]:
    """Get all pending content"""
    response = await client.get(f"{API_BASE}/governance/hitl-queue?status=pending")
    return response.json()

async def main():
    """Main execution"""
    print("="*70)
    print("🚀 POPULATING TEST DATA FOR DASHBOARD")
    print("="*70)

    async with httpx.AsyncClient() as client:
        # Create campaigns
        created_campaigns = []
        for i, campaign_data in enumerate(TEST_CAMPAIGNS):
            try:
                campaign = await create_campaign(client, campaign_data)
                created_campaigns.append(campaign)

                # Wait a bit between campaigns
                await asyncio.sleep(3)

            except Exception as e:
                print(f"❌ Error creating campaign: {e}")
                continue

        print(f"\n{'='*70}")
        print(f"✅ Created {len(created_campaigns)} campaigns")
        print(f"{'='*70}\n")

        # Wait for all content to be generated
        print("⏳ Waiting for all content to be generated...")
        await asyncio.sleep(10)

        # Get all pending content
        pending_items = await get_pending_content(client)
        print(f"\n📋 Found {len(pending_items)} pending reviews")

        if not pending_items:
            print("⚠️  No pending content found. Campaigns may have failed.")
            return

        # Approve most, reject one for testing
        for i, item in enumerate(pending_items):
            content_id = item.get('content_id')

            try:
                # Reject the first one for testing
                if i == 0:
                    await reject_content(client, content_id)
                else:
                    await approve_content(client, content_id)

                # Wait between approvals
                await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ Error processing content {content_id}: {e}")
                continue

        print(f"\n{'='*70}")
        print("✅ TEST DATA POPULATION COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📊 Summary:")
        print(f"   - Campaigns Created: {len(created_campaigns)}")
        print(f"   - Content Reviewed: {len(pending_items)}")
        print(f"   - Approved: {len(pending_items) - 1}")
        print(f"   - Rejected: 1")
        print(f"\n🌐 View results:")
        print(f"   - Dashboard: http://localhost:8501")
        print(f"   - API Docs: http://localhost:8000/docs")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    asyncio.run(main())
