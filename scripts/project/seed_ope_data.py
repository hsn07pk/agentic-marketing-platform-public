#!/usr/bin/env python3
"""
Seed historical experiment data for OPE testing
Creates bandit arm data that the OPE system can use
"""
import asyncio
import random
from datetime import datetime, timedelta
from uuid import UUID, uuid4

async def seed_ope_data():
    from src.data_layer.database.connection import get_async_session
    from src.data_layer.database.models import Experiment, BanditArm, Campaign, Platform, CampaignStatus
    from sqlalchemy import delete
    
    campaign_ids = []
    
    arm_names = ["variant_a", "variant_b", "variant_c", "variant_d"]
    platforms = ["linkedin", "twitter", "email"]
    personas = ["decision_maker", "influencer", "technical_buyer"]
    
    async with get_async_session() as session:
        # Create a Campaign first to satisfy FK
        new_campaign_id = uuid4()
        campaign = Campaign(
            id=new_campaign_id,
            name=f"OPE Data Seed Campaign {datetime.now().strftime('%Y%m%d%H%M')}",
            status=CampaignStatus.COMPLETED,
            budget_total=50000.0,
            platform=Platform.LINKEDIN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        session.add(campaign)
        await session.flush()
        campaign_ids.append(str(new_campaign_id))
        print(f"✅ Created Campaign: {new_campaign_id}")

        # Continue with logic...
        # Create experiments with bandit arms
        count = 0
        for campaign_id in campaign_ids:
            for i in range(50):  # 50 experiments per campaign = 150 total
                exp_id = uuid4()
                started_at = datetime.now() - timedelta(days=random.randint(1, 30), hours=random.randint(0, 23))
                
                experiment = Experiment(
                    id=exp_id,
                    campaign_id=UUID(campaign_id),
                    name=f"ope_test_exp_{count}",
                    type="content_bandit",
                    algorithm="thompson_sampling",
                    variants=arm_names,
                    parameters={
                        "platform": random.choice(platforms),
                        "persona": random.choice(personas),
                        "budget": random.uniform(1000, 10000),
                        "num_arms": len(arm_names)
                    },
                    started_at=started_at,
                    ended_at=started_at + timedelta(hours=random.randint(1, 48)),
                    is_active=False  # Must be False for OPE
                )
                session.add(experiment)
                
                # Create bandit arms for this experiment
                for arm_name in arm_names:
                    pulls = random.randint(10, 100)
                    success_count = int(pulls * random.uniform(0.01, 0.15))  # 1-15% CTR
                    
                    arm = BanditArm(
                        id=uuid4(),
                        experiment_id=exp_id,
                        arm_id=arm_name,
                        pulls=pulls,
                        successes=success_count,
                        failures=pulls - success_count,
                        total_reward=float(success_count),
                        alpha=1.0 + success_count,
                        beta=1.0 + (pulls - success_count)
                    )
                    session.add(arm)
                
                count += 1
        
        await session.commit()
        print(f"✅ Created {count} experiments with bandit arms")
        return count

if __name__ == "__main__":
    asyncio.run(seed_ope_data())
