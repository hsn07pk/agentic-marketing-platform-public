import logging
from datetime import datetime, timedelta
from redis import Redis
from rq_scheduler import Scheduler

from ..config.settings import settings

logger = logging.getLogger(__name__)


def setup_scheduler() -> Scheduler:
    redis = Redis.from_url(settings.REDIS_URL)
    scheduler = Scheduler(connection=redis)
    
    for job in scheduler.get_jobs():
        scheduler.cancel(job)
    
    from .tasks import (
        process_delayed_rewards,
        poll_platform_metrics,
        sync_hubspot_deals,
        cleanup_agent_memories,
        run_scheduled_maintenance,
        run_autonomous_mlops_check,
        save_daily_governance_metrics,
        generate_weekly_learning_report,
        scrape_market_data
    )

    scheduler.schedule(
        scheduled_time=datetime.utcnow(),
        func=process_delayed_rewards,
        interval=3600,
        repeat=None,
        queue_name='high',
        meta={'description': 'Hourly delayed reward processing'}
    )

    # Offset by 5min to avoid overlap with reward processing
    scheduler.schedule(
        scheduled_time=datetime.utcnow() + timedelta(minutes=5),
        func=poll_platform_metrics,
        interval=3600,
        repeat=None,
        queue_name='default',
        meta={'description': 'Hourly platform metrics polling (LinkedIn/Twitter)'}
    )

    scheduler.schedule(
        scheduled_time=datetime.utcnow() + timedelta(minutes=10),
        func=sync_hubspot_deals,
        interval=7200,
        repeat=None,
        queue_name='default',
        meta={'description': 'HubSpot deal stage sync for reward attribution'}
    )

    # Research Plan Section 6.4: 90-day retention policy
    scheduler.schedule(
        scheduled_time=datetime.utcnow() + timedelta(hours=1),
        func=cleanup_agent_memories,
        kwargs={'retention_days': 90},
        interval=86400,
        repeat=None,
        queue_name='low',
        meta={'description': 'Daily agent memory cleanup (90-day retention)'}
    )

    scheduler.schedule(
        scheduled_time=datetime.utcnow().replace(hour=3, minute=0, second=0) + timedelta(days=1),
        func=run_scheduled_maintenance,
        interval=86400,
        repeat=None,
        queue_name='low',
        meta={'description': 'Daily comprehensive maintenance (memory cleanup, metrics, rewards)'}
    )

    scheduler.schedule(
        scheduled_time=datetime.utcnow() + timedelta(minutes=2),
        func=run_autonomous_mlops_check,
        interval=300,
        repeat=None,
        queue_name='high',
        meta={'description': 'Autonomous MLOps: experiment completion, model logging, auto-promotion'}
    )

    # Research Plan Section 10.2: Track Human Override Rate (<5% target)
    # 5min past midnight to avoid contention with other midnight jobs
    scheduler.schedule(
        scheduled_time=datetime.utcnow().replace(hour=0, minute=5, second=0) + timedelta(days=1),
        func=save_daily_governance_metrics,
        interval=86400,
        repeat=None,
        queue_name='default',
        meta={'description': 'Daily governance metrics (override rate, safety scores)'}
    )

    # Research Plan Section 10.2: Weekly Uplift Summary
    days_until_monday = (7 - datetime.utcnow().weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7

    next_monday = datetime.utcnow().replace(hour=6, minute=0, second=0) + timedelta(days=days_until_monday)

    scheduler.schedule(
        scheduled_time=next_monday,
        func=generate_weekly_learning_report,
        interval=604800,
        repeat=None,
        queue_name='default',
        meta={'description': 'Weekly learning report (best hooks, insights, recommendations)'}
    )

    # Competitive intelligence scraping every 6 hours
    scheduler.schedule(
        scheduled_time=datetime.utcnow() + timedelta(minutes=15),
        func=scrape_market_data,
        kwargs={'keywords': ['AI marketing', 'marketing automation'], 'platform': 'linkedin', 'limit': 10},
        interval=21600,
        repeat=None,
        queue_name='low',
        meta={'description': 'Periodic market intelligence scraping (Apify/web)'}
    )

    logger.info("Scheduler configured with recurring tasks")
    return scheduler


def get_scheduler_info() -> dict:
    try:
        redis = Redis.from_url(settings.REDIS_URL)
        scheduler = Scheduler(connection=redis)
        
        jobs = []
        for job in scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'func_name': job.func_name,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'meta': job.meta
            })
        
        return {
            'scheduled_jobs': len(jobs),
            'jobs': jobs
        }
    except Exception as e:
        return {'error': str(e), 'scheduled_jobs': 0}
