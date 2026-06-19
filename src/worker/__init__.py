"""
RQ Worker Tasks Module

This module contains background tasks processed by the RQ worker.
Tasks are enqueued from the API and processed asynchronously.

Usage:
    from src.worker import get_queue_service
    
    queue = get_queue_service()
    job = queue.enqueue_simulation(campaign_id, ...)
    status = queue.get_job_status(job.id)
"""
from .tasks import (
    process_delayed_rewards,
    run_campaign_simulation,
    generate_content_async,
    scrape_market_data,
    send_campaign_emails
)
from .queue_service import QueueService, get_queue_service

__all__ = [
    # Tasks
    'process_delayed_rewards',
    'run_campaign_simulation', 
    'generate_content_async',
    'scrape_market_data',
    'send_campaign_emails',
    # Queue Service
    'QueueService',
    'get_queue_service'
]
