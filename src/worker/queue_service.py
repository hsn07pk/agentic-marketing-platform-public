"""
RQ Queue Service for Production

Provides simple interface to enqueue background tasks to the RQ worker.
"""
import logging
from typing import Any, Optional, Dict
from redis import Redis
from rq import Queue
from rq.job import Job

from ..config.settings import settings

logger = logging.getLogger(__name__)


class QueueService:
    """
    Service for enqueuing background tasks to RQ worker.
    
    Usage:
        queue = QueueService()
        job = queue.enqueue_delayed_rewards()
        job = queue.enqueue_simulation(campaign_id, ...)
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None
        self._queues = {}
    
    @property
    def redis(self) -> Redis:
        if self._redis is None:
            self._redis = Redis.from_url(self.redis_url)
        return self._redis
    
    def get_queue(self, name: str = 'default') -> Queue:
        """Get or create a queue by name"""
        if name not in self._queues:
            self._queues[name] = Queue(name, connection=self.redis)
        return self._queues[name]
    
    def enqueue_delayed_rewards(self) -> Job:
        """
        Enqueue delayed rewards processing task.
        Should run hourly.
        """
        from .tasks import process_delayed_rewards
        
        queue = self.get_queue('high')
        job = queue.enqueue(
            process_delayed_rewards,
            job_timeout='10m',
            result_ttl=3600
        )
        logger.info(f"Enqueued delayed rewards job: {job.id}")
        return job
    
    def enqueue_simulation(
        self,
        campaign_id: str,
        platform: str,
        persona: str,
        content: Dict[str, Any],
        budget: float,
        duration_days: int = 7
    ) -> Job:
        """
        Enqueue campaign simulation task.
        """
        from .tasks import run_campaign_simulation
        
        queue = self.get_queue('default')
        job = queue.enqueue(
            run_campaign_simulation,
            campaign_id,
            platform,
            persona,
            content,
            budget,
            duration_days,
            job_timeout='15m',
            result_ttl=86400  # 24 hours
        )
        logger.info(f"Enqueued simulation job for campaign {campaign_id}: {job.id}")
        return job
    
    def enqueue_content_generation(
        self,
        campaign_id: str,
        platform: str,
        persona: str,
        campaign_config: Dict[str, Any]
    ) -> Job:
        """
        Enqueue content generation task.
        """
        from .tasks import generate_content_async
        
        queue = self.get_queue('default')
        job = queue.enqueue(
            generate_content_async,
            campaign_id,
            platform,
            persona,
            campaign_config,
            job_timeout='5m',
            result_ttl=86400
        )
        logger.info(f"Enqueued content generation for campaign {campaign_id}: {job.id}")
        return job
    
    def enqueue_market_scrape(
        self,
        keywords: list,
        platform: str = 'linkedin',
        limit: int = 10
    ) -> Job:
        """
        Enqueue market scraping task.
        """
        from .tasks import scrape_market_data
        
        queue = self.get_queue('low')
        job = queue.enqueue(
            scrape_market_data,
            keywords,
            platform,
            limit,
            job_timeout='10m',
            result_ttl=3600
        )
        logger.info(f"Enqueued market scrape job: {job.id}")
        return job
    
    def enqueue_email_campaign(
        self,
        campaign_id: str,
        recipients: list,
        subject: str,
        html_content: str,
        from_email: str = None
    ) -> Job:
        """
        Enqueue email campaign task.
        """
        from .tasks import send_campaign_emails
        
        queue = self.get_queue('high')
        job = queue.enqueue(
            send_campaign_emails,
            campaign_id,
            recipients,
            subject,
            html_content,
            from_email,
            job_timeout='30m',
            result_ttl=86400
        )
        logger.info(f"Enqueued email campaign {campaign_id}: {job.id}")
        return job
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a queued job"""
        try:
            job = Job.fetch(job_id, connection=self.redis)
            return {
                'id': job.id,
                'status': job.get_status(),
                'result': job.result if job.is_finished else None,
                'error': str(job.exc_info) if job.is_failed else None,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'ended_at': job.ended_at.isoformat() if job.ended_at else None
            }
        except Exception as e:
            return {'id': job_id, 'status': 'not_found', 'error': str(e)}
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues"""
        stats = {}
        for name in ['high', 'default', 'low']:
            queue = self.get_queue(name)
            stats[name] = {
                'length': len(queue),
                'failed': queue.failed_job_registry.count,
                'scheduled': queue.scheduled_job_registry.count
            }
        return stats


# Singleton instance for easy access
_queue_service: Optional[QueueService] = None


def get_queue_service() -> QueueService:
    """Get or create the queue service singleton"""
    global _queue_service
    if _queue_service is None:
        _queue_service = QueueService()
    return _queue_service
