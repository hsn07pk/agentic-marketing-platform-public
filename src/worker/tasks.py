"""RQ background tasks processed by the worker container."""
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async functions in sync RQ context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def process_delayed_rewards() -> Dict[str, int]:
    """Check Cal.com for bookings and process pending delayed rewards."""
    from ..ai_layer.learning.reward_tracker import RewardTracker
    
    logger.info("Processing delayed rewards...")
    tracker = RewardTracker()
    result = _run_async(tracker.process_pending_rewards())
    logger.info(f"Delayed rewards processed: {result}")
    return result


def run_campaign_simulation(
    campaign_id: str,
    platform: str,
    persona: str,
    content: Dict[str, Any],
    budget: float,
    duration_days: int = 7
) -> Dict[str, Any]:
    """Run pre-deployment simulation for a campaign."""
    from ..simulation.environment import MarketingEnvironment, SimulationConfig
    
    logger.info(f"Running simulation for campaign {campaign_id}")
    
    config = SimulationConfig(
        duration_days=duration_days,
        platforms=[platform],
        seed=42  # Reproducible simulation results
    )
    
    env = MarketingEnvironment(config)
    
    campaign_config = {
        'platform': platform,
        'content': content,
        'targeting': {'persona': persona},
        'budget': budget,
        'duration': duration_days
    }
    
    results = env.run_campaign(campaign_config)
    logger.info(f"Simulation complete for {campaign_id}: CTR={results.get('ctr', 0):.2%}")
    
    return results


def generate_content_async(
    campaign_id: str,
    platform: str,
    persona: str,
    campaign_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate campaign content asynchronously via background worker."""
    from ..ai_layer.agents.content_generator import ContentGeneratorAgent
    
    logger.info(f"Generating content for campaign {campaign_id} on {platform}")
    
    generator = ContentGeneratorAgent()
    
    async def _generate():
        content, metadata = await generator.generate_content(
            platform=platform,
            persona=persona,
            campaign_config=campaign_config
        )
        return {
            'headline': content.headline if hasattr(content, 'headline') else content.get('headline'),
            'body': content.body if hasattr(content, 'body') else content.get('body'),
            'cta': content.cta if hasattr(content, 'cta') else content.get('cta'),
            'metadata': metadata
        }
    
    result = _run_async(_generate())
    logger.info(f"Content generated for campaign {campaign_id}")
    return result


def scrape_market_data(
    keywords: list,
    platform: str = 'linkedin',
    limit: int = 10
) -> Dict[str, Any]:
    """Scrape market data for competitive intelligence."""
    from ..ai_layer.agents.market_scraper import MarketScraperAgent
    
    logger.info(f"Scraping market data for keywords: {keywords}")
    
    scraper = MarketScraperAgent()
    
    async def _scrape():
        result = await scraper.get_inspiration_for_campaign(
            keywords=keywords,
            limit=limit
        )
        return result
    
    result = _run_async(_scrape())
    logger.info(f"Market scraping complete: {result.get('success', False)}")
    return result


def send_campaign_emails(
    campaign_id: str,
    recipients: list,
    subject: str,
    html_content: str,
    from_email: str = None
) -> Dict[str, Any]:
    """Send batch campaign emails using the configured email provider (Mailgun > Mailchimp > SendGrid)."""
    from ..config.configuration_service import ConfigurationService
    from ..data_layer.database.connection import get_sync_session
    
    logger.info(f"Sending {len(recipients)} emails for campaign {campaign_id}")
    
    # Respect the same email provider priority as the deployer
    connector = None
    sync_db = get_sync_session()
    try:
        config_service = ConfigurationService(sync_db)
        mailgun_key = config_service.get_value("MAILGUN_API_KEY")
        mailgun_domain = config_service.get_value("MAILGUN_DOMAIN")
        if mailgun_key and mailgun_domain:
            from ..automation_layer.connectors.mailgun_api import MailgunConnector
            connector = MailgunConnector()
            logger.info("Using Mailgun for campaign emails")
        else:
            mailchimp_key = config_service.get_value("MAILCHIMP_API_KEY")
            if mailchimp_key:
                from ..automation_layer.connectors.mailchimp_api import MailchimpConnector
                connector = MailchimpConnector()
                logger.info("Using Mailchimp for campaign emails")
            else:
                from ..automation_layer.connectors.email_api import EmailConnector
                connector = EmailConnector()
                logger.info("Using SendGrid for campaign emails")
    finally:
        sync_db.close()
    
    if connector is None:
        from ..automation_layer.connectors.email_api import EmailConnector
        connector = EmailConnector()
    
    async def _send():
        results = {'sent': 0, 'failed': 0, 'errors': []}
        
        for recipient in recipients:
            try:
                response = await connector.send_email(
                    to_email=recipient,
                    subject=subject,
                    html_content=html_content,
                    from_email=from_email
                )
                if response.success:
                    results['sent'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(response.error)
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(str(e))
        
        return results
    
    result = _run_async(_send())
    logger.info(f"Email campaign {campaign_id} complete: {result['sent']} sent, {result['failed']} failed")
    return result


def poll_platform_metrics(campaign_id: str = None) -> Dict[str, Any]:
    """
    Poll platform APIs for actual campaign metrics after deployment.

    HIGH PRIORITY FIX: This connects deployed campaigns to their actual
    performance metrics from LinkedIn/Twitter APIs.
    """
    from ..data_layer.database.connection import async_session_maker
    from ..data_layer.database.models import Campaign, CampaignStatus
    from ..automation_layer.connectors.linkedin_api import LinkedInConnector
    from ..automation_layer.connectors.x_api import XConnector
    from sqlalchemy import select
    from datetime import datetime

    logger.info(f"Polling platform metrics for campaign: {campaign_id or 'all active'}")

    async def _poll():
        results = {'campaigns_checked': 0, 'metrics_updated': 0, 'errors': []}

        async with async_session_maker() as session:
            if campaign_id:
                stmt = select(Campaign).where(Campaign.id == campaign_id)
            else:
                stmt = select(Campaign).where(
                    Campaign.status.in_([CampaignStatus.RUNNING, CampaignStatus.APPROVED])
                )

            result = await session.execute(stmt)
            campaigns = result.scalars().all()

            # Connectors may fail if credentials not configured
            linkedin_connector = None
            x_connector = None
            blog_connector = None
            try:
                linkedin_connector = LinkedInConnector()
            except Exception as e:
                logger.warning(f"Could not initialize LinkedIn connector: {e}")
            try:
                x_connector = XConnector()
            except Exception as e:
                logger.warning(f"Could not initialize X connector: {e}")
            try:
                from ..automation_layer.connectors.blog_api import BlogConnector
                blog_connector = BlogConnector()
            except Exception as e:
                logger.warning(f"Could not initialize Blog connector: {e}")

            for campaign in campaigns:
                results['campaigns_checked'] += 1

                try:
                    platform = str(campaign.platform.value) if campaign.platform else 'linkedin'
                    config = campaign.config or {}
                    campaign_ext_id = config.get('campaign_id') or config.get('post_id') or str(campaign.id)

                    if not campaign_ext_id:
                        logger.debug(f"Campaign {campaign.id} has no external ID, skipping")
                        continue

                    if platform == 'linkedin':
                        if not linkedin_connector:
                            logger.debug(f"LinkedIn connector not available for campaign {campaign.id}")
                            continue
                        metrics = await linkedin_connector.get_campaign_metrics(campaign_ext_id)
                    elif platform in ['twitter', 'x']:
                        if not x_connector:
                            logger.debug(f"X connector not available for campaign {campaign.id}")
                            continue
                        metrics = await x_connector.get_campaign_metrics(campaign_ext_id)
                    elif platform == 'blog':
                        if not blog_connector:
                            logger.debug(f"Blog connector not available for campaign {campaign.id}")
                            continue
                        metrics = await blog_connector.get_post_metrics(campaign_ext_id)
                    else:
                        continue

                    if metrics.success and metrics.response_data:
                        data = metrics.response_data
                        campaign.impressions = data.get('impressions', campaign.impressions)
                        campaign.clicks = data.get('clicks', campaign.clicks)
                        campaign.conversions = data.get('conversions', campaign.conversions)
                        campaign.updated_at = datetime.utcnow()

                        if not campaign.config:
                            campaign.config = {}
                        campaign.config['last_poll'] = datetime.utcnow().isoformat()
                        campaign.config['platform_metrics'] = data
                        
                        results['metrics_updated'] += 1
                        logger.info(f"Updated metrics for campaign {campaign.id}: {data}")
                        
                except Exception as e:
                    error_msg = f"Error polling campaign {campaign.id}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            await session.commit()
        
        return results
    
    result = _run_async(_poll())
    logger.info(f"Platform metrics poll complete: {result}")
    return result


def sync_hubspot_deals() -> Dict[str, Any]:
    """
    Sync HubSpot deal stage changes to DelayedReward status.

    HIGH PRIORITY FIX: When deals move to 'Closed Won' or 'Closed Lost',
    update the corresponding DelayedReward status for proper attribution.
    """
    from ..data_layer.database.database import async_session_maker
    from ..data_layer.database.models import DelayedReward
    from ..automation_layer.connectors.hubspot_api import HubSpotAPIConnector
    from sqlalchemy import select
    from datetime import datetime
    
    logger.info("Syncing HubSpot deals to reward status...")
    
    async def _sync():
        results = {'deals_checked': 0, 'rewards_updated': 0, 'closed_won': 0, 'closed_lost': 0}
        
        connector = HubSpotAPIConnector()
        
        deals_result = await connector.get_deals(limit=100)
        
        if not deals_result.success:
            logger.error(f"Failed to fetch HubSpot deals: {deals_result.error}")
            return {'error': deals_result.error}
        
        deals = deals_result.data.get('deals', [])
        results['deals_checked'] = len(deals)
        
        async with async_session_maker() as session:
            for deal in deals:
                deal_props = deal.get('properties', {})
                email = deal_props.get('email') or deal_props.get('contact_email')
                stage = deal_props.get('dealstage', '')
                
                if not email:
                    continue
                
                stmt = select(DelayedReward).where(
                    DelayedReward.lead_email == email,
                    DelayedReward.status.in_(['pending', 'booked'])
                )
                
                result = await session.execute(stmt)
                reward = result.scalar_one_or_none()
                
                if not reward:
                    continue
                
                stage_lower = stage.lower()
                
                if 'closedwon' in stage_lower or 'closed-won' in stage_lower or stage_lower == 'won':
                    reward.status = 'converted'
                    reward.current_reward += 50.0  # Bonus for closed deal
                    reward.updated_at = datetime.utcnow()
                    if not reward.booking_data:
                        reward.booking_data = {}
                    reward.booking_data['hubspot_deal'] = deal
                    reward.booking_data['deal_closed_at'] = datetime.utcnow().isoformat()
                    
                    results['rewards_updated'] += 1
                    results['closed_won'] += 1
                    logger.info(f"Reward for {email} marked as converted (deal won)")
                    
                elif 'closedlost' in stage_lower or 'closed-lost' in stage_lower or stage_lower == 'lost':
                    reward.status = 'expired'
                    reward.updated_at = datetime.utcnow()
                    if not reward.booking_data:
                        reward.booking_data = {}
                    reward.booking_data['hubspot_deal'] = deal
                    reward.booking_data['deal_lost_at'] = datetime.utcnow().isoformat()
                    
                    results['rewards_updated'] += 1
                    results['closed_lost'] += 1
                    logger.info(f"Reward for {email} marked as expired (deal lost)")
            
            await session.commit()
        
        return results
    
    result = _run_async(_sync())
    logger.info(f"HubSpot sync complete: {result}")
    return result


def update_bandit_from_platform_metrics(experiment_id: str) -> Dict[str, Any]:
    """
    HIGH PRIORITY FIX: Connects real deployment performance back to
    the experiment learning loop for proper bandit updates.
    """
    from ..data_layer.database.database import async_session_maker
    from ..data_layer.database.models import BanditArm, Campaign, Experiment
    from sqlalchemy import select
    from datetime import datetime
    from uuid import UUID
    
    logger.info(f"Updating bandit arms from platform metrics for experiment {experiment_id}")
    
    async def _update():
        results = {'arms_updated': 0, 'total_pulls': 0}
        
        async with async_session_maker() as session:
            stmt = select(Experiment).where(Experiment.id == UUID(experiment_id))
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return {'error': f'Experiment {experiment_id} not found'}
            
            campaign_stmt = select(Campaign).where(Campaign.id == experiment.campaign_id)
            campaign_result = await session.execute(campaign_stmt)
            campaign = campaign_result.scalar_one_or_none()
            
            if not campaign or not campaign.impressions:
                return {'error': 'No campaign metrics available'}
            
            actual_ctr = campaign.clicks / campaign.impressions if campaign.impressions > 0 else 0
            
            arms_stmt = select(BanditArm).where(BanditArm.experiment_id == UUID(experiment_id))
            arms_result = await session.execute(arms_stmt)
            arms = arms_result.scalars().all()
            
            for arm in arms:
                if arm.arm_name == experiment.winner_variant:
                    arm.pulls = (arm.pulls or 0) + 1
                    
                    # Success threshold: CTR > 2%
                    if actual_ctr > 0.02:
                        arm.successes = (arm.successes or 0) + 1
                        arm.alpha = (arm.alpha or 1.0) + 1.0
                    else:
                        arm.failures = (arm.failures or 0) + 1  
                        arm.beta = (arm.beta or 1.0) + 1.0
                    
                    arm.total_reward = (arm.total_reward or 0.0) + actual_ctr
                    arm.updated_at = datetime.utcnow()
                    
                    results['arms_updated'] += 1
                    results['total_pulls'] = arm.pulls
                    logger.info(f"Updated arm {arm.arm_name}: CTR={actual_ctr:.2%}, α={arm.alpha}, β={arm.beta}")
            
            await session.commit()
        
        return results
    
    result = _run_async(_update())
    logger.info(f"Bandit update complete: {result}")
    return result


def cleanup_agent_memories(retention_days: int = 90) -> Dict[str, Any]:
    """
    Clean up old agent memories to prevent unbounded storage growth.

    Per Research Plan Section 6.4 - Agent Memory and Self-Improvement:
    "The system should schedule memory cleanup (daily, 90-day retention)"
    """
    from ..ai_layer.memory.episodic_memory import EpisodicMemoryStore

    logger.info(f"Starting agent memory cleanup (retention: {retention_days} days)...")

    async def _cleanup():
        results = {
            'agents_cleaned': [],
            'total_deleted': 0,
            'timestamp': datetime.now().isoformat()
        }

        # Agent types with episodic memory (Research Plan Section 6.4)
        agent_types = [
            'content_generator',
            'strategy_optimizer',
            'safety_validator',
            'market_scraper'
        ]

        for agent_name in agent_types:
            try:
                memory_store = EpisodicMemoryStore(agent_name)
                deleted_count = await memory_store.clear_old_memories(days=retention_days)

                results['agents_cleaned'].append({
                    'agent': agent_name,
                    'deleted': deleted_count
                })
                results['total_deleted'] += deleted_count

                logger.info(f"Cleaned {deleted_count} old memories for agent: {agent_name}")

            except Exception as e:
                logger.error(f"Failed to clean memories for {agent_name}: {e}")
                results['agents_cleaned'].append({
                    'agent': agent_name,
                    'error': str(e)
                })

        return results

    result = _run_async(_cleanup())
    logger.info(f"Agent memory cleanup complete: {result['total_deleted']} memories deleted")
    return result


def run_scheduled_maintenance() -> Dict[str, Any]:
    """Run all daily scheduled maintenance tasks."""
    logger.info("Starting scheduled maintenance tasks...")

    results = {
        'timestamp': datetime.now().isoformat(),
        'tasks': {}
    }

    # Research Plan Section 6.4
    try:
        memory_result = cleanup_agent_memories(retention_days=90)
        results['tasks']['memory_cleanup'] = memory_result
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")
        results['tasks']['memory_cleanup'] = {'error': str(e)}

    # Research Plan Section 2.3
    try:
        rewards_result = process_delayed_rewards()
        results['tasks']['delayed_rewards'] = rewards_result
    except Exception as e:
        logger.error(f"Delayed rewards processing failed: {e}")
        results['tasks']['delayed_rewards'] = {'error': str(e)}

    # Research Plan Section 8.3
    try:
        metrics_result = poll_platform_metrics()
        results['tasks']['platform_metrics'] = metrics_result
    except Exception as e:
        logger.error(f"Platform metrics polling failed: {e}")
        results['tasks']['platform_metrics'] = {'error': str(e)}

    # Research Plan Section 5
    try:
        hubspot_result = sync_hubspot_deals()
        results['tasks']['hubspot_sync'] = hubspot_result
    except Exception as e:
        logger.error(f"HubSpot sync failed: {e}")
        results['tasks']['hubspot_sync'] = {'error': str(e)}

    logger.info(f"Scheduled maintenance complete: {results}")
    return results


def run_autonomous_mlops_check() -> Dict[str, Any]:
    """
    Autonomous MLOps: scan active experiments, complete/log/promote automatically.

    Per Research Plan Section 8.1 - MLOps Infrastructure
    Per Research Plan Section 2.3 - MARL Promotion Gating
    """
    from ..ai_layer.learning.autonomous_mlops import run_autonomous_mlops_check as _check

    logger.info("Running autonomous MLOps check...")

    try:
        result = _check()
        logger.info(f"Autonomous MLOps check complete: {result}")
        return result
    except Exception as e:
        logger.error(f"Autonomous MLOps check failed: {e}")
        return {
            'error': str(e),
            'checked': 0,
            'completed': 0,
            'models_logged': 0,
            'models_promoted': 0
        }


def save_daily_governance_metrics() -> Dict[str, Any]:
    """
    Save daily governance metrics.

    Research Plan Section 10.2: Track Human Override Rate (<5% target)
    """
    from ..ai_layer.learning.governance_metrics_tracker import save_daily_governance_metrics as _save

    logger.info("Saving daily governance metrics...")

    try:
        record_id = _run_async(_save())
        logger.info(f"Daily governance metrics saved: {record_id}")
        return {
            'status': 'success',
            'record_id': record_id
        }
    except Exception as e:
        logger.error(f"Failed to save governance metrics: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def generate_weekly_learning_report() -> Dict[str, Any]:
    """
    Research Plan Section 10.2: Weekly Uplift Summary
    "An automated report showing the best-performing hooks and content angles."
    """
    from ..ai_layer.learning.weekly_learning_report import generate_weekly_report as _generate

    logger.info("Generating weekly learning report...")

    try:
        report = _run_async(_generate())
        logger.info(f"Weekly learning report generated: week {report.get('week_number')}/{report.get('year')}")
        return {
            'status': 'success',
            'week_number': report.get('week_number'),
            'year': report.get('year'),
            'best_hooks_count': len(report.get('best_hooks', [])),
            'recommendations_count': len(report.get('recommendations', []))
        }
    except Exception as e:
        logger.error(f"Failed to generate weekly report: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

