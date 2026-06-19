"""
Campaign deployment orchestrator
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import asyncio
from enum import Enum

from .connectors.linkedin_api import LinkedInConnector
from .connectors.x_api import XConnector
from .connectors.email_api import EmailConnector
from .connectors.mailchimp_api import MailchimpConnector
from .connectors.mailgun_api import MailgunConnector
from .connectors.blog_api import BlogConnector
from .mock_deployer import MockDeployer
from ..data_layer.database.models import Campaign, Content, Platform
from ..config.configuration_service import ConfigurationService
from ..data_layer.database.connection import get_async_session, get_sync_session
from ..config.settings import settings
from ..governance.content_formatter import format_content_for_platform

logger = logging.getLogger(__name__)

class DeploymentStatus(str, Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    PARTIALLY_DEPLOYED = "partially_deployed"

class CampaignDeployer:
    """
    Orchestrates campaign deployment across platforms
    """

    def __init__(self):
        self.connectors = {}
        self.mock_deployer = MockDeployer()
        self._initialized = False

    async def initialize(self):
        """Initialize connectors using database configuration"""
        if self._initialized:
            return

        sync_db = get_sync_session()
        try:
            config_service = ConfigurationService(sync_db)
            
            linkedin_client_id = config_service.get_value("LINKEDIN_CLIENT_ID")
            linkedin_token = config_service.get_value("LINKEDIN_ACCESS_TOKEN")
            
            twitter_api_key = config_service.get_value("TWITTER_API_KEY")
            
            mailgun_key = config_service.get_value("MAILGUN_API_KEY")
            mailgun_domain = config_service.get_value("MAILGUN_DOMAIN")
            
            if linkedin_client_id and linkedin_token:
                linkedin_config = {
                    'client_id': linkedin_client_id,
                    'access_token': linkedin_token,
                    'client_secret': config_service.get_value("LINKEDIN_CLIENT_SECRET"),
                    'account_id': config_service.get_value("LINKEDIN_ACCOUNT_ID"),
                    'organization_id': config_service.get_value("LINKEDIN_ORGANIZATION_ID")
                }
                self.connectors['linkedin'] = LinkedInConnector(config=linkedin_config)
                logger.info("LinkedIn connector initialized from DB config")

            if twitter_api_key:
                 self.connectors['twitter'] = XConnector()
                 self.connectors['x'] = self.connectors['twitter']

            if mailgun_key and mailgun_domain:
                self.connectors['email'] = MailgunConnector()
                self.connectors['mailgun'] = self.connectors['email']
                logger.info("Mailgun connector initialized from DB config (primary email)")
            else:
                mailchimp_key = config_service.get_value("MAILCHIMP_API_KEY")
                if mailchimp_key:
                    self.connectors['email'] = MailchimpConnector()
                    self.connectors['mailchimp'] = self.connectors['email']
                    logger.info("Mailchimp connector initialized from DB config (secondary email)")
                else:
                    sendgrid_key = config_service.get_value("SENDGRID_API_KEY")
                    if sendgrid_key:
                        self.connectors['email'] = EmailConnector()
                        self.connectors['sendgrid'] = self.connectors['email']
                        logger.info("SendGrid connector initialized from DB config (fallback email)")
            
            blog_cms_url = config_service.get_value("BLOG_CMS_URL")
            if blog_cms_url:
                self.connectors['blog'] = BlogConnector()
                logger.info("Blog connector initialized from DB config")

            self.mock_mode_enabled = config_service.get_value("MOCK_MODE_ENABLED", False)
            self.enable_mock_deployment = config_service.get_value("ENABLE_MOCK_DEPLOYMENT", False)
        finally:
            sync_db.close()
            
            if not self.connectors:
                if self.enable_mock_deployment:
                    logger.info("🎭 No external API connectors - Using MockDeployer (DB Config)")
                else:
                    logger.warning("⚠️ No connectors and Mock Disabled - Deployments will fail")
            else:
                 logger.info(f"✅ Configured connectors: {', '.join(self.connectors.keys())}")
            
            self._initialized = True

    def _initialize_connectors(self):
        """Initialize platform connectors based on configuration"""

        if settings.LINKEDIN_CLIENT_ID:
            self.connectors['linkedin'] = LinkedInConnector()
            logger.info("LinkedIn connector initialized")

        if settings.TWITTER_API_KEY:
            x_connector = XConnector()
            self.connectors['twitter'] = x_connector
            self.connectors['x'] = x_connector
            logger.info("Twitter/X connector initialized")

        if getattr(settings, 'MAILGUN_API_KEY', None) and getattr(settings, 'MAILGUN_DOMAIN', None):
            self.connectors['email'] = MailgunConnector()
            self.connectors['mailgun'] = self.connectors['email']
            logger.info("Mailgun connector initialized (primary email)")
        elif getattr(settings, 'MAILCHIMP_API_KEY', None):
            self.connectors['email'] = MailchimpConnector()
            self.connectors['mailchimp'] = self.connectors['email']
            logger.info("Mailchimp connector initialized (secondary email)")
        elif settings.SENDGRID_API_KEY:
            self.connectors['email'] = EmailConnector()
            self.connectors['sendgrid'] = self.connectors['email']
            logger.info("SendGrid connector initialized (legacy fallback email)")

        if getattr(settings, 'BLOG_CMS_URL', None):
            self.connectors['blog'] = BlogConnector()
            logger.info("Blog connector initialized")

        if not self.connectors:
            if settings.ENABLE_MOCK_DEPLOYMENT:
                logger.info("🎭 No external API connectors configured - MockDeployer will generate realistic metrics (ENABLE_MOCK_DEPLOYMENT=True)")
            else:
                logger.warning("⚠️  No external API connectors configured and ENABLE_MOCK_DEPLOYMENT=False - deployments will fail")
        else:
            logger.info(f"✅ Configured connectors: {', '.join(self.connectors.keys())}")

    async def update_arm_from_deployment(
        self,
        campaign_id: str,
        arm_id: str,
        deployment_metrics: Dict[str, Any]
    ) -> bool:
        """
        Update bandit arm with actual deployment metrics.

        Per Research Plan - connects deployment metrics to BanditArm.update
        for automatic experiment arm reward updates.

        Args:
            campaign_id: Campaign ID
            arm_id: Arm ID (variant being tested)
            deployment_metrics: Metrics from deployment (clicks, impressions, conversions)

        Returns:
            True if update successful
        """
        try:
            from ..data_layer.database.connection import get_async_session
            from ..data_layer.database.models import Experiment, BanditArm
            from sqlalchemy import select
            from uuid import UUID

            async with get_async_session() as session:
                stmt = select(Experiment).where(
                    Experiment.campaign_id == UUID(campaign_id)
                )
                result = await session.execute(stmt)
                experiment = result.scalar_one_or_none()

                if not experiment:
                    logger.debug(f"No experiment found for campaign {campaign_id}")
                    return False

                arm_stmt = select(BanditArm).where(
                    BanditArm.experiment_id == experiment.id,
                    BanditArm.arm_id == arm_id
                )
                arm_result = await session.execute(arm_stmt)
                arm = arm_result.scalar_one_or_none()

                if not arm:
                    logger.debug(f"No arm found with id {arm_id} for experiment {experiment.id}")
                    return False

                clicks = deployment_metrics.get('clicks', 0)
                impressions = deployment_metrics.get('impressions', 0)
                conversions = deployment_metrics.get('conversions', 0)

                arm.pulls = (arm.pulls or 0) + impressions
                arm.successes = (arm.successes or 0) + clicks

                # Thompson Sampling Bayesian update: α += successes, β += failures
                arm.alpha = (arm.alpha or 1.0) + clicks
                arm.beta = (arm.beta or 1.0) + (impressions - clicks)

                conversion_bonus = conversions * 10.0
                arm.total_reward = (arm.total_reward or 0) + clicks + conversion_bonus

                arm.last_pulled_at = datetime.utcnow()

                experiment.total_impressions = (experiment.total_impressions or 0) + impressions
                experiment.total_conversions = (experiment.total_conversions or 0) + conversions

                await session.commit()

                logger.info(
                    f"Updated arm {arm_id} with deployment metrics",
                    extra={
                        "event": "arm_updated_from_deployment",
                        "campaign_id": campaign_id,
                        "experiment_id": str(experiment.id),
                        "arm_id": arm_id,
                        "impressions": impressions,
                        "clicks": clicks,
                        "conversions": conversions,
                        "new_alpha": arm.alpha,
                        "new_beta": arm.beta
                    }
                )

                return True

        except Exception as e:
            logger.error(f"Failed to update arm from deployment: {e}")
            return False

    async def deploy(
        self,
        content_id: str,
        platform: str,
        content: Dict[str, Any],
        campaign_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deploy campaign content to specified platform

        Args:
            content_id: Content ID
            platform: Target platform (linkedin, twitter, email)
            content: Content dictionary with headline, body, etc.
            campaign_config: Optional campaign configuration

        Returns:
            Deployment result with success status
        """
        await self.initialize()

        platform = platform.value if hasattr(platform, 'value') else str(platform)

        start_time = datetime.now()

        try:
            # Expand [CLM_XXX] citation placeholders to human-readable text for external platforms
            formatted_content = format_content_for_platform(content, platform)
            
            logger.info(
                "Starting deployment",
                extra={
                    "event": "deployment_started",
                    "content_id": content_id,
                    "platform": platform,
                    "headline": formatted_content.get('headline', '')[:50] if formatted_content.get('headline') else None,
                    "citations_expanded": formatted_content.get('_formatting', {}).get('citations_expanded', False)
                }
            )

            connector = self.connectors.get(platform)

            if not connector:
                mock_enabled = self.mock_mode_enabled and self.enable_mock_deployment
                if mock_enabled:
                    logger.info(
                        "No connector available - using MockDeployer (MOCK_MODE_ENABLED=True)",
                        extra={
                            "event": "mock_deployment_fallback",
                            "content_id": content_id,
                            "platform": platform,
                            "reason": "Platform connector not configured",
                            "is_mock": True
                        }
                    )

                    from ..data_layer.repositories.campaign_repo import CampaignRepository
                    from ..data_layer.database.connection import get_async_session

                    async with get_async_session() as session:
                        campaign_repo = CampaignRepository(session)

                        campaign_id = campaign_config.get('campaign_id') if campaign_config else None

                        simulation_results = {}
                        budget = 1000.0
                        duration_days = 7

                        if campaign_id:
                            campaign = await campaign_repo.get_by_id(campaign_id)
                            if campaign and campaign.config:
                                simulation_results = campaign.config.get('simulation_results', {})
                                budget = campaign.budget_total or 1000.0
                                if campaign.start_date and campaign.end_date:
                                    duration_days = (campaign.end_date - campaign.start_date).days

                    mock_result = self.mock_deployer.deploy_campaign(
                        campaign_id=campaign_id or content_id,
                        simulation_results=simulation_results,
                        budget=budget,
                        duration_days=duration_days
                    )

                    duration = (datetime.now() - start_time).total_seconds()
                    logger.info(
                        "Mock deployment completed",
                        extra={
                            "event": "mock_deployment_completed",
                            "content_id": content_id,
                            "platform": platform,
                            "campaign_id": campaign_id,
                            "mape": mock_result['validation']['overall_mape'],
                            "duration_seconds": round(duration, 3)
                        }
                    )

                    return {
                        'success': True,
                        'post_id': mock_result['external_campaign_id'],
                        'platform': platform,
                        'deployment_type': 'mock',
                        'is_mock': True,
                        'metrics': mock_result['actual_metrics'],
                        'validation': mock_result['validation'],
                        'daily_metrics': mock_result['daily_metrics'],
                        'timestamp': datetime.utcnow()
                    }
                else:
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.error(
                        "No connector available for platform",
                        extra={
                            "event": "deployment_error",
                            "content_id": content_id,
                            "platform": platform,
                            "error": "Platform not configured",
                            "mock_deployment_enabled": False,
                            "duration_seconds": round(duration, 3)
                        }
                    )
                    return {
                        'success': False,
                        'error': f"Platform {platform} not configured (MOCK_MODE_ENABLED=False — real APIs required)",
                        'is_mock': False,
                        'timestamp': datetime.utcnow()
                    }

            if not await connector.validate_credentials():
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    "Invalid credentials for platform",
                    extra={
                        "event": "deployment_error",
                        "content_id": content_id,
                        "platform": platform,
                        "error": "Invalid credentials",
                        "duration_seconds": round(duration, 3)
                    }
                )
                return {
                    'success': False,
                    'error': f"Invalid credentials for {platform}",
                    'timestamp': datetime.utcnow()
                }

            if platform == 'linkedin':
                result = await connector.create_sponsored_content({
                    'headline': formatted_content.get('headline', ''),
                    'body': formatted_content.get('body', ''),
                    'landing_url': campaign_config.get('landing_url', 'https://example.com/book-a-demo') if campaign_config else 'https://example.com/book-a-demo',
                    'image_url': formatted_content.get('image_url')
                })
                if result.success and result.response_data:
                    result = {
                        'success': True,
                        'post_id': result.response_data.get('id', f"li_{datetime.now().timestamp()}"),
                        'platform': 'linkedin'
                    }
                else:
                    result = {
                        'success': False,
                        'error': result.error if hasattr(result, 'error') else 'LinkedIn deployment failed'
                    }
            elif platform in ['twitter', 'x']:
                tweet_content = f"{formatted_content.get('headline', '')}\n\n{formatted_content.get('body', '')}"
                if formatted_content.get('cta'):
                    tweet_content = f"{tweet_content}\n\n{formatted_content.get('cta')}"
                result = await connector.create_tweet(tweet_content[:280])
                if result.success and result.response_data:
                    result = {
                        'success': True,
                        'post_id': result.response_data.get('tweet_id', f"tw_{datetime.now().timestamp()}"),
                        'platform': platform
                    }
                else:
                    result = {
                        'success': False,
                        'error': result.error if hasattr(result, 'error') else 'Twitter/X deployment failed'
                    }
            elif platform == 'email':
                recipients = campaign_config.get('email_recipients', []) if campaign_config else []
                if not recipients:
                    try:
                        sync_db = get_sync_session()
                        try:
                            config_service = ConfigurationService(sync_db)
                            default_recipients = config_service.get_value("DEFAULT_EMAIL_RECIPIENTS", "")
                        finally:
                            sync_db.close()
                        if default_recipients:
                            recipients = [{'email': r.strip(), 'name': 'Subscriber'}
                                          for r in default_recipients.split(',') if r.strip()]
                    except Exception:
                        pass
                if not recipients:
                    return {
                        'success': False,
                        'error': ("No email recipients configured. Set email_recipients in campaign config "
                                  "or DEFAULT_EMAIL_RECIPIENTS in System Settings → Email Configuration"),
                        'timestamp': datetime.utcnow()
                    }
                result = await connector.send_bulk_emails(
                    recipients=recipients,
                    subject=formatted_content.get('headline', ''),
                    html_template=f"<html><body><p>{formatted_content.get('body', '')}</p></body></html>",
                    text_template=formatted_content.get('body', '')
                )
                if result.success and result.response_data:
                    result = {
                        'success': True,
                        'post_id': result.response_data.get('batch_id', f"email_{datetime.now().timestamp()}"),
                        'platform': 'email'
                    }
                else:
                    result = {
                        'success': False,
                        'error': result.error if hasattr(result, 'error') else 'Email deployment failed'
                    }
            elif platform == 'blog':
                # Extract blog-specific fields from formatted content
                title = formatted_content.get('headline', formatted_content.get('title', ''))
                body = formatted_content.get('body', formatted_content.get('content', ''))
                meta_desc = formatted_content.get('meta_description', '')
                seo_keywords = formatted_content.get('seo_keywords', '')
                tags = [t.strip() for t in seo_keywords.split(',') if t.strip()] if seo_keywords else []
                blog_status = campaign_config.get('blog_publish_status', 'draft') if campaign_config else 'draft'

                result = await connector.create_post(
                    title=title,
                    content=body,
                    meta_description=meta_desc,
                    tags=tags,
                    status=blog_status,
                )
                if result.success and result.response_data:
                    result = {
                        'success': True,
                        'post_id': result.response_data.get('post_id', f"blog_{datetime.now().timestamp()}"),
                        'url': result.response_data.get('url', ''),
                        'platform': 'blog'
                    }
                else:
                    result = {
                        'success': False,
                        'error': result.error if hasattr(result, 'error') else 'Blog deployment failed'
                    }
            else:
                result = {
                    'success': False,
                    'error': f"Unsupported platform: {platform}"
                }

            duration = (datetime.now() - start_time).total_seconds()

            if result.get('success', False):
                logger.info(
                    "Deployment completed successfully",
                    extra={
                        "event": "deployment_completed",
                        "content_id": content_id,
                        "platform": platform,
                        "post_id": result.get('post_id', ''),
                        "duration_seconds": round(duration, 3)
                    }
                )
                await self._send_slack_notification(
                    f"✅ Deployment succeeded on {platform} (content: {content_id}, "
                    f"post: {result.get('post_id', 'N/A')}, {duration:.1f}s)"
                )
            else:
                logger.error(
                    f"Deployment failed on {platform}: {result.get('error', 'Unknown error')}",
                    extra={
                        "event": "deployment_failed",
                        "content_id": content_id,
                        "platform": platform,
                        "error": result.get('error', 'Unknown error'),
                        "duration_seconds": round(duration, 3)
                    }
                )
                await self._send_slack_notification(
                    f"❌ Deployment failed on {platform} (content: {content_id}): "
                    f"{result.get('error', 'Unknown error')}"
                )

            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                "Deployment failed with exception",
                extra={
                    "event": "deployment_exception",
                    "content_id": content_id,
                    "platform": platform,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 3)
                },
                exc_info=True
            )
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }
        finally:
            if connector and hasattr(connector, 'close'):
                try:
                    await connector.close()
                    if hasattr(connector, 'session'):
                        connector.session = None
                except Exception:
                    pass

    async def _send_slack_notification(self, message: str):
        """Send a Slack notification if webhook is configured."""
        try:
            sync_db = get_sync_session()
            try:
                config_service = ConfigurationService(sync_db)
                webhook_url = config_service.get_value("SLACK_WEBHOOK_URL")
            finally:
                sync_db.close()

            if not webhook_url or not webhook_url.strip():
                return

            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    webhook_url,
                    json={"text": message, "username": "Agentic Deployer", "icon_emoji": ":rocket:"}
                )
        except Exception as e:
            logger.debug(f"Slack notification skipped: {e}")

    async def _deploy_to_linkedin(self, connector: LinkedInConnector, campaign: Campaign, content: Content) -> Dict[str, Any]:
        """Deploy to LinkedIn"""
        try:
            from ..governance.content_formatter import expand_claim_citations
            formatted_headline = expand_claim_citations(content.headline, "inline")
            formatted_body = expand_claim_citations(content.body, "inline")
            
            if not campaign.config.get('linkedin_campaign_id'):
                campaign_data = {
                    'name': campaign.name,
                    'account_id': settings.get('LINKEDIN_ACCOUNT_ID'),
                    'daily_budget': campaign.budget_daily_limit,
                    'duration_days': (campaign.end_date - campaign.start_date).days,
                    'targeting': {
                        'locations': campaign.target_demographics.get('regions', []),
                        'industries': campaign.target_demographics.get('industries', []),
                        'company_sizes': campaign.target_demographics.get('company_sizes', []),
                        'job_titles': campaign.target_demographics.get('job_titles', [])
                    }
                }
                
                campaign_result = await connector.create_campaign(campaign_data)
                
                if not campaign_result.success:
                    return {
                        'status': DeploymentStatus.FAILED,
                        'error': campaign_result.error,
                        'platform': 'linkedin'
                    }
                
                if campaign_result.response_data:
                    linkedin_campaign_id = campaign_result.response_data.get('id')
                else:
                    return {
                        'status': DeploymentStatus.FAILED,
                        'error': 'No response data from LinkedIn campaign creation',
                        'platform': 'linkedin'
                    }
            else:
                linkedin_campaign_id = campaign.config['linkedin_campaign_id']
            
            content_data = {
                'organization_id': getattr(settings, 'LINKEDIN_ORGANIZATION_ID', None),
                'headline': formatted_headline,
                'body': formatted_body,
                'landing_url': campaign.config.get('landing_url', 'https://example.com/book-a-demo') if campaign.config else 'https://example.com/book-a-demo',
                'image_url': content.image_url if hasattr(content, 'image_url') else None
            }
            
            content_result = await connector.create_sponsored_content(content_data)
            
            if content_result.success and content_result.response_data:
                return {
                    'status': DeploymentStatus.DEPLOYED,
                    'platform': 'linkedin',
                    'campaign_id': linkedin_campaign_id,
                    'content_id': content_result.response_data.get('id'),
                    'timestamp': datetime.utcnow()
                }
            else:
                return {
                    'status': DeploymentStatus.FAILED,
                    'error': content_result.error,
                    'platform': 'linkedin'
                }
                
        except Exception as e:
            logger.error(f"LinkedIn deployment failed: {e}")
            return {
                'status': DeploymentStatus.FAILED,
                'error': str(e),
                'platform': 'linkedin'
            }
    
    async def _deploy_to_twitter(self, connector: XConnector, campaign: Campaign, content: Content) -> Dict[str, Any]:
        """Deploy to Twitter/X"""
        try:
            from ..governance.content_formatter import expand_claim_citations
            formatted_body = expand_claim_citations(content.body, "inline")
            formatted_cta = expand_claim_citations(content.cta, "inline") if content.cta else content.cta
            
            tweets = self._split_content_for_twitter(formatted_body, formatted_cta)
            
            if len(tweets) == 1:
                result = await connector.create_tweet(tweets[0])
            else:
                created_tweets = []
                for i, tweet in enumerate(tweets):
                    if i > 0:
                        twitter_username = getattr(settings, 'TWITTER_USERNAME', 'agentic')
                        tweet = f"@{twitter_username} {tweet}"
                    
                    result = await connector.create_tweet(tweet)
                    if result.success and result.response_data:
                        created_tweets.append(result.response_data)
                    else:
                        break
                
                if created_tweets:
                    return {
                        'status': DeploymentStatus.DEPLOYED,
                        'platform': 'twitter',
                        'tweet_ids': [t['tweet_id'] for t in created_tweets],
                        'thread_size': len(created_tweets),
                        'timestamp': datetime.utcnow()
                    }
                else:
                    return {
                        'status': DeploymentStatus.FAILED,
                        'platform': 'twitter',
                        'error': 'Failed to create tweets'
                    }
            
            if result.success:
                return {
                    'status': DeploymentStatus.DEPLOYED,
                    'platform': 'twitter',
                    'tweet_id': result.response_data.get('tweet_id'),
                    'timestamp': datetime.utcnow()
                }
            else:
                return {
                    'status': DeploymentStatus.FAILED,
                    'error': result.error,
                    'platform': 'twitter'
                }
                
        except Exception as e:
            logger.error(f"Twitter deployment failed: {e}")
            return {
                'status': DeploymentStatus.FAILED,
                'error': str(e),
                'platform': 'twitter'
            }
    
    def _split_content_for_twitter(self, content: str, cta: str) -> List[str]:
        """Split content into tweet-sized chunks"""
        max_length = 270  # Leave room for links/hashtags
        
        full_content = f"{content}\n\n{cta}"
        if len(full_content) <= max_length:
            return [full_content]
        
        tweets = []
        words = content.split()
        current_tweet = ""
        
        for word in words:
            if len(current_tweet) + len(word) + 1 <= max_length:
                current_tweet += f" {word}"
            else:
                tweets.append(current_tweet.strip())
                current_tweet = word
        
        if current_tweet:
            tweets.append(current_tweet.strip())
        
        if tweets:
            last_tweet = tweets[-1]
            if len(last_tweet) + len(cta) + 2 <= max_length:
                tweets[-1] = f"{last_tweet}\n\n{cta}"
            else:
                tweets.append(cta)
        
        return tweets
    
    async def _deploy_to_email(self, connector: EmailConnector, campaign: Campaign, content: Content) -> Dict[str, Any]:
        """Deploy to email"""
        try:
            from ..governance.content_formatter import expand_claim_citations
            formatted_headline = expand_claim_citations(content.headline, "footnote")
            formatted_body = expand_claim_citations(content.body, "footnote")
            
            recipients = campaign.config.get('email_recipients', [])
            
            if not recipients:
                recipients = [
                    {'email': 'test@example.com', 'name': 'Test User', 'persona': campaign.target_persona}
                ]
            
            class FormattedContent:
                def __init__(self, headline, body, cta):
                    self.headline = headline
                    self.body = body
                    self.cta = cta
            
            formatted_content_obj = FormattedContent(
                formatted_headline,
                formatted_body,
                expand_claim_citations(content.cta, "footnote") if content.cta else content.cta
            )
            
            result = await connector.send_bulk_emails(
                recipients=recipients,
                subject=formatted_headline,
                html_template=self._format_email_html(formatted_content_obj),
                text_template=formatted_body
            )
            
            if result.success:
                return {
                    'status': DeploymentStatus.DEPLOYED,
                    'platform': 'email',
                    'batch_id': result.response_data.get('batch_id'),
                    'recipients_count': result.response_data.get('recipients_count'),
                    'timestamp': datetime.utcnow()
                }
            else:
                return {
                    'status': DeploymentStatus.FAILED,
                    'error': result.error,
                    'platform': 'email'
                }
                
        except Exception as e:
            logger.error(f"Email deployment failed: {e}")
            return {
                'status': DeploymentStatus.FAILED,
                'error': str(e),
                'platform': 'email'
            }
    
    def _format_email_html(self, content: Content) -> str:
        """Format content as HTML email"""
        body_html = content.body.replace('\n', '<br>')

        html_template = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #1e3a8a;">{content.headline}</h2>

                <div style="margin: 20px 0;">
                    {body_html}
                </div>

                <div style="margin-top: 30px; text-align: center;">
                    <a href="https://example.com/demo"
                       style="display: inline-block; padding: 12px 30px; background-color: #667eea;
                              color: white; text-decoration: none; border-radius: 5px;">
                        {content.cta}
                    </a>
                </div>

                <hr style="margin-top: 40px; border: none; border-top: 1px solid #e5e5e5;">

                <p style="font-size: 12px; color: #888; text-align: center;">
                    © 2025 Agentic AI. All rights reserved.<br>
                    <a href="{{{{unsubscribe}}}}" style="color: #888;">Unsubscribe</a>
                </p>
            </div>
        </body>
        </html>
        """
        return html_template
    
    async def deploy_batch(self, deployments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deploy multiple campaigns in batch"""
        results = []
        
        for deployment in deployments:
            campaign = deployment['campaign']
            content = deployment['content']
            
            result = await self.deploy(campaign, content)
            results.append({
                'campaign_id': campaign.id,
                'platform': campaign.platform,
                'result': result
            })
            
            await asyncio.sleep(1)
        
        successful = sum(1 for r in results if r['result']['status'] == DeploymentStatus.DEPLOYED)
        failed = sum(1 for r in results if r['result']['status'] == DeploymentStatus.FAILED)
        
        return {
            'total': len(results),
            'successful': successful,
            'failed': failed,
            'results': results,
            'timestamp': datetime.utcnow()
        }
    
    async def close_all(self):
        """Close all connector sessions"""
        for connector in self.connectors.values():
            if hasattr(connector, 'close'):
                await connector.close()