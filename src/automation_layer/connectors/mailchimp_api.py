"""
Mailchimp Email Marketing Connector

Per Research Plan Section 6.3 - Deployer Tools:
"LinkedIn Ads API, X Ads API, Mailchimp API"

Provides email marketing capabilities through Mailchimp's Marketing API:
- Campaign creation and management
- Audience/list management
- Email sending and scheduling
- Engagement metrics and analytics
"""
import logging
import aiohttp
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_connector import BaseConnector, PlatformResponse
from ...config.settings import settings

logger = logging.getLogger(__name__)


class MailchimpConnector(BaseConnector):
    """
    Mailchimp Marketing API connector for email campaigns

    Per Research Plan Section 6.3 - Uses Mailchimp API for email pilot campaigns
    """

    def __init__(self):
        # Get API key and extract data center from key format: xxxx-us1
        api_key = getattr(settings, 'MAILCHIMP_API_KEY', None)
        self.data_center = "us1"  # Default

        if api_key and "-" in api_key:
            self.data_center = api_key.split("-")[-1]

        config = {
            'api_key': api_key,
            'rate_limit': 10,  # Mailchimp has strict rate limits
            'data_center': self.data_center,
            'from_email': getattr(settings, 'MAILCHIMP_FROM_EMAIL', 'marketing@example.com'),
            'from_name': getattr(settings, 'MAILCHIMP_FROM_NAME', 'Agentic AI'),
            'list_id': getattr(settings, 'MAILCHIMP_LIST_ID', None)
        }
        super().__init__(config)

        self.base_url = f"https://{self.data_center}.api.mailchimp.com/3.0"
        self.enabled = bool(api_key)

        if not self.enabled:
            logger.warning("Mailchimp connector not configured (MAILCHIMP_API_KEY not set)")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Mailchimp API"""
        import base64
        # Mailchimp uses HTTP Basic Auth with "anystring" as username
        auth_string = f"anystring:{self.config['api_key']}"
        encoded = base64.b64encode(auth_string.encode()).decode()
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def _get_subscriber_hash(email: str) -> str:
        """Get MD5 hash of lowercase email for Mailchimp API"""
        return hashlib.md5(email.lower().encode()).hexdigest()

    async def validate_credentials(self) -> bool:
        """Validate Mailchimp API credentials"""
        if not self.enabled:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/",
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Mailchimp credentials validated: {data.get('account_name')}")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Mailchimp validation failed: {response.status} - {error}")
                        return False

        except Exception as e:
            logger.error(f"Mailchimp credential validation error: {e}")
            return False

    async def get_lists(self) -> PlatformResponse:
        """Get all audience lists"""
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="get_lists",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/lists",
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        lists = [
                            {
                                "id": lst["id"],
                                "name": lst["name"],
                                "member_count": lst["stats"]["member_count"],
                                "created": lst["date_created"]
                            }
                            for lst in data.get("lists", [])
                        ]
                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="get_lists",
                            response_data={"lists": lists, "total": len(lists)}
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="get_lists",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to get Mailchimp lists: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="get_lists",
                response_data={},
                error=str(e)
            )

    async def send_bulk_emails(
        self,
        recipients: List[Dict[str, Any]],
        subject: str,
        html_template: str,
        text_template: str = None
    ) -> PlatformResponse:
        """Send bulk emails via Mailchimp campaign create-and-send flow.
        
        Matches the interface used by CampaignDeployer.deploy(platform='email').
        """
        if not self.enabled:
            return PlatformResponse(
                success=False, platform="mailchimp", action="send_bulk",
                response_data={}, error="Mailchimp not configured"
            )

        try:
            list_id = self.config.get('list_id')
            if not list_id:
                return PlatformResponse(
                    success=False, platform="mailchimp", action="send_bulk",
                    response_data={}, error="MAILCHIMP_LIST_ID not configured"
                )

            # Create campaign
            campaign_resp = await self.create_campaign({
                'list_id': list_id,
                'subject': subject,
                'from_name': self.config.get('from_name', 'Agentic AI'),
                'from_email': self.config.get('from_email', 'marketing@example.com'),
                'html_content': html_template,
                'text_content': text_template or '',
            })

            if not campaign_resp.success:
                return PlatformResponse(
                    success=False, platform="mailchimp", action="send_bulk",
                    response_data=campaign_resp.response_data,
                    error=f"Campaign creation failed: {campaign_resp.error_message}"
                )

            campaign_id = campaign_resp.response_data.get('campaign_id')
            if not campaign_id:
                return PlatformResponse(
                    success=False, platform="mailchimp", action="send_bulk",
                    response_data={}, error="No campaign ID returned"
                )

            # Send the campaign
            send_resp = await self.send_campaign(campaign_id)
            if send_resp.success:
                logger.info(f"Bulk email sent via Mailchimp campaign {campaign_id}: {len(recipients)} intended recipients")
                return PlatformResponse(
                    success=True, platform="mailchimp", action="send_bulk",
                    response_data={
                        'campaign_id': campaign_id,
                        'recipients_count': len(recipients)
                    }
                )
            else:
                return PlatformResponse(
                    success=False, platform="mailchimp", action="send_bulk",
                    response_data={'campaign_id': campaign_id},
                    error=f"Send failed: {send_resp.error_message}"
                )

        except Exception as e:
            logger.error(f"Mailchimp send_bulk_emails error: {e}")
            return PlatformResponse(
                success=False, platform="mailchimp", action="send_bulk",
                response_data={}, error=str(e)
            )

    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        """
        Create an email marketing campaign

        Args:
            campaign_data: Campaign configuration including:
                - list_id: Audience list ID
                - subject: Email subject line
                - from_name: Sender name
                - reply_to: Reply-to email
                - title: Campaign title (internal name)
        """
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="create_campaign",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            list_id = campaign_data.get('list_id') or self.config.get('list_id')
            if not list_id:
                return PlatformResponse(
                    success=False,
                    platform="mailchimp",
                    action="create_campaign",
                    response_data={},
                    error="No list_id provided"
                )

            payload = {
                "type": "regular",
                "recipients": {
                    "list_id": list_id
                },
                "settings": {
                    "subject_line": campaign_data.get("subject", "Newsletter"),
                    "title": campaign_data.get("title", f"Campaign {datetime.now().strftime('%Y-%m-%d')}"),
                    "from_name": campaign_data.get("from_name", self.config["from_name"]),
                    "reply_to": campaign_data.get("reply_to", self.config["from_email"]),
                    "auto_footer": True,
                    "inline_css": True
                },
                "tracking": {
                    "opens": True,
                    "html_clicks": True,
                    "text_clicks": True
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/campaigns",
                    headers=self._get_auth_headers(),
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        campaign_id = data.get("id")

                        # If HTML content provided, set it
                        if campaign_data.get("html_content"):
                            await self._set_campaign_content(
                                session,
                                campaign_id,
                                campaign_data["html_content"]
                            )

                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="create_campaign",
                            response_data={
                                "campaign_id": campaign_id,
                                "status": data.get("status"),
                                "web_id": data.get("web_id")
                            }
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="create_campaign",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to create Mailchimp campaign: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="create_campaign",
                response_data={},
                error=str(e)
            )

    async def _set_campaign_content(
        self,
        session: aiohttp.ClientSession,
        campaign_id: str,
        html_content: str
    ) -> bool:
        """Set HTML content for a campaign"""
        try:
            async with session.put(
                f"{self.base_url}/campaigns/{campaign_id}/content",
                headers=self._get_auth_headers(),
                json={"html": html_content}
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to set campaign content: {e}")
            return False

    async def send_campaign(self, campaign_id: str) -> PlatformResponse:
        """Send a campaign immediately"""
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="send_campaign",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/campaigns/{campaign_id}/actions/send",
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status == 204:
                        logger.info(f"Mailchimp campaign {campaign_id} sent successfully")
                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="send_campaign",
                            response_data={"campaign_id": campaign_id, "status": "sent"}
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="send_campaign",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to send Mailchimp campaign: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="send_campaign",
                response_data={},
                error=str(e)
            )

    async def schedule_campaign(
        self,
        campaign_id: str,
        schedule_time: datetime
    ) -> PlatformResponse:
        """Schedule a campaign for future delivery"""
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="schedule_campaign",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            payload = {
                "schedule_time": schedule_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/campaigns/{campaign_id}/actions/schedule",
                    headers=self._get_auth_headers(),
                    json=payload
                ) as response:
                    if response.status == 204:
                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="schedule_campaign",
                            response_data={
                                "campaign_id": campaign_id,
                                "scheduled_for": schedule_time.isoformat()
                            }
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="schedule_campaign",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to schedule Mailchimp campaign: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="schedule_campaign",
                response_data={},
                error=str(e)
            )

    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        """Get campaign performance metrics"""
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="get_metrics",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/reports/{campaign_id}",
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        metrics = {
                            "emails_sent": data.get("emails_sent", 0),
                            "opens": data.get("opens", {}).get("opens_total", 0),
                            "unique_opens": data.get("opens", {}).get("unique_opens", 0),
                            "open_rate": data.get("opens", {}).get("open_rate", 0),
                            "clicks": data.get("clicks", {}).get("clicks_total", 0),
                            "unique_clicks": data.get("clicks", {}).get("unique_clicks", 0),
                            "click_rate": data.get("clicks", {}).get("click_rate", 0),
                            "unsubscribes": data.get("unsubscribed", 0),
                            "bounces": data.get("bounces", {}).get("hard_bounces", 0) +
                                       data.get("bounces", {}).get("soft_bounces", 0),
                            "abuse_reports": data.get("abuse_reports", 0)
                        }

                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="get_metrics",
                            response_data=metrics
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="get_metrics",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to get Mailchimp metrics: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="get_metrics",
                response_data={},
                error=str(e)
            )

    async def add_subscriber(
        self,
        email: str,
        list_id: str = None,
        merge_fields: Dict[str, str] = None,
        tags: List[str] = None
    ) -> PlatformResponse:
        """Add a subscriber to an audience list"""
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="add_subscriber",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            list_id = list_id or self.config.get('list_id')
            if not list_id:
                return PlatformResponse(
                    success=False,
                    platform="mailchimp",
                    action="add_subscriber",
                    response_data={},
                    error="No list_id provided"
                )

            payload = {
                "email_address": email,
                "status": "subscribed"
            }

            if merge_fields:
                payload["merge_fields"] = merge_fields

            if tags:
                payload["tags"] = tags

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/lists/{list_id}/members",
                    headers=self._get_auth_headers(),
                    json=payload
                ) as response:
                    if response.status in [200, 201]:
                        data = await response.json()
                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="add_subscriber",
                            response_data={
                                "id": data.get("id"),
                                "email": email,
                                "status": data.get("status")
                            }
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="add_subscriber",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to add Mailchimp subscriber: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="add_subscriber",
                response_data={},
                error=str(e)
            )

    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        """Pause a scheduled campaign"""
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="pause_campaign",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/campaigns/{campaign_id}/actions/unschedule",
                    headers=self._get_auth_headers()
                ) as response:
                    if response.status == 204:
                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="pause_campaign",
                            response_data={"campaign_id": campaign_id, "status": "paused"}
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="pause_campaign",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to pause Mailchimp campaign: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="pause_campaign",
                response_data={},
                error=str(e)
            )

    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        """Resume/reschedule a paused campaign - returns to draft state"""
        # Note: Mailchimp doesn't have a direct "resume" - unscheduled campaigns
        # go back to draft state and need to be scheduled again
        return PlatformResponse(
            success=False,
            platform="mailchimp",
            action="resume_campaign",
            response_data={"campaign_id": campaign_id},
            error="Use schedule_campaign to reschedule after pausing"
        )

    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any]
    ) -> PlatformResponse:
        """Update campaign settings"""
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="update_campaign",
                response_data={},
                error="Mailchimp not configured"
            )

        try:
            await self._check_rate_limit()

            payload = {"settings": {}}

            if "subject" in updates:
                payload["settings"]["subject_line"] = updates["subject"]
            if "title" in updates:
                payload["settings"]["title"] = updates["title"]
            if "from_name" in updates:
                payload["settings"]["from_name"] = updates["from_name"]

            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.base_url}/campaigns/{campaign_id}",
                    headers=self._get_auth_headers(),
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return PlatformResponse(
                            success=True,
                            platform="mailchimp",
                            action="update_campaign",
                            response_data={"campaign_id": campaign_id, "updated": True}
                        )
                    else:
                        error = await response.text()
                        return PlatformResponse(
                            success=False,
                            platform="mailchimp",
                            action="update_campaign",
                            response_data={},
                            error=f"Failed: {response.status} - {error}"
                        )

        except Exception as e:
            logger.error(f"Failed to update Mailchimp campaign: {e}")
            return PlatformResponse(
                success=False,
                platform="mailchimp",
                action="update_campaign",
                response_data={},
                error=str(e)
            )
