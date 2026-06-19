"""Mailgun email connector for transactional and marketing emails."""
import httpx
import json
import base64
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from .base_connector import BaseConnector, PlatformResponse
from ...config.settings import settings

logger = logging.getLogger(__name__)


class MailgunConnector(BaseConnector):
    """Mailgun connector for alerts, transactional, and campaign emails."""
    
    def __init__(self):
        api_key = getattr(settings, 'MAILGUN_API_KEY', None)
        domain = getattr(settings, 'MAILGUN_DOMAIN', None)
        region = getattr(settings, 'MAILGUN_REGION', 'eu')
        
        config = {
            'api_key': api_key,
            'domain': domain,
            'region': region,
            'rate_limit': 300,
            'from_email': getattr(settings, 'MAILGUN_FROM_EMAIL', 'alerts@example.com'),
            'from_name': getattr(settings, 'MAILGUN_FROM_NAME', 'Agentic AI')
        }
        self.config = config
        
        self.enabled = bool(api_key and domain)
        # EU domains use api.eu.mailgun.net, US domains use api.mailgun.net
        api_host = "api.eu.mailgun.net" if region == "eu" else "api.mailgun.net"
        self.api_host = api_host
        self.base_url = f"https://{api_host}/v3/{domain}" if domain else None
        
        super().__init__(name="mailgun", base_url=self.base_url or "", rate_limit=300)
        
        if not self.enabled:
            logger.warning("Mailgun connector not configured (MAILGUN_API_KEY or MAILGUN_DOMAIN not set)")
    
    def _get_auth(self) -> tuple:
        return ("api", self.config['api_key'])
    
    async def validate_credentials(self) -> bool:
        if not self.enabled:
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://{self.api_host}/v3/domains/{self.config['domain']}",
                    auth=self._get_auth(),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info("Mailgun credentials validated successfully")
                    return True
                else:
                    logger.error(f"Mailgun validation failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Mailgun credential validation error: {e}")
            return False
    
    async def send_email(self, email_data: Dict[str, Any]) -> PlatformResponse:
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailgun",
                action="send_email",
                response_data={},
                error="Mailgun not configured"
            )
        
        try:
            await self.check_rate_limit()
            
            from_email = email_data.get('from_email') or self.config['from_email']
            from_name = email_data.get('from_name') or self.config['from_name']
            
            data = {
                'from': f"{from_name} <{from_email}>",
                'to': email_data.get('to_email'),
                'subject': email_data.get('subject')
            }
            
            if email_data.get('html_content'):
                data['html'] = email_data['html_content']
            if email_data.get('text_content') or email_data.get('plain_content'):
                data['text'] = email_data.get('text_content') or email_data.get('plain_content')
            
            data['o:tracking'] = 'yes'
            data['o:tracking-clicks'] = 'yes'
            data['o:tracking-opens'] = 'yes'
            
            if email_data.get('campaign_id'):
                data['o:tag'] = [f"campaign:{email_data['campaign_id']}"]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    auth=self._get_auth(),
                    data=data,
                    timeout=15.0
                )
                
                result = response.json()
                
                if response.status_code == 200:
                    logger.info(f"Email sent via Mailgun: {result.get('id')}")
                    return PlatformResponse(
                        success=True,
                        platform="mailgun",
                        action="send_email",
                        response_data={
                            'message_id': result.get('id'),
                            'message': result.get('message')
                        }
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="mailgun",
                        action="send_email",
                        response_data={},
                        error=f"Failed: {result.get('message', response.status_code)}"
                    )
        except Exception as e:
            logger.error(f"Failed to send email via Mailgun: {e}")
            return PlatformResponse(
                success=False,
                platform="mailgun",
                action="send_email",
                response_data={},
                error=str(e)
            )
    
    async def send_alert(
        self,
        to_email: str,
        subject: str,
        message: str,
        alert_type: str = "warning"
    ) -> PlatformResponse:
        """Used for canary rollback alerts, system errors, etc."""
        # Format alert message
        alert_emoji = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "ℹ️",
            "success": "✅"
        }.get(alert_type, "📧")
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
            <h2 style="color: #333;">{alert_emoji} {subject}</h2>
            <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid {'#dc3545' if alert_type == 'critical' else '#ffc107' if alert_type == 'warning' else '#17a2b8'};">
                <pre style="white-space: pre-wrap; font-family: monospace;">{message}</pre>
            </div>
            <p style="color: #666; font-size: 12px; margin-top: 20px;">
                Sent by Agentic AI Platform at {datetime.utcnow().isoformat()}
            </p>
        </div>
        """
        
        return await self.send_email({
            'to_email': to_email,
            'subject': f"{alert_emoji} {subject}",
            'html_content': html_content,
            'text_content': message
        })
    
    async def send_bulk_emails(
        self,
        recipients: List[Dict[str, Any]],
        subject: str,
        html_template: str,
        text_template: str = None
    ) -> PlatformResponse:
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailgun",
                action="send_bulk",
                response_data={},
                error="Mailgun not configured"
            )
        
        try:
            to_list = [r.get('email') for r in recipients]
            recipient_variables = {
                r.get('email'): {k: v for k, v in r.items() if k != 'email'}
                for r in recipients
            }
            
            data = {
                'from': f"{self.config['from_name']} <{self.config['from_email']}>",
                'to': to_list,
                'subject': subject,
                'html': html_template,
                'recipient-variables': json.dumps(recipient_variables),
                'o:tracking': 'yes'
            }
            
            if text_template:
                data['text'] = text_template
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    auth=self._get_auth(),
                    data=data,
                    timeout=30.0
                )
                
                result = response.json()
                
                if response.status_code == 200:
                    logger.info(f"Bulk email sent via Mailgun: {len(recipients)} recipients")
                    return PlatformResponse(
                        success=True,
                        platform="mailgun",
                        action="send_bulk",
                        response_data={
                            'message_id': result.get('id'),
                            'recipients_count': len(recipients)
                        }
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="mailgun",
                        action="send_bulk",
                        response_data={},
                        error=f"Failed: {result.get('message', response.status_code)}"
                    )
        except Exception as e:
            logger.error(f"Failed to send bulk email via Mailgun: {e}")
            return PlatformResponse(
                success=False,
                platform="mailgun",
                action="send_bulk",
                response_data={},
                error=str(e)
            )
    
    async def get_stats(self, days: int = 7) -> PlatformResponse:
        if not self.enabled:
            return PlatformResponse(
                success=False,
                platform="mailgun",
                action="get_stats",
                response_data={},
                error="Mailgun not configured"
            )
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/stats/total",
                    auth=self._get_auth(),
                    params={
                        'event': ['accepted', 'delivered', 'failed', 'opened', 'clicked'],
                        'duration': f'{days}d'
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return PlatformResponse(
                        success=True,
                        platform="mailgun",
                        action="get_stats",
                        response_data=data.get('stats', {})
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="mailgun",
                        action="get_stats",
                        response_data={},
                        error=f"Failed: {response.status_code}"
                    )
        except Exception as e:
            logger.error(f"Failed to get Mailgun stats: {e}")
            return PlatformResponse(
                success=False,
                platform="mailgun",
                action="get_stats",
                response_data={},
                error=str(e)
            )
    
    # Deployer interface methods
    async def create_campaign(self, content_data: Dict[str, Any]) -> PlatformResponse:
        return await self.send_email(content_data)
    
    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=True,
            platform="mailgun",
            action="pause",
            response_data={"message": "Mailgun is transactional - no pause needed"}
        )
    
    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=True,
            platform="mailgun",
            action="resume",
            response_data={"message": "Mailgun is transactional - no resume needed"}
        )
    
    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        return await self.get_stats()

    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        return PlatformResponse(
            success=True,
            platform="mailgun",
            action="update",
            response_data={"message": "Mailgun is transactional - no updates needed"}
        )
