"""SendGrid email marketing connector."""
try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType
except ImportError:
    sendgrid = None
    Mail = Email = To = Content = Attachment = FileContent = FileName = FileType = None
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
import base64

from .base_connector import BaseConnector, PlatformResponse
from ...config.settings import settings

logger = logging.getLogger(__name__)

class EmailConnector(BaseConnector):
    """SendGrid email marketing connector"""
    
    def __init__(self):
        if sendgrid is None:
            logger.warning("SendGrid package not installed. Email functions will fail.")
            self.sg = None
            return

        config = {
            'api_key': settings.SENDGRID_API_KEY,
            'rate_limit': 100,  # SendGrid rate limits vary by plan
            'from_email': settings.get('FROM_EMAIL', 'marketing@example.com'),
            'from_name': settings.get('FROM_NAME', 'Agentic AI')
        }
        super().__init__(config)
        
        self.sg = sendgrid.SendGridAPIClient(api_key=config['api_key'])
        self.base_url = "https://api.sendgrid.com/v3"
    
    async def validate_credentials(self) -> bool:
        try:
            response = self.sg.client.api_keys.get()
            
            if response.status_code == 200:
                logger.info("SendGrid credentials validated successfully")
                return True
            else:
                logger.error(f"SendGrid credential validation failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"SendGrid credential validation error: {e}")
            return False
    
    async def send_email(self, email_data: Dict[str, Any]) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            message = Mail(
                from_email=(self.config['from_email'], self.config['from_name']),
                to_emails=email_data.get('to_email'),
                subject=email_data.get('subject'),
                html_content=email_data.get('html_content')
            )
            
            if email_data.get('plain_content'):
                message.plain_text_content = email_data['plain_content']
            
            if email_data.get('attachments'):
                for attachment_data in email_data['attachments']:
                    attachment = Attachment()
                    attachment.file_content = FileContent(attachment_data['content'])
                    attachment.file_type = FileType(attachment_data['type'])
                    attachment.file_name = FileName(attachment_data['name'])
                    message.add_attachment(attachment)
            
            message.tracking_settings = {
                'click_tracking': {'enable': True},
                'open_tracking': {'enable': True},
                'subscription_tracking': {'enable': False}
            }
            
            if email_data.get('campaign_id'):
                message.custom_arg = {
                    'campaign_id': str(email_data['campaign_id']),
                    'persona': email_data.get('persona', 'unknown')
                }
            
            response = self.sg.send(message)
            
            return PlatformResponse(
                success=response.status_code in [200, 202],
                platform="email",
                action="send_email",
                response_data={
                    'message_id': response.headers.get('X-Message-Id'),
                    'status_code': response.status_code
                },
                error=None if response.status_code in [200, 202] else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return PlatformResponse(
                success=False,
                platform="email",
                action="send_email",
                response_data={},
                error=str(e)
            )
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            campaign_payload = {
                'title': campaign_data.get('name'),
                'subject': campaign_data.get('subject'),
                'sender_id': await self._get_or_create_sender(),
                'list_ids': campaign_data.get('list_ids', []),
                'suppression_group_id': campaign_data.get('suppression_group_id'),
                'custom_unsubscribe_url': campaign_data.get('unsubscribe_url'),
                'html_content': campaign_data.get('html_content'),
                'plain_content': campaign_data.get('plain_content', ''),
                'categories': campaign_data.get('categories', ['marketing'])
            }
            
            response = self.sg.client.campaigns.post(request_body=campaign_payload)
            
            if response.status_code == 201:
                campaign_id = json.loads(response.body)['id']
                
                return PlatformResponse(
                    success=True,
                    platform="email",
                    action="create_campaign",
                    response_data={
                        'campaign_id': campaign_id,
                        'status': 'created'
                    }
                )
            else:
                return PlatformResponse(
                    success=False,
                    platform="email",
                    action="create_campaign",
                    response_data=json.loads(response.body),
                    error=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Email campaign creation failed: {e}")
            return PlatformResponse(
                success=False,
                platform="email",
                action="create_campaign",
                response_data={},
                error=str(e)
            )
    
    async def _get_or_create_sender(self) -> int:
        try:
            response = self.sg.client.senders.get()
            
            if response.status_code == 200:
                senders = json.loads(response.body)
                if senders:
                    return senders[0]['id']
            
            sender_data = {
                'nickname': 'Agentic Marketing',
                'from': {
                    'email': self.config['from_email'],
                    'name': self.config['from_name']
                },
                'reply_to': {
                    'email': self.config['from_email'],
                    'name': self.config['from_name']
                },
                'address': '123 AI Street',
                'city': 'San Francisco',
                'state': 'CA',
                'zip': '94105',
                'country': 'USA'
            }
            
            response = self.sg.client.senders.post(request_body=sender_data)
            
            if response.status_code == 201:
                return json.loads(response.body)['id']
            
            return 1
            
        except Exception as e:
            logger.error(f"Sender management failed: {e}")
            return 1
    
    async def send_bulk_emails(self, recipients: List[Dict[str, Any]], template_data: Dict[str, Any]) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            personalizations = []
            for recipient in recipients[:1000]:  # SendGrid: max 1000 per request
                personalizations.append({
                    'to': [{'email': recipient['email'], 'name': recipient.get('name', '')}],
                    'substitutions': recipient.get('substitutions', {}),
                    'custom_args': {
                        'recipient_id': str(recipient.get('id', '')),
                        'persona': recipient.get('persona', '')
                    }
                })
            
            message = {
                'personalizations': personalizations,
                'from': {
                    'email': self.config['from_email'],
                    'name': self.config['from_name']
                },
                'subject': template_data.get('subject'),
                'content': [
                    {
                        'type': 'text/html',
                        'value': template_data.get('html_content')
                    }
                ],
                'tracking_settings': {
                    'click_tracking': {'enable': True},
                    'open_tracking': {'enable': True}
                },
                'batch_id': self._generate_batch_id()
            }
            
            response = self.sg.client.mail.send.post(request_body=message)
            
            return PlatformResponse(
                success=response.status_code == 202,
                platform="email",
                action="send_bulk",
                response_data={
                    'batch_id': message['batch_id'],
                    'recipients_count': len(personalizations),
                    'status_code': response.status_code
                },
                error=None if response.status_code == 202 else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            logger.error(f"Bulk email send failed: {e}")
            return PlatformResponse(
                success=False,
                platform="email",
                action="send_bulk",
                response_data={},
                error=str(e)
            )
    
    def _generate_batch_id(self) -> str:
        import uuid
        return str(uuid.uuid4())
    
    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            response = self.sg.client.campaigns._(campaign_id).stats.get()
            
            if response.status_code == 200:
                stats = json.loads(response.body)
                
                metrics = {
                    'sent': sum(s.get('stats', [{}])[0].get('metrics', {}).get('requests', 0) for s in stats),
                    'delivered': sum(s.get('stats', [{}])[0].get('metrics', {}).get('delivered', 0) for s in stats),
                    'opens': sum(s.get('stats', [{}])[0].get('metrics', {}).get('opens', 0) for s in stats),
                    'unique_opens': sum(s.get('stats', [{}])[0].get('metrics', {}).get('unique_opens', 0) for s in stats),
                    'clicks': sum(s.get('stats', [{}])[0].get('metrics', {}).get('clicks', 0) for s in stats),
                    'unique_clicks': sum(s.get('stats', [{}])[0].get('metrics', {}).get('unique_clicks', 0) for s in stats),
                    'bounces': sum(s.get('stats', [{}])[0].get('metrics', {}).get('bounces', 0) for s in stats),
                    'spam_reports': sum(s.get('stats', [{}])[0].get('metrics', {}).get('spam_reports', 0) for s in stats),
                    'unsubscribes': sum(s.get('stats', [{}])[0].get('metrics', {}).get('unsubscribes', 0) for s in stats)
                }
                
                if metrics['sent'] > 0:
                    metrics['open_rate'] = (metrics['unique_opens'] / metrics['sent']) * 100
                    metrics['click_rate'] = (metrics['unique_clicks'] / metrics['sent']) * 100
                    metrics['bounce_rate'] = (metrics['bounces'] / metrics['sent']) * 100
                else:
                    metrics['open_rate'] = 0
                    metrics['click_rate'] = 0
                    metrics['bounce_rate'] = 0
                
                return PlatformResponse(
                    success=True,
                    platform="email",
                    action="get_metrics",
                    response_data=metrics
                )
            else:
                return PlatformResponse(
                    success=False,
                    platform="email",
                    action="get_metrics",
                    response_data={},
                    error=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Failed to get email metrics: {e}")
            return PlatformResponse(
                success=False,
                platform="email",
                action="get_metrics",
                response_data={},
                error=str(e)
            )
    
    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            response = self.sg.client.campaigns._(campaign_id).patch(request_body=updates)
            
            return PlatformResponse(
                success=response.status_code == 200,
                platform="email",
                action="update_campaign",
                response_data={'campaign_id': campaign_id, 'updates': updates},
                error=None if response.status_code == 200 else f"Status: {response.status_code}"
            )
            
        except Exception as e:
            logger.error(f"Email campaign update failed: {e}")
            return PlatformResponse(
                success=False,
                platform="email",
                action="update_campaign",
                response_data={},
                error=str(e)
            )
    
    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        return await self.update_campaign(campaign_id, {'status': 'Paused'})
    
    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        return await self.update_campaign(campaign_id, {'status': 'Scheduled'})
    
    async def get_engagement_data(self, start_date: str, end_date: str) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            params = {
                'start_date': start_date,
                'end_date': end_date,
                'aggregated_by': 'day'
            }
            
            response = self.sg.client.stats.get(query_params=params)
            
            if response.status_code == 200:
                return PlatformResponse(
                    success=True,
                    platform="email",
                    action="get_engagement",
                    response_data=json.loads(response.body)
                )
            else:
                return PlatformResponse(
                    success=False,
                    platform="email",
                    action="get_engagement",
                    response_data={},
                    error=f"Failed with status {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Failed to get engagement data: {e}")
            return PlatformResponse(
                success=False,
                platform="email",
                action="get_engagement",
                response_data={},
                error=str(e)
            )
