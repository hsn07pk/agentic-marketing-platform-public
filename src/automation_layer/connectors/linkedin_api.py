"""LinkedIn API connector for B2B marketing campaigns."""
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import json

from .base_connector import BaseConnector, PlatformResponse
from .base_connector import BaseConnector, PlatformResponse
# Removed static settings import
# from ...config.settings import settings

logger = logging.getLogger(__name__)

class LinkedInConnector(BaseConnector):
    """LinkedIn Marketing API connector"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="linkedin",
            base_url="https://api.linkedin.com/v2",
            rate_limit=100  # LinkedIn: 100 calls/day for some endpoints
        )
        if config:
            self.config = config
        else:
            # Fallback for unexpected direct usage
            from ...config.settings import settings
            self.config = {
                'client_id': settings.LINKEDIN_CLIENT_ID,
                'client_secret': settings.LINKEDIN_CLIENT_SECRET,
                'access_token': settings.LINKEDIN_ACCESS_TOKEN,
                'api_version': 'v2',
            }
        
        self.ads_base_url = "https://api.linkedin.com/rest/adCampaigns"
        self.session = None
    
    async def _get_session(self):
        import aiohttp
        if not self.session:
            headers = {
                'Authorization': f"Bearer {self.config['access_token']}",
                'Content-Type': 'application/json',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def validate_credentials(self) -> bool:
        try:
            session = await self._get_session()
            await self.check_rate_limit()
            
            async with session.get(f"{self.base_url}/me") as response:
                if response.status == 200:
                    logger.info("LinkedIn credentials validated successfully")
                    return True
                else:
                    logger.error(f"LinkedIn credential validation failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"LinkedIn credential validation error: {e}")
            return False
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        try:
            session = await self._get_session()
            await self.check_rate_limit()
            
            payload = {
                "account": f"urn:li:sponsoredAccount:{campaign_data.get('account_id')}",
                "name": campaign_data.get('name'),
                "status": "ACTIVE",
                "type": "SPONSORED_CONTENT",
                "objectiveType": "LEAD_GENERATION",
                "dailyBudget": {
                    "amount": str(campaign_data.get('daily_budget', 100)),
                    "currencyCode": "EUR"
                },
                "runSchedule": {
                    "start": int(datetime.utcnow().timestamp() * 1000),
                    "end": int((datetime.utcnow() + timedelta(days=campaign_data.get('duration_days', 30))).timestamp() * 1000)
                },
                "targeting": self._build_targeting(campaign_data.get('targeting', {}))
            }
            
            request_id = self._generate_request_id(payload)
            
            async with session.post(self.ads_base_url, json=payload) as response:
                response_data = await response.json()
                
                if response.status in [200, 201]:
                    return PlatformResponse(
                        success=True,
                        platform="linkedin",
                        action="create_campaign",
                        response_data=response_data,
                        request_id=request_id
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="linkedin",
                        action="create_campaign",
                        response_data=response_data,
                        error=f"Failed with status {response.status}",
                        request_id=request_id
                    )
                    
        except Exception as e:
            logger.error(f"LinkedIn campaign creation failed: {e}")
            return PlatformResponse(
                success=False,
                platform="linkedin",
                action="create_campaign",
                response_data={},
                error=str(e)
            )
    
    def _build_targeting(self, targeting: Dict[str, Any]) -> Dict[str, Any]:
        criteria = {
            "include": {
                "and": []
            }
        }
        
        # Geographic targeting
        if targeting.get('locations'):
            criteria["include"]["and"].append({
                "or": {
                    "urn:li:adTargetingFacet:locations": targeting['locations']
                }
            })
        
        if targeting.get('company_sizes'):
            criteria["include"]["and"].append({
                "or": {
                    "urn:li:adTargetingFacet:staffCountRanges": targeting['company_sizes']
                }
            })
        
        if targeting.get('job_titles'):
            criteria["include"]["and"].append({
                "or": {
                    "urn:li:adTargetingFacet:titles": targeting['job_titles']
                }
            })
        
        if targeting.get('industries'):
            criteria["include"]["and"].append({
                "or": {
                    "urn:li:adTargetingFacet:industries": targeting['industries']
                }
            })
        
        return criteria
    
    async def create_sponsored_content(self, content_data: Dict[str, Any]) -> PlatformResponse:
        try:
            session = await self._get_session()
            await self.check_rate_limit()
            
            payload = {
                "owner": f"urn:li:organization:{content_data.get('organization_id')}",
                "text": {
                    "text": content_data.get('body')
                },
                "shareCommentary": {
                    "text": content_data.get('headline')
                },
                "visibility": "PUBLIC",
                "distribution": {
                    "linkedInDistributionTarget": {}
                },
                "lifecycleState": "PUBLISHED",
                "isReshareDisabledByAuthor": False
            }
            
            if content_data.get('image_url'):
                payload["content"] = {
                    "contentEntities": [{
                        "entityLocation": content_data['image_url']
                    }],
                    "title": content_data.get('headline', ''),
                    "landingPageUrl": content_data.get('landing_url')
                }
            
            async with session.post(f"{self.base_url}/ugcPosts", json=payload) as response:
                response_data = await response.json()
                
                return PlatformResponse(
                    success=response.status in [200, 201],
                    platform="linkedin",
                    action="create_content",
                    response_data=response_data,
                    error=None if response.status in [200, 201] else f"Status: {response.status}"
                )
                
        except Exception as e:
            logger.error(f"LinkedIn content creation failed: {e}")
            return PlatformResponse(
                success=False,
                platform="linkedin",
                action="create_content",
                response_data={},
                error=str(e)
            )
    
    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        try:
            session = await self._get_session()
            await self.check_rate_limit()
            
            params = {
                'q': 'campaign',
                'campaign': f'urn:li:sponsoredCampaign:{campaign_id}',
                'dateRange.start.day': 1,
                'dateRange.start.month': 1,
                'dateRange.start.year': 2025,
                'fields': 'impressions,clicks,costInUsd,leads,conversionValueInLocalCurrency'
            }
            
            async with session.get(f"{self.base_url}/adAnalytics", params=params) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    metrics = self._process_analytics(response_data)
                    
                    return PlatformResponse(
                        success=True,
                        platform="linkedin",
                        action="get_metrics",
                        response_data=metrics
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="linkedin",
                        action="get_metrics",
                        response_data=response_data,
                        error=f"Failed with status {response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get LinkedIn metrics: {e}")
            return PlatformResponse(
                success=False,
                platform="linkedin",
                action="get_metrics",
                response_data={},
                error=str(e)
            )
    
    def _process_analytics(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        elements = raw_data.get('elements', [])
        
        if not elements:
            return {
                'impressions': 0,
                'clicks': 0,
                'conversions': 0,
                'cost': 0.0,
                'ctr': 0.0,
                'cpl': 0.0
            }
        
        total_impressions = sum(e.get('impressions', 0) for e in elements)
        total_clicks = sum(e.get('clicks', 0) for e in elements)
        total_leads = sum(e.get('leads', 0) for e in elements)
        total_cost = sum(e.get('costInUsd', 0) for e in elements)
        
        return {
            'impressions': total_impressions,
            'clicks': total_clicks,
            'conversions': total_leads,
            'cost': total_cost,
            'ctr': (total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
            'cpl': (total_cost / total_leads) if total_leads > 0 else 0
        }
    
    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        try:
            session = await self._get_session()
            await self.check_rate_limit()
            
            payload = {}
            
            if 'status' in updates:
                payload['status'] = updates['status']
            
            if 'daily_budget' in updates:
                payload['dailyBudget'] = {
                    'amount': str(updates['daily_budget']),
                    'currencyCode': 'EUR'
                }
            
            if 'end_date' in updates:
                payload['runSchedule'] = {
                    'end': int(updates['end_date'].timestamp() * 1000)
                }
            
            async with session.patch(f"{self.ads_base_url}/{campaign_id}", json=payload) as response:
                
                return PlatformResponse(
                    success=response.status == 200,
                    platform="linkedin",
                    action="update_campaign",
                    response_data={'campaign_id': campaign_id, 'updates': updates},
                    error=None if response.status == 200 else f"Status: {response.status}"
                )
                
        except Exception as e:
            logger.error(f"LinkedIn campaign update failed: {e}")
            return PlatformResponse(
                success=False,
                platform="linkedin",
                action="update_campaign",
                response_data={},
                error=str(e)
            )
    
    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        return await self.update_campaign(campaign_id, {'status': 'PAUSED'})
    
    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        return await self.update_campaign(campaign_id, {'status': 'ACTIVE'})
    
    async def close(self):
        if self.session:
            await self.session.close()