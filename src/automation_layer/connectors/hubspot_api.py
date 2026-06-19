"""HubSpot CRM connector for full-funnel attribution."""
import httpx
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_connector import BaseConnector, PlatformResponse
from ...config.settings import settings

logger = logging.getLogger(__name__)

class HubSpotAPIConnector(BaseConnector):
    """HubSpot CRM API connector for lead funnel tracking and attribution."""
    
    def __init__(self):
        super().__init__(
            name="hubspot",
            base_url="https://api.hubapi.com",
            rate_limit=100
        )
        self.api_key = settings.HUBSPOT_API_KEY
        self.portal_id = settings.HUBSPOT_PORTAL_ID
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def validate_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/account-info/v3/api-usage/daily",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info("HubSpot credentials validated successfully")
                    return True
                else:
                    logger.error(f"HubSpot validation failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Failed to validate HubSpot credentials: {e}")
            return False
    
    async def create_contact(
        self,
        email: str,
        properties: Dict[str, Any]
    ) -> PlatformResponse:
        try:
            contact_props = {
                "email": email,
                **properties
            }
            
            payload = {
                "properties": contact_props
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/crm/v3/objects/contacts",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 201:
                    contact = data
                    logger.info(f"Created HubSpot contact: {email}")
                    
                    return PlatformResponse(
                        success=True,
                        data={"contact": contact, "action": "created"},
                        message="Contact created successfully",
                        platform_id=contact.get("id")
                    )
                elif response.status_code == 409:
                    return await self.update_contact_by_email(email, properties)
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to create contact: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to create contact: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def update_contact_by_email(
        self,
        email: str,
        properties: Dict[str, Any]
    ) -> PlatformResponse:
        try:
            # First, search for contact by email
            search_response = await self.search_contact_by_email(email)
            
            if not search_response.success:
                return search_response
            
            results = search_response.data.get("results", [])
            if not results:
                return PlatformResponse(
                    success=False,
                    error="Contact not found"
                )
            
            contact_id = results[0].get("id")
            
            payload = {
                "properties": properties
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.base_url}/crm/v3/objects/contacts/{contact_id}",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 200:
                    logger.info(f"Updated HubSpot contact: {email}")
                    
                    return PlatformResponse(
                        success=True,
                        data={"contact": data, "action": "updated"},
                        message="Contact updated successfully",
                        platform_id=contact_id
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to update contact: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to update contact: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def search_contact_by_email(self, email: str) -> PlatformResponse:
        try:
            payload = {
                "filterGroups": [
                    {
                        "filters": [
                            {
                                "propertyName": "email",
                                "operator": "EQ",
                                "value": email
                            }
                        ]
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/crm/v3/objects/contacts/search",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 200:
                    return PlatformResponse(
                        success=True,
                        data=data,
                        message="Search completed"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Search failed: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to search contact: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def create_deal(
        self,
        contact_id: str,
        deal_name: str,
        amount: float,
        stage: str = "appointmentscheduled",
        properties: Optional[Dict[str, Any]] = None
    ) -> PlatformResponse:
        try:
            deal_props = {
                "dealname": deal_name,
                "amount": str(amount),
                "dealstage": stage,
                "pipeline": "default"
            }
            
            if properties:
                deal_props.update(properties)
            
            payload = {
                "properties": deal_props
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/crm/v3/objects/deals",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 201:
                    deal_id = data.get("id")
                    
                    await self.associate_contact_to_deal(contact_id, deal_id)
                    
                    logger.info(f"Created HubSpot deal: {deal_id}")
                    
                    return PlatformResponse(
                        success=True,
                        data={"deal": data},
                        message="Deal created successfully",
                        platform_id=deal_id
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to create deal: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to create deal: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def associate_contact_to_deal(
        self,
        contact_id: str,
        deal_id: str
    ) -> PlatformResponse:
        try:
            payload = {
                "inputs": [
                    {
                        "from": {
                            "id": contact_id
                        },
                        "to": {
                            "id": deal_id
                        },
                        "type": "deal_to_contact"
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/crm/v3/associations/contacts/deals/batch/create",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                if response.status_code == 201:
                    logger.info(f"Associated contact {contact_id} with deal {deal_id}")
                    
                    return PlatformResponse(
                        success=True,
                        message="Association created"
                    )
                else:
                    data = response.json()
                    return PlatformResponse(
                        success=False,
                        error=f"Association failed: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to associate contact to deal: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def track_campaign_conversion(
        self,
        campaign_id: str,
        lead_email: str,
        lead_data: Dict[str, Any]
    ) -> PlatformResponse:
        try:
            properties = {
                "firstname": lead_data.get("first_name", ""),
                "lastname": lead_data.get("last_name", ""),
                "company": lead_data.get("company", ""),
                "phone": lead_data.get("phone", ""),
                "jobtitle": lead_data.get("job_title", ""),
                "hs_lead_status": "NEW",
                "agentic_campaign_id": campaign_id,
                "agentic_conversion_date": datetime.utcnow().isoformat()
            }
            
            contact_response = await self.create_contact(lead_email, properties)
            
            if not contact_response.success:
                return contact_response
            
            contact_id = contact_response.platform_id
            
            if lead_data.get("estimated_value"):
                deal_name = f"Agentic Lead - {lead_email}"
                await self.create_deal(
                    contact_id,
                    deal_name,
                    lead_data["estimated_value"],
                    stage="appointmentscheduled"
                )
            
            logger.info(f"Tracked campaign {campaign_id} conversion for {lead_email} in HubSpot")
            
            return PlatformResponse(
                success=True,
                data={
                    "contact_id": contact_id,
                    "campaign_id": campaign_id,
                    "email": lead_email
                },
                message="Conversion tracked in HubSpot"
            )
        except Exception as e:
            logger.error(f"Failed to track conversion: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_contact_lifecycle_stage(
        self,
        email: str
    ) -> Optional[str]:
        try:
            search_response = await self.search_contact_by_email(email)
            
            if not search_response.success:
                return None
            
            results = search_response.data.get("results", [])
            if not results:
                return None
            
            properties = results[0].get("properties", {})
            return properties.get("lifecyclestage")
        except Exception as e:
            logger.error(f"Failed to get lifecycle stage: {e}")
            return None
    
    async def update_deal_stage(
        self,
        deal_id: str,
        new_stage: str
    ) -> PlatformResponse:
        try:
            payload = {
                "properties": {
                    "dealstage": new_stage
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.base_url}/crm/v3/objects/deals/{deal_id}",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 200:
                    logger.info(f"Updated deal {deal_id} stage to {new_stage}")
                    
                    return PlatformResponse(
                        success=True,
                        data={"deal": data},
                        message="Deal stage updated"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to update deal: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to update deal stage: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_deal_by_id(self, deal_id: int) -> PlatformResponse:
        """Per Research Plan - connects HubSpot deal pipeline to DelayedReward status."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/crm/v3/objects/deals/{deal_id}",
                    headers=self._get_headers(),
                    params={"properties": "dealname,amount,dealstage,closedate"},
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to get deal: {response.status_code}",
                        status_code=response.status_code
                    )
                
                deal_data = response.json()
                deal_props = deal_data.get("properties", {})
                
                assoc_response = await client.get(
                    f"{self.base_url}/crm/v3/objects/deals/{deal_id}/associations/contacts",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                
                contact_email = None
                if assoc_response.status_code == 200:
                    associations = assoc_response.json().get("results", [])
                    if associations:
                        contact_id = associations[0].get("id")
                        contact_response = await client.get(
                            f"{self.base_url}/crm/v3/objects/contacts/{contact_id}",
                            headers=self._get_headers(),
                            params={"properties": "email,firstname,lastname"},
                            timeout=10.0
                        )
                        if contact_response.status_code == 200:
                            contact_data = contact_response.json()
                            contact_email = contact_data.get("properties", {}).get("email")
                
                logger.info(f"Retrieved deal {deal_id} with contact email: {contact_email}")
                
                return PlatformResponse(
                    success=True,
                    data={
                        "deal_id": deal_id,
                        "dealname": deal_props.get("dealname"),
                        "amount": float(deal_props.get("amount", 0) or 0),
                        "dealstage": deal_props.get("dealstage"),
                        "closedate": deal_props.get("closedate"),
                        "contact_email": contact_email
                    },
                    message="Deal retrieved successfully"
                )
                
        except Exception as e:
            logger.error(f"Failed to get deal by ID: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_deals(
        self,
        limit: int = 100,
        properties: Optional[List[str]] = None
    ) -> PlatformResponse:
        try:
            if properties is None:
                properties = [
                    "dealname", "amount", "dealstage", "pipeline",
                    "closedate", "createdate", "hs_lastmodifieddate"
                ]
            
            params = {
                "limit": min(limit, 100),
                "properties": ",".join(properties)
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/crm/v3/objects/deals",
                    headers=self._get_headers(),
                    params=params,
                    timeout=15.0
                )
                
                data = response.json()
                
                if response.status_code == 200:
                    deals = data.get("results", [])
                    
                    by_stage = {}
                    for deal in deals:
                        props = deal.get("properties", {})
                        stage = props.get("dealstage", "unknown")
                        if stage not in by_stage:
                            by_stage[stage] = 0
                        by_stage[stage] += 1
                    
                    logger.info(f"Retrieved {len(deals)} deals from HubSpot")
                    
                    return PlatformResponse(
                        success=True,
                        data={
                            "deals": deals,
                            "by_stage": by_stage,
                            "total": len(deals)
                        },
                        message=f"Retrieved {len(deals)} deals"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to get deals: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to get deals: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_lifecycle_stages(self) -> PlatformResponse:
        try:
            stages = [
                "subscriber", "lead", "marketingqualifiedlead",
                "salesqualifiedlead", "opportunity", "customer", "evangelist", "other"
            ]
            
            stage_counts = {}
            
            async with httpx.AsyncClient() as client:
                for stage in stages:
                    payload = {
                        "filterGroups": [
                            {
                                "filters": [
                                    {
                                        "propertyName": "lifecyclestage",
                                        "operator": "EQ",
                                        "value": stage
                                    }
                                ]
                            }
                        ],
                        "limit": 1
                    }
                    
                    response = await client.post(
                        f"{self.base_url}/crm/v3/objects/contacts/search",
                        headers=self._get_headers(),
                        json=payload,
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        stage_counts[stage] = data.get("total", 0)
                    else:
                        stage_counts[stage] = 0
            
            logger.info(f"Retrieved lifecycle stages: {stage_counts}")
            
            return PlatformResponse(
                success=True,
                data={"stages": stage_counts},
                message="Retrieved lifecycle stage counts"
            )
        except Exception as e:
            logger.error(f"Failed to get lifecycle stages: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_lead_quality_scores(
        self,
        limit: int = 100
    ) -> PlatformResponse:
        try:
            params = {
                "limit": min(limit, 100),
                "properties": "email,hubspotscore,hs_lead_status,lifecyclestage"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/crm/v3/objects/contacts",
                    headers=self._get_headers(),
                    params=params,
                    timeout=15.0
                )
                
                data = response.json()
                
                if response.status_code == 200:
                    contacts = data.get("results", [])
                    
                    scores = []
                    distribution = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
                    
                    for contact in contacts:
                        props = contact.get("properties", {})
                        score_str = props.get("hubspotscore", "0")
                        try:
                            score = int(float(score_str)) if score_str else 0
                        except (ValueError, TypeError):
                            score = 0
                        
                        scores.append(score)
                        
                        if score <= 20:
                            distribution["0-20"] += 1
                        elif score <= 40:
                            distribution["21-40"] += 1
                        elif score <= 60:
                            distribution["41-60"] += 1
                        elif score <= 80:
                            distribution["61-80"] += 1
                        else:
                            distribution["81-100"] += 1
                    
                    avg_score = sum(scores) / len(scores) if scores else 0
                    
                    logger.info(f"Analyzed {len(contacts)} contacts for lead quality")
                    
                    return PlatformResponse(
                        success=True,
                        data={
                            "avg_score": round(avg_score, 1),
                            "distribution": distribution,
                            "total_contacts": len(contacts)
                        },
                        message=f"Analyzed {len(contacts)} contacts"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to get contacts: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to get lead quality scores: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_pipelines(self) -> PlatformResponse:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/crm/v3/pipelines/deals",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 200:
                    pipelines = data.get("results", [])
                    
                    formatted = [
                        {
                            "id": p.get("id"),
                            "label": p.get("label"),
                            "stages": [
                                {"id": s.get("id"), "label": s.get("label")}
                                for s in p.get("stages", [])
                            ]
                        }
                        for p in pipelines
                    ]
                    
                    return PlatformResponse(
                        success=True,
                        data={"pipelines": formatted},
                        message=f"Retrieved {len(pipelines)} pipelines"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to get pipelines: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to get pipelines: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def count_contacts(self) -> PlatformResponse:
        try:
            payload = {
                "filterGroups": [],
                "limit": 1
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/crm/v3/objects/contacts/search",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 200:
                    total = data.get("total", 0)
                    
                    return PlatformResponse(
                        success=True,
                        data={"count": total},
                        message=f"Total contacts: {total}"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to count contacts: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to count contacts: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    # Required abstract method implementations
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign creation not supported for CRM platform"
        )
    
    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign updates not supported for CRM platform"
        )
    
    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign metrics not supported for CRM platform"
        )
    
    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign pause not supported for CRM platform"
        )
    
    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign resume not supported for CRM platform"
        )