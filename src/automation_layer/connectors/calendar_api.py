"""Cal.com API connector for demo booking tracking."""
import httpx
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_connector import BaseConnector, PlatformResponse
from ...config.settings import settings

logger = logging.getLogger(__name__)

class CalendarAPIConnector(BaseConnector):
    """
    Cal.com API v2 integration for tracking booked demos.
    Using v1 API - compatible with cal_live_* API keys.
    v2 API requires OAuth which is not yet configured.
    API Reference: https://cal.com/docs/api-reference/v1
    """
    
    def __init__(self):
        super().__init__(
            name="cal.com",
            base_url="https://api.cal.com/v1",
            rate_limit=100  # v1 API rate limits
        )
        self.api_key = settings.CALENDAR_API_KEY
        
        if not self.api_key:
            try:
                from ...data_layer.database.connection import sync_session_maker
                from ...config.configuration_service import ConfigurationService
                
                with sync_session_maker() as db:
                    svc = ConfigurationService(db)
                    self.api_key = svc.get_value("CALENDAR_API_KEY", "")
                    if self.api_key:
                        logger.info("Loaded CALENDAR_API_KEY from database")
            except Exception as e:
                logger.error(f"Failed to load CALENDAR_API_KEY from database: {e}")
    
    def _get_params(self, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """v1 API authenticates via apiKey query param."""
        params = {"apiKey": self.api_key}
        if extra_params:
            params.update(extra_params)
        return params
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def validate_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                # v1 uses /event-types endpoint for validation
                response = await client.get(
                    f"{self.base_url}/event-types",
                    params=self._get_params(),
                    headers=self._get_headers(),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "event_types" in data:
                        logger.info("Cal.com v1 credentials validated successfully")
                        return True
                    logger.error(f"Cal.com v1 validation response error: {data}")
                    return False
                else:
                    err_body = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    err_msg = err_body.get('message') or err_body.get('error') or response.text[:100]
                    logger.error(f"Cal.com v1 validation failed: HTTP {response.status_code} — {err_msg}")
                    return False
        except Exception as e:
            logger.error(f"Failed to validate Cal.com v1 credentials: {e}")
            return False
    
    async def get_event_types(self) -> PlatformResponse:
        """v1 API: GET /event-types?apiKey=..."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/event-types",
                    params=self._get_params(),
                    headers=self._get_headers(),
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 200 and "event_types" in data:
                    event_types = data.get("event_types", [])
                    
                    logger.info(f"Retrieved {len(event_types)} event types from Cal.com v1")
                    
                    return PlatformResponse(
                        success=True,
                        response_data={"event_types": event_types}
                    )
                else:
                    error_msg = data.get('message') or data.get('error') or f"HTTP {response.status_code}"
                    return PlatformResponse(
                        success=False,
                        error=f"Cal.com: {error_msg}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to get event types: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_bookings(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None
    ) -> PlatformResponse:
        """v1 API: GET /bookings?apiKey=... with optional status filter."""
        try:
            # v1 API returns all bookings; date filtering done client-side
            extra_params = {}
            if status:
                extra_params["status"] = status
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/bookings",
                    params=self._get_params(extra_params),
                    headers=self._get_headers(),
                    timeout=15.0
                )
                
                data = response.json()
                
                if response.status_code == 200 and "bookings" in data:
                    bookings = data.get("bookings", [])
                    
                    # Filter by date if specified
                    if start_date or end_date:
                        filtered = []
                        for b in bookings:
                            booking_time = datetime.fromisoformat(b.get("startTime", "").replace("Z", "+00:00"))
                            if start_date and booking_time < start_date.replace(tzinfo=booking_time.tzinfo):
                                continue
                            if end_date and booking_time > end_date.replace(tzinfo=booking_time.tzinfo):
                                continue
                            filtered.append(b)
                        bookings = filtered
                    
                    logger.info(f"Retrieved {len(bookings)} bookings from Cal.com v1")
                    
                    return PlatformResponse(
                        success=True,
                        response_data={"bookings": bookings, "count": len(bookings)}
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to get bookings: {data.get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to get bookings: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_booking_by_id(self, booking_id: str) -> PlatformResponse:
        """v2 API: GET /bookings/{uid}"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/bookings/{booking_id}",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code == 200 and data.get("status") == "success":
                    booking = data.get("data", {})
                    
                    logger.info(f"Retrieved booking {booking_id}")
                    
                    return PlatformResponse(
                        success=True,
                        data={"booking": booking},
                        message="Booking retrieved successfully"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to get booking: {data.get('error', {}).get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to get booking {booking_id}: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def track_conversion_to_booking(
        self,
        campaign_id: str,
        lead_email: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PlatformResponse:
        """Critical delayed-reward signal: tracks when a campaign lead books a demo."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/bookings",
                    headers=self._get_headers(),
                    params={
                        "attendeeEmail": lead_email,
                        "status": "upcoming"
                    },
                    timeout=10.0
                )
                
                data = response.json()
                
                # v2 API returns data in 'data' field
                if response.status_code == 200 and data.get("status") == "success":
                    bookings = data.get("data", [])
                    
                    if bookings:
                        booking = bookings[0]
                        
                        logger.info(
                            f"Tracked conversion for campaign {campaign_id}: "
                            f"{lead_email} booked demo {booking.get('id')}"
                        )
                        
                        return PlatformResponse(
                            success=True,
                            data={
                                "matched": True,
                                "booking": booking,
                                "campaign_id": campaign_id,
                                "lead_email": lead_email,
                                "booking_time": booking.get("startTime"),
                                "event_type": booking.get("eventType", {}).get("title")
                            },
                            message="Conversion tracked to booking"
                        )
                    else:
                        return PlatformResponse(
                            success=True,
                            data={
                                "matched": False,
                                "campaign_id": campaign_id,
                                "lead_email": lead_email
                            },
                            message="No booking found for this lead yet"
                        )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to search bookings: {data.get('error', {}).get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to track conversion: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    async def get_booking_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        try:
            response = await self.get_bookings(start_date, end_date)
            
            if not response.success:
                return {
                    "total_bookings": 0,
                    "upcoming": 0,
                    "completed": 0,
                    "cancelled": 0,
                    "no_show": 0
                }
            
            bookings = response.data.get("bookings", [])
            
            metrics = {
                "total_bookings": len(bookings),
                "upcoming": 0,
                "completed": 0,
                "cancelled": 0,
                "no_show": 0
            }
            
            for booking in bookings:
                status = booking.get("status", "").lower()
                if status in metrics:
                    metrics[status] += 1
            
            if metrics["total_bookings"] > 0:
                metrics["completion_rate"] = (
                    metrics["completed"] / metrics["total_bookings"] * 100
                )
                metrics["no_show_rate"] = (
                    metrics["no_show"] / metrics["total_bookings"] * 100
                )
            else:
                metrics["completion_rate"] = 0.0
                metrics["no_show_rate"] = 0.0
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get booking metrics: {e}")
            return {
                "total_bookings": 0,
                "upcoming": 0,
                "completed": 0,
                "cancelled": 0,
                "no_show": 0
            }
    
    async def create_booking_webhook(
        self,
        webhook_url: str,
        event_types: List[str] = None
    ) -> PlatformResponse:
        """v2 API: POST /webhooks for real-time booking notifications."""
        try:
            if event_types is None:
                event_types = [
                    "BOOKING_CREATED",
                    "BOOKING_RESCHEDULED",
                    "BOOKING_CANCELLED",
                    "MEETING_STARTED",
                    "MEETING_ENDED"
                ]
            
            payload = {
                "subscriberUrl": webhook_url,
                "triggers": event_types,
                "active": True,
                "payloadTemplate": None  # Use default payload
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/webhooks",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=10.0
                )
                
                data = response.json()
                
                if response.status_code in [200, 201] and data.get("status") == "success":
                    webhook = data.get("data", {})
                    
                    logger.info(f"Created Cal.com v2 webhook: {webhook.get('id')}")
                    
                    return PlatformResponse(
                        success=True,
                        data={"webhook": webhook},
                        message="Webhook created successfully"
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        error=f"Failed to create webhook: {data.get('error', {}).get('message', 'Unknown error')}",
                        status_code=response.status_code
                    )
        except Exception as e:
            logger.error(f"Failed to create webhook: {e}")
            return PlatformResponse(
                success=False,
                error=str(e)
            )
    
    # Required abstract method implementations
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign creation not supported for calendar platform"
        )
    
    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign updates not supported for calendar platform"
        )
    
    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign metrics not supported for calendar platform"
        )
    
    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign pause not supported for calendar platform"
        )
    
    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=False,
            error="Campaign resume not supported for calendar platform"
        )