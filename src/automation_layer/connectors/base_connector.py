import asyncio
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlatformResponse:
    """Standardized response from platform APIs"""
    success: bool
    platform: str = ""
    action: str = ""
    response_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    platform_id: Optional[str] = None
    
    @property
    def data(self) -> Dict[str, Any]:
        """Alias for response_data"""
        return self.response_data or {}

class BaseConnector(ABC):
    """
    Base class for all platform connectors
    Provides common functionality for API interactions
    """
    
    def __init__(self, name: str, base_url: str, rate_limit: int = 100):
        self.name = name
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.request_count = 0
        self.last_reset = datetime.utcnow()
    
    async def check_rate_limit(self) -> bool:
        """
        Check if rate limit is exceeded
        
        Returns:
            Boolean indicating if request is allowed
        """
        current_time = datetime.utcnow()
        time_diff = (current_time - self.last_reset).total_seconds()
        
        if time_diff >= 60:
            self.request_count = 0
            self.last_reset = current_time
        
        if self.request_count >= self.rate_limit:
            logger.warning(f"Rate limit exceeded for {self.name}")
            return False
        
        self.request_count += 1
        return True
    
    async def execute_with_retry(
        self,
        func,
        max_retries: int = 3,
        delay: float = 1.0
    ):
        """
        Execute function with retry logic
        
        Args:
            func: Async function to execute
            max_retries: Maximum retry attempts
            delay: Delay between retries in seconds
        
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if not await self.check_rate_limit():
                    await asyncio.sleep(60)
                
                result = await func()
                return result
            
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {self.name}, "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
        
        logger.error(f"All {max_retries} attempts failed for {self.name}")
        raise last_exception
    
    def _generate_request_id(self, data: Dict[str, Any]) -> str:
        """
        Generate unique request ID for idempotency
        
        Args:
            data: Request data
        
        Returns:
            Unique request ID
        """
        import json
        content = json.dumps(data, sort_keys=True) + str(datetime.utcnow())
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _format_error_response(
        self,
        error: Exception,
        action: str
    ) -> PlatformResponse:
        """
        Format error as PlatformResponse
        
        Args:
            error: Exception that occurred
            action: Action that was being performed
        
        Returns:
            PlatformResponse with error details
        """
        return PlatformResponse(
            success=False,
            platform=self.name,
            action=action,
            error=str(error)
        )
    
    def _format_success_response(
        self,
        data: Dict[str, Any],
        action: str,
        platform_id: Optional[str] = None
    ) -> PlatformResponse:
        """
        Format success as PlatformResponse
        
        Args:
            data: Response data
            action: Action that was performed
            platform_id: Platform-specific ID
        
        Returns:
            PlatformResponse with success data
        """
        return PlatformResponse(
            success=True,
            platform=self.name,
            action=action,
            response_data=data,
            platform_id=platform_id
        )
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validate API credentials
        
        Returns:
            Boolean indicating if credentials are valid
        """
        raise NotImplementedError("Subclass must implement validate_credentials")
    
    @abstractmethod
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        """
        Create a new campaign
        
        Args:
            campaign_data: Campaign configuration
        
        Returns:
            PlatformResponse with campaign details
        """
        raise NotImplementedError("Subclass must implement create_campaign")
    
    @abstractmethod
    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        """
        Update existing campaign
        
        Args:
            campaign_id: Platform campaign ID
            updates: Updates to apply
        
        Returns:
            PlatformResponse with updated campaign
        """
        raise NotImplementedError("Subclass must implement update_campaign")
    
    @abstractmethod
    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        """
        Get campaign performance metrics
        
        Args:
            campaign_id: Platform campaign ID
        
        Returns:
            PlatformResponse with metrics
        """
        raise NotImplementedError("Subclass must implement get_campaign_metrics")
    
    @abstractmethod
    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        """
        Pause running campaign
        
        Args:
            campaign_id: Platform campaign ID
        
        Returns:
            PlatformResponse
        """
        raise NotImplementedError("Subclass must implement pause_campaign")
    
    @abstractmethod
    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        """
        Resume paused campaign
        
        Args:
            campaign_id: Platform campaign ID
        
        Returns:
            PlatformResponse
        """
        raise NotImplementedError("Subclass must implement resume_campaign")