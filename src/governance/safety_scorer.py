"""
Safety scoring implementation
"""
from typing import Dict, List, Optional, Any
import logging

from ..ai_layer.agents.safety_validator import SafetyValidatorAgent

logger = logging.getLogger(__name__)

class SafetyScorer:
    """
    Wrapper for safety scoring functionality
    """
    
    def __init__(self):
        self.validator = SafetyValidatorAgent()
    
    async def validate(
        self,
        content: str,
        headline: Optional[str] = None,
        claims_used: Optional[List[str]] = None,
        platform: str = "general"
    ) -> Dict[str, Any]:
        """
        Validate content safety
        
        Args:
            content: Content to validate
            headline: Optional headline
            claims_used: Claims used in content
            platform: Target platform
        
        Returns:
            Validation results
        """
        return await self.validator.validate_content(
            content_text=content,
            headline=headline,
            claims_used=claims_used,
            platform=platform
        )