"""
Multi-Touch Attribution Models
Research Plan Section 1.2 (Enhancement)

This module implements various multi-touch attribution models beyond
simple first/last touch attribution. These models provide more nuanced
credit distribution across multiple touchpoints in the customer journey.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttributionModel(Enum):
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    U_SHAPED = "u_shaped"
    W_SHAPED = "w_shaped"
    POSITION_BASED = "position_based"


@dataclass
class Touchpoint:
    touchpoint_id: str
    campaign_id: str
    channel: str
    timestamp: datetime
    action_type: str  # impression, click, engagement
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "touchpoint_id": self.touchpoint_id,
            "campaign_id": self.campaign_id,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "action_type": self.action_type,
            "metadata": self.metadata
        }


@dataclass
class AttributionResult:
    conversion_id: str
    conversion_value: float
    model_type: AttributionModel
    touchpoint_credits: Dict[str, float]  # touchpoint_id -> credit
    campaign_credits: Dict[str, float]  # campaign_id -> credit
    channel_credits: Dict[str, float]  # channel -> credit
    total_touchpoints: int


class MultiTouchAttributionEngine:
    """
    Engine for calculating multi-touch attribution across various models.
    
    This implements multiple attribution strategies to provide a more
    accurate picture of campaign contribution to conversions.
    
    Research Plan Reference:
    Section 1.2: "metrics tied directly to the Agentic sales funnel"
    Section 10.2: Full-funnel tracking and attribution
    """
    
    def __init__(self, decay_half_life_days: float = 7.0):
        self.decay_half_life_days = decay_half_life_days
        logger.info(f"MultiTouchAttributionEngine initialized: decay_half_life={decay_half_life_days} days")
    
    def attribute(
        self,
        touchpoints: List[Touchpoint],
        conversion_value: float,
        conversion_id: str,
        model: AttributionModel = AttributionModel.LINEAR
    ) -> AttributionResult:
        if not touchpoints:
            logger.warning(f"No touchpoints for conversion {conversion_id}")
            return AttributionResult(
                conversion_id=conversion_id,
                conversion_value=conversion_value,
                model_type=model,
                touchpoint_credits={},
                campaign_credits={},
                channel_credits={},
                total_touchpoints=0
            )
        
        sorted_touchpoints = sorted(touchpoints, key=lambda t: t.timestamp)
        
        if model == AttributionModel.FIRST_TOUCH:
            credits = self._first_touch(sorted_touchpoints)
        elif model == AttributionModel.LAST_TOUCH:
            credits = self._last_touch(sorted_touchpoints)
        elif model == AttributionModel.LINEAR:
            credits = self._linear(sorted_touchpoints)
        elif model == AttributionModel.TIME_DECAY:
            credits = self._time_decay(sorted_touchpoints)
        elif model == AttributionModel.U_SHAPED:
            credits = self._u_shaped(sorted_touchpoints)
        elif model == AttributionModel.W_SHAPED:
            credits = self._w_shaped(sorted_touchpoints)
        elif model == AttributionModel.POSITION_BASED:
            credits = self._position_based(sorted_touchpoints)
        else:
            credits = self._linear(sorted_touchpoints)
        
        campaign_credits = self._aggregate_by_campaign(sorted_touchpoints, credits, conversion_value)
        channel_credits = self._aggregate_by_channel(sorted_touchpoints, credits, conversion_value)
        touchpoint_credits = {tp.touchpoint_id: credits[i] * conversion_value 
                            for i, tp in enumerate(sorted_touchpoints)}
        
        return AttributionResult(
            conversion_id=conversion_id,
            conversion_value=conversion_value,
            model_type=model,
            touchpoint_credits=touchpoint_credits,
            campaign_credits=campaign_credits,
            channel_credits=channel_credits,
            total_touchpoints=len(sorted_touchpoints)
        )
    
    def _first_touch(self, touchpoints: List[Touchpoint]) -> List[float]:
        credits = [0.0] * len(touchpoints)
        credits[0] = 1.0
        return credits
    
    def _last_touch(self, touchpoints: List[Touchpoint]) -> List[float]:
        credits = [0.0] * len(touchpoints)
        credits[-1] = 1.0
        return credits
    
    def _linear(self, touchpoints: List[Touchpoint]) -> List[float]:
        n = len(touchpoints)
        return [1.0 / n] * n
    
    def _time_decay(self, touchpoints: List[Touchpoint]) -> List[float]:
        """
        Exponential decay - recent touchpoints get more credit.
        Credit = 2^(-(time_to_conversion / half_life))
        """
        if len(touchpoints) == 0:
            return []
        
        conversion_time = touchpoints[-1].timestamp
        half_life_seconds = self.decay_half_life_days * 24 * 3600
        
        raw_credits = []
        for tp in touchpoints:
            time_diff = (conversion_time - tp.timestamp).total_seconds()
            credit = 2 ** (-time_diff / half_life_seconds) if half_life_seconds > 0 else 1.0
            raw_credits.append(credit)
        
        total = sum(raw_credits)
        return [c / total for c in raw_credits] if total > 0 else [1.0 / len(touchpoints)] * len(touchpoints)
    
    def _u_shaped(self, touchpoints: List[Touchpoint]) -> List[float]:
        """
        U-shaped / Position-based (40-20-40)
        First and last touchpoints get 40% each, middle shares 20%
        """
        n = len(touchpoints)
        
        if n == 1:
            return [1.0]
        elif n == 2:
            return [0.5, 0.5]
        
        credits = [0.0] * n
        credits[0] = 0.4  # First touch gets 40%
        credits[-1] = 0.4  # Last touch gets 40%
        
        # Middle touchpoints share remaining 20%
        middle_count = n - 2
        if middle_count > 0:
            middle_credit = 0.2 / middle_count
            for i in range(1, n - 1):
                credits[i] = middle_credit
        
        return credits
    
    def _w_shaped(self, touchpoints: List[Touchpoint]) -> List[float]:
        """
        W-shaped (30-10-30-10-20)
        First touch, lead creation, and opportunity creation emphasized
        For simplicity: first, middle, last get higher weights
        """
        n = len(touchpoints)
        
        if n == 1:
            return [1.0]
        elif n == 2:
            return [0.5, 0.5]
        elif n == 3:
            return [0.35, 0.3, 0.35]
        
        credits = [0.0] * n
        credits[0] = 0.30  # First touch
        credits[-1] = 0.30  # Last touch
        
        mid = n // 2
        credits[mid] = 0.30  # Middle touch (lead creation proxy)
        
        remaining = 0.10
        other_count = n - 3
        if other_count > 0:
            other_credit = remaining / other_count
            for i in range(1, n - 1):
                if i != mid:
                    credits[i] = other_credit
        
        return credits
    
    def _position_based(self, touchpoints: List[Touchpoint]) -> List[float]:
        return self._u_shaped(touchpoints)  # Default to U-shaped
    
    def _aggregate_by_campaign(
        self,
        touchpoints: List[Touchpoint],
        credits: List[float],
        conversion_value: float
    ) -> Dict[str, float]:
        campaign_credits: Dict[str, float] = {}
        
        for tp, credit in zip(touchpoints, credits):
            campaign_id = tp.campaign_id
            if campaign_id not in campaign_credits:
                campaign_credits[campaign_id] = 0.0
            campaign_credits[campaign_id] += credit * conversion_value
        
        return campaign_credits
    
    def _aggregate_by_channel(
        self,
        touchpoints: List[Touchpoint],
        credits: List[float],
        conversion_value: float
    ) -> Dict[str, float]:
        channel_credits: Dict[str, float] = {}
        
        for tp, credit in zip(touchpoints, credits):
            channel = tp.channel
            if channel not in channel_credits:
                channel_credits[channel] = 0.0
            channel_credits[channel] += credit * conversion_value
        
        return channel_credits
    
    def compare_models(
        self,
        touchpoints: List[Touchpoint],
        conversion_value: float,
        conversion_id: str
    ) -> Dict[str, AttributionResult]:
        results = {}
        for model in AttributionModel:
            result = self.attribute(
                touchpoints=touchpoints,
                conversion_value=conversion_value,
                conversion_id=conversion_id,
                model=model
            )
            results[model.value] = result
        
        return results
    
    def get_model_summary(self) -> Dict:
        return {
            "available_models": [m.value for m in AttributionModel],
            "default_model": AttributionModel.LINEAR.value,
            "decay_half_life_days": self.decay_half_life_days,
            "model_descriptions": {
                "first_touch": "100% credit to first interaction",
                "last_touch": "100% credit to last interaction before conversion",
                "linear": "Equal credit distributed across all touchpoints",
                "time_decay": "More credit to recent touchpoints (exponential decay)",
                "u_shaped": "40% first, 40% last, 20% middle touchpoints",
                "w_shaped": "30% first, 30% middle, 30% last, 10% others",
                "position_based": "Custom weights based on funnel position"
            }
        }


_attribution_engine_instance: Optional[MultiTouchAttributionEngine] = None


def get_attribution_engine() -> MultiTouchAttributionEngine:
    global _attribution_engine_instance
    if _attribution_engine_instance is None:
        _attribution_engine_instance = MultiTouchAttributionEngine()
    return _attribution_engine_instance
