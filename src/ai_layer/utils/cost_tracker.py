"""
Cost Tracking Utility

Tracks LLM API costs and saves to database for the Cost Control dashboard.
Supports both OpenAI and local LLM cost estimation.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    LOCAL = "local"


# Cost rates per 1K tokens (EUR)
COST_RATES = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
    "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
    # Local LLM cost estimates (based on compute/electricity, roughly 1/10 of API)
    "llama3.1": {"prompt": 0.0001, "completion": 0.0002},
    "llama3:8b": {"prompt": 0.0001, "completion": 0.0002},
    "llama3.2": {"prompt": 0.00005, "completion": 0.0001},
    "default_local": {"prompt": 0.0001, "completion": 0.0002},
}


def calculate_cost(
    model: str,
    tokens_prompt: int,
    tokens_completion: int,
    provider: Optional[str] = None
) -> float:
    rates = COST_RATES.get(model)
    
    if rates is None:
        if provider in ["ollama", "local"] or model.startswith("llama"):
            rates = COST_RATES.get("default_local")
        else:
            rates = COST_RATES.get("gpt-3.5-turbo")
    
    prompt_cost = (tokens_prompt / 1000) * rates["prompt"]
    completion_cost = (tokens_completion / 1000) * rates["completion"]
    
    return prompt_cost + completion_cost


async def track_llm_cost(
    agent_type: str,
    model: str,
    tokens_prompt: int,
    tokens_completion: int,
    provider: str = "openai",
    campaign_id: Optional[str] = None,
    content_id: Optional[str] = None,
    action: str = "llm_call"
) -> Dict[str, Any]:
    from ...data_layer.database.connection import async_session_maker
    from ...data_layer.database.models import CostTracking
    
    cost = calculate_cost(model, tokens_prompt, tokens_completion, provider)
    
    cost_record = {
        "model": model,
        "provider": provider,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "tokens_total": tokens_prompt + tokens_completion,
        "cost": cost,
        "agent_type": agent_type,
        "campaign_id": campaign_id,
        "action": action
    }
    
    try:
        async with async_session_maker() as session:
            tracking = CostTracking(
                source_type="llm_api",
                source_id=f"{agent_type}:{action}",
                cost_amount=cost,
                cost_currency="EUR",
                campaign_id=campaign_id if campaign_id else None,
                agent_type=agent_type,
                provider=provider,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion
            )
            session.add(tracking)
            await session.commit()
            
            logger.info(
                f"Tracked LLM cost: €{cost:.6f}",
                extra={
                    "event": "cost_tracked",
                    "agent_type": agent_type,
                    "model": model,
                    "provider": provider,
                    "tokens_total": tokens_prompt + tokens_completion,
                    "cost_eur": cost
                }
            )
            
            cost_record["tracking_id"] = str(tracking.id)
            
    except Exception as e:
        logger.error(f"Failed to track cost: {e}")
        # Don't fail the main operation if cost tracking fails
    
    return cost_record


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token for English text."""
    if not text:
        return 0
    return len(text) // 4


class CostTracker:
    
    def __init__(
        self,
        agent_type: str,
        campaign_id: Optional[str] = None
    ):
        self.agent_type = agent_type
        self.campaign_id = campaign_id
        self.operations: list = []
        self.total_cost: float = 0.0
        self.total_tokens: int = 0
    
    def add_operation(
        self,
        model: str,
        tokens_prompt: int,
        tokens_completion: int,
        provider: str = "openai",
        action: str = "llm_call"
    ):
        cost = calculate_cost(model, tokens_prompt, tokens_completion, provider)
        
        self.operations.append({
            "model": model,
            "provider": provider,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "cost": cost,
            "action": action
        })
        
        self.total_cost += cost
        self.total_tokens += tokens_prompt + tokens_completion
    
    async def save(self) -> Dict[str, Any]:
        for op in self.operations:
            await track_llm_cost(
                agent_type=self.agent_type,
                model=op["model"],
                tokens_prompt=op["tokens_prompt"],
                tokens_completion=op["tokens_completion"],
                provider=op["provider"],
                campaign_id=self.campaign_id,
                action=op["action"]
            )
        
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "operations_count": len(self.operations)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "operations": self.operations
        }
