"""
Token usage tracking for LLM calls
"""
from typing import Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TokenTracker:

    
    # Approximate costs per 1K tokens (as of 2024)
    TOKEN_COSTS = {
        "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
    }
    
    def __init__(self):
        self.usage_history = []
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def track_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track token usage and calculate cost."""
        costs = self.TOKEN_COSTS.get(model, {"prompt": 0.01, "completion": 0.03})
        
        prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
        completion_cost = (completion_tokens / 1000) * costs["completion"]
        total_cost = prompt_cost + completion_cost
        
        usage = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
            "metadata": metadata or {}
        }
        
        self.usage_history.append(usage)
        self.total_tokens += usage["total_tokens"]
        self.total_cost += total_cost
        
        if total_cost > 1.0:
            logger.warning(f"High cost API call: €{total_cost:.2f} for {model}")
        
        return usage
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.usage_history:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
        
        by_model = {}
        for usage in self.usage_history:
            model = usage["model"]
            if model not in by_model:
                by_model[model] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            
            by_model[model]["calls"] += 1
            by_model[model]["tokens"] += usage["total_tokens"]
            by_model[model]["cost"] += usage["total_cost"]
        
        return {
            "total_calls": len(self.usage_history),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_cost_per_call": self.total_cost / len(self.usage_history),
            "by_model": by_model,
            "recent_calls": self.usage_history[-10:]  # Last 10 calls
        }
