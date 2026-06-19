from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
from pydantic import BaseModel
import logging

from ...ai_layer.agents.strategy_optimizer import StrategyOptimizerAgent
from ..dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter()
strategy_optimizer = StrategyOptimizerAgent()

class StrategyRequest(BaseModel):
    campaign_id: str | None = None  # Optional for exploratory queries
    platform: str
    target_persona: str
    budget: float
    goal: str | None = None  # Optional campaign goal
    context: Dict[str, Any] | None = None

class StrategyUpdateRequest(BaseModel):
    campaign_id: str
    strategy_action: str
    reward: float
    context: Dict[str, Any] | None = None

@router.post("/optimize", response_model=Dict[str, Any])
async def get_optimal_strategy(
    request: StrategyRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        campaign_id = request.campaign_id or "exploratory"
        
        context = request.context or {}
        if request.goal:
            context["goal"] = request.goal
        
        strategy = await strategy_optimizer.get_optimal_strategy(
            campaign_id=campaign_id,
            platform=request.platform,
            target_persona=request.target_persona,
            budget=request.budget,
            context=context
        )

        logger.info(f"Strategy optimized for campaign {campaign_id}: {strategy.get('strategy_name')}")

        return strategy

    except Exception as e:
        logger.error(f"Strategy optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-performance", response_model=Dict[str, str])
async def update_strategy_performance(
    request: StrategyUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        await strategy_optimizer.update_strategy_performance(
            campaign_id=request.campaign_id,
            action=request.strategy_action,
            reward=request.reward,
            context=request.context
        )

        logger.info(f"Strategy performance updated for campaign {request.campaign_id}: action={request.strategy_action}, reward={request.reward}")

        return {
            "status": "success",
            "message": f"Updated performance for strategy '{request.strategy_action}'"
        }

    except Exception as e:
        logger.error(f"Strategy performance update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{campaign_id}", response_model=Dict[str, Any])
async def get_strategy_performance(
    campaign_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        report = strategy_optimizer.get_performance_report()

        if not report or report.get('message') == 'No performance data available':
            return {
                "campaign_id": campaign_id,
                "status": "no_data",
                "message": "No performance data available yet",
                "total_strategies": 0,
                "best_ctr": 0.0,
                "best_conversions": 0,
                "lowest_cpl": 0.0,
                "success_rate": 0.0,
                "strategy_history": []
            }

        action_stats = report.get('action_statistics', {})
        total_decisions = report.get('total_decisions', 0)
        unique_actions = report.get('unique_actions', 0)
        
        best_reward = report.get('best_action_reward', 0)
        
        strategy_history = []
        for action, stats in action_stats.items():
            avg_reward = stats.get('average_reward', 0)
            count = stats.get('count', 0)
            
            strategy_history.append({
                "strategy": action.replace("_", " ").title(),
                "action": action,
                "count": count,
                "total_reward": stats.get('total_reward', 0),
                "average_reward": avg_reward,
                "estimated_ctr": round(2.0 + avg_reward * 3.0, 2),  # 2-5% CTR range
                "estimated_conversions": int(count * avg_reward * 10),  # Based on pulls and reward
                "estimated_cpl": round(max(20, 50 - avg_reward * 30), 2)  # Lower CPL = better
            })
        
        strategy_history.sort(key=lambda x: x['average_reward'], reverse=True)
        
        if strategy_history:
            best_ctr = max(s['estimated_ctr'] for s in strategy_history)
            best_conversions = max(s['estimated_conversions'] for s in strategy_history)
            lowest_cpl = min(s['estimated_cpl'] for s in strategy_history)
            success_rate = sum(1 for s in strategy_history if s['average_reward'] > 0.5) / len(strategy_history) if strategy_history else 0
        else:
            best_ctr = 0.0
            best_conversions = 0
            lowest_cpl = 0.0
            success_rate = 0.0

        return {
            "campaign_id": campaign_id,
            "status": "success",
            "total_strategies": unique_actions,
            "total_decisions": total_decisions,
            "best_ctr": round(best_ctr, 2),
            "best_conversions": best_conversions,
            "lowest_cpl": round(lowest_cpl, 2),
            "success_rate": round(success_rate, 2),
            "best_action": report.get('best_action', 'N/A'),
            "best_action_reward": round(best_reward, 4),
            "strategy_history": strategy_history[:10]  # Limit to 10 entries
        }

    except Exception as e:
        logger.error(f"Failed to get strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
