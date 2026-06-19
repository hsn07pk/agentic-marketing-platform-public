from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

from ...ai_layer.memory.episodic_memory import EpisodicMemoryStore
from ..dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

class MemoryQueryRequest(BaseModel):
    agent_name: str
    query: str
    k: int | None = 5

@router.get("/agents", response_model=List[str])
async def list_agents_with_memory(
    db: AsyncSession = Depends(get_db)
):
    """
    List all agents that have episodic memory
    """
    try:
        # Known agents with memory
        agents = [
            "content_generator",
            "strategy_optimizer",
            "safety_validator",
            "market_scraper"
        ]

        return agents

    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_name}/stats", response_model=Dict[str, Any])
async def get_agent_memory_stats(
    agent_name: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics about an agent's episodic memory

    Returns total memories, success rate, common failure patterns
    """
    try:
        memory_store = EpisodicMemoryStore(agent_name=agent_name)

        stats = await memory_store.get_memory_stats()

        return {
            "agent_name": agent_name,
            "total_memories": stats.get("total_count", 0),
            "success_count": stats.get("success_count", 0),
            "failure_count": stats.get("failure_count", 0),
            "success_rate": stats.get("success_rate", 0.0),
            "avg_cost": stats.get("avg_cost", 0.0),
            "avg_duration": stats.get("avg_duration", 0.0)
        }

    except Exception as e:
        logger.error(f"Failed to get memory stats for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=List[Dict[str, Any]])
async def query_agent_memory(
    request: MemoryQueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Query an agent's episodic memory by semantic similarity

    Returns relevant past experiences for a given task description
    """
    try:
        memory_store = EpisodicMemoryStore(agent_name=request.agent_name)

        memories = await memory_store.retrieve_relevant_memories(
            query=request.query,  # Fixed: was 'task_description', should be 'query'
            k=request.k
        )

        # Format response - memory is a dict with 'content', 'metadata', 'similarity_score' keys
        formatted = []
        for memory in memories:
            content = memory.get('content', '')
            metadata = memory.get('metadata', {})
            
            # Parse content to extract task description and actions
            lines = content.split('\n')
            task_desc = lines[0].replace('Task: ', '') if lines else 'Unknown'
            
            # Extract actions
            actions = []
            for line in lines:
                if line.startswith('Actions: '):
                    actions = line.replace('Actions: ', '').split('; ')
                    break
            
            formatted.append({
                "task_description": task_desc,
                "outcome": memory.get('outcome') or metadata.get('outcome', 'unknown'),
                "actions_taken": actions,
                "metrics": {},  # Not stored in vector format
                "human_feedback": metadata.get('human_feedback'),
                "lessons_learned": metadata.get('lessons_learned'),
                "timestamp": memory.get('timestamp') or metadata.get('timestamp'),
                "similarity_score": memory.get('similarity_score', 0)
            })
        
        return formatted

    except Exception as e:
        logger.error(f"Failed to query memory for {request.agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_name}/recent", response_model=List[Dict[str, Any]])
async def get_recent_memories(
    agent_name: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent memories for an agent (chronological order)
    """
    try:
        memory_store = EpisodicMemoryStore(agent_name=agent_name)

        memories = await memory_store.get_recent_memories(limit=limit)

        return [
            {
                "task_description": memory.task_description,
                "outcome": memory.outcome,
                "actions_taken": memory.actions_taken,
                "metrics": memory.metrics,
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None
            }
            for memory in memories
        ]

    except Exception as e:
        logger.error(f"Failed to get recent memories for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_name}/failures", response_model=List[Dict[str, Any]])
async def get_failure_patterns(
    agent_name: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get common failure patterns for an agent

    Helps identify areas where the agent struggles
    """
    try:
        memory_store = EpisodicMemoryStore(agent_name=agent_name)

        failures = await memory_store.get_failure_patterns(limit=limit)

        return failures

    except Exception as e:
        logger.error(f"Failed to get failure patterns for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{agent_name}/clear", response_model=Dict[str, str])
async def clear_agent_memory(
    agent_name: str,
    confirm: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Clear all episodic memories for an agent

    WARNING: This is irreversible and will reset the agent's learning
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to clear memory"
        )

    try:
        memory_store = EpisodicMemoryStore(agent_name=agent_name)

        await memory_store.clear_all_memories()

        logger.warning(f"All memories cleared for agent: {agent_name}")

        return {
            "status": "success",
            "message": f"All memories for {agent_name} have been cleared"
        }

    except Exception as e:
        logger.error(f"Failed to clear memory for {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
