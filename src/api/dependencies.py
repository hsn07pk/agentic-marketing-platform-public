"""
FastAPI dependencies
"""
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
import jwt
from datetime import datetime, timedelta

from ..config.settings import settings
from ..data_layer.database.connection import get_async_session

# Security
security = HTTPBearer()

async def get_db() -> AsyncSession:
    """Get database session"""
    async with get_async_session() as session:
        yield session

async def get_redis() -> redis.Redis:
    """Get Redis connection"""
    client = await redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    try:
        yield client
    finally:
        await client.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current user from JWT token"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        return {
            "username": payload.get("sub"),
            "user_id": payload.get("user_id")
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

async def get_orchestrator():
    """Get orchestrator instance"""
    from ..ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator
    return MarketingOrchestrator()