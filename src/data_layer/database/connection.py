"""
Database connection management
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager
from typing import Generator

from ...config.settings import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'),
    echo=settings.DEBUG,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    poolclass=NullPool if settings.DEBUG else None
)

# Create sync engine for synchronous operations
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create sync session factory
sync_session_maker = sessionmaker(
    sync_engine,
    class_=Session,
    expire_on_commit=False
)

def get_engine():
    """Get async database engine"""
    return engine

def get_sync_engine():
    """Get sync database engine"""
    return sync_engine

@asynccontextmanager
async def get_async_session():
    """Get async database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for synchronous database sessions.
    Use this for sync operations like configuration management.
    """
    db = sync_session_maker()
    try:
        yield db
    finally:
        db.close()


def get_sync_session() -> Session:
    """
    Get a synchronous database session directly (not as a generator).
    Caller is responsible for closing the session.
    """
    return sync_session_maker()


from contextlib import contextmanager

@contextmanager
def get_sync_db_session():
    """
    Context manager for synchronous database sessions.
    Ensures proper cleanup of the session.
    
    Usage:
        with get_sync_db_session() as db:
            service = ConfigurationService(db)
            value = service.get_value('KEY')
    """
    session = sync_session_maker()
    try:
        yield session
    finally:
        session.close()