# tests/conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.data_layer.database.models import Base
from src.config.settings import settings

TEST_DATABASE_URL = "postgresql+asyncpg://agentic:changeme@localhost:5432/agentic_test"

# ---------------------------------------------------------------------------
# Global Mocks for missing dependencies (FastAPI, LangChain, etc.)
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

def _ensure_mock(name: str):
    """Insert a MagicMock module into sys.modules if not already present."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()

# Dependency Mocks (copied from unit/conftest.py)
_ensure_mock("optuna")
_ensure_mock("transformers")
_ensure_mock("sklearn")
_ensure_mock("sklearn.ensemble")
_ensure_mock("sklearn.gaussian_process")
_ensure_mock("sklearn.gaussian_process.kernels")
_ensure_mock("sklearn.model_selection")
_ensure_mock("sklearn.metrics")
_ensure_mock("fastapi")
_ensure_mock("fastapi.middleware")
_ensure_mock("fastapi.middleware.cors")
_ensure_mock("fastapi.middleware.gzip")
_ensure_mock("fastapi.responses")
_ensure_mock("fastapi.exceptions")
_ensure_mock("fastapi.staticfiles")
_ensure_mock("fastapi.templating")
_ensure_mock("starlette")
_ensure_mock("starlette.responses")
_ensure_mock("starlette.middleware")
_ensure_mock("uvicorn")
_ensure_mock("pydantic")
_ensure_mock("fastapi.security")
_ensure_mock("prometheus_client")
_ensure_mock("tenacity")
_ensure_mock("langgraph")
_ensure_mock("langgraph.graph")
_ensure_mock("langchain")
_ensure_mock("langchain_openai")
_ensure_mock("langchain.schema")
_ensure_mock("langchain_core")
_ensure_mock("langchain_core.messages")
_ensure_mock("langchain_core.prompts")
_ensure_mock("langchain_community")
_ensure_mock("langchain_community.chat_models")
_ensure_mock("langchain_community.callbacks")
_ensure_mock("langchain.prompts")
_ensure_mock("langchain.chat_models")
_ensure_mock("langchain.embeddings")
_ensure_mock("redis")
_ensure_mock("redis.asyncio")
_ensure_mock("frontmatter")
_ensure_mock("sendgrid")
_ensure_mock("apify_client")
_ensure_mock("bs4")
_ensure_mock("selenium")
_ensure_mock("selenium.webdriver")
_ensure_mock("watchdog")
_ensure_mock("watchdog.observers")
_ensure_mock("rq")
_ensure_mock("schedule")
_ensure_mock("psutil")
_ensure_mock("sentence_transformers")
_ensure_mock("pgvector")
_ensure_mock("pgvector.sqlalchemy")
pass # numpy, pandas, simpy are installed

@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def mock_openai_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "Test generated content with CLAIM_001 [Source: Test]"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }

@pytest.fixture
def sample_campaign_data():
    return {
        "name": "Test Campaign",
        "platform": "linkedin",
        "status": "draft",
        "budget": 1000.0,
        "target_persona": "decision_maker",
        "goal": "lead_generation"
    }

@pytest.fixture
def sample_content_data():
    return {
        "headline": "Test Headline",
        "body": "Test body content with CLAIM_001 [Source: Test]",
        "platform": "linkedin",
        "persona": "decision_maker"
    }

@pytest.mark.asyncio
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )