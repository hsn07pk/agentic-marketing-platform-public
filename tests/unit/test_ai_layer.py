# tests/unit/test_ai_layer.py
"""
Comprehensive unit tests for the AI layer.

Covers:
  - OllamaClient (local LLM integration)
  - ContentGeneratorAgent (RAG pipeline, claim injection, content parsing)
  - StrategyOptimizerAgent (bandit selection, strategy building, GPU-accelerated neural bandit)
  - MarketScraperAgent (competitor analysis, content pattern analysis, demo posts)
  - MarketingOrchestrator (LangGraph supervisor, OODA-G loop nodes)

All external dependencies (LLM APIs, database, Redis, Apify) are mocked.
"""
import sys
import pytest
import json
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Pre-mock heavy dependencies so module-level imports don't fail
# ---------------------------------------------------------------------------
_STUB_MODULES = {}


def _ensure_stub(name):
    """Insert a MagicMock into sys.modules if the real module is unavailable."""
    if name not in sys.modules:
        _STUB_MODULES[name] = MagicMock()
        sys.modules[name] = _STUB_MODULES[name]


# Redis
_ensure_stub("redis")
_ensure_stub("redis.asyncio")

# pgvector
_ensure_stub("pgvector")
_ensure_stub("pgvector.sqlalchemy")

# LangChain / LangGraph / OpenAI
for _m in [
    "langchain", "langchain.chat_models", "langchain.prompts",
    "langchain.schema", "langchain.callbacks",
    "langchain_openai", "langchain_community",
    "langchain_community.callbacks",
    "sentence_transformers",
    "langgraph", "langgraph.graph",
    "openai",
]:
    _ensure_stub(_m)

# Torch — force CUDA to appear unavailable so module-level GPU detection
# in strategy_optimizer.py is skipped (avoids MagicMock format errors)
import torch as _real_torch
_real_torch.cuda.is_available = lambda: False

# httpx / BeautifulSoup / aiohttp (for MarketScraper / connectors)
_ensure_stub("httpx")
_ensure_stub("bs4")
_ensure_stub("aiohttp")

# Apify
_ensure_stub("apify_client")

# watchdog (for data_file_monitor)
_ensure_stub("watchdog")
_ensure_stub("watchdog.events")
_ensure_stub("watchdog.observers")

# rq / rq_scheduler (for worker)
_ensure_stub("rq")
_ensure_stub("rq.job")
_ensure_stub("rq_scheduler")

# mlflow
_ensure_stub("mlflow")


# ===========================================================================
# 1. OllamaClient Tests
# ===========================================================================

class TestOllamaClient:
    """Tests for src/ai_layer/agents/ollama_integration.py"""

    def _make_client(self, host="http://localhost:11434"):
        from src.ai_layer.agents.ollama_integration import OllamaClient
        return OllamaClient(host=host)

    def test_init_defaults(self):
        client = self._make_client()
        assert client.host == "http://localhost:11434"
        assert client.model is None

    def test_init_custom_host(self):
        client = self._make_client("http://gpu-server:11434")
        assert client.host == "http://gpu-server:11434"

    def test_set_model(self):
        client = self._make_client()
        client.set_model("mistral:7b")
        assert client.model == "mistral:7b"

    @pytest.mark.asyncio
    async def test_generate_non_streaming(self):
        client = self._make_client()
        client.set_model("mistral:7b")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text here"}

        with patch("requests.post", return_value=mock_response):
            result = await client.generate("Write a tagline")
            assert result == "Generated text here"

    @pytest.mark.asyncio
    async def test_generate_uses_specified_model_over_default(self):
        client = self._make_client()
        client.set_model("default_model")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            await client.generate("prompt", model="override_model")
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["model"] == "override_model"

    @pytest.mark.asyncio
    async def test_generate_raises_on_error(self):
        client = self._make_client()
        client.set_model("mistral:7b")

        with patch("requests.post", side_effect=ConnectionError("down")):
            with pytest.raises(ConnectionError):
                await client.generate("prompt")

    @pytest.mark.asyncio
    async def test_generate_custom_params(self):
        client = self._make_client()
        client.set_model("mistral:7b")

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            await client.generate("prompt", temperature=0.2, max_tokens=500)
            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 0.2
            assert payload["num_predict"] == 500

    def test_list_models_success(self):
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "mistral:7b"}, {"name": "llama3:8b"}]
        }
        with patch("requests.get", return_value=mock_response):
            models = client.list_models()
            assert len(models) == 2
            assert models[0]["name"] == "mistral:7b"

    def test_list_models_error_returns_empty(self):
        client = self._make_client()
        with patch("requests.get", side_effect=Exception("offline")):
            assert client.list_models() == []

    def test_select_best_model_high_vram(self):
        from src.ai_layer.agents.ollama_integration import OllamaClient
        result = OllamaClient.select_best_model("content_generation", 30000)
        assert result == "mixtral:8x7b"

    def test_select_best_model_low_vram(self):
        from src.ai_layer.agents.ollama_integration import OllamaClient
        result = OllamaClient.select_best_model("content_generation", 500)
        assert result == "tinyllama"

    def test_select_best_model_safety_task(self):
        from src.ai_layer.agents.ollama_integration import OllamaClient
        result = OllamaClient.select_best_model("safety_validation", 50000)
        assert result == "llama3:70b"

    def test_select_best_model_unknown_task(self):
        from src.ai_layer.agents.ollama_integration import OllamaClient
        result = OllamaClient.select_best_model("unknown_task", 5000)
        assert result == "mistral:7b"

    def test_handle_stream(self):
        client = self._make_client()
        mock_response = MagicMock()
        lines = [
            json.dumps({"response": "Hello "}).encode(),
            json.dumps({"response": "world"}).encode(),
        ]
        mock_response.iter_lines.return_value = lines
        result = client._handle_stream(mock_response)
        assert result == "Hello world"

    def test_pull_model(self):
        client = self._make_client()
        mock_response = MagicMock()
        lines = [
            json.dumps({"status": "downloading"}).encode(),
            json.dumps({"status": "success"}).encode(),
        ]
        mock_response.iter_lines.return_value = lines
        with patch("requests.post", return_value=mock_response):
            client.pull_model("mistral:7b")  # Should not raise


# ===========================================================================
# 2. ContentGeneratorAgent Tests
# ===========================================================================

class TestContentGeneratorAgent:
    """Tests for src/ai_layer/agents/content_generator.py

    The agent __init__ has deep dependencies (LLM, Redis, VectorStore,
    SemanticCache, EpisodicMemory, BrandVoice, PromptShield, MarketScraper).
    We bypass __init__ entirely with object.__new__ and set only the attributes
    needed to test pure-logic methods.
    """

    def _make_agent(self):
        """Create a ContentGeneratorAgent WITHOUT calling __init__."""
        from src.ai_layer.agents.content_generator import ContentGeneratorAgent
        agent = object.__new__(ContentGeneratorAgent)

        # Minimal attributes required by the methods under test
        agent.claim_library = {
            "claims": [
                {"id": "CLM_001", "text": "50% ROI improvement", "source": "Gartner", "personas": ["cfo"], "goals": ["roi"], "priority": 9},
                {"id": "CLM_002", "text": "2x engagement rate", "source": "Internal Study", "personas": ["cmo"], "goals": ["engagement"], "priority": 7},
                {"id": "CLM_003", "text": "30% cost reduction", "source": "McKinsey", "personas": ["cfo", "coo"], "goals": ["cost"], "priority": 8},
            ],
            "version": "1.0.0",
            "validation_rules": {
                "min_claims_per_content": 1,
                "max_claims_per_content": 3,
                "require_citation": True,
                "citation_format": "[CLAIM_ID]",
            },
        }
        agent.total_cost = 0.0
        agent.total_tokens = 0
        agent.company = MagicMock()
        agent.company.name = "Agentic AI"

        # Mock settings at module level for MAX_CLAIMS_PER_CONTENT
        with patch("src.ai_layer.agents.content_generator.settings") as mock_s:
            mock_s.MAX_CLAIMS_PER_CONTENT = 3
            # Store ref so tests can re-patch if needed
            agent._mock_settings = mock_s

        return agent

    # -- Claim selection logic --

    def test_select_claims_filters_by_persona(self):
        agent = self._make_agent()
        with patch("src.ai_layer.agents.content_generator.settings") as ms:
            ms.MAX_CLAIMS_PER_CONTENT = 3
            claims = agent._select_claims("cfo", {"goal": "roi"})
        claim_ids = [c["id"] for c in claims]
        assert "CLM_001" in claim_ids
        assert "CLM_003" in claim_ids

    def test_select_claims_all_when_no_match(self):
        agent = self._make_agent()
        with patch("src.ai_layer.agents.content_generator.settings") as ms:
            ms.MAX_CLAIMS_PER_CONTENT = 3
            claims = agent._select_claims("unknown_persona", {})
        # Should fall back to returning all claims
        assert len(claims) > 0

    def test_select_claims_respects_max(self):
        agent = self._make_agent()
        with patch("src.ai_layer.agents.content_generator.settings") as ms:
            ms.MAX_CLAIMS_PER_CONTENT = 1
            claims = agent._select_claims("cfo", {})
        assert len(claims) == 1

    # -- Claim citation extraction --

    def test_extract_claim_citations_bracket_format(self):
        agent = self._make_agent()
        result = agent._extract_claim_citations("Results [CLM_001] show [CLM_002] growth.")
        assert "CLM_001" in result
        assert "CLM_002" in result

    def test_extract_claim_citations_prefixed_format(self):
        agent = self._make_agent()
        result = agent._extract_claim_citations("See [CLAIM_ID:CLM_001] for details.")
        assert "CLM_001" in result

    def test_extract_claim_citations_no_duplicates(self):
        agent = self._make_agent()
        result = agent._extract_claim_citations("[CLM_001] and [CLM_001] again")
        assert result.count("CLM_001") == 1

    def test_extract_claim_citations_invalid_id(self):
        agent = self._make_agent()
        result = agent._extract_claim_citations("[CLM_999] not in library")
        assert result == []

    def test_extract_claim_citations_empty(self):
        agent = self._make_agent()
        result = agent._extract_claim_citations("No claims here")
        assert result == []

    # -- Content parsing --

    def test_parse_generated_content_with_headline(self):
        agent = self._make_agent()
        raw = "Headline: Boost Your ROI\nBody: Our platform delivers results."
        result = agent._parse_generated_content(raw, "linkedin")
        assert result["headline"] == "Boost Your ROI"
        assert "delivers results" in result["body"]

    def test_parse_generated_content_body_only(self):
        agent = self._make_agent()
        raw = "Just some content without a headline marker."
        result = agent._parse_generated_content(raw, "twitter")
        assert result["body"] == raw

    def test_parse_generated_content_with_cta(self):
        agent = self._make_agent()
        raw = "Headline: Test\nBody: Content here\nCTA: Learn More"
        result = agent._parse_generated_content(raw, "linkedin")
        assert result["headline"] == "Test"
        assert "Content here" in result["body"]
        assert result["cta"] == "Learn More"

    def test_parse_generated_content_tweet(self):
        agent = self._make_agent()
        raw = "Tweet: Check out our latest results #AI"
        result = agent._parse_generated_content(raw, "twitter")
        assert "Check out" in result["body"]

    def test_parse_generated_content_strips_claims_metadata(self):
        agent = self._make_agent()
        raw = "Headline: Test\nBody: Great results\nClaims Used: CLM_001, CLM_002"
        result = agent._parse_generated_content(raw, "linkedin")
        assert "Claims Used" not in result.get("body", "")

    # -- Claim usage validation --

    def test_validate_claim_usage_valid(self):
        agent = self._make_agent()
        result = agent._validate_claim_usage(
            content="This shows [CLM_001] improvement",
            used_claims=["CLM_001"],
            selected_claims=[{"id": "CLM_001", "text": "50% ROI improvement", "source": "Gartner"}],
        )
        assert result["valid"] is True

    def test_validate_claim_usage_no_claims_invalid(self):
        agent = self._make_agent()
        result = agent._validate_claim_usage(
            content="No claims at all",
            used_claims=[],
            selected_claims=[{"id": "CLM_001", "text": "50% ROI improvement", "source": "Gartner"}],
        )
        assert result["valid"] is False
        assert "Insufficient" in result["reason"]

    def test_validate_claim_usage_too_many_claims(self):
        agent = self._make_agent()
        result = agent._validate_claim_usage(
            content="[CLM_001] [CLM_002] [CLM_003] and [CLM_001] again",
            used_claims=["CLM_001", "CLM_002", "CLM_003", "CLM_001"],
            selected_claims=[
                {"id": "CLM_001"}, {"id": "CLM_002"}, {"id": "CLM_003"}, {"id": "CLM_001"}
            ],
        )
        # 4 claims > max of 3
        assert result["valid"] is False

    def test_validate_claim_usage_invalid_claim(self):
        agent = self._make_agent()
        result = agent._validate_claim_usage(
            content="[CLM_999] referenced",
            used_claims=["CLM_999"],
            selected_claims=[{"id": "CLM_001"}],
        )
        assert result["valid"] is False
        assert "not provided" in result["reason"]

    # -- Context formatting --

    def test_format_context_with_items(self):
        agent = self._make_agent()
        context = [
            {"text": "Document 1 text about marketing", "source": "kb", "score": 0.9},
            {"text": "Document 2 text about AI", "source": "web", "score": 0.8},
        ]
        result = agent._format_context(context)
        assert "Document 1" in result
        assert "Document 2" in result

    def test_format_context_empty(self):
        agent = self._make_agent()
        result = agent._format_context([])
        assert "No specific context" in result

    def test_format_claims(self):
        agent = self._make_agent()
        claims = [
            {"id": "CLM_001", "text": "50% ROI", "source": "Gartner"},
        ]
        result = agent._format_claims(claims)
        assert "CLM_001" in result
        assert "50% ROI" in result
        assert "Gartner" in result

    def test_format_claims_empty(self):
        agent = self._make_agent()
        result = agent._format_claims([])
        assert "No specific claims" in result

    # -- Generation stats --

    def test_get_generation_stats_initial(self):
        agent = self._make_agent()
        stats = agent.get_generation_stats()
        assert isinstance(stats, dict)
        assert stats["total_cost"] == 0.0
        assert stats["total_tokens"] == 0


# ===========================================================================
# 3. StrategyOptimizerAgent Tests
# ===========================================================================

class TestStrategyOptimizerAgent:
    """Tests for src/ai_layer/agents/strategy_optimizer.py"""

    def _make_agent(self):
        """Create a StrategyOptimizerAgent WITHOUT calling __init__."""
        from src.ai_layer.agents.strategy_optimizer import StrategyOptimizerAgent, StrategyConfig
        agent = object.__new__(StrategyOptimizerAgent)

        # Minimal attributes used by the methods under test
        agent.config = StrategyConfig(
            algorithm="thompson_sampling",
            exploration_rate=0.1,
            learning_rate=0.01,
            use_gpu=False,
            context_dim=50,
        )
        agent.active_bandits = {}
        agent.performance_history = []
        agent.memory = MagicMock()
        agent.memory.retrieve_relevant_memories = AsyncMock(return_value=[])
        agent.memory.store_memory = AsyncMock()
        agent.research_mode = False
        agent.experiment_runner = None
        agent.tracker = MagicMock()
        agent.tracker.record_simulation_predictions = AsyncMock()
        return agent

    def test_init_creates_agent(self):
        agent = self._make_agent()
        assert agent is not None
        assert agent.config is not None

    def test_build_context_vector_shape(self):
        agent = self._make_agent()
        ctx = agent._build_context_vector("linkedin", "cfo", 5000.0)
        assert isinstance(ctx, np.ndarray)
        assert len(ctx) > 0

    def test_build_context_vector_different_platforms(self):
        agent = self._make_agent()
        ctx_li = agent._build_context_vector("linkedin", "cfo", 5000.0)
        ctx_tw = agent._build_context_vector("twitter", "cmo", 5000.0)
        # Different platforms should produce different context vectors
        assert not np.array_equal(ctx_li, ctx_tw)

    def test_get_platform_optimizations_linkedin(self):
        agent = self._make_agent()
        opts = agent._get_platform_optimizations("linkedin")
        assert isinstance(opts, dict)
        assert len(opts) > 0

    def test_get_platform_optimizations_twitter(self):
        agent = self._make_agent()
        opts = agent._get_platform_optimizations("twitter")
        assert isinstance(opts, dict)

    def test_get_platform_optimizations_unknown(self):
        agent = self._make_agent()
        opts = agent._get_platform_optimizations("tiktok")
        assert isinstance(opts, dict)

    def test_optimize_budget_allocation(self):
        agent = self._make_agent()
        allocation = agent._optimize_budget_allocation(10000.0, 0.8)
        assert isinstance(allocation, dict)
        # Total allocated should approximately equal budget
        total = sum(allocation.values())
        assert abs(total - 10000.0) < 100  # Allow small rounding

    def test_get_optimal_timing(self):
        agent = self._make_agent()
        timing = agent._get_optimal_timing("linkedin", "cfo")
        assert isinstance(timing, dict)

    def test_get_default_strategy(self):
        agent = self._make_agent()
        strategy = agent._get_default_strategy("linkedin", "cfo")
        assert isinstance(strategy, dict)
        assert "platform" in strategy or "action" in strategy or len(strategy) > 0

    def test_build_strategy_hook_transform(self):
        agent = self._make_agent()
        strategy = agent._build_strategy(
            action="hook_transform",
            confidence=0.85,
            platform="linkedin",
            persona="decision_maker",
            budget=5000.0,
        )
        assert isinstance(strategy, dict)
        assert strategy["confidence"] == 0.85
        assert strategy["strategy_name"] == "Transformation Focus"

    def test_build_strategy_unknown_action_fallback(self):
        agent = self._make_agent()
        strategy = agent._build_strategy(
            action="unknown_action",
            confidence=0.5,
            platform="linkedin",
            persona="decision_maker",
            budget=1000.0,
        )
        # Unknown action falls back to hook_transform
        assert strategy["strategy_name"] == "Transformation Focus"

    def test_get_performance_report_empty(self):
        agent = self._make_agent()
        with patch.object(agent, '_load_performance_from_db'):
            report = agent.get_performance_report()
        assert isinstance(report, dict)
        assert "message" in report  # No performance data

    def test_get_performance_report_with_data(self):
        agent = self._make_agent()
        agent.performance_history = [
            {"campaign_id": "c1", "action": "hook_transform", "reward": 0.8, "timestamp": datetime.utcnow()},
            {"campaign_id": "c1", "action": "hook_transform", "reward": 0.9, "timestamp": datetime.utcnow()},
            {"campaign_id": "c2", "action": "hook_problem", "reward": 0.6, "timestamp": datetime.utcnow()},
        ]
        report = agent.get_performance_report()
        assert report["total_decisions"] == 3
        assert report["best_action"] == "hook_transform"

    @pytest.mark.asyncio
    async def test_get_optimal_strategy(self):
        agent = self._make_agent()
        # Mock the bandit
        mock_bandit = MagicMock()
        mock_bandit.select_arm = MagicMock(return_value=("hook_transform", 0.85))
        agent.active_bandits = {"camp-1": mock_bandit}

        with patch("src.ai_layer.agents.strategy_optimizer.get_sync_session"):
            result = await agent.get_optimal_strategy(
                campaign_id="camp-1",
                platform="linkedin",
                target_persona="decision_maker",
                budget=5000.0,
            )
        assert isinstance(result, dict)
        assert result["strategy_name"] == "Transformation Focus"

    @pytest.mark.asyncio
    async def test_update_strategy_performance(self):
        agent = self._make_agent()
        mock_bandit = MagicMock()
        agent.active_bandits = {"camp-1": mock_bandit}

        with patch("src.ai_layer.agents.strategy_optimizer.get_sync_session"):
            await agent.update_strategy_performance(
                campaign_id="camp-1",
                action="hook_transform",
                reward=0.7,
            )
        mock_bandit.update_arm.assert_called_once_with("hook_transform", 0.7)
        assert len(agent.performance_history) == 1


# ===========================================================================
# 4. MarketScraperAgent Tests
# ===========================================================================

class TestMarketScraperAgent:
    """Tests for src/ai_layer/agents/market_scraper.py"""

    def _make_agent(self):
        """Create a MarketScraperAgent with mocked dependencies."""
        with patch("src.ai_layer.agents.market_scraper.settings") as mock_settings, \
             patch("src.ai_layer.agents.market_scraper._get_config_value", return_value=None), \
             patch("src.ai_layer.agents.market_scraper.get_async_session"), \
             patch("src.ai_layer.agents.market_scraper.get_sync_session"):

            mock_settings.APIFY_API_TOKEN = ""
            mock_settings.ENABLE_SCRAPING = False
            mock_settings.PROJECT_ROOT = "/tmp"
            mock_settings.get = MagicMock(side_effect=lambda k, d=None: d)

            from src.ai_layer.agents.market_scraper import MarketScraperAgent
            agent = MarketScraperAgent()

        return agent

    def test_init(self):
        agent = self._make_agent()
        assert agent is not None
        assert agent.enabled is False  # Scraping disabled in test

    def test_get_competitor_profiles_default(self):
        agent = self._make_agent()
        profiles = agent.get_competitor_profiles()
        # Should return dict (possibly empty or with defaults)
        assert isinstance(profiles, (dict, list))

    @pytest.mark.asyncio
    async def test_get_inspiration_disabled_returns_empty(self):
        agent = self._make_agent()
        result = await agent.get_inspiration_for_campaign(
            keywords=["employee experience", "people analytics"],
            limit=5,
            platform="linkedin",
        )
        assert isinstance(result, dict)
        # When disabled, should return empty (no demo fallback)
        assert result.get('success') is False

    def test_extract_keywords(self):
        agent = self._make_agent()
        texts = [
            "Employee engagement analytics drives better retention outcomes",
            "AI-powered people analytics improves workplace wellbeing",
        ]
        keywords = agent._extract_keywords(texts)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should find domain-relevant keywords
        assert any(kw in keywords for kw in ['engagement', 'analytics', 'retention', 'wellbeing', 'ai'])

    def test_get_differentiation_opportunities(self):
        agent = self._make_agent()
        agent.competitor_profiles = {
            "competitor_a": {
                "name": "Competitor A",
                "strengths": ["pricing"],
                "weaknesses": ["support"],
                "our_differentiators": ["AI-driven insights"],
            }
        }
        opps = agent.get_differentiation_opportunities()
        assert isinstance(opps, list)

    @pytest.mark.asyncio
    async def test_analyze_content_patterns(self):
        agent = self._make_agent()
        posts = [
            {
                "text": "Research shows employee engagement drives 21% higher productivity. Book a demo to learn more.",
                "likes": 150,
                "comments": 30,
                "shares": 20,
                "platform": "linkedin",
            },
            {
                "text": "New study finds QWL-based interventions reduce attrition by 35%. Download the whitepaper.",
                "likes": 80,
                "comments": 10,
                "shares": 5,
                "platform": "linkedin",
            },
        ]
        analysis = await agent.analyze_content_patterns(posts)
        assert isinstance(analysis, dict)
        assert "top_hooks" in analysis
        assert "top_ctas" in analysis
        assert "common_themes" in analysis
        assert analysis["total_analyzed"] == 2

    def test_format_competitive_insights(self):
        agent = self._make_agent()
        agent.competitor_profiles = {
            "competitor_a": {
                "name": "Competitor A",
                "strengths": ["analytics"],
                "weaknesses": ["UX"],
                "our_differentiators": ["Better UX"],
            }
        }
        insights = agent.format_competitive_insights_for_content()
        assert isinstance(insights, str)

    def test_format_competitive_insights_with_patterns(self):
        agent = self._make_agent()
        agent.competitor_profiles = {"competitors": [
            {"name": "Test", "differentiation_opportunities": ["QWL science approach"]}
        ]}
        patterns = {
            "top_hooks": [{"text": "Research shows engagement drives productivity"}],
            "common_themes": [{"theme": "engagement"}, {"theme": "analytics"}],
            "top_ctas": [{"cta": "book a demo"}],
        }
        insights = agent.format_competitive_insights_for_content(content_patterns=patterns)
        assert isinstance(insights, str)
        assert "engagement" in insights.lower() or "research" in insights.lower()

    def test_get_scrape_sources_default(self):
        agent = self._make_agent()
        sources = agent._get_scrape_sources()
        assert isinstance(sources, list)
        assert len(sources) > 0
        # Should have domain-relevant sources, not generic marketing blogs
        names = [s['name'] for s in sources]
        assert not any('HubSpot' in n for n in names)
        assert not any('Neil Patel' in n for n in names)


# ===========================================================================
# 5. StrategyConfig Tests
# ===========================================================================

class TestStrategyConfig:
    """Tests for the StrategyConfig dataclass."""

    def test_default_values(self):
        from src.ai_layer.agents.strategy_optimizer import StrategyConfig
        config = StrategyConfig()
        assert config.algorithm == "thompson_sampling"
        assert config.exploration_rate == 0.1
        assert config.learning_rate == 0.01
        assert config.use_gpu is True
        assert config.batch_size == 32
        assert config.context_dim == 50

    def test_custom_values(self):
        from src.ai_layer.agents.strategy_optimizer import StrategyConfig
        config = StrategyConfig(
            algorithm="linucb",
            exploration_rate=0.2,
            use_gpu=False,
        )
        assert config.algorithm == "linucb"
        assert config.exploration_rate == 0.2
        assert config.use_gpu is False


# ===========================================================================
# 6. NeuralBandit Tests
# ===========================================================================

class TestNeuralBandit:
    """Tests for the NeuralBandit PyTorch module."""

    def test_init(self):
        """NeuralBandit should initialize with correct architecture."""
        with patch("src.ai_layer.agents.strategy_optimizer.DEVICE", "cpu"):
            from src.ai_layer.agents.strategy_optimizer import NeuralBandit
            import torch

            # Since torch is mocked, just test initialization succeeds
            bandit = NeuralBandit(input_dim=50, hidden_dim=128, num_arms=4)
            assert bandit is not None


# ===========================================================================
# 7. MarketingOrchestrator Tests
# ===========================================================================

class TestMarketingOrchestrator:
    """Tests for the LangGraph-based MarketingOrchestrator.

    The orchestrator has deep dependencies. We mock heavily but ensure
    the node functions produce expected state transformations.
    """

    def _make_orchestrator(self):
        """Create orchestrator with all deps mocked."""
        with patch("src.ai_layer.orchestration.langgraph_supervisor.settings") as mock_settings, \
             patch("src.ai_layer.orchestration.langgraph_supervisor.ContentGeneratorAgent") as MockCGA, \
             patch("src.ai_layer.orchestration.langgraph_supervisor.StrategyOptimizerAgent") as MockSOA, \
             patch("src.ai_layer.orchestration.langgraph_supervisor.MarketScraperAgent") as MockMSA, \
             patch("src.ai_layer.orchestration.langgraph_supervisor.SafetyValidatorAgent") as MockSVA, \
             patch("src.ai_layer.orchestration.langgraph_supervisor.BudgetManager") as MockBM, \
             patch("src.ai_layer.orchestration.langgraph_supervisor.HITLQueueManager") as MockHITL, \
             patch("src.ai_layer.orchestration.langgraph_supervisor.MarketingEnvironment"), \
             patch("src.ai_layer.orchestration.langgraph_supervisor.PersonaFactory"), \
             patch("src.ai_layer.orchestration.langgraph_supervisor.MARLGatekeeper"), \
             patch("src.ai_layer.orchestration.langgraph_supervisor.EpisodicMemoryStore"), \
             patch("src.ai_layer.orchestration.langgraph_supervisor.ConfigurationService"), \
             patch("src.ai_layer.orchestration.langgraph_supervisor.StateGraph") as MockSG:

            mock_settings.get = MagicMock(side_effect=lambda k, d=None: d)
            mock_settings.MOCK_MODE_ENABLED = True
            mock_settings.DATABASE_URL = "postgresql://fake"
            mock_settings.REDIS_URL = "redis://localhost"
            mock_settings.OPENAI_API_KEY = "fake"
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.SAFETY_THRESHOLD = 0.7
            mock_settings.HITL_SAFETY_THRESHOLD = 0.5
            mock_settings.ENABLE_SCRAPING = False
            mock_settings.PROJECT_ROOT = "/tmp"

            # Make StateGraph mock chainable
            mock_graph = MagicMock()
            mock_graph.compile.return_value = MagicMock()
            MockSG.return_value = mock_graph

            from src.ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator
            orch = MarketingOrchestrator()

        return orch

    def test_init_creates_orchestrator(self):
        orch = self._make_orchestrator()
        assert orch is not None

    def test_build_graph_called(self):
        orch = self._make_orchestrator()
        # The compiled graph should exist after init (stored in self.app)
        assert hasattr(orch, 'app') or hasattr(orch, 'graph')

    def test_get_config_value_fallback(self):
        orch = self._make_orchestrator()
        # Should return default when no DB config
        result = orch._get_config_value("NONEXISTENT_KEY", "fallback_val")
        assert result == "fallback_val"

    def test_state_to_features(self):
        orch = self._make_orchestrator()
        state = {
            "budget": 5000.0,
            "platform": "linkedin",
            "persona": "cfo",
            "ctr": 0.05,
            "cvr": 0.02,
            "impressions": 10000,
            "clicks": 500,
            "conversions": 100,
            "safety_score": 0.9,
            "content_quality": 0.8,
            "day_of_week": 2,
            "hour_of_day": 14,
        }
        features = orch._state_to_features(state)
        assert isinstance(features, np.ndarray)
        # May be 1D (13,) or 2D (1,13) depending on implementation
        total_elements = features.size
        assert total_elements == 13

    @pytest.mark.asyncio
    async def test_observe_market_node(self):
        orch = self._make_orchestrator()
        orch.scraper = MagicMock()
        orch.scraper.get_inspiration_for_campaign = AsyncMock(return_value={
            "posts": [{"text": "test post", "engagement": {"likes": 10}}],
            "insights": {"top_themes": ["AI"]},
        })
        orch.scraper.format_competitive_insights_for_content = MagicMock(return_value="Insights text")

        state = {
            "campaign_id": "test-1",
            "platform": "linkedin",
            "persona": "cfo",
            "budget": 5000.0,
            "keywords": ["AI", "marketing"],
            "market_insights": None,
            "step": "observe",
        }
        result = await orch._observe_market_node(state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_optimize_strategy_node(self):
        orch = self._make_orchestrator()
        orch.strategy_optimizer = MagicMock()
        orch.strategy_optimizer.get_optimal_strategy = AsyncMock(return_value={
            "action": "aggressive_content",
            "confidence": 0.85,
            "timing": {"best_hour": 9},
        })
        orch.memory_store = MagicMock()
        orch.memory_store.query_memories = MagicMock(return_value=[])

        state = {
            "campaign_id": "test-1",
            "platform": "linkedin",
            "persona": "cfo",
            "budget": 5000.0,
            "market_insights": {"top_themes": ["AI"]},
            "strategy": None,
            "step": "orient",
        }
        result = await orch._optimize_strategy_node(state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_content_node(self):
        orch = self._make_orchestrator()
        orch.content_generator = MagicMock()
        orch.content_generator.generate_content = AsyncMock(return_value={
            "headline": "Boost ROI",
            "body": "Our platform delivers results [CLM_001].",
            "claims_used": ["CLM_001"],
        })
        orch.memory_store = MagicMock()
        orch.memory_store.query_memories = MagicMock(return_value=[])

        state = {
            "campaign_id": "test-1",
            "platform": "linkedin",
            "persona": "cfo",
            "budget": 5000.0,
            "strategy": {"action": "aggressive_content"},
            "market_insights": {},
            "content": None,
            "step": "decide",
        }
        result = await orch._generate_content_node(state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_validate_safety_node_passes(self):
        orch = self._make_orchestrator()
        orch.safety_validator = MagicMock()
        orch.safety_validator.validate_content = AsyncMock(return_value={
            "overall_score": 0.95,
            "safe": True,
        })
        orch.memory_store = MagicMock()
        orch.memory_store.query_memories = MagicMock(return_value=[])

        state = {
            "campaign_id": "test-1",
            "content": {"headline": "Test", "body": "Safe content"},
            "safety_score": None,
            "safety_passed": None,
            "step": "govern_safety",
        }
        result = await orch._validate_safety_node(state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_check_cost_node(self):
        orch = self._make_orchestrator()
        orch.budget_manager = MagicMock()
        orch.budget_manager.check_budget = AsyncMock(return_value=True)

        state = {
            "campaign_id": "test-1",
            "budget": 5000.0,
            "cost_approved": None,
            "step": "govern_cost",
        }
        result = await orch._check_cost_node(state)
        assert isinstance(result, dict)

    def test_route_after_simulation(self):
        orch = self._make_orchestrator()
        state = {
            "simulation_passed": True,
            "marl_enabled": False,
        }
        result = orch._route_after_simulation(state)
        assert isinstance(result, str)
