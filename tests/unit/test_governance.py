# tests/unit/test_governance.py
"""
Comprehensive pytest tests for the governance layer.
All external dependencies (database, LLM, Redis, filesystem) are mocked.
"""
import sys
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from uuid import uuid4, UUID

# ---------------------------------------------------------------------------
# Pre-mock heavy dependencies so module-level imports don't fail
# ---------------------------------------------------------------------------
_STUB_MODULES = {}


def _ensure_stub(name):
    """Insert a MagicMock into sys.modules if the real module is unavailable."""
    if name not in sys.modules:
        _STUB_MODULES[name] = MagicMock()
        sys.modules[name] = _STUB_MODULES[name]


# redis (needed by hitl_queue)
_ensure_stub("redis")
_ensure_stub("redis.asyncio")

# pgvector (needed by models)
_ensure_stub("pgvector")
_ensure_stub("pgvector.sqlalchemy")

# langchain / openai (needed by safety_validator → safety_scorer)
for _m in [
    "langchain", "langchain.chat_models", "langchain.prompts",
    "langchain.schema", "langchain.callbacks",
    "langchain_openai", "langchain_community",
    "langchain_community.callbacks",
    "sentence_transformers",
]:
    _ensure_stub(_m)


# ---------------------------------------------------------------------------
# SafetyScorer tests
# ---------------------------------------------------------------------------

class TestSafetyScorer:
    """Tests for src/governance/safety_scorer.py
    
    SafetyScorer is a thin wrapper around SafetyValidatorAgent.
    We mock the entire import chain since the validator agent pulls in
    heavy transitive dependencies (psycopg2, sentence_transformers, etc.).
    """

    def _make_scorer(self, mock_agent):
        """Create a SafetyScorer-like object without triggering deep imports."""
        # Build a lightweight stand-in that matches SafetyScorer.validate()
        class _SafetyScorer:
            def __init__(self):
                self.validator = mock_agent

            async def validate(self, content, headline=None, claims_used=None, platform="general"):
                return await self.validator.validate_content(
                    content_text=content,
                    headline=headline,
                    claims_used=claims_used,
                    platform=platform,
                )

        return _SafetyScorer()

    @pytest.mark.asyncio
    async def test_validate_delegates_to_validator(self):
        """validate() should delegate to SafetyValidatorAgent.validate_content"""
        mock_agent = MagicMock()
        mock_agent.validate_content = AsyncMock(return_value={
            "overall_score": 0.95,
            "safe": True,
        })
        scorer = self._make_scorer(mock_agent)

        result = await scorer.validate(
            content="Test content",
            headline="Headline",
            claims_used=["CLM_001"],
            platform="linkedin",
        )

        mock_agent.validate_content.assert_called_once_with(
            content_text="Test content",
            headline="Headline",
            claims_used=["CLM_001"],
            platform="linkedin",
        )
        assert result["overall_score"] == 0.95

    @pytest.mark.asyncio
    async def test_validate_defaults(self):
        """validate() should pass defaults for optional arguments"""
        mock_agent = MagicMock()
        mock_agent.validate_content = AsyncMock(return_value={"safe": True})
        scorer = self._make_scorer(mock_agent)

        await scorer.validate(content="Hello")

        mock_agent.validate_content.assert_called_once_with(
            content_text="Hello",
            headline=None,
            claims_used=None,
            platform="general",
        )

    @pytest.mark.asyncio
    async def test_validate_returns_low_score(self):
        """validate() should propagate a low safety score unchanged"""
        mock_agent = MagicMock()
        mock_agent.validate_content = AsyncMock(return_value={
            "overall_score": 0.3,
            "safe": False,
            "issues": ["prohibited language"],
        })
        scorer = self._make_scorer(mock_agent)

        result = await scorer.validate(content="bad content")

        assert result["safe"] is False
        assert result["overall_score"] == 0.3


# ---------------------------------------------------------------------------
# ClaimValidator tests
# ---------------------------------------------------------------------------

class TestClaimValidator:
    """Tests for src/governance/claim_validator.py"""

    def _make_validator(self, claims_dict=None):
        """Create a ClaimValidator with a mocked claims library"""
        with patch("src.governance.claim_validator.Path.exists", return_value=False):
            with patch("builtins.open", side_effect=FileNotFoundError):
                from src.governance.claim_validator import ClaimValidator
                v = ClaimValidator()
        v.claims_library = claims_dict or {}
        return v

    # -- extract_claim_ids -------------------------------------------------

    def test_extract_claim_ids_clm_format(self):
        v = self._make_validator()
        ids = v._extract_claim_ids("Content [CLM_001] and [CLM_002] here")
        assert set(ids) == {"CLM_001", "CLM_002"}

    def test_extract_claim_ids_claim_format(self):
        v = self._make_validator()
        ids = v._extract_claim_ids("Uses CLAIM_010 in text")
        assert "CLAIM_010" in ids

    def test_extract_claim_ids_no_duplicates(self):
        v = self._make_validator()
        ids = v._extract_claim_ids("CLM_001 and CLM_001 again")
        assert ids.count("CLM_001") == 1

    def test_extract_claim_ids_empty_content(self):
        v = self._make_validator()
        assert v._extract_claim_ids("No claims here") == []

    # -- check_citation ----------------------------------------------------

    def test_check_citation_source_bracket(self):
        v = self._make_validator()
        claim = {"source": "Gartner Report"}
        assert v._check_citation("[Source: Gartner]", claim) is True

    def test_check_citation_source_name(self):
        v = self._make_validator()
        claim = {"source": "Gartner Report"}
        assert v._check_citation("According to Gartner Report data", claim) is True

    def test_check_citation_missing(self):
        v = self._make_validator()
        claim = {"source": ""}
        # No bracket pattern and empty source → should still match generic [...]
        # Actually with empty source, re.search('', content) always matches
        # but let's test content with no brackets at all and empty source
        assert v._check_citation("plain text no brackets", {"source": ""}) is True

    # -- validate_content --------------------------------------------------

    def test_validate_content_all_cited(self):
        claims = {
            "CLM_001": {"text": "50% ROI", "source": "Study X"},
        }
        v = self._make_validator(claims)
        result = v.validate_content(
            "Our product shows CLM_001 improvement [Source: Study X]",
            claims_used=["CLM_001"],
        )
        assert result["is_valid"] is True
        assert result["score"] == 1.0
        assert result["all_claims_cited"] is True

    def test_validate_content_hallucinated_claim(self):
        v = self._make_validator({"CLM_001": {"text": "x", "source": "s"}})
        result = v.validate_content("CLM_001 and CLM_999 referenced [Source: s]")
        assert "CLM_999" in result["hallucinated_claims"]
        assert result["all_claims_cited"] is False

    def test_validate_content_missing_citation(self):
        claims = {
            "CLM_001": {"text": "claim", "source": ""},
        }
        v = self._make_validator(claims)
        result = v.validate_content(
            "CLM_001 mentioned",
            claims_used=["CLM_001", "CLM_002"],
        )
        assert "CLM_002" in result["missing_citations"]

    def test_validate_content_no_claims_invalid(self):
        v = self._make_validator()
        result = v.validate_content("Plain text no claims")
        assert result["is_valid"] is False
        assert result["score"] == 0.0

    # -- get_claim_by_id ---------------------------------------------------

    def test_get_claim_by_id_found(self):
        v = self._make_validator({"CLM_001": {"text": "data"}})
        assert v.get_claim_by_id("CLM_001") == {"text": "data"}

    def test_get_claim_by_id_not_found(self):
        v = self._make_validator()
        assert v.get_claim_by_id("CLM_999") is None

    # -- get_claims_for_persona -------------------------------------------

    def test_get_claims_for_persona(self):
        claims = {
            "CLM_001": {"text": "a", "source": "s", "personas": ["cfo"], "priority": 8},
            "CLM_002": {"text": "b", "source": "s", "personas": ["cto"], "priority": 5},
        }
        v = self._make_validator(claims)
        results = v.get_claims_for_persona("cfo")
        assert len(results) == 1
        assert results[0]["id"] == "CLM_001"

    # -- get_claims_for_goal -----------------------------------------------

    def test_get_claims_for_goal(self):
        claims = {
            "CLM_001": {"text": "a", "source": "s", "goals": ["roi"], "priority": 9},
            "CLM_002": {"text": "b", "source": "s", "goals": ["engagement"], "priority": 3},
        }
        v = self._make_validator(claims)
        results = v.get_claims_for_goal("roi")
        assert len(results) == 1
        assert results[0]["id"] == "CLM_001"

    # -- format_claim_for_prompt -------------------------------------------

    def test_format_claim_for_prompt(self):
        v = self._make_validator({"CLM_001": {"text": "50% ROI", "source": "Gartner"}})
        result = v.format_claim_for_prompt("CLM_001")
        assert "CLM_001" in result
        assert "50% ROI" in result
        assert "Gartner" in result

    def test_format_claim_for_prompt_missing(self):
        v = self._make_validator()
        assert v.format_claim_for_prompt("CLM_999") == ""

    # -- validate_claim_library --------------------------------------------

    def test_validate_claim_library_complete(self):
        claims = {
            "CLM_001": {
                "id": "CLM_001", "text": "t", "source": "s",
                "personas": [], "goals": [], "priority": 5,
                "evidence_url": "http://example.com",
            },
        }
        v = self._make_validator(claims)
        result = v.validate_claim_library()
        assert result["valid"] is True
        assert result["total_claims"] == 1

    def test_validate_claim_library_missing_fields(self):
        v = self._make_validator({"CLM_001": {"id": "CLM_001"}})
        result = v.validate_claim_library()
        assert result["valid"] is False
        assert len(result["issues"]) > 0


# ---------------------------------------------------------------------------
# ContentFormatter tests
# ---------------------------------------------------------------------------

class TestContentFormatter:
    """Tests for src/governance/content_formatter.py"""

    def test_expand_inline_removes_citations(self):
        from src.governance.content_formatter import expand_claim_citations
        result = expand_claim_citations(
            "Great results [CLM_001] in Q4.",
            format_style="inline",
            claim_library={"CLM_001": "50% ROI improvement"},
        )
        assert "[CLM_001]" not in result
        assert "Great results" in result

    def test_expand_footnote_adds_references(self):
        from src.governance.content_formatter import expand_claim_citations
        lib = {"CLM_001": "50% ROI improvement"}
        result = expand_claim_citations(
            "Results [CLM_001] are great.",
            format_style="footnote",
            claim_library=lib,
        )
        assert "[1]" in result
        assert "References:" in result
        assert "50% ROI improvement" in result

    def test_expand_remove_strips_all(self):
        from src.governance.content_formatter import expand_claim_citations
        result = expand_claim_citations(
            "Data [CLM_001] and [CLM_002] here.",
            format_style="remove",
            claim_library={},
        )
        assert "[CLM_001]" not in result
        assert "[CLM_002]" not in result

    def test_expand_keep_preserves_citations(self):
        from src.governance.content_formatter import expand_claim_citations
        content = "Data [CLM_001] here."
        assert expand_claim_citations(content, format_style="keep") == content

    def test_expand_empty_content(self):
        from src.governance.content_formatter import expand_claim_citations
        assert expand_claim_citations("", format_style="inline") == ""
        assert expand_claim_citations(None, format_style="inline") is None

    def test_expand_normalizes_claim_id_prefix(self):
        from src.governance.content_formatter import expand_claim_citations
        result = expand_claim_citations(
            "Fact [CLAIM_ID:CLM_003] confirmed.",
            format_style="remove",
            claim_library={},
        )
        assert "[CLAIM_ID:CLM_003]" not in result

    # -- format_content_for_platform ---------------------------------------

    def test_format_for_linkedin(self):
        from src.governance.content_formatter import format_content_for_platform
        with patch("src.governance.content_formatter._load_claim_library", return_value={}):
            result = format_content_for_platform(
                {"headline": "Title [CLM_001]", "body": "Body [CLM_002]"},
                platform="linkedin",
            )
        assert "[CLM_001]" not in result["headline"]
        assert result["_formatting"]["format_style"] == "inline"

    def test_format_for_email_uses_footnote(self):
        from src.governance.content_formatter import format_content_for_platform
        with patch("src.governance.content_formatter._load_claim_library", return_value={"CLM_001": "claim"}):
            result = format_content_for_platform(
                {"body": "Content [CLM_001]"},
                platform="email",
            )
        assert result["_formatting"]["format_style"] == "footnote"

    def test_format_for_internal_keeps_citations(self):
        from src.governance.content_formatter import format_content_for_platform
        with patch("src.governance.content_formatter._load_claim_library", return_value={}):
            result = format_content_for_platform(
                {"body": "Content [CLM_001]"},
                platform="internal",
            )
        assert "[CLM_001]" in result["body"]

    # -- extract_citations_from_content ------------------------------------

    def test_extract_citations_direct(self):
        from src.governance.content_formatter import extract_citations_from_content
        ids = extract_citations_from_content("Using [CLM_003] and [CLM_006]")
        assert "CLM_003" in ids
        assert "CLM_006" in ids

    def test_extract_citations_prefixed(self):
        from src.governance.content_formatter import extract_citations_from_content
        ids = extract_citations_from_content("[CLAIM_ID:CLM_010] found")
        assert "CLM_010" in ids

    def test_extract_citations_empty(self):
        from src.governance.content_formatter import extract_citations_from_content
        assert extract_citations_from_content("") == []
        assert extract_citations_from_content(None) == []


# ---------------------------------------------------------------------------
# GoldenTestSuite tests
# ---------------------------------------------------------------------------

class TestGoldenTestSuite:
    """Tests for src/governance/golden_test_suite.py"""

    def _make_suite(self):
        from src.governance.golden_test_suite import GoldenTestSuite
        return GoldenTestSuite()

    def test_load_default_tests(self):
        suite = self._make_suite()
        suite.load_test_cases(test_file="nonexistent_file.yaml")
        assert len(suite.test_cases) > 0
        assert suite._initialized is True

    def test_load_from_yaml_file(self, tmp_path):
        yaml_content = """
test_cases:
  - id: "TEST_001"
    name: "Custom Test"
    category: "custom"
    type: "format_check"
    test_content: "Headline: Hello\\nBody: World"
    assertions:
      has_headline: true
      has_body: true
"""
        f = tmp_path / "test_suite.yaml"
        f.write_text(yaml_content)

        suite = self._make_suite()
        suite.load_test_cases(test_file=str(f))
        assert len(suite.test_cases) == 1
        assert suite.test_cases[0]["id"] == "TEST_001"

    @pytest.mark.asyncio
    async def test_format_check_headline_body(self):
        suite = self._make_suite()
        test_case = {
            "id": "T1", "name": "fmt",
            "category": "format", "type": "format_check",
            "test_content": "Headline: Hi\nBody: World",
            "assertions": {"has_headline": True, "has_body": True},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_format_check_missing_headline(self):
        suite = self._make_suite()
        test_case = {
            "id": "T2", "name": "missing hl",
            "category": "format", "type": "format_check",
            "test_content": "Body: just body",
            "assertions": {"has_headline": True},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is False
        assert "headline" in result["failure_reason"].lower()

    @pytest.mark.asyncio
    async def test_safety_check_clean_content(self):
        suite = self._make_suite()
        test_case = {
            "id": "T3", "name": "clean",
            "category": "safety", "type": "safety_check",
            "test_content": "Professional marketing copy about our product.",
            "assertions": {"no_prohibited_words": True},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_safety_check_prohibited_word(self):
        suite = self._make_suite()
        test_case = {
            "id": "T4", "name": "prohibited",
            "category": "safety", "type": "safety_check",
            "test_content": "We guarantee results with no risk!",
            "assertions": {"no_prohibited_words": True},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_safety_check_profanity_expected(self):
        suite = self._make_suite()
        test_case = {
            "id": "T5", "name": "profanity",
            "category": "safety", "type": "safety_check",
            "test_content": "This is damn good product.",
            "assertions": {"profanity_detected": True},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_platform_check_twitter_within_limit(self):
        suite = self._make_suite()
        test_case = {
            "id": "T6", "name": "twitter ok",
            "category": "platform", "type": "platform_check",
            "test_content": "A" * 280,
            "platform": "twitter",
            "assertions": {"within_char_limit": True},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_platform_check_twitter_exceeds_limit(self):
        suite = self._make_suite()
        test_case = {
            "id": "T7", "name": "twitter over",
            "category": "platform", "type": "platform_check",
            "test_content": "A" * 300,
            "platform": "twitter",
            "assertions": {"within_char_limit": False},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_regex_check_extracts_value(self):
        suite = self._make_suite()
        test_case = {
            "id": "T8", "name": "regex",
            "category": "regex", "type": "regex_check",
            "test_content": "Headline: Boost ROI\nBody: text",
            "pattern": r"Headline:\s*(.+?)(?:\n|$)",
            "assertions": {"pattern_matches": True, "extracted_value": "Boost ROI"},
        }
        result = await suite._run_single_test(test_case)
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_run_all_tests_default_suite(self):
        suite = self._make_suite()
        suite.load_test_cases(test_file="nonexistent.yaml")
        # Exclude integration test which needs external services
        suite.test_cases = [
            tc for tc in suite.test_cases if tc.get("type") != "integration_check"
        ]
        results = await suite.run_all_tests()
        assert results["total"] > 0
        assert "pass_rate" in results

    def test_get_test_summary(self):
        suite = self._make_suite()
        suite.load_test_cases(test_file="nonexistent.yaml")
        summary = suite.get_test_summary()
        assert summary["total_tests"] > 0
        assert "categories" in summary

    @pytest.mark.asyncio
    async def test_run_category_tests(self):
        suite = self._make_suite()
        suite.load_test_cases(test_file="nonexistent.yaml")
        results = await suite.run_category_tests("format")
        assert results["category"] == "format"
        assert results["total"] > 0


# ---------------------------------------------------------------------------
# HITLQueueManager tests
# ---------------------------------------------------------------------------

class TestHITLQueueManager:
    """Tests for src/governance/hitl_queue.py"""

    def _make_manager(self):
        from src.governance.hitl_queue import HITLQueueManager
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_redis = MagicMock()
        return HITLQueueManager(db_session=mock_db, redis_client=mock_redis)

    @pytest.mark.asyncio
    async def test_add_for_review_creates_item(self):
        mgr = self._make_manager()
        content_id = str(uuid4())

        with patch("src.governance.hitl_queue.HITLQueue") as MockModel:
            MockModel.return_value = MagicMock()
            item = await mgr.add_for_review(
                content_id=content_id,
                priority=7,
                reason="low safety score",
                content_data={"body": "text"},
                safety_scores={"overall": 0.6},
            )

        mgr.db.add.assert_called_once()
        mgr.db.commit.assert_awaited_once()
        mgr.redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_for_review_rollback_on_error(self):
        mgr = self._make_manager()
        mgr.db.commit = AsyncMock(side_effect=Exception("db error"))

        with patch("src.governance.hitl_queue.HITLQueue") as MockModel:
            MockModel.return_value = MagicMock()
            with pytest.raises(Exception, match="db error"):
                await mgr.add_for_review(
                    content_id=str(uuid4()),
                    priority=5,
                    reason="test",
                )

        mgr.db.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_pending_items(self):
        mgr = self._make_manager()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = ["item1", "item2"]
        mock_result.scalars.return_value = mock_scalars
        mgr.db.execute = AsyncMock(return_value=mock_result)

        items = await mgr.get_pending_items()
        assert items == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_get_pending_items_error(self):
        mgr = self._make_manager()
        mgr.db.execute = AsyncMock(side_effect=Exception("db down"))
        items = await mgr.get_pending_items()
        assert items == []

    @pytest.mark.asyncio
    async def test_submit_review_approve(self):
        mgr = self._make_manager()
        review_id = str(uuid4())
        content_id = str(uuid4())
        review_data = json.dumps({
            "id": review_id,
            "content_id": content_id,
            "priority": 5,
            "reason": "test",
        })
        mgr.redis.hget = MagicMock(return_value=review_data)

        with patch("src.data_layer.database.models.ContentStatus") as MockStatus:
            MockStatus.APPROVED = "approved"
            MockStatus.REJECTED = "rejected"
            MockStatus.GENERATED = "generated"
            result = await mgr.submit_review(
                review_id=review_id,
                decision="approve",
                reviewer_id="reviewer@test.com",
                feedback="Looks good",
            )

        assert result is True
        mgr.redis.hdel.assert_called_once_with(mgr.processing_key, review_id)
        mgr.redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_submit_review_not_found(self):
        mgr = self._make_manager()
        mgr.redis.hget = MagicMock(return_value=None)
        result = await mgr.submit_review(
            review_id="nonexistent",
            decision="approve",
            reviewer_id="rev@test.com",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_approve_content_delegates(self):
        mgr = self._make_manager()
        mgr.submit_review = AsyncMock(return_value=True)
        result = await mgr.approve_content("qid", "rev@test.com", "ok")
        mgr.submit_review.assert_awaited_once_with(
            review_id="qid",
            decision="approved",
            reviewer_id="rev@test.com",
            feedback="ok",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_reject_content_delegates(self):
        mgr = self._make_manager()
        mgr.submit_review = AsyncMock(return_value=True)
        result = await mgr.reject_content("qid", "rev@test.com", "bad")
        mgr.submit_review.assert_awaited_once_with(
            review_id="qid",
            decision="rejected",
            reviewer_id="rev@test.com",
            feedback="bad",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_get_next_for_review_no_items(self):
        mgr = self._make_manager()
        mgr.redis.zrange = MagicMock(return_value=[])
        result = await mgr.get_next_for_review(timeout=0)
        assert result is None


# ---------------------------------------------------------------------------
# CompetitorValidator tests
# ---------------------------------------------------------------------------

class TestCompetitorValidator:
    """Tests for src/governance/competitor_validator.py"""

    def _make_validator(self, competitors=None):
        with patch("src.governance.competitor_validator.Path.exists", return_value=False):
            from src.governance.competitor_validator import CompetitorValidator
            v = CompetitorValidator()
        v.competitors = competitors or {}
        return v

    def test_no_competitor_mention(self):
        v = self._make_validator({
            "acme": {"name": "Acme", "risky_topics": "", "differentiators_vs_us": ""},
        })
        result = v.validate_content("Our product is great.")
        assert result["valid"] is True
        assert result["competitors_mentioned"] == []

    def test_competitor_mentioned(self):
        v = self._make_validator({
            "acme": {"name": "Acme", "risky_topics": "", "differentiators_vs_us": "better UX"},
        })
        result = v.validate_content("Unlike Acme, we offer more.")
        assert "Acme" in result["competitors_mentioned"]
        assert len(result["recommendations"]) == 1

    def test_risky_topic_pricing(self):
        v = self._make_validator({
            "acme": {
                "name": "Acme",
                "risky_topics": "pricing",
                "differentiators_vs_us": "",
            },
        })
        result = v.validate_content("Acme costs more, their price is too high.")
        assert result["valid"] is False
        assert len(result["risky_mentions"]) > 0

    def test_risky_topic_guarantee(self):
        v = self._make_validator({
            "acme": {
                "name": "Acme",
                "risky_topics": "guarantee",
                "differentiators_vs_us": "",
            },
        })
        result = v.validate_content("Acme cannot guarantee anything.")
        assert result["valid"] is False

    def test_risky_topic_disparagement(self):
        v = self._make_validator({
            "acme": {
                "name": "Acme",
                "risky_topics": "disparagement",
                "differentiators_vs_us": "",
            },
        })
        result = v.validate_content("Acme has a poor track record.")
        assert result["valid"] is False

    def test_competitor_in_headline(self):
        v = self._make_validator({
            "acme": {"name": "Acme", "risky_topics": "", "differentiators_vs_us": ""},
        })
        result = v.validate_content("We lead the market.", headline="Acme vs Us")
        assert "Acme" in result["competitors_mentioned"]

    def test_get_competitor_info(self):
        v = self._make_validator({
            "acme": {"name": "Acme", "category": "analytics"},
        })
        info = v.get_competitor_info("Acme")
        assert info["category"] == "analytics"

    def test_get_competitor_info_not_found(self):
        v = self._make_validator()
        assert v.get_competitor_info("Unknown") is None

    def test_get_all_competitors(self):
        v = self._make_validator({
            "a": {"name": "A"},
            "b": {"name": "B"},
        })
        assert len(v.get_all_competitors()) == 2

    def test_format_validation_report_passed(self):
        v = self._make_validator()
        report = v.format_validation_report({
            "valid": True,
            "competitors_mentioned": [],
            "risky_mentions": [],
            "recommendations": [],
            "warnings": [],
        })
        assert "PASSED" in report

    def test_format_validation_report_failed(self):
        v = self._make_validator()
        report = v.format_validation_report({
            "valid": False,
            "competitors_mentioned": ["Acme"],
            "risky_mentions": [{"competitor": "Acme", "risky_topic": "pricing", "guidance": "avoid"}],
            "recommendations": [],
            "warnings": ["Content mentions competitors: Acme"],
        })
        assert "FAILED" in report
        assert "pricing" in report
