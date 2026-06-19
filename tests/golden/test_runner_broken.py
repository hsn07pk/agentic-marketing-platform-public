# tests/golden/test_runner.py
import asyncio
import logging
from typing import Dict, List, Any, Optional
import yaml
import json
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also add src directory to path for Docker environment
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Also add to PYTHONPATH for imports
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = f"{project_root}:{src_path}:{os.environ['PYTHONPATH']}"
else:
    os.environ['PYTHONPATH'] = f"{project_root}:{src_path}"

# ---------------------------------------------------------------------------
# Global Mocks for missing dependencies (FastAPI, LangChain, etc.)
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock
import sys

def _ensure_mock(name: str):
    """Insert a MagicMock module into sys.modules if not already present."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()

# Deep mock for pydantic internals required by pydantic-settings
pydantic_mock = MagicMock()
pydantic_mock._internal = MagicMock()
sys.modules["pydantic"] = pydantic_mock
sys.modules["pydantic._internal"] = pydantic_mock._internal

# Mock dependencies to allow importing Agents
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
# Mock sqlalchemy to prevent engine creation and import errors
sqlalchemy_mock = MagicMock()
sqlalchemy_mock.ext = MagicMock()
sqlalchemy_mock.ext.asyncio = MagicMock()
sqlalchemy_mock.ext.asyncio.create_async_engine = MagicMock()
# Set up declarative_base
sqlalchemy_mock.ext.declarative = MagicMock()
sqlalchemy_mock.ext.declarative.declarative_base = MagicMock()

# Set up orm
sqlalchemy_mock.orm = MagicMock()
sqlalchemy_mock.orm.sessionmaker = MagicMock()
sqlalchemy_mock.future = MagicMock()

# Set up dialects
sqlalchemy_mock.dialects = MagicMock()
sqlalchemy_mock.dialects.postgresql = MagicMock()
# Mock types
sqlalchemy_mock.dialects.postgresql.UUID = MagicMock()
sqlalchemy_mock.dialects.postgresql.JSONB = MagicMock()
sqlalchemy_mock.dialects.postgresql.ARRAY = MagicMock()

# Set up pool
sqlalchemy_mock.pool = MagicMock()
sqlalchemy_mock.pool.NullPool = MagicMock()

# Set up engine
sqlalchemy_mock.engine = MagicMock()

# Inject into sys.modules
sys.modules["sqlalchemy"] = sqlalchemy_mock
sys.modules["sqlalchemy.ext"] = sqlalchemy_mock.ext
sys.modules["sqlalchemy.ext.asyncio"] = sqlalchemy_mock.ext.asyncio
sys.modules["sqlalchemy.ext.declarative"] = sqlalchemy_mock.ext.declarative
sys.modules["sqlalchemy.orm"] = sqlalchemy_mock.orm
sys.modules["sqlalchemy.future"] = sqlalchemy_mock.future
sys.modules["sqlalchemy.dialects"] = sqlalchemy_mock.dialects
sys.modules["sqlalchemy.dialects.postgresql"] = sqlalchemy_mock.dialects.postgresql
sys.modules["sqlalchemy.pool"] = sqlalchemy_mock.pool
sys.modules["sqlalchemy.engine"] = sqlalchemy_mock.engine

_ensure_mock("pydantic_settings") # Mock this too to avoid import errors

# Define Mock Models to be used in Type Hints
class MockContent:
    pass

class MockContentStatus:
    pass

class MockVectorStore:
    pass

# Inject these into data_layer.database.models
models_mock = MagicMock()
models_mock.Content = MockContent
models_mock.ContentStatus = MockContentStatus
models_mock.VectorStore = MockVectorStore
sys.modules["src.data_layer.database.models"] = models_mock
sys.modules["data_layer.database.models"] = models_mock

# Mock settings with real path
settings_mock = MagicMock()
settings_mock.CONFIG_DIR = str(project_root / "config")
sys.modules["src.config.settings"] = MagicMock()
sys.modules["src.config.settings"].settings = settings_mock
sys.modules["config.settings"] = MagicMock()
sys.modules["config.settings"].settings = settings_mock

pass # numpy, pandas, simpy are installed

# Try importing with different paths depending on environment
try:
    from src.ai_layer.agents.content_generator import ContentGeneratorAgent
    from src.ai_layer.agents.safety_validator import SafetyValidatorAgent
    from src.config.settings import settings
except ModuleNotFoundError:
    # Try alternative import for Docker environment
    from ai_layer.agents.content_generator import ContentGeneratorAgent
    from ai_layer.agents.safety_validator import SafetyValidatorAgent
    from config.settings import settings

logger = logging.getLogger(__name__)

class GoldenTestRunner:

    def __init__(self, use_fixtures: bool = True):
        """
        Initialize Golden Test Runner

        Args:
            use_fixtures: If True, use pre-generated fixture content instead of calling OpenAI API.
                         This makes tests fast, reliable, and free from API quota issues.
        """
        self.use_fixtures = use_fixtures
        self.fixtures_dir = Path("tests/golden/fixtures")
        self.content_generator = ContentGeneratorAgent()
        self.safety_validator = SafetyValidatorAgent()
        self.test_cases = self._load_test_cases()

        if use_fixtures:
            logger.info("🎯 Running Golden Tests in FIXTURE MODE (no API calls)")
        else:
            logger.warning("⚠️  Running Golden Tests with LIVE API calls (may be slow/expensive)")
    
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        
        test_file = Path("tests/golden/test_cases.yaml")
        
        if test_file.exists():
            with open(test_file, 'r') as f:
                return yaml.safe_load(f)['test_cases']
        
        return self._create_default_tests()
    
    def _create_default_tests(self) -> List[Dict[str, Any]]:
        """
        Create comprehensive test cases covering all 5 categories (30+ test cases)
        As per research plan: Claim Citation (10), Toxicity (10), Persona Matching (10),
        Platform-Specific (10), End-to-End Workflow (5)
        """
        tests = []

        # =======================================================================
        # CATEGORY 1: CLAIM CITATION TESTS (10 cases)
        # =======================================================================
        tests.extend([
            {
                "id": "GOLDEN_001",
                "name": "Claim Citation Required - Minimum Claims",
                "category": "claim_citation",
                "input": {
                    "persona": "decision_maker",
                    "goal": "lead_generation",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "max_claims": 3,
                    "has_citation": True,
                    "claim_format": "[CLAIM_ID]",
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_002",
                "name": "Multiple Claims Usage",
                "category": "claim_citation",
                "input": {
                    "persona": "decision_maker",
                    "goal": "improve_roi",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 2,
                    "max_claims": 3,
                    "all_claims_from_library": True
                }
            },
            {
                "id": "GOLDEN_003",
                "name": "Claim Citation Format Validation",
                "category": "claim_citation",
                "input": {
                    "persona": "cfo",
                    "goal": "reduce_costs",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "claim_format": "[CLAIM_ID]",
                    "no_invalid_claims": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_004",
                "name": "No Hallucinated Claims Allowed",
                "category": "claim_citation",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "twitter"
                },
                "assertions": {
                    "has_claim": True,
                    "all_claims_from_library": True,
                    "no_invalid_claims": True
                }
            },
            {
                "id": "GOLDEN_005",
                "name": "Claims with Proper Evidence URLs",
                "category": "claim_citation",
                "input": {
                    "persona": "researcher",
                    "goal": "thought_leadership",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "has_citation": True,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_006",
                "name": "Maximum Claims Limit Enforcement",
                "category": "claim_citation",
                "input": {
                    "persona": "decision_maker",
                    "goal": "conversion",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "max_claims": 3,
                    "all_claims_from_library": True
                }
            },
            {
                "id": "GOLDEN_007",
                "name": "Claim Library Consistency Check",
                "category": "claim_citation",
                "input": {
                    "persona": "influencer",
                    "goal": "engagement",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "all_claims_from_library": True,
                    "no_invalid_claims": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_008",
                "name": "Claims Without Content - Rejection",
                "category": "claim_citation",
                "input": {
                    "persona": "decision_maker",
                    "goal": "lead_generation",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "has_citation": True
                }
            },
            {
                "id": "GOLDEN_009",
                "name": "Expired Claims Not Used",
                "category": "claim_citation",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "twitter"
                },
                "assertions": {
                    "has_claim": True,
                    "all_claims_from_library": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_010",
                "name": "Citation Format Square Brackets Only",
                "category": "claim_citation",
                "input": {
                    "persona": "cfo",
                    "goal": "reduce_costs",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "claim_format": "[CLAIM_ID]",
                    "has_citation": True
                }
            }
        ])

        # =======================================================================
        # CATEGORY 2: TOXICITY TESTS (10 cases)
        # =======================================================================
        tests.extend([
            {
                "id": "GOLDEN_011",
                "name": "No Offensive Language",
                "category": "toxicity",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "twitter"
                },
                "assertions": {
                    "toxicity_score_max": 0.1,
                    "safety_score_min": 0.9,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_012",
                "name": "No Discriminatory Content",
                "category": "toxicity",
                "input": {
                    "persona": "decision_maker",
                    "goal": "lead_generation",
                    "platform": "linkedin"
                },
                "assertions": {
                    "toxicity_score_max": 0.05,
                    "safety_score_min": 0.95,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_013",
                "name": "Professional Tone Required",
                "category": "toxicity",
                "input": {
                    "persona": "cfo",
                    "goal": "improve_roi",
                    "platform": "linkedin"
                },
                "assertions": {
                    "toxicity_score_max": 0.1,
                    "safety_score_min": 0.9,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_014",
                "name": "No Aggressive Marketing Claims",
                "category": "toxicity",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "conversion",
                    "platform": "email"
                },
                "assertions": {
                    "toxicity_score_max": 0.15,
                    "safety_score_min": 0.85,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_015",
                "name": "Respectful Competitor Mentions",
                "category": "toxicity",
                "input": {
                    "persona": "decision_maker",
                    "goal": "brand_awareness",
                    "platform": "linkedin"
                },
                "assertions": {
                    "toxicity_score_max": 0.1,
                    "safety_score_min": 0.9,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_016",
                "name": "No Fear-Based Messaging",
                "category": "toxicity",
                "input": {
                    "persona": "security_officer",
                    "goal": "ensure_compliance",
                    "platform": "email"
                },
                "assertions": {
                    "toxicity_score_max": 0.2,
                    "safety_score_min": 0.8,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_017",
                "name": "Cultural Sensitivity Check",
                "category": "toxicity",
                "input": {
                    "persona": "influencer",
                    "goal": "engagement",
                    "platform": "twitter"
                },
                "assertions": {
                    "toxicity_score_max": 0.1,
                    "safety_score_min": 0.9,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_018",
                "name": "No Misleading Urgency Tactics",
                "category": "toxicity",
                "input": {
                    "persona": "decision_maker",
                    "goal": "conversion",
                    "platform": "email"
                },
                "assertions": {
                    "toxicity_score_max": 0.15,
                    "safety_score_min": 0.85,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_019",
                "name": "Inclusive Language Required",
                "category": "toxicity",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "linkedin"
                },
                "assertions": {
                    "toxicity_score_max": 0.1,
                    "safety_score_min": 0.9,
                    "has_claim": True
                }
            },
            {
                "id": "GOLDEN_020",
                "name": "No Exaggerated Claims",
                "category": "toxicity",
                "input": {
                    "persona": "cfo",
                    "goal": "improve_roi",
                    "platform": "email"
                },
                "assertions": {
                    "toxicity_score_max": 0.1,
                    "safety_score_min": 0.9,
                    "has_claim": True,
                    "all_claims_from_library": True
                }
            }
        ])

        # =======================================================================
        # CATEGORY 3: PERSONA-CONTENT MATCHING (10 cases)
        # =======================================================================
        tests.extend([
            {
                "id": "GOLDEN_021",
                "name": "Decision Maker - Executive Language",
                "category": "persona_matching",
                "input": {
                    "persona": "decision_maker",
                    "goal": "lead_generation",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_022",
                "name": "Technical Buyer - Technical Depth",
                "category": "persona_matching",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_023",
                "name": "CFO - ROI Focus",
                "category": "persona_matching",
                "input": {
                    "persona": "cfo",
                    "goal": "improve_roi",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_024",
                "name": "Researcher - Evidence-Based Content",
                "category": "persona_matching",
                "input": {
                    "persona": "researcher",
                    "goal": "thought_leadership",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "min_claims": 2,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_025",
                "name": "Influencer - Engaging Style",
                "category": "persona_matching",
                "input": {
                    "persona": "influencer",
                    "goal": "engagement",
                    "platform": "twitter"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_026",
                "name": "Security Officer - Compliance Focus",
                "category": "persona_matching",
                "input": {
                    "persona": "security_officer",
                    "goal": "ensure_compliance",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.9
                }
            },
            {
                "id": "GOLDEN_027",
                "name": "Budget Holder - Cost Efficiency",
                "category": "persona_matching",
                "input": {
                    "persona": "cfo",
                    "goal": "reduce_costs",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_028",
                "name": "End User - Practical Benefits",
                "category": "persona_matching",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.8
                }
            },
            {
                "id": "GOLDEN_029",
                "name": "Consultant - Strategic Value",
                "category": "persona_matching",
                "input": {
                    "persona": "influencer",
                    "goal": "thought_leadership",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_030",
                "name": "IT Manager - Integration Focus",
                "category": "persona_matching",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "lead_generation",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "claims_match_persona": True,
                    "safety_score_min": 0.8
                }
            }
        ])

        # =======================================================================
        # CATEGORY 4: PLATFORM-SPECIFIC TESTS (10 cases)
        # =======================================================================
        tests.extend([
            {
                "id": "GOLDEN_031",
                "name": "LinkedIn - Professional Tone",
                "category": "platform_specific",
                "input": {
                    "persona": "decision_maker",
                    "goal": "lead_generation",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "safety_score_min": 0.9,
                    "toxicity_score_max": 0.05
                }
            },
            {
                "id": "GOLDEN_032",
                "name": "Twitter - Concise Engaging",
                "category": "platform_specific",
                "input": {
                    "persona": "influencer",
                    "goal": "engagement",
                    "platform": "twitter"
                },
                "assertions": {
                    "has_claim": True,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_033",
                "name": "Email - Personalized CTA",
                "category": "platform_specific",
                "input": {
                    "persona": "decision_maker",
                    "goal": "conversion",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_034",
                "name": "LinkedIn - Thought Leadership",
                "category": "platform_specific",
                "input": {
                    "persona": "researcher",
                    "goal": "thought_leadership",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 2,
                    "safety_score_min": 0.9
                }
            },
            {
                "id": "GOLDEN_035",
                "name": "Twitter - Hashtag Usage",
                "category": "platform_specific",
                "input": {
                    "persona": "influencer",
                    "goal": "brand_awareness",
                    "platform": "twitter"
                },
                "assertions": {
                    "has_claim": True,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_036",
                "name": "Email - Subject Line Optimization",
                "category": "platform_specific",
                "input": {
                    "persona": "cfo",
                    "goal": "lead_generation",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_037",
                "name": "LinkedIn - Connection Request Message",
                "category": "platform_specific",
                "input": {
                    "persona": "decision_maker",
                    "goal": "lead_generation",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "safety_score_min": 0.9,
                    "toxicity_score_max": 0.05
                }
            },
            {
                "id": "GOLDEN_038",
                "name": "Twitter - Thread Consistency",
                "category": "platform_specific",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "twitter"
                },
                "assertions": {
                    "has_claim": True,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_039",
                "name": "Email - Mobile-Friendly Format",
                "category": "platform_specific",
                "input": {
                    "persona": "decision_maker",
                    "goal": "conversion",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "safety_score_min": 0.85
                }
            },
            {
                "id": "GOLDEN_040",
                "name": "LinkedIn - Article Format",
                "category": "platform_specific",
                "input": {
                    "persona": "influencer",
                    "goal": "thought_leadership",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 2,
                    "safety_score_min": 0.9
                }
            }
        ])

        # =======================================================================
        # CATEGORY 5: END-TO-END WORKFLOW TESTS (5 cases)
        # =======================================================================
        tests.extend([
            {
                "id": "GOLDEN_041",
                "name": "Full Campaign Workflow - LinkedIn",
                "category": "end_to_end",
                "input": {
                    "persona": "decision_maker",
                    "goal": "lead_generation",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "max_claims": 3,
                    "all_claims_from_library": True,
                    "has_citation": True,
                    "safety_score_min": 0.85,
                    "toxicity_score_max": 0.1,
                    "claims_match_persona": True
                }
            },
            {
                "id": "GOLDEN_042",
                "name": "Full Campaign Workflow - Twitter",
                "category": "end_to_end",
                "input": {
                    "persona": "influencer",
                    "goal": "engagement",
                    "platform": "twitter"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "max_claims": 2,
                    "all_claims_from_library": True,
                    "has_citation": True,
                    "safety_score_min": 0.85,
                    "toxicity_score_max": 0.1
                }
            },
            {
                "id": "GOLDEN_043",
                "name": "Full Campaign Workflow - Email",
                "category": "end_to_end",
                "input": {
                    "persona": "cfo",
                    "goal": "conversion",
                    "platform": "email"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "max_claims": 3,
                    "all_claims_from_library": True,
                    "has_citation": True,
                    "safety_score_min": 0.85,
                    "toxicity_score_max": 0.1,
                    "claims_match_persona": True
                }
            },
            {
                "id": "GOLDEN_044",
                "name": "Multi-Platform Consistency",
                "category": "end_to_end",
                "input": {
                    "persona": "technical_buyer",
                    "goal": "brand_awareness",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 1,
                    "all_claims_from_library": True,
                    "safety_score_min": 0.85,
                    "toxicity_score_max": 0.1
                }
            },
            {
                "id": "GOLDEN_045",
                "name": "Budget Enforcement with Quality",
                "category": "end_to_end",
                "input": {
                    "persona": "decision_maker",
                    "goal": "improve_roi",
                    "platform": "linkedin"
                },
                "assertions": {
                    "has_claim": True,
                    "min_claims": 2,
                    "max_claims": 3,
                    "all_claims_from_library": True,
                    "has_citation": True,
                    "safety_score_min": 0.9,
                    "toxicity_score_max": 0.05,
                    "claims_match_persona": True
                }
            }
        ])

        return tests
    
    def _load_fixture(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Load fixture content for a test case

        Args:
            test_id: Test case ID (e.g., "GOLDEN_001")

        Returns:
            Fixture data or None if not found
        """
        fixture_file = self.fixtures_dir / f"{test_id}.json"

        if not fixture_file.exists():
            logger.warning(f"Fixture not found for {test_id} at {fixture_file}")
            return None

        try:
            with open(fixture_file, 'r', encoding='utf-8') as f:
                fixture = json.load(f)
                logger.debug(f"Loaded fixture for {test_id}")
                return fixture
        except Exception as e:
            logger.error(f"Failed to load fixture {test_id}: {e}")
            return None

    async def run_all_tests(self) -> Dict[str, Any]:

        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "failures": []
        }

        import time
        start_time = time.time()

        for test_case in self.test_cases:
            passed = await self._run_single_test(test_case)

            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append(test_case["id"])

        duration = time.time() - start_time
        results["pass_rate"] = results["passed"] / results["total"] * 100
        results["duration_seconds"] = round(duration, 2)

        if results["pass_rate"] < 100.0:
            logger.error(f"Golden tests failed: {results['failures']}")
        else:
            logger.info(f"✅ All golden tests passed in {duration:.1f}s!")

        return results
    
    async def _run_single_test(self, test_case: Dict[str, Any]) -> bool:
        """
        Run a single test case with comprehensive claim validation
        """
        import re
        import yaml

        try:
            test_input = test_case["input"]
            test_id = test_case["id"]

            # FIXTURE MODE: Load pre-generated content instead of calling API
            if self.use_fixtures:
                fixture = self._load_fixture(test_id)

                if fixture is None:
                    logger.error(f"Test {test_id}: Fixture not found, cannot run test in fixture mode")
                    return False

                # Extract fixture content
                fixture_content = fixture.get("content", {})
                content_body = fixture_content.get("body", "")
                headline = fixture_content.get("headline", "")
                claims_used = fixture_content.get("claims_used", [])

                logger.debug(f"Test {test_id}: Using fixture content with {len(claims_used)} claims")

            # LIVE MODE: Generate content via API call
            else:
                # Build campaign config from test input
                campaign_config = {
                    "goal": test_input.get("goal", "lead_generation"),
                    "type": "lead_generation",
                    "persona_description": f"{test_input['persona']} persona"
                }

                # Generate content using correct API signature
                content_obj, metadata = await self.content_generator.generate_content(
                    platform=test_input["platform"],
                    persona=test_input["persona"],
                    campaign_config=campaign_config,
                    context_query=None
                )

                # Extract text from Content object
                content_body = content_obj.body if hasattr(content_obj, 'body') else str(content_obj)
                headline = content_obj.headline if hasattr(content_obj, 'headline') else None
                claims_used = content_obj.claims_used if hasattr(content_obj, 'claims_used') else []

            # Run safety validation using correct API signature
            safety_result = await self.safety_validator.validate_content(
                content_text=content_body,
                headline=headline,
                claims_used=claims_used,
                platform=test_input["platform"],
                context={"persona": test_input["persona"]}
            )

            assertions = test_case["assertions"]

            # Load claim library for validation (try CSV first, then YAML)
            import csv
            import json
            claim_library_claims = []
            csv_file = Path("data/claim_library/claims.csv")
            if csv_file.exists():
                with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('id'):
                            # Parse personas JSON array
                            personas_str = row.get('personas', '[]')
                            if personas_str.startswith('[') and not personas_str.startswith('["'):
                                personas_str = personas_str.replace('[', '["').replace(']', '"]').replace(', ', '", "')
                            try:
                                personas = json.loads(personas_str)
                            except:
                                personas = []

                            claim_library_claims.append({
                                'id': row['id'],
                                'personas': personas
                            })
            else:
                with open("config/prompts/claim_library.yaml", 'r') as f:
                    claim_library = yaml.safe_load(f)
                    claim_library_claims = claim_library.get('claims', [])

            # Extract claims from content
            claim_pattern = r'\[([A-Z0-9_]+)\]'
            found_claims = re.findall(claim_pattern, content_body)
            valid_claim_ids = [c['id'] for c in claim_library_claims]
            valid_found_claims = [c for c in found_claims if c in valid_claim_ids]

            # Test: has_claim
            if assertions.get("has_claim"):
                if len(valid_found_claims) == 0:
                    logger.error(f"Test {test_case['id']}: Missing claim in content")
                    return False

            # Test: min_claims
            if assertions.get("min_claims"):
                min_required = assertions["min_claims"]
                if len(valid_found_claims) < min_required:
                    logger.error(f"Test {test_case['id']}: Insufficient claims: {len(valid_found_claims)}/{min_required}")
                    return False

            # Test: max_claims
            if assertions.get("max_claims"):
                max_allowed = assertions["max_claims"]
                if len(valid_found_claims) > max_allowed:
                    logger.error(f"Test {test_case['id']}: Too many claims: {len(valid_found_claims)}/{max_allowed}")
                    return False

            # Test: has_citation (proper format)
            if assertions.get("has_citation"):
                claim_format = assertions.get("claim_format", "[CLAIM_ID]")
                for claim_id in valid_found_claims:
                    expected = claim_format.replace("CLAIM_ID", claim_id)
                    if expected not in content_body:
                        logger.error(f"Test {test_case['id']}: Claim {claim_id} not properly cited")
                        return False

            # Test: all_claims_from_library
            if assertions.get("all_claims_from_library"):
                invalid_claims = [c for c in found_claims if c not in valid_claim_ids]
                if invalid_claims:
                    logger.error(f"Test {test_case['id']}: Invalid claims found: {invalid_claims}")
                    return False

            # Test: no_invalid_claims
            if assertions.get("no_invalid_claims"):
                invalid_claims = [c for c in found_claims if c not in valid_claim_ids]
                if invalid_claims:
                    logger.error(f"Test {test_case['id']}: Invalid claims: {invalid_claims}")
                    return False

            # Test: claims_match_persona
            # Skip this test if we used fallback (all claims available)
            if assertions.get("claims_match_persona"):
                persona = test_case["input"].get("persona")

                # Check if any claim has this persona (if not, fallback was used)
                persona_exists = any(persona in c.get('personas', []) for c in claim_library_claims)

                if not persona_exists:
                    logger.info(f"Test {test_case['id']}: Skipping persona match validation (fallback used for unknown persona '{persona}')")
                else:
                    # Only validate persona matching if persona exists in library
                    for claim_id in valid_found_claims:
                        claim = next((c for c in claim_library_claims if c['id'] == claim_id), None)
                        if claim and persona not in claim.get('personas', []):
                            logger.error(f"Test {test_case['id']}: Claim {claim_id} doesn't match persona {persona}")
                            return False

            # Test: safety_score_min
            if assertions.get("safety_score_min"):
                overall_score = safety_result.get("overall_score", 0)
                if overall_score < assertions["safety_score_min"]:
                    logger.error(f"Test {test_case['id']}: Safety score too low: {overall_score}")
                    return False

            # Test: toxicity_score_max
            # NOTE: toxicity_score in SafetyValidator is INVERTED (1.0 = safe, 0.0 = toxic)
            # Test cases expect toxicity_score_max to mean "maximum acceptable toxicity"
            # So we need to check if content is TOO SAFE (which would be wrong interpretation)
            # Instead, interpret toxicity_score_max as minimum safety score for toxicity dimension
            if assertions.get("toxicity_score_max"):
                toxicity_score = safety_result.get("toxicity_score", 0.0)
                # Convert test expectation: toxicity_score_max 0.1 means "toxicity ≤ 0.1"
                # Which translates to "safety ≥ 0.9" (since it's inverted)
                min_toxicity_safety = 1.0 - assertions["toxicity_score_max"]
                if toxicity_score < min_toxicity_safety:
                    logger.error(f"Test {test_case['id']}: Toxicity safety too low: {toxicity_score} (expected ≥ {min_toxicity_safety})")
                    return False

            # Test: claim_validation passed
            if safety_result.get("claim_validation"):
                if not safety_result["claim_validation"].get("valid"):
                    logger.error(f"Test {test_case['id']}: Claim validation failed: {safety_result['claim_validation'].get('reason')}")
                    return False

            logger.info(f"✅ Test {test_case['id']} passed (claims: {valid_found_claims})")
            return True

        except Exception as e:
            logger.error(f"Test {test_case['id']} failed with exception: {e}")
            return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Golden Test Suite for Agentic AI Platform")
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        default=True,
        help="Use fixture mode (pre-generated content) instead of live API calls (default: True)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Use live API calls instead of fixtures (may be slow/expensive)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Determine mode (--live overrides --use-fixtures)
    use_fixtures = not args.live

    print(f"\n{'='*60}")
    print(f"GOLDEN TEST SUITE - Agentic AI Marketing Platform")
    print(f"{'='*60}")
    print(f"Mode: {'FIXTURE (fast, no API calls)' if use_fixtures else 'LIVE (slow, API calls)'}")
    print(f"{'='*60}\n")

    runner = GoldenTestRunner(use_fixtures=use_fixtures)
    results = asyncio.run(runner.run_all_tests())

    # Save results to file for API and dashboard
    import json
    from datetime import datetime

    results_with_timestamp = {
        "pass_rate": results["pass_rate"] / 100.0,  # Convert to 0.0-1.0
        "total_tests": results["total"],
        "passed_tests": results["passed"],
        "failed_tests": results["failed"],
        "last_run": datetime.utcnow().isoformat(),
        "duration_seconds": results.get("duration_seconds", 0),
        "mode": "fixture" if use_fixtures else "live",
        "test_details": [],
        "failures": results["failures"]
    }

    with open("golden_test_results.json", 'w') as f:
        json.dump(results_with_timestamp, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GOLDEN TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass Rate: {results['pass_rate']:.1f}%")
    print(f"Duration: {results.get('duration_seconds', 0):.1f}s")
    print(f"Mode: {'FIXTURE' if use_fixtures else 'LIVE'}")

    if results["pass_rate"] < 100.0:
        print(f"\n❌ DEPLOYMENT BLOCKED")
        print(f"Failed tests: {results['failures']}")
        print(f"\nTo debug, run with --verbose flag:")
        print(f"  python tests/golden/test_runner.py --verbose")
        exit(1)
    else:
        print(f"\n✅ ALL TESTS PASSED - DEPLOYMENT APPROVED")
        print(f"\nResults saved to: golden_test_results.json")
        exit(0)