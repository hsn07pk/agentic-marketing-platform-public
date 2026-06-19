# tests/unit/conftest.py
"""
Fix the scipy/torch v2.10 compatibility bug that crashes with:
  TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union

The crash happens when scipy.stats initializes and calls is_torch_array()
which does issubclass(cls, torch.Tensor). On torch 2.10+, torch.Tensor
triggers this TypeError in certain issubclass() edge cases.

The fix: monkey-patch the problematic scipy helper BEFORE scipy.stats loads.
We also pre-mock optional heavy packages (optuna, transformers, sklearn)
that are imported by the application but not needed for unit tests.
"""
import sys
from unittest.mock import MagicMock


def _ensure_mock(name: str):
    """Insert a MagicMock module into sys.modules if not already present."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


# ---------------------------------------------------------------------------
# Fix scipy/torch compatibility BEFORE scipy.stats is imported
# ---------------------------------------------------------------------------
try:
    # Monkey-patch the broken helper before scipy.stats initializes
    import scipy._lib.array_api_compat.common._helpers as _helpers

    _original_issubclass_fast = _helpers._issubclass_fast

    def _safe_issubclass_fast(cls, *args, **kwargs):
        try:
            return _original_issubclass_fast(cls, *args, **kwargs)
        except TypeError:
            return False

    _helpers._issubclass_fast = _safe_issubclass_fast
except (ImportError, AttributeError):
    pass  # scipy not installed or different version — no fix needed

# ---------------------------------------------------------------------------
# optuna  (required by advanced_experiments.py, may not be installed)
# ---------------------------------------------------------------------------
_ensure_mock("optuna")

# ---------------------------------------------------------------------------
# transformers  (optional, may not be installed)
# ---------------------------------------------------------------------------
_ensure_mock("transformers")

# ---------------------------------------------------------------------------
# sklearn and sub-modules  (heavy, not needed for unit tests)
# ---------------------------------------------------------------------------
_ensure_mock("sklearn")
_ensure_mock("sklearn.ensemble")
_ensure_mock("sklearn.gaussian_process")
_ensure_mock("sklearn.gaussian_process.kernels")
_ensure_mock("sklearn.model_selection")
_ensure_mock("sklearn.metrics")

# ---------------------------------------------------------------------------
# FastAPI and its submodules (not installed in test environment)
# The import chain: campaign_monitor → api.dependencies → api.__init__
# → api.main → fastapi.middleware.cors, fastapi.middleware.gzip, etc.
# ---------------------------------------------------------------------------
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


