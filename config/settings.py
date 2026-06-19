"""
Central configuration for the Agentic AI Agent Platform.

Infrastructure-only Pydantic fields live here (DB, Redis, paths, server).
ALL application settings (LLM, credentials, thresholds, feature flags …)
are served exclusively by the DB-backed Configuration Service and accessed
transparently via the ``__getattribute__`` / ``get()`` proxy on Settings.
"""
import logging
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path

_logger = logging.getLogger(__name__)

# Keys resolved locally — needed before the DB-backed config service is up.
_INFRA_KEYS = frozenset({
    "APP_NAME", "APP_VERSION", "ENVIRONMENT", "DEBUG", "LOG_LEVEL", "PORT",
    "PROJECT_ROOT", "DATA_DIR", "LOG_DIR", "CONFIG_DIR", "BACKUP_DIR",
    "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD",
    "DATABASE_URL", "DB_POOL_SIZE", "DB_MAX_OVERFLOW",
    "REDIS_URL", "REDIS_MAX_CONNECTIONS",
    "SECRET_KEY",
    "API_HOST", "API_PORT", "API_PREFIX", "CORS_ORIGINS",
    "DASHBOARD_HOST", "DASHBOARD_PORT",
})


_MISSING = object()  # sentinel – distinguishes "not found" from empty/None values


def _from_config_service(key: str) -> Any:
    """Try to read a value from the DB-backed Configuration Service.
    
    Returns the value if found (including empty string / False / 0), or
    ``None`` when the key genuinely does not exist anywhere.
    """
    # --- fast path: full config service available ---
    try:
        from src.config.configuration_service import get_runtime_config, DEFAULT_CONFIGURATIONS
        val = get_runtime_config(key, default=_MISSING)
        if val is _MISSING:
            # Key not in DB or defaults at all
            return None
        if val is None and key in DEFAULT_CONFIGURATIONS:
            # Known key whose DB/default value is empty → return "" not None
            # so the proxy doesn't fall through to AttributeError.
            return ""
        return val
    except Exception:
        pass

    # --- fallback: module still loading (circular import) → read defaults dict ---
    try:
        from src.config.configuration_service import DEFAULT_CONFIGURATIONS
        cfg = DEFAULT_CONFIGURATIONS.get(key)
        if cfg is not None:
            return _convert_default(cfg["default"], cfg["value_type"])
    except Exception:
        pass
    return None


def _convert_default(raw, vtype: str) -> Any:
    """Lightweight type coercion for DEFAULT_CONFIGURATIONS entries.
    
    Unlike the full config service helper, this intentionally keeps empty
    strings as ``""`` so that the proxy never collapses them to ``None``.
    """
    if raw is None or (isinstance(raw, str) and raw == ""):
        return ""
    try:
        if vtype == "integer":
            return int(raw)
        elif vtype == "float":
            return float(raw)
        elif vtype == "boolean":
            return str(raw).lower() in ("true", "1", "yes", "on")
        return raw
    except (ValueError, TypeError):
        return raw


class Settings(BaseSettings):
    """Thin infrastructure shell.  Non-infra reads are proxied to config service."""

    # ── Core / bootstrap ────────────────────────────────────────────────
    APP_NAME: str = "Agentic AI Agent Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    PORT: int = Field(default=8000)

    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent)

    # ── PostgreSQL (composed into DATABASE_URL) ─────────────────────────
    POSTGRES_DB: str = Field(default="agentic", env="POSTGRES_DB")
    POSTGRES_USER: str = Field(default="agentic", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="agentic", env="POSTGRES_PASSWORD")
    DATABASE_URL: str = Field(default="", env="DATABASE_URL")
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40

    # ── Redis ───────────────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = 50

    # ── Security ────────────────────────────────────────────────────────
    SECRET_KEY: str = Field(default="change-me-to-a-random-string", env="SECRET_KEY")

    # ── Server / networking ─────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default="/api/v1")
    CORS_ORIGINS: list = Field(default=["http://localhost:8501", "http://localhost:3000"])
    DASHBOARD_HOST: str = Field(default="0.0.0.0")
    DASHBOARD_PORT: int = Field(default=8501)

    # ── Directories (auto-created) ──────────────────────────────────────
    DATA_DIR: Path = Field(default=Path("data"))
    LOG_DIR: Path = Field(default=Path("logs"))
    CONFIG_DIR: Path = Field(default=Path("config"))
    BACKUP_DIR: Path = Field(default=Path("backups"))

    # ── Pydantic config ─────────────────────────────────────────────────
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    # ── Validators ──────────────────────────────────────────────────────
    @validator("DATABASE_URL", pre=True, always=True)
    def assemble_database_url(cls, v, values):
        if v:
            return v
        user = values.get("POSTGRES_USER", "agentic")
        password = values.get("POSTGRES_PASSWORD", "agentic")
        db = values.get("POSTGRES_DB", "agentic")
        return f"postgresql://{user}:{password}@localhost:5432/{db}"

    @validator("DATA_DIR", "LOG_DIR", "CONFIG_DIR", "BACKUP_DIR")
    def create_directories(cls, v):
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v.startswith("postgresql://"):
            raise ValueError("Database URL must start with postgresql://")
        return v

    # ── Convenience helpers (use self.get() → config service) ───────────
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    def get_llm_config(self) -> Dict[str, Any]:
        if self.get("USE_LOCAL_LLM", False):
            return {
                "provider": "ollama",
                "host": self.get("OLLAMA_HOST", "http://localhost:11434"),
                "model": self.get("OLLAMA_MODEL", "qwen3:8b"),
                "temperature": self.get("OPENAI_TEMPERATURE", 0.7),
            }
        return {
            "provider": "openai",
            "api_key": self.get("OPENAI_API_KEY", ""),
            "model": self.get("OPENAI_MODEL", "gpt-4-turbo-preview"),
            "temperature": self.get("OPENAI_TEMPERATURE", 0.7),
            "max_tokens": self.get("OPENAI_MAX_TOKENS", 2000),
        }

    def get_database_config(self) -> Dict[str, Any]:
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DB_POOL_SIZE,
            "max_overflow": self.DB_MAX_OVERFLOW,
            "pool_pre_ping": True,
            "echo": self.DEBUG,
        }

    def get_redis_config(self) -> Dict[str, Any]:
        return {
            "url": self.REDIS_URL,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "decode_responses": True,
        }

    def get_cost_limits(self) -> Dict[str, float]:
        return {
            "daily_max": float(self.get("MAX_DAILY_API_COST", 1000.0)),
            "campaign_max": float(self.get("MAX_CAMPAIGN_COST", 500.0)),
        }

    def get_safety_thresholds(self) -> Dict[str, float]:
        return {
            "overall": float(self.get("SAFETY_SCORE_THRESHOLD", 0.8)),
            "toxicity": float(self.get("TOXICITY_THRESHOLD", 0.1)),
            "auto_approve": float(self.get("AUTO_APPROVE_THRESHOLD", 0.95)),
        }

    # ── Config-service proxy ────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        """DB config service → Pydantic field → *default*."""
        if key not in _INFRA_KEYS:
            db_val = _from_config_service(key)
            if db_val is not None:
                return db_val
        try:
            return super().__getattribute__(key)
        except AttributeError:
            return default

    def __getattribute__(self, key: str) -> Any:
        if (
            not key.startswith("_")
            and key == key.upper()
            and key not in _INFRA_KEYS
        ):
            db_val = _from_config_service(key)
            if db_val is not None:
                return db_val
        return super().__getattribute__(key)

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(f"Setting '{key}' not found")
        return val


settings = Settings()

# Module-level convenience exports (infra-only)
DEBUG = settings.DEBUG
DATABASE_URL = settings.DATABASE_URL
REDIS_URL = settings.REDIS_URL