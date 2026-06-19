"""
Configuration service for managing database-backed system configurations.
Provides CRUD operations, caching, and encryption for sensitive values.
"""
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from src.data_layer.database.models import SystemConfiguration, ConfigurationCategory
from src.config.encryption import encrypt_value, decrypt_value, mask_sensitive_value

logger = logging.getLogger(__name__)


DEFAULT_CONFIGURATIONS = {
    "DATABASE_URL": {
        "category": ConfigurationCategory.DATABASE,
        "default": "postgresql://agentic:changeme@postgres:5432/agentic",
        "description": "PostgreSQL database connection URL",
        "is_secret": True,
        "value_type": "string",
    },
    "DB_POOL_SIZE": {
        "category": ConfigurationCategory.DATABASE,
        "default": "20",
        "description": "Database connection pool size",
        "is_secret": False,
        "value_type": "integer",
    },
    "DB_MAX_OVERFLOW": {
        "category": ConfigurationCategory.DATABASE,
        "default": "40",
        "description": "Maximum overflow connections",
        "is_secret": False,
        "value_type": "integer",
    },
    
    "REDIS_URL": {
        "category": ConfigurationCategory.REDIS,
        "default": "redis://localhost:6379",
        "description": "Redis connection URL for caching and queues",
        "is_secret": False,
        "value_type": "string",
    },
    "REDIS_MAX_CONNECTIONS": {
        "category": ConfigurationCategory.REDIS,
        "default": "50",
        "description": "Maximum Redis connections",
        "is_secret": False,
        "value_type": "integer",
    },
    
    "USE_LOCAL_LLM": {
        "category": ConfigurationCategory.LLM,
        "default": "True",
        "description": "Use local Ollama LLM instead of OpenAI",
        "is_secret": False,
        "value_type": "boolean",
    },
    "OLLAMA_HOST": {
        "category": ConfigurationCategory.LLM,
        "default": "http://localhost:11434",
        "description": "Ollama server host URL",
        "is_secret": False,
        "value_type": "string",
    },
    "OLLAMA_MODEL": {
        "category": ConfigurationCategory.LLM,
        "default": "qwen3:8b",
        "description": "Ollama model to use",
        "is_secret": False,
        "value_type": "string",
    },
    "OLLAMA_TEMPERATURE": {
        "category": ConfigurationCategory.LLM,
        "default": "0.7",
        "description": "Ollama model temperature",
        "is_secret": False,
        "value_type": "float",
    },
    "OLLAMA_MAX_TOKENS": {
        "category": ConfigurationCategory.LLM,
        "default": "2000",
        "description": "Maximum tokens for Ollama responses",
        "is_secret": False,
        "value_type": "integer",
    },
    "OPENAI_API_KEY": {
        "category": ConfigurationCategory.LLM,
        "default": "",
        "description": "OpenAI API key (required if not using local LLM)",
        "is_secret": True,
        "value_type": "string",
    },
    "OPENAI_MODEL": {
        "category": ConfigurationCategory.LLM,
        "default": "gpt-4-turbo-preview",
        "description": "OpenAI model to use",
        "is_secret": False,
        "value_type": "string",
    },
    "OPENAI_TEMPERATURE": {
        "category": ConfigurationCategory.LLM,
        "default": "0.7",
        "description": "OpenAI model temperature",
        "is_secret": False,
        "value_type": "float",
    },
    "OPENAI_MAX_TOKENS": {
        "category": ConfigurationCategory.LLM,
        "default": "2000",
        "description": "Maximum tokens for OpenAI responses",
        "is_secret": False,
        "value_type": "integer",
    },
    "EMBEDDING_MODEL": {
        "category": ConfigurationCategory.LLM,
        "default": "text-embedding-ada-002",
        "description": "Embedding model for RAG/vector search",
        "is_secret": False,
        "value_type": "string",
    },
    "EMBEDDING_DIMENSION": {
        "category": ConfigurationCategory.LLM,
        "default": "1536",
        "description": "Embedding vector dimension",
        "is_secret": False,
        "value_type": "integer",
    },
    
    "LINKEDIN_CLIENT_ID": {
        "category": ConfigurationCategory.LINKEDIN,
        "default": "",
        "description": "LinkedIn OAuth client ID",
        "is_secret": True,
        "value_type": "string",
    },
    "LINKEDIN_CLIENT_SECRET": {
        "category": ConfigurationCategory.LINKEDIN,
        "default": "",
        "description": "LinkedIn OAuth client secret",
        "is_secret": True,
        "value_type": "string",
    },
    "LINKEDIN_ACCESS_TOKEN": {
        "category": ConfigurationCategory.LINKEDIN,
        "default": "",
        "description": "LinkedIn access token",
        "is_secret": True,
        "value_type": "string",
    },
    "LINKEDIN_ACCOUNT_ID": {
        "category": ConfigurationCategory.LINKEDIN,
        "default": "",
        "description": "LinkedIn advertising account ID",
        "is_secret": False,
        "value_type": "string",
    },
    "LINKEDIN_ORGANIZATION_ID": {
        "category": ConfigurationCategory.LINKEDIN,
        "default": "",
        "description": "LinkedIn organization ID",
        "is_secret": False,
        "value_type": "string",
    },
    
    "TWITTER_API_KEY": {
        "category": ConfigurationCategory.TWITTER,
        "default": "",
        "description": "Twitter API key",
        "is_secret": True,
        "value_type": "string",
    },
    "TWITTER_API_SECRET": {
        "category": ConfigurationCategory.TWITTER,
        "default": "",
        "description": "Twitter API secret",
        "is_secret": True,
        "value_type": "string",
    },
    "TWITTER_ACCESS_TOKEN": {
        "category": ConfigurationCategory.TWITTER,
        "default": "",
        "description": "Twitter access token",
        "is_secret": True,
        "value_type": "string",
    },
    "TWITTER_ACCESS_TOKEN_SECRET": {
        "category": ConfigurationCategory.TWITTER,
        "default": "",
        "description": "Twitter access token secret",
        "is_secret": True,
        "value_type": "string",
    },
    "TWITTER_USERNAME": {
        "category": ConfigurationCategory.TWITTER,
        "default": "agentic",
        "description": "Twitter username",
        "is_secret": False,
        "value_type": "string",
    },
    
    # Blog / CMS
    "BLOG_CMS_URL": {
        "category": ConfigurationCategory.BLOG,
        "default": "https://example.com",
        "description": "Blog CMS base URL (WordPress site URL)",
        "is_secret": False,
        "value_type": "string",
    },
    "BLOG_API_KEY": {
        "category": ConfigurationCategory.BLOG,
        "default": "",
        "description": "Blog CMS API key or application password",
        "is_secret": True,
        "value_type": "string",
    },
    "BLOG_USERNAME": {
        "category": ConfigurationCategory.BLOG,
        "default": "",
        "description": "Blog CMS username for API authentication",
        "is_secret": False,
        "value_type": "string",
    },
    "BLOG_APP_PASSWORD": {
        "category": ConfigurationCategory.BLOG,
        "default": "",
        "description": "WordPress application password for REST API auth",
        "is_secret": True,
        "value_type": "string",
    },
    "BLOG_DEFAULT_AUTHOR": {
        "category": ConfigurationCategory.BLOG,
        "default": "Agentic AI",
        "description": "Default author name for blog posts",
        "is_secret": False,
        "value_type": "string",
    },
    "BLOG_DEFAULT_CATEGORY": {
        "category": ConfigurationCategory.BLOG,
        "default": "Marketing",
        "description": "Default category for auto-generated blog posts",
        "is_secret": False,
        "value_type": "string",
    },

    "SENDGRID_API_KEY": {
        "category": ConfigurationCategory.EMAIL,
        "default": "",
        "description": "SendGrid API key for email campaigns",
        "is_secret": True,
        "value_type": "string",
    },
    "SENDGRID_FROM_EMAIL": {
        "category": ConfigurationCategory.EMAIL,
        "default": "noreply@example.com",
        "description": "Default sender email address",
        "is_secret": False,
        "value_type": "string",
    },
    "SENDGRID_FROM_NAME": {
        "category": ConfigurationCategory.EMAIL,
        "default": "Agentic AI",
        "description": "Default sender name",
        "is_secret": False,
        "value_type": "string",
    },

    "MAILGUN_API_KEY": {
        "category": ConfigurationCategory.EMAIL,
        "default": "",
        "description": "Mailgun API key (Primary Email Service)",
        "is_secret": True,
        "value_type": "string",
    },
    "MAILGUN_DOMAIN": {
        "category": ConfigurationCategory.EMAIL,
        "default": "",
        "description": "Mailgun domain (e.g., mg.agentic.ai)",
        "is_secret": False,
        "value_type": "string",
    },
    "MAILGUN_REGION": {
        "category": ConfigurationCategory.EMAIL,
        "default": "eu",
        "description": "Mailgun API region: 'eu' for European domains, 'us' for US domains",
        "is_secret": False,
        "value_type": "string",
    },
    "MAILGUN_FROM_EMAIL": {
        "category": ConfigurationCategory.EMAIL,
        "default": "alerts@example.com",
        "description": "Mailgun sender email",
        "is_secret": False,
        "value_type": "string",
    },
    "MAILGUN_FROM_NAME": {
        "category": ConfigurationCategory.EMAIL,
        "default": "Agentic AI",
        "description": "Mailgun sender name",
        "is_secret": False,
        "value_type": "string",
    },
    "DEFAULT_EMAIL_RECIPIENTS": {
        "category": ConfigurationCategory.EMAIL,
        "default": "",
        "description": "Comma-separated default email recipients for campaign deployments (e.g. user1@example.com,user2@example.com)",
        "is_secret": False,
        "value_type": "string",
    },
    
    "HUBSPOT_API_KEY": {
        "category": ConfigurationCategory.HUBSPOT,
        "default": "",
        "description": "HubSpot API key for CRM integration",
        "is_secret": True,
        "value_type": "string",
    },
    "HUBSPOT_PORTAL_ID": {
        "category": ConfigurationCategory.HUBSPOT,
        "default": "",
        "description": "HubSpot portal ID",
        "is_secret": False,
        "value_type": "string",
    },
    "HUBSPOT_LIFECYCLE_STAGES": {
        "category": ConfigurationCategory.HUBSPOT,
        "default": '["lead", "marketingqualifiedlead"]',
        "description": "JSON array of HubSpot lifecycle stages to track in funnel attribution.",
        "is_secret": False,
        "value_type": "string",
    },
    "HUBSPOT_DEAL_STAGES": {
        "category": ConfigurationCategory.HUBSPOT,
        "default": '["appointmentscheduled", "closedwon"]',
        "description": "JSON array of HubSpot deal stages to track in funnel attribution.",
        "is_secret": False,
        "value_type": "string",
    },
    "HUBSPOT_CLOSED_WON_STAGES": {
        "category": ConfigurationCategory.HUBSPOT,
        "default": '["closedwon", "closed_won", "won", "qualifiedtobuy"]',
        "description": "JSON array of HubSpot deal stage names that indicate a closed-won deal.",
        "is_secret": False,
        "value_type": "string",
    },
    
    "CALENDAR_API_KEY": {
        "category": ConfigurationCategory.CALENDAR,
        "default": "",
        "description": "Cal.com API key",
        "is_secret": True,
        "value_type": "string",
    },
    "CALENDAR_API_URL": {
        "category": ConfigurationCategory.CALENDAR,
        "default": "https://api.cal.com/v1",
        "description": "Calendar API URL",
        "is_secret": False,
        "value_type": "string",
    },
    "CALCOM_WEBHOOK_EVENTS": {
        "category": ConfigurationCategory.CALENDAR,
        "default": '["BOOKING_CREATED", "BOOKING_RESCHEDULED", "BOOKING_CANCELLED"]',
        "description": "JSON array of Cal.com webhook event types to subscribe to.",
        "is_secret": False,
        "value_type": "string",
    },
    
    "APIFY_API_TOKEN": {
        "category": ConfigurationCategory.APIFY,
        "default": "",
        "description": "Apify API token for web scraping",
        "is_secret": True,
        "value_type": "string",
    },
    "ENABLE_COMPETITIVE_ANALYSIS": {
        "category": ConfigurationCategory.APIFY,
        "default": "True",
        "description": "Enable competitive intelligence features",
        "is_secret": False,
        "value_type": "boolean",
    },
    "ENABLE_SCRAPING": {
        "category": ConfigurationCategory.APIFY,
        "default": "True",
        "description": "Enable web scraping features",
        "is_secret": False,
        "value_type": "boolean",
    },
    "MARKET_SCRAPE_SOURCES": {
        "category": ConfigurationCategory.APIFY,
        "default": "",
        "description": "JSON array of web scraping sources. Each source: {url, fallback, name, extractor, category}. Leave empty for built-in HR tech sources (Personnel Today, HRD, Lattice, Culture Amp, Visier, 15Five).",
        "is_secret": False,
        "value_type": "string",
    },
    
    "SAFETY_SCORE_THRESHOLD": {
        "category": ConfigurationCategory.GOVERNANCE,
        "default": "0.8",
        "description": "Minimum safety score for content approval",
        "is_secret": False,
        "value_type": "float",
    },
    "TOXICITY_THRESHOLD": {
        "category": ConfigurationCategory.GOVERNANCE,
        "default": "0.1",
        "description": "Maximum allowed toxicity score",
        "is_secret": False,
        "value_type": "float",
    },
    "REQUIRE_HUMAN_APPROVAL": {
        "category": ConfigurationCategory.GOVERNANCE,
        "default": "True",
        "description": "Require human approval for content",
        "is_secret": False,
        "value_type": "boolean",
    },
    "AUTO_APPROVE_THRESHOLD": {
        "category": ConfigurationCategory.GOVERNANCE,
        "default": "0.95",
        "description": "Safety score threshold for auto-approval",
        "is_secret": False,
        "value_type": "float",
    },
    "MIN_SAFETY_SCORE": {
        "category": ConfigurationCategory.GOVERNANCE,
        "default": "0.70",
        "description": "Minimum acceptable safety score",
        "is_secret": False,
        "value_type": "float",
    },
    
    "MAX_DAILY_API_COST": {
        "category": ConfigurationCategory.COST_CONTROL,
        "default": "1000.0",
        "description": "Maximum daily API cost in EUR",
        "is_secret": False,
        "value_type": "float",
    },
    "MAX_CAMPAIGN_COST": {
        "category": ConfigurationCategory.COST_CONTROL,
        "default": "500.0",
        "description": "Maximum cost per campaign in EUR",
        "is_secret": False,
        "value_type": "float",
    },
    "ENABLE_SEMANTIC_CACHE": {
        "category": ConfigurationCategory.COST_CONTROL,
        "default": "True",
        "description": "Enable semantic caching to reduce API costs",
        "is_secret": False,
        "value_type": "boolean",
    },
    "CACHE_TTL_SECONDS": {
        "category": ConfigurationCategory.COST_CONTROL,
        "default": "3600",
        "description": "Cache time-to-live in seconds",
        "is_secret": False,
        "value_type": "integer",
    },
    
    "SIMULATION_ACCURACY_TARGET": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "0.9",
        "description": "Target simulation accuracy (1 - MAPE)",
        "is_secret": False,
        "value_type": "float",
    },
    "SIMULATION_TIME_STEP": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "3600",
        "description": "Simulation time step in seconds",
        "is_secret": False,
        "value_type": "integer",
    },
    "DEFAULT_SIMULATION_DAYS": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "30",
        "description": "Default simulation duration in days",
        "is_secret": False,
        "value_type": "integer",
    },
    "SIM_MIN_CTR": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "0.01",
        "description": "Minimum CTR threshold for simulation pass (1% = 0.01)",
        "is_secret": False,
        "value_type": "float",
    },
    "SIM_MIN_CONVERSIONS": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "1",
        "description": "Minimum conversions required for simulation pass",
        "is_secret": False,
        "value_type": "integer",
    },
    "SIM_MAX_CPL": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "2000.0",
        "description": "Maximum cost per lead for simulation pass",
        "is_secret": False,
        "value_type": "float",
    },
    
    "BANDIT_ALGORITHM": {
        "category": ConfigurationCategory.LEARNING,
        "default": "thompson_sampling",
        "description": "Contextual bandit algorithm (thompson_sampling, linucb, neural)",
        "is_secret": False,
        "value_type": "string",
    },
    "BANDIT_EXPLORATION_RATE": {
        "category": ConfigurationCategory.LEARNING,
        "default": "0.1",
        "description": "Exploration rate for bandit algorithms",
        "is_secret": False,
        "value_type": "float",
    },
    "BANDIT_LEARNING_RATE": {
        "category": ConfigurationCategory.LEARNING,
        "default": "0.01",
        "description": "Learning rate for bandit algorithms",
        "is_secret": False,
        "value_type": "float",
    },
    "ENABLE_REWARD_TRACKING": {
        "category": ConfigurationCategory.LEARNING,
        "default": "True",
        "description": "Enable reward tracking for learning",
        "is_secret": False,
        "value_type": "boolean",
    },
    "REWARD_DELAY_WINDOW_HOURS": {
        "category": ConfigurationCategory.LEARNING,
        "default": "72",
        "description": "Reward attribution window in hours",
        "is_secret": False,
        "value_type": "integer",
    },
    "MIN_SAMPLES_FOR_DECISION": {
        "category": ConfigurationCategory.LEARNING,
        "default": "10",
        "description": "Minimum samples before making decisions",
        "is_secret": False,
        "value_type": "integer",
    },
    "ENABLE_NEURAL_BANDIT": {
        "category": ConfigurationCategory.LEARNING,
        "default": "False",
        "description": "Enable neural network bandit",
        "is_secret": False,
        "value_type": "boolean",
    },
    "USE_GPU": {
        "category": ConfigurationCategory.LEARNING,
        "default": "False",
        "description": "Use GPU for neural computations",
        "is_secret": False,
        "value_type": "boolean",
    },
    "STRATEGY_HOOK_TEMPLATES": {
        "category": ConfigurationCategory.LEARNING,
        "default": "",
        "description": "JSON object of strategy hook templates (RL bandit arm priors). Keys: hook_transform, hook_problem, hook_success, hook_question. Leave empty for built-in defaults.",
        "is_secret": False,
        "value_type": "string",
    },
    "STRATEGY_PERSONA_CUSTOMIZATIONS": {
        "category": ConfigurationCategory.LEARNING,
        "default": "",
        "description": "JSON object mapping persona IDs to customization values ({pain_point, industry, metric, benefit}). Leave empty for built-in defaults.",
        "is_secret": False,
        "value_type": "string",
    },
    "OPTIMAL_TIMING_DEFAULTS": {
        "category": ConfigurationCategory.LEARNING,
        "default": "",
        "description": "JSON object of optimal posting times per platform/persona (initial priors for RL). Leave empty for built-in B2B heuristics.",
        "is_secret": False,
        "value_type": "string",
    },
    
    "ENABLE_MARL": {
        "category": ConfigurationCategory.MARL,
        "default": "False",
        "description": "Enable multi-agent reinforcement learning",
        "is_secret": False,
        "value_type": "boolean",
    },
    "MARL_POLICY_PROMOTION_THRESHOLD": {
        "category": ConfigurationCategory.MARL,
        "default": "0.2",
        "description": "Required lift for policy promotion (20%)",
        "is_secret": False,
        "value_type": "float",
    },
    "ENABLE_CANARY_DEPLOYMENT": {
        "category": ConfigurationCategory.MARL,
        "default": "True",
        "description": "Enable canary deployment for policies",
        "is_secret": False,
        "value_type": "boolean",
    },
    "CANARY_TRAFFIC_PERCENTAGE": {
        "category": ConfigurationCategory.MARL,
        "default": "0.05",
        "description": "Initial canary traffic percentage",
        "is_secret": False,
        "value_type": "float",
    },
    "CANARY_PERCENTAGE": {
        "category": ConfigurationCategory.MARL,
        "default": "10",
        "description": "Canary group percentage",
        "is_secret": False,
        "value_type": "integer",
    },
    "CANARY_DURATION_HOURS": {
        "category": ConfigurationCategory.MARL,
        "default": "24",
        "description": "Canary deployment duration in hours",
        "is_secret": False,
        "value_type": "integer",
    },
    "ROLLBACK_THRESHOLD": {
        "category": ConfigurationCategory.MARL,
        "default": "0.05",
        "description": "Performance drop threshold for rollback",
        "is_secret": False,
        "value_type": "float",
    },
    "OPE_CONFIDENCE_LEVEL": {
        "category": ConfigurationCategory.MARL,
        "default": "0.95",
        "description": "Off-Policy Evaluation confidence level",
        "is_secret": False,
        "value_type": "float",
    },
    "OPE_MIN_SAMPLES": {
        "category": ConfigurationCategory.MARL,
        "default": "1000",
        "description": "Minimum samples for OPE",
        "is_secret": False,
        "value_type": "integer",
    },
    "MARL_MIN_TRAINING_SAMPLES": {
        "category": ConfigurationCategory.MARL,
        "default": "20",
        "description": "Minimum samples required to auto-train MARL policy",
        "is_secret": False,
        "value_type": "integer",
    },
    
    "PROMETHEUS_PORT": {
        "category": ConfigurationCategory.MONITORING,
        "default": "8080",
        "description": "Prometheus metrics port",
        "is_secret": False,
        "value_type": "integer",
    },
    "ENABLE_METRICS": {
        "category": ConfigurationCategory.MONITORING,
        "default": "True",
        "description": "Enable Prometheus metrics",
        "is_secret": False,
        "value_type": "boolean",
    },
    "ENABLE_TRACING": {
        "category": ConfigurationCategory.MONITORING,
        "default": "True",
        "description": "Enable distributed tracing",
        "is_secret": False,
        "value_type": "boolean",
    },
    "SLACK_WEBHOOK_URL": {
        "category": ConfigurationCategory.MONITORING,
        "default": "",
        "description": "Slack webhook URL for alerts",
        "is_secret": True,
        "value_type": "string",
    },
    "GRAFANA_PASSWORD": {
        "category": ConfigurationCategory.MONITORING,
        "default": "admin123",
        "description": "Grafana admin password",
        "is_secret": True,
        "value_type": "string",
    },
    
    "SECRET_KEY": {
        "category": ConfigurationCategory.SECURITY,
        "default": "dev-secret-key-change-in-production-minimum-32-chars",
        "description": "Secret key for JWT and session encryption",
        "is_secret": True,
        "value_type": "string",
    },
    "JWT_ALGORITHM": {
        "category": ConfigurationCategory.SECURITY,
        "default": "HS256",
        "description": "JWT signing algorithm",
        "is_secret": False,
        "value_type": "string",
    },
    "ACCESS_TOKEN_EXPIRE_MINUTES": {
        "category": ConfigurationCategory.SECURITY,
        "default": "30",
        "description": "Access token expiration in minutes",
        "is_secret": False,
        "value_type": "integer",
    },
    
    "MOCK_MODE_ENABLED": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "True",
        "description": "Master toggle for mock mode. When ON: uses test data/mock deployments. When OFF: requires real APIs, shows only authenticated data.",
        "is_secret": False,
        "value_type": "boolean",
    },
    "ENABLE_MOCK_DEPLOYMENT": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "True",
        "description": "Enable mock deployment mode for testing (inherits from MOCK_MODE_ENABLED)",
        "is_secret": False,
        "value_type": "boolean",
    },
    "ENABLE_MOCK_EXPERIMENTS": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "True",
        "description": "Use mock experiment runner for testing (inherits from MOCK_MODE_ENABLED)",
        "is_secret": False,
        "value_type": "boolean",
    },
    "INCLUDE_MOCK_IN_METRICS": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "True",
        "description": "Include mock campaign data in platform metrics and KPIs. When ON, mock data is included but clearly labeled. When OFF, only real production data is shown.",
        "is_secret": False,
        "value_type": "boolean",
    },
    "ENABLE_SURVIVAL_MODEL": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "False",
        "description": "Enable survival analysis model",
        "is_secret": False,
        "value_type": "boolean",
    },
    "AUTO_INIT_DB": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "True",
        "description": "Auto-initialize database on startup",
        "is_secret": False,
        "value_type": "boolean",
    },
    "AUTO_SEED_DATA": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "False",
        "description": "Auto-seed sample data on startup",
        "is_secret": False,
        "value_type": "boolean",
    },
    
    "ENVIRONMENT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "development",
        "description": "Environment (development, staging, production)",
        "is_secret": False,
        "value_type": "string",
    },
    "DEBUG": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "False",
        "description": "Enable debug mode",
        "is_secret": False,
        "value_type": "boolean",
    },
    "LOG_LEVEL": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "INFO",
        "description": "Logging level (DEBUG, INFO, WARNING, ERROR)",
        "is_secret": False,
        "value_type": "string",
    },
    "API_HOST": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "0.0.0.0",
        "description": "API server host",
        "is_secret": False,
        "value_type": "string",
    },
    "API_PORT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "8000",
        "description": "API server port",
        "is_secret": False,
        "value_type": "integer",
    },
    "API_PREFIX": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "/api/v1",
        "description": "API URL prefix",
        "is_secret": False,
        "value_type": "string",
    },
    "DASHBOARD_HOST": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "0.0.0.0",
        "description": "Dashboard server host",
        "is_secret": False,
        "value_type": "string",
    },
    "DASHBOARD_PORT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "8501",
        "description": "Dashboard server port",
        "is_secret": False,
        "value_type": "integer",
    },
    "DASHBOARD_REFRESH_RATE": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "5",
        "description": "Dashboard auto-refresh rate in seconds",
        "is_secret": False,
        "value_type": "integer",
    },
    "API_VERSION": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "1.0.0",
        "description": "API version string",
        "is_secret": False,
        "value_type": "string",
    },
    
    "SAFETY_VALIDATOR_TEMPERATURE": {
        "category": ConfigurationCategory.LLM,
        "default": "0.1",
        "description": "Temperature for safety validation (low for consistency)",
        "is_secret": False,
        "value_type": "float",
    },
    "SAFETY_VALIDATOR_MAX_TOKENS": {
        "category": ConfigurationCategory.LLM,
        "default": "1000",
        "description": "Maximum tokens for safety validation responses",
        "is_secret": False,
        "value_type": "integer",
    },
    
    # Timeouts (previously hardcoded)
    "HTTP_TIMEOUT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "10.0",
        "description": "Default HTTP request timeout in seconds",
        "is_secret": False,
        "value_type": "float",
    },
    "SCRAPER_TIMEOUT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "30.0",
        "description": "Market scraper request timeout in seconds",
        "is_secret": False,
        "value_type": "float",
    },
    "HEALTH_CHECK_TIMEOUT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "5.0",
        "description": "Health check request timeout in seconds",
        "is_secret": False,
        "value_type": "float",
    },
    "SUBPROCESS_TIMEOUT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "60",
        "description": "Subprocess execution timeout in seconds",
        "is_secret": False,
        "value_type": "integer",
    },
    
    # Retries (previously hardcoded)
    "DB_MAX_RETRIES": {
        "category": ConfigurationCategory.DATABASE,
        "default": "5",
        "description": "Maximum database connection retries on startup",
        "is_secret": False,
        "value_type": "integer",
    },
    "DB_RETRY_DELAY": {
        "category": ConfigurationCategory.DATABASE,
        "default": "5",
        "description": "Delay between database connection retries in seconds",
        "is_secret": False,
        "value_type": "integer",
    },
    "CONTENT_MAX_RETRIES": {
        "category": ConfigurationCategory.LLM,
        "default": "3",
        "description": "Maximum retries for content generation",
        "is_secret": False,
        "value_type": "integer",
    },
    "CONNECTOR_MAX_RETRIES": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "3",
        "description": "Maximum retries for external API connectors",
        "is_secret": False,
        "value_type": "integer",
    },
    "CONNECTOR_RETRY_DELAY": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "2.0",
        "description": "Delay between connector retries in seconds",
        "is_secret": False,
        "value_type": "float",
    },
    
    # Simulation - MAPE and Calibration (previously hardcoded)
    "MAPE_TARGET_THRESHOLD": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "10.0",
        "description": "Target MAPE threshold percentage for passing calibration",
        "is_secret": False,
        "value_type": "float",
    },
    "AD_FATIGUE_THRESHOLD": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "5",
        "description": "Impressions before ad fatigue kicks in",
        "is_secret": False,
        "value_type": "integer",
    },
    "AD_FATIGUE_DECAY": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "0.1",
        "description": "Ad fatigue decay rate",
        "is_secret": False,
        "value_type": "float",
    },
    "VALIDATION_ACCURACY_THRESHOLD": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "0.9",
        "description": "Validation accuracy threshold (0.9 = 90%)",
        "is_secret": False,
        "value_type": "float",
    },
    "USE_LEGACY_CALIBRATION": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "False",
        "description": "Use legacy calibration method instead of advanced",
        "is_secret": False,
        "value_type": "boolean",
    },
    "SIMULATION_DEFAULT_PERSONAS": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "",
        "description": "JSON array of default simulation personas. Each: {id, name, daily_active_prob, click_prob, conversion_prob}. Leave empty for built-in defaults.",
        "is_secret": False,
        "value_type": "string",
    },
    "SIMULATION_TRENDING_TOPICS": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "",
        "description": "JSON array of trending topics for simulation market state. Leave empty for HR tech domain defaults.",
        "is_secret": False,
        "value_type": "string",
    },
    "EMAIL_BENCHMARK_RATES": {
        "category": ConfigurationCategory.SIMULATION,
        "default": "",
        "description": "JSON object with email benchmark rates: {open_rate, click_rate, unsubscribe_rate}. Leave empty for B2B industry defaults (21.5% open, 2.5% click, 0.2% unsub).",
        "is_secret": False,
        "value_type": "string",
    },
    
    "PROMETHEUS_URL": {
        "category": ConfigurationCategory.MONITORING,
        "default": "http://localhost:9090",
        "description": "Prometheus server URL for metrics queries",
        "is_secret": False,
        "value_type": "string",
    },
    "ALERT_EMAIL": {
        "category": ConfigurationCategory.MONITORING,
        "default": "",
        "description": "Email address for system alerts",
        "is_secret": False,
        "value_type": "string",
    },
    
    "HITL_DEFAULT_TIMEOUT": {
        "category": ConfigurationCategory.GOVERNANCE,
        "default": "3600",
        "description": "Human-in-the-loop review timeout in seconds",
        "is_secret": False,
        "value_type": "integer",
    },
    
    "TRANSFORMER_MODEL": {
        "category": ConfigurationCategory.LEARNING,
        "default": "bert-base-uncased",
        "description": "Transformer model for contextual embeddings",
        "is_secret": False,
        "value_type": "string",
    },
    "CAUSAL_MODEL": {
        "category": ConfigurationCategory.LEARNING,
        "default": "doubly_robust",
        "description": "Causal inference model type",
        "is_secret": False,
        "value_type": "string",
    },
    
    "MLFLOW_TRACKING_URI": {
        "category": ConfigurationCategory.MLOPS,
        "default": "http://localhost:5000",
        "description": "MLflow tracking server URL",
        "is_secret": False,
        "value_type": "string",
    },
    "GRAFANA_URL": {
        "category": ConfigurationCategory.MONITORING,
        "default": "http://localhost:3000",
        "description": "Grafana dashboard URL",
        "is_secret": False,
        "value_type": "string",
    },

    # --- Keys migrated from settings.py (no more Pydantic field duplication) ---

    "EXPERIMENT_NAME": {
        "category": ConfigurationCategory.MLOPS,
        "default": "agentic-marketing-agents",
        "description": "MLflow experiment name",
        "is_secret": False,
        "value_type": "string",
    },
    "CLAIM_LIBRARY_PATH": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "config/prompts/claim_library.yaml",
        "description": "Path to the claim library YAML",
        "is_secret": False,
        "value_type": "string",
    },
    "MAX_CLAIMS_PER_CONTENT": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "3",
        "description": "Maximum number of claims to inject per piece of content",
        "is_secret": False,
        "value_type": "integer",
    },
    "GOLDEN_TEST_PATH": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "tests/golden/test_suite.yaml",
        "description": "Path to golden test suite YAML",
        "is_secret": False,
        "value_type": "string",
    },
    "GOLDEN_TEST_MIN_PASS_RATE": {
        "category": ConfigurationCategory.APPLICATION,
        "default": "1.0",
        "description": "Minimum golden test pass rate (0.0-1.0)",
        "is_secret": False,
        "value_type": "float",
    },
    "ENABLE_RESEARCH_MODE": {
        "category": ConfigurationCategory.FEATURE_FLAGS,
        "default": "true",
        "description": "Enable advanced research/experiment mode",
        "is_secret": False,
        "value_type": "boolean",
    },
    "STRATEGY_OPTIMIZER_CONTEXT_DIM": {
        "category": ConfigurationCategory.LEARNING,
        "default": "50",
        "description": "Strategy optimizer context vector dimension",
        "is_secret": False,
        "value_type": "integer",
    },
    "NEURAL_BANDIT_HIDDEN_DIM": {
        "category": ConfigurationCategory.LEARNING,
        "default": "128",
        "description": "Neural bandit hidden layer dimension",
        "is_secret": False,
        "value_type": "integer",
    },
    "OLLAMA_NUM_GPU": {
        "category": ConfigurationCategory.LLM,
        "default": "1",
        "description": "Number of GPUs for Ollama",
        "is_secret": False,
        "value_type": "integer",
    },
    "OLLAMA_GPU_LAYERS": {
        "category": ConfigurationCategory.LLM,
        "default": "35",
        "description": "Number of GPU layers for Ollama model offloading",
        "is_secret": False,
        "value_type": "integer",
    },
    "VECTOR_SEARCH_K": {
        "category": ConfigurationCategory.LLM,
        "default": "5",
        "description": "Number of results for vector similarity search",
        "is_secret": False,
        "value_type": "integer",
    },
    "VECTOR_SEARCH_METRIC": {
        "category": ConfigurationCategory.LLM,
        "default": "cosine",
        "description": "Vector similarity search metric",
        "is_secret": False,
        "value_type": "string",
    },
    "PERSPECTIVE_API_KEY": {
        "category": ConfigurationCategory.GOVERNANCE,
        "default": "",
        "description": "Google Perspective API key for toxicity scoring",
        "is_secret": True,
        "value_type": "string",
    },
    "MAILCHIMP_API_KEY": {
        "category": ConfigurationCategory.EMAIL,
        "default": "",
        "description": "Mailchimp API key",
        "is_secret": True,
        "value_type": "string",
    },
    "MAILCHIMP_FROM_EMAIL": {
        "category": ConfigurationCategory.EMAIL,
        "default": "marketing@example.com",
        "description": "Mailchimp sender email address",
        "is_secret": False,
        "value_type": "string",
    },
    "MAILCHIMP_FROM_NAME": {
        "category": ConfigurationCategory.EMAIL,
        "default": "Agentic AI",
        "description": "Mailchimp sender display name",
        "is_secret": False,
        "value_type": "string",
    },
    "MAILCHIMP_LIST_ID": {
        "category": ConfigurationCategory.EMAIL,
        "default": "",
        "description": "Mailchimp audience list ID",
        "is_secret": False,
        "value_type": "string",
    },
}


class ConfigurationService:
    """
    Service for managing system configurations in the database.
    Handles encryption, caching, and CRUD operations.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self._cache: Dict[str, Any] = {}
        self._cache_loaded = False
    
    def initialize_defaults(self) -> int:
        """
        Initialize database with default configurations.
        Only creates entries that don't already exist.
        
        Returns:
            Number of configurations created
        """
        created = 0
        for key, config in DEFAULT_CONFIGURATIONS.items():
            existing = self.db.query(SystemConfiguration).filter_by(key=key).first()
            if not existing:
                value = config["default"]
                if config["is_secret"] and value:
                    value = encrypt_value(value)
                
                new_config = SystemConfiguration(
                    key=key,
                    value=value,
                    category=config["category"],
                    is_secret=config["is_secret"],
                    description=config["description"],
                    default_value=config["default"],
                    value_type=config["value_type"],
                )
                self.db.add(new_config)
                created += 1
        
        if created > 0:
            self.db.commit()
            logger.info(f"Initialized {created} default configurations")
        
        return created
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        Checks: DB → environment variable → DEFAULT_CONFIGURATIONS → default param.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value (decrypted if secret)
        """
        config = self.db.query(SystemConfiguration).filter_by(key=key).first()
        
        if not config:
            env_val = os.environ.get(key)
            if env_val:
                return env_val
            if key in DEFAULT_CONFIGURATIONS:
                return self._convert_value(
                    DEFAULT_CONFIGURATIONS[key]["default"],
                    DEFAULT_CONFIGURATIONS[key]["value_type"]
                )
            return default
        
        value = config.value
        
        # If DB value is empty, check environment variable
        if not value or not str(value).strip():
            env_val = os.environ.get(key)
            if env_val:
                return env_val
        
        if config.is_secret and value:
            try:
                value = decrypt_value(value)
            except Exception as e:
                logger.error(f"Failed to decrypt {key}: {e}")
                return default
        
        return self._convert_value(value, config.value_type)
    
    def set_value(self, key: str, value: Any, category: ConfigurationCategory = None) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            category: Category (required for new keys)
            
        Returns:
            True if successful
        """
        config = self.db.query(SystemConfiguration).filter_by(key=key).first()
        
        if config:
            str_value = str(value) if value is not None else ""
            
            if config.is_secret and str_value:
                str_value = encrypt_value(str_value)
            
            config.value = str_value
            config.updated_at = datetime.utcnow()
        else:
            if category is None:
                if key in DEFAULT_CONFIGURATIONS:
                    category = DEFAULT_CONFIGURATIONS[key]["category"]
                else:
                    raise ValueError(f"Category required for new configuration: {key}")
            
            is_secret = DEFAULT_CONFIGURATIONS.get(key, {}).get("is_secret", False)
            value_type = DEFAULT_CONFIGURATIONS.get(key, {}).get("value_type", "string")
            description = DEFAULT_CONFIGURATIONS.get(key, {}).get("description", "")
            
            str_value = str(value) if value is not None else ""
            if is_secret and str_value:
                str_value = encrypt_value(str_value)
            
            config = SystemConfiguration(
                key=key,
                value=str_value,
                category=category,
                is_secret=is_secret,
                value_type=value_type,
                description=description,
            )
            self.db.add(config)
        
        try:
            self.db.commit()
            self._cache[key] = value
            return True
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Failed to set {key}: {e}")
            return False
    
    def get_by_category(self, category: ConfigurationCategory) -> List[Dict[str, Any]]:
        """
        Get all configurations for a category.
        
        Args:
            category: Configuration category
            
        Returns:
            List of configuration dictionaries
        """
        configs = self.db.query(SystemConfiguration).filter_by(category=category).all()
        
        result = []
        for config in configs:
            value = config.value
            display_value = value
            
            if config.is_secret and value:
                try:
                    decrypted = decrypt_value(value)
                    display_value = mask_sensitive_value(decrypted)
                except Exception:
                    display_value = "••••••••"
            
            result.append({
                "key": config.key,
                "value": value if not config.is_secret else None,  # Don't expose encrypted value
                "display_value": display_value,
                "category": config.category.value,
                "is_secret": config.is_secret,
                "description": config.description,
                "default_value": config.default_value,
                "value_type": config.value_type,
                "updated_at": config.updated_at.isoformat() if config.updated_at else None,
            })
        
        return result
    
    def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        Get summary of all configuration categories.
        
        Returns:
            List of category summaries
        """
        categories = []
        for cat in ConfigurationCategory:
            count = self.db.query(SystemConfiguration).filter_by(category=cat).count()
            
            configured = self.db.query(SystemConfiguration).filter(
                SystemConfiguration.category == cat,
                SystemConfiguration.value.isnot(None),
                SystemConfiguration.value != ""
            ).count()
            
            categories.append({
                "category": cat.value,
                "display_name": cat.value.replace("_", " ").title(),
                "total_settings": count,
                "configured_count": configured,
            })
        
        return categories
    
    def is_configured(self) -> bool:
        """
        Check if the system has been configured.
        
        Returns:
            True if essential configurations are set
        """
        essential_keys = ["DATABASE_URL", "REDIS_URL"]
        
        for key in essential_keys:
            config = self.db.query(SystemConfiguration).filter_by(key=key).first()
            if not config or not config.value:
                return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall configuration status.
        
        Returns:
            Status dictionary
        """
        total = self.db.query(SystemConfiguration).count()
        configured = self.db.query(SystemConfiguration).filter(
            SystemConfiguration.value.isnot(None),
            SystemConfiguration.value != ""
        ).count()
        
        return {
            "is_configured": self.is_configured(),
            "total_settings": total,
            "configured_count": configured,
            "categories": self.get_all_categories(),
        }
    
    def bulk_update(self, updates: Dict[str, Any]) -> Dict[str, bool]:
        """
        Update multiple configurations at once.
        
        Args:
            updates: Dictionary of key-value pairs
            
        Returns:
            Dictionary of results per key
        """
        results = {}
        for key, value in updates.items():
            results[key] = self.set_value(key, value)
        return results
    
    def _convert_value(self, value: str, value_type: str) -> Any:
        """Convert string value to appropriate type."""
        if value is None or value == "":
            return None
        
        try:
            if value_type == "integer":
                return int(value)
            elif value_type == "float":
                return float(value)
            elif value_type == "boolean":
                return value.lower() in ("true", "1", "yes", "on")
            elif value_type == "json":
                import json
                return json.loads(value)
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    def export_to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export all configurations to a dictionary.
        
        Args:
            include_secrets: Whether to include decrypted secrets
            
        Returns:
            Dictionary of all configurations
        """
        configs = self.db.query(SystemConfiguration).all()
        result = {}
        
        for config in configs:
            value = config.value
            if config.is_secret and value and include_secrets:
                try:
                    value = decrypt_value(value)
                except Exception:
                    value = None
            elif config.is_secret:
                value = None
            
            result[config.key] = self._convert_value(value, config.value_type)
        
        return result


def get_configuration_service(db_session: Session) -> ConfigurationService:
    """Factory function to get ConfigurationService instance."""
    return ConfigurationService(db_session)


_runtime_config_cache: Dict[str, Any] = {}
_cache_timestamp: Optional[datetime] = None
_CACHE_TTL_SECONDS = 60  # Cache config for 60 seconds


def get_runtime_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value at runtime from the database.
    Uses caching to avoid excessive database queries.
    
    This is the SINGLE SOURCE OF TRUTH for configuration values.
    All workflow code should use this instead of env vars.
    
    Args:
        key: Configuration key (e.g., 'REQUIRE_HUMAN_APPROVAL')
        default: Default value if not found
        
    Returns:
        Configuration value (properly typed)
    """
    global _runtime_config_cache, _cache_timestamp
    
    now = datetime.utcnow()
    
    if _cache_timestamp and (now - _cache_timestamp).total_seconds() < _CACHE_TTL_SECONDS:
        if key in _runtime_config_cache:
            return _runtime_config_cache[key]
    
    try:
        from src.data_layer.database.connection import get_sync_db_session
        
        with get_sync_db_session() as db:
            service = ConfigurationService(db)
            value = service.get_value(key, default)
            _runtime_config_cache[key] = value
            _cache_timestamp = now
            return value
    except Exception as e:
        logger.warning(f"Failed to get config {key} from database: {e}, using default")
        if key in DEFAULT_CONFIGURATIONS:
            config_def = DEFAULT_CONFIGURATIONS[key]
            return _convert_value_static(config_def["default"], config_def["value_type"])
        return default


def refresh_runtime_config_cache() -> Dict[str, Any]:
    """
    Force refresh of the runtime configuration cache.
    Returns all configuration values.
    """
    global _runtime_config_cache, _cache_timestamp
    
    try:
        from src.data_layer.database.connection import get_sync_db_session
        
        with get_sync_db_session() as db:
            service = ConfigurationService(db)
            _runtime_config_cache = service.export_to_dict(include_secrets=False)
            _cache_timestamp = datetime.utcnow()
            logger.info(f"Refreshed runtime config cache with {len(_runtime_config_cache)} values")
            return _runtime_config_cache
    except Exception as e:
        logger.error(f"Failed to refresh config cache: {e}")
        return {}


_get_config_value = get_runtime_config


def _convert_value_static(value: str, value_type: str) -> Any:
    """Static version of value conversion for use without service instance."""
    if value is None or value == "":
        return None
    
    try:
        if value_type == "integer":
            return int(value)
        elif value_type == "float":
            return float(value)
        elif value_type == "boolean":
            return value.lower() in ("true", "1", "yes", "on")
        else:
            return value
    except (ValueError, TypeError):
        return value
