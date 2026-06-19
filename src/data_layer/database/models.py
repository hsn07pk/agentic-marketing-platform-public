"""
SQLAlchemy database models for the Agentic AI Agent Platform

NOTE: Enums are imported from src/shared/constants.py - the SINGLE SOURCE OF TRUTH.
Do NOT define duplicate enums here.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, UniqueConstraint, Index, Enum as SQLEnum, 
    create_engine, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from pgvector.sqlalchemy import Vector
import uuid

from src.shared.constants import (
    CampaignStatus,
    ContentStatus,
    Platform,
    WorkflowEventType,
    AlertSeverity,
    CampaignGoal,
)

Base = declarative_base()

class AgentType(str, Enum):
    """Types of agents in the system"""
    CONTENT_GENERATOR = "content_generator"
    STRATEGY_OPTIMIZER = "strategy_optimizer"
    SAFETY_VALIDATOR = "safety_validator"
    MARKET_SCRAPER = "market_scraper"
    DEPLOYER = "deployer"

class ConfigurationCategory(str, Enum):
    """Configuration categories for system settings"""
    DATABASE = "database"
    REDIS = "redis"
    LLM = "llm"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    EMAIL = "email"
    BLOG = "blog"
    HUBSPOT = "hubspot"
    CALENDAR = "calendar"
    APIFY = "apify"
    GOVERNANCE = "governance"
    COST_CONTROL = "cost_control"
    SIMULATION = "simulation"
    LEARNING = "learning"
    MARL = "marl"
    MONITORING = "monitoring"
    MLOPS = "mlops"
    SECURITY = "security"
    FEATURE_FLAGS = "feature_flags"
    APPLICATION = "application"

class SystemConfiguration(Base):
    """
    Database-backed system configuration storage.
    Replaces .env file configuration with encrypted storage for sensitive values.
    """
    __tablename__ = "system_configurations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    key = Column(String(100), unique=True, nullable=False, index=True)
    
    value = Column(Text)
    
    category = Column(SQLEnum(ConfigurationCategory), nullable=False, index=True)
    
    is_secret = Column(Boolean, default=False)
    
    description = Column(Text)
    
    default_value = Column(Text)
    
    value_type = Column(String(20), default="string")
    
    validation = Column(JSONB, default={})
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_config_category", "category"),
        Index("idx_config_key", "key"),
    )
    
    def __repr__(self):
        return f"<SystemConfiguration(key='{self.key}', category='{self.category}')>"

class Campaign(Base):
    """Marketing campaign model"""
    __tablename__ = "campaigns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    platform = Column(SQLEnum(Platform), nullable=False)
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT)
    goal = Column(SQLEnum(CampaignGoal), nullable=True)

    target_persona = Column(String(100))
    target_keywords = Column(ARRAY(String))
    target_demographics = Column(JSONB)
    
    budget_total = Column(Float, default=0.0)
    budget_spent = Column(Float, default=0.0)
    budget_daily_limit = Column(Float)
    
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    demos_booked = Column(Integer, default=0)  # Cal.com booking conversions
    ctr = Column(Float, default=0.0)
    cpl = Column(Float, default=0.0)
    
    config = Column(JSONB, default={})
    
    is_mock = Column(Boolean, default=False, index=True)
    
    contents = relationship("Content", back_populates="campaign", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="campaign")
    metrics = relationship("Metric", back_populates="campaign")
    
    __table_args__ = (
        Index("idx_campaign_status", "status"),
        Index("idx_campaign_platform", "platform"),
    )

class Content(Base):
    """Generated marketing content"""
    __tablename__ = "contents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    
    content_type = Column(String(50))
    headline = Column(Text)
    body = Column(Text, nullable=False)
    cta = Column(Text)
    image_url = Column(String(500))
    platform_specific = Column(JSONB, default={})

    generated_by = Column(SQLEnum(AgentType), default=AgentType.CONTENT_GENERATOR)
    prompt_used = Column(Text)
    model_used = Column(String(50))
    claims_used = Column(ARRAY(String))
    
    status = Column(SQLEnum(ContentStatus), default=ContentStatus.GENERATED)
    safety_score = Column(Float)
    toxicity_score = Column(Float)
    factuality_score = Column(Float)
    brand_alignment_score = Column(Float)
    review_notes = Column(Text)
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    
    impressions = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime)
    
    generation_cost = Column(Float, default=0.0)
    
    is_mock = Column(Boolean, default=False, index=True)
    
    campaign = relationship("Campaign", back_populates="contents")
    
    __table_args__ = (
        Index("idx_content_status", "status"),
        Index("idx_content_safety_score", "safety_score"),
    )

class Persona(Base):
    """Customer persona definitions"""
    __tablename__ = "personas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    title = Column(String(200))
    description = Column(Text)
    role = Column(String(100))
    seniority = Column(String(50))
    industry = Column(String(100))
    
    daily_active_prob = Column(Float, default=0.5)
    content_engagement_prob = Column(Float, default=0.1)
    click_prob = Column(Float, default=0.05)
    conversion_prob = Column(Float, default=0.01)
    share_prob = Column(Float, default=0.02)
    
    preferred_content_types = Column(ARRAY(String))
    preferred_channels = Column(ARRAY(String))
    active_hours = Column(ARRAY(Integer))
    
    company_size = Column(String(50))
    budget_range = Column(String(50))
    pain_points = Column(ARRAY(String))
    goals = Column(ARRAY(String))
    
    attributes = Column(JSONB, default={})
    preferences = Column(JSONB, default={})

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Experiment(Base):
    """A/B/n testing experiments"""
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"))
    name = Column(String(255), nullable=False)
    type = Column(String(50))
    
    variants = Column(JSONB)
    algorithm = Column(String(50))
    parameters = Column(JSONB)
    
    is_active = Column(Boolean, default=True)
    winner_variant = Column(String(100))
    results = Column(JSONB)

    total_impressions = Column(Integer, default=0)
    total_conversions = Column(Integer, default=0)

    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)

    campaign = relationship("Campaign", back_populates="experiments")
    arms = relationship("BanditArm", back_populates="experiment")

class BanditArm(Base):
    """Bandit algorithm arms for experiments"""
    __tablename__ = "bandit_arms"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"))
    
    arm_id = Column(String(100), nullable=False)
    variant_data = Column(JSONB)
    
    alpha = Column(Float, default=1.0)  # Success count + 1
    beta = Column(Float, default=1.0)   # Failure count + 1
    
    pulls = Column(Integer, default=0)
    successes = Column(Integer, default=0)
    failures = Column(Integer, default=0)
    total_reward = Column(Float, default=0.0)
    
    context_vector = Column(Vector(dim=50))
    
    last_pulled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    experiment = relationship("Experiment", back_populates="arms")
    
    __table_args__ = (
        UniqueConstraint("experiment_id", "arm_id"),
    )

class Metric(Base):
    """Time-series metrics storage"""
    __tablename__ = "metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50))
    
    platform = Column(String(50))
    agent_type = Column(String(50))
    tags = Column(JSONB, default={})
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    campaign = relationship("Campaign", back_populates="metrics")
    
    __table_args__ = (
        Index("idx_metrics_timestamp", "timestamp"),
        Index("idx_metrics_name", "metric_name"),
    )

class AgentAction(Base):
    """Audit log of all agent actions"""
    __tablename__ = "agent_actions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    agent_type = Column(SQLEnum(AgentType), nullable=False)
    action = Column(String(100), nullable=False)
    
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    content_id = Column(UUID(as_uuid=True), ForeignKey("contents.id"), nullable=True)
    
    input_data = Column(JSONB)
    output_data = Column(JSONB)
    error = Column(Text)
    
    tokens_used = Column(Integer, default=0)
    api_cost = Column(Float, default=0.0)
    
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    
    __table_args__ = (
        Index("idx_agent_actions_timestamp", "started_at"),
        Index("idx_agent_actions_agent", "agent_type"),
    )

class HITLQueue(Base):
    """Human-in-the-loop review queue"""
    __tablename__ = "hitl_queue"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_id = Column(UUID(as_uuid=True), ForeignKey("contents.id"), nullable=False)
    
    priority = Column(Integer, default=0)
    reason = Column(Text)
    
    status = Column(String(50), default="pending")
    assigned_to = Column(String(100))
    
    decision = Column(String(50))
    feedback = Column(Text)
    modifications = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    assigned_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    __table_args__ = (
        Index("idx_hitl_status", "status"),
        Index("idx_hitl_priority", "priority"),
    )

class VectorStore(Base):
    """Storage for embeddings and vector search"""
    __tablename__ = "vector_store"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    doc_type = Column(String(50), nullable=False)
    doc_id = Column(String(255))
    doc_text = Column(Text, nullable=False)
    
    embedding = Column(Vector(dim=1536))  # OpenAI embedding dimension

    meta_data = Column(JSONB, default={})
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_vector_doc_type", "doc_type"),
    )

class CostTracking(Base):
    """Track all costs in the system"""
    __tablename__ = "cost_tracking"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    source_type = Column(String(50))
    source_id = Column(String(255))
    
    cost_amount = Column(Float, nullable=False)
    cost_currency = Column(String(3), default="EUR")
    
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    agent_type = Column(String(50))
    
    provider = Column(String(50))
    tokens_prompt = Column(Integer)
    tokens_completion = Column(Integer)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_cost_timestamp", "timestamp"),
        Index("idx_cost_campaign", "campaign_id"),
    )

class DelayedReward(Base):
    """Track delayed rewards for campaign conversions"""
    __tablename__ = "delayed_rewards"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)

    lead_email = Column(String(255), nullable=False)
    lead_data = Column(JSON)

    initial_reward = Column(Float, default=1.0)
    current_reward = Column(Float, default=1.0)
    status = Column(String(20), default="pending")

    registered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)

    meeting_scheduled = Column(Boolean, default=False)
    meeting_date = Column(DateTime, nullable=True)
    meeting_attended = Column(Boolean, default=False)

    hubspot_contact_id = Column(String(50))
    lead_score = Column(Integer)

    __table_args__ = (
        Index("idx_delayed_reward_campaign", "campaign_id"),
        Index("idx_delayed_reward_status", "status"),
        Index("idx_delayed_reward_registered", "registered_at"),
    )

class ScrapedContent(Base):
    """Scraped market intelligence and competitive insights"""
    __tablename__ = "scraped_content"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    source = Column(String(50), nullable=False)
    source_url = Column(String(500))

    keywords = Column(Text)

    raw_content = Column(Text)
    insights = Column(JSONB, default={})

    scraped_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_scraped_content_source", "source"),
        Index("idx_scraped_content_scraped_at", "scraped_at"),
    )

class CalibrationRun(Base):
    """Track simulation calibration experiments"""
    __tablename__ = "calibration_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    name = Column(String(255), nullable=False)
    description = Column(Text)

    historical_data_source = Column(String(255))
    num_training_campaigns = Column(Integer)
    num_validation_campaigns = Column(Integer)

    validation_mape = Column(Float)
    validation_accuracy = Column(Float)  # 100 - MAPE
    passes_threshold = Column(Boolean, default=False)  # MAPE < 10%

    metrics = Column(JSONB, default={})  # Per-metric MAPE, etc.

    optimization_method = Column(String(50))  # 'differential_evolution', 'minimize'
    random_seed = Column(Integer, default=42)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)

    # Status
    status = Column(String(20), default="running")
    error_message = Column(Text)

    __table_args__ = (
        Index("idx_calibration_status", "status"),
        Index("idx_calibration_started", "started_at"),
    )

class WorkflowEvent(Base):
    """Track all workflow events for complete transparency"""
    __tablename__ = "workflow_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    content_id = Column(UUID(as_uuid=True), ForeignKey("contents.id"), nullable=True)

    event_type = Column(SQLEnum(WorkflowEventType), nullable=False)
    severity = Column(SQLEnum(AlertSeverity), default=AlertSeverity.INFO, nullable=False)

    workflow_node = Column(String(100))
    workflow_state = Column(String(50))

    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)

    details = Column(JSONB, default={})

    is_user_actionable = Column(Boolean, default=False)  # Requires user action
    is_dismissed = Column(Boolean, default=False)
    dismissed_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    campaign = relationship("Campaign", backref="workflow_events")
    content = relationship("Content", backref="workflow_events")

    __table_args__ = (
        Index("idx_workflow_event_campaign", "campaign_id"),
        Index("idx_workflow_event_type", "event_type"),
        Index("idx_workflow_event_severity", "severity"),
        Index("idx_workflow_event_created", "created_at"),
        Index("idx_workflow_event_actionable", "is_user_actionable"),
        Index("idx_workflow_event_dismissed", "is_dismissed"),
    )

class FileIngestionStatus(str, Enum):
    """File ingestion status"""
    PENDING = "pending"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    FAILED = "failed"
    DELETED = "deleted"

class DataFileIngestion(Base):
    """Track ingestion status of files in data/ directory"""
    __tablename__ = "data_file_ingestion"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    file_path = Column(String(500), nullable=False, unique=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_category = Column(String(100))

    file_size_bytes = Column(Integer)
    file_hash = Column(String(64))
    last_modified_at = Column(DateTime)

    status = Column(SQLEnum(FileIngestionStatus), default=FileIngestionStatus.PENDING, nullable=False)
    ingestion_started_at = Column(DateTime)
    ingestion_completed_at = Column(DateTime)
    ingestion_duration_seconds = Column(Float)

    error_message = Column(Text)
    retry_count = Column(Integer, default=0)

    document_ids = Column(ARRAY(Integer))
    num_chunks = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_file_ingestion_status", "status"),
        Index("idx_file_ingestion_category", "file_category"),
        Index("idx_file_ingestion_path", "file_path"),
    )

class SimulationLiveAccuracy(Base):
    """
    Track simulation-to-live accuracy for RQ2 (>90% accuracy target)

    Research Plan Section 5.3: The goal is to achieve a MAPE of less than 10%,
    satisfying the ">90% accuracy" requirement.
    """
    __tablename__ = "simulation_live_accuracy"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)

    simulated_impressions = Column(Integer, default=0)
    simulated_clicks = Column(Integer, default=0)
    simulated_conversions = Column(Integer, default=0)
    simulated_ctr = Column(Float, default=0.0)
    simulated_cpl = Column(Float, default=0.0)

    actual_impressions = Column(Integer, default=0)
    actual_clicks = Column(Integer, default=0)
    actual_conversions = Column(Integer, default=0)
    actual_ctr = Column(Float, default=0.0)
    actual_cpl = Column(Float, default=0.0)

    mape_impressions = Column(Float)
    mape_clicks = Column(Float)
    mape_conversions = Column(Float)
    mape_ctr = Column(Float)
    mape_cpl = Column(Float)

    overall_mape = Column(Float)
    overall_accuracy = Column(Float)  # 100 - MAPE (target: >90%)
    passes_threshold = Column(Boolean, default=False)  # accuracy >= 90%

    rq2_target = Column(Float, default=90.0)
    rq2_gap = Column(Float)  # Gap to target (negative = exceeds target)

    simulation_timestamp = Column(DateTime)
    measurement_timestamp = Column(DateTime, default=datetime.utcnow)
    measurement_type = Column(String(50))

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    campaign = relationship("Campaign", backref="simulation_accuracy_records")

    __table_args__ = (
        Index("idx_sim_accuracy_campaign", "campaign_id"),
        Index("idx_sim_accuracy_passes", "passes_threshold"),
        Index("idx_sim_accuracy_created", "created_at"),
    )

class GovernanceMetrics(Base):
    """
    Track governance metrics including human override rate

    Research Plan Section 10.2: Human Override Rate < 5% target
    """
    __tablename__ = "governance_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20))

    total_reviews = Column(Integer, default=0)
    approved_count = Column(Integer, default=0)
    rejected_count = Column(Integer, default=0)
    modified_count = Column(Integer, default=0)

    # Override rate (Research Plan target: <5%)
    human_override_rate = Column(Float, default=0.0)  # (rejected + modified) / total * 100
    override_rate_target = Column(Float, default=5.0)
    meets_override_target = Column(Boolean, default=True)

    avg_safety_score = Column(Float)
    avg_toxicity_score = Column(Float)
    avg_factuality_score = Column(Float)
    avg_brand_alignment_score = Column(Float)

    golden_test_pass_rate = Column(Float)
    golden_tests_run = Column(Integer, default=0)
    golden_tests_passed = Column(Integer, default=0)

    auto_approved_count = Column(Integer, default=0)
    auto_approval_rate = Column(Float, default=0.0)

    avg_review_time_minutes = Column(Float)
    median_review_time_minutes = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_gov_metrics_period", "period_start", "period_end"),
        Index("idx_gov_metrics_type", "period_type"),
    )

class WeeklyLearningReport(Base):
    """
    Automated weekly learning reports

    Research Plan Section 10.2: Weekly Uplift Summary
    """
    __tablename__ = "weekly_learning_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    week_start = Column(DateTime, nullable=False)
    week_end = Column(DateTime, nullable=False)
    week_number = Column(Integer)
    year = Column(Integer)

    best_hooks = Column(JSONB, default=[])  # [{hook: str, ctr: float, conversions: int}]
    worst_hooks = Column(JSONB, default=[])

    platform_performance = Column(JSONB, default={})

    persona_performance = Column(JSONB, default={})

    bandit_insights = Column(JSONB, default={})
    regret_cumulative = Column(Float)
    exploration_exploitation_ratio = Column(Float)

    ctr_this_week = Column(Float)
    ctr_last_week = Column(Float)
    ctr_change_pct = Column(Float)

    conversions_this_week = Column(Integer)
    conversions_last_week = Column(Integer)
    conversions_change_pct = Column(Float)

    cpl_this_week = Column(Float)
    cpl_last_week = Column(Float)
    cpl_change_pct = Column(Float)

    recommendations = Column(JSONB, default=[])

    generated_at = Column(DateTime, default=datetime.utcnow)
    generated_by = Column(String(100), default="system")

    __table_args__ = (
        Index("idx_weekly_report_period", "week_start", "week_end"),
        UniqueConstraint("week_number", "year", name="uq_weekly_report_week_year"),
    )

class PersonaCalibration(Base):
    """Store calibrated persona parameters"""
    __tablename__ = "persona_calibrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    calibration_run_id = Column(UUID(as_uuid=True), ForeignKey("calibration_runs.id"), nullable=False)

    persona_name = Column(String(100), nullable=False)
    platform = Column(String(50))

    daily_active_prob = Column(Float, nullable=False)
    click_prob = Column(Float, nullable=False)
    conversion_prob = Column(Float, nullable=False)
    content_engagement_prob = Column(Float, default=0.1)
    share_prob = Column(Float, default=0.02)

    ad_fatigue_threshold = Column(Integer, default=5)
    ad_fatigue_decay = Column(Float, default=0.15)
    influence_factor = Column(Float, default=0.3)

    active_hours = Column(JSONB, default=[9, 10, 11, 14, 15, 16])

    training_mape = Column(Float)
    num_training_samples = Column(Integer)

    is_active = Column(Boolean, default=True)
    version = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    calibration_run = relationship("CalibrationRun", backref="persona_calibrations")

    __table_args__ = (
        Index("idx_persona_calibration_name", "persona_name"),
        Index("idx_persona_calibration_active", "is_active"),
        UniqueConstraint("persona_name", "platform", "version", name="uq_persona_platform_version"),
    )

class CanaryDeployment(Base):
    """
    Canary Deployment tracking for production readiness.
    
    Per Research Plan Section 2.3: Canary Rollout with monitoring and rollback.
    Persists deployment state to database for durability across restarts.
    """
    __tablename__ = "canary_deployments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_id = Column(String(255), unique=True, nullable=False, index=True)
    
    policy_id = Column(String(255), nullable=False)
    policy_version = Column(String(50), default="1.0.0")
    deployment_type = Column(String(50), default="policy")
    
    status = Column(String(50), nullable=False, default="pending")
    # Status values: pending, canary_5_percent, canary_25_percent, canary_50_percent, 
    #                canary_75_percent, full_rollout_100_percent, rolled_back, failed
    
    current_traffic_percentage = Column(Float, default=0.05)
    
    baseline_ctr = Column(Float)
    baseline_conversion_rate = Column(Float)
    baseline_cpl = Column(Float)
    
    auto_rollback_enabled = Column(Boolean, default=True)
    rollback_ctr_threshold = Column(Float, default=10.0)
    rollback_error_threshold = Column(Float, default=5.0)
    rollback_latency_threshold = Column(Integer, default=500)
    
    rollback_reason = Column(Text, nullable=True)
    
    # Extra data (campaign_id, description, etc.) - named 'extra_data' to avoid SQLAlchemy reserved 'metadata'
    extra_data = Column(JSONB, default={})
    
    metrics_history = Column(JSONB, default=[])
    
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_canary_status", "status"),
        Index("idx_canary_policy", "policy_id"),
        Index("idx_canary_started", "started_at"),
    )

class StrategyPerformance(Base):
    """
    Strategy Performance tracking for bandit optimizer.
    
    Per Research Plan Section 4.1: Multi-Armed Bandit for strategy selection.
    Persists strategy decisions and rewards to database for durability across restarts.
    """
    __tablename__ = "strategy_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    campaign_id = Column(String(255), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    
    reward = Column(Float, nullable=False)
    
    context = Column(JSONB, default={})
    
    estimated_ctr = Column(Float, nullable=True)
    estimated_conversions = Column(Integer, nullable=True)
    estimated_cpl = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_strategy_campaign", "campaign_id"),
        Index("idx_strategy_action", "action"),
        Index("idx_strategy_created", "created_at"),
    )

def init_database(engine):
    """Initialize database with all tables"""
    Base.metadata.create_all(engine)
    
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()