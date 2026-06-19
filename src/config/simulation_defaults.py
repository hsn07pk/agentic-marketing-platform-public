"""
Centralized simulation defaults — single source of truth for all simulation parameters.

These defaults are used as fallbacks when no configuration is set in the config service.
All values can be overridden via Operations → System Settings in the dashboard.

Research Plan Reference:
- RQ2: Simulation accuracy (MAPE < 10%)
- Persona probabilities and benchmark rates directly affect simulation fidelity
"""

# Default persona parameters for simulation when none loaded from database.
# These serve as initial priors; calibrated personas from PersonaCalibration
# table should be used for production simulation runs.
DEFAULT_SIMULATION_PERSONAS = [
    {
        "id": "default_decision_maker",
        "name": "Decision Maker",
        "daily_active_prob": 0.5,
        "click_prob": 0.04,
        "conversion_prob": 0.02,
    },
    {
        "id": "default_technical",
        "name": "Technical Buyer",
        "daily_active_prob": 0.6,
        "click_prob": 0.06,
        "conversion_prob": 0.015,
    },
    {
        "id": "default_budget",
        "name": "Budget Holder",
        "daily_active_prob": 0.3,
        "click_prob": 0.03,
        "conversion_prob": 0.025,
    },
]

# Email platform benchmark rates (B2B industry averages).
# Source: Campaign Monitor / Mailchimp B2B benchmarks 2024.
DEFAULT_EMAIL_BENCHMARKS = {
    "open_rate": 0.215,
    "click_rate": 0.025,
    "unsubscribe_rate": 0.002,
}

# Trending topics for simulation market state evolution.
# Domain-relevant to HR tech / Employee Experience (Agentic).
DEFAULT_TRENDING_TOPICS = [
    "employee experience platforms",
    "people analytics adoption",
    "workplace wellbeing ROI",
    "AI-driven HR technology",
    "remote work engagement",
    "talent retention strategies",
    "QWL measurement science",
    "leadership action analytics",
    "organizational health metrics",
]

# Email subject line scoring word lists.
DEFAULT_SPAM_TRIGGERS = [
    "free", "act now", "urgent", "!!!",
    "click here", "guaranteed", "winner", "$$$",
]

DEFAULT_URGENCY_WORDS = [
    "limited", "exclusive", "today", "now", "deadline",
]

DEFAULT_PERSONALIZATION_TOKENS = [
    "{first_name}", "{company}", "{{", "}}",
]
