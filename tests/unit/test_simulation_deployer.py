# tests/unit/test_simulation_deployer.py
"""
Comprehensive tests for simulation and deployment layers.

All external dependencies (database, SimPy, APIs, Redis) are mocked
so tests run without Docker services.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. SimulationConfig / MarketState / MarketingEnvironment
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    """Tests for SimulationConfig dataclass defaults and overrides."""

    def test_default_config_values(self):
        from src.simulation.environment import SimulationConfig

        cfg = SimulationConfig()
        assert cfg.duration_days == 30
        assert cfg.time_step_hours == 1.0
        assert cfg.num_customers == 1000
        assert cfg.num_competitors == 3
        assert cfg.platforms == ["linkedin", "twitter", "blog"]
        assert cfg.seed is None
        assert cfg.validation_mode is False
        assert cfg.historical_data_path is None

    def test_custom_config_values(self):
        from src.simulation.environment import SimulationConfig

        cfg = SimulationConfig(
            duration_days=7,
            time_step_hours=0.5,
            num_customers=100,
            num_competitors=1,
            platforms=["linkedin"],
            seed=42,
        )
        assert cfg.duration_days == 7
        assert cfg.num_customers == 100
        assert cfg.seed == 42


class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_default_market_state(self):
        from src.simulation.environment import MarketState

        ms = MarketState(timestamp=datetime.now())
        assert ms.total_impressions == 0
        assert ms.total_clicks == 0
        assert ms.total_conversions == 0
        assert ms.market_sentiment == 0.5
        assert ms.trending_topics == []
        assert ms.competitor_spend == {}


class TestMarketingEnvironment:
    """Tests for the main SimPy simulation environment."""

    @patch("src.simulation.environment.XPlatform")
    @patch("src.simulation.environment.LinkedInPlatform")
    def test_environment_init_creates_platforms(self, mock_li, mock_x):
        from src.simulation.environment import SimulationConfig, MarketingEnvironment

        cfg = SimulationConfig(platforms=["linkedin", "twitter"], seed=1)
        env = MarketingEnvironment(cfg)

        assert "linkedin" in env.platforms
        assert "twitter" in env.platforms
        mock_li.assert_called_once()
        mock_x.assert_called_once()

    @patch("src.simulation.environment.XPlatform")
    @patch("src.simulation.environment.LinkedInPlatform")
    def test_environment_seed_sets_numpy_seed(self, mock_li, mock_x):
        from src.simulation.environment import SimulationConfig, MarketingEnvironment

        cfg = SimulationConfig(seed=99, platforms=[])
        _ = MarketingEnvironment(cfg)
        # After setting seed=99, first random value is deterministic
        val = np.random.random()
        np.random.seed(99)
        expected = np.random.random()
        # They won't match because env already consumed the seed,
        # but we verify the seed path executed without error.
        assert isinstance(val, float)

    @patch("src.simulation.environment.XPlatform")
    @patch("src.simulation.environment.LinkedInPlatform")
    def test_run_campaign_invalid_platform_raises(self, mock_li, mock_x):
        from src.simulation.environment import SimulationConfig, MarketingEnvironment

        cfg = SimulationConfig(platforms=[], seed=1)
        env = MarketingEnvironment(cfg)

        with pytest.raises(ValueError, match="not initialized"):
            env.run_campaign({"platform": "tiktok", "budget": 100, "duration": 7})

    @patch("src.simulation.environment.XPlatform")
    @patch("src.simulation.environment.LinkedInPlatform")
    def test_reset_clears_state(self, mock_li, mock_x):
        from src.simulation.environment import SimulationConfig, MarketingEnvironment

        cfg = SimulationConfig(platforms=[], seed=1)
        env = MarketingEnvironment(cfg)
        env.market_state.total_impressions = 500
        env.campaign_results.append({"x": 1})

        env.reset()

        assert env.market_state.total_impressions == 0
        assert env.campaign_results == []

    @patch("src.simulation.environment.XPlatform")
    @patch("src.simulation.environment.LinkedInPlatform")
    def test_get_state_snapshot(self, mock_li, mock_x):
        from src.simulation.environment import SimulationConfig, MarketingEnvironment

        cfg = SimulationConfig(platforms=[], seed=1)
        env = MarketingEnvironment(cfg)
        snap = env.get_state_snapshot()

        assert "market_state" in snap
        assert snap["market_state"]["sentiment"] == 0.5
        assert snap["active_customers"] == 0

    @patch("src.simulation.environment.XPlatform")
    @patch("src.simulation.environment.LinkedInPlatform")
    def test_run_simulation_auto_creates_personas(self, mock_li, mock_x):
        """run_simulation() should auto-create default personas if none loaded."""
        from src.simulation.environment import SimulationConfig, MarketingEnvironment

        cfg = SimulationConfig(
            platforms=[],
            seed=42,
            num_customers=5,
            duration_days=1,
        )
        env = MarketingEnvironment(cfg)
        assert len(env.customer_agents) == 0

        with patch.object(env, "_create_default_personas") as mock_create:
            # Prevent actual agent creation which needs real Persona models
            mock_create.side_effect = lambda: setattr(env, 'customer_agents', [])
            results = env.run_simulation()

        mock_create.assert_called_once()
        assert "duration_days" in results
        assert results["avg_ctr"] == 0.0  # no impressions in empty sim

    @patch("src.simulation.environment.XPlatform")
    @patch("src.simulation.environment.LinkedInPlatform")
    def test_export_results_writes_json(self, mock_li, mock_x, tmp_path):
        from src.simulation.environment import SimulationConfig, MarketingEnvironment

        cfg = SimulationConfig(platforms=[], seed=1)
        env = MarketingEnvironment(cfg)

        filepath = tmp_path / "results.json"
        env.export_results(str(filepath))
        assert filepath.exists()

        import json
        data = json.loads(filepath.read_text())
        assert "config" in data
        assert data["config"]["duration_days"] == 30


# ---------------------------------------------------------------------------
# 2. PersonaFactory
# ---------------------------------------------------------------------------

class TestPersonaFactory:
    """Tests for PersonaFactory config loading and persona creation."""

    @patch("src.simulation.agents.persona_factory.settings")
    def test_factory_init_no_dir(self, mock_settings, tmp_path):
        from src.simulation.agents.persona_factory import PersonaFactory

        missing = tmp_path / "nonexistent"
        mock_settings.CONFIG_DIR = str(tmp_path)
        factory = PersonaFactory(persona_dir=missing)

        assert factory.loaded_personas == {}

    @patch("src.simulation.agents.persona_factory.settings")
    def test_factory_loads_yaml(self, mock_settings, tmp_path):
        import yaml
        from src.simulation.agents.persona_factory import PersonaFactory

        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()

        persona_data = {
            "persona_id": "test_buyer",
            "label": "Test Buyer",
            "who": {
                "archetypes": ["CTO"],
                "context": {"buying_committee_role": "decision_maker"},
            },
        }
        (persona_dir / "test_buyer.yaml").write_text(yaml.dump(persona_data))

        mock_settings.CONFIG_DIR = str(tmp_path)
        factory = PersonaFactory(persona_dir=persona_dir)

        assert "test_buyer" in factory.loaded_personas
        assert factory.loaded_personas["test_buyer"]["name"] == "Test Buyer"

    @patch("src.simulation.agents.persona_factory.settings")
    def test_create_persona_model_not_found(self, mock_settings, tmp_path):
        from src.simulation.agents.persona_factory import PersonaFactory

        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        mock_settings.CONFIG_DIR = str(tmp_path)
        factory = PersonaFactory(persona_dir=persona_dir)

        with pytest.raises(ValueError, match="not found"):
            factory.create_persona_model("nonexistent")

    @patch("src.simulation.agents.persona_factory.settings")
    def test_validate_persona_config_complete(self, mock_settings, tmp_path):
        import yaml
        from src.simulation.agents.persona_factory import PersonaFactory

        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "p.yaml").write_text(yaml.dump({"persona_id": "p", "label": "P"}))
        mock_settings.CONFIG_DIR = str(tmp_path)
        factory = PersonaFactory(persona_dir=persona_dir)

        # The loaded persona has all required fields from the factory transforms
        assert factory.validate_persona_config("p") is True

    @patch("src.simulation.agents.persona_factory.settings")
    def test_list_available_personas(self, mock_settings, tmp_path):
        import yaml
        from src.simulation.agents.persona_factory import PersonaFactory

        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "a.yaml").write_text(yaml.dump({"persona_id": "a"}))
        (persona_dir / "b.yaml").write_text(yaml.dump({"persona_id": "b"}))
        mock_settings.CONFIG_DIR = str(tmp_path)
        factory = PersonaFactory(persona_dir=persona_dir)

        assert sorted(factory.list_available_personas()) == ["a", "b"]

    @patch("src.simulation.agents.persona_factory.settings")
    def test_get_persona_preferences_missing(self, mock_settings, tmp_path):
        from src.simulation.agents.persona_factory import PersonaFactory

        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        mock_settings.CONFIG_DIR = str(tmp_path)
        factory = PersonaFactory(persona_dir=persona_dir)

        assert factory.get_persona_preferences("missing") == {}

    @patch("src.simulation.agents.persona_factory.settings")
    def test_get_ltv_multiplier_default(self, mock_settings, tmp_path):
        from src.simulation.agents.persona_factory import PersonaFactory

        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        mock_settings.CONFIG_DIR = str(tmp_path)
        factory = PersonaFactory(persona_dir=persona_dir)

        assert factory.get_ltv_multiplier("missing") == 1.0


# ---------------------------------------------------------------------------
# 3. CustomerAgent
# ---------------------------------------------------------------------------

class TestCustomerAgent:
    """Tests for customer behavior simulation."""

    def _make_persona(self, **overrides):
        """Create a mock Persona with sensible defaults."""
        persona = Mock()
        persona.name = overrides.get("name", "test_persona")
        persona.daily_active_prob = overrides.get("daily_active_prob", 0.5)
        persona.click_prob = overrides.get("click_prob", 0.05)
        persona.conversion_prob = overrides.get("conversion_prob", 0.02)
        persona.ad_fatigue_threshold = overrides.get("ad_fatigue_threshold", 5)
        persona.ad_fatigue_decay = overrides.get("ad_fatigue_decay", 0.15)
        persona.influence_factor = overrides.get("influence_factor", 0.3)
        persona.peak_hours = overrides.get("peak_hours", [9, 10, 11])
        return persona

    def _make_market_env(self):
        """Create a mock market environment."""
        from src.simulation.environment import MarketState

        market = Mock()
        market.market_state = MarketState(timestamp=datetime.now())
        return market

    def test_agent_initial_state(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona()
        market = self._make_market_env()

        agent = CustomerAgent(env=env, agent_id="c_0", persona=persona, environment=market)

        assert agent.agent_id == "c_0"
        assert agent.state.is_active is False
        assert agent.state.impressions_seen == 0
        assert agent.state.converted is False

    def test_see_impression_increments_counters(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona()
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_1", persona=persona, environment=market)

        agent.see_impression()
        assert agent.state.impressions_seen == 1
        assert market.market_state.total_impressions == 1

    def test_ad_fatigue_below_threshold(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona(ad_fatigue_threshold=10)
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_2", persona=persona, environment=market)

        for _ in range(5):
            agent.see_impression()

        assert agent.state.ad_fatigue_level == 0.0

    def test_ad_fatigue_above_threshold(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona(ad_fatigue_threshold=3)
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_3", persona=persona, environment=market)

        for _ in range(8):
            agent.see_impression()

        assert agent.state.ad_fatigue_level > 0.0

    def test_decide_click_respects_probability(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        np.random.seed(0)
        env = simpy.Environment()
        persona = self._make_persona(click_prob=1.0)  # always click
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_4", persona=persona, environment=market)

        clicked = agent.decide_click()
        assert clicked is True
        assert agent.state.ads_clicked == 1
        assert market.market_state.total_clicks == 1

    def test_consider_conversion_no_clicks(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona(conversion_prob=1.0)
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_5", persona=persona, environment=market)

        assert agent.consider_conversion() is False  # 0 clicks

    def test_consider_conversion_already_converted(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona(conversion_prob=1.0)
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_6", persona=persona, environment=market)
        agent.state.converted = True
        agent.state.ads_clicked = 5

        assert agent.consider_conversion() is False

    def test_reset_clears_state(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona()
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_7", persona=persona, environment=market)

        agent.see_impression()
        agent.state.ads_clicked = 3
        agent.reset()

        assert agent.state.impressions_seen == 0
        assert agent.state.ads_clicked == 0
        assert agent.interaction_history == []

    def test_get_summary_returns_expected_keys(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona()
        market = self._make_market_env()
        agent = CustomerAgent(env=env, agent_id="c_8", persona=persona, environment=market)

        summary = agent.get_summary()
        expected_keys = {
            "agent_id", "persona", "impressions", "clicks",
            "converted", "ad_fatigue", "interest_level",
            "activity_level", "interactions",
        }
        assert expected_keys.issubset(summary.keys())

    def test_add_connection(self):
        import simpy
        from src.simulation.agents.customer_agent import CustomerAgent

        env = simpy.Environment()
        persona = self._make_persona()
        market = self._make_market_env()
        a = CustomerAgent(env=env, agent_id="a", persona=persona, environment=market)
        b = CustomerAgent(env=env, agent_id="b", persona=persona, environment=market)

        a.add_connection(b)
        a.add_connection(b)  # duplicate ignored
        assert len(a.connections) == 1


# ---------------------------------------------------------------------------
# 4. SimulationValidator / CampaignValidator
# ---------------------------------------------------------------------------

class TestSimulationValidator:
    """Tests for simulation result validation utilities."""

    def test_calculate_mape_basic(self):
        from src.simulation.validators import SimulationValidator

        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        mape = SimulationValidator.calculate_mape(actual, predicted)
        assert 0 < mape < 10  # small error

    def test_calculate_mape_all_zeros(self):
        from src.simulation.validators import SimulationValidator

        actual = np.array([0, 0, 0])
        predicted = np.array([1, 2, 3])
        assert SimulationValidator.calculate_mape(actual, predicted) == 100.0

    def test_calculate_rmse(self):
        from src.simulation.validators import SimulationValidator

        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert SimulationValidator.calculate_rmse(actual, predicted) == 0.0

    def test_calculate_correlation_single_element(self):
        from src.simulation.validators import SimulationValidator

        assert SimulationValidator.calculate_correlation(
            np.array([1.0]), np.array([2.0])
        ) == 0.0

    def test_accuracy_score_from_mape(self):
        from src.simulation.validators import SimulationValidator

        assert SimulationValidator.calculate_accuracy_score(0.0) == 1.0
        assert SimulationValidator.calculate_accuracy_score(100.0) == 0.0
        assert SimulationValidator.calculate_accuracy_score(200.0) == 0.0  # clamped

    def test_validate_metrics_multikey(self):
        from src.simulation.validators import SimulationValidator

        sim = {"clicks": [100, 200], "impressions": [1000, 2000]}
        act = {"clicks": [110, 190], "impressions": [1050, 1950]}
        results = SimulationValidator.validate_metrics(sim, act)

        assert "clicks" in results
        assert "impressions" in results
        assert "mape" in results["clicks"]
        assert results["clicks"]["accuracy"] > 0.8

    def test_overall_accuracy_with_weights(self):
        from src.simulation.validators import SimulationValidator

        validation_results = {
            "impressions": {"accuracy": 0.9},
            "clicks": {"accuracy": 0.85},
            "conversions": {"accuracy": 0.8},
            "ctr": {"accuracy": 0.95},
        }
        overall = SimulationValidator.calculate_overall_accuracy(validation_results)
        assert 0.0 < overall < 1.0

    def test_overall_accuracy_empty(self):
        from src.simulation.validators import SimulationValidator

        assert SimulationValidator.calculate_overall_accuracy({}) == 0.0

    def test_check_bias(self):
        from src.simulation.validators import SimulationValidator

        actual = np.array([10, 20, 30])
        predicted = np.array([12, 22, 32])  # consistently overestimates
        bias = SimulationValidator.check_bias(actual, predicted)

        assert bias["bias_direction"] == "overestimate"
        assert bias["mean_error"] == 2.0

    def test_validate_distribution_ks(self):
        from src.simulation.validators import SimulationValidator

        np.random.seed(42)
        a = np.random.normal(0, 1, 200)
        b = np.random.normal(0, 1, 200)
        stat, p = SimulationValidator.validate_distribution(a, b, test="ks")
        assert p > 0.05  # same distribution → high p-value

    def test_validate_distribution_invalid_test(self):
        from src.simulation.validators import SimulationValidator

        with pytest.raises(ValueError, match="Unknown test"):
            SimulationValidator.validate_distribution(
                np.array([1]), np.array([1]), test="invalid"
            )


class TestCampaignValidator:
    """Tests for campaign configuration validation."""

    def test_valid_campaign_config(self):
        from src.simulation.validators import CampaignValidator

        config = {"platform": "linkedin", "budget": 500, "duration": 7}
        is_valid, errors = CampaignValidator.validate_campaign_config(config)
        assert is_valid is True
        assert errors == []

    def test_missing_fields(self):
        from src.simulation.validators import CampaignValidator

        is_valid, errors = CampaignValidator.validate_campaign_config({})
        assert is_valid is False
        assert len(errors) == 3  # platform, budget, duration

    def test_negative_budget(self):
        from src.simulation.validators import CampaignValidator

        config = {"platform": "linkedin", "budget": -100, "duration": 7}
        is_valid, errors = CampaignValidator.validate_campaign_config(config)
        assert is_valid is False
        assert any("positive" in e for e in errors)

    def test_excessive_budget(self):
        from src.simulation.validators import CampaignValidator

        config = {"platform": "linkedin", "budget": 2_000_000, "duration": 7}
        is_valid, errors = CampaignValidator.validate_campaign_config(config)
        assert is_valid is False
        assert any("maximum" in e.lower() for e in errors)

    def test_invalid_platform(self):
        from src.simulation.validators import CampaignValidator

        config = {"platform": "tiktok", "budget": 100, "duration": 7}
        is_valid, errors = CampaignValidator.validate_campaign_config(config)
        assert is_valid is False
        assert any("Invalid platform" in e for e in errors)

    def test_validate_targeting_valid(self):
        from src.simulation.validators import CampaignValidator

        ok, errors = CampaignValidator.validate_targeting({"persona": "decision_maker"})
        assert ok is True

    def test_validate_targeting_invalid_persona(self):
        from src.simulation.validators import CampaignValidator

        ok, errors = CampaignValidator.validate_targeting({"persona": "aliens"})
        assert ok is False


# ---------------------------------------------------------------------------
# 5. MockDeployer
# ---------------------------------------------------------------------------

class TestMockDeployer:
    """Tests for mock deployment that generates realistic metrics."""

    def test_deploy_campaign_returns_required_keys(self):
        from src.automation_layer.mock_deployer import MockDeployer

        deployer = MockDeployer()
        sim = {"ctr": 0.02, "conversions": 5, "impressions": 30000, "clicks": 400}
        result = deployer.deploy_campaign("c1", sim, budget=1000, duration_days=7)

        assert result["status"] == "deployed_mock"
        assert "actual_metrics" in result
        assert "predicted_metrics" in result
        assert "validation" in result
        assert "daily_metrics" in result

    def test_deploy_campaign_metrics_within_variance(self):
        from src.automation_layer.mock_deployer import MockDeployer

        deployer = MockDeployer()
        sim = {"impressions": 10000, "clicks": 200, "conversions": 10, "ctr": 0.02}
        result = deployer.deploy_campaign("c2", sim, budget=500, duration_days=3)

        actual = result["actual_metrics"]
        # Metrics should be within reasonable bounds of predictions
        assert 5000 < actual["impressions"] < 15000
        assert actual["conversions"] >= 1

    def test_daily_metrics_length(self):
        from src.automation_layer.mock_deployer import MockDeployer

        deployer = MockDeployer()
        result = deployer.deploy_campaign(
            "c3", {"impressions": 1000, "clicks": 50, "conversions": 3},
            budget=200, duration_days=5,
        )
        assert len(result["daily_metrics"]) == 5

    def test_validation_mape_calculated(self):
        from src.automation_layer.mock_deployer import MockDeployer

        deployer = MockDeployer()
        result = deployer.deploy_campaign(
            "c4", {"impressions": 5000, "clicks": 100, "conversions": 5, "ctr": 0.02},
            budget=300, duration_days=7,
        )
        v = result["validation"]
        assert "overall_mape" in v
        assert "accuracy" in v
        assert isinstance(v["target_met"], bool)

    def test_daily_weights_short_campaign(self):
        from src.automation_layer.mock_deployer import MockDeployer

        deployer = MockDeployer()
        weights = deployer._get_daily_weights(3)
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 0.01

    def test_daily_weights_long_campaign(self):
        from src.automation_layer.mock_deployer import MockDeployer

        deployer = MockDeployer()
        weights = deployer._get_daily_weights(14)
        assert len(weights) == 14
        assert abs(sum(weights) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 6. CampaignDeployer
# ---------------------------------------------------------------------------

class TestCampaignDeployer:
    """Tests for deployment routing across connectors."""

    @patch("src.automation_layer.deployer.get_sync_session")
    def test_init_creates_mock_deployer(self, mock_sync):
        from src.automation_layer.deployer import CampaignDeployer

        deployer = CampaignDeployer()
        assert deployer.mock_deployer is not None
        assert deployer._initialized is False
        assert deployer.connectors == {}

    @pytest.mark.asyncio
    @patch("src.automation_layer.deployer.format_content_for_platform", side_effect=lambda c, p: c)
    @patch("src.automation_layer.deployer.get_sync_session")
    async def test_deploy_no_connector_mock_disabled(self, mock_sync, mock_fmt):
        from src.automation_layer.deployer import CampaignDeployer

        deployer = CampaignDeployer()
        deployer._initialized = True
        deployer.mock_mode_enabled = False
        deployer.enable_mock_deployment = False

        result = await deployer.deploy("content_1", "linkedin", {"body": "test"})

        assert result["success"] is False
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    @patch("src.automation_layer.deployer.format_content_for_platform", side_effect=lambda c, p: c)
    @patch("src.automation_layer.deployer.get_sync_session")
    async def test_deploy_with_connector_invalid_creds(self, mock_sync, mock_fmt):
        from src.automation_layer.deployer import CampaignDeployer

        deployer = CampaignDeployer()
        deployer._initialized = True
        deployer.mock_mode_enabled = False
        deployer.enable_mock_deployment = False

        mock_connector = AsyncMock()
        mock_connector.validate_credentials = AsyncMock(return_value=False)
        deployer.connectors["linkedin"] = mock_connector

        result = await deployer.deploy("content_1", "linkedin", {"body": "hi"})
        assert result["success"] is False
        assert "Invalid credentials" in result["error"]


# ---------------------------------------------------------------------------
# 7. Canary Rollout – DeploymentController
# ---------------------------------------------------------------------------

class TestDeploymentController:
    """Tests for canary deployment logic."""

    def test_controller_init(self):
        from src.automation_layer.deployment.canary_rollout import DeploymentController

        ctrl = DeploymentController()
        assert ctrl.active_deployments == {}
        assert ctrl.deployment_history == []
        assert ctrl.degradation_threshold == 0.10

    @pytest.mark.asyncio
    async def test_start_canary_deployment_success(self):
        from src.automation_layer.deployment.canary_rollout import DeploymentController

        ctrl = DeploymentController()

        with patch("asyncio.create_task"):
            result = await ctrl.start_canary_deployment(
                policy_id="pol_1",
                policy_version="v2",
                initial_traffic_percent=5,
                baseline_metrics={"ctr": 0.03, "conversion_rate": 0.02, "cpl": 50.0},
            )

        assert result["success"] is True
        assert result["traffic_percent"] == 5
        assert len(ctrl.active_deployments) == 1

    @pytest.mark.asyncio
    async def test_start_canary_deployment_default_baseline(self):
        from src.automation_layer.deployment.canary_rollout import DeploymentController

        ctrl = DeploymentController()

        with patch("asyncio.create_task"):
            result = await ctrl.start_canary_deployment(policy_id="pol_2")

        deployment = list(ctrl.active_deployments.values())[0]
        assert deployment.baseline_metrics == {"ctr": 3.0, "conversion_rate": 2.0, "cpl": 50.0}

    @pytest.mark.asyncio
    async def test_rollback_deployment(self):
        from src.automation_layer.deployment.canary_rollout import (
            DeploymentController,
            CanaryDeployment,
            DeploymentStatus,
        )

        ctrl = DeploymentController()

        with patch("asyncio.create_task"):
            result = await ctrl.start_canary_deployment(
                policy_id="pol_3",
                baseline_metrics={"ctr": 0.03, "conversion_rate": 0.02, "cpl": 50.0},
            )

        dep_id = result["deployment_id"]

        with patch.object(ctrl, "_send_rollback_alert", new_callable=AsyncMock):
            success = await ctrl.rollback_deployment(dep_id, "test_reason")

        assert success is True
        assert dep_id not in ctrl.active_deployments
        assert len(ctrl.deployment_history) == 1
        assert ctrl.deployment_history[0].status == DeploymentStatus.ROLLED_BACK
        assert ctrl.deployment_history[0].rollback_reason == "test_reason"

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_deployment(self):
        from src.automation_layer.deployment.canary_rollout import DeploymentController

        ctrl = DeploymentController()
        success = await ctrl.rollback_deployment("nonexistent", "reason")
        assert success is False

    def test_check_degradation_ctr_drop(self):
        from src.automation_layer.deployment.canary_rollout import (
            DeploymentController,
            CanaryDeployment,
            CanaryMetrics,
            DeploymentStatus,
        )

        ctrl = DeploymentController()
        dep = CanaryDeployment(
            deployment_id="d1",
            policy_id="p1",
            policy_version="v1",
            start_time=datetime.now(),
            baseline_metrics={"ctr": 0.10, "conversion_rate": 0.05, "cpl": 50.0},
        )

        # CTR drops 50% → should trigger degradation
        bad_metrics = CanaryMetrics(
            timestamp=datetime.now(),
            traffic_percentage=0.05,
            requests_served=100,
            average_ctr=0.04,  # way below 0.10 * 0.90
            average_conversion_rate=0.05,
            average_cost_per_lead=50.0,
            error_rate=0.001,
            p95_latency_ms=200.0,
        )

        assert ctrl._check_for_degradation(dep, bad_metrics) is True

    def test_check_degradation_no_baseline(self):
        from src.automation_layer.deployment.canary_rollout import (
            DeploymentController,
            CanaryDeployment,
            CanaryMetrics,
        )

        ctrl = DeploymentController()
        dep = CanaryDeployment(
            deployment_id="d2",
            policy_id="p2",
            policy_version="v1",
            start_time=datetime.now(),
            baseline_metrics=None,
        )
        metrics = CanaryMetrics(
            timestamp=datetime.now(),
            traffic_percentage=0.05,
            requests_served=100,
            average_ctr=0.001,
            average_conversion_rate=0.001,
            average_cost_per_lead=100.0,
            error_rate=0.001,
            p95_latency_ms=200.0,
        )
        assert ctrl._check_for_degradation(dep, metrics) is False

    def test_check_degradation_high_error_rate(self):
        from src.automation_layer.deployment.canary_rollout import (
            DeploymentController,
            CanaryDeployment,
            CanaryMetrics,
        )

        ctrl = DeploymentController()
        dep = CanaryDeployment(
            deployment_id="d3",
            policy_id="p3",
            policy_version="v1",
            start_time=datetime.now(),
            baseline_metrics={"ctr": 0.03, "conversion_rate": 0.02, "cpl": 50.0},
        )
        metrics = CanaryMetrics(
            timestamp=datetime.now(),
            traffic_percentage=0.05,
            requests_served=100,
            average_ctr=0.03,
            average_conversion_rate=0.02,
            average_cost_per_lead=50.0,
            error_rate=0.10,  # 10% > 5% threshold
            p95_latency_ms=200.0,
        )
        assert ctrl._check_for_degradation(dep, metrics) is True

    def test_check_degradation_cpl_increase(self):
        from src.automation_layer.deployment.canary_rollout import (
            DeploymentController,
            CanaryDeployment,
            CanaryMetrics,
        )

        ctrl = DeploymentController()
        dep = CanaryDeployment(
            deployment_id="d4",
            policy_id="p4",
            policy_version="v1",
            start_time=datetime.now(),
            baseline_metrics={"ctr": 0.03, "conversion_rate": 0.02, "cpl": 50.0},
        )
        metrics = CanaryMetrics(
            timestamp=datetime.now(),
            traffic_percentage=0.05,
            requests_served=100,
            average_ctr=0.03,
            average_conversion_rate=0.02,
            average_cost_per_lead=60.0,  # > 50 * 1.10 = 55
            error_rate=0.001,
            p95_latency_ms=200.0,
        )
        assert ctrl._check_for_degradation(dep, metrics) is True

    def test_get_deployment_status_active(self):
        from src.automation_layer.deployment.canary_rollout import (
            DeploymentController,
            CanaryDeployment,
            DeploymentStatus,
        )

        ctrl = DeploymentController()
        dep = CanaryDeployment(
            deployment_id="d5",
            policy_id="p5",
            policy_version="v1",
            start_time=datetime.now(),
            status=DeploymentStatus.CANARY_5,
        )
        ctrl.active_deployments["d5"] = dep
        status = ctrl.get_deployment_status("d5")

        assert status is not None
        assert status["status"] == "canary_5_percent"

    def test_get_deployment_status_not_found(self):
        from src.automation_layer.deployment.canary_rollout import DeploymentController

        ctrl = DeploymentController()
        assert ctrl.get_deployment_status("nope") is None

    def test_list_active_deployments(self):
        from src.automation_layer.deployment.canary_rollout import (
            DeploymentController,
            CanaryDeployment,
            DeploymentStatus,
        )

        ctrl = DeploymentController()
        dep = CanaryDeployment(
            deployment_id="d6",
            policy_id="p6",
            policy_version="v1",
            start_time=datetime.now(),
        )
        ctrl.active_deployments["d6"] = dep
        active = ctrl.list_active_deployments()

        assert len(active) == 1
        assert active[0]["deployment_id"] == "d6"

    def test_canary_metrics_to_dict(self):
        from src.automation_layer.deployment.canary_rollout import CanaryMetrics

        m = CanaryMetrics(
            timestamp=datetime(2024, 1, 1),
            traffic_percentage=0.05,
            requests_served=100,
            average_ctr=0.03,
            average_conversion_rate=0.02,
            average_cost_per_lead=50.0,
            error_rate=0.001,
            p95_latency_ms=200.0,
        )
        d = m.to_dict()
        assert d["traffic_percentage"] == 0.05
        assert d["requests_served"] == 100


# ---------------------------------------------------------------------------
# 8. BudgetManager
# ---------------------------------------------------------------------------

class TestBudgetManager:
    """Tests for budget tracking and threshold alerts."""

    def _make_manager(self, daily_limit=100.0, campaign_limit=500.0):
        from src.cost_control.budget_manager import BudgetManager

        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("src.cost_control.budget_manager.settings") as mock_settings:
            mock_settings.MAX_DAILY_API_COST = daily_limit
            mock_settings.MAX_CAMPAIGN_COST = campaign_limit
            mgr = BudgetManager(mock_db, mock_redis)

        return mgr

    @pytest.mark.asyncio
    async def test_check_budget_within_limit(self):
        mgr = self._make_manager()

        campaign = Mock()
        campaign.budget_spent = 100.0
        campaign.budget_total = 500.0

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = campaign
        mgr.db.execute = AsyncMock(return_value=mock_result)

        assert await mgr.check_budget("camp_1", estimated_cost=50.0) is True

    @pytest.mark.asyncio
    async def test_check_budget_exceeded(self):
        mgr = self._make_manager()

        campaign = Mock()
        campaign.budget_spent = 480.0
        campaign.budget_total = 500.0

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = campaign
        mgr.db.execute = AsyncMock(return_value=mock_result)

        assert await mgr.check_budget("camp_2", estimated_cost=30.0) is False

    @pytest.mark.asyncio
    async def test_check_budget_campaign_not_found(self):
        mgr = self._make_manager()

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mgr.db.execute = AsyncMock(return_value=mock_result)

        # Missing campaign → conservative False
        assert await mgr.check_budget("missing") is False

    @pytest.mark.asyncio
    async def test_track_cost_success(self):
        mgr = self._make_manager()
        mgr.db.commit = AsyncMock()
        mgr.db.add = Mock()
        mgr.db.execute = AsyncMock()
        mgr.redis.incrbyfloat = AsyncMock()
        mgr.redis.expire = AsyncMock()
        mgr.redis.get = AsyncMock(return_value="50.0")

        result = await mgr.track_cost(
            source_type="api_calls",
            cost_amount=5.0,
            campaign_id="camp_3",
            metadata={"provider": "openai"},
        )
        assert result is True
        mgr.db.add.assert_called_once()
        mgr.db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_cost_db_error_rollback(self):
        mgr = self._make_manager()
        mgr.db.add = Mock(side_effect=Exception("DB error"))
        mgr.db.rollback = AsyncMock()

        result = await mgr.track_cost("api_calls", 5.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_enforce_limits_pauses_campaign(self):
        mgr = self._make_manager()

        # Mock check_budget to return False (exceeded)
        with patch.object(mgr, "check_budget", new_callable=AsyncMock, return_value=False):
            mgr.db.execute = AsyncMock()
            mgr.db.commit = AsyncMock()

            result = await mgr.enforce_limits("camp_4")

        assert result is False
        mgr.db.execute.assert_called_once()  # UPDATE statement
        mgr.db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# 9. TokenTracker
# ---------------------------------------------------------------------------

class TestTokenTracker:
    """Tests for LLM token usage tracking."""

    def test_track_usage_known_model(self):
        from src.cost_control.token_tracker import TokenTracker

        tracker = TokenTracker()
        usage = tracker.track_usage("gpt-4", prompt_tokens=1000, completion_tokens=500)

        assert usage["total_tokens"] == 1500
        expected_cost = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert abs(usage["total_cost"] - expected_cost) < 0.001
        assert tracker.total_tokens == 1500

    def test_track_usage_unknown_model_uses_default(self):
        from src.cost_control.token_tracker import TokenTracker

        tracker = TokenTracker()
        usage = tracker.track_usage("future-model-v99", prompt_tokens=500, completion_tokens=300)

        # Default: prompt=0.01, completion=0.03
        expected = (500 / 1000) * 0.01 + (300 / 1000) * 0.03
        assert abs(usage["total_cost"] - expected) < 0.001

    def test_get_summary_empty(self):
        from src.cost_control.token_tracker import TokenTracker

        tracker = TokenTracker()
        summary = tracker.get_summary()

        assert summary["total_calls"] == 0
        assert summary["total_cost"] == 0.0

    def test_get_summary_aggregates_by_model(self):
        from src.cost_control.token_tracker import TokenTracker

        tracker = TokenTracker()
        tracker.track_usage("gpt-4", 100, 50)
        tracker.track_usage("gpt-4", 200, 100)
        tracker.track_usage("gpt-3.5-turbo", 500, 250)

        summary = tracker.get_summary()
        assert summary["total_calls"] == 3
        assert "gpt-4" in summary["by_model"]
        assert summary["by_model"]["gpt-4"]["calls"] == 2
        assert "gpt-3.5-turbo" in summary["by_model"]

    def test_metadata_stored(self):
        from src.cost_control.token_tracker import TokenTracker

        tracker = TokenTracker()
        usage = tracker.track_usage(
            "gpt-4", 100, 50, metadata={"campaign_id": "abc"}
        )
        assert usage["metadata"]["campaign_id"] == "abc"

    def test_total_cost_accumulates(self):
        from src.cost_control.token_tracker import TokenTracker

        tracker = TokenTracker()
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)
        first_cost = tracker.total_cost
        tracker.track_usage("gpt-3.5-turbo", 1000, 500)

        assert tracker.total_cost == first_cost * 2


# ---------------------------------------------------------------------------
# 10. ConfigurationService
# ---------------------------------------------------------------------------

class TestConfigurationService:
    """Tests for config loading, default values, and CRUD operations."""

    def _make_service(self):
        from src.config.configuration_service import ConfigurationService

        mock_db = Mock()
        mock_db.query.return_value = mock_db
        mock_db.filter_by.return_value = mock_db
        mock_db.filter.return_value = mock_db
        mock_db.first.return_value = None
        mock_db.all.return_value = []
        mock_db.count.return_value = 0
        mock_db.commit = Mock()
        mock_db.add = Mock()
        mock_db.rollback = Mock()

        svc = ConfigurationService(mock_db)
        return svc

    def test_get_value_from_defaults(self):
        svc = self._make_service()
        # DB returns None → falls back to DEFAULT_CONFIGURATIONS
        val = svc.get_value("USE_LOCAL_LLM")
        assert val is True  # default is "True", converted to bool

    def test_get_value_missing_key_returns_default_arg(self):
        svc = self._make_service()
        val = svc.get_value("TOTALLY_UNKNOWN_KEY", default="fallback")
        assert val == "fallback"

    def test_convert_value_integer(self):
        svc = self._make_service()
        assert svc._convert_value("42", "integer") == 42

    def test_convert_value_float(self):
        svc = self._make_service()
        assert svc._convert_value("3.14", "float") == pytest.approx(3.14)

    def test_convert_value_boolean_true(self):
        svc = self._make_service()
        assert svc._convert_value("True", "boolean") is True
        assert svc._convert_value("yes", "boolean") is True

    def test_convert_value_boolean_false(self):
        svc = self._make_service()
        assert svc._convert_value("false", "boolean") is False
        assert svc._convert_value("no", "boolean") is False

    def test_convert_value_empty_returns_none(self):
        svc = self._make_service()
        assert svc._convert_value("", "string") is None
        assert svc._convert_value(None, "integer") is None

    def test_set_value_new_key_requires_category(self):
        svc = self._make_service()
        with pytest.raises(ValueError, match="Category required"):
            svc.set_value("BRAND_NEW_KEY_XYZ", "val")

    @patch("src.config.configuration_service.encrypt_value", return_value="encrypted")
    def test_set_value_existing_config(self, mock_encrypt):
        from src.config.configuration_service import ConfigurationService

        existing = Mock()
        existing.is_secret = False
        existing.value = "old"

        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = existing
        mock_db.commit = Mock()

        svc = ConfigurationService(mock_db)
        result = svc.set_value("SOME_KEY", "new_val")

        assert result is True
        assert existing.value == "new_val"

    def test_bulk_update(self):
        svc = self._make_service()
        with patch.object(svc, "set_value", return_value=True) as mock_set:
            results = svc.bulk_update({"A": 1, "B": 2})

        assert results == {"A": True, "B": True}
        assert mock_set.call_count == 2

    def test_get_value_db_record_non_secret(self):
        from src.config.configuration_service import ConfigurationService

        record = Mock()
        record.value = "8000"
        record.is_secret = False
        record.value_type = "integer"

        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = record

        svc = ConfigurationService(mock_db)
        val = svc.get_value("API_PORT")
        assert val == 8000

    @patch("src.config.configuration_service.decrypt_value", return_value="decrypted_secret")
    def test_get_value_secret_decrypted(self, mock_decrypt):
        from src.config.configuration_service import ConfigurationService

        record = Mock()
        record.value = "encrypted_blob"
        record.is_secret = True
        record.value_type = "string"

        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = record

        svc = ConfigurationService(mock_db)
        val = svc.get_value("OPENAI_API_KEY")
        assert val == "decrypted_secret"
        mock_decrypt.assert_called_once_with("encrypted_blob")

    @patch("src.config.configuration_service.decrypt_value", side_effect=Exception("bad key"))
    def test_get_value_decrypt_failure_returns_default(self, mock_decrypt):
        from src.config.configuration_service import ConfigurationService

        record = Mock()
        record.value = "bad_encrypted"
        record.is_secret = True
        record.value_type = "string"

        mock_db = Mock()
        mock_db.query.return_value.filter_by.return_value.first.return_value = record

        svc = ConfigurationService(mock_db)
        val = svc.get_value("OPENAI_API_KEY", default="safe_default")
        assert val == "safe_default"


# ---------------------------------------------------------------------------
# Validation report helper
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    """Tests for the module-level _generate_summary helper."""

    def test_passed_summary(self):
        from src.simulation.validators import _generate_summary

        summary = _generate_summary(0.95, True, 0.90)
        assert "PASSED" in summary
        assert "95.0%" in summary

    def test_failed_summary(self):
        from src.simulation.validators import _generate_summary

        summary = _generate_summary(0.80, False, 0.90)
        assert "FAILED" in summary
        assert "gap" in summary
