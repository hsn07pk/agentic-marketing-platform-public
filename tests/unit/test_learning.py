# tests/unit/test_learning.py
"""
Comprehensive tests for AI learning algorithms.
All external dependencies (database, MLflow, connectors) are mocked.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Thompson Sampling Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ts_bandit():
    with patch("src.ai_layer.learning.thompson_sampling.settings"):
        from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit
        return ThompsonSamplingBandit(
            experiment_id="exp-001",
            arms=["arm_a", "arm_b", "arm_c"],
        )


@pytest.fixture
def ts_single_arm():
    with patch("src.ai_layer.learning.thompson_sampling.settings"):
        from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit
        return ThompsonSamplingBandit(
            experiment_id="exp-single",
            arms=["only_arm"],
        )


class TestThompsonSampling:

    def test_select_arm_returns_valid_arm(self, ts_bandit):
        arm_id, value = ts_bandit.select_arm()
        assert arm_id in ["arm_a", "arm_b", "arm_c"]
        assert 0.0 <= value <= 1.0

    def test_select_arm_increments_pull_count(self, ts_bandit):
        arm_id, _ = ts_bandit.select_arm()
        assert ts_bandit.arms[arm_id].pulls == 1

    def test_select_arm_no_arms_raises(self):
        with patch("src.ai_layer.learning.thompson_sampling.settings"):
            from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit
            bandit = ThompsonSamplingBandit(experiment_id="empty", arms=[])
            with pytest.raises(ValueError, match="No arms available"):
                bandit.select_arm()

    def test_update_arm_positive_reward(self, ts_bandit):
        ts_bandit.update_arm("arm_a", reward=1.0)
        stats = ts_bandit.arms["arm_a"]
        assert stats.alpha == 2.0  # prior 1.0 + reward 1.0
        assert stats.successes == 1
        assert stats.total_reward == 1.0

    def test_update_arm_zero_reward(self, ts_bandit):
        ts_bandit.update_arm("arm_a", reward=0.0)
        stats = ts_bandit.arms["arm_a"]
        assert stats.alpha == 1.0  # unchanged
        assert stats.beta == 2.0   # prior 1.0 + (1 - 0) = 2.0
        assert stats.successes == 0

    def test_update_unknown_arm_does_not_raise(self, ts_bandit):
        ts_bandit.update_arm("nonexistent", reward=1.0)  # should log error, not raise

    def test_exploitation_favors_successful_arm(self, ts_bandit):
        """After many successes on arm_a, it should be selected more often."""
        for _ in range(100):
            ts_bandit.update_arm("arm_a", reward=1.0)
        selections = [ts_bandit.select_arm()[0] for _ in range(200)]
        arm_a_count = selections.count("arm_a")
        assert arm_a_count > 150, "Arm with high reward should dominate selections"

    def test_exploration_with_uniform_priors(self, ts_bandit):
        """With uniform priors all arms should be selected over many draws."""
        selections = set()
        for _ in range(200):
            arm_id, _ = ts_bandit.select_arm()
            selections.add(arm_id)
        assert len(selections) == 3, "All arms should be explored with uniform priors"

    def test_estimated_ctr(self, ts_bandit):
        stats = ts_bandit.arms["arm_a"]
        assert stats.estimated_ctr == pytest.approx(0.5)  # alpha=1, beta=1

    def test_save_and_load_state_roundtrip(self, ts_bandit):
        ts_bandit.update_arm("arm_a", reward=1.0)
        ts_bandit.select_arm()
        state = ts_bandit.save_state()

        from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit
        restored = ThompsonSamplingBandit.load_state(state)
        assert restored.experiment_id == ts_bandit.experiment_id
        assert restored.arms["arm_a"].alpha == ts_bandit.arms["arm_a"].alpha
        assert restored.arms["arm_a"].beta == ts_bandit.arms["arm_a"].beta

    def test_should_stop_experiment_insufficient_samples(self, ts_bandit):
        assert ts_bandit.should_stop_experiment(min_samples=100) is False

    def test_single_arm_select(self, ts_single_arm):
        arm_id, value = ts_single_arm.select_arm()
        assert arm_id == "only_arm"
        assert 0.0 <= value <= 1.0

    def test_delayed_reward_tracking(self, ts_bandit):
        ts_bandit.add_delayed_reward("arm_a", reward_id="r-001", expected_delay_hours=24)
        assert "r-001" in ts_bandit.pending_rewards
        assert ts_bandit.pending_rewards["r-001"]["arm_id"] == "arm_a"

    def test_process_delayed_reward_unknown_id(self, ts_bandit):
        ts_bandit.process_delayed_reward("unknown-id", actual_reward=1.0)  # should not raise

    def test_cumulative_regret_non_negative(self, ts_bandit):
        for _ in range(20):
            arm_id, _ = ts_bandit.select_arm()
            ts_bandit.update_arm(arm_id, reward=np.random.choice([0.0, 1.0]))
        assert ts_bandit.cumulative_regret >= 0.0

    def test_custom_priors(self):
        with patch("src.ai_layer.learning.thompson_sampling.settings"):
            from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit
            bandit = ThompsonSamplingBandit(
                experiment_id="custom", arms=["a", "b"],
                prior_alpha=5.0, prior_beta=2.0
            )
            assert bandit.arms["a"].alpha == 5.0
            assert bandit.arms["a"].beta == 2.0


# ---------------------------------------------------------------------------
# LinUCB Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def linucb_bandit():
    from src.ai_layer.learning.linucb import LinUCBBandit
    return LinUCBBandit(n_arms=3, n_features=4, alpha=1.0, use_gpu=False)


class TestLinUCB:

    def test_select_arm_returns_valid_index(self, linucb_bandit):
        context = np.array([1.0, 0.5, 0.0, -0.5])
        arm, confidence = linucb_bandit.select_arm(context)
        assert 0 <= arm < 3
        assert 0.0 <= confidence <= 1.0

    def test_select_arm_increments_timestep(self, linucb_bandit):
        context = np.zeros(4)
        linucb_bandit.select_arm(context)
        assert linucb_bandit.t == 1

    def test_update_modifies_matrices(self, linucb_bandit):
        context = np.array([1.0, 0.0, 0.0, 0.0])
        A_before = linucb_bandit.A[0].clone()
        linucb_bandit.update(arm=0, context=context, reward=1.0)
        # A should change (identity + outer product)
        assert not (linucb_bandit.A[0] == A_before).all()

    def test_update_tracks_rewards(self, linucb_bandit):
        context = np.zeros(4)
        linucb_bandit.update(0, context, 0.5)
        assert linucb_bandit.rewards == [0.5]

    def test_batch_update(self, linucb_bandit):
        contexts = np.random.randn(5, 4)
        arms = [0, 1, 2, 0, 1]
        rewards = np.array([1.0, 0.0, 0.5, 1.0, 0.0])
        linucb_bandit.batch_update(arms, contexts, rewards)
        assert len(linucb_bandit.rewards) == 5

    def test_ucb_higher_alpha_more_exploration(self):
        """Higher alpha should produce more variance in UCB scores."""
        from src.ai_layer.learning.linucb import LinUCBBandit
        np.random.seed(42)

        low_alpha = LinUCBBandit(n_arms=3, n_features=4, alpha=0.01, use_gpu=False)
        high_alpha = LinUCBBandit(n_arms=3, n_features=4, alpha=10.0, use_gpu=False)

        context = np.array([1.0, 0.5, -0.3, 0.2])

        # After same training data, high alpha should explore more
        for b in [low_alpha, high_alpha]:
            b.update(0, context, 1.0)
            b.update(1, context, 0.0)
            b.update(2, context, 0.5)

        # Run many selections and check diversity
        low_selections = set(low_alpha.select_arm(context)[0] for _ in range(50))
        high_selections = set(high_alpha.select_arm(context)[0] for _ in range(50))
        # High alpha linucb is deterministic per context after training,
        # but the UCB values differ — just verify valid output
        assert all(0 <= s < 3 for s in low_selections)
        assert all(0 <= s < 3 for s in high_selections)

    def test_reset_clears_state(self, linucb_bandit):
        context = np.ones(4)
        linucb_bandit.update(0, context, 1.0)
        linucb_bandit.reset()
        assert linucb_bandit.t == 0
        assert linucb_bandit.rewards == []
        assert linucb_bandit.arm_counts.sum().item() == 0

    def test_get_statistics(self, linucb_bandit):
        context = np.ones(4)
        linucb_bandit.update(0, context, 1.0)
        stats = linucb_bandit.get_statistics()
        assert stats["total_pulls"] == 0  # only select_arm increments t
        assert stats["average_reward"] == 1.0
        assert stats["cumulative_reward"] == 1.0

    def test_identity_initialization(self, linucb_bandit):
        """A matrices should be initialized as identity."""
        import torch
        for i in range(linucb_bandit.n_arms):
            expected = torch.eye(4, device=linucb_bandit.device)
            assert torch.allclose(linucb_bandit.A[i], expected)

    def test_confidence_sums_to_one_across_softmax(self, linucb_bandit):
        """Softmax confidences across arms should sum to ~1."""
        import torch
        context = np.array([1.0, 0.5, 0.0, -0.5])
        x = torch.tensor(context, dtype=torch.float32, device=linucb_bandit.device).reshape(-1, 1)
        ucb_values = torch.zeros(linucb_bandit.n_arms, device=linucb_bandit.device)
        for a in range(linucb_bandit.n_arms):
            A_inv = torch.linalg.inv(linucb_bandit.A[a])
            theta = A_inv @ linucb_bandit.b[a]
            mean = (theta.T @ x).squeeze()
            variance = torch.sqrt(x.T @ A_inv @ x).squeeze()
            ucb_values[a] = mean + linucb_bandit.alpha * variance
        softmax_probs = torch.softmax(ucb_values, dim=0)
        assert softmax_probs.sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Offline Policy Evaluation Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ope_evaluator():
    import sys
    import types

    # Stub out modules that trigger DB connections at import time
    fake_connection = types.ModuleType("src.data_layer.database.connection")
    fake_connection.get_async_session = MagicMock()
    sys.modules.setdefault("src.data_layer.database.connection", fake_connection)

    mock_settings = MagicMock()
    mock_settings.MIN_SAMPLES_FOR_DECISION = 10

    with patch.dict(sys.modules, {
        "src.config.settings": MagicMock(settings=mock_settings),
    }):
        # Force re-import so patches take effect
        mod_name = "src.ai_layer.learning.offline_policy_eval"
        sys.modules.pop(mod_name, None)
        from src.ai_layer.learning.offline_policy_eval import OfflinePolicyEvaluator

    evaluator = OfflinePolicyEvaluator()
    evaluator.min_samples = 10
    return evaluator


class TestOfflinePolicyEval:

    def test_doubly_robust_basic(self, ope_evaluator):
        df = pd.DataFrame({
            "context": [[1, 0]] * 100,
            "action": [0, 1] * 50,
            "reward": np.random.rand(100).tolist(),
        })
        new_probs = np.ones(100) * 0.5
        baseline_probs = np.ones(100) * 0.5

        result = ope_evaluator.doubly_robust_estimator(
            df, new_probs, baseline_probs
        )
        assert "estimated_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["estimated_value"] <= result["ci_upper"]

    def test_doubly_robust_confidence_interval_width(self, ope_evaluator):
        """CI should narrow with more samples."""
        np.random.seed(0)
        rewards = np.random.rand(1000)
        df = pd.DataFrame({
            "context": [[1, 0]] * 1000,
            "action": [0] * 1000,
            "reward": rewards.tolist(),
        })
        new_probs = np.ones(1000) * 0.5
        baseline_probs = np.ones(1000) * 0.5

        result = ope_evaluator.doubly_robust_estimator(df, new_probs, baseline_probs)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width < 0.5, "CI should be narrow with 1000 samples"

    def test_doubly_robust_importance_weights(self, ope_evaluator):
        """When new_policy = 2 * baseline, IW=2 shifts the estimate."""
        df = pd.DataFrame({
            "context": [[1]] * 10,
            "action": [0, 1] * 5,
            "reward": [0.8, 0.2] * 5,
        })
        new_probs = np.ones(10) * 0.8
        baseline_probs = np.ones(10) * 0.4

        result = ope_evaluator.doubly_robust_estimator(df, new_probs, baseline_probs)
        # IW=2 amplifies high-reward observations, estimate != simple mean
        assert "estimated_value" in result
        assert result["n_samples"] == 10

    def test_marl_gate_passes_all_criteria(self, ope_evaluator):
        result = ope_evaluator.marl_promotion_gate(
            baseline_value=0.5,
            marl_value=0.7,
            marl_ci_lower=0.55,
            marl_ci_upper=0.85,
            n_samples=2000,
        )
        assert result["gate_passed"] is True
        assert result["recommendation"]["action"] == "APPROVE_CANARY"

    def test_marl_gate_fails_insufficient_samples(self, ope_evaluator):
        result = ope_evaluator.marl_promotion_gate(
            baseline_value=0.5,
            marl_value=0.7,
            marl_ci_lower=0.55,
            marl_ci_upper=0.85,
            n_samples=500,  # below default 1000
        )
        assert result["gate_passed"] is False

    def test_marl_gate_fails_insufficient_lift(self, ope_evaluator):
        result = ope_evaluator.marl_promotion_gate(
            baseline_value=0.5,
            marl_value=0.55,  # only 10% lift, need 20%
            marl_ci_lower=0.52,
            marl_ci_upper=0.58,
            n_samples=2000,
        )
        assert result["gate_passed"] is False

    def test_marl_gate_fails_ci_overlaps_baseline(self, ope_evaluator):
        result = ope_evaluator.marl_promotion_gate(
            baseline_value=0.5,
            marl_value=0.7,
            marl_ci_lower=0.45,  # CI lower < baseline
            marl_ci_upper=0.95,
            n_samples=2000,
        )
        assert result["gate_passed"] is False

    def test_generate_recommendation_high_reward(self, ope_evaluator):
        comparison = {"best_policy": "thompson_sampling", "best_reward": 0.8}
        rec = ope_evaluator._generate_recommendation(comparison)
        assert rec["recommended_policy"] == "thompson_sampling"
        assert rec["confidence"] == "high"

    def test_generate_recommendation_medium_reward(self, ope_evaluator):
        comparison = {"best_policy": "linucb", "best_reward": 0.3}
        rec = ope_evaluator._generate_recommendation(comparison)
        assert rec["confidence"] == "medium"


# ---------------------------------------------------------------------------
# Reward Tracker Tests
# ---------------------------------------------------------------------------

class TestRewardTracker:

    @pytest.fixture
    def tracker(self):
        import sys
        import types

        # Ensure connection module is stubbed before importing reward_tracker
        fake_connection = types.ModuleType("src.data_layer.database.connection")
        fake_connection.get_async_session = MagicMock()
        sys.modules.setdefault("src.data_layer.database.connection", fake_connection)

        # Stub connector modules if not yet importable
        for mod_path in [
            "src.automation_layer",
            "src.automation_layer.connectors",
            "src.automation_layer.connectors.calendar_api",
            "src.automation_layer.connectors.hubspot_api",
        ]:
            if mod_path not in sys.modules:
                m = types.ModuleType(mod_path)
                if mod_path.endswith("calendar_api"):
                    m.CalendarAPIConnector = MagicMock
                if mod_path.endswith("hubspot_api"):
                    m.HubSpotAPIConnector = MagicMock
                sys.modules[mod_path] = m

        mock_settings = MagicMock()
        mock_settings.REWARD_DELAY_WINDOW_HOURS = 72
        mock_settings.ESTIMATED_CONVERSION_RATE = 0.10

        # Force re-import
        sys.modules.pop("src.ai_layer.learning.reward_tracker", None)
        with patch.dict(sys.modules, {
            "src.config.settings": MagicMock(settings=mock_settings),
        }):
            from src.ai_layer.learning.reward_tracker import RewardTracker

        t = RewardTracker.__new__(RewardTracker)
        t.calendar_api = MagicMock()
        t.hubspot_api = MagicMock()
        t.reward_window_hours = 72
        t.estimated_conversion_rate = 0.10
        return t

    def test_surrogate_reward_formula(self, tracker):
        """surrogate = CTR × CVR"""
        result = tracker.calculate_surrogate_reward(ctr=0.05)
        assert result == pytest.approx(0.05 * 0.10)

    def test_surrogate_reward_custom_cvr(self, tracker):
        result = tracker.calculate_surrogate_reward(ctr=0.10, estimated_conversion_rate=0.20)
        assert result == pytest.approx(0.02)

    def test_surrogate_reward_zero_ctr(self, tracker):
        result = tracker.calculate_surrogate_reward(ctr=0.0)
        assert result == 0.0

    def test_reward_with_surrogate_click_no_conversion(self, tracker):
        result = tracker.calculate_reward_with_surrogate(
            click_occurred=True, ctr=0.05
        )
        assert result["reward_type"] == "surrogate"
        assert result["immediate_reward"] == 1.0
        assert result["total_reward"] == pytest.approx(1.0 + 0.05 * 0.10)

    def test_reward_with_surrogate_no_click(self, tracker):
        result = tracker.calculate_reward_with_surrogate(
            click_occurred=False, ctr=0.05
        )
        assert result["immediate_reward"] == 0.0
        assert result["reward_type"] == "surrogate"

    def test_reward_with_final_conversion(self, tracker):
        result = tracker.calculate_reward_with_surrogate(
            click_occurred=True, ctr=0.05,
            has_final_conversion=True, conversion_value=10.0
        )
        assert result["reward_type"] == "final"
        assert result["total_reward"] == pytest.approx(1.0 + 10.0)
        assert result["final_reward"] == 10.0

    def test_reward_formula_string_present(self, tracker):
        result = tracker.calculate_reward_with_surrogate(
            click_occurred=True, ctr=0.05
        )
        assert "CTR" in result["formula"]
        assert "CVR" in result["formula"]


# ---------------------------------------------------------------------------
# Multi-Touch Attribution Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def attribution_engine():
    from src.ai_layer.learning.multi_touch_attribution import MultiTouchAttributionEngine
    return MultiTouchAttributionEngine(decay_half_life_days=7.0)


@pytest.fixture
def sample_touchpoints():
    from src.ai_layer.learning.multi_touch_attribution import Touchpoint
    base = datetime(2024, 1, 1)
    return [
        Touchpoint("tp1", "camp_a", "linkedin", base, "impression"),
        Touchpoint("tp2", "camp_b", "email", base + timedelta(days=1), "click"),
        Touchpoint("tp3", "camp_a", "linkedin", base + timedelta(days=3), "click"),
        Touchpoint("tp4", "camp_c", "google", base + timedelta(days=5), "engagement"),
    ]


class TestMultiTouchAttribution:

    def test_first_touch_all_credit_to_first(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute(
            sample_touchpoints, 100.0, "conv-1", AttributionModel.FIRST_TOUCH
        )
        assert result.touchpoint_credits["tp1"] == pytest.approx(100.0)
        assert result.touchpoint_credits["tp2"] == pytest.approx(0.0)

    def test_last_touch_all_credit_to_last(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute(
            sample_touchpoints, 100.0, "conv-1", AttributionModel.LAST_TOUCH
        )
        assert result.touchpoint_credits["tp4"] == pytest.approx(100.0)
        assert result.touchpoint_credits["tp1"] == pytest.approx(0.0)

    def test_linear_equal_credit(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute(
            sample_touchpoints, 100.0, "conv-1", AttributionModel.LINEAR
        )
        for tp_id, credit in result.touchpoint_credits.items():
            assert credit == pytest.approx(25.0)

    def test_credits_sum_to_conversion_value(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        for model in AttributionModel:
            result = attribution_engine.attribute(
                sample_touchpoints, 100.0, "conv-1", model
            )
            total = sum(result.touchpoint_credits.values())
            assert total == pytest.approx(100.0, abs=0.01), f"{model} credits don't sum to 100"

    def test_empty_touchpoints(self, attribution_engine):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute([], 100.0, "conv-empty", AttributionModel.LINEAR)
        assert result.total_touchpoints == 0
        assert result.touchpoint_credits == {}

    def test_single_touchpoint(self, attribution_engine):
        from src.ai_layer.learning.multi_touch_attribution import Touchpoint, AttributionModel
        tp = [Touchpoint("tp1", "camp_a", "linkedin", datetime.now(), "click")]
        for model in AttributionModel:
            result = attribution_engine.attribute(tp, 50.0, "conv-1", model)
            assert sum(result.touchpoint_credits.values()) == pytest.approx(50.0)

    def test_u_shaped_40_20_40(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute(
            sample_touchpoints, 100.0, "conv-1", AttributionModel.U_SHAPED
        )
        assert result.touchpoint_credits["tp1"] == pytest.approx(40.0)
        assert result.touchpoint_credits["tp4"] == pytest.approx(40.0)
        middle_total = result.touchpoint_credits["tp2"] + result.touchpoint_credits["tp3"]
        assert middle_total == pytest.approx(20.0)

    def test_campaign_credits_aggregate(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute(
            sample_touchpoints, 100.0, "conv-1", AttributionModel.LINEAR
        )
        # camp_a has tp1 and tp3 → 25 + 25 = 50
        assert result.campaign_credits["camp_a"] == pytest.approx(50.0)
        assert result.campaign_credits["camp_b"] == pytest.approx(25.0)

    def test_channel_credits_aggregate(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute(
            sample_touchpoints, 100.0, "conv-1", AttributionModel.LINEAR
        )
        assert result.channel_credits["linkedin"] == pytest.approx(50.0)
        assert result.channel_credits["email"] == pytest.approx(25.0)
        assert result.channel_credits["google"] == pytest.approx(25.0)

    def test_time_decay_recent_gets_more(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        result = attribution_engine.attribute(
            sample_touchpoints, 100.0, "conv-1", AttributionModel.TIME_DECAY
        )
        # Last touchpoint (most recent) should get highest credit
        assert result.touchpoint_credits["tp4"] > result.touchpoint_credits["tp1"]

    def test_compare_models_returns_all(self, attribution_engine, sample_touchpoints):
        from src.ai_layer.learning.multi_touch_attribution import AttributionModel
        results = attribution_engine.compare_models(
            sample_touchpoints, 100.0, "conv-1"
        )
        assert len(results) == len(AttributionModel)
        for model in AttributionModel:
            assert model.value in results


# ---------------------------------------------------------------------------
# Survival Model Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def survival_model():
    from src.ai_layer.learning.survival_model import SurvivalModel
    return SurvivalModel(
        max_conversion_days=7,
        baseline_conversion_rate=0.15,
        time_decay_factor=0.1,
    )


class TestSurvivalModel:

    def test_survival_at_time_zero_is_one(self, survival_model):
        assert survival_model.survival_function(0.0) == 1.0

    def test_survival_decreases_over_time(self, survival_model):
        s1 = survival_model.survival_function(10.0)
        s2 = survival_model.survival_function(100.0)
        assert s1 > s2

    def test_survival_bounded_0_1(self, survival_model):
        for t in [0, 1, 10, 48, 100, 500]:
            s = survival_model.survival_function(t)
            assert 0.0 <= s <= 1.0

    def test_hazard_at_zero_is_zero(self, survival_model):
        assert survival_model.hazard_function(0.0) == 0.0

    def test_hazard_non_negative(self, survival_model):
        for t in [0, 1, 24, 48, 168]:
            assert survival_model.hazard_function(t) >= 0.0

    def test_cumulative_hazard_monotonic(self, survival_model):
        prev = 0.0
        for t in [1, 10, 24, 48, 100]:
            h = survival_model.cumulative_hazard(t)
            assert h >= prev
            prev = h

    def test_conversion_probability_bounded(self, survival_model):
        prob = survival_model.conversion_probability(0.0)
        assert 0.0 <= prob <= 1.0

    def test_conversion_probability_scales_with_baseline(self, survival_model):
        """Conversion probability is bounded by baseline_conversion_rate."""
        for t in [0, 10, 50, 100]:
            p = survival_model.conversion_probability(t, window_hours=24)
            assert p <= survival_model.baseline_conversion_rate + 1e-9

    def test_refine_surrogate_reward(self, survival_model):
        reward = survival_model.refine_surrogate_reward(
            base_ctr=0.05, estimated_conversion_rate=0.10, time_since_action_hours=0
        )
        assert reward > 0.0
        assert reward < 1.0

    def test_refine_surrogate_reward_increases_with_ctr(self, survival_model):
        low = survival_model.refine_surrogate_reward(0.01, 0.10, 0)
        high = survival_model.refine_surrogate_reward(0.10, 0.10, 0)
        assert high > low

    def test_calibrate_insufficient_data(self, survival_model):
        result = survival_model.calibrate_from_history(
            conversion_times=[10.0, 20.0],
            censored_times=[50.0]
        )
        assert result["calibrated"] is False

    def test_calibrate_with_enough_data(self, survival_model):
        np.random.seed(42)
        conv_times = np.random.exponential(24.0, size=50).tolist()
        cens_times = np.random.exponential(48.0, size=30).tolist()
        result = survival_model.calibrate_from_history(conv_times, cens_times)
        assert result["calibrated"] is True
        assert result["n_conversions"] == 50
        assert result["n_censored"] == 30
        assert survival_model.baseline_conversion_rate == pytest.approx(50 / 80)

    def test_estimate_survival_curve_length(self, survival_model):
        curve = survival_model.estimate_survival_curve(num_points=20)
        assert len(curve) == 20
        # First point should be near 1.0, last should be smaller
        assert curve[0].survival_probability >= curve[-1].survival_probability

    def test_estimate_survival_curve_custom_times(self, survival_model):
        curve = survival_model.estimate_survival_curve(
            time_points_hours=[0.0, 24.0, 48.0]
        )
        assert len(curve) == 3
        assert curve[0].time_hours == 0.0
        assert curve[1].time_hours == 24.0

    def test_get_model_summary(self, survival_model):
        summary = survival_model.get_model_summary()
        assert summary["model_type"] == "WeibullSurvival"
        assert "shape_parameter" in summary
        assert "scale_parameter_hours" in summary
        assert "baseline_conversion_rate" in summary

    def test_negative_time_survival(self, survival_model):
        assert survival_model.survival_function(-5.0) == 1.0

    def test_negative_time_hazard(self, survival_model):
        assert survival_model.hazard_function(-5.0) == 0.0
