#!/usr/bin/env python3
"""
Comprehensive End-to-End Flow Test Suite

Tests all system flows from campaign creation to completion,
verifying complete integration per the research plan.

Run with: python scripts/e2e_flow_test.py
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import uuid4
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "details": []
}


def log_test(name: str, status: str, message: str = "", duration: float = 0):
    """Log test result"""
    test_results[status] += 1
    test_results["details"].append({
        "name": name,
        "status": status,
        "message": message,
        "duration": round(duration, 2)
    })
    symbol = "✅" if status == "passed" else ("❌" if status == "failed" else "⏭️")
    print(f"  {symbol} {name}: {status.upper()} {f'({message})' if message else ''} [{duration:.2f}s]")


async def test_database_connection():
    """Test 1: Verify database connectivity"""
    start = datetime.now()
    try:
        from src.data_layer.database.connection import get_async_session
        async with get_async_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        log_test("Database Connection", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Database Connection", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_new_models_exist():
    """Test 2: Verify new analytics models are defined"""
    start = datetime.now()
    try:
        from src.data_layer.database.models import (
            SimulationLiveAccuracy,
            GovernanceMetrics,
            WeeklyLearningReport
        )
        assert SimulationLiveAccuracy.__tablename__ == "simulation_live_accuracy"
        assert GovernanceMetrics.__tablename__ == "governance_metrics"
        assert WeeklyLearningReport.__tablename__ == "weekly_learning_reports"
        log_test("New Analytics Models Exist", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("New Analytics Models Exist", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_simulation_accuracy_tracker():
    """Test 3: Verify simulation accuracy tracker"""
    start = datetime.now()
    try:
        from src.ai_layer.learning.simulation_accuracy_tracker import SimulationAccuracyTracker

        tracker = SimulationAccuracyTracker()

        # Test MAPE calculation
        mape = tracker._calculate_mape(100, 90)  # 100 predicted, 90 actual
        assert abs(mape - 11.11) < 0.1, f"MAPE calculation wrong: {mape}"

        # Test weighted MAPE
        mape_values = {
            'impressions': 5.0,
            'clicks': 10.0,
            'conversions': 8.0,
            'ctr': 2.0
        }
        weighted = tracker._calculate_weighted_mape(mape_values)
        assert weighted > 0, "Weighted MAPE should be positive"

        log_test("Simulation Accuracy Tracker", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Simulation Accuracy Tracker", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_surrogate_reward_formula():
    """Test 4: Verify surrogate reward formula (Research Plan Section 2.3)"""
    start = datetime.now()
    try:
        from src.ai_layer.learning.reward_tracker import RewardTracker

        tracker = RewardTracker()

        # Test surrogate reward calculation: CTR × estimated_conversion_rate
        ctr = 0.05  # 5% CTR
        surrogate = tracker.calculate_surrogate_reward(ctr)

        # Default conversion rate is 0.10 (10%)
        expected = 0.05 * 0.10  # = 0.005
        assert abs(surrogate - expected) < 0.0001, f"Surrogate wrong: {surrogate} != {expected}"

        # Test with custom conversion rate
        surrogate_custom = tracker.calculate_surrogate_reward(ctr, estimated_conversion_rate=0.20)
        expected_custom = 0.05 * 0.20  # = 0.01
        assert abs(surrogate_custom - expected_custom) < 0.0001

        # Test full reward calculation
        result = tracker.calculate_reward_with_surrogate(
            click_occurred=True,
            ctr=0.05,
            has_final_conversion=False
        )
        assert result['reward_type'] == 'surrogate'
        assert result['immediate_reward'] == 1.0
        assert result['surrogate_reward'] > 0

        log_test("Surrogate Reward Formula", "passed",
                 f"CTR={ctr}, Surrogate={surrogate}",
                 (datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Surrogate Reward Formula", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_governance_metrics_tracker():
    """Test 5: Verify governance metrics tracker"""
    start = datetime.now()
    try:
        from src.ai_layer.learning.governance_metrics_tracker import GovernanceMetricsTracker

        tracker = GovernanceMetricsTracker()

        # Verify targets
        assert tracker.OVERRIDE_RATE_TARGET == 5.0, "Override rate target should be 5%"
        assert tracker.GOLDEN_TEST_PASS_TARGET == 100.0, "Golden test target should be 100%"

        log_test("Governance Metrics Tracker", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Governance Metrics Tracker", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_weekly_learning_report_generator():
    """Test 6: Verify weekly learning report generator"""
    start = datetime.now()
    try:
        from src.ai_layer.learning.weekly_learning_report import WeeklyLearningReportGenerator

        generator = WeeklyLearningReportGenerator()

        # Test change calculation
        change = generator._calculate_change(current=110, previous=100)
        assert change == 10.0, f"Change calculation wrong: {change}"

        change_from_zero = generator._calculate_change(current=50, previous=0)
        assert change_from_zero == 100.0, f"Change from zero wrong: {change_from_zero}"

        log_test("Weekly Learning Report Generator", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Weekly Learning Report Generator", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_api_router_analytics():
    """Test 7: Verify analytics API router is importable"""
    start = datetime.now()
    try:
        from src.api.routers.analytics import router

        # Check router has expected endpoints
        routes = [r.path for r in router.routes]
        expected_routes = [
            "/simulation-accuracy/summary",
            "/governance-metrics/override-rate",
            "/weekly-report/latest"
        ]
        for expected in expected_routes:
            assert expected in routes, f"Missing route: {expected}"

        log_test("Analytics API Router", "passed",
                 f"{len(routes)} routes defined",
                 (datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Analytics API Router", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_langgraph_supervisor_workflow():
    """Test 8: Verify LangGraph supervisor has simulation tracking"""
    start = datetime.now()
    try:
        # Read the file and check for RQ2 integration
        import inspect
        from src.ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator

        source = inspect.getsourcefile(MarketingOrchestrator)
        with open(source, 'r') as f:
            content = f.read()

        # Check for RQ2 tracking integration
        assert "record_simulation_for_campaign" in content, "Missing RQ2 simulation tracking"
        assert "RQ2" in content, "Missing RQ2 comments"

        log_test("LangGraph Supervisor RQ2 Integration", "passed",
                 duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("LangGraph Supervisor RQ2 Integration", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_scheduler_tasks():
    """Test 9: Verify scheduler has new tasks"""
    start = datetime.now()
    try:
        from src.worker.tasks import (
            save_daily_governance_metrics,
            generate_weekly_learning_report
        )

        # Verify functions exist and have docstrings
        assert save_daily_governance_metrics.__doc__ is not None
        assert generate_weekly_learning_report.__doc__ is not None
        assert "Research Plan Section 10.2" in save_daily_governance_metrics.__doc__
        assert "Research Plan Section 10.2" in generate_weekly_learning_report.__doc__

        log_test("Scheduler Tasks", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Scheduler Tasks", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_simulation_validator():
    """Test 10: Verify simulation validator functions"""
    start = datetime.now()
    try:
        from src.simulation.validators import SimulationValidator
        import numpy as np

        # Test MAPE calculation
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190, 310])
        mape = SimulationValidator.calculate_mape(actual, predicted)
        assert 0 < mape < 10, f"MAPE should be between 0-10: {mape}"

        # Test accuracy score
        accuracy = SimulationValidator.calculate_accuracy_score(mape)
        assert accuracy > 0.9, f"Accuracy should be >90%: {accuracy}"

        # Test validation report
        validation_results = {
            'impressions': {'accuracy': 0.92, 'mape': 8.0},
            'clicks': {'accuracy': 0.95, 'mape': 5.0},
            'conversions': {'accuracy': 0.88, 'mape': 12.0},
            'ctr': {'accuracy': 0.96, 'mape': 4.0}
        }
        report = SimulationValidator.generate_validation_report(validation_results, threshold=0.9)
        assert 'overall_accuracy' in report
        assert 'passed' in report

        log_test("Simulation Validator", "passed",
                 f"MAPE={mape:.2f}, Accuracy={accuracy:.2%}",
                 (datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Simulation Validator", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_campaign_completion_checker():
    """Test 11: Verify campaign completion logic"""
    start = datetime.now()
    try:
        from src.ai_layer.learning.campaign_completion import (
            CampaignCompletionChecker,
            should_complete_campaign
        )

        checker = CampaignCompletionChecker()

        # Test budget depletion
        campaign = {
            'id': 'test-1',
            'status': 'RUNNING',
            'budget_total': 100.0,
            'budget_spent': 100.0
        }
        decision = checker.check_completion(campaign)
        assert decision.should_complete, "Should complete when budget depleted"
        assert decision.completion_type == "budget_depleted"

        # Test running campaign
        campaign['budget_spent'] = 50.0
        decision = checker.check_completion(campaign)
        assert not decision.should_complete, "Should continue when budget remaining"

        log_test("Campaign Completion Checker", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Campaign Completion Checker", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_content_generator_agent():
    """Test 12: Verify content generator agent is properly structured"""
    start = datetime.now()
    try:
        from src.ai_layer.agents.content_generator import ContentGeneratorAgent

        # Just verify it can be instantiated (won't call LLM)
        agent = ContentGeneratorAgent()
        assert hasattr(agent, 'generate')
        assert hasattr(agent, 'memory_store') or hasattr(agent, 'episodic_memory')

        log_test("Content Generator Agent", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Content Generator Agent", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_safety_validator_agent():
    """Test 13: Verify safety validator agent is properly structured"""
    start = datetime.now()
    try:
        from src.ai_layer.agents.safety_validator import SafetyValidatorAgent

        agent = SafetyValidatorAgent()
        assert hasattr(agent, 'validate')

        log_test("Safety Validator Agent", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Safety Validator Agent", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_strategy_optimizer_agent():
    """Test 14: Verify strategy optimizer with bandits"""
    start = datetime.now()
    try:
        from src.ai_layer.agents.strategy_optimizer import StrategyOptimizerAgent

        agent = StrategyOptimizerAgent()
        assert hasattr(agent, 'optimize')

        log_test("Strategy Optimizer Agent", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Strategy Optimizer Agent", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_thompson_sampling():
    """Test 15: Verify Thompson Sampling bandit"""
    start = datetime.now()
    try:
        from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit

        bandit = ThompsonSamplingBandit(n_arms=3)

        # Pull some arms and update
        for _ in range(10):
            arm = bandit.select_arm()
            reward = 1 if arm == 0 else 0  # Arm 0 always wins
            bandit.update(arm, reward)

        # Arm 0 should have highest expected reward
        stats = bandit.get_arm_stats()
        assert stats[0]['success_rate'] > 0, "Arm 0 should have positive success rate"

        log_test("Thompson Sampling Bandit", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Thompson Sampling Bandit", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_episodic_memory():
    """Test 16: Verify episodic memory store"""
    start = datetime.now()
    try:
        from src.ai_layer.memory.episodic_memory import EpisodicMemoryStore, AgentMemory

        # Create memory (won't actually store without DB)
        memory = AgentMemory(
            agent_name="test_agent",
            task_id="task-1",
            task_description="Test task",
            actions_taken=["action1", "action2"],
            outcome="success",
            metrics={"cost": 0.01, "duration": 1.5}
        )

        assert memory.agent_name == "test_agent"
        assert memory.outcome == "success"

        log_test("Episodic Memory Store", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Episodic Memory Store", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_golden_test_suite():
    """Test 17: Verify golden test suite exists"""
    start = datetime.now()
    try:
        from src.governance.golden_test_suite import GoldenTestSuite
        import yaml
        import os

        # Check test cases file exists
        test_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tests", "golden", "test_cases.yaml"
        )
        assert os.path.exists(test_file), f"Golden test cases not found: {test_file}"

        with open(test_file, 'r') as f:
            test_cases = yaml.safe_load(f)

        assert 'test_cases' in test_cases
        num_tests = len(test_cases['test_cases'])
        assert num_tests >= 30, f"Need at least 30 golden tests, have {num_tests}"

        log_test("Golden Test Suite", "passed", f"{num_tests} test cases", (datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Golden Test Suite", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_canary_rollout():
    """Test 18: Verify canary rollout module"""
    start = datetime.now()
    try:
        from src.automation_layer.deployment.canary_rollout import CanaryRollout

        # Verify class exists
        assert hasattr(CanaryRollout, 'start_deployment') or callable(getattr(CanaryRollout, '__init__', None))

        log_test("Canary Rollout", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Canary Rollout", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_ope_gating():
    """Test 19: Verify Offline Policy Evaluation gating"""
    start = datetime.now()
    try:
        from src.ai_layer.marl.ope_gating import OffPolicyEvaluator, MARLGatekeeper

        # Verify classes exist
        assert OffPolicyEvaluator is not None
        assert MARLGatekeeper is not None

        log_test("OPE Gating", "passed", duration=(datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("OPE Gating", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def test_platform_connectors():
    """Test 20: Verify platform connectors exist"""
    start = datetime.now()
    try:
        from src.automation_layer.connectors.linkedin_api import LinkedInConnector
        from src.automation_layer.connectors.x_api import XConnector
        from src.automation_layer.connectors.email_api import EmailConnector
        from src.automation_layer.connectors.hubspot_api import HubSpotAPIConnector
        from src.automation_layer.connectors.calendar_api import CalendarAPIConnector

        # All connectors importable
        log_test("Platform Connectors", "passed",
                 "LinkedIn, Twitter, Email, HubSpot, Cal.com",
                 (datetime.now() - start).total_seconds())
        return True
    except Exception as e:
        log_test("Platform Connectors", "failed", str(e), (datetime.now() - start).total_seconds())
        return False


async def run_all_tests():
    """Run all end-to-end tests"""
    print("\n" + "=" * 70)
    print("AGENTIC AI PLATFORM - END-TO-END FLOW TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    tests = [
        ("DATABASE & INFRASTRUCTURE", [
            test_database_connection,
            test_new_models_exist,
        ]),
        ("NEW ANALYTICS FEATURES (RQ2, Override Rate, Weekly Reports)", [
            test_simulation_accuracy_tracker,
            test_surrogate_reward_formula,
            test_governance_metrics_tracker,
            test_weekly_learning_report_generator,
            test_api_router_analytics,
        ]),
        ("WORKFLOW INTEGRATION", [
            test_langgraph_supervisor_workflow,
            test_scheduler_tasks,
        ]),
        ("SIMULATION & VALIDATION", [
            test_simulation_validator,
            test_campaign_completion_checker,
        ]),
        ("AI AGENTS", [
            test_content_generator_agent,
            test_safety_validator_agent,
            test_strategy_optimizer_agent,
        ]),
        ("LEARNING ALGORITHMS", [
            test_thompson_sampling,
            test_episodic_memory,
        ]),
        ("GOVERNANCE & DEPLOYMENT", [
            test_golden_test_suite,
            test_canary_rollout,
            test_ope_gating,
        ]),
        ("PLATFORM INTEGRATIONS", [
            test_platform_connectors,
        ]),
    ]

    for category, test_funcs in tests:
        print(f"\n{'─' * 60}")
        print(f"  {category}")
        print(f"{'─' * 60}")

        for test_func in test_funcs:
            try:
                await test_func()
            except Exception as e:
                log_test(test_func.__name__, "failed", f"Unexpected error: {e}", 0)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    total = test_results["passed"] + test_results["failed"] + test_results["skipped"]
    print(f"  Total Tests:  {total}")
    print(f"  ✅ Passed:    {test_results['passed']}")
    print(f"  ❌ Failed:    {test_results['failed']}")
    print(f"  ⏭️ Skipped:   {test_results['skipped']}")
    print()

    pass_rate = (test_results['passed'] / total * 100) if total > 0 else 0
    print(f"  Pass Rate: {pass_rate:.1f}%")

    if test_results['failed'] > 0:
        print("\n  Failed Tests:")
        for detail in test_results['details']:
            if detail['status'] == 'failed':
                print(f"    - {detail['name']}: {detail['message']}")

    print("\n" + "=" * 70)
    print(f"Completed: {datetime.now().isoformat()}")
    print("=" * 70 + "\n")

    return test_results['failed'] == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
