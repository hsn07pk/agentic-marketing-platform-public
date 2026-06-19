# Per Research Plan Section 8.1 - MLOps Infrastructure
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ...ai_layer.learning.mlflow_integration import get_mlflow_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mlflow", tags=["MLflow"])


class PolicyLogRequest(BaseModel):
    policy_name: str = Field(..., description="Name for the policy")
    policy_type: str = Field(default="thompson_sampling", description="Algorithm type")
    arms_data: List[Dict[str, Any]] = Field(..., description="Arm configurations")
    metrics: Dict[str, float] = Field(default={}, description="Performance metrics")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Algorithm parameters")
    tags: Optional[Dict[str, str]] = Field(default=None, description="Additional tags")


class PolicyUpdateRequest(BaseModel):
    policy_name: str
    experiment_id: str
    arm_updates: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]


class PolicyPromoteRequest(BaseModel):
    policy_name: str
    version: str
    stage: str = Field(default="Production", description="Target stage (Staging or Production)")


class MLflowStatusResponse(BaseModel):
    connected: bool
    tracking_uri: str
    experiment_name: str
    message: str


@router.get("/status", response_model=MLflowStatusResponse)
async def get_mlflow_status():
    registry = get_mlflow_registry()
    connected = registry._ensure_initialized()
    
    return {
        "connected": connected,
        "tracking_uri": registry.tracking_uri,
        "experiment_name": registry.experiment_name,
        "message": "MLflow connected" if connected else "MLflow not available"
    }


@router.post("/policies/log")
async def log_policy(request: PolicyLogRequest):
    registry = get_mlflow_registry()
    
    run_id = registry.log_bandit_policy(
        policy_name=request.policy_name,
        policy_type=request.policy_type,
        arms_data=request.arms_data,
        metrics=request.metrics,
        parameters=request.parameters,
        tags=request.tags
    )
    
    if run_id:
        return {
            "success": True,
            "run_id": run_id,
            "policy_name": request.policy_name,
            "message": f"Policy logged successfully"
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="MLflow not available or logging failed"
        )


@router.post("/policies/update")
async def update_policy_metrics(request: PolicyUpdateRequest):
    registry = get_mlflow_registry()
    
    run_id = registry.update_bandit_metrics(
        policy_name=request.policy_name,
        experiment_id=request.experiment_id,
        arm_updates=request.arm_updates,
        performance_metrics=request.performance_metrics
    )
    
    if run_id:
        return {
            "success": True,
            "run_id": run_id,
            "message": "Policy metrics updated"
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="MLflow not available or update failed"
        )


@router.get("/policies/{policy_name}/versions")
async def list_policy_versions(policy_name: str):
    registry = get_mlflow_registry()
    versions = registry.list_policy_versions(policy_name)
    
    return {
        "policy_name": policy_name,
        "versions": versions,
        "count": len(versions)
    }


@router.post("/policies/promote")
async def promote_policy(request: PolicyPromoteRequest):
    if request.stage not in ["Staging", "Production", "Archived", "None"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage: {request.stage}. Must be Staging, Production, Archived, or None"
        )
    
    registry = get_mlflow_registry()
    success = registry.promote_policy(
        policy_name=request.policy_name,
        version=request.version,
        stage=request.stage
    )
    
    if success:
        return {
            "success": True,
            "policy_name": request.policy_name,
            "version": request.version,
            "stage": request.stage,
            "message": f"Policy promoted to {request.stage}"
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="Failed to promote policy"
        )


@router.get("/runs/latest")
async def get_latest_runs(max_results: int = Query(default=10, le=100)):
    registry = get_mlflow_registry()
    runs = registry.get_latest_runs(max_results=max_results)
    
    return {
        "runs": runs,
        "count": len(runs),
        "experiment_name": registry.experiment_name
    }


@router.get("/health")
async def mlflow_health():
    registry = get_mlflow_registry()
    connected = registry._ensure_initialized()

    if connected:
        return {"status": "healthy", "service": "mlflow"}
    else:
        raise HTTPException(
            status_code=503,
            detail="MLflow not available"
        )


@router.post("/autonomous/check")
async def trigger_autonomous_check():
    """Trigger the autonomous MLOps check. Runs the same logic as the scheduled task
    (check experiments for completion, log to MLflow, auto-promote models)."""
    from ...ai_layer.learning.autonomous_mlops import get_mlops_orchestrator

    orchestrator = get_mlops_orchestrator()
    result = await orchestrator.check_and_complete_experiments()

    return {
        "success": True,
        "message": "Autonomous MLOps check completed",
        "result": result
    }


@router.get("/autonomous/active-policy/{campaign_id}")
async def get_active_policy(campaign_id: str):
    """
    Get the currently active policy for a campaign.

    Automatically selects the best available model:
    - Production stage if available
    - Staging stage if no production
    - Latest development model as fallback
    """
    from ...ai_layer.learning.autonomous_mlops import get_mlops_orchestrator

    orchestrator = get_mlops_orchestrator()
    policy = await orchestrator.get_active_policy(campaign_id)

    if policy:
        return {
            "success": True,
            "model_name": policy["model_name"],
            "version": policy["version"],
            "stage": policy["stage"],
            "available": True
        }
    else:
        return {
            "success": False,
            "message": f"No active policy for campaign {campaign_id}",
            "available": False
        }


@router.post("/autonomous/select-arm/{campaign_id}")
async def select_arm_for_campaign(campaign_id: str, context: Dict[str, Any] = None):
    """Select the best arm using the active (production) policy for autonomous arm selection."""
    from ...ai_layer.learning.autonomous_mlops import get_mlops_orchestrator

    orchestrator = get_mlops_orchestrator()
    result = await orchestrator.select_arm_for_campaign(campaign_id, context)

    if result:
        return {
            "success": True,
            "selected_arm": result["selected_arm"],
            "sample_value": result.get("sample_value"),
            "expected_ctr": result.get("expected_ctr"),
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "model_stage": result["model_stage"],
            "all_samples": result.get("all_samples", [])
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No active policy available for campaign {campaign_id}"
        )


@router.post("/test/log-experiment")
async def test_log_experiment(
    experiment_name: str = "test-experiment",
    num_arms: int = 3,
    total_pulls: int = 1000
):
    """TEST: Create and log a mock experiment to MLflow to verify logging works."""
    import random
    import numpy as np
    from datetime import timedelta
    from ...ai_layer.learning.autonomous_mlops import (
        get_mlops_orchestrator,
        ModelType,
        ProfessionalNamingConvention
    )

    num_arms = max(2, min(5, num_arms))
    total_pulls = max(100, min(10000, total_pulls))

    true_ctrs = [random.uniform(0.02, 0.15) for _ in range(num_arms)]
    best_arm_idx = np.argmax(true_ctrs)

    arms_data = []
    remaining_pulls = total_pulls
    pulls_distribution = np.random.dirichlet([2] * num_arms) * total_pulls

    for i in range(num_arms):
        arm_pulls = int(pulls_distribution[i])
        if i == num_arms - 1:
            arm_pulls = remaining_pulls
        remaining_pulls -= arm_pulls

        successes = np.random.binomial(arm_pulls, true_ctrs[i])
        failures = arm_pulls - successes

        # Bayesian posterior: Beta(alpha=1+successes, beta=1+failures)
        alpha = 1 + successes
        beta = 1 + failures

        arms_data.append({
            "arm_id": f"variant_{chr(65+i)}",  # variant_A, variant_B, etc.
            "pulls": arm_pulls,
            "successes": int(successes),
            "failures": int(failures),
            "alpha": float(alpha),
            "beta": float(beta),
            "total_reward": float(successes),
            "variant_data": {
                "headline": f"Test Headline {chr(65+i)}",
                "cta": f"CTA {i+1}"
            }
        })

    winner_arm = max(arms_data, key=lambda a: a["successes"] / max(a["pulls"], 1))

    n_samples = 10000
    arm_wins = {arm["arm_id"]: 0 for arm in arms_data}
    for _ in range(n_samples):
        samples = {}
        for arm in arms_data:
            samples[arm["arm_id"]] = np.random.beta(arm["alpha"], arm["beta"])
        winner_id = max(samples, key=samples.get)
        arm_wins[winner_id] += 1

    confidence = arm_wins[winner_arm["arm_id"]] / n_samples

    class MockExperiment:
        def __init__(self):
            self.id = f"test-{random.randint(1000, 9999)}"
            self.name = experiment_name
            self.algorithm = "thompson_sampling"
            self.campaign_id = f"campaign-{random.randint(100, 999)}"
            self.started_at = datetime.utcnow() - timedelta(hours=random.uniform(24, 168))
            self.parameters = {
                "confidence_threshold": 0.95,
                "target_sample_size": total_pulls,
                "duration": 14
            }

    class MockArm:
        def __init__(self, data):
            self.arm_id = data["arm_id"]
            self.pulls = data["pulls"]
            self.successes = data["successes"]
            self.failures = data["failures"]
            self.alpha = data["alpha"]
            self.beta = data["beta"]
            self.total_reward = data["total_reward"]
            self.variant_data = data["variant_data"]

    mock_experiment = MockExperiment()
    mock_arms = [MockArm(d) for d in arms_data]

    completion_result = {
        "should_complete": True,
        "reason": "Test experiment - statistical significance reached",
        "winner_arm": winner_arm["arm_id"],
        "confidence": confidence,
        "auto_promote": confidence >= 0.99
    }

    orchestrator = get_mlops_orchestrator()
    run_id = await orchestrator._log_completed_experiment(
        mock_experiment,
        mock_arms,
        completion_result
    )

    if run_id:
        return {
            "success": True,
            "run_id": run_id,
            "experiment_name": experiment_name,
            "num_arms": num_arms,
            "total_pulls": total_pulls,
            "winner": {
                "arm_id": winner_arm["arm_id"],
                "confidence": f"{confidence:.2%}",
                "ctr": f"{(winner_arm['successes'] / max(winner_arm['pulls'], 1)) * 100:.2f}%"
            },
            "arms": [
                {
                    "arm_id": arm["arm_id"],
                    "pulls": arm["pulls"],
                    "ctr": f"{(arm['successes'] / max(arm['pulls'], 1)) * 100:.2f}%"
                }
                for arm in arms_data
            ],
            "message": "Test experiment logged successfully. Check MLflow dashboard."
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="Failed to log test experiment to MLflow"
        )
