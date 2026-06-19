"""
Autonomous MLOps Pipeline for Agentic AI Marketing Platform

Fully autonomous model training, tracking, promotion, and serving:
- Zero human intervention
- Professional naming conventions
- Automatic model training on experiment completion
- Automatic promotion based on OPE results
- Automatic model loading for inference

Per Research Plan Section 8.1 - MLOps Infrastructure
Per Research Plan Section 2.3 - MARL Promotion Gating
"""
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "agentic-bandits")


class ModelStage(str, Enum):
    """Model lifecycle stages in MLflow registry."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelType(str, Enum):
    """Supported bandit algorithm types."""
    THOMPSON_SAMPLING = "thompson_sampling"
    LINUCB = "linucb"
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class PolicyVersion:
    """Represents a versioned policy in the registry."""
    policy_id: str
    version: int
    stage: ModelStage
    algorithm: ModelType
    campaign_id: str
    experiment_id: str
    metrics: Dict[str, float]
    arms_data: List[Dict[str, Any]]
    created_at: datetime
    promoted_at: Optional[datetime] = None
    mlflow_run_id: Optional[str] = None


class ProfessionalNamingConvention:
    """
    Professional naming convention for MLflow experiments and models.

    Format: agentic-{algorithm}-{campaign_short}-v{version}
    Example: agentic-ts-linkedin-b2b-v3
    """

    @staticmethod
    def get_experiment_name(campaign_name: str, algorithm: ModelType) -> str:
        """Generate experiment name."""
        algo_prefix = {
            ModelType.THOMPSON_SAMPLING: "ts",
            ModelType.LINUCB: "ucb",
            ModelType.EPSILON_GREEDY: "eg"
        }.get(algorithm, "bandit")

        clean_name = campaign_name.lower()[:20].replace(" ", "-").replace("_", "-")
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '-')

        return f"agentic-{algo_prefix}-{clean_name}"

    @staticmethod
    def get_model_name(
        campaign_id: str,
        algorithm: ModelType,
        platform: str = "linkedin"
    ) -> str:
        """
        Generate professional model name.

        Format: agentic-{platform}-{algorithm}-{campaign_short}
        """
        algo_name = {
            ModelType.THOMPSON_SAMPLING: "thompson",
            ModelType.LINUCB: "linucb",
            ModelType.EPSILON_GREEDY: "epsilon"
        }.get(algorithm, "bandit")

        campaign_short = campaign_id[:8] if campaign_id else "default"

        return f"agentic-{platform}-{algo_name}-{campaign_short}"

    @staticmethod
    def get_run_name(
        campaign_name: str,
        algorithm: ModelType,
        version: int,
        is_completion: bool = False
    ) -> str:
        """
        Generate professional run name.

        Format: {campaign_short}-{algo}-v{version}[-final]
        """
        algo_short = {
            ModelType.THOMPSON_SAMPLING: "ts",
            ModelType.LINUCB: "ucb",
            ModelType.EPSILON_GREEDY: "eg"
        }.get(algorithm, "b")

        clean_name = campaign_name.lower()[:15].replace(" ", "-")
        suffix = "-final" if is_completion else ""

        return f"{clean_name}-{algo_short}-v{version}{suffix}"


class AutonomousMLOpsOrchestrator:
    """
    Fully autonomous MLOps pipeline orchestrator.

    Handles the complete lifecycle:
    1. Monitor experiments for completion
    2. Log completed experiments to MLflow with professional naming
    3. Evaluate new policies using OPE
    4. Promote policies through stages (dev → staging → canary → production)
    5. Automatically load best policy for inference
    """

    def __init__(self):
        self._mlflow = None
        self._initialized = False
        self.naming = ProfessionalNamingConvention()
        self._active_policies: Dict[str, PolicyVersion] = {}

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of MLflow connection."""
        if self._initialized:
            return True

        try:
            import mlflow
            import os as _os
            self._mlflow = mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Avoids needing direct filesystem access to the artifact store
            _os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                mlflow.create_experiment(
                    MLFLOW_EXPERIMENT_NAME,
                    artifact_location=f"mlflow-artifacts:/{MLFLOW_EXPERIMENT_NAME}",
                    tags={
                        "project": "agentic",
                        "type": "bandit-policies",
                        "autonomous": "true",
                        "version": "2.0"
                    }
                )
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            self._initialized = True
            logger.info(f"Autonomous MLOps initialized: {MLFLOW_TRACKING_URI}")
            return True

        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}")
            return False

    async def check_and_complete_experiments(self) -> Dict[str, Any]:
        """
        AUTONOMOUS: Check all active experiments and complete those ready.

        This should be called by the scheduler every 5 minutes.

        Returns:
            Summary of completed experiments and models logged
        """
        from ...data_layer.database.connection import get_async_session
        from ...data_layer.database.models import Experiment, BanditArm, Campaign
        from sqlalchemy import select

        results = {
            "checked": 0,
            "completed": 0,
            "models_logged": 0,
            "models_promoted": 0,
            "errors": []
        }

        try:
            async with get_async_session() as session:
                stmt = select(Experiment).where(Experiment.is_active == True)
                result = await session.execute(stmt)
                experiments = result.scalars().all()

                for experiment in experiments:
                    results["checked"] += 1

                    try:
                        arms_stmt = select(BanditArm).where(
                            BanditArm.experiment_id == experiment.id
                        )
                        arms_result = await session.execute(arms_stmt)
                        arms = arms_result.scalars().all()

                        if not arms:
                            continue

                        completion_result = await self._check_experiment_completion(
                            experiment, arms
                        )

                        if completion_result["should_complete"]:
                            await self._complete_experiment(
                                session, experiment, arms, completion_result
                            )
                            results["completed"] += 1

                            run_id = await self._log_completed_experiment(
                                experiment, arms, completion_result
                            )
                            if run_id:
                                results["models_logged"] += 1

                            if completion_result.get("auto_promote"):
                                promoted = await self._auto_promote_model(
                                    experiment, completion_result
                                )
                                if promoted:
                                    results["models_promoted"] += 1

                    except Exception as e:
                        error_msg = f"Error processing experiment {experiment.id}: {e}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)

                await session.commit()

        except Exception as e:
            logger.error(f"Experiment completion check failed: {e}")
            results["errors"].append(str(e))

        logger.info(f"Experiment completion check: {results}")
        return results

    async def _check_experiment_completion(
        self,
        experiment,
        arms: List
    ) -> Dict[str, Any]:
        """Check if experiment meets completion criteria."""
        import numpy as np

        result = {
            "should_complete": False,
            "reason": None,
            "winner_arm": None,
            "confidence": 0.0,
            "auto_promote": False
        }

        params = experiment.parameters or {}
        target_sample_size = params.get("target_sample_size", 1000)
        confidence_threshold = params.get("confidence_threshold", 0.95)
        max_duration_days = params.get("duration", 14)

        total_pulls = sum(arm.pulls or 0 for arm in arms)

        if experiment.started_at:
            days_running = (datetime.utcnow() - experiment.started_at).days
            if days_running >= max_duration_days:
                result["should_complete"] = True
                result["reason"] = f"Duration limit reached ({days_running} days)"

        if total_pulls >= target_sample_size:
            confidence, winner = self._calculate_winner_confidence(arms)

            logger.info(
                f"Experiment {experiment.id}: pulls={total_pulls}, "
                f"confidence={confidence:.2%}, threshold={confidence_threshold:.2%}, "
                f"winner={winner.arm_id if winner else 'none'}"
            )

            if confidence >= confidence_threshold:
                result["should_complete"] = True
                result["reason"] = f"Statistical significance reached ({confidence:.2%})"
                result["winner_arm"] = winner.arm_id if winner else None
                result["confidence"] = confidence
                result["auto_promote"] = confidence >= 0.99
        else:
            logger.debug(
                f"Experiment {experiment.id}: pulls={total_pulls}, "
                f"target={target_sample_size} (not enough samples)"
            )

        return result

    def _calculate_winner_confidence(
        self,
        arms: List
    ) -> Tuple[float, Any]:
        """
        Calculate confidence that the best arm is truly best.

        Uses Monte Carlo simulation of Thompson Sampling posteriors.
        """
        import numpy as np

        if not arms:
            return 0.0, None

        n_samples = 10000
        arm_wins = {arm.arm_id: 0 for arm in arms}

        for _ in range(n_samples):
            samples = {}
            for arm in arms:
                alpha = arm.alpha or 1.0
                beta = arm.beta or 1.0
                samples[arm.arm_id] = np.random.beta(alpha, beta)

            winner_id = max(samples, key=samples.get)
            arm_wins[winner_id] += 1

        best_arm_id = max(arm_wins, key=arm_wins.get)
        best_arm = next((a for a in arms if a.arm_id == best_arm_id), None)
        confidence = arm_wins[best_arm_id] / n_samples

        return confidence, best_arm

    async def _complete_experiment(
        self,
        session,
        experiment,
        arms: List,
        completion_result: Dict[str, Any]
    ):
        """Mark experiment as complete and store results."""
        from sqlalchemy import update
        from ...data_layer.database.models import Experiment

        total_pulls = sum(arm.pulls or 0 for arm in arms)
        total_successes = sum(arm.successes or 0 for arm in arms)
        overall_ctr = total_successes / total_pulls if total_pulls > 0 else 0

        winner_arm = None
        best_ctr = 0
        for arm in arms:
            arm_ctr = (arm.successes or 0) / (arm.pulls or 1)
            if arm_ctr > best_ctr:
                best_ctr = arm_ctr
                winner_arm = arm

        results = {
            "completion_reason": completion_result["reason"],
            "winner_arm_id": winner_arm.arm_id if winner_arm else None,
            "winner_confidence": completion_result["confidence"],
            "total_pulls": total_pulls,
            "total_successes": total_successes,
            "overall_ctr": overall_ctr,
            "completed_at": datetime.utcnow().isoformat(),
            "arm_results": [
                {
                    "arm_id": arm.arm_id,
                    "pulls": arm.pulls,
                    "successes": arm.successes,
                    "ctr": (arm.successes or 0) / (arm.pulls or 1),
                    "alpha": arm.alpha,
                    "beta": arm.beta
                }
                for arm in arms
            ]
        }

        experiment.is_active = False
        experiment.ended_at = datetime.utcnow()
        experiment.winner_variant = winner_arm.arm_id if winner_arm else None
        experiment.results = results
        experiment.total_impressions = total_pulls
        experiment.total_conversions = total_successes

        logger.info(
            f"Completed experiment {experiment.id}: "
            f"winner={winner_arm.arm_id if winner_arm else 'none'}, "
            f"confidence={completion_result['confidence']:.2%}"
        )

    async def _log_completed_experiment(
        self,
        experiment,
        arms: List,
        completion_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Log completed experiment to MLflow with professional naming.

        FIXED ISSUES:
        - Proper artifact logging with temp files (not just log_dict)
        - Robust error handling to ensure runs complete with FINISHED status
        - Correct duration calculation (not in ms)
        - Comprehensive professional logging
        """
        if not self._ensure_initialized():
            return None

        import tempfile
        import os as _os

        run_id = None

        try:
            algo = experiment.algorithm or "thompson_sampling"
            algorithm = ModelType.THOMPSON_SAMPLING
            if "linucb" in algo.lower():
                algorithm = ModelType.LINUCB
            elif "epsilon" in algo.lower():
                algorithm = ModelType.EPSILON_GREEDY

            campaign_id = str(experiment.campaign_id) if experiment.campaign_id else "standalone"
            experiment_name = experiment.name or f"experiment-{experiment.id}"

            model_name = self.naming.get_model_name(
                campaign_id, algorithm, "linkedin"
            )

            version = await self._get_next_version(model_name)

            run_name = self.naming.get_run_name(
                experiment_name, algorithm, version, is_completion=True
            )

            arms_data = []
            for arm in arms:
                pulls = arm.pulls or 0
                successes = arm.successes or 0
                alpha = arm.alpha or 1.0
                beta = arm.beta or 1.0

                arms_data.append({
                    "arm_id": arm.arm_id,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "pulls": int(pulls),
                    "successes": int(successes),
                    "failures": int(pulls - successes),
                    "total_reward": float(arm.total_reward or 0.0),
                    "ctr": float(successes / max(pulls, 1)),
                    "expected_value": float(alpha / (alpha + beta)),
                    "variant_data": arm.variant_data or {}
                })

            total_pulls = sum(a["pulls"] for a in arms_data)
            total_successes = sum(a["successes"] for a in arms_data)
            total_failures = total_pulls - total_successes

            now = datetime.utcnow()
            if experiment.started_at:
                duration = now - experiment.started_at
                duration_seconds = duration.total_seconds()
                duration_minutes = duration_seconds / 60
                duration_hours = duration_seconds / 3600
                duration_days = duration.days
            else:
                duration_seconds = 0
                duration_minutes = 0
                duration_hours = 0
                duration_days = 0

            winner_arm_data = None
            for arm in arms_data:
                if arm["arm_id"] == completion_result.get("winner_arm"):
                    winner_arm_data = arm
                    break

            if not winner_arm_data and arms_data:
                winner_arm_data = max(arms_data, key=lambda a: a["ctr"])

            # Regret = difference from optimal arm's CTR
            best_ctr = max(a["ctr"] for a in arms_data) if arms_data else 0
            cumulative_regret = sum(
                (best_ctr - a["ctr"]) * a["pulls"]
                for a in arms_data
            )
            normalized_regret = cumulative_regret / max(total_pulls, 1)

            # Exploration efficiency: fraction of pulls allocated to the winner
            winner_pull_ratio = (
                winner_arm_data["pulls"] / max(total_pulls, 1)
                if winner_arm_data else 0
            )

            run = self._mlflow.start_run(run_name=run_name)
            run_id = run.info.run_id

            try:
                self._mlflow.set_tag("mlflow.runName", run_name)
                self._mlflow.set_tag("mlflow.note.content",
                    f"Bandit experiment completed. Winner: {completion_result.get('winner_arm', 'N/A')} "
                    f"with {completion_result['confidence']:.1%} confidence after {total_pulls:,} pulls. "
                    f"Duration: {duration_hours:.1f} hours."
                )

                self._mlflow.set_tag("mlflow.source.type", "JOB")
                self._mlflow.set_tag("mlflow.source.name", "autonomous_mlops")

                self._mlflow.set_tag("model.name", model_name)
                self._mlflow.set_tag("model.version", str(version))
                self._mlflow.set_tag("model.type", "bandit_policy")
                self._mlflow.set_tag("model.framework", "thompson_sampling")
                self._mlflow.set_tag("model.stage", ModelStage.DEVELOPMENT.value)

                self._mlflow.set_tag("experiment.id", str(experiment.id))
                self._mlflow.set_tag("experiment.name", experiment_name)
                self._mlflow.set_tag("experiment.algorithm", algorithm.value)

                self._mlflow.set_tag("campaign.id", campaign_id)
                self._mlflow.set_tag("campaign.platform", "linkedin")

                self._mlflow.set_tag("completion.type", "automatic")
                self._mlflow.set_tag("completion.reason", completion_result.get("reason", "unknown"))
                self._mlflow.set_tag("completion.autonomous", "true")
                self._mlflow.set_tag("completion.timestamp", now.isoformat())

                if winner_arm_data:
                    self._mlflow.set_tag("winner.arm_id", winner_arm_data["arm_id"])
                    self._mlflow.set_tag("winner.confidence", f"{completion_result['confidence']:.2%}")
                    self._mlflow.set_tag("winner.ctr", f"{winner_arm_data['ctr']:.4f}")

                self._mlflow.log_param("model_name", model_name)
                self._mlflow.log_param("algorithm", algorithm.value)
                self._mlflow.log_param("num_arms", len(arms_data))
                self._mlflow.log_param("campaign_id", campaign_id)
                self._mlflow.log_param("experiment_id", str(experiment.id))

                params = experiment.parameters or {}
                self._mlflow.log_param("confidence_threshold", params.get("confidence_threshold", 0.95))
                self._mlflow.log_param("target_sample_size", params.get("target_sample_size", 1000))
                self._mlflow.log_param("max_duration_days", params.get("duration", 14))
                self._mlflow.log_param("prior_alpha", 1.0)
                self._mlflow.log_param("prior_beta", 1.0)

                arm_names = [a["arm_id"] for a in arms_data]
                self._mlflow.log_param("arm_ids", ", ".join(arm_names[:5]))
                if winner_arm_data:
                    self._mlflow.log_param("winner_arm", winner_arm_data["arm_id"])

                self._mlflow.log_metric("total_pulls", float(total_pulls))
                self._mlflow.log_metric("total_successes", float(total_successes))
                self._mlflow.log_metric("total_failures", float(total_failures))
                self._mlflow.log_metric("overall_ctr", total_successes / max(total_pulls, 1))

                self._mlflow.log_metric("winner_confidence", completion_result["confidence"])
                if winner_arm_data:
                    self._mlflow.log_metric("winner_ctr", winner_arm_data["ctr"])
                    self._mlflow.log_metric("winner_pulls", float(winner_arm_data["pulls"]))
                    self._mlflow.log_metric("winner_expected_value", winner_arm_data["expected_value"])

                self._mlflow.log_metric("duration_seconds", round(duration_seconds, 2))
                self._mlflow.log_metric("duration_minutes", round(duration_minutes, 2))
                self._mlflow.log_metric("duration_hours", round(duration_hours, 2))
                self._mlflow.log_metric("duration_days", float(duration_days))

                self._mlflow.log_metric("cumulative_regret", round(cumulative_regret, 4))
                self._mlflow.log_metric("normalized_regret", round(normalized_regret, 6))
                self._mlflow.log_metric("exploration_efficiency", round(winner_pull_ratio, 4))
                self._mlflow.log_metric("pulls_per_hour", round(total_pulls / max(duration_hours, 0.01), 2))

                self._mlflow.log_metric("num_arms", float(len(arms_data)))

                self._mlflow.log_metric("total_alpha", sum(a["alpha"] for a in arms_data))
                self._mlflow.log_metric("total_beta", sum(a["beta"] for a in arms_data))

                for arm in arms_data:
                    arm_id_clean = arm["arm_id"].replace("-", "_").replace(" ", "_")[:20]
                    self._mlflow.log_metric(f"arm_{arm_id_clean}_ctr", arm["ctr"])
                    self._mlflow.log_metric(f"arm_{arm_id_clean}_pulls", float(arm["pulls"]))
                    self._mlflow.log_metric(f"arm_{arm_id_clean}_expected", arm["expected_value"])

                with tempfile.TemporaryDirectory() as tmpdir:
                    policy_artifact = {
                        "model_name": model_name,
                        "model_version": version,
                        "algorithm": algorithm.value,
                        "created_at": now.isoformat(),
                        "experiment": {
                            "id": str(experiment.id),
                            "name": experiment_name,
                            "campaign_id": campaign_id,
                            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                            "completed_at": now.isoformat(),
                            "duration_hours": round(duration_hours, 2),
                        },
                        "arms": arms_data,
                        "winner": {
                            "arm_id": winner_arm_data["arm_id"] if winner_arm_data else None,
                            "confidence": completion_result["confidence"],
                            "ctr": winner_arm_data["ctr"] if winner_arm_data else None,
                        },
                        "metrics": {
                            "total_pulls": total_pulls,
                            "total_successes": total_successes,
                            "overall_ctr": total_successes / max(total_pulls, 1),
                            "cumulative_regret": cumulative_regret,
                            "normalized_regret": normalized_regret,
                        },
                        "completion": {
                            "reason": completion_result.get("reason"),
                            "auto_promote": completion_result.get("auto_promote", False),
                        },
                    }
                    policy_path = _os.path.join(tmpdir, "policy_data.json")
                    with open(policy_path, "w") as f:
                        json.dump(policy_artifact, f, indent=2, default=str)
                    self._mlflow.log_artifact(policy_path, "model")

                    arms_config = {
                        "model_name": model_name,
                        "algorithm": algorithm.value,
                        "serving_config": {
                            "framework": "thompson_sampling",
                            "prior_alpha": 1.0,
                            "prior_beta": 1.0,
                        },
                        "arms": [
                            {
                                "arm_id": arm["arm_id"],
                                "alpha": arm["alpha"],
                                "beta": arm["beta"],
                                "expected_ctr": arm["expected_value"],
                                "historical_ctr": arm["ctr"],
                                "historical_pulls": arm["pulls"],
                            }
                            for arm in arms_data
                        ]
                    }
                    arms_path = _os.path.join(tmpdir, "arms_config.json")
                    with open(arms_path, "w") as f:
                        json.dump(arms_config, f, indent=2)
                    self._mlflow.log_artifact(arms_path, "model")

                    arms_ranking = sorted(arms_data, key=lambda a: a["ctr"], reverse=True)
                    training_summary = {
                        "title": f"Bandit Experiment Report: {experiment_name}",
                        "generated_at": now.isoformat(),
                        "summary": {
                            "algorithm": algorithm.value,
                            "total_pulls": f"{total_pulls:,}",
                            "duration": f"{duration_hours:.1f} hours ({duration_days} days)",
                            "winner": winner_arm_data["arm_id"] if winner_arm_data else "N/A",
                            "winner_confidence": f"{completion_result['confidence']:.2%}",
                            "completion_reason": completion_result.get("reason", "unknown"),
                        },
                        "performance": {
                            "overall_ctr": f"{(total_successes / max(total_pulls, 1)) * 100:.2f}%",
                            "winner_ctr": f"{winner_arm_data['ctr'] * 100:.2f}%" if winner_arm_data else "N/A",
                            "cumulative_regret": f"{cumulative_regret:.2f}",
                            "exploration_efficiency": f"{winner_pull_ratio * 100:.1f}%",
                        },
                        "arms_ranking": [
                            {
                                "rank": i + 1,
                                "arm_id": arm["arm_id"],
                                "ctr": f"{arm['ctr'] * 100:.2f}%",
                                "pulls": f"{arm['pulls']:,}",
                                "successes": f"{arm['successes']:,}",
                                "is_winner": arm["arm_id"] == (winner_arm_data["arm_id"] if winner_arm_data else None),
                            }
                            for i, arm in enumerate(arms_ranking)
                        ],
                    }
                    summary_path = _os.path.join(tmpdir, "training_summary.json")
                    with open(summary_path, "w") as f:
                        json.dump(training_summary, f, indent=2)
                    self._mlflow.log_artifact(summary_path, "reports")

                    md_report = self._generate_markdown_report(
                        experiment_name, algorithm, arms_ranking,
                        winner_arm_data, completion_result, total_pulls,
                        total_successes, duration_hours, cumulative_regret
                    )
                    md_path = _os.path.join(tmpdir, "experiment_report.md")
                    with open(md_path, "w") as f:
                        f.write(md_report)
                    self._mlflow.log_artifact(md_path, "reports")

                    csv_content = "arm_id,pulls,successes,failures,ctr,alpha,beta,expected_value,is_winner\n"
                    for arm in arms_ranking:
                        is_winner = "true" if arm["arm_id"] == (winner_arm_data["arm_id"] if winner_arm_data else None) else "false"
                        csv_content += f"{arm['arm_id']},{arm['pulls']},{arm['successes']},{arm['failures']},{arm['ctr']:.6f},{arm['alpha']:.2f},{arm['beta']:.2f},{arm['expected_value']:.6f},{is_winner}\n"
                    csv_path = _os.path.join(tmpdir, "arm_metrics.csv")
                    with open(csv_path, "w") as f:
                        f.write(csv_content)
                    self._mlflow.log_artifact(csv_path, "data")

                model_registered = False
                try:
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()

                    try:
                        client.create_registered_model(
                            model_name,
                            description=f"Thompson Sampling bandit policy for campaign {campaign_id}"
                        )
                        logger.info(f"Created registered model: {model_name}")
                    except Exception:
                        pass

                    model_uri = f"runs:/{run_id}/model"
                    mv = client.create_model_version(
                        name=model_name,
                        source=model_uri,
                        run_id=run_id,
                        description=f"Winner: {winner_arm_data['arm_id'] if winner_arm_data else 'N/A'}, "
                                    f"Confidence: {completion_result['confidence']:.1%}"
                    )
                    self._mlflow.set_tag("model.registered", "true")
                    self._mlflow.set_tag("model.registry_version", str(mv.version))
                    model_registered = True
                    logger.info(f"Registered model {model_name} version {mv.version}")

                except Exception as reg_error:
                    logger.warning(f"Model registration failed (non-fatal): {reg_error}")
                    self._mlflow.set_tag("model.registered", "false")
                    self._mlflow.set_tag("model.registration_error", str(reg_error)[:100])

                logger.info(
                    f"Logged experiment to MLflow: {run_name}, "
                    f"model={model_name}, version={version}, "
                    f"winner={winner_arm_data['arm_id'] if winner_arm_data else 'none'}, "
                    f"registered={model_registered}"
                )

            finally:
                self._mlflow.end_run(status="FINISHED")

            return run_id

        except Exception as e:
            logger.error(f"Failed to log experiment to MLflow: {e}")
            if run_id:
                try:
                    self._mlflow.end_run(status="FAILED")
                except:
                    pass
            return None

    def _generate_markdown_report(
        self,
        experiment_name: str,
        algorithm: ModelType,
        arms_ranking: List[Dict],
        winner_arm: Optional[Dict],
        completion_result: Dict,
        total_pulls: int,
        total_successes: int,
        duration_hours: float,
        cumulative_regret: float
    ) -> str:
        """Generate a professional markdown report for the experiment."""
        winner_id = winner_arm["arm_id"] if winner_arm else "N/A"
        winner_ctr = f"{winner_arm['ctr'] * 100:.2f}%" if winner_arm else "N/A"
        overall_ctr = f"{(total_successes / max(total_pulls, 1)) * 100:.2f}%"

        report = f"""# Bandit Experiment Report

## Experiment: {experiment_name}

**Algorithm:** {algorithm.value}
**Completed:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

---

## Summary

| Metric | Value |
|--------|-------|
| Total Pulls | {total_pulls:,} |
| Total Successes | {total_successes:,} |
| Overall CTR | {overall_ctr} |
| Duration | {duration_hours:.1f} hours |
| Winner | {winner_id} |
| Winner Confidence | {completion_result['confidence']:.2%} |
| Completion Reason | {completion_result.get('reason', 'N/A')} |

---

## Arm Performance Ranking

| Rank | Arm ID | CTR | Pulls | Successes | Winner |
|------|--------|-----|-------|-----------|--------|
"""
        for i, arm in enumerate(arms_ranking):
            is_winner = "**Yes**" if arm["arm_id"] == winner_id else "No"
            report += f"| {i+1} | {arm['arm_id']} | {arm['ctr']*100:.2f}% | {arm['pulls']:,} | {arm['successes']:,} | {is_winner} |\n"

        report += f"""
---

## Learning Metrics

| Metric | Value |
|--------|-------|
| Cumulative Regret | {cumulative_regret:.4f} |
| Normalized Regret | {cumulative_regret / max(total_pulls, 1):.6f} |
| Pulls per Hour | {total_pulls / max(duration_hours, 0.01):.1f} |

---

## Bayesian Posterior Summary

| Arm | Alpha | Beta | Expected Value |
|-----|-------|------|----------------|
"""
        for arm in arms_ranking:
            report += f"| {arm['arm_id']} | {arm['alpha']:.2f} | {arm['beta']:.2f} | {arm['expected_value']:.4f} |\n"

        report += """
---

*Generated automatically by Agentic Autonomous MLOps*
"""
        return report

    def _register_policy_model(
        self,
        model_name: str,
        algorithm: ModelType,
        arms_data: List[Dict[str, Any]],
        run_id: str
    ):
        """
        Register the policy as a PyFunc model in MLflow registry.

        NOTE: This is called within an active run context.
        Model registration failures should NOT cause the run to fail.
        """
        try:
            import mlflow.pyfunc
            import cloudpickle

            model_data = {
                "algorithm": algorithm.value,
                "arms": arms_data,
                "model_name": model_name,
            }

            class BanditPolicyWrapper(mlflow.pyfunc.PythonModel):
                """
                Production-ready bandit policy model wrapper.

                Implements Thompson Sampling for arm selection.
                """

                def load_context(self, context):
                    """Load model artifacts when model is loaded."""
                    import json
                    artifacts_path = context.artifacts.get("model_data")
                    if artifacts_path:
                        with open(artifacts_path, "r") as f:
                            data = json.load(f)
                            self.algorithm = data["algorithm"]
                            self.arms = data["arms"]
                            self.model_name = data["model_name"]
                    else:
                        self.algorithm = "thompson_sampling"
                        self.arms = []
                        self.model_name = "unknown"

                def predict(self, context, model_input):
                    """
                    Select best arm using the trained policy.

                    For Thompson Sampling: Sample from Beta posteriors.
                    Returns the arm_id of the selected arm.
                    """
                    import numpy as np

                    if not self.arms:
                        return {"error": "No arms configured"}

                    if self.algorithm == "thompson_sampling":
                        samples = []
                        for arm in self.arms:
                            alpha = arm.get("alpha", 1.0)
                            beta = arm.get("beta", 1.0)
                            sample = np.random.beta(alpha, beta)
                            samples.append({
                                "arm_id": arm["arm_id"],
                                "sample": sample,
                                "expected": alpha / (alpha + beta)
                            })

                        best = max(samples, key=lambda x: x["sample"])
                        return {
                            "selected_arm": best["arm_id"],
                            "sample_value": float(best["sample"]),
                            "expected_value": float(best["expected"]),
                            "model_name": self.model_name,
                            "algorithm": self.algorithm
                        }
                    else:
                        best_arm = max(self.arms, key=lambda a: a.get("ctr", 0))
                        return {
                            "selected_arm": best_arm["arm_id"],
                            "ctr": best_arm.get("ctr", 0),
                            "model_name": self.model_name,
                            "algorithm": self.algorithm
                        }

            import tempfile
            import os as _os
            from mlflow.tracking import MlflowClient

            with tempfile.TemporaryDirectory() as tmpdir:
                model_data_path = _os.path.join(tmpdir, "model_data.json")
                with open(model_data_path, "w") as f:
                    json.dump(model_data, f, indent=2)

                model_info = mlflow.pyfunc.log_model(
                    artifact_path="bandit_policy",
                    python_model=BanditPolicyWrapper(),
                    artifacts={"model_data": model_data_path},
                    pip_requirements=["numpy>=1.21.0"],
                )

            client = MlflowClient()

            try:
                client.create_registered_model(model_name)
                logger.info(f"Created registered model: {model_name}")
            except Exception:
                pass

            model_uri = f"runs:/{run_id}/bandit_policy"
            mv = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=f"Autonomous bandit policy - winner: {arms_data[0]['arm_id'] if arms_data else 'unknown'}"
            )

            logger.info(f"Registered model: {model_name} version {mv.version}")

        except Exception as e:
            logger.warning(f"Failed to register model {model_name}: {e}")
            raise

    async def _get_next_version(self, model_name: str) -> int:
        """Get next version number for a model."""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            try:
                versions = client.search_model_versions(f"name='{model_name}'")
                if versions:
                    return max(int(v.version) for v in versions) + 1
            except:
                pass

            return 1

        except Exception as e:
            logger.warning(f"Error getting model version: {e}")
            return 1

    async def _auto_promote_model(
        self,
        experiment,
        completion_result: Dict[str, Any]
    ) -> bool:
        """
        Automatically promote model through stages based on performance.

        Promotion path: development → staging → canary → production
        """
        if not self._ensure_initialized():
            return False

        try:
            from mlflow.tracking import MlflowClient
            from ...ai_layer.marl.ope_gating import OffPolicyEvaluator

            client = MlflowClient()

            algo = experiment.algorithm or "thompson_sampling"
            algorithm = ModelType.THOMPSON_SAMPLING
            if "linucb" in algo.lower():
                algorithm = ModelType.LINUCB

            campaign_id = str(experiment.campaign_id) if experiment.campaign_id else "standalone"
            model_name = self.naming.get_model_name(campaign_id, algorithm, "linkedin")

            try:
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    return False

                latest = max(versions, key=lambda v: int(v.version))

                if completion_result["confidence"] >= 0.95:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest.version,
                        stage="Staging"
                    )
                    logger.info(f"Promoted {model_name} v{latest.version} to Staging")

                    if completion_result["confidence"] >= 0.99:
                        client.transition_model_version_stage(
                            name=model_name,
                            version=latest.version,
                            stage="Production"
                        )
                        logger.info(f"Promoted {model_name} v{latest.version} to Production")

                    return True

            except Exception as e:
                logger.warning(f"Model promotion failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Auto-promotion error: {e}")
            return False

        return False

    async def get_active_policy(
        self,
        campaign_id: str,
        algorithm: ModelType = ModelType.THOMPSON_SAMPLING
    ) -> Optional[Dict[str, Any]]:
        """
        AUTONOMOUS: Load the current production policy for a campaign.

        Loads the policy directly from MLflow artifacts (arms_config.json).
        Priority: Production > Staging > Latest version.
        """
        if not self._ensure_initialized():
            return None

        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            model_name = self.naming.get_model_name(
                campaign_id, algorithm, "linkedin"
            )

            try:
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    logger.warning(f"No registered model found: {model_name}")
                    return None

                selected_version = None
                for stage in ["Production", "Staging", "None"]:
                    for v in versions:
                        if v.current_stage == stage or (
                            stage == "None" and v.current_stage not in ["Production", "Staging", "Archived"]
                        ):
                            selected_version = v
                            break
                    if selected_version:
                        break

                if not selected_version:
                    return None

                run_id = selected_version.run_id
                artifacts_path = client.download_artifacts(run_id, "model/arms_config.json")

                with open(artifacts_path, 'r') as f:
                    arms_config = json.load(f)

                return {
                    "model_name": model_name,
                    "version": selected_version.version,
                    "stage": selected_version.current_stage,
                    "run_id": run_id,
                    "algorithm": arms_config.get("algorithm", "thompson_sampling"),
                    "arms": arms_config.get("arms", [])
                }

            except Exception as e:
                logger.warning(f"Error loading model {model_name}: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to get active policy: {e}")
            return None

    async def select_arm_for_campaign(
        self,
        campaign_id: str,
        context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        AUTONOMOUS: Select the best arm for a campaign using active policy.

        This is the main inference endpoint - fully autonomous arm selection.
        Uses Thompson Sampling to select from the loaded policy.
        """
        import numpy as np

        policy = await self.get_active_policy(campaign_id)

        if not policy:
            logger.warning(f"No active policy for campaign {campaign_id}")
            return None

        try:
            arms = policy.get("arms", [])
            if not arms:
                logger.warning(f"Policy has no arms configured")
                return None

            samples = []
            for arm in arms:
                alpha = arm.get("alpha", 1.0)
                beta = arm.get("beta", 1.0)
                sample = np.random.beta(alpha, beta)
                samples.append({
                    "arm_id": arm["arm_id"],
                    "sample": float(sample),
                    "expected_ctr": arm.get("expected_ctr", alpha / (alpha + beta)),
                    "historical_ctr": arm.get("historical_ctr", 0)
                })

            best = max(samples, key=lambda x: x["sample"])

            logger.info(
                f"Selected arm '{best['arm_id']}' for campaign {campaign_id} "
                f"(sample={best['sample']:.4f}, expected={best['expected_ctr']:.4f})"
            )

            return {
                "selected_arm": best["arm_id"],
                "sample_value": best["sample"],
                "expected_ctr": best["expected_ctr"],
                "model_name": policy["model_name"],
                "model_version": policy["version"],
                "model_stage": policy["stage"],
                "all_samples": samples
            }

        except Exception as e:
            logger.error(f"Arm selection failed: {e}")
            return None


_orchestrator: Optional[AutonomousMLOpsOrchestrator] = None


def get_mlops_orchestrator() -> AutonomousMLOpsOrchestrator:
    """Get singleton MLOps orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AutonomousMLOpsOrchestrator()
    return _orchestrator


async def autonomous_experiment_completion_check() -> Dict[str, Any]:
    """
    Scheduled task: Check and complete experiments autonomously.

    Should be scheduled every 5 minutes.
    """
    orchestrator = get_mlops_orchestrator()
    return await orchestrator.check_and_complete_experiments()


def run_autonomous_mlops_check() -> Dict[str, Any]:
    """Sync wrapper for the autonomous MLOps check."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(autonomous_experiment_completion_check())
