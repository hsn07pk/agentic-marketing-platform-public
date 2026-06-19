"""
MLflow Integration for Agentic AI Marketing Platform

Complete MLflow Model Registry integration for:
- Experiment tracking
- Model logging and versioning
- Model registry management
- Bandit policy storage and retrieval

Per Research Plan Section 8.1 - MLOps Infrastructure
"""
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import pickle

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "agentic-bandits")


class MLflowModelRegistry:
    """
    MLflow Model Registry integration for bandit policies.
    Per Research Plan Section 8.1 - MLOps Infrastructure.
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or MLFLOW_EXPERIMENT_NAME
        self._mlflow = None
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        
        try:
            import mlflow
            self._mlflow = mlflow
            
            mlflow.set_tracking_uri(self.tracking_uri)
            
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    tags={"project": "agentic", "type": "bandit-policies"}
                )
            mlflow.set_experiment(self.experiment_name)
            
            self._initialized = True
            logger.info(f"MLflow initialized: {self.tracking_uri}, experiment: {self.experiment_name}")
            return True
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            return False
    
    def log_bandit_policy(
        self,
        policy_name: str,
        policy_type: str,
        arms_data: List[Dict[str, Any]],
        metrics: Dict[str, float],
        parameters: Dict[str, Any] = None,
        tags: Dict[str, str] = None
    ) -> Optional[str]:
        if not self._ensure_initialized():
            logger.warning("MLflow not available, skipping policy logging")
            return None
        
        try:
            with self._mlflow.start_run() as run:
                run_id = run.info.run_id
                
                self._mlflow.log_param("policy_name", policy_name)
                self._mlflow.log_param("policy_type", policy_type)
                self._mlflow.log_param("num_arms", len(arms_data))
                
                if parameters:
                    for key, value in parameters.items():
                        self._mlflow.log_param(f"param_{key}", value)
                
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self._mlflow.log_metric(metric_name, metric_value)
                
                arms_artifact = {
                    "policy_name": policy_name,
                    "policy_type": policy_type,
                    "arms": arms_data,
                    "logged_at": datetime.utcnow().isoformat()
                }
                self._mlflow.log_dict(arms_artifact, "arms_data.json")
                
                self._log_policy_model(policy_name, policy_type, arms_data, parameters)
                
                self._mlflow.set_tag("policy_name", policy_name)
                self._mlflow.set_tag("policy_type", policy_type)
                if tags:
                    for key, value in tags.items():
                        self._mlflow.set_tag(key, value)
                
                logger.info(f"Logged bandit policy '{policy_name}' to MLflow, run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to log bandit policy: {e}")
            return None
    
    def _log_policy_model(
        self,
        policy_name: str,
        policy_type: str,
        arms_data: List[Dict[str, Any]],
        parameters: Dict[str, Any] = None
    ):
        try:
            import mlflow.pyfunc
            
            # Create a simple wrapper class for the bandit policy
            class BanditPolicyModel(mlflow.pyfunc.PythonModel):
                def __init__(self, policy_type, arms, parameters):
                    self.policy_type = policy_type
                    self.arms = arms
                    self.parameters = parameters or {}
                
                def predict(self, context, model_input):
                    import numpy as np
                    
                    if self.policy_type == "thompson_sampling":
                        samples = []
                        for arm in self.arms:
                            alpha = arm.get("alpha", 1.0)
                            beta = arm.get("beta", 1.0)
                            sample = np.random.beta(alpha, beta)
                            samples.append(sample)
                        best_arm_idx = np.argmax(samples)
                        return self.arms[best_arm_idx]
                    else:
                        return self.arms[0] if self.arms else None
            
            mlflow.pyfunc.log_model(
                artifact_path="bandit_policy",
                python_model=BanditPolicyModel(policy_type, arms_data, parameters),
                registered_model_name=policy_name,
                pip_requirements=["numpy>=1.21.0"]
            )
            
        except Exception as e:
            logger.warning(f"Failed to log PyFunc model: {e}")
    
    def update_bandit_metrics(
        self,
        policy_name: str,
        experiment_id: str,
        arm_updates: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ) -> Optional[str]:
        """Log incremental updates to a bandit policy after deployment."""
        if not self._ensure_initialized():
            return None
        
        try:
            with self._mlflow.start_run() as run:
                run_id = run.info.run_id
                
                self._mlflow.set_tag("run_type", "update")
                self._mlflow.set_tag("policy_name", policy_name)
                self._mlflow.set_tag("experiment_id", experiment_id)
                
                for name, value in performance_metrics.items():
                    if isinstance(value, (int, float)):
                        self._mlflow.log_metric(name, value)
                
                total_pulls = sum(arm.get("pulls", 0) for arm in arm_updates)
                total_successes = sum(arm.get("successes", 0) for arm in arm_updates)
                
                self._mlflow.log_metric("total_pulls", total_pulls)
                self._mlflow.log_metric("total_successes", total_successes)
                if total_pulls > 0:
                    self._mlflow.log_metric("overall_ctr", total_successes / total_pulls)
                
                self._mlflow.log_dict({
                    "arms": arm_updates,
                    "updated_at": datetime.utcnow().isoformat()
                }, "arm_updates.json")
                
                logger.info(f"Logged bandit update for '{policy_name}', run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logger.error(f"Failed to log bandit update: {e}")
            return None
    
    def load_policy(self, policy_name: str, version: str = "latest") -> Optional[Any]:
        if not self._ensure_initialized():
            return None
        
        try:
            if version in ["latest", "staging", "production"]:
                model_uri = f"models:/{policy_name}/{version}"
            else:
                model_uri = f"models:/{policy_name}/{version}"
            
            model = self._mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded policy '{policy_name}' version '{version}' from MLflow")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load policy from MLflow: {e}")
            return None
    
    def list_policy_versions(self, policy_name: str) -> List[Dict[str, Any]]:
        if not self._ensure_initialized():
            return []
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(self.tracking_uri)
            
            versions = client.search_model_versions(f"name='{policy_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "created_at": v.creation_timestamp
                }
                for v in versions
            ]
            
        except Exception as e:
            logger.error(f"Failed to list policy versions: {e}")
            return []
    
    def promote_policy(
        self,
        policy_name: str,
        version: str,
        stage: str = "Production"
    ) -> bool:
        if not self._ensure_initialized():
            return False
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(self.tracking_uri)
            
            client.transition_model_version_stage(
                name=policy_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Promoted '{policy_name}' v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote policy: {e}")
            return False
    
    def get_latest_runs(self, max_results: int = 10) -> List[Dict[str, Any]]:
        if not self._ensure_initialized():
            return []

        try:
            runs = self._mlflow.search_runs(
                experiment_names=[self.experiment_name],
                max_results=max_results,
                order_by=["start_time DESC"]
            )

            if runs.empty:
                return []

            # Convert to records and clean NaN/Inf values (not JSON compliant)
            import math
            records = runs.to_dict('records')
            def clean_value(v):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v

            cleaned_records = []
            for record in records:
                cleaned = {k: clean_value(v) for k, v in record.items()}
                cleaned_records.append(cleaned)

            return cleaned_records

        except Exception as e:
            logger.error(f"Failed to get runs: {e}")
            return []


_registry: Optional[MLflowModelRegistry] = None


def get_mlflow_registry() -> MLflowModelRegistry:
    global _registry
    if _registry is None:
        _registry = MLflowModelRegistry()
    return _registry


async def log_experiment_completion(
    experiment_id: str,
    experiment_name: str,
    algorithm: str,
    arms: List[Dict[str, Any]],
    winner_arm: str,
    metrics: Dict[str, float],
    campaign_id: str = None
) -> Optional[str]:
    """Log experiment completion to MLflow. Called when a bandit experiment completes."""
    registry = get_mlflow_registry()
    
    policy_name = f"experiment_{experiment_name.replace(' ', '_').lower()}"
    
    return registry.log_bandit_policy(
        policy_name=policy_name,
        policy_type=algorithm,
        arms_data=arms,
        metrics=metrics,
        parameters={"winner_arm": winner_arm},
        tags={
            "experiment_id": experiment_id,
            "campaign_id": campaign_id or "unknown",
            "winner": winner_arm
        }
    )
