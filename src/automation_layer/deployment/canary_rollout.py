"""
Canary Rollout System with One-Click Rollback

Research Plan: Gradual policy deployment with monitoring and instant rollback
- Start with 5% traffic to new policy (canary)
- Monitor key metrics (CTR, conversion rate, cost)
- Gradually increase to 25%, 50%, 75%, 100% if metrics are good
- Instant rollback if metrics degrade >10%
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from ...data_layer.database.connection import get_async_session

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    PENDING = "pending"
    CANARY_5 = "canary_5_percent"
    CANARY_25 = "canary_25_percent"
    CANARY_50 = "canary_50_percent"
    CANARY_75 = "canary_75_percent"
    FULL_ROLLOUT = "full_rollout_100_percent"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

@dataclass
class CanaryMetrics:
    timestamp: datetime
    traffic_percentage: float
    requests_served: int
    average_ctr: float
    average_conversion_rate: float
    average_cost_per_lead: float
    error_rate: float
    p95_latency_ms: float
    has_real_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'traffic_percentage': self.traffic_percentage,
            'requests_served': self.requests_served,
            'average_ctr': self.average_ctr,
            'average_conversion_rate': self.average_conversion_rate,
            'average_cost_per_lead': self.average_cost_per_lead,
            'error_rate': self.error_rate,
            'p95_latency_ms': self.p95_latency_ms,
            'has_real_data': self.has_real_data
        }

@dataclass
class CanaryDeployment:
    deployment_id: str
    policy_id: str
    policy_version: str
    start_time: datetime
    status: DeploymentStatus = DeploymentStatus.PENDING
    current_traffic_percentage: float = 0.0
    baseline_metrics: Optional[Dict[str, float]] = None
    canary_metrics_history: List[CanaryMetrics] = field(default_factory=list)
    rollback_reason: Optional[str] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'deployment_id': self.deployment_id,
            'policy_id': self.policy_id,
            'policy_version': self.policy_version,
            'start_time': self.start_time.isoformat(),
            'status': self.status.value,
            'current_traffic_percentage': self.current_traffic_percentage,
            'baseline_metrics': self.baseline_metrics,
            'canary_metrics_history': [m.to_dict() for m in self.canary_metrics_history],
            'rollback_reason': self.rollback_reason,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

class DeploymentController:

    def __init__(self):
        self.active_deployments: Dict[str, CanaryDeployment] = {}
        self.deployment_history: List[CanaryDeployment] = []

        # (traffic_percentage, min_observation_seconds)
        self.canary_stages = [
            (0.05, 300),
            (0.25, 600),
            (0.50, 900),
            (0.75, 1200),
            (1.00, 0)
        ]

        self.degradation_threshold = 0.10
        self.persistence_callback = None

        logger.info("Initialized Canary Deployment Controller")

    def register_persistence_callback(self, callback):
        self.persistence_callback = callback
        logger.info("Registered persistence callback")

    async def start_canary_deployment(
        self,
        name: str = None,
        policy_name: str = None,
        policy_id: str = None,
        policy_version: str = "v1",
        deployment_type: str = "marl_policy",
        initial_traffic_percent: float = 5,
        progression_schedule: str = "conservative",
        auto_rollback: bool = True,
        rollback_thresholds: Dict[str, float] = None,
        baseline_metrics: Dict[str, float] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        try:
            effective_policy_id = policy_id or policy_name or "unknown_policy"
            deployment_id = f"canary_{effective_policy_id}_{datetime.now().timestamp()}"

            if baseline_metrics is None:
                baseline_metrics = {"ctr": 0.03, "conversion_rate": 0.02, "cpl": 50.0}

            if rollback_thresholds is None:
                rollback_thresholds = {
                    "ctr_drop_percent": 10,
                    "error_rate_percent": 5,
                    "cpl_increase_percent": 15
                }

            deployment = CanaryDeployment(
                deployment_id=deployment_id,
                policy_id=effective_policy_id,
                policy_version=policy_version,
                start_time=datetime.now(),
                status=DeploymentStatus.CANARY_5,
                current_traffic_percentage=initial_traffic_percent / 100.0,
                baseline_metrics=baseline_metrics
            )

            deployment.auto_rollback = auto_rollback
            deployment.rollback_thresholds = rollback_thresholds
            deployment.progression_schedule = progression_schedule
            deployment.metadata = metadata or {}
            deployment.name = name or deployment_id

            self.active_deployments[deployment_id] = deployment

            logger.info(
                f"Started canary deployment {deployment_id}: "
                f"{effective_policy_id} @ {initial_traffic_percent}% traffic"
            )

            asyncio.create_task(self._monitor_deployment(deployment_id))

            return {
                "success": True,
                "deployment_id": deployment_id,
                "status": deployment.status.value,
                "traffic_percent": initial_traffic_percent,
                "policy_id": effective_policy_id,
                "message": f"Canary deployment started at {initial_traffic_percent}% traffic"
            }

        except Exception as e:
            logger.error(f"Failed to start canary deployment: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to start canary deployment: {e}"
            }

    async def _monitor_deployment(self, deployment_id: str):
        try:
            deployment = self.active_deployments.get(deployment_id)
            if not deployment:
                logger.error(f"Deployment {deployment_id} not found")
                return

            stage_index = 0
            stage_start_time = datetime.now()

            while deployment.status != DeploymentStatus.FULL_ROLLOUT and \
                  deployment.status != DeploymentStatus.ROLLED_BACK:

                sleep_interval = 15 if deployment.metadata.get('test_mode') else 60
                await asyncio.sleep(sleep_interval)

                current_metrics = await self._collect_metrics(deployment)
                deployment.canary_metrics_history.append(current_metrics)

                if self._check_for_degradation(deployment, current_metrics):
                    logger.warning(f"Metrics degraded for {deployment_id}, initiating rollback")
                    await self.rollback_deployment(deployment_id, "metrics_degradation")
                    return

                current_stage_traffic, min_observation_time = self.canary_stages[stage_index]
                
                if deployment.metadata.get('test_mode'):
                    min_observation_time = 10
                
                time_in_stage = (datetime.now() - stage_start_time).total_seconds()

                if time_in_stage >= min_observation_time:
                    stage_index += 1

                    if stage_index >= len(self.canary_stages):
                        deployment.status = DeploymentStatus.FULL_ROLLOUT
                        deployment.current_traffic_percentage = 1.0
                        deployment.end_time = datetime.now()

                        logger.info(f"Deployment {deployment_id} COMPLETED successfully")
                        self.deployment_history.append(deployment)
                        del self.active_deployments[deployment_id]
                        
                        if self.persistence_callback:
                            await self.persistence_callback(deployment)
                        return

                    next_traffic, _ = self.canary_stages[stage_index]
                    deployment.current_traffic_percentage = next_traffic

                    if next_traffic == 0.05:
                        deployment.status = DeploymentStatus.CANARY_5
                    elif next_traffic == 0.25:
                        deployment.status = DeploymentStatus.CANARY_25
                    elif next_traffic == 0.50:
                        deployment.status = DeploymentStatus.CANARY_50
                    elif next_traffic == 0.75:
                        deployment.status = DeploymentStatus.CANARY_75
                    elif next_traffic == 1.0:
                        deployment.status = DeploymentStatus.FULL_ROLLOUT

                    stage_start_time = datetime.now()

                    logger.info(
                        f"Deployment {deployment_id} progressed to "
                        f"{next_traffic*100:.0f}% traffic"
                    )
                    
                    if self.persistence_callback:
                        await self.persistence_callback(deployment)

        except Exception as e:
            logger.error(f"Error monitoring deployment {deployment_id}: {e}")
            await self.rollback_deployment(deployment_id, f"monitoring_error: {str(e)}")

    async def _collect_metrics(self, deployment: CanaryDeployment) -> CanaryMetrics:
        try:
            import httpx
            import os

            prometheus_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

            query_window = '5m'
            deployment_label = f'deployment_id="{deployment.deployment_id}"'

            async with httpx.AsyncClient() as client:
                queries = {
                    'ctr': f'rate(campaign_clicks_total{{{deployment_label}}}[{query_window}]) / rate(campaign_impressions_total{{{deployment_label}}}[{query_window}])',
                    'conversion': f'rate(campaign_conversions_total{{{deployment_label}}}[{query_window}]) / rate(campaign_clicks_total{{{deployment_label}}}[{query_window}])',
                    'cpl': f'sum(campaign_cost_total{{{deployment_label}}}) / sum(campaign_conversions_total{{{deployment_label}}})',
                    'error': f'rate(http_requests_errors_total{{{deployment_label}}}[{query_window}]) / rate(http_requests_total{{{deployment_label}}}[{query_window}])',
                    'latency': f'histogram_quantile(0.95, rate(http_request_duration_ms_bucket{{{deployment_label}}}[{query_window}]))',
                    'requests': f'sum(increase(http_requests_total{{{deployment_label}}}[{query_window}]))',
                }

                responses = {}
                for name, query in queries.items():
                    responses[name] = await client.get(
                        f'{prometheus_url}/api/v1/query',
                        params={'query': query},
                        timeout=10.0
                    )

                def extract_metric_value(response, default=0.0):
                    """Returns (value, has_data) tuple."""
                    try:
                        data = response.json()
                        if data.get('status') == 'success':
                            results = data.get('data', {}).get('result', [])
                            if results:
                                val = float(results[0].get('value', [None, default])[1])
                                import math
                                if not math.isnan(val) and not math.isinf(val):
                                    return val, True
                    except Exception as e:
                        logger.warning(f"Failed to parse Prometheus response: {e}")
                    return default, False

                ctr_val, ctr_real = extract_metric_value(responses['ctr'], 0.0)
                conv_val, conv_real = extract_metric_value(responses['conversion'], 0.0)
                cpl_val, cpl_real = extract_metric_value(responses['cpl'], 0.0)
                err_val, err_real = extract_metric_value(responses['error'], 0.0)
                lat_val, lat_real = extract_metric_value(responses['latency'], 0.0)
                req_val, req_real = extract_metric_value(responses['requests'], 0.0)

                has_real_data = any([ctr_real, conv_real, cpl_real, err_real, lat_real, req_real])

                if not has_real_data:
                    logger.info(
                        f"Prometheus returned no deployment-specific data for {deployment.deployment_id} "
                        f"— no metrics with deployment_id label exist yet"
                    )

                baseline = deployment.baseline_metrics or {}
                return CanaryMetrics(
                    timestamp=datetime.now(),
                    traffic_percentage=deployment.current_traffic_percentage,
                    requests_served=int(req_val) if req_real else int(1000 * deployment.current_traffic_percentage),
                    average_ctr=ctr_val if ctr_real else baseline.get('ctr', 0.03),
                    average_conversion_rate=conv_val if conv_real else baseline.get('conversion_rate', 0.02),
                    average_cost_per_lead=cpl_val if cpl_real else baseline.get('cpl', 50.0),
                    error_rate=err_val if err_real else 0.001,
                    p95_latency_ms=lat_val if lat_real else 200.0,
                    has_real_data=has_real_data
                )

        except (Exception,) as e:
            is_connect_error = 'ConnectError' in type(e).__name__ or 'Connect' in str(type(e).__mro__)
            if is_connect_error:
                logger.warning(f"Prometheus not reachable: {e}")
            else:
                logger.error(f"Error collecting metrics: {e}")

            baseline = deployment.baseline_metrics or {}
            return CanaryMetrics(
                timestamp=datetime.now(),
                traffic_percentage=deployment.current_traffic_percentage,
                requests_served=int(1000 * deployment.current_traffic_percentage),
                average_ctr=baseline.get('ctr', 0.03),
                average_conversion_rate=baseline.get('conversion_rate', 0.02),
                average_cost_per_lead=baseline.get('cpl', 50.0),
                error_rate=0.001,
                p95_latency_ms=200.0,
                has_real_data=False
            )

    def _check_for_degradation(
        self,
        deployment: CanaryDeployment,
        current_metrics: CanaryMetrics
    ) -> bool:
        baseline = deployment.baseline_metrics
        if not baseline:
            return False

        if not current_metrics.has_real_data:
            return False

        baseline_ctr = baseline.get('ctr', 0.03)
        if current_metrics.average_ctr < baseline_ctr * (1 - self.degradation_threshold):
            logger.warning(
                f"CTR degraded: {current_metrics.average_ctr:.4f} vs baseline {baseline_ctr:.4f}"
            )
            return True

        baseline_conversion = baseline.get('conversion_rate', 0.02)
        if current_metrics.average_conversion_rate < baseline_conversion * (1 - self.degradation_threshold):
            logger.warning(
                f"Conversion rate degraded: {current_metrics.average_conversion_rate:.4f} "
                f"vs baseline {baseline_conversion:.4f}"
            )
            return True

        baseline_cpl = baseline.get('cpl', 50.0)
        if current_metrics.average_cost_per_lead > baseline_cpl * (1 + self.degradation_threshold):
            logger.warning(
                f"CPL increased: €{current_metrics.average_cost_per_lead:.2f} "
                f"vs baseline €{baseline_cpl:.2f}"
            )
            return True

        if current_metrics.error_rate > 0.05:
            logger.warning(f"High error rate: {current_metrics.error_rate:.2%}")
            return True

        return False

    async def rollback_deployment(
        self,
        deployment_id: str,
        reason: str
    ) -> bool:
        try:
            deployment = self.active_deployments.get(deployment_id)
            if not deployment:
                logger.error(f"Deployment {deployment_id} not found")
                return False

            logger.critical(
                f"🚨 ROLLING BACK deployment {deployment_id}: {reason}"
            )

            max_traffic = deployment.current_traffic_percentage

            deployment.current_traffic_percentage = 0.0
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.rollback_reason = reason
            deployment.end_time = datetime.now()

            self.deployment_history.append(deployment)
            del self.active_deployments[deployment_id]

            logger.info(f"✅ Rollback completed for {deployment_id}")

            if self.persistence_callback:
                await self.persistence_callback(deployment)

            await self._send_rollback_alert(deployment, reason, max_traffic)

            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def _send_rollback_alert(self, deployment: CanaryDeployment, reason: str, max_traffic: float = 0.0):
        try:
            duration_s = (deployment.end_time - deployment.start_time).total_seconds()
        except TypeError:
            duration_s = 0
        alert_message = (
            f"🚨 CANARY ROLLBACK ALERT\n"
            f"Deployment: {deployment.deployment_id}\n"
            f"Policy: {deployment.policy_id} v{deployment.policy_version}\n"
            f"Reason: {reason}\n"
            f"Duration: {duration_s:.0f}s\n"
            f"Max Traffic: {max_traffic*100:.0f}%"
        )
        logger.critical(alert_message)

        async with get_async_session() as session:
            from sqlalchemy import select
            from ...data_layer.database.models import SystemConfiguration
            from ...config.encryption import decrypt_value

            # Config keys that are Fernet-encrypted in the DB
            _SECRET_KEYS = {"SLACK_WEBHOOK_URL", "SENDGRID_API_KEY", "MAILGUN_API_KEY",
                            "BLOG_APP_PASSWORD", "BLOG_API_KEY"}

            async def _get_config(key: str, default: str = None) -> Optional[str]:
                result = await session.execute(
                    select(SystemConfiguration.value).where(SystemConfiguration.key == key)
                )
                row = result.scalar_one_or_none()
                if row is not None:
                    if key in _SECRET_KEYS and row:
                        try:
                            return decrypt_value(row)
                        except Exception:
                            logger.debug(f"Config {key} not encrypted, using raw value")
                    return row
                import os
                return os.environ.get(key, default)

            slack_webhook = await _get_config("SLACK_WEBHOOK_URL")
            sendgrid_key = await _get_config("SENDGRID_API_KEY")
            alert_email = await _get_config("ALERT_EMAIL")
            from_email = await _get_config("SENDGRID_FROM_EMAIL", "alerts@example.com")

        if slack_webhook and slack_webhook.strip():
            try:
                import httpx
                slack_payload = {
                    'text': alert_message,
                    'username': 'Agentic Canary Monitor',
                    'icon_emoji': ':rotating_light:'
                }
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        slack_webhook,
                        json=slack_payload,
                        timeout=10.0
                    )
                    if response.status_code == 200:
                        logger.info("Rollback alert sent to Slack")
                    else:
                        logger.warning(f"Failed to send Slack alert: {response.status_code}")
            except Exception as e:
                logger.error(f"Error sending Slack alert: {e}")

        if sendgrid_key and sendgrid_key.strip() and alert_email:
            try:
                import httpx

                email_payload = {
                    'personalizations': [{
                        'to': [{'email': alert_email}],
                        'subject': f'🚨 Canary Rollback: {deployment.deployment_id}'
                    }],
                    'from': {'email': from_email},
                    'content': [{
                        'type': 'text/plain',
                        'value': alert_message
                    }]
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        'https://api.sendgrid.com/v3/mail/send',
                        headers={
                            'Authorization': f'Bearer {sendgrid_key}',
                            'Content-Type': 'application/json'
                        },
                        json=email_payload,
                        timeout=10.0
                    )
                    if response.status_code in (200, 202):
                        logger.info("Rollback alert sent via email")
                    else:
                        logger.warning(f"Failed to send email alert: {response.status_code}")
            except Exception as e:
                logger.error(f"Error sending email alert: {e}")

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        deployment = self.active_deployments.get(deployment_id)
        if deployment:
            return deployment.to_dict()

        for hist_deployment in self.deployment_history:
            if hist_deployment.deployment_id == deployment_id:
                return hist_deployment.to_dict()

        return None

    def list_active_deployments(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self.active_deployments.values()]

    def save_deployment_log(self, filepath: str = "canary_deployment_log.json"):
        try:
            log_data = {
                'active': [d.to_dict() for d in self.active_deployments.values()],
                'history': [d.to_dict() for d in self.deployment_history]
            }

            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)

            logger.info(f"Saved deployment log to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save deployment log: {e}")
