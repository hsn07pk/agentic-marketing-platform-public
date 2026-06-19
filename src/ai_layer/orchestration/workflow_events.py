"""
Workflow event logging for complete transparency
Tracks all workflow state changes and alerts for dashboard visibility
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from ...data_layer.database.models import (
    WorkflowEvent,
    WorkflowEventType,
    AlertSeverity
)

logger = logging.getLogger(__name__)


class WorkflowEventLogger:

    def __init__(self, db: AsyncSession, campaign_id: str):
        self.db = db
        self.campaign_id = campaign_id

    @staticmethod
    def _sanitize_json(obj):
        """Replace JSON-incompatible float values (Infinity, NaN) with safe alternatives."""
        import math
        if isinstance(obj, dict):
            return {k: WorkflowEventLogger._sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [WorkflowEventLogger._sanitize_json(v) for v in obj]
        if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
            return None
        return obj

    async def log_event(
        self,
        event_type: WorkflowEventType,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        workflow_node: Optional[str] = None,
        workflow_state: Optional[str] = None,
        content_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        is_user_actionable: bool = False
    ) -> str:
        try:
            safe_details = self._sanitize_json(details or {})
            stmt = insert(WorkflowEvent).values(
                campaign_id=self.campaign_id,
                content_id=content_id,
                event_type=event_type,
                severity=severity,
                workflow_node=workflow_node,
                workflow_state=workflow_state,
                title=title,
                message=message,
                details=safe_details,
                is_user_actionable=is_user_actionable,
                created_at=datetime.utcnow()
            ).returning(WorkflowEvent.id)

            result = await self.db.execute(stmt)
            await self.db.commit()
            event_id = result.scalar_one()

            logger.info(
                f"Workflow event logged: {event_type.value} - {title}",
                extra={
                    "campaign_id": self.campaign_id,
                    "event_type": event_type.value,
                    "severity": severity.value,
                    "event_id": str(event_id)
                }
            )

            return str(event_id)

        except Exception as e:
            logger.error(f"Failed to log workflow event: {e}", exc_info=True)
            # Don't fail the workflow if logging fails
            await self.db.rollback()
            return None

    async def log_workflow_started(self):
        return await self.log_event(
            event_type=WorkflowEventType.WORKFLOW_STARTED,
            title="Campaign Workflow Started",
            message="Content generation workflow has been initiated. Generating campaign content.",
            severity=AlertSeverity.INFO,
            workflow_state="running"
        )

    async def log_workflow_completed(self, content_id: str):
        return await self.log_event(
            event_type=WorkflowEventType.WORKFLOW_COMPLETED,
            title="Workflow Completed Successfully",
            message="Content has been generated, validated, and deployed successfully.",
            severity=AlertSeverity.INFO,
            workflow_state="completed",
            content_id=content_id
        )

    async def log_workflow_failed(self, error: str):
        return await self.log_event(
            event_type=WorkflowEventType.WORKFLOW_FAILED,
            title="Workflow Failed",
            message=f"Workflow encountered a critical error: {error}",
            severity=AlertSeverity.ERROR,
            workflow_state="failed",
            details={"error": error}
        )

    async def log_content_generated(self, content_id: str, strategy: str):
        return await self.log_event(
            event_type=WorkflowEventType.CONTENT_GENERATED,
            title="Content Generated",
            message=f"New content generated using '{strategy}' strategy. Proceeding to safety validation.",
            severity=AlertSeverity.INFO,
            workflow_node="content_generation",
            content_id=content_id,
            details={"strategy": strategy}
        )

    async def log_safety_check_passed(self, content_id: str, safety_score: float, **extra_scores):
        details = {"safety_score": safety_score}
        details.update(extra_scores)
        return await self.log_event(
            event_type=WorkflowEventType.SAFETY_CHECK_PASSED,
            title="Safety Check Passed",
            message=f"Content passed safety validation with score {safety_score:.2f}. Proceeding to budget check.",
            severity=AlertSeverity.INFO,
            workflow_node="safety_validation",
            content_id=content_id,
            details=details
        )

    async def log_safety_check_failed(self, content_id: str, safety_score: float, issues: list, **extra_scores):
        details = {"safety_score": safety_score, "issues": issues}
        details.update(extra_scores)
        return await self.log_event(
            event_type=WorkflowEventType.SAFETY_CHECK_FAILED,
            title="Safety Check Failed",
            message=f"Content failed safety validation (score: {safety_score:.2f}). Issues: {', '.join(issues)}",
            severity=AlertSeverity.WARNING,
            workflow_node="safety_validation",
            content_id=content_id,
            details=details
        )

    async def log_hitl_queue_added(self, content_id: str, safety_score: float, reason: str):
        return await self.log_event(
            event_type=WorkflowEventType.HITL_QUEUE_ADDED,
            title="Pending Human Review",
            message=f"Content added to review queue (safety score: {safety_score:.2f}). Reason: {reason}. Workflow paused until approval.",
            severity=AlertSeverity.WARNING,
            workflow_node="human_review",
            workflow_state="paused",
            content_id=content_id,
            details={"safety_score": safety_score, "reason": reason},
            is_user_actionable=True
        )

    async def log_hitl_reviewed(self, content_id: str, decision: str, reviewer: str):
        severity = AlertSeverity.INFO if decision == "approve" else AlertSeverity.WARNING
        return await self.log_event(
            event_type=WorkflowEventType.HITL_REVIEWED,
            title=f"Content {decision.capitalize()}d",
            message=f"Content has been {decision}d by {reviewer}. {'Resuming workflow.' if decision == 'approve' else 'Regenerating content.'}",
            severity=severity,
            workflow_node="human_review",
            content_id=content_id,
            details={"decision": decision, "reviewer": reviewer}
        )

    async def log_workflow_resumed(self, content_id: str):
        return await self.log_event(
            event_type=WorkflowEventType.WORKFLOW_RESUMED,
            title="Workflow Resumed",
            message="Content approved. Resuming workflow from budget check.",
            severity=AlertSeverity.INFO,
            workflow_state="running",
            content_id=content_id
        )

    async def log_budget_warning(self, current_cost: float, limit: float, percentage: float):
        return await self.log_event(
            event_type=WorkflowEventType.BUDGET_WARNING,
            title="Budget Warning",
            message=f"Daily API costs at {percentage:.0f}% of limit (€{current_cost:.2f} / €{limit:.2f})",
            severity=AlertSeverity.WARNING,
            workflow_node="cost_check",
            details={"current_cost": current_cost, "limit": limit, "percentage": percentage},
            is_user_actionable=True
        )

    async def log_budget_exceeded(self, current_cost: float, limit: float):
        return await self.log_event(
            event_type=WorkflowEventType.BUDGET_EXCEEDED,
            title="Budget Limit Exceeded",
            message=f"Daily API cost limit exceeded (€{current_cost:.2f} / €{limit:.2f}). Workflow halted. Increase MAX_DAILY_API_COST in settings to continue.",
            severity=AlertSeverity.CRITICAL,
            workflow_node="cost_check",
            workflow_state="paused",
            details={"current_cost": current_cost, "limit": limit},
            is_user_actionable=True
        )

    async def log_cost_check_passed(self, current_cost: float, limit: float):
        return await self.log_event(
            event_type=WorkflowEventType.COST_CHECK_PASSED,
            title="Budget Check Passed",
            message=f"Daily costs within budget (€{current_cost:.2f} / €{limit:.2f}). Proceeding to deployment.",
            severity=AlertSeverity.INFO,
            workflow_node="cost_check",
            details={"current_cost": current_cost, "limit": limit}
        )

    async def log_content_deployed(self, content_id: str, platform: str):
        return await self.log_event(
            event_type=WorkflowEventType.CONTENT_DEPLOYED,
            title="Content Deployed",
            message=f"Content successfully deployed to {platform}. Campaign is now live!",
            severity=AlertSeverity.INFO,
            workflow_node="deployment",
            content_id=content_id,
            details={"platform": platform}
        )

    async def log_error(self, error: str, node: str, content_id: Optional[str] = None):
        return await self.log_event(
            event_type=WorkflowEventType.ERROR_OCCURRED,
            title=f"Error in {node}",
            message=f"An error occurred during {node}: {error}",
            severity=AlertSeverity.ERROR,
            workflow_node=node,
            content_id=content_id,
            details={"error": error}
        )

    async def log_node_started(
        self,
        node: str,
        content_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        node_titles = {
            "market_observation": "Gathering Market Intelligence",
            "strategy_optimization": "Optimizing Strategy",
            "content_generation": "Generating Content",
            "safety_validation": "Validating Safety",
            "human_review": "Awaiting Human Review",
            "cost_check": "Checking Budget",
            "simulation": "Simulating Campaign Performance",
            "golden_test_gate": "Running Golden Tests",
            "marl_gating": "Evaluating MARL Policy",
            "deployment": "Deploying Content",
            "canary_deployment": "Starting Canary Deployment",
            "monitoring": "Setting Up Monitoring"
        }
        
        node_descriptions = {
            "market_observation": "Analyzing competitive landscape and market trends.",
            "strategy_optimization": "Selecting optimal content strategy using bandits.",
            "content_generation": "Creating personalized content for target persona.",
            "safety_validation": "Checking content for safety, toxicity, and brand compliance.",
            "human_review": "Content requires human review before proceeding.",
            "cost_check": "Verifying budget constraints and API costs.",
            "simulation": "Running Monte Carlo simulation to predict campaign performance.",
            "golden_test_gate": "Executing golden test suite to validate deployment safety.",
            "marl_gating": "Evaluating if new policy should replace current policy.",
            "deployment": "Publishing content to target platform.",
            "canary_deployment": "Starting progressive rollout at 5% traffic.",
            "monitoring": "Setting up performance monitoring and alerting."
        }
        
        return await self.log_event(
            event_type=WorkflowEventType.NODE_STARTED,
            title=node_titles.get(node, f"Starting {node}"),
            message=node_descriptions.get(node, f"Processing: {node_titles.get(node, node)}."),
            severity=AlertSeverity.INFO,
            workflow_node=node,
            content_id=content_id,
            details=details or {}
        )

    async def log_node_completed(
        self,
        node: str,
        content_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None
    ):
        node_success_messages = {
            "market_observation": "Market intelligence gathered successfully.",
            "strategy_optimization": "Optimal strategy selected.",
            "content_generation": "Content generated successfully.",
            "safety_validation": "Safety validation completed.",
            "human_review": "Human review decision received.",
            "cost_check": "Budget check passed.",
            "simulation": "Campaign simulation completed.",
            "golden_test_gate": "Golden test suite completed.",
            "marl_gating": "MARL policy evaluation completed.",
            "deployment": "Content deployed to platform.",
            "canary_deployment": "Canary deployment initiated.",
            "monitoring": "Monitoring activated."
        }
        
        # Merge duration and result into details if provided
        event_details = details or {}
        if duration is not None:
            event_details["duration_seconds"] = round(duration, 2)
        if result is not None:
            event_details.update(result)
        
        return await self.log_event(
            event_type=WorkflowEventType.NODE_COMPLETED,
            title=f"{node.replace('_', ' ').title()} Complete",
            message=node_success_messages.get(node, f"Successfully completed {node} phase."),
            severity=AlertSeverity.INFO,
            workflow_node=node,
            content_id=content_id,
            details=event_details
        )

    async def log_simulation_passed(
        self,
        content_id: str,
        predicted_ctr: float,
        predicted_conversions: int,
        predicted_cpl: float
    ):
        return await self.log_event(
            event_type=WorkflowEventType.NODE_COMPLETED,
            title="Simulation Passed",
            message=f"Campaign simulation successful! Predicted CTR: {predicted_ctr:.2%}, Conversions: {predicted_conversions}, CPL: €{predicted_cpl:.2f}. Proceeding to deployment.",
            severity=AlertSeverity.INFO,
            workflow_node="simulation",
            content_id=content_id,
            details={
                "predicted_ctr": predicted_ctr,
                "predicted_conversions": predicted_conversions,
                "predicted_cpl": predicted_cpl,
                "simulation_passed": True
            }
        )

    async def log_simulation_failed(
        self,
        content_id: str,
        predicted_ctr: float,
        predicted_conversions: int,
        min_ctr: float,
        min_conversions: int
    ):
        return await self.log_event(
            event_type=WorkflowEventType.ERROR_OCCURRED,
            title="Simulation Failed",
            message=f"Campaign simulation did not meet thresholds. CTR: {predicted_ctr:.2%} (need ≥{min_ctr:.2%}), Conversions: {predicted_conversions} (need ≥{min_conversions}). Regenerating content.",
            severity=AlertSeverity.WARNING,
            workflow_node="simulation",
            content_id=content_id,
            details={
                "predicted_ctr": predicted_ctr,
                "predicted_conversions": predicted_conversions,
                "min_ctr": min_ctr,
                "min_conversions": min_conversions,
                "simulation_passed": False
            }
        )

    async def log_golden_test_passed(self, pass_rate: float, tests_passed: int, tests_total: int):
        return await self.log_event(
            event_type=WorkflowEventType.NODE_COMPLETED,
            title="Golden Tests Passed",
            message=f"Golden test suite passed ({tests_passed}/{tests_total} tests, {pass_rate:.1f}% pass rate). Deployment approved.",
            severity=AlertSeverity.INFO,
            workflow_node="golden_test_gate",
            details={
                "pass_rate": pass_rate,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "approved_for_deployment": True
            }
        )

    async def log_golden_test_failed(self, pass_rate: float, tests_passed: int, tests_failed: int, failures: list):
        return await self.log_event(
            event_type=WorkflowEventType.ERROR_OCCURRED,
            title="Golden Tests Failed - Deployment Blocked",
            message=f"Golden test suite FAILED ({tests_passed} passed, {tests_failed} failed, {pass_rate:.1f}% pass rate). Deployment is blocked until tests pass.",
            severity=AlertSeverity.CRITICAL,
            workflow_node="golden_test_gate",
            details={
                "pass_rate": pass_rate,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "failures": failures[:5],  # Limit to first 5 failures
                "approved_for_deployment": False
            },
            is_user_actionable=True
        )

    async def log_canary_started(self, canary_id: str, policy_id: str, traffic_percent: int = 5):
        return await self.log_event(
            event_type=WorkflowEventType.CANARY_STARTED,
            title="Canary Deployment Started",
            message=f"Progressive rollout initiated. Policy '{policy_id}' deployed to {traffic_percent}% of traffic. ID: {canary_id}",
            severity=AlertSeverity.INFO,
            workflow_node="canary_deployment",
            details={
                "canary_id": canary_id,
                "policy_id": policy_id,
                "traffic_percent": traffic_percent
            }
        )

    async def log_canary_promoted(self, canary_id: str, new_traffic_percent: int):
        return await self.log_event(
            event_type=WorkflowEventType.CANARY_PROMOTED,
            title="Canary Promoted",
            message=f"Canary deployment promoted to {new_traffic_percent}% traffic. Metrics are stable.",
            severity=AlertSeverity.INFO,
            workflow_node="canary_deployment",
            details={
                "canary_id": canary_id,
                "traffic_percent": new_traffic_percent
            }
        )

    async def log_canary_rolled_back(self, canary_id: str, reason: str, metrics: Optional[Dict[str, Any]] = None):
        return await self.log_event(
            event_type=WorkflowEventType.CANARY_ROLLED_BACK,
            title="Canary Rolled Back",
            message=f"Canary deployment rolled back due to: {reason}. All traffic restored to baseline.",
            severity=AlertSeverity.WARNING,
            workflow_node="canary_deployment",
            details={
                "canary_id": canary_id,
                "reason": reason,
                "metrics": metrics or {}
            },
            is_user_actionable=True
        )

    async def log_marl_approved(
        self,
        policy_id: str,
        lift_percentage: float,
        baseline_reward: float,
        new_policy_reward: float
    ):
        return await self.log_event(
            event_type=WorkflowEventType.NODE_COMPLETED,
            title="MARL Policy Approved",
            message=f"MARL policy '{policy_id}' approved! Offline evaluation shows {lift_percentage:.1f}% lift over baseline.",
            severity=AlertSeverity.INFO,
            workflow_node="marl_gating",
            details={
                "policy_id": policy_id,
                "lift_percentage": lift_percentage,
                "baseline_reward": baseline_reward,
                "new_policy_reward": new_policy_reward,
                "approved": True
            }
        )

    async def log_marl_rejected(self, policy_id: str, reason: str, lift_percentage: float = 0.0):
        return await self.log_event(
            event_type=WorkflowEventType.NODE_COMPLETED,
            title="MARL Policy Rejected",
            message=f"MARL policy '{policy_id}' did not meet promotion criteria: {reason}. Using baseline strategy (policies require ≥20% lift).",
            severity=AlertSeverity.INFO,
            workflow_node="marl_gating",
            details={
                "policy_id": policy_id,
                "reason": reason,
                "lift_percentage": lift_percentage,
                "approved": False
            },
            is_user_actionable=False
        )

