"""
Automatic Campaign & Experiment Completion Monitor

Runs in background and automatically completes:

CAMPAIGNS when:
- Budget threshold reached (98%)
- End date reached
- Daily budget limits exceeded

EXPERIMENTS when:
- Target sample size reached + statistical significance
- Duration expired
- Early stopping criteria met (Thompson Sampling >95% confidence)

Per Research Plan Section 7.4 (Cost Control), Section 5.2 (Campaign Duration),
and Section 2.3 (Multi-Armed Bandits)
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from ..data_layer.database.models import Campaign, CampaignStatus, Experiment, BanditArm
from ..ai_layer.learning.campaign_completion import should_complete_campaign
from ..ai_layer.learning.experiment_completion import should_complete_experiment
from ..ai_layer.learning.mlflow_integration import log_experiment_completion
from ..api.dependencies import get_db

logger = logging.getLogger(__name__)

class CampaignCompletionMonitor:

    def __init__(self, check_interval_seconds: int = 300):  # Check every 5 minutes
        self.check_interval_seconds = check_interval_seconds
        self.running = False

    async def check_all_campaigns(self, db_session: AsyncSession) -> dict:
        """Check all running campaigns and complete if criteria met."""
        try:
            result = await db_session.execute(
                select(Campaign).where(Campaign.status == CampaignStatus.RUNNING)
            )
            running_campaigns = result.scalars().all()

            if not running_campaigns:
                logger.debug("No running campaigns to check")
                return {"checked": 0, "completed": 0}

            completed_count = 0
            checked_count = len(running_campaigns)

            for campaign in running_campaigns:
                campaign_dict = {
                    "id": str(campaign.id),
                    "name": campaign.name,
                    "status": campaign.status.value,
                    "budget_total": campaign.budget_total or 0,
                    "budget_spent": campaign.budget_spent or 0,
                    "budget_daily_limit": campaign.budget_daily_limit or 0,
                    "end_date": campaign.end_date,
                    "total_impressions": campaign.impressions or 0,
                    "total_clicks": campaign.clicks or 0,
                    "total_conversions": campaign.conversions or 0
                }

                decision = should_complete_campaign(campaign_dict, budget_threshold=0.98)

                if decision.should_complete:
                    campaign.status = CampaignStatus.COMPLETED
                    campaign.updated_at = datetime.utcnow()

                    logger.info(
                        f"✅ Auto-completed campaign: {campaign.name} "
                        f"(ID: {campaign.id}) - {decision.reason}"
                    )

                    completed_count += 1

            if completed_count > 0:
                await db_session.commit()
                logger.info(
                    f"Campaign completion check: {completed_count}/{checked_count} campaigns auto-completed"
                )

            return {
                "checked": checked_count,
                "completed": completed_count,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error checking campaign completions: {e}", exc_info=True)
            await db_session.rollback()
            return {"checked": 0, "completed": 0, "error": str(e)}

    async def check_all_experiments(self, db_session: AsyncSession) -> dict:
        """Check all active experiments and complete if criteria met."""
        try:
            result = await db_session.execute(
                select(Experiment).where(Experiment.is_active == True)
            )
            active_experiments = result.scalars().all()

            if not active_experiments:
                logger.debug("No active experiments to check")
                return {"checked": 0, "completed": 0}

            completed_count = 0
            checked_count = len(active_experiments)

            for experiment in active_experiments:
                arms_result = await db_session.execute(
                    select(BanditArm).where(BanditArm.experiment_id == experiment.id)
                )
                arms = arms_result.scalars().all()

                experiment_dict = {
                    "id": str(experiment.id),
                    "name": experiment.name,
                    "algorithm": experiment.algorithm,
                    "is_active": experiment.is_active,
                    "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                    "parameters": experiment.parameters or {}
                }

                arms_data = [
                    {
                        "arm_id": arm.arm_id,
                        "pulls": arm.pulls,
                        "successes": arm.successes,
                        "failures": arm.failures,
                        "alpha": arm.alpha,
                        "beta": arm.beta,
                        "total_reward": arm.total_reward
                    }
                    for arm in arms
                ]

                decision = should_complete_experiment(experiment_dict, arms_data)

                if decision.should_complete:
                    experiment.is_active = False
                    experiment.ended_at = datetime.utcnow()
                    experiment.winner_variant = decision.winner

                    logger.info(
                        f"✅ Auto-completed experiment: {experiment.name} "
                        f"(ID: {experiment.id}) - {decision.reason}"
                    )

                    # Log experiment completion to MLflow (Research Plan Section 8.1)
                    try:
                        metrics = {
                            "total_pulls": sum(arm.get("pulls", 0) for arm in arms_data),
                            "total_successes": sum(arm.get("successes", 0) for arm in arms_data),
                            "confidence": decision.confidence,
                            "winner_ctr": next(
                                (arm.get("successes", 0) / max(arm.get("pulls", 1), 1)
                                 for arm in arms_data if arm.get("arm_id") == decision.winner),
                                0.0
                            )
                        }
                        run_id = await log_experiment_completion(
                            experiment_id=str(experiment.id),
                            experiment_name=experiment.name,
                            algorithm=experiment.algorithm,
                            arms=arms_data,
                            winner_arm=decision.winner,
                            metrics=metrics,
                            campaign_id=str(experiment.campaign_id) if experiment.campaign_id else None
                        )
                        if run_id:
                            logger.info(f"📊 Logged experiment to MLflow: run_id={run_id}")
                    except Exception as mlflow_error:
                        logger.warning(f"Failed to log experiment to MLflow: {mlflow_error}")

                    completed_count += 1

            if completed_count > 0:
                await db_session.commit()
                logger.info(
                    f"Experiment completion check: {completed_count}/{checked_count} experiments auto-completed"
                )

            return {
                "checked": checked_count,
                "completed": completed_count,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error checking experiment completions: {e}", exc_info=True)
            await db_session.rollback()
            return {"checked": 0, "completed": 0, "error": str(e)}

    async def run_forever(self):
        """Main monitor loop - checks campaigns and experiments periodically."""
        self.running = True
        logger.info(
            f"🔍 Campaign & Experiment completion monitor started "
            f"(check interval: {self.check_interval_seconds}s)"
        )

        while self.running:
            try:
                async for db_session in get_db():
                    campaign_summary = await self.check_all_campaigns(db_session)

                    if campaign_summary.get("completed", 0) > 0:
                        logger.info(
                            f"✅ {campaign_summary['completed']} campaigns auto-completed "
                            f"(checked {campaign_summary['checked']} total)"
                        )

                    experiment_summary = await self.check_all_experiments(db_session)

                    if experiment_summary.get("completed", 0) > 0:
                        logger.info(
                            f"✅ {experiment_summary['completed']} experiments auto-completed "
                            f"(checked {experiment_summary['checked']} total)"
                        )

                    break  # Exit the async for to close session

                await asyncio.sleep(self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute on error

    def stop(self):
        logger.info("Stopping campaign completion monitor")
        self.running = False

_monitor: Optional[CampaignCompletionMonitor] = None

async def start_campaign_monitor(check_interval_seconds: int = 300):
    """Start the global campaign completion monitor."""
    global _monitor

    if _monitor is None:
        _monitor = CampaignCompletionMonitor(check_interval_seconds)
        asyncio.create_task(_monitor.run_forever())
        logger.info(
            f"✅ Campaign completion monitor initialized "
            f"(checks every {check_interval_seconds}s)"
        )

async def stop_campaign_monitor():
    """Stop the global campaign completion monitor."""
    global _monitor

    if _monitor:
        _monitor.stop()
        _monitor = None

async def trigger_completion_check():
    """Manually trigger campaign and experiment completion check."""
    global _monitor

    if _monitor is None:
        _monitor = CampaignCompletionMonitor()

    async for db_session in get_db():
        campaign_summary = await _monitor.check_all_campaigns(db_session)
        experiment_summary = await _monitor.check_all_experiments(db_session)

        return {
            "campaigns": campaign_summary,
            "experiments": experiment_summary,
            "total_checked": campaign_summary.get("checked", 0) + experiment_summary.get("checked", 0),
            "total_completed": campaign_summary.get("completed", 0) + experiment_summary.get("completed", 0)
        }
