"""
Simulation-to-Live Accuracy Tracker

Research Plan RQ2: Achieve >90% simulation-to-live accuracy.
Research Plan Section 5.3: MAPE < 10% target.

This module tracks and measures the accuracy of simulated campaign predictions
against actual live performance, supporting RQ2 validation.
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID
import numpy as np
from sqlalchemy import select, func, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from ...data_layer.database.models import (
    SimulationLiveAccuracy, Campaign, CalibrationRun
)
from ...data_layer.database.connection import get_async_session
from ...simulation.validators import SimulationValidator

logger = logging.getLogger(__name__)


class SimulationAccuracyTracker:
    """
    Tracks simulation-to-live accuracy for RQ2 validation.

    Research Plan Requirements:
    - Section 5.3: MAPE < 10% (>90% accuracy)
    - RQ2: Simulation must accurately predict live performance
    """

    RQ2_TARGET_ACCURACY = 90.0  # Target: >90%
    RQ2_TARGET_MAPE = 10.0  # Target: MAPE < 10%

    # Weights for overall accuracy calculation (per Research Plan Section 5.3)
    METRIC_WEIGHTS = {
        'impressions': 0.2,
        'clicks': 0.3,
        'conversions': 0.4,
        'ctr': 0.1
    }

    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session

    async def record_simulation_predictions(
        self,
        campaign_id: str,
        simulated_impressions: int,
        simulated_clicks: int,
        simulated_conversions: int,
        simulated_ctr: float,
        simulated_cpl: float,
        simulation_timestamp: Optional[datetime] = None
    ) -> str:
        try:
            async with get_async_session() as session:
                record = SimulationLiveAccuracy(
                    campaign_id=UUID(campaign_id),
                    simulated_impressions=simulated_impressions,
                    simulated_clicks=simulated_clicks,
                    simulated_conversions=simulated_conversions,
                    simulated_ctr=simulated_ctr,
                    simulated_cpl=simulated_cpl,
                    simulation_timestamp=simulation_timestamp or datetime.utcnow(),
                    measurement_type="pending",
                    rq2_target=self.RQ2_TARGET_ACCURACY
                )

                session.add(record)
                await session.commit()
                await session.refresh(record)

                logger.info(
                    f"Recorded simulation predictions for campaign {campaign_id}",
                    extra={
                        "event": "simulation_predictions_recorded",
                        "campaign_id": campaign_id,
                        "simulated_ctr": simulated_ctr,
                        "simulated_conversions": simulated_conversions
                    }
                )

                return str(record.id)

        except Exception as e:
            logger.error(f"Failed to record simulation predictions: {e}")
            raise

    async def measure_accuracy(
        self,
        campaign_id: str,
        actual_impressions: int,
        actual_clicks: int,
        actual_conversions: int,
        actual_ctr: float,
        actual_cpl: float,
        measurement_type: str = "interim"
    ) -> Dict[str, Any]:
        try:
            async with get_async_session() as session:
                stmt = select(SimulationLiveAccuracy).where(
                    SimulationLiveAccuracy.campaign_id == UUID(campaign_id)
                ).order_by(SimulationLiveAccuracy.created_at.desc()).limit(1)

                result = await session.execute(stmt)
                record = result.scalar_one_or_none()

                if not record:
                    logger.warning(f"No simulation prediction found for campaign {campaign_id}")
                    return {
                        "error": "No simulation prediction found",
                        "campaign_id": campaign_id
                    }

                mape_impressions = self._calculate_mape(
                    record.simulated_impressions, actual_impressions
                )
                mape_clicks = self._calculate_mape(
                    record.simulated_clicks, actual_clicks
                )
                mape_conversions = self._calculate_mape(
                    record.simulated_conversions, actual_conversions
                )
                mape_ctr = self._calculate_mape(
                    record.simulated_ctr, actual_ctr
                )
                mape_cpl = self._calculate_mape(
                    record.simulated_cpl, actual_cpl
                ) if actual_cpl > 0 else None

                overall_mape = self._calculate_weighted_mape({
                    'impressions': mape_impressions,
                    'clicks': mape_clicks,
                    'conversions': mape_conversions,
                    'ctr': mape_ctr
                })

                overall_accuracy = max(0.0, 100.0 - overall_mape)

                passes_threshold = overall_accuracy >= self.RQ2_TARGET_ACCURACY
                rq2_gap = overall_accuracy - self.RQ2_TARGET_ACCURACY

                record.actual_impressions = actual_impressions
                record.actual_clicks = actual_clicks
                record.actual_conversions = actual_conversions
                record.actual_ctr = actual_ctr
                record.actual_cpl = actual_cpl

                record.mape_impressions = mape_impressions
                record.mape_clicks = mape_clicks
                record.mape_conversions = mape_conversions
                record.mape_ctr = mape_ctr
                record.mape_cpl = mape_cpl

                record.overall_mape = overall_mape
                record.overall_accuracy = overall_accuracy
                record.passes_threshold = passes_threshold
                record.rq2_gap = rq2_gap

                record.measurement_timestamp = datetime.utcnow()
                record.measurement_type = measurement_type

                await session.commit()

                result_data = {
                    "campaign_id": campaign_id,
                    "record_id": str(record.id),
                    "mape": {
                        "impressions": round(mape_impressions, 2),
                        "clicks": round(mape_clicks, 2),
                        "conversions": round(mape_conversions, 2),
                        "ctr": round(mape_ctr, 2),
                        "cpl": round(mape_cpl, 2) if mape_cpl else None,
                        "overall": round(overall_mape, 2)
                    },
                    "accuracy": {
                        "overall": round(overall_accuracy, 2),
                        "target": self.RQ2_TARGET_ACCURACY,
                        "gap": round(rq2_gap, 2)
                    },
                    "rq2_compliance": {
                        "passes": passes_threshold,
                        "target": f">{self.RQ2_TARGET_ACCURACY}%",
                        "achieved": f"{overall_accuracy:.1f}%",
                        "status": "PASS" if passes_threshold else "FAIL"
                    },
                    "measurement_type": measurement_type,
                    "measured_at": datetime.utcnow().isoformat()
                }

                logger.info(
                    f"Measured simulation accuracy for campaign {campaign_id}: "
                    f"{overall_accuracy:.1f}% (RQ2: {'PASS' if passes_threshold else 'FAIL'})",
                    extra={
                        "event": "simulation_accuracy_measured",
                        "campaign_id": campaign_id,
                        "overall_accuracy": overall_accuracy,
                        "overall_mape": overall_mape,
                        "passes_rq2": passes_threshold
                    }
                )

                return result_data

        except Exception as e:
            logger.error(f"Failed to measure simulation accuracy: {e}")
            raise

    async def get_aggregate_accuracy(
        self,
        days: int = 30,
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            async with get_async_session() as session:
                from datetime import timedelta

                cutoff = datetime.utcnow() - timedelta(days=days)

                query = select(
                    func.avg(SimulationLiveAccuracy.overall_accuracy).label('avg_accuracy'),
                    func.avg(SimulationLiveAccuracy.overall_mape).label('avg_mape'),
                    func.count(SimulationLiveAccuracy.id).label('total_campaigns'),
                    func.sum(
                        func.cast(SimulationLiveAccuracy.passes_threshold, Integer)
                    ).label('passing_campaigns')
                ).where(
                    SimulationLiveAccuracy.created_at >= cutoff,
                    SimulationLiveAccuracy.overall_accuracy.isnot(None)
                )

                result = await session.execute(query)
                row = result.first()

                total = row.total_campaigns or 0
                passing = row.passing_campaigns or 0
                avg_accuracy = row.avg_accuracy or 0.0
                avg_mape = row.avg_mape or 0.0

                rq2_pass_rate = (passing / total * 100) if total > 0 else 0.0

                return {
                    "period_days": days,
                    "total_campaigns": total,
                    "passing_campaigns": passing,
                    "failing_campaigns": total - passing,
                    "avg_accuracy": round(avg_accuracy, 2),
                    "avg_mape": round(avg_mape, 2),
                    "rq2_pass_rate": round(rq2_pass_rate, 2),
                    "rq2_target": self.RQ2_TARGET_ACCURACY,
                    "rq2_status": "MEETING_TARGET" if rq2_pass_rate >= 80 else "BELOW_TARGET"
                }

        except Exception as e:
            logger.error(f"Failed to get aggregate accuracy: {e}")
            return {
                "error": str(e),
                "period_days": days,
                "total_campaigns": 0
            }

    async def get_accuracy_trend(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        try:
            async with get_async_session() as session:
                from datetime import timedelta
                from sqlalchemy import func, cast, Date

                cutoff = datetime.utcnow() - timedelta(days=days)

                query = select(
                    cast(SimulationLiveAccuracy.created_at, Date).label('date'),
                    func.avg(SimulationLiveAccuracy.overall_accuracy).label('avg_accuracy'),
                    func.count(SimulationLiveAccuracy.id).label('count')
                ).where(
                    SimulationLiveAccuracy.created_at >= cutoff,
                    SimulationLiveAccuracy.overall_accuracy.isnot(None)
                ).group_by(
                    cast(SimulationLiveAccuracy.created_at, Date)
                ).order_by(
                    cast(SimulationLiveAccuracy.created_at, Date)
                )

                result = await session.execute(query)
                rows = result.fetchall()

                return [
                    {
                        "date": row.date.isoformat(),
                        "avg_accuracy": round(row.avg_accuracy, 2),
                        "count": row.count,
                        "meets_target": row.avg_accuracy >= self.RQ2_TARGET_ACCURACY
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get accuracy trend: {e}")
            return []

    def _calculate_mape(self, predicted: float, actual: float) -> float:
        """Calculate Mean Absolute Percentage Error for a single value."""
        if actual == 0:
            return 100.0 if predicted > 0 else 0.0
        return abs(predicted - actual) / actual * 100

    def _calculate_weighted_mape(self, mape_values: Dict[str, float]) -> float:
        """Calculate weighted MAPE using Research Plan weights."""
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, mape in mape_values.items():
            if mape is not None and metric in self.METRIC_WEIGHTS:
                weight = self.METRIC_WEIGHTS[metric]
                weighted_sum += mape * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight


# Convenience functions
async def record_simulation_for_campaign(
    campaign_id: str,
    simulation_results: Dict[str, Any]
) -> str:
    """
    Record simulation results from workflow node.

    Called from langgraph_supervisor._simulate_campaign_node()
    """
    tracker = SimulationAccuracyTracker()
    return await tracker.record_simulation_predictions(
        campaign_id=campaign_id,
        simulated_impressions=simulation_results.get('impressions', 0),
        simulated_clicks=simulation_results.get('clicks', 0),
        simulated_conversions=simulation_results.get('conversions', 0),
        simulated_ctr=simulation_results.get('ctr', 0.0),
        simulated_cpl=simulation_results.get('cpl', 0.0)
    )


async def measure_campaign_accuracy(
    campaign_id: str,
    measurement_type: str = "interim"
) -> Dict[str, Any]:
    """
    Measure accuracy for a campaign using its current live metrics.

    Can be called during campaign execution or at completion.
    """
    async with get_async_session() as session:
        result = await session.execute(
            select(Campaign).where(Campaign.id == UUID(campaign_id))
        )
        campaign = result.scalar_one_or_none()

        if not campaign:
            return {"error": "Campaign not found"}

        tracker = SimulationAccuracyTracker()
        return await tracker.measure_accuracy(
            campaign_id=campaign_id,
            actual_impressions=campaign.impressions or 0,
            actual_clicks=campaign.clicks or 0,
            actual_conversions=campaign.conversions or 0,
            actual_ctr=campaign.ctr or 0.0,
            actual_cpl=campaign.cpl or 0.0,
            measurement_type=measurement_type
        )
