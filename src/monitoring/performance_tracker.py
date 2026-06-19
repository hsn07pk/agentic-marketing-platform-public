"""
Automatic performance monitoring for deployed campaigns

Tracks real vs predicted metrics and calculates accuracy.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import json

from ..data_layer.database.models import Campaign, Content, Metric
from ..simulation.calibration_utils import calculate_mape

logger = logging.getLogger(__name__)

class PerformanceTracker:

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def track_campaign_performance(self, campaign_id: str) -> Dict:
        """Track a campaign's actual vs predicted performance."""
        try:
            result = await self.db_session.execute(
                select(Campaign).where(Campaign.id == campaign_id)
            )
            campaign = result.scalar_one_or_none()

            if not campaign:
                logger.error(f"Campaign {campaign_id} not found")
                return {}

            simulation_results = campaign.config.get('simulation_results', {}) if campaign.config else {}

            if not simulation_results:
                logger.warning(f"No simulation results found for campaign {campaign_id}")
                return {}

            predicted_ctr = simulation_results.get('ctr', 0)
            predicted_conversions = simulation_results.get('conversions', 0)
            predicted_clicks = simulation_results.get('clicks', 0)
            predicted_cpl = simulation_results.get('cpl', 0)

            metrics_result = await self.db_session.execute(
                select(Metric).where(
                    and_(
                        Metric.campaign_id == campaign_id,
                        Metric.metric_type == 'campaign_summary'
                    )
                ).order_by(Metric.timestamp.desc())
            )
            latest_metrics = metrics_result.scalars().first()

            if not latest_metrics:
                actual_clicks = campaign.clicks or 0
                actual_conversions = campaign.conversions or 0
                actual_impressions = campaign.impressions or 0
                actual_ctr = campaign.ctr or 0
                actual_cpl = campaign.cpl or 0
            else:
                actual_clicks = latest_metrics.value.get('clicks', 0) if isinstance(latest_metrics.value, dict) else 0
                actual_conversions = latest_metrics.value.get('conversions', 0) if isinstance(latest_metrics.value, dict) else 0
                actual_impressions = latest_metrics.value.get('impressions', 0) if isinstance(latest_metrics.value, dict) else 0
                actual_ctr = actual_clicks / actual_impressions if actual_impressions > 0 else 0
                actual_cpl = (campaign.budget_spent / actual_conversions) if actual_conversions > 0 else 0

            if actual_clicks > 0 or actual_conversions > 0:
                import pandas as pd

                ctr_error = abs(actual_ctr - predicted_ctr) / actual_ctr * 100 if actual_ctr > 0 else 100

                conv_error = abs(actual_conversions - predicted_conversions) / actual_conversions * 100 if actual_conversions > 0 else 100

                mape = (ctr_error + conv_error) / 2
                accuracy = 100 - mape

                logger.info(f"📊 Campaign {campaign_id} Performance:")
                logger.info(f"   Predicted CTR: {predicted_ctr:.4f} | Actual: {actual_ctr:.4f} | Error: {ctr_error:.2f}%")
                logger.info(f"   Predicted Conversions: {predicted_conversions} | Actual: {actual_conversions} | Error: {conv_error:.2f}%")
                logger.info(f"   Overall MAPE: {mape:.2f}% | Accuracy: {accuracy:.2f}%")

                return {
                    'campaign_id': str(campaign_id),
                    'campaign_name': campaign.name,
                    'predicted': {
                        'ctr': predicted_ctr,
                        'clicks': predicted_clicks,
                        'conversions': predicted_conversions,
                        'cpl': predicted_cpl
                    },
                    'actual': {
                        'ctr': actual_ctr,
                        'clicks': actual_clicks,
                        'conversions': actual_conversions,
                        'cpl': actual_cpl,
                        'impressions': actual_impressions
                    },
                    'accuracy': {
                        'ctr_error_pct': ctr_error,
                        'conversions_error_pct': conv_error,
                        'mape': mape,
                        'accuracy': accuracy,
                        'meets_threshold': mape < 10.0
                    },
                    'status': campaign.status,
                    'last_updated': datetime.utcnow().isoformat()
                }
            else:
                logger.info(f"📊 Campaign {campaign_id}: No actual data yet (still running)")
                return {
                    'campaign_id': str(campaign_id),
                    'campaign_name': campaign.name,
                    'predicted': {
                        'ctr': predicted_ctr,
                        'clicks': predicted_clicks,
                        'conversions': predicted_conversions,
                        'cpl': predicted_cpl
                    },
                    'actual': {
                        'ctr': 0,
                        'clicks': 0,
                        'conversions': 0,
                        'cpl': 0,
                        'impressions': 0
                    },
                    'status': 'running',
                    'message': 'Campaign running - no actual data yet',
                    'last_updated': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to track campaign performance: {e}", exc_info=True)
            return {}

    async def get_all_campaigns_performance(self) -> List[Dict]:
        """Get performance tracking for all active campaigns."""
        try:
            result = await self.db_session.execute(
                select(Campaign).where(Campaign.status == 'RUNNING')
            )
            campaigns = result.scalars().all()

            performance_data = []

            for campaign in campaigns:
                perf = await self.track_campaign_performance(str(campaign.id))
                if perf:
                    performance_data.append(perf)

            return performance_data

        except Exception as e:
            logger.error(f"Failed to get campaigns performance: {e}", exc_info=True)
            return []

    async def log_performance_alert(self, campaign_id: str, alert_type: str, message: str):
        """Log performance alert."""
        logger.warning(f"⚠️  Performance Alert [{alert_type}] for campaign {campaign_id}: {message}")
