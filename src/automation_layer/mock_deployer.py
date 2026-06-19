"""
Mock Deployment Service for Testing Without External APIs

Generates realistic campaign metrics based on simulation predictions
with controlled variance for RQ2 validation.
"""

import logging
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MockDeployer:

    def __init__(self):
        self.name = "MockDeployer"
        np.random.seed(42)

    def deploy_campaign(
        self,
        campaign_id: str,
        simulation_results: Dict[str, Any],
        budget: float,
        duration_days: int = 7
    ) -> Dict[str, Any]:
        logger.info(f"🎭 MOCK DEPLOYMENT for campaign {campaign_id}")

        predicted_ctr = simulation_results.get('ctr', 0.02)
        predicted_conversions = simulation_results.get('conversions', 5)
        predicted_impressions = simulation_results.get('impressions', 30000)
        predicted_clicks = simulation_results.get('clicks', 400)
        predicted_cpl = simulation_results.get('cpl', budget / max(predicted_conversions, 1))

        variance_factor = np.random.uniform(0.8, 1.2)

        actual_impressions = int(predicted_impressions * variance_factor)
        actual_clicks = int(predicted_clicks * variance_factor * np.random.uniform(0.85, 1.15))
        actual_ctr = actual_clicks / actual_impressions if actual_impressions > 0 else 0

        # Conversions have higher variance than clicks (user behavior less predictable)
        conversion_variance = np.random.uniform(0.7, 1.3)
        actual_conversions = max(1, int(predicted_conversions * conversion_variance))

        budget_spent = min(budget, budget * np.random.uniform(0.7, 1.0))
        actual_cpl = budget_spent / actual_conversions if actual_conversions > 0 else 0

        mape_clicks = abs(actual_clicks - predicted_clicks) / max(predicted_clicks, 1) * 100
        mape_conversions = abs(actual_conversions - predicted_conversions) / max(predicted_conversions, 1) * 100
        mape_ctr = abs(actual_ctr - predicted_ctr) / max(predicted_ctr, 0.001) * 100

        overall_mape = (mape_clicks + mape_conversions + mape_ctr) / 3

        daily_metrics = self._generate_daily_metrics(
            duration_days,
            actual_impressions,
            actual_clicks,
            actual_conversions,
            budget_spent
        )

        mock_result = {
            "status": "deployed_mock",
            "deployment_type": "mock",
            "message": "✅ Mock deployment successful - realistic metrics generated for testing",
            "deployment_timestamp": datetime.utcnow().isoformat(),
            "campaign_id": campaign_id,

            "actual_metrics": {
                "impressions": actual_impressions,
                "clicks": actual_clicks,
                "conversions": actual_conversions,
                "ctr": round(actual_ctr, 6),
                "cpl": round(actual_cpl, 2),
                "budget_spent": round(budget_spent, 2)
            },

            "predicted_metrics": {
                "impressions": predicted_impressions,
                "clicks": predicted_clicks,
                "conversions": predicted_conversions,
                "ctr": round(predicted_ctr, 6),
                "cpl": round(predicted_cpl, 2)
            },

            "validation": {
                "mape_clicks": round(mape_clicks, 2),
                "mape_conversions": round(mape_conversions, 2),
                "mape_ctr": round(mape_ctr, 2),
                "overall_mape": round(overall_mape, 2),
                "accuracy": round(100 - overall_mape, 2),
                "target_met": overall_mape < 10.0
            },

            "daily_metrics": daily_metrics,

            "platform": "mock_platform",
            "external_campaign_id": f"MOCK_{campaign_id[:8]}"
        }

        logger.info(f"   📊 Mock Metrics Generated:")
        logger.info(f"      Impressions: {actual_impressions:,} (predicted: {predicted_impressions:,})")
        logger.info(f"      Clicks: {actual_clicks:,} (predicted: {predicted_clicks:,})")
        logger.info(f"      Conversions: {actual_conversions} (predicted: {predicted_conversions})")
        logger.info(f"      CTR: {actual_ctr:.4f} (predicted: {predicted_ctr:.4f})")
        logger.info(f"      CPL: €{actual_cpl:.2f} (predicted: €{predicted_cpl:.2f})")
        logger.info(f"   🎯 Validation MAPE: {overall_mape:.2f}% (Target: <10%)")

        return mock_result

    def _generate_daily_metrics(
        self,
        duration_days: int,
        total_impressions: int,
        total_clicks: int,
        total_conversions: int,
        total_spent: float
    ) -> list:
        daily_data = []

        # Ramp-up distribution: days 1-2 (30%), days 3-5 (50%), days 6-7 (20%)
        weights = self._get_daily_weights(duration_days)

        for day in range(duration_days):
            weight = weights[day]

            daily_impressions = int(total_impressions * weight)
            daily_clicks = int(total_clicks * weight * np.random.uniform(0.9, 1.1))
            daily_conversions = int(total_conversions * weight * np.random.uniform(0.8, 1.2))
            daily_spent = total_spent * weight

            date = (datetime.utcnow() - timedelta(days=duration_days-day-1)).strftime('%Y-%m-%d')

            daily_data.append({
                "date": date,
                "impressions": daily_impressions,
                "clicks": daily_clicks,
                "conversions": max(0, daily_conversions),
                "spent": round(daily_spent, 2)
            })

        return daily_data

    def _get_daily_weights(self, duration_days: int) -> list:

        if duration_days <= 3:
            return [1/duration_days] * duration_days
        elif duration_days <= 7:
            return [0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05][:duration_days]
        else:
            # Gradual ramp-up (20%), plateau (50%), wind-down (30%)
            weights = []
            for day in range(duration_days):
                if day < duration_days * 0.2:
                    weight = 0.02 + (day / (duration_days * 0.2)) * 0.03
                elif day < duration_days * 0.7:
                    weight = 0.05
                else:
                    weight = 0.05 * (1 - (day - duration_days * 0.7) / (duration_days * 0.3))
                weights.append(weight)

            total = sum(weights)
            return [w / total for w in weights]
