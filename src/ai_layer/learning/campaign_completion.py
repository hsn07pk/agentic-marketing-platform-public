"""
Campaign Completion Logic - Research Plan Sections 7.4, 5.2, 2.1

Determines when campaigns should complete based on:
- Budget depletion (Section 7.4: Cost Control)
- End date reached (Section 5.2: Campaign Duration)
- Daily budget limit violations
- Manual completion

Per Research Plan Section 2.1: Campaigns must complete to measure final ROI.
No real campaign has infinite budget or time.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CampaignCompletionDecision:
    should_complete: bool
    reason: str
    completion_type: str  # budget_depleted, end_date_reached, manual, daily_limit
    final_metrics: Dict[str, Any]


class CampaignCompletionChecker:
    """
    Checks if campaigns meet completion criteria per Research Plan

    Research Plan Requirements:
    - Section 7.4: Budget tracking and enforcement
    - Section 5.2: Finite campaign duration (7-day campaigns)
    - Section 2.1: Final ROI measurement requires completion
    """

    def __init__(
        self,
        budget_threshold: float = 0.98,  # Complete at 98% budget spent
        allow_overspend: bool = False
    ):
        self.budget_threshold = budget_threshold
        self.allow_overspend = allow_overspend

    def check_completion(
        self,
        campaign: Dict[str, Any]
    ) -> CampaignCompletionDecision:
        campaign_id = campaign.get('id', 'unknown')
        status = campaign.get('status', 'UNKNOWN')

        if status in ['COMPLETED', 'CANCELLED', 'FAILED']:
            return CampaignCompletionDecision(
                should_complete=False,
                reason=f"Campaign already {status}",
                completion_type="already_completed",
                final_metrics=self._extract_final_metrics(campaign)
            )

        budget_total = campaign.get('budget_total', 0)
        budget_spent = campaign.get('budget_spent', 0)

        if budget_total > 0:
            budget_pct = (budget_spent / budget_total) * 100

            if budget_spent >= budget_total:
                logger.info(f"Campaign {campaign_id}: Budget depleted (€{budget_spent:.2f}/€{budget_total:.2f})")
                return CampaignCompletionDecision(
                    should_complete=True,
                    reason=f"Budget fully depleted: €{budget_spent:.2f}/€{budget_total:.2f} (100%)",
                    completion_type="budget_depleted",
                    final_metrics=self._extract_final_metrics(campaign)
                )

            if budget_pct >= (self.budget_threshold * 100):
                logger.info(f"Campaign {campaign_id}: Budget threshold reached ({budget_pct:.1f}%)")
                return CampaignCompletionDecision(
                    should_complete=True,
                    reason=f"Budget threshold reached: €{budget_spent:.2f}/€{budget_total:.2f} ({budget_pct:.1f}%)",
                    completion_type="budget_depleted",
                    final_metrics=self._extract_final_metrics(campaign)
                )

        end_date = campaign.get('end_date')
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00')).date()
            elif isinstance(end_date, datetime):
                end_date = end_date.date()

            today = date.today()

            if today >= end_date:
                logger.info(f"Campaign {campaign_id}: End date reached ({end_date})")
                return CampaignCompletionDecision(
                    should_complete=True,
                    reason=f"End date reached: {end_date}",
                    completion_type="end_date_reached",
                    final_metrics=self._extract_final_metrics(campaign)
                )

        budget_daily_limit = campaign.get('budget_daily_limit', 0)
        if budget_daily_limit > 0:
            # This would require daily spend tracking in a separate table
            # For now, we'll skip this check and implement later if needed
            pass

        remaining_budget = budget_total - budget_spent if budget_total > 0 else 0
        days_remaining = (end_date - date.today()).days if end_date else None

        return CampaignCompletionDecision(
            should_complete=False,
            reason=f"Campaign active (€{remaining_budget:.2f} remaining, {days_remaining} days left)" if days_remaining else f"Campaign active (€{remaining_budget:.2f} remaining)",
            completion_type="running",
            final_metrics={}
        )

    def _extract_final_metrics(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract final metrics for ROI calculation (Section 2.1)

        Research Plan Section 2.1: Key Business Metrics
        - Demo Sign-ups
        - Cost-Per-Lead (CPL)
        - Booked Call Rate
        - Show Rate & Lead Quality
        """
        total_impressions = campaign.get('total_impressions', 0)
        total_clicks = campaign.get('total_clicks', 0)
        total_conversions = campaign.get('total_conversions', 0)
        budget_spent = campaign.get('budget_spent', 0)

        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        cpl = (budget_spent / total_conversions) if total_conversions > 0 else 0

        return {
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_conversions': total_conversions,
            'budget_spent': budget_spent,
            'ctr': round(ctr, 2),
            'conversion_rate': round(conversion_rate, 2),
            'cost_per_lead': round(cpl, 2),
            'roi': self._calculate_roi(campaign)
        }

    def _calculate_roi(self, campaign: Dict[str, Any]) -> float:
        """
        Calculate ROI per Research Plan Section 2.1

        Simplified ROI = (Revenue - Cost) / Cost * 100

        For now, we assume revenue = conversions * estimated_value_per_lead
        """
        budget_spent = campaign.get('budget_spent', 0)
        total_conversions = campaign.get('total_conversions', 0)

        # Assume €200 value per lead (configurable)
        estimated_value_per_lead = 200.0
        revenue = total_conversions * estimated_value_per_lead

        if budget_spent > 0:
            roi = ((revenue - budget_spent) / budget_spent) * 100
            return round(roi, 2)

        return 0.0


# Convenience function
def should_complete_campaign(
    campaign: Dict[str, Any],
    budget_threshold: float = 0.98
) -> CampaignCompletionDecision:
    checker = CampaignCompletionChecker(budget_threshold=budget_threshold)
    return checker.check_completion(campaign)
