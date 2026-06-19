"""
Weekly Learning Report Generator

Research Plan Section 10.2: Weekly Uplift Summary
"An automated report showing the best-performing hooks and content angles."

Generates automated weekly reports with:
- Best/worst performing hooks
- Platform performance comparison
- Persona performance analysis
- Bandit learning insights
- Recommendations
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ...data_layer.database.models import (
    WeeklyLearningReport, Campaign, Content, Experiment, BanditArm,
    Platform, Metric
)
from ...data_layer.database.connection import get_async_session

logger = logging.getLogger(__name__)


class WeeklyLearningReportGenerator:
    """
    Generates automated weekly learning reports per Research Plan Section 10.2.
    """

    async def generate_report(
        self,
        week_start: Optional[datetime] = None,
        week_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        try:
            if week_start is None:
                now = datetime.utcnow()
                # Use THIS Monday (not previous) to include recently deployed content
                week_start = now - timedelta(days=now.weekday())
                week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

            # Allow custom week_end or default to 7 days from start
            if week_end is None:
                week_end = week_start + timedelta(days=7)
            
            week_number = week_start.isocalendar()[1]
            year = week_start.year

            period_days = (week_end - week_start).days
            prev_week_start = week_start - timedelta(days=period_days)
            prev_week_end = week_start

            best_hooks = await self._get_best_hooks(week_start, week_end)
            worst_hooks = await self._get_worst_hooks(week_start, week_end)
            platform_perf = await self._get_platform_performance(week_start, week_end)
            persona_perf = await self._get_persona_performance(week_start, week_end)
            bandit_insights = await self._get_bandit_insights(week_start, week_end)

            this_week_metrics = await self._get_week_metrics(week_start, week_end)
            last_week_metrics = await self._get_week_metrics(prev_week_start, prev_week_end)

            ctr_change = self._calculate_change(
                this_week_metrics.get('ctr', 0),
                last_week_metrics.get('ctr', 0)
            )
            conv_change = self._calculate_change(
                this_week_metrics.get('conversions', 0),
                last_week_metrics.get('conversions', 0)
            )
            cpl_change = self._calculate_change(
                this_week_metrics.get('cpl', 0),
                last_week_metrics.get('cpl', 0)
            )

            recommendations = self._generate_recommendations(
                best_hooks, worst_hooks, platform_perf, persona_perf, bandit_insights
            )

            report_data = {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "week_number": week_number,
                "year": year,
                "best_hooks": best_hooks,
                "worst_hooks": worst_hooks,
                "platform_performance": platform_perf,
                "persona_performance": persona_perf,
                "bandit_insights": bandit_insights,
                "metrics": {
                    "this_week": this_week_metrics,
                    "last_week": last_week_metrics,
                    "changes": {
                        "ctr_change_pct": ctr_change,
                        "conversions_change_pct": conv_change,
                        "cpl_change_pct": cpl_change
                    }
                },
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }

            await self._save_report(report_data)

            logger.info(
                f"Generated weekly learning report for week {week_number}/{year}",
                extra={
                    "event": "weekly_report_generated",
                    "week_number": week_number,
                    "year": year
                }
            )

            return report_data

        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")
            return {"error": str(e)}

    async def _get_best_hooks(
        self,
        start: datetime,
        end: datetime,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            async with get_async_session() as session:
                query = select(
                    Content.headline,
                    Content.body,
                    func.sum(Content.impressions).label('impressions'),
                    func.sum(Content.clicks).label('clicks'),
                    func.sum(Content.conversions).label('conversions')
                ).where(
                    Content.created_at >= start,
                    Content.created_at < end,
                    Content.status == 'deployed'
                ).group_by(
                    Content.headline, Content.body
                ).having(
                    func.sum(Content.impressions) > 0
                ).order_by(
                    desc(func.sum(Content.clicks) / func.sum(Content.impressions))
                ).limit(limit)

                result = await session.execute(query)
                rows = result.fetchall()

                return [
                    {
                        "hook": row.headline[:100] if row.headline else "N/A",
                        "impressions": row.impressions or 0,
                        "clicks": row.clicks or 0,
                        "conversions": row.conversions or 0,
                        "ctr": round((row.clicks / row.impressions * 100), 2) if row.impressions else 0
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get best hooks: {e}")
            return []

    async def _get_worst_hooks(
        self,
        start: datetime,
        end: datetime,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        try:
            async with get_async_session() as session:
                query = select(
                    Content.headline,
                    func.sum(Content.impressions).label('impressions'),
                    func.sum(Content.clicks).label('clicks')
                ).where(
                    Content.created_at >= start,
                    Content.created_at < end,
                    Content.status == 'deployed'
                ).group_by(
                    Content.headline
                ).having(
                    func.sum(Content.impressions) > 100
                ).order_by(
                    func.sum(Content.clicks) / func.sum(Content.impressions)
                ).limit(limit)

                result = await session.execute(query)
                rows = result.fetchall()

                return [
                    {
                        "hook": row.headline[:100] if row.headline else "N/A",
                        "impressions": row.impressions or 0,
                        "clicks": row.clicks or 0,
                        "ctr": round((row.clicks / row.impressions * 100), 2) if row.impressions else 0
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get worst hooks: {e}")
            return []

    async def _get_platform_performance(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, Dict[str, Any]]:
        try:
            async with get_async_session() as session:
                query = select(
                    Campaign.platform,
                    func.sum(Campaign.impressions).label('impressions'),
                    func.sum(Campaign.clicks).label('clicks'),
                    func.sum(Campaign.conversions).label('conversions'),
                    func.sum(Campaign.budget_spent).label('spend')
                ).where(
                    Campaign.created_at >= start,
                    Campaign.created_at < end
                ).group_by(Campaign.platform)

                result = await session.execute(query)
                rows = result.fetchall()

                platform_data = {}
                for row in rows:
                    if hasattr(row.platform, 'value'):
                        platform_name = row.platform.value
                    else:
                        platform_name = str(row.platform) if row.platform else 'unknown'

                    impressions = row.impressions or 0
                    clicks = row.clicks or 0
                    conversions = row.conversions or 0
                    spend = row.spend or 0

                    platform_data[platform_name] = {
                        "impressions": impressions,
                        "clicks": clicks,
                        "conversions": conversions,
                        "spend": round(spend, 2),
                        "ctr": round((clicks / impressions * 100), 2) if impressions else 0,
                        "cpl": round(spend / conversions, 2) if conversions else 0
                    }

                return platform_data

        except Exception as e:
            logger.error(f"Failed to get platform performance: {e}")
            return {}

    async def _get_persona_performance(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, Dict[str, Any]]:
        try:
            async with get_async_session() as session:
                query = select(
                    Campaign.target_persona,
                    func.sum(Campaign.impressions).label('impressions'),
                    func.sum(Campaign.clicks).label('clicks'),
                    func.sum(Campaign.conversions).label('conversions')
                ).where(
                    Campaign.created_at >= start,
                    Campaign.created_at < end,
                    Campaign.target_persona.isnot(None)
                ).group_by(Campaign.target_persona)

                result = await session.execute(query)
                rows = result.fetchall()

                persona_data = {}
                for row in rows:
                    persona = row.target_persona or 'unknown'
                    impressions = row.impressions or 0
                    clicks = row.clicks or 0
                    conversions = row.conversions or 0

                    persona_data[persona] = {
                        "impressions": impressions,
                        "clicks": clicks,
                        "conversions": conversions,
                        "ctr": round((clicks / impressions * 100), 2) if impressions else 0,
                        "conversion_rate": round((conversions / clicks * 100), 2) if clicks else 0
                    }

                return persona_data

        except Exception as e:
            logger.error(f"Failed to get persona performance: {e}")
            return {}

    async def _get_bandit_insights(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        try:
            async with get_async_session() as session:
                query = select(
                    BanditArm.arm_id,
                    BanditArm.pulls,
                    BanditArm.successes,
                    BanditArm.alpha,
                    BanditArm.beta,
                    BanditArm.total_reward
                ).join(Experiment).where(
                    Experiment.started_at >= start,
                    Experiment.started_at < end
                ).order_by(desc(BanditArm.successes))

                result = await session.execute(query)
                arms = result.fetchall()

                if not arms:
                    return {"message": "No bandit experiments in this period"}

                # Calculate regret (simplified)
                total_pulls = sum(a.pulls for a in arms)
                total_successes = sum(a.successes for a in arms)
                best_arm_rate = max(
                    (a.successes / a.pulls if a.pulls else 0) for a in arms
                )

                exploration_pulls = sum(
                    a.pulls for a in arms if (a.successes / a.pulls if a.pulls else 0) < best_arm_rate * 0.8
                )
                exploitation_pulls = total_pulls - exploration_pulls
                exp_ratio = exploration_pulls / total_pulls if total_pulls else 0

                return {
                    "total_arms": len(arms),
                    "total_pulls": total_pulls,
                    "total_successes": total_successes,
                    "best_arm": arms[0].arm_id if arms else None,
                    "best_arm_success_rate": round(best_arm_rate * 100, 2),
                    "exploration_exploitation_ratio": round(exp_ratio, 2),
                    "regret_estimate": round((best_arm_rate * total_pulls - total_successes), 2),
                    "arms_summary": [
                        {
                            "arm_id": a.arm_id,
                            "pulls": a.pulls,
                            "successes": a.successes,
                            "success_rate": round((a.successes / a.pulls * 100), 2) if a.pulls else 0
                        }
                        for a in arms[:5]
                    ]
                }

        except Exception as e:
            logger.error(f"Failed to get bandit insights: {e}")
            return {}

    async def _get_week_metrics(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        try:
            async with get_async_session() as session:
                query = select(
                    func.sum(Campaign.impressions).label('impressions'),
                    func.sum(Campaign.clicks).label('clicks'),
                    func.sum(Campaign.conversions).label('conversions'),
                    func.sum(Campaign.budget_spent).label('spend')
                ).where(
                    Campaign.created_at >= start,
                    Campaign.created_at < end
                )

                result = await session.execute(query)
                row = result.first()

                impressions = row.impressions or 0
                clicks = row.clicks or 0
                conversions = row.conversions or 0
                spend = row.spend or 0

                return {
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": round(spend, 2),
                    "ctr": round((clicks / impressions * 100), 2) if impressions else 0,
                    "cpl": round(spend / conversions, 2) if conversions else 0
                }

        except Exception as e:
            logger.error(f"Failed to get week metrics: {e}")
            return {}

    def _calculate_change(self, current: float, previous: float) -> float:
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return round(((current - previous) / previous) * 100, 2)

    def _generate_recommendations(
        self,
        best_hooks: List,
        worst_hooks: List,
        platform_perf: Dict,
        persona_perf: Dict,
        bandit_insights: Dict
    ) -> List[str]:
        recommendations = []

        if best_hooks:
            top_hook = best_hooks[0]
            recommendations.append(
                f"Top performing hook achieved {top_hook.get('ctr', 0):.1f}% CTR. "
                f"Consider using similar messaging patterns."
            )

        if platform_perf:
            best_platform = max(
                platform_perf.items(),
                key=lambda x: x[1].get('ctr', 0),
                default=(None, {})
            )
            if best_platform[0]:
                recommendations.append(
                    f"{best_platform[0].title()} shows highest CTR at "
                    f"{best_platform[1].get('ctr', 0):.1f}%. Consider increasing budget allocation."
                )

        if persona_perf:
            best_persona = max(
                persona_perf.items(),
                key=lambda x: x[1].get('conversion_rate', 0),
                default=(None, {})
            )
            if best_persona[0]:
                recommendations.append(
                    f"{best_persona[0]} persona has highest conversion rate at "
                    f"{best_persona[1].get('conversion_rate', 0):.1f}%. Prioritize this segment."
                )

        if bandit_insights.get('exploration_exploitation_ratio', 0) > 0.3:
            recommendations.append(
                "High exploration rate detected. Consider narrowing to top-performing variants."
            )

        if worst_hooks:
            recommendations.append(
                f"Avoid hooks similar to lowest performers (CTR < {worst_hooks[0].get('ctr', 0):.1f}%)."
            )

        return recommendations

    async def _save_report(self, report_data: Dict[str, Any]) -> Optional[str]:
        try:
            async with get_async_session() as session:
                report = WeeklyLearningReport(
                    week_start=datetime.fromisoformat(report_data['week_start']),
                    week_end=datetime.fromisoformat(report_data['week_end']),
                    week_number=report_data['week_number'],
                    year=report_data['year'],
                    best_hooks=report_data['best_hooks'],
                    worst_hooks=report_data['worst_hooks'],
                    platform_performance=report_data['platform_performance'],
                    persona_performance=report_data['persona_performance'],
                    bandit_insights=report_data['bandit_insights'],
                    ctr_this_week=report_data['metrics']['this_week'].get('ctr', 0),
                    ctr_last_week=report_data['metrics']['last_week'].get('ctr', 0),
                    ctr_change_pct=report_data['metrics']['changes'].get('ctr_change_pct', 0),
                    conversions_this_week=report_data['metrics']['this_week'].get('conversions', 0),
                    conversions_last_week=report_data['metrics']['last_week'].get('conversions', 0),
                    conversions_change_pct=report_data['metrics']['changes'].get('conversions_change_pct', 0),
                    cpl_this_week=report_data['metrics']['this_week'].get('cpl', 0),
                    cpl_last_week=report_data['metrics']['last_week'].get('cpl', 0),
                    cpl_change_pct=report_data['metrics']['changes'].get('cpl_change_pct', 0),
                    recommendations=report_data['recommendations']
                )

                session.add(report)
                await session.commit()
                await session.refresh(report)

                return str(report.id)

        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return None

    async def get_latest_report(self) -> Optional[Dict[str, Any]]:
        try:
            async with get_async_session() as session:
                query = select(WeeklyLearningReport).order_by(
                    desc(WeeklyLearningReport.week_start)
                ).limit(1)

                result = await session.execute(query)
                report = result.scalar_one_or_none()

                if not report:
                    return None

                return {
                    "id": str(report.id),
                    "week_start": report.week_start.isoformat(),
                    "week_end": report.week_end.isoformat(),
                    "week_number": report.week_number,
                    "year": report.year,
                    "best_hooks": report.best_hooks,
                    "worst_hooks": report.worst_hooks,
                    "platform_performance": report.platform_performance,
                    "persona_performance": report.persona_performance,
                    "bandit_insights": report.bandit_insights,
                    "metrics": {
                        "ctr_this_week": report.ctr_this_week,
                        "ctr_last_week": report.ctr_last_week,
                        "ctr_change_pct": report.ctr_change_pct,
                        "conversions_this_week": report.conversions_this_week,
                        "conversions_last_week": report.conversions_last_week,
                        "conversions_change_pct": report.conversions_change_pct,
                        "cpl_this_week": report.cpl_this_week,
                        "cpl_last_week": report.cpl_last_week,
                        "cpl_change_pct": report.cpl_change_pct
                    },
                    "recommendations": report.recommendations,
                    "generated_at": report.generated_at.isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get latest report: {e}")
            return None


# Convenience functions for scheduler
async def generate_weekly_report() -> Dict[str, Any]:
    generator = WeeklyLearningReportGenerator()
    return await generator.generate_report()


async def get_latest_weekly_report() -> Optional[Dict[str, Any]]:
    generator = WeeklyLearningReportGenerator()
    return await generator.get_latest_report()
