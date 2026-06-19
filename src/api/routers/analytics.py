from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from ..dependencies import get_db
from ...ai_layer.learning.simulation_accuracy_tracker import (
    SimulationAccuracyTracker,
    measure_campaign_accuracy
)
from ...ai_layer.learning.governance_metrics_tracker import (
    GovernanceMetricsTracker,
    get_current_override_rate
)
from ...ai_layer.learning.weekly_learning_report import (
    WeeklyLearningReportGenerator,
    generate_weekly_report,
    get_latest_weekly_report
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/simulation-accuracy/summary", response_model=Dict[str, Any])
async def get_simulation_accuracy_summary(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Research Plan RQ2: Target >90% accuracy."""
    try:
        tracker = SimulationAccuracyTracker()
        return await tracker.get_aggregate_accuracy(days=days)
    except Exception as e:
        logger.error(f"Failed to get simulation accuracy summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation-accuracy/trend", response_model=List[Dict[str, Any]])
async def get_simulation_accuracy_trend(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    try:
        tracker = SimulationAccuracyTracker()
        return await tracker.get_accuracy_trend(days=days)
    except Exception as e:
        logger.error(f"Failed to get accuracy trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulation-accuracy/campaign/{campaign_id}", response_model=Dict[str, Any])
async def get_campaign_simulation_accuracy(
    campaign_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        return await measure_campaign_accuracy(campaign_id, measurement_type="interim")
    except Exception as e:
        logger.error(f"Failed to get campaign accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulation-accuracy/measure/{campaign_id}", response_model=Dict[str, Any])
async def measure_campaign_simulation_accuracy(
    campaign_id: str,
    measurement_type: str = Query(default="interim"),
    db: AsyncSession = Depends(get_db)
):
    try:
        return await measure_campaign_accuracy(campaign_id, measurement_type=measurement_type)
    except Exception as e:
        logger.error(f"Failed to measure accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance-metrics/override-rate", response_model=Dict[str, Any])
async def get_override_rate(
    days: int = Query(default=7, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Research Plan Section 10.2: Target < 5%."""
    try:
        tracker = GovernanceMetricsTracker()
        return await tracker.calculate_current_override_rate(days=days)
    except Exception as e:
        logger.error(f"Failed to get override rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance-metrics/safety-scores", response_model=Dict[str, Any])
async def get_safety_score_averages(
    days: int = Query(default=7, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    try:
        tracker = GovernanceMetricsTracker()
        return await tracker.calculate_safety_score_averages(days=days)
    except Exception as e:
        logger.error(f"Failed to get safety scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance-metrics/summary", response_model=Dict[str, Any])
async def get_governance_summary(
    db: AsyncSession = Depends(get_db)
):
    try:
        tracker = GovernanceMetricsTracker()
        return await tracker.get_dashboard_summary()
    except Exception as e:
        logger.error(f"Failed to get governance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance-metrics/override-trend", response_model=List[Dict[str, Any]])
async def get_override_rate_trend(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    try:
        tracker = GovernanceMetricsTracker()
        return await tracker.get_override_rate_trend(days=days)
    except Exception as e:
        logger.error(f"Failed to get override trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance-metrics/save-daily", response_model=Dict[str, Any])
async def save_daily_metrics(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    try:
        tracker = GovernanceMetricsTracker()
        record_id = await tracker.save_period_metrics("daily")
        return {
            "status": "success",
            "record_id": record_id,
            "message": "Daily governance metrics saved"
        }
    except Exception as e:
        logger.error(f"Failed to save daily metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly-report/latest", response_model=Dict[str, Any])
async def get_latest_report(
    db: AsyncSession = Depends(get_db)
):
    """Research Plan Section 10.2: Weekly Uplift Summary."""
    try:
        report = await get_latest_weekly_report()
        if not report:
            return {
                "status": "no_report",
                "message": "No weekly reports generated yet"
            }
        return report
    except Exception as e:
        logger.error(f"Failed to get latest report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weekly-report/generate", response_model=Dict[str, Any])
async def generate_report(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    try:
        report = await generate_weekly_report()
        
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])
            
        return {
            "status": "success",
            "report": report
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class CustomDateRangeRequest(BaseModel):
    start_date: str
    end_date: str


@router.post("/weekly-report/generate-custom", response_model=Dict[str, Any])
async def generate_custom_report(
    request: CustomDateRangeRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        from datetime import datetime
        
        try:
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {ve}")
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        generator = WeeklyLearningReportGenerator()
        report = await generator.generate_report(
            week_start=start_date,
            week_end=end_date
        )
        
        if not report or "error" in report:
            raise HTTPException(status_code=500, detail=report.get("error", "Failed to generate report"))
            
        return {
            "status": "success",
            "report": report
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate custom report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly-report/history", response_model=List[Dict[str, Any]])
async def get_report_history(
    limit: int = Query(default=10, ge=1, le=52),
    db: AsyncSession = Depends(get_db)
):
    try:
        from sqlalchemy import select, desc
        from ...data_layer.database.models import WeeklyLearningReport

        query = select(WeeklyLearningReport).order_by(
            desc(WeeklyLearningReport.week_start)
        ).limit(limit)

        result = await db.execute(query)
        reports = result.scalars().all()

        return [
            {
                "id": str(r.id),
                "week_start": r.week_start.isoformat(),
                "week_end": r.week_end.isoformat(),
                "week_number": r.week_number,
                "year": r.year,
                "ctr_this_week": r.ctr_this_week,
                "ctr_change_pct": r.ctr_change_pct,
                "conversions_this_week": r.conversions_this_week,
                "generated_at": r.generated_at.isoformat()
            }
            for r in reports
        ]
    except Exception as e:
        logger.error(f"Failed to get report history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/kpis", response_model=Dict[str, Any])
async def get_dashboard_kpis(
    db: AsyncSession = Depends(get_db)
):
    """RQ2 Simulation Accuracy, Human Override Rate, Safety Scores, Weekly Insights."""
    try:
        sim_tracker = SimulationAccuracyTracker()
        gov_tracker = GovernanceMetricsTracker()

        sim_accuracy = await sim_tracker.get_aggregate_accuracy(days=30)
        gov_summary = await gov_tracker.get_dashboard_summary()
        latest_report = await get_latest_weekly_report()

        return {
            "rq2_simulation_accuracy": {
                "avg_accuracy": sim_accuracy.get('avg_accuracy', 0),
                "target": 90.0,
                "pass_rate": sim_accuracy.get('rq2_pass_rate', 0),
                "status": sim_accuracy.get('rq2_status', 'UNKNOWN'),
                "total_campaigns": sim_accuracy.get('total_campaigns', 0)
            },
            "governance": gov_summary,
            "weekly_insights": {
                "has_report": latest_report is not None,
                "ctr_trend": latest_report.get('metrics', {}).get('ctr_change_pct') if latest_report else None,
                "top_recommendation": latest_report.get('recommendations', [None])[0] if latest_report else None
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance-metrics/review-time", response_model=Dict[str, Any])
async def get_review_time_metrics(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """Research Plan Section 10.2: Human Review Time Saved - quantify efficiency gains."""
    try:
        from sqlalchemy import select, func
        from ...data_layer.database.models import HITLQueue, Content

        since_date = datetime.utcnow() - timedelta(days=days)

        # Get completed reviews with timing data
        query = select(HITLQueue).where(
            HITLQueue.status == "completed",
            HITLQueue.completed_at.isnot(None),
            HITLQueue.created_at >= since_date
        )

        result = await db.execute(query)
        reviews = result.scalars().all()

        if not reviews:
            return {
                "period_days": days,
                "total_reviews": 0,
                "avg_review_duration_seconds": 0,
                "avg_review_duration_minutes": 0,
                "min_review_duration_seconds": 0,
                "max_review_duration_seconds": 0,
                "estimated_time_saved_hours": 0,
                "baseline_manual_minutes_per_content": 30,
                "message": "No completed reviews in period"
            }

        durations = []
        for review in reviews:
            if review.completed_at and review.created_at:
                # Use assigned_at if available, otherwise created_at
                start_time = review.assigned_at if review.assigned_at else review.created_at
                duration = (review.completed_at - start_time).total_seconds()
                if duration > 0:  # Filter out negative or zero durations
                    durations.append(duration)

        if not durations:
            return {
                "period_days": days,
                "total_reviews": len(reviews),
                "avg_review_duration_seconds": 0,
                "avg_review_duration_minutes": 0,
                "message": "No valid duration data available"
            }

        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        # Baseline: manual content creation ~30 min; AI generates instantly, human only reviews
        baseline_manual_minutes = 30
        baseline_manual_seconds = baseline_manual_minutes * 60

        time_saved_per_review = baseline_manual_seconds - avg_duration
        if time_saved_per_review < 0:
            time_saved_per_review = 0

        total_time_saved_seconds = time_saved_per_review * len(reviews)
        total_time_saved_hours = total_time_saved_seconds / 3600

        efficiency_gain_pct = ((baseline_manual_seconds - avg_duration) / baseline_manual_seconds * 100) if baseline_manual_seconds > 0 else 0

        approved_count = sum(1 for r in reviews if r.decision == "approve")
        approval_rate = (approved_count / len(reviews) * 100) if reviews else 0

        decision_durations = {}
        for review in reviews:
            decision = review.decision or "unknown"
            if review.completed_at and review.created_at:
                start_time = review.assigned_at if review.assigned_at else review.created_at
                duration = (review.completed_at - start_time).total_seconds()
                if duration > 0:
                    if decision not in decision_durations:
                        decision_durations[decision] = []
                    decision_durations[decision].append(duration)

        avg_by_decision = {}
        for decision, durs in decision_durations.items():
            avg_by_decision[decision] = {
                "count": len(durs),
                "avg_seconds": round(sum(durs) / len(durs), 1) if durs else 0
            }

        return {
            "period_days": days,
            "total_reviews": len(reviews),
            "avg_review_duration_seconds": round(avg_duration, 1),
            "avg_review_duration_minutes": round(avg_duration / 60, 2),
            "min_review_duration_seconds": round(min_duration, 1),
            "max_review_duration_seconds": round(max_duration, 1),
            "baseline_manual_minutes_per_content": baseline_manual_minutes,
            "estimated_time_saved_hours": round(total_time_saved_hours, 2),
            "estimated_time_saved_minutes": round(total_time_saved_seconds / 60, 1),
            "efficiency_gain_pct": round(efficiency_gain_pct, 1),
            "approval_rate": round(approval_rate, 1),
            "avg_by_decision_type": avg_by_decision,
            "status": "excellent" if efficiency_gain_pct > 80 else "good" if efficiency_gain_pct > 50 else "moderate"
        }
    except Exception as e:
        logger.error(f"Failed to get review time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
