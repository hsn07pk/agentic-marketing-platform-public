"""
Research Plan Reference: Section 10.2 - "automated weekly learning report"
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_scheduler_task: Optional[asyncio.Task] = None
_scheduler_running: bool = False


async def generate_weekly_report_task():
    try:
        from ...ai_layer.learning.weekly_learning_report import (
            generate_weekly_report,
            save_weekly_report
        )
        from ...data_layer.database.connection import get_async_session
        
        logger.info("🗓️ Starting automated weekly report generation...")
        
        async with get_async_session() as session:
            report = await generate_weekly_report(session)
            
            if report:
                await save_weekly_report(session, report)
                logger.info(f"✅ Weekly report generated successfully: {report.get('report_id', 'unknown')}")
                return report
            else:
                logger.warning("⚠️ Weekly report generation returned empty result")
                return None
                
    except Exception as e:
        logger.error(f"❌ Failed to generate weekly report: {e}", exc_info=True)
        return None


def _get_next_monday_9am() -> datetime:
    now = datetime.now()
    
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0 and now.hour >= 9:
        days_until_monday = 7  # Already past 9am Monday, schedule for next week
    
    next_monday = now + timedelta(days=days_until_monday)
    next_monday_9am = next_monday.replace(hour=9, minute=0, second=0, microsecond=0)
    
    return next_monday_9am


async def _scheduler_loop():
    global _scheduler_running
    
    logger.info("📅 Weekly report scheduler started")
    
    while _scheduler_running:
        try:
            next_run = _get_next_monday_9am()
            now = datetime.now()
            wait_seconds = (next_run - now).total_seconds()
            
            if wait_seconds > 0:
                logger.info(f"⏰ Next weekly report scheduled for: {next_run.isoformat()}")
                logger.info(f"   Waiting {wait_seconds / 3600:.1f} hours...")
                
                while wait_seconds > 0 and _scheduler_running:
                    sleep_time = min(wait_seconds, 3600)
                    await asyncio.sleep(sleep_time)
                    wait_seconds -= sleep_time
            
            if not _scheduler_running:
                break
            
            logger.info("⏰ Scheduled time reached - generating weekly report...")
            await generate_weekly_report_task()
            
            # Avoid duplicate runs within the same minute
            await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            logger.info("📅 Weekly report scheduler cancelled")
            break
        except Exception as e:
            logger.error(f"❌ Scheduler error: {e}", exc_info=True)
            await asyncio.sleep(300)


async def start_weekly_report_scheduler():
    global _scheduler_task, _scheduler_running
    
    if _scheduler_running:
        logger.warning("Weekly report scheduler is already running")
        return
    
    _scheduler_running = True
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    
    logger.info("✅ Weekly report scheduler started")


async def stop_weekly_report_scheduler():
    global _scheduler_task, _scheduler_running
    
    _scheduler_running = False
    
    if _scheduler_task:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None
    
    logger.info("🛑 Weekly report scheduler stopped")


async def trigger_manual_report():
    logger.info("🔧 Manual weekly report generation triggered")
    return await generate_weekly_report_task()


def get_scheduler_status() -> dict:
    next_run = _get_next_monday_9am()
    
    return {
        "running": _scheduler_running,
        "next_scheduled_run": next_run.isoformat() if _scheduler_running else None,
        "schedule": "Every Monday at 9:00 AM",
        "task_active": _scheduler_task is not None and not _scheduler_task.done()
    }
