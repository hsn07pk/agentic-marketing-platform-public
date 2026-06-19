"""
Automated Calibration Scheduler

IMPORTANT: Calibration runs as BACKGROUND TASK to avoid blocking API requests.
Each background calibration creates its own database session to maintain isolation.
"""

import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import json

from ..api.dependencies import get_db
from ..data_layer.database.models import CalibrationRun, PersonaCalibration
from .calibration_utils import calibrate_and_validate
from ..config.settings import settings

logger = logging.getLogger(__name__)

_last_file_state: Dict[str, any] = {}


class CalibrationScheduler:
    """Automated calibration scheduler that runs in background."""

    def __init__(self, historical_data_path: str = "data/historical/campaign_results.csv"):
        self.historical_data_path = Path(historical_data_path)
        self.calibration_interval_hours = 24
        self.calibration_hour = 2
        self.running = False

        self._last_file_hash: Optional[str] = None
        self._last_row_count: Optional[int] = None
        self._last_data_checksum: Optional[str] = None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file contents."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _calculate_data_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum of dataframe content (ignoring whitespace differences)."""
        data_str = df.to_csv(index=False).replace(' ', '')
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_file_state(self) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Get current file state: (hash, row_count, data_checksum)."""
        if not self.historical_data_path.exists():
            return None, None, None

        try:
            file_hash = self._calculate_file_hash(self.historical_data_path)
            df = pd.read_csv(self.historical_data_path)
            row_count = len(df)
            data_checksum = self._calculate_data_checksum(df)
            return file_hash, row_count, data_checksum
        except Exception as e:
            logger.error(f"Error reading file state: {e}")
            return None, None, None

    def _has_file_changed(self) -> Tuple[bool, str]:
        """Check if historical data file has changed. Returns (changed, reason)."""
        current_hash, current_rows, current_checksum = self._get_file_state()

        if current_hash is None:
            return False, "File not found"

        if self._last_file_hash is None:
            self._last_file_hash = current_hash
            self._last_row_count = current_rows
            self._last_data_checksum = current_checksum
            return False, "Initial state recorded"

        reasons = []

        if current_hash != self._last_file_hash:
            reasons.append(f"file_hash changed")

        if current_rows != self._last_row_count:
            reasons.append(f"row_count: {self._last_row_count} → {current_rows}")

        if current_checksum != self._last_data_checksum:
            reasons.append(f"data_checksum changed")

        if reasons:
            self._last_file_hash = current_hash
            self._last_row_count = current_rows
            self._last_data_checksum = current_checksum
            return True, "; ".join(reasons)

        return False, "No changes detected"

    async def should_run_calibration(self, db_session) -> Tuple[bool, str]:
        """Check if calibration should run. Returns (should_run, reason)."""
        from sqlalchemy import select, func

        result = await db_session.execute(
            select(func.count(CalibrationRun.id))
        )
        count = result.scalar()

        if count == 0:
            logger.info("🔍 No calibrations found - triggering initial calibration")
            return True, "No existing calibrations"

        file_changed, change_reason = self._has_file_changed()
        if file_changed:
            logger.info(f"📊 Historical data changed: {change_reason}")
            return True, change_reason

        result = await db_session.execute(
            select(CalibrationRun)
            .order_by(CalibrationRun.started_at.desc())
            .limit(1)
        )
        last_calibration = result.scalar_one_or_none()

        if last_calibration and self.historical_data_path.exists():
            file_modified = datetime.fromtimestamp(self.historical_data_path.stat().st_mtime)
            if file_modified > last_calibration.started_at:
                logger.info(f"📊 File mtime changed ({file_modified} > {last_calibration.started_at})")
                return True, f"File modified: {file_modified}"

        return False, "No changes detected"

    async def _run_calibration_with_session(self) -> Optional[str]:
        """Create an isolated DB session for background calibration task."""
        async for db_session in get_db():
            try:
                return await self.run_calibration(db_session)
            finally:
                pass

    async def run_calibration(self, db_session) -> Optional[str]:
        """
        Run full calibration and store results in database.

        NOTE: Long-running (5-10 min). Runs in thread pool to avoid blocking event loop.
        """
        try:
            logger.info("🚀 AUTOMATED CALIBRATION STARTED (Background Thread)")
            logger.info("⚠️  This will take 5-10 minutes - APIs remain operational")

            if not self.historical_data_path.exists():
                logger.error(f"Historical data not found: {self.historical_data_path}")
                return None

            df = pd.read_csv(self.historical_data_path)
            logger.info(f"Loaded {len(df)} campaigns from {self.historical_data_path}")

            # 'legacy' method: data-driven calibration achieving MAPE <10% (Research Plan RQ2)
            calibrations, validation_result = await asyncio.to_thread(
                calibrate_and_validate,
                historical_df=df,
                train_ratio=0.7,
                random_seed=42,
                method='legacy'
            )

            train_count = int(len(df) * 0.7)
            val_count = len(df) - train_count

            calibration_run = CalibrationRun(
                name=f"Automated Calibration {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                description="Automated calibration triggered by data-driven scheduler",
                historical_data_source=str(self.historical_data_path),
                num_training_campaigns=train_count,
                num_validation_campaigns=val_count,
                validation_mape=validation_result.validation_mape,
                validation_accuracy=validation_result.validation_accuracy,
                passes_threshold=validation_result.passes_threshold,
                metrics={
                    'per_metric_mape': validation_result.per_metric_mape,
                    'per_campaign_results': validation_result.per_campaign_results,
                    'threshold': 10.0,
                    'automated': True,
                    'total_campaigns': len(df)
                },
                optimization_method='grid_search',
                random_seed=42,
                status="completed"
            )

            db_session.add(calibration_run)
            await db_session.flush()

            for calib in calibrations:
                persona_calib = PersonaCalibration(
                    calibration_run_id=calibration_run.id,
                    persona_name=calib.persona_name,
                    daily_active_prob=calib.daily_active_prob,
                    click_prob=calib.click_prob,
                    conversion_prob=calib.conversion_prob,
                    content_engagement_prob=calib.content_engagement_prob,
                    share_prob=calib.share_prob,
                    training_mape=calib.training_mape,
                    num_training_samples=calib.num_training_samples,
                    is_active=True
                )
                db_session.add(persona_calib)

            calibration_run.completed_at = datetime.utcnow()
            calibration_run.duration_seconds = (calibration_run.completed_at - calibration_run.started_at).total_seconds()

            await db_session.commit()

            logger.info(f"CALIBRATION COMPLETE: Run ID = {calibration_run.id}")
            logger.info(f"  Validation MAPE: {validation_result.validation_mape:.2f}%")
            logger.info(f"  Threshold Met: {'✅ YES' if validation_result.passes_threshold else '❌ NO'}")
            logger.info(f"  Personas Calibrated: {len(calibrations)}")

            return str(calibration_run.id)

        except Exception as e:
            logger.error(f"Calibration failed: {e}", exc_info=True)
            await db_session.rollback()
            return None

    async def run_forever(self):
        """Main scheduler loop - checks for data changes every 5 minutes."""
        self.running = True
        check_interval_seconds = 300

        logger.info("📊 CALIBRATION SCHEDULER STARTED")
        logger.info(f"   Check interval: Every {check_interval_seconds // 60} minutes")

        initial_hash, initial_rows, _ = self._get_file_state()
        if initial_hash:
            logger.info(f"   Initial file state: {initial_rows} rows, hash={initial_hash[:12]}...")

        while self.running:
            try:
                async for db_session in get_db():
                    should_run, reason = await self.should_run_calibration(db_session)

                    if should_run:
                        logger.info(f"🎯 TRIGGERING CALIBRATION: {reason}")

                        asyncio.create_task(self._run_calibration_with_session())
                        break

                await asyncio.sleep(check_interval_seconds)

            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(60)

    def stop(self):
        logger.info("Stopping calibration scheduler")
        self.running = False


_scheduler: Optional[CalibrationScheduler] = None


async def start_calibration_scheduler():
    """Start the global calibration scheduler."""
    global _scheduler

    if _scheduler is None:
        _scheduler = CalibrationScheduler()
        asyncio.create_task(_scheduler.run_forever())
        logger.info("Calibration scheduler initialized")


async def stop_calibration_scheduler():
    """Stop the global calibration scheduler."""
    global _scheduler

    if _scheduler:
        _scheduler.stop()
        _scheduler = None


async def trigger_calibration_now():
    """Manually trigger calibration. Returns calibration run ID if successful."""
    global _scheduler

    if _scheduler is None:
        _scheduler = CalibrationScheduler()

    async for db_session in get_db():
        return await _scheduler.run_calibration(db_session)
