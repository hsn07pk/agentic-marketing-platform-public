"""
Automatic calibration initialization.

Ensures the system uses data-driven parameters per research plan by auto-calibrating
from historical data if no calibrations exist.
"""

import logging
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..data_layer.database.models import PersonaCalibration, CalibrationRun
from .calibration_utils import calibrate_and_validate
import pandas as pd
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

HISTORICAL_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "historical" / "campaign_results.csv"


async def ensure_calibration_exists(db_session: AsyncSession) -> bool:
    """Ensure calibration exists; auto-calibrate from historical data if not."""
    try:
        result = await db_session.execute(
            select(PersonaCalibration).where(PersonaCalibration.is_active == True)
        )
        existing_calibrations = result.scalars().all()

        if existing_calibrations:
            logger.info(f"✅ Found {len(existing_calibrations)} active calibrations - using data-driven parameters")
            return True

        logger.warning("⚠️  No calibrations found - simulation will use hardcoded defaults")

        if not HISTORICAL_DATA_PATH.exists():
            logger.warning(f"❌ Historical data not found: {HISTORICAL_DATA_PATH}")
            logger.warning("   Simulation will use default persona parameters (NOT research-compliant)")
            return False

        logger.info("🔄 Auto-calibrating from historical data...")

        df = pd.read_csv(HISTORICAL_DATA_PATH)

        calibrations, validation_result = calibrate_and_validate(
            historical_df=df,
            train_ratio=0.7,
            random_seed=42
        )

        calib_run = CalibrationRun(
            name="Auto-Calibration from Historical Data",
            description=f"Automatic calibration from {HISTORICAL_DATA_PATH.name}",
            historical_data_source=str(HISTORICAL_DATA_PATH),
            num_training_campaigns=int(len(df) * 0.7),
            num_validation_campaigns=int(len(df) * 0.3),
            optimization_method="auto_calibration",
            status="completed",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            validation_mape=validation_result.validation_mape,
            validation_accuracy=validation_result.validation_accuracy,
            passes_threshold=validation_result.passes_threshold
        )
        db_session.add(calib_run)
        await db_session.flush()

        for calib in calibrations:
            persona_calib = PersonaCalibration(
                calibration_run_id=calib_run.id,
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

        await db_session.commit()

        logger.info(f"✅ Auto-calibration completed! MAPE: {validation_result.validation_mape:.2f}%")

        for calib in calibrations:
            logger.info(f"      • {calib.persona_name}: CTR={calib.click_prob:.4f}, Conv={calib.conversion_prob:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Auto-calibration failed: {e}", exc_info=True)
        return False


def sync_ensure_calibration_exists(db_session) -> bool:
    """Synchronous version for non-async contexts."""
    try:
        result = db_session.execute(
            select(PersonaCalibration).where(PersonaCalibration.is_active == True)
        )
        existing_calibrations = result.scalars().all()

        if existing_calibrations:
            logger.info(f"✅ Found {len(existing_calibrations)} active calibrations - using data-driven parameters")
            return True

        logger.warning("⚠️  No calibrations found - checking historical data")

        if not HISTORICAL_DATA_PATH.exists():
            logger.warning(f"❌ Historical data not found: {HISTORICAL_DATA_PATH}")
            logger.warning("   Simulation will use default persona parameters (NOT research-compliant)")
            return False

        logger.info("🔄 Auto-calibrating from historical data...")

        df = pd.read_csv(HISTORICAL_DATA_PATH)

        calibrations, validation_result = calibrate_and_validate(
            historical_df=df,
            train_ratio=0.7,
            random_seed=42
        )

        calib_run = CalibrationRun(
            name="Auto-Calibration from Historical Data",
            description=f"Automatic calibration from {HISTORICAL_DATA_PATH.name}",
            historical_data_source=str(HISTORICAL_DATA_PATH),
            num_training_campaigns=int(len(df) * 0.7),
            num_validation_campaigns=int(len(df) * 0.3),
            optimization_method="auto_calibration",
            status="completed",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            validation_mape=validation_result.validation_mape,
            validation_accuracy=validation_result.validation_accuracy,
            passes_threshold=validation_result.passes_threshold
        )
        db_session.add(calib_run)
        db_session.flush()

        for calib in calibrations:
            persona_calib = PersonaCalibration(
                calibration_run_id=calib_run.id,
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

        db_session.commit()

        logger.info(f"✅ Auto-calibration completed! MAPE: {validation_result.validation_mape:.2f}%")

        for calib in calibrations:
            logger.info(f"      • {calib.persona_name}: CTR={calib.click_prob:.4f}, Conv={calib.conversion_prob:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Auto-calibration failed: {e}", exc_info=True)
        return False
