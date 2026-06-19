import logging
import asyncio
from typing import Dict, List, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import io
from datetime import datetime
import time

from ..dependencies import get_db
from ...data_layer.database.models import CalibrationRun, PersonaCalibration
from ...simulation.calibration_utils import calibrate_and_validate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["calibration"])


async def run_calibration_task(
    historical_csv_content: bytes,
    calibration_run_id: UUID,
    db_session: AsyncSession,
    method: str = "legacy"  # Default to data-driven method for MAPE <10%
):
    start_time = time.time()

    try:
        df = pd.read_csv(io.BytesIO(historical_csv_content))

        logger.info(f"Loaded {len(df)} historical campaigns for calibration")
        logger.info(f"Using calibration method: {method}")

        # Run calibration in thread pool to avoid blocking event loop
        # This is CRITICAL for keeping the API responsive during long calibrations
        logger.info("Running calibration in thread pool (non-blocking)...")
        calibrations, validation_result = await asyncio.to_thread(
            calibrate_and_validate,
            historical_df=df,
            train_ratio=0.7,
            random_seed=42,
            method=method
        )

        for calib in calibrations:
            persona_calib = PersonaCalibration(
                calibration_run_id=calibration_run_id,
                persona_name=calib.persona_name,
                daily_active_prob=calib.daily_active_prob,
                click_prob=calib.click_prob,
                conversion_prob=calib.conversion_prob,
                content_engagement_prob=calib.content_engagement_prob,
                share_prob=calib.share_prob,
                training_mape=calib.training_mape,
                num_training_samples=calib.num_training_samples,
                is_active=False  # Not active until manually activated
            )
            db_session.add(persona_calib)

        result = await db_session.execute(
            select(CalibrationRun).where(CalibrationRun.id == calibration_run_id)
        )
        calib_run = result.scalar_one()

        calib_run.status = "completed"
        calib_run.completed_at = datetime.utcnow()
        calib_run.duration_seconds = time.time() - start_time
        calib_run.validation_mape = validation_result.validation_mape
        calib_run.validation_accuracy = validation_result.validation_accuracy
        calib_run.passes_threshold = validation_result.passes_threshold

        await db_session.commit()

        logger.info(f"Calibration {calibration_run_id} completed: MAPE={validation_result.validation_mape:.2f}%, {len(calibrations)} personas calibrated")

    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)

        result = await db_session.execute(
            select(CalibrationRun).where(CalibrationRun.id == calibration_run_id)
        )
        calib_run = result.scalar_one_or_none()
        if calib_run:
            calib_run.status = "failed"
            calib_run.error_message = str(e)
            await db_session.commit()


@router.post("/upload")
async def upload_historical_data(
    file: UploadFile = File(...),
    name: str = "Historical Data Calibration",
    method: str = "legacy",  # Default to data-driven method for MAPE <10%
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Upload historical campaign CSV and start calibration.

    **Required CSV columns:**
    - campaign_id, platform, target_persona, duration_days
    - budget_total, impressions, clicks, conversions, ctr, cpl

    **Calibration Methods:**
    - `auto`: Automatically selects best method based on data size (recommended)
    - `adaptive`: Same as auto, uses LOOCV/GP/DE based on data
    - `hierarchical`: Bayesian hierarchical - borrows strength across personas
    - `ensemble`: Multiple calibration runs averaged for robustness
    - `legacy`: Original differential evolution (for comparison)

    **Process:**
    1. Upload your real Agentic historical data CSV
    2. System selects optimal calibration strategy
    3. For small data (<20): Uses LOOCV with Bayesian regularization
    4. For medium data (20-100): Uses K-fold CV with GP surrogate
    5. For large data (100+): Uses standard differential evolution
    6. Validates and saves calibrated parameters

    **Research Plan Target:** MAPE < 10% (Accuracy > 90%)
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))

        column_mapping = {
            'persona': 'target_persona',
            'budget_spent': 'budget_total'
        }
        df.rename(columns=column_mapping, inplace=True)

        required_cols = [
            'campaign_id', 'platform', 'target_persona', 'duration_days',
            'budget_total', 'impressions', 'clicks', 'conversions', 'ctr'
        ]

        missing = set(required_cols) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
            )

        calib_run = CalibrationRun(
            name=name,
            description=f"Calibration from {file.filename}",
            historical_data_source=file.filename,
            num_training_campaigns=int(len(df) * 0.7),
            num_validation_campaigns=int(len(df) * 0.3),
            optimization_method="differential_evolution",
            status="running"
        )

        db.add(calib_run)
        await db.commit()
        await db.refresh(calib_run)

        logger.info(f"Created calibration run {calib_run.id}")

        if background_tasks:
            background_tasks.add_task(
                run_calibration_task,
                content,
                calib_run.id,
                db,
                method
            )

        method_descriptions = {
            "auto": "Adaptive (LOOCV/GP/DE based on data size)",
            "adaptive": "Adaptive (LOOCV/GP/DE based on data size)",
            "hierarchical": "Hierarchical Bayesian (borrows strength across personas)",
            "ensemble": "Ensemble (multiple runs averaged)",
            "legacy": "Legacy differential evolution"
        }

        return {
            "calibration_run_id": str(calib_run.id),
            "status": "running",
            "message": f"Calibration started with {len(df)} campaigns",
            "method": method,
            "method_description": method_descriptions.get(method, "Unknown"),
            "training_campaigns": int(len(df) * 0.7),
            "validation_campaigns": int(len(df) * 0.3),
            "check_status_url": f"/api/v1/calibration/{calib_run.id}"
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        logger.error(f"Error processing CSV: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


@router.get("/{calibration_run_id}")
async def get_calibration_status(
    calibration_run_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:

    result = await db.execute(
        select(CalibrationRun).where(CalibrationRun.id == calibration_run_id)
    )
    calib_run = result.scalar_one_or_none()

    if not calib_run:
        raise HTTPException(status_code=404, detail="Calibration run not found")

    persona_result = await db.execute(
        select(PersonaCalibration).where(
            PersonaCalibration.calibration_run_id == calibration_run_id
        )
    )
    persona_calibs = persona_result.scalars().all()

    return {
        "id": str(calib_run.id),
        "name": calib_run.name,
        "status": calib_run.status,
        "started_at": calib_run.started_at.isoformat() if calib_run.started_at else None,
        "completed_at": calib_run.completed_at.isoformat() if calib_run.completed_at else None,
        "duration_seconds": calib_run.duration_seconds,
        "num_training_campaigns": calib_run.num_training_campaigns,
        "num_validation_campaigns": calib_run.num_validation_campaigns,
        "validation_mape": calib_run.validation_mape,
        "validation_accuracy": calib_run.validation_accuracy,
        "passes_threshold": calib_run.passes_threshold,
        "target": "MAPE < 10% (Accuracy > 90%)",
        "error_message": calib_run.error_message,
        "persona_calibrations": [
            {
                "persona_name": pc.persona_name,
                "daily_active_prob": pc.daily_active_prob,
                "click_prob": pc.click_prob,
                "conversion_prob": pc.conversion_prob,
                "training_mape": pc.training_mape,
                "is_active": pc.is_active
            }
            for pc in persona_calibs
        ]
    }


@router.get("/")
async def list_calibration_runs(
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:

    result = await db.execute(
        select(CalibrationRun)
        .order_by(CalibrationRun.started_at.desc())
        .limit(limit)
    )
    calib_runs = result.scalars().all()

    return [
        {
            "id": str(cr.id),
            "name": cr.name,
            "status": cr.status,
            "started_at": cr.started_at.isoformat() if cr.started_at else None,
            "validation_mape": cr.validation_mape,
            "validation_accuracy": cr.validation_accuracy,
            "passes_threshold": cr.passes_threshold,
            "num_training_campaigns": cr.num_training_campaigns,
            "num_validation_campaigns": cr.num_validation_campaigns
        }
        for cr in calib_runs
    ]


@router.get("/personas/active")
async def get_active_calibrations(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:

    result = await db.execute(
        select(PersonaCalibration).where(PersonaCalibration.is_active == True)
    )
    calibrations = result.scalars().all()

    if not calibrations:
        return {
            "has_calibrations": False,
            "message": "No calibrations active. Using default persona parameters.",
            "personas": []
        }

    return {
        "has_calibrations": True,
        "message": f"Using calibrated parameters for {len(calibrations)} personas",
        "personas": [
            {
                "persona_name": pc.persona_name,
                "calibration_id": str(pc.id),
                "daily_active_prob": pc.daily_active_prob,
                "click_prob": pc.click_prob,
                "conversion_prob": pc.conversion_prob,
                "training_mape": pc.training_mape,
                "calibrated_at": pc.created_at.isoformat()
            }
            for pc in calibrations
        ]
    }


@router.post("/personas/{persona_calibration_id}/activate")
async def activate_persona_calibration(
    persona_calibration_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:

    result = await db.execute(
        select(PersonaCalibration).where(PersonaCalibration.id == persona_calibration_id)
    )
    calibration = result.scalar_one_or_none()

    if not calibration:
        raise HTTPException(status_code=404, detail="Persona calibration not found")

    deactivate_result = await db.execute(
        select(PersonaCalibration).where(
            PersonaCalibration.persona_name == calibration.persona_name,
            PersonaCalibration.is_active == True
        )
    )
    for pc in deactivate_result.scalars():
        pc.is_active = False

    calibration.is_active = True

    await db.commit()

    logger.info(f"Activated calibration {persona_calibration_id} for {calibration.persona_name}")

    return {
        "status": "activated",
        "persona_name": calibration.persona_name,
        "calibration_id": str(calibration.id)
    }
