"""
Simulation Validation Script - Section 5.3 of Research Plan

Validates simulation-to-live accuracy by comparing simulated results
against historical campaign data. Target: >90% accuracy (MAPE <10%)

Usage:
    python scripts/validate_simulation.py --data data/historical/campaign_results.csv --output validation_report.json
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.simulation.environment import MarketingEnvironment, SimulationConfig
from src.simulation.validators import SimulationValidator
from src.simulation.calibration_utils import calibrate_and_validate
from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: Calibration functions moved to src/simulation/calibration_utils.py
# This script now uses the shared calibration utilities for consistency


def main():
    parser = argparse.ArgumentParser(description='Validate simulation accuracy')
    parser.add_argument(
        '--data',
        type=str,
        default='data/historical/campaign_results.csv',
        help='Path to historical campaign data CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validation_report.json',
        help='Output path for validation report JSON'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='Minimum number of campaigns required for validation'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("SIMULATION VALIDATION - Section 5.3")
    logger.info("Target: >90% accuracy (MAPE <10%)")
    logger.info("="*60)

    # Load historical data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Historical data not found: {data_path}")
        logger.error("Please provide anonymized campaign data with columns:")
        logger.error("  - campaign_id, platform, persona, date")
        logger.error("  - impressions, clicks, conversions, ctr, cpl, budget_spent")
        return 1

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} campaigns from {data_path}")

    # Check minimum sample size
    if len(df) < args.min_samples:
        logger.warning(f"Insufficient data: {len(df)} campaigns < {args.min_samples} minimum")
        logger.warning("Validation requires at least 10 campaigns for statistical significance")
        logger.warning("Results may not be reliable with limited data")

    # Run calibration and validation using shared utility
    logger.info("\n" + "="*60)
    logger.info("CALIBRATION & VALIDATION (using shared calibration_utils)")
    logger.info("="*60)

    calibrations, validation_result = calibrate_and_validate(
        historical_df=df,
        train_ratio=0.7,
        random_seed=42
    )

    # Format results for compatibility with original output format
    validation_results = {
        'overall': {
            'mape': validation_result.validation_mape,
            'accuracy': validation_result.validation_accuracy,
            'target_met': validation_result.passes_threshold,
            'worst_metric': 'clicks' if validation_result.per_metric_mape.get('clicks', 0) > validation_result.per_metric_mape.get('conversions', 0) else 'conversions',
            'worst_accuracy': 100 - max(validation_result.per_metric_mape.values()) if validation_result.per_metric_mape else 0,
            'summary': f"Validation {'PASSED' if validation_result.passes_threshold else 'FAILED'} - MAPE: {validation_result.validation_mape:.2f}%"
        },
        'metrics': {
            metric: {
                'mape': mape,
                'accuracy': 100 - mape
            }
            for metric, mape in validation_result.per_metric_mape.items()
        },
        'per_campaign': validation_result.per_campaign_results
    }

    # Store calibrated parameters
    calibrated_params = {
        'calibrations': [
            {
                'persona_name': calib.persona_name,
                'daily_active_prob': calib.daily_active_prob,
                'click_prob': calib.click_prob,
                'conversion_prob': calib.conversion_prob,
                'training_mape': calib.training_mape,
                'num_samples': calib.num_training_samples
            }
            for calib in calibrations
        ]
    }

    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Overall MAPE: {validation_results['overall']['mape']:.2f}%")
    logger.info(f"Overall Accuracy: {validation_results['overall']['accuracy']:.2f}%")
    logger.info(f"Target Met (>90%): {'✅ YES' if validation_results['overall']['target_met'] else '❌ NO'}")

    # Save report
    output_path = Path(args.output)

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            # Handle NaN
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    # Calculate train/test split sizes
    n_train = int(len(df) * 0.7)
    n_test = len(df) - n_train

    report = {
        'timestamp': datetime.now().isoformat(),
        'data_source': str(data_path),
        'n_campaigns': int(len(df)),
        'n_train': n_train,
        'n_test': n_test,
        'calibrated_params': convert_numpy(calibrated_params),
        'validation_results': convert_numpy(validation_results)
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nValidation report saved to: {output_path}")

    # Exit code
    if validation_results['overall']['target_met']:
        logger.info("\n✅ Simulation validation PASSED - Ready for deployment")
        return 0
    else:
        logger.warning("\n⚠️ Simulation validation FAILED - Accuracy below 90% threshold")
        logger.warning("Recommendation: Collect more historical data or refine simulation parameters")
        return 1


if __name__ == "__main__":
    sys.exit(main())
