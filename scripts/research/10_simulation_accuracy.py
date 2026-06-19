#!/usr/bin/env python3
"""
Thesis Research: Simulation Accuracy & Calibration Analysis (H1/RQ2)

This script implements the supervisor's request to validatethe simulation against real data.
It calculates MAPE (Mean Absolute Percentage Error) and generates confirmation visualizations.

Outputs:
- visualizations/fig_simulation_accuracy.png: Regression plot of Real vs Simulated
- tables/simulation_accuracy.csv: Detailed error metrics per KPI
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Add source path
sys.path.insert(0, '/app' if os.path.exists('/app/src') else str(Path(__file__).parent.parent.parent))

# Import existing calibration tools to reuse logic
try:
    from src.simulation.calibration_utils import calibrate_and_validate, CalibrationResult
except ImportError:
    # Fallback or local definition if needed
    print("Warning: Could not import src.simulation.calibration_utils")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'data'
VIZ_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'visualizations'
TABLE_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research' / 'tables'

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

def load_data():
    """Load real campaign data."""
    # Try to load existing historical results first (the 'ground truth')
    historical_path = Path('data/historical/campaign_results.csv')
    if historical_path.exists():
        return pd.read_csv(historical_path)
    
    # Fallback to extracted campaigns.json if no CSV
    json_path = OUTPUT_DIR / 'campaigns.json'
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data.get('campaigns', []))
        # Ensure numeric columns
        cols = ['impressions', 'clicks', 'conversions', 'ctr', 'budget_spent']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    
    return pd.DataFrame()

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data.get('campaigns', []))
        
        # Ensure numeric columns
        cols = ['impressions', 'clicks', 'conversions', 'ctr', 'budget_spent']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # FILTER: Prioritize REAL completed campaigns
        # 1. Try: Real + Completed + Data
        real_df = df[
            (df['status'] == 'completed') & 
            (df['is_mock'] == False) & 
            (df['impressions'] > 0)
        ]
        
        if not real_df.empty:
            print(f"Loaded {len(real_df)} valid REAL campaigns.")
            return real_df
            
        print("⚠️  No valid REAL campaigns found (with data). Checking MOCK data...")
        
        # 2. Fallback: Mock + Completed + Data
        mock_df = df[
            (df['status'] == 'completed') & 
            (df['is_mock'] == True) & 
            (df['impressions'] > 0)
        ]
        
        if not mock_df.empty:
            print(f"⚠️  Loaded {len(mock_df)} MOCK campaigns for simulation verification.")
            return mock_df

        print("❌ No valid campaigns found (Real or Mock).")
        return pd.DataFrame()

    return pd.DataFrame()

def run_accuracy_analysis(df):
    """Run calibration validation and calculate MAPE."""
    if df.empty:
        print("No data to analyze.")
        return

    print(f"Running accuracy analysis on {len(df)} campaigns...")

    # We use the existing calibration utility to split Train/Test and validate
    # This ensures we adhere to the methodology defined in the specificatoins
    calibrations, val_result = calibrate_and_validate(
        historical_df=df,
        train_ratio=0.7,
        random_seed=42
    )

    # Extract validation results (Real vs Simulated) from the hold-out set
    results = val_result.per_campaign_results
    if not results:
        print("No validation results generated.")
        return

    val_df = pd.DataFrame(results)
    
    # 1. Generate Table
    generate_mape_table(val_result)

    # 2. Generate Visualization
    generate_accuracy_plot(val_df)

def generate_mape_table(val_result):
    """Generate CSV table for MAPE metrics."""
    metrics = {
        'Metric': ['Clicks', 'Conversions', 'Overall Accuracy'],
        'MAPE (%)': [
            f"{val_result.per_metric_mape.get('clicks', 0):.2f}",
            f"{val_result.per_metric_mape.get('conversions', 0):.2f}",
            f"{val_result.validation_accuracy * 100:.2f}"
        ],
        'Target Met': [
            "YES" if val_result.per_metric_mape.get('clicks', 100) < 15 else "NO",
            "YES" if val_result.per_metric_mape.get('conversions', 100) < 20 else "NO",
            "YES" if val_result.passes_threshold else "NO"
        ]
    }
    
    df = pd.DataFrame(metrics)
    csv_path = TABLE_DIR / 'simulation_accuracy.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✅ simulation_accuracy.csv (Overall MAPE: {val_result.validation_mape:.2f}%)")

def generate_accuracy_plot(val_df):
    """Generate Simulated vs Real regression plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter Plot: Clicks
    real_clicks = val_df['actual_clicks']
    sim_clicks = val_df['simulated_clicks']
    
    # Plot Clicks
    axes[0].scatter(real_clicks, sim_clicks, alpha=0.6, color=COLORS[0], label='Campaigns')
    
    # Perfect alignment line
    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),
    ]
    axes[0].plot(lims, lims, 'r--', alpha=0.75, label='Perfect Match (y=x)')
    
    # Regression Stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(real_clicks, sim_clicks)
    r_squared = r_value**2
    
    x_vals = np.array(real_clicks)
    axes[0].plot(x_vals, intercept + slope * x_vals, color='black', alpha=0.3, 
                 label=f'Fit (R²={r_squared:.2f})')

    axes[0].set_xlabel('Real Clicks (Hold-out Set)')
    axes[0].set_ylabel('Simulated Clicks')
    axes[0].set_title(f'Click Simulation Accuracy\nMAPE: {np.mean(np.abs((real_clicks - sim_clicks) / real_clicks)) * 100:.1f}%')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter Plot: Conversions
    real_conv = val_df['actual_conversions']
    sim_conv = val_df['simulated_conversions']

    axes[1].scatter(real_conv, sim_conv, alpha=0.6, color=COLORS[1], label='Campaigns')
    
    lims_c = [
        np.min([axes[1].get_xlim(), axes[1].get_ylim()]),
        np.max([axes[1].get_xlim(), axes[1].get_ylim()]),
    ]
    axes[1].plot(lims_c, lims_c, 'r--', alpha=0.75, label='Perfect Match (y=x)')
    
    slope_c, intercept_c, r_value_c, p_value_c, std_err_c = stats.linregress(real_conv, sim_conv)
    x_vals_c = np.array(real_conv)
    axes[1].plot(x_vals_c, intercept_c + slope_c * x_vals_c, color='black', alpha=0.3,
                 label=f'Fit (R²={r_value_c**2:.2f})')

    axes[1].set_xlabel('Real Conversions (Hold-out Set)')
    axes[1].set_ylabel('Simulated Conversions')
    axes[1].set_title(f'Conversion Simulation Accuracy\nMAPE: {np.mean(np.abs((real_conv - sim_conv) / real_conv)) * 100:.1f}%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Simulation Accuracy Validation (H1)\nBased on {len(val_df)} hold-out campaigns', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = VIZ_DIR / 'fig_simulation_accuracy.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {output_path.name}")

if __name__ == "__main__":
    print("=" * 60)
    print("THESIS RESEARCH: Simulation Accuracy (Requests H1/RQ2)")
    print("=" * 60)
    
    try:
        # Create output dirs if they don't exist
        for d in [OUTPUT_DIR, VIZ_DIR, TABLE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        df = load_data()
        if not df.empty:
            run_accuracy_analysis(df)
            print("\n✅ Accuracy analysis complete!")
        else:
            print("⚠️  No data found to analyze.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
