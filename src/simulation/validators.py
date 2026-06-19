"""Validation utilities for simulation accuracy."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class SimulationValidator:
    """Validates simulation accuracy against historical data."""
    
    @staticmethod
    def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error as percentage."""
        mask = actual != 0
        if not mask.any():
            return 100.0
        
        actual_filtered = actual[mask]
        predicted_filtered = predicted[mask]
        
        mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
        return mape
    
    @staticmethod
    def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def calculate_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
        if len(actual) < 2:
            return 0.0
        
        correlation, _ = stats.pearsonr(actual, predicted)
        return correlation
    
    @staticmethod
    def calculate_accuracy_score(mape: float) -> float:
        """Convert MAPE to accuracy score [0-1]: Accuracy = 1 - (MAPE / 100)."""
        return max(0.0, 1.0 - (mape / 100))
    
    @staticmethod
    def validate_metrics(
        simulated_metrics: Dict[str, List[float]],
        actual_metrics: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Validate multiple metrics, returning validation results per metric."""
        results = {}
        
        for metric_name in simulated_metrics.keys():
            if metric_name not in actual_metrics:
                continue
            
            sim = np.array(simulated_metrics[metric_name])
            act = np.array(actual_metrics[metric_name])
            
            # Ensure same length
            min_len = min(len(sim), len(act))
            sim = sim[:min_len]
            act = act[:min_len]
            
            if len(sim) == 0:
                continue
            
            results[metric_name] = {
                'mape': SimulationValidator.calculate_mape(act, sim),
                'rmse': SimulationValidator.calculate_rmse(act, sim),
                'correlation': SimulationValidator.calculate_correlation(act, sim),
                'accuracy': SimulationValidator.calculate_accuracy_score(
                    SimulationValidator.calculate_mape(act, sim)
                )
            }
        
        return results
    
    @staticmethod
    def calculate_overall_accuracy(validation_results: Dict[str, Dict[str, float]]) -> float:
        """Calculate weighted overall accuracy (impressions:0.2, clicks:0.3, conversions:0.4, ctr:0.1)."""
        weights = {
            'impressions': 0.2,
            'clicks': 0.3,
            'conversions': 0.4,
            'ctr': 0.1
        }
        
        weighted_accuracy = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in validation_results:
                accuracy = validation_results[metric].get('accuracy', 0.0)
                weighted_accuracy += accuracy * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_accuracy / total_weight
    
    @staticmethod
    def validate_distribution(
        simulated: np.ndarray,
        actual: np.ndarray,
        test: str = 'ks'
    ) -> Tuple[float, float]:
        """Test if simulated and actual distributions match. Returns (statistic, p_value)."""
        if test == 'ks':
            statistic, p_value = stats.ks_2samp(simulated, actual)
        elif test == 'ttest':
            statistic, p_value = stats.ttest_ind(simulated, actual)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        return statistic, p_value
    
    @staticmethod
    def check_bias(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Check for systematic bias in predictions."""
        errors = predicted - actual
        
        return {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'mean_abs_error': np.mean(np.abs(errors)),
            'bias_direction': 'overestimate' if np.mean(errors) > 0 else 'underestimate'
        }
    
    @staticmethod
    def validate_temporal_patterns(
        simulated_series: pd.Series,
        actual_series: pd.Series
    ) -> Dict[str, float]:
        """Validate temporal pattern similarity (trend, autocorrelation)."""
        results = {}
        
        sim_trend = np.polyfit(range(len(simulated_series)), simulated_series, 1)[0]
        act_trend = np.polyfit(range(len(actual_series)), actual_series, 1)[0]
        
        results['trend_similarity'] = 1 - abs(sim_trend - act_trend) / max(abs(act_trend), 1)
        
        if len(simulated_series) > 2:
            sim_autocorr = simulated_series.autocorr(lag=1)
            act_autocorr = actual_series.autocorr(lag=1)
            
            results['autocorr_similarity'] = 1 - abs(sim_autocorr - act_autocorr)
        else:
            results['autocorr_similarity'] = 0.0
        
        return results
    
    @staticmethod
    def generate_validation_report(
        validation_results: Dict[str, Dict[str, float]],
        threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        overall_accuracy = SimulationValidator.calculate_overall_accuracy(validation_results)
        
        passed = overall_accuracy >= threshold
        
        worst_metric = None
        worst_accuracy = 1.0
        
        for metric, results in validation_results.items():
            accuracy = results.get('accuracy', 0.0)
            if accuracy < worst_accuracy:
                worst_accuracy = accuracy
                worst_metric = metric
        
        report = {
            'overall_accuracy': overall_accuracy,
            'passed': passed,
            'threshold': threshold,
            'metrics': validation_results,
            'worst_metric': worst_metric,
            'worst_accuracy': worst_accuracy,
            'summary': _generate_summary(overall_accuracy, passed, threshold)
        }
        
        return report

def _generate_summary(accuracy: float, passed: bool, threshold: float) -> str:
    """Generate human-readable summary"""
    accuracy_pct = accuracy * 100
    threshold_pct = threshold * 100
    
    if passed:
        return (
            f"✅ Validation PASSED with {accuracy_pct:.1f}% accuracy "
            f"(threshold: {threshold_pct:.0f}%)"
        )
    else:
        gap = threshold_pct - accuracy_pct
        return (
            f"❌ Validation FAILED with {accuracy_pct:.1f}% accuracy "
            f"(threshold: {threshold_pct:.0f}%, gap: {gap:.1f}%)"
        )

class CampaignValidator:
    """Validates individual campaigns against constraints."""
    
    @staticmethod
    def validate_campaign_config(config: Dict) -> Tuple[bool, List[str]]:
        """Validate campaign configuration. Returns (is_valid, errors)."""
        errors = []
        
        required = ['platform', 'budget', 'duration']
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        if 'budget' in config:
            if config['budget'] <= 0:
                errors.append("Budget must be positive")
            if config['budget'] > 1000000:
                errors.append("Budget exceeds maximum (1M)")
        
        if 'duration' in config:
            if config['duration'] <= 0:
                errors.append("Duration must be positive")
            if config['duration'] > 365:
                errors.append("Duration exceeds maximum (365 days)")
        
        if 'platform' in config:
            valid_platforms = ['linkedin', 'twitter', 'email']
            if config['platform'] not in valid_platforms:
                errors.append(f"Invalid platform. Must be one of: {valid_platforms}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_targeting(targeting: Dict) -> Tuple[bool, List[str]]:
        errors = []
        
        if 'persona' in targeting:
            valid_personas = ['decision_maker', 'influencer', 'researcher', 'all']
            if targeting['persona'] not in valid_personas:
                errors.append(f"Invalid persona. Must be one of: {valid_personas}")
        
        return len(errors) == 0, errors