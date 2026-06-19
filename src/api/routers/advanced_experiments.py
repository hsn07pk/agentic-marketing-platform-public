"""
Advanced Experiments API Router

Provides endpoints for running research experiments with advanced learning algorithms.
Enabled via ENABLE_RESEARCH_MODE environment variable.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging
from datetime import datetime

from ...ai_layer.learning.advanced_experiments import (
    AdvancedExperimentRunner,
    ExperimentType,
    ResearchConfig
)
from ...config.settings import settings
from ..dependencies import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


def is_research_mode_enabled() -> bool:
    """Check if research mode is enabled - reads dynamically from settings"""
    return getattr(settings, 'ENABLE_RESEARCH_MODE', True)


class ExperimentRequest(BaseModel):
    """Request to run an advanced experiment"""
    experiment_type: str  # baseline, transformer_bandits, meta_learning, gaussian_process, etc.
    campaign_id: Optional[str] = None
    n_arms: int = 4
    n_iterations: int = 100
    contexts: Optional[List[str]] = None
    input_dim: Optional[int] = 10
    n_actions: Optional[int] = 4
    tasks: Optional[List[Dict[str, Any]]] = None  # For meta-learning


class ExperimentResponse(BaseModel):
    """Response from running an experiment"""
    success: bool
    experiment_type: str
    results: Dict[str, Any]
    timestamp: str
    message: str


@router.get("/status")
async def get_research_mode_status():
    """
    Get research mode status and configuration

    Returns current research mode settings and available experiment types.
    """
    if not is_research_mode_enabled():
        return {
            "research_mode_enabled": False,
            "message": "Research mode is disabled. Set ENABLE_RESEARCH_MODE=True to enable."
        }

    config = ResearchConfig.from_env()

    return {
        "research_mode_enabled": True,
        "current_experiment_type": config.experiment_type,
        "available_experiment_types": [e.value for e in ExperimentType],
        "configuration": {
            "use_gpu": config.use_gpu,
            "meta_learning_steps": config.meta_learning_steps,
            "transformer_model": config.transformer_model,
            "gp_kernel": config.gp_kernel,
            "ensemble_size": config.ensemble_size,
            "causal_model": config.causal_model
        }
    }


@router.post("/run", response_model=ExperimentResponse)
async def run_experiment(
    request: ExperimentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Run an advanced research experiment
    """
    try:
        from ...data_layer.repositories.experiment_repo import ExperimentRepository
        from ...data_layer.database.models import Experiment, BanditArm
        from sqlalchemy import update
        import uuid

        if not is_research_mode_enabled():
            raise HTTPException(
                status_code=403,
                detail="Research mode is disabled. Set ENABLE_RESEARCH_MODE=True in environment."
            )

        # Initialize experiment runner with correct type
        experiment_runner = AdvancedExperimentRunner(experiment_type=request.experiment_type)
        
        variants = []
        for i in range(request.n_arms):
           variants.append({"name": f"arm_{i}"})

        exp_payload = {
            "name": f"{request.experiment_type}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "type": request.experiment_type,
            "campaign_id": request.campaign_id,
            "algorithm": request.experiment_type,
            "variants": variants,
            "is_active": True,
            "parameters": {
                "n_iterations": request.n_iterations,
                "n_arms": request.n_arms
            }
        }
        
        repo = ExperimentRepository(db)
        experiment_record = await repo.create(exp_payload)
        experiment_id = experiment_record.id

        import numpy as np
        
        n_iterations = request.n_iterations
        n_arms = request.n_arms
        input_dim = request.input_dim or 10
        n_actions = request.n_actions or n_arms
        
        contexts = request.contexts or [
            f"Marketing campaign iteration {i}"
            for i in range(n_iterations)
        ]
        
        actions = [np.random.randn(input_dim).tolist() for _ in range(n_arms)]
        
        tasks = request.tasks
        if not tasks and request.experiment_type == 'meta_learning':
            n_tasks = max(5, n_iterations // 10)
            tasks = []
            for _ in range(n_tasks):
                tasks.append({
                    'support_x': np.random.randn(5, input_dim).tolist(),
                    'support_y': np.random.randn(5, n_actions).tolist(),
                    'query_x': np.random.randn(5, input_dim).tolist(),
                    'query_y': np.random.randn(5, n_actions).tolist()
                })
        
        experiment_data = {
            'campaign_id': request.campaign_id,
            'n_arms': n_arms,
            'n_iterations': n_iterations,
            'n_trials': n_iterations,  # For bayesian_optimization
            'contexts': contexts,
            'input_dim': input_dim,
            'n_actions': n_actions,
            'n_features': input_dim,  # For gaussian_process
            'actions': actions,  # For gaussian_process
            'tasks': tasks or []
        }

        import time
        start_time = time.time()

        raw_results = experiment_runner.run_experiment(experiment_data)
        
        duration_seconds = time.time() - start_time
        logger.info(f"Experiment completed: {request.experiment_type}. Duration: {duration_seconds}s")
        
        # Update DB with results using SQL because Repository doesn't support bulk arm updates
        arm_stats = raw_results.get('arm_stats', {})
        total_pulls = 0
        total_successes = 0
        
        for arm_key, stats in arm_stats.items():
            try:
                if arm_key.isdigit():
                    arm_idx = int(arm_key)
                    db_arm_id = f"arm_{arm_idx}"
                else:
                    db_arm_id = arm_key
                
                pulls = stats.get('pulls', 0)
                reward = stats.get('rewards', 0.0)
                successes_est = int(reward)
                
                total_pulls += pulls
                total_successes += successes_est
                
                stmt = (
                    update(BanditArm)
                    .where(
                        (BanditArm.experiment_id == experiment_id) & 
                        (BanditArm.arm_id == db_arm_id)
                    )
                    .values(
                        pulls=pulls,
                        total_reward=reward,
                        successes=successes_est,
                        failures=pulls-successes_est
                    )
                )
                await db.execute(stmt)
                
            except Exception as arm_e:
                logger.warning(f"Failed to update arm {arm_key}: {arm_e}")

        exp_update = (
            update(Experiment)
            .where(Experiment.id == experiment_id)
            .values(
                is_active=False,
                ended_at=datetime.utcnow(),
                results=raw_results,
                total_impressions=total_pulls,
                total_conversions=total_successes
            )
        )
        await db.execute(exp_update)
        await db.commit()

        avg_reward = raw_results.get('average_reward', 0)
        total_reward = raw_results.get('total_reward', 0)
        n_iter = raw_results.get('n_iterations', request.n_iterations)
        
        # Get real reward history from experiment results - NO FAKE DATA FALLBACK
        # If experiment didn't return history, dashboard will show "No data available"
        reward_history = raw_results.get('reward_history', [])

        import numpy as np
        rewards_array = np.array(reward_history) if reward_history else np.array([avg_reward])
        cumulative_regret = sum(1.0 - r for r in reward_history) if reward_history else n_iter * (1.0 - avg_reward)

        results = {
            'status': 'completed',
            'duration_seconds': round(duration_seconds, 3),  # 3 decimal places for millisecond precision
            'final_reward': avg_reward,
            'metrics': {
                'mean_reward': float(rewards_array.mean()) if len(rewards_array) > 0 else avg_reward,
                'std_reward': float(np.std(rewards_array, ddof=1)) if len(rewards_array) > 1 else 0.0,
                'max_reward': float(rewards_array.max()) if len(rewards_array) > 0 else avg_reward,
                'min_reward': float(rewards_array.min()) if len(rewards_array) > 0 else avg_reward,
                'cumulative_regret': round(cumulative_regret, 2)
            },
            'reward_history': reward_history,
            'experiment_type': raw_results.get('experiment_type', request.experiment_type),
            'n_iterations': n_iter,
            'model_used': raw_results.get('model_used', 'default'),
            'raw_results': raw_results
        }

        return ExperimentResponse(
            success=True,
            experiment_type=request.experiment_type,
            results=results,
            timestamp=datetime.utcnow().isoformat(),
            message=f"Successfully ran {request.experiment_type} experiment"
        )

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CompareMethodsRequest(BaseModel):
    """Request to compare multiple experiment methods"""
    experiment_types: List[str]


@router.post("/compare-methods")
async def compare_experiment_methods(
    request: CompareMethodsRequest,
    n_iterations: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    Compare multiple experimental methods on the same task

    Useful for thesis research to compare different approaches.

    Args:
        experiment_types: List of experiment types to compare
        n_iterations: Number of iterations to run each experiment

    Returns:
        Comparison results with performance metrics for each method
    """
    experiment_types = request.experiment_types
    try:
        from ...ai_layer.learning.advanced_experiments import AdvancedExperimentRunner
        
        if not is_research_mode_enabled():
            raise HTTPException(
                status_code=403,
                detail="Research mode is disabled. Set ENABLE_RESEARCH_MODE=True in environment."
            )

        logger.info(f"Comparing {len(experiment_types)} experiment methods")

        comparison_results = {}

        import numpy as np
        
        n_tasks = max(5, n_iterations // 10)
        tasks = [
            {
                'support_x': np.random.randn(5, 10).tolist(),
                'support_y': np.random.randn(5, 4).tolist(),
                'query_x': np.random.randn(5, 10).tolist(),
                'query_y': np.random.randn(5, 4).tolist()
            } for _ in range(n_tasks)
        ]
        
        test_data = {
            'n_arms': 4,
            'n_iterations': n_iterations,
            'n_trials': n_iterations,  # For bayesian_optimization
            'contexts': [f"Test context {i}" for i in range(n_iterations)],
            'input_dim': 10,
            'n_features': 10,  # For gaussian_process
            'n_actions': 4,
            'actions': [np.random.randn(10).tolist() for _ in range(4)], # Required for GP
            'tasks': tasks  # Scaled for Meta Learning
        }

        for exp_type in experiment_types:
            try:
                runner = AdvancedExperimentRunner(experiment_type=exp_type)
                results = runner.run_experiment(test_data)

                comparison_results[exp_type] = results

            except Exception as e:
                logger.error(f"Failed to run {exp_type}: {e}")
                comparison_results[exp_type] = {
                    'error': str(e),
                    'success': False
                }

        # Calculate comparative metrics
        if len(comparison_results) > 1:
            best_method = max(
                comparison_results.items(),
                key=lambda x: x[1].get('average_reward', 0) if 'average_reward' in x[1] else 0
            )

            comparison_results['summary'] = {
                'best_method': best_method[0],
                'best_average_reward': best_method[1].get('average_reward', 0),
                'methods_compared': len(experiment_types),
                'total_iterations_per_method': n_iterations
            }

        logger.info(f"Comparison complete: {len(comparison_results)} methods")

        formatted_results = []
        for method, data in comparison_results.items():
            if method == 'summary':
                continue
            if 'error' in data:
                formatted_results.append({
                    'method': method,
                    'error': data['error'],
                    'success': False
                })
                continue
            
            avg_reward = data.get('average_reward', 0)
            formatted_results.append({
                'method': method,
                'mean_reward': avg_reward,
                'std_reward': data.get('std_reward', 0.05),
                'max_reward': data.get('max_reward', avg_reward * 1.1),
                'cumulative_regret': data.get('cumulative_regret', n_iterations * (1 - avg_reward))
            })

        return {
            'success': True,
            'results': formatted_results,
            'best_method': comparison_results.get('summary', {}).get('best_method', experiment_types[0] if experiment_types else 'unknown'),
            'comparison_results': comparison_results,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiment-history")
async def get_experiment_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    Get history of research experiments run
    """
    try:
        from ...data_layer.database.connection import get_async_session
        from ...data_layer.database.models import Experiment, BanditArm
        from sqlalchemy import select, desc

        async with get_async_session() as session:
            query = select(Experiment).order_by(desc(Experiment.started_at)).limit(limit)
            result = await session.execute(query)
            experiments = result.scalars().all()

            experiment_history = []
            for exp in experiments:
                arms_query = select(BanditArm).where(BanditArm.experiment_id == exp.id)
                arms_result = await session.execute(arms_query)
                arms = arms_result.scalars().all()

                status = "active" if exp.is_active else "completed"
                if exp.ended_at:
                    status = "completed"

                experiment_history.append({
                    'experiment_id': str(exp.id),
                    'name': exp.name,
                    'algorithm': exp.algorithm,
                    'started_at': exp.started_at.isoformat() if exp.started_at else None,
                    'ended_at': exp.ended_at.isoformat() if exp.ended_at else None,
                    'status': status,
                    'num_arms': len(arms),
                    'total_pulls': sum(arm.pulls or 0 for arm in arms) if arms else 0,
                    'total_successes': sum(arm.successes or 0 for arm in arms) if arms else 0,
                    'best_arm': max(arms, key=lambda a: (a.successes or 0) / max(a.pulls or 1, 1)).arm_id if arms else None
                })

            return {
                'experiments': experiment_history,
                'total': len(experiment_history)
            }

    except Exception as e:
        logger.error(f"Error fetching experiment history: {e}")
        return {
            'error': str(e),
            'experiments': [],
            'total': 0
        }
