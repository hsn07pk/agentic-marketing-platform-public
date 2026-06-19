from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
import logging

from ..database.models import Experiment, BanditArm
from ...config.settings import settings

logger = logging.getLogger(__name__)

class ExperimentRepository:
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, experiment_data: Dict[str, Any]) -> Experiment:

        try:
            # Map 'experiment_type' to 'type' for model compatibility
            if 'experiment_type' in experiment_data:
                experiment_data['type'] = experiment_data.pop('experiment_type')

            # Map 'experiment_type' to 'algorithm' if algorithm not set
            if 'algorithm' not in experiment_data and 'type' in experiment_data:
                experiment_data['algorithm'] = experiment_data['type']

            # Move extra fields to parameters JSONB field
            extra_fields = ['target_sample_size', 'confidence_threshold', 'duration']
            parameters = experiment_data.get('parameters', {})

            for field in extra_fields:
                if field in experiment_data:
                    parameters[field] = experiment_data.pop(field)

            if parameters:
                experiment_data['parameters'] = parameters

            variants = experiment_data.get('variants', [])

            experiment = Experiment(**experiment_data)
            self.session.add(experiment)
            await self.session.flush()  # Flush to get experiment.id without committing

            if variants:
                from ..database.models import BanditArm

                for idx, variant in enumerate(variants):
                    if isinstance(variant, str):
                        arm_id = variant
                        variant_data = {"name": variant}
                    else:
                        arm_id = variant.get('name', f"variant_{idx}")
                        variant_data = variant

                    bandit_arm = BanditArm(
                        experiment_id=experiment.id,
                        arm_id=arm_id,
                        variant_data=variant_data,
                        alpha=1.0,
                        beta=1.0,
                        pulls=0,
                        successes=0,
                        failures=0,
                        total_reward=0.0
                    )
                    self.session.add(bandit_arm)

                logger.info(f"Created {len(variants)} bandit arms for experiment {experiment.id}")

            await self.session.commit()
            await self.session.refresh(experiment)

            logger.info(f"Created experiment: {experiment.id}")
            return experiment
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def get_by_id(self, experiment_id: str) -> Optional[Experiment]:
        
        try:
            stmt = select(Experiment).where(Experiment.id == UUID(experiment_id))
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get experiment: {e}")
            return None
    
    async def get_by_campaign(
        self,
        campaign_id: str
    ) -> List[Experiment]:
        
        try:
            stmt = (
                select(Experiment)
                .where(Experiment.campaign_id == UUID(campaign_id))
                .order_by(desc(Experiment.started_at))
            )
            result = await self.session.execute(stmt)
            experiments = result.scalars().all()
            return list(experiments)
        except Exception as e:
            logger.error(f"Failed to get experiments: {e}")
            return []
    
    async def update(
        self,
        experiment_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Experiment]:
        
        try:
            updates['updated_at'] = datetime.utcnow()
            
            stmt = (
                update(Experiment)
                .where(Experiment.id == UUID(experiment_id))
                .values(**updates)
                .returning(Experiment)
            )
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return result.scalar_one_or_none()
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update experiment: {e}")
            return None
    
    async def update_results(
        self,
        experiment_id: str,
        results: Dict[str, Any]
    ) -> bool:
        
        try:
            stmt = (
                update(Experiment)
                .where(Experiment.id == UUID(experiment_id))
                .values(results=results, updated_at=datetime.utcnow())
            )
            await self.session.execute(stmt)
            await self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update experiment results: {e}")
            return False
    
    async def get_active_experiments(self) -> List[Experiment]:
        
        try:
            stmt = (
                select(Experiment)
                .where(Experiment.status == "running")
                .order_by(desc(Experiment.started_at))
            )
            result = await self.session.execute(stmt)
            experiments = result.scalars().all()
            return list(experiments)
        except Exception as e:
            logger.error(f"Failed to get active experiments: {e}")
            return []
    
    async def delete(self, experiment_id: str) -> bool:
        
        try:
            stmt = delete(Experiment).where(Experiment.id == UUID(experiment_id))
            await self.session.execute(stmt)
            await self.session.commit()
            
            logger.info(f"Deleted experiment: {experiment_id}")
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete experiment: {e}")
            return False
    
    async def get_experiment_summary(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        
        try:
            stmt = (
                select(
                    func.count(Experiment.id).label('total'),
                    func.sum(
                        func.cast(
                            func.json_extract_path_text(Experiment.results, 'total_samples'),
                            func.Integer
                        )
                    ).label('total_samples')
                )
                .where(Experiment.campaign_id == UUID(campaign_id))
            )
            
            result = await self.session.execute(stmt)
            row = result.first()
            
            return {
                "total_experiments": row.total or 0,
                "total_samples": row.total_samples or 0
            }
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {
                "total_experiments": 0,
                "total_samples": 0
            }
