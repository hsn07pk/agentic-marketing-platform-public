"""Factory for creating customer agents from persona configurations."""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ...data_layer.database.models import Persona
from ...config.settings import settings

logger = logging.getLogger(__name__)

class PersonaFactory:
    """Factory for loading and creating persona-based agents."""
    
    def __init__(self, persona_dir: Optional[Path] = None):
        self.persona_dir = persona_dir or (Path(settings.CONFIG_DIR) / "personas")
        self.loaded_personas: Dict[str, Dict] = {}
        self._load_all_personas()
    
    def _load_all_personas(self):
        """Load all persona YAML files from config/personas/ and config/personas/custom/."""
        if not self.persona_dir.exists():
            logger.warning(f"Persona directory not found: {self.persona_dir}")
            return

        yaml_files = list(self.persona_dir.glob("*.yaml"))

        custom_dir = self.persona_dir / "custom"
        if custom_dir.exists():
            yaml_files.extend(custom_dir.glob("*.yaml"))
            logger.info(f"Scanning custom persona directory: {custom_dir}")

        logger.info(f"Found {len(yaml_files)} persona YAML files to load")

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    raw_data = yaml.safe_load(f)

                    # Transform flat YAML structure to expected format
                    persona_id = raw_data.get('persona_id', yaml_file.stem)

                    persona_data = {
                        'id': persona_id,
                        'name': raw_data.get('label', persona_id.replace('_', ' ').title()),
                        'description': self._build_description(raw_data),
                        'demographics': self._extract_demographics(raw_data),
                        'behavior': self._extract_behavior(raw_data),
                        'content_preferences': self._extract_content_preferences(raw_data),
                        'platform_behavior': self._extract_platform_behavior(raw_data),
                        'simulation': self._extract_simulation_config(raw_data)
                    }

                    self.loaded_personas[persona_id] = persona_data
                    logger.info(f"Loaded persona: {persona_id}")
            except Exception as e:
                logger.error(f"Failed to load persona from {yaml_file}: {e}", exc_info=True)

    def _build_description(self, raw_data: Dict) -> str:
        """Build description from who/goals/pains"""
        who = raw_data.get('who', {})
        archetypes = who.get('archetypes', [])
        if archetypes:
            return f"{', '.join(archetypes)}. {who.get('context', {}).get('buying_committee_role', '')}"
        return raw_data.get('label', 'Persona')

    def _extract_demographics(self, raw_data: Dict) -> Dict:
        """Extract demographics from who section"""
        who = raw_data.get('who', {})
        return {
            'archetypes': who.get('archetypes', []),
            'role': who.get('context', {}).get('buying_committee_role', 'unknown'),
            'risk_posture': who.get('context', {}).get('risk_posture', 'balanced'),
            'time_pressure': who.get('context', {}).get('time_pressure', 'medium')
        }

    def _extract_behavior(self, raw_data: Dict) -> Dict:
        return {
            'daily_active_probability': 0.4,
            'engagement_rates': {
                'ad_click_probability': 0.03,
                'content_like_probability': 0.10,
                'content_share_probability': 0.02,
                'comment_probability': 0.01
            },
            'conversion_funnel': {
                'overall_conversion': 0.01,
                'awareness_to_consideration': 0.3,
                'consideration_to_intent': 0.2,
                'intent_to_purchase': 0.15
            },
            'ad_fatigue': {
                'threshold_impressions': 5,
                'decay_rate': 0.15
            },
            'network_influence': {
                'influences_others': 0.3,
                'influenced_by_others': 0.2
            },
            'peak_activity_hours': [9, 10, 11, 14, 15, 16]
        }

    def _extract_content_preferences(self, raw_data: Dict) -> Dict:
        """Extract content preferences"""
        tone = raw_data.get('tone_style', {})
        retrieval = raw_data.get('retrieval_preferences', {})
        proof = raw_data.get('proof_preferences', {})

        return {
            'formality': tone.get('formality', 'medium'),
            'directness': tone.get('directness', 'high'),
            'warmth': tone.get('warmth', 'medium'),
            'reading_level': tone.get('reading_level', 'B2-C1'),
            'avoid_phrases': tone.get('avoid', []),
            'require_citations': proof.get('require_claim_ids', True),
            'minimum_confidence': proof.get('minimum_claim_confidence', 3),
            'preferred_tags': retrieval.get('must_include_tags', []),
            'preferred_categories': retrieval.get('prefer_categories', [])
        }

    def _extract_platform_behavior(self, raw_data: Dict) -> Dict:
        """Extract platform-specific behavior"""
        channel_guidance = raw_data.get('channel_guidance', {})
        platforms = {}

        for platform, config in channel_guidance.items():
            platforms[platform] = {
                'activity_level': 'high',  # Default
                'target_length': config.get('target_length_words') or config.get('target_length_chars', ''),
                'structure': config.get('structure', []),
                'ctas': config.get('ctas', []),
                'language': config.get('language', 'en')
            }

        return platforms

    def _extract_simulation_config(self, raw_data: Dict) -> Dict:
        return {
            'spawn_probability': 0.33,
            'lifetime_value_multiplier': 1.0,
            'influence_multiplier': 1.0
        }

    def create_persona_model(self, persona_id: str) -> Persona:
        if persona_id not in self.loaded_personas:
            raise ValueError(f"Persona {persona_id} not found")

        config = self.loaded_personas[persona_id]
        behavior = config.get('behavior', {})
        demographics = config.get('demographics', {})

        persona = Persona(
            name=persona_id,
            title=config.get('name', persona_id.replace('_', ' ').title()),
            description=config.get('description', ''),
            role=demographics.get('role', 'unknown'),
            daily_active_prob=behavior.get('daily_active_probability', 0.4),
            content_engagement_prob=behavior.get('engagement_rates', {}).get('content_like_probability', 0.1),
            click_prob=behavior.get('engagement_rates', {}).get('ad_click_probability', 0.03),
            conversion_prob=behavior.get('conversion_funnel', {}).get('overall_conversion', 0.01),
            share_prob=behavior.get('engagement_rates', {}).get('content_share_probability', 0.02),
            active_hours=behavior.get('peak_activity_hours', [9, 10, 11, 14, 15, 16]),
            attributes={
                'ad_fatigue_threshold': behavior.get('ad_fatigue', {}).get('threshold_impressions', 5),
                'ad_fatigue_decay': behavior.get('ad_fatigue', {}).get('decay_rate', 0.15),
                'influence_factor': behavior.get('network_influence', {}).get('influences_others', 0.3),
                'risk_posture': demographics.get('risk_posture', 'balanced'),
                'time_pressure': demographics.get('time_pressure', 'medium')
            },
            preferences=config.get('content_preferences', {})
        )

        return persona
    
    def get_persona_config(self, persona_id: str) -> Dict:
        if persona_id not in self.loaded_personas:
            raise ValueError(f"Persona {persona_id} not found")
        
        return self.loaded_personas[persona_id]
    
    def generate_persona_distribution(
        self,
        total_count: int,
        distribution: Optional[Dict[str, float]] = None
    ) -> List[Persona]:
        """Generate a list of personas based on distribution weights."""
        if not self.loaded_personas:
            raise ValueError("No personas loaded")
        
        if distribution is None:
            distribution = {}
            total_prob = 0.0
            
            for persona_id, config in self.loaded_personas.items():
                prob = config.get('simulation', {}).get('spawn_probability', 0.33)
                distribution[persona_id] = prob
                total_prob += prob
            
            if total_prob > 0:
                distribution = {k: v/total_prob for k, v in distribution.items()}
        
        personas = []
        persona_ids = list(distribution.keys())
        probabilities = [distribution[pid] for pid in persona_ids]
        
        sampled_ids = np.random.choice(
            persona_ids,
            size=total_count,
            p=probabilities
        )
        
        for persona_id in sampled_ids:
            persona = self.create_persona_model(persona_id)
            personas.append(persona)
        
        unique, counts = np.unique(sampled_ids, return_counts=True)
        for pid, count in zip(unique, counts):
            logger.info(f"Generated {count} {pid} personas ({count/total_count*100:.1f}%)")
        
        return personas
    
    def get_persona_preferences(self, persona_id: str) -> Dict:
        if persona_id not in self.loaded_personas:
            return {}
        
        config = self.loaded_personas[persona_id]
        return config.get('content_preferences', {})
    
    def get_platform_behavior(self, persona_id: str, platform: str) -> Dict:
        if persona_id not in self.loaded_personas:
            return {}
        
        config = self.loaded_personas[persona_id]
        platform_behaviors = config.get('platform_behavior', {})
        return platform_behaviors.get(platform, {})
    
    def get_buying_behavior(self, persona_id: str) -> Dict:
        if persona_id not in self.loaded_personas:
            return {}
        
        config = self.loaded_personas[persona_id]
        return config.get('buying_behavior', {})
    
    def get_campaign_response_patterns(self, persona_id: str) -> Dict:
        if persona_id not in self.loaded_personas:
            return {}
        
        config = self.loaded_personas[persona_id]
        return config.get('campaign_response', {})
    
    def list_available_personas(self) -> List[str]:
        return list(self.loaded_personas.keys())
    
    def validate_persona_config(self, persona_id: str) -> bool:
        if persona_id not in self.loaded_personas:
            return False
        
        config = self.loaded_personas[persona_id]
        
        required_fields = [
            'id', 'name', 'description',
            'demographics', 'behavior',
            'content_preferences', 'platform_behavior'
        ]
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Persona {persona_id} missing required field: {field}")
                return False
        
        # Check behavior parameters
        behavior = config.get('behavior', {})
        required_behavior = [
            'daily_active_probability',
            'engagement_rates',
            'conversion_funnel'
        ]
        
        for field in required_behavior:
            if field not in behavior:
                logger.error(f"Persona {persona_id} missing behavior field: {field}")
                return False
        
        logger.info(f"Persona {persona_id} validation passed")
        return True
    
    def get_persona_summary(self, persona_id: str) -> str:
        if persona_id not in self.loaded_personas:
            return f"Persona {persona_id} not found"
        
        config = self.loaded_personas[persona_id]
        behavior = config.get('behavior', {})
        
        summary = f"""
Persona: {config['name']}
Description: {config['description']}

Key Metrics:
- Daily Activity: {behavior.get('daily_active_probability', 0)*100:.0f}%
- Click Rate: {behavior.get('engagement_rates', {}).get('ad_click_probability', 0)*100:.1f}%
- Conversion Rate: {behavior.get('conversion_funnel', {}).get('overall_conversion', 0)*100:.2f}%

Platform Activity:
"""
        
        # Add platform info
        for platform, data in config.get('platform_behavior', {}).items():
            activity = data.get('activity_level', 'unknown')
            summary += f"- {platform.capitalize()}: {activity}\n"
        
        return summary.strip()
    
    def create_mixed_population(
        self,
        total_count: int,
        persona_weights: Optional[Dict[str, float]] = None
    ) -> List[Persona]:
        """Create a mixed population with specified weights."""
        if persona_weights is None:
            return self.generate_persona_distribution(total_count)
        
        total_weight = sum(persona_weights.values())
        if not np.isclose(total_weight, 1.0):
            persona_weights = {
                k: v/total_weight for k, v in persona_weights.items()
            }
        
        return self.generate_persona_distribution(total_count, persona_weights)
    
    def get_ltv_multiplier(self, persona_id: str) -> float:
        if persona_id not in self.loaded_personas:
            return 1.0
        
        config = self.loaded_personas[persona_id]
        sim_config = config.get('simulation', {})
        return sim_config.get('lifetime_value_multiplier', 1.0)
    
    def get_influence_multiplier(self, persona_id: str) -> float:
        if persona_id not in self.loaded_personas:
            return 1.0
        
        config = self.loaded_personas[persona_id]
        sim_config = config.get('simulation', {})
        return sim_config.get('influence_multiplier', 1.0)