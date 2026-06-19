"""
Brand Voice Configuration Loader

Centralizes brand voice settings from data/company/brand_voice.json
Replaces hardcoded company references in prompts
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class BrandVoice:
    tone_attributes: List[str]
    tone_description: str
    formality_level: str
    use_first_person: bool
    use_second_person: bool
    preferred_terms: List[str]
    prohibited_terms: List[str]
    technical_jargon_level: str
    use_statistics: bool
    use_case_studies: bool
    use_metaphors: str
    use_humor: str
    use_emojis: bool
    use_questions: str

@dataclass
class Company:
    name: str
    legal_name: str
    website: str
    founded_year: int
    headquarters: str
    industry: str
    company_size: str
    tagline: str
    mission: str
    description: str
    logo_url: Optional[str] = None
    social_handles: Optional[Dict[str, str]] = None

@dataclass
class ValueProposition:
    id: str
    title: str
    description: str
    target_personas: List[str]

class BrandVoiceConfig:

    def __init__(self, config_path: Optional[Path] = None):

        if config_path is None:
            config_path = Path("data/company/brand_voice.json")

        self.config_path = config_path
        self._config_data: Optional[Dict[str, Any]] = None

        self._load_config()

    def _load_config(self):
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                # Check if this is the Agentic-specific format (has 'profile' key)
                # or the generic format (has 'company' key)
                if 'profile' in raw_data:
                    self._config_data = self._transform_agentic_format(raw_data)
                    logger.info(f"Loaded Agentic brand voice from {self.config_path}")
                else:
                    self._config_data = raw_data
                    logger.info(f"Loaded brand voice from {self.config_path}")
            else:
                logger.warning(
                    f"Brand voice file not found: {self.config_path}. Using defaults."
                )
                self._config_data = self._get_default_config()

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.config_path}: {e}")
            logger.warning("Using default brand voice configuration")
            self._config_data = self._get_default_config()

        except Exception as e:
            logger.error(f"Failed to load brand voice: {e}", exc_info=True)
            self._config_data = self._get_default_config()

    def _transform_agentic_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Agentic-specific brand voice format to standard format."""
        profile = raw_data.get('profile', {})
        tone = raw_data.get('tone', {})
        style = raw_data.get('style', {})
        audience = raw_data.get('audience', {})
        goals = raw_data.get('goals', {})

        company = {
            "name": profile.get('company', 'Agentic'),
            "legal_name": f"{profile.get('company', 'Agentic')} AI",
            "website": "https://example.com",
            "founded_year": 2024,
            "headquarters": "Finland",
            "industry": "B2B SaaS - HR Technology & QWL Analytics",
            "company_size": "11-50 employees",
            "tagline": "QWL Science + RL AI for team performance",
            "mission": goals.get('primary', [''])[0] if goals.get('primary') else "",
            "description": profile.get('summary', ''),
            "spokesperson": {
                "name": profile.get('name', ''),
                "role": profile.get('role', ''),
                "summary": profile.get('summary', '')
            }
        }

        characteristics = tone.get('characteristics', [])
        avoid = tone.get('avoid', [])
        style_attrs = style.get('emoji_usage', {})

        brand_voice = {
            "tone_attributes": characteristics[:5] if characteristics else ['professional', 'data-driven'],
            "tone_description": tone.get('default', ''),
            "formality_level": style.get('formality', {}).get('level', 'professional_casual'),
            "use_first_person": False,  # Inferred from content
            "use_second_person": True,   # Common in B2B
            "vocabulary": {
                "preferred_terms": [],  # Extract from content_themes if needed
                "prohibited_terms": avoid,
                "technical_jargon_level": "high"
            },
            "content_preferences": {
                "use_statistics": True,  # QWL is data-driven
                "use_case_studies": True,
                "use_metaphors": "sparingly",
                "use_humor": style_attrs.get('frequency', 'low'),
                "use_emojis": style_attrs.get('enabled', False),
                "use_questions": "moderate"
            },
            "sentence_structure": {
                "average_length": style.get('sentence_length', 'short_to_medium'),
                "paragraph_length": style.get('paragraph_length', 'short'),
                "variety": "high"
            },
            "raw_agentic_data": raw_data  # Preserve full data for detailed use
        }

        value_propositions = []
        for idx, goal in enumerate(goals.get('primary', [])[:3]):
            value_propositions.append({
                "id": f"vp_{idx+1:03d}",
                "title": goal.split('.')[0][:50],  # First sentence as title
                "description": goal,
                "target_personas": audience.get('primary_personas', [])
            })

        return {
            "company": company,
            "brand_voice": brand_voice,
            "value_propositions": value_propositions,
            "key_differentiators": [],
            "_raw_format": "agentic_specific",
            "_version": raw_data.get('version', '1.0')
        }

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "company": {
                "name": "Agentic AI",
                "legal_name": "Agentic Technologies Inc.",
                "website": "https://example.com",
                "founded_year": 2024,
                "headquarters": "San Francisco, CA, USA",
                "industry": "B2B SaaS - Marketing Technology",
                "company_size": "11-50 employees",
                "tagline": "Autonomous Marketing Intelligence",
                "mission": "Empower B2B marketers with AI-driven autonomous campaign orchestration.",
                "description": "AI-powered platform for autonomous marketing campaign generation and deployment.",
                "social_handles": {
                    "linkedin": "agentic-ai",
                    "twitter": "@agentic"
                }
            },
            "brand_voice": {
                "tone_attributes": [
                    "data-driven",
                    "professional",
                    "innovative",
                    "trustworthy",
                    "confident"
                ],
                "tone_description": "Authoritative yet approachable. Data-driven claims, never hyperbole.",
                "formality_level": "business_professional",
                "use_first_person": False,
                "use_second_person": True,
                "sentence_structure": {
                    "average_length": "medium",
                    "variety": "high",
                    "active_voice_preference": 0.9
                },
                "vocabulary": {
                    "preferred_terms": [
                        "autonomous agents",
                        "simulation-first",
                        "governed AI",
                        "cost-per-booked-call",
                        "human-in-the-loop"
                    ],
                    "prohibited_terms": [
                        "cheap",
                        "revolutionary",
                        "game-changer",
                        "guaranteed results",
                        "magic"
                    ],
                    "technical_jargon_level": "high"
                },
                "content_preferences": {
                    "use_statistics": True,
                    "use_case_studies": True,
                    "use_metaphors": "sparingly",
                    "use_humor": "minimal",
                    "use_emojis": False,
                    "use_questions": "moderate"
                }
            },
            "value_propositions": [
                {
                    "id": "vp_001",
                    "title": "Simulation-Before-Deployment",
                    "description": "Test campaigns in digital twin market before spending budget.",
                    "target_personas": ["decision_maker"]
                }
            ],
            "key_differentiators": [
                "Only platform with >90% simulation-to-live accuracy",
                "Six-layer OODA-G architecture with explicit governance",
                "Cost-per-booked-call optimization"
            ]
        }

    def get_company(self) -> Company:
        company_data = self._config_data.get('company', {})
        return Company(
            name=company_data.get('name', 'Agentic AI'),
            legal_name=company_data.get('legal_name', 'Agentic Technologies Inc.'),
            website=company_data.get('website', 'https://example.com'),
            founded_year=company_data.get('founded_year', 2024),
            headquarters=company_data.get('headquarters', 'San Francisco, CA'),
            industry=company_data.get('industry', 'B2B SaaS'),
            company_size=company_data.get('company_size', '11-50 employees'),
            tagline=company_data.get('tagline', 'Autonomous Marketing Intelligence'),
            mission=company_data.get('mission', ''),
            description=company_data.get('description', ''),
            logo_url=company_data.get('logo_url'),
            social_handles=company_data.get('social_handles', {})
        )

    def get_brand_voice(self) -> BrandVoice:
        bv_data = self._config_data.get('brand_voice', {})
        vocab = bv_data.get('vocabulary', {})
        prefs = bv_data.get('content_preferences', {})

        return BrandVoice(
            tone_attributes=bv_data.get('tone_attributes', ['professional']),
            tone_description=bv_data.get('tone_description', ''),
            formality_level=bv_data.get('formality_level', 'business_professional'),
            use_first_person=bv_data.get('use_first_person', False),
            use_second_person=bv_data.get('use_second_person', True),
            preferred_terms=vocab.get('preferred_terms', []),
            prohibited_terms=vocab.get('prohibited_terms', []),
            technical_jargon_level=vocab.get('technical_jargon_level', 'medium'),
            use_statistics=prefs.get('use_statistics', True),
            use_case_studies=prefs.get('use_case_studies', True),
            use_metaphors=prefs.get('use_metaphors', 'sparingly'),
            use_humor=prefs.get('use_humor', 'minimal'),
            use_emojis=prefs.get('use_emojis', False),
            use_questions=prefs.get('use_questions', 'moderate')
        )

    def get_value_propositions(self) -> List[ValueProposition]:
        vp_data = self._config_data.get('value_propositions', [])
        return [
            ValueProposition(
                id=vp['id'],
                title=vp['title'],
                description=vp['description'],
                target_personas=vp.get('target_personas', [])
            )
            for vp in vp_data
        ]

    def get_key_differentiators(self) -> List[str]:
        return self._config_data.get('key_differentiators', [])

    def format_for_prompt(self, persona: Optional[str] = None) -> str:
        """Format brand voice for inclusion in LLM prompt."""
        company = self.get_company()
        brand_voice = self.get_brand_voice()
        value_props = self.get_value_propositions()
        differentiators = self.get_key_differentiators()

        if persona:
            value_props = [
                vp for vp in value_props
                if persona in vp.target_personas
            ]

        prompt = f"""# Brand Voice Guidelines for {company.name}

## Company Overview
{company.description}

**Tagline:** {company.tagline}
**Mission:** {company.mission}

## Tone & Style
- **Tone Attributes:** {', '.join(brand_voice.tone_attributes)}
- **Description:** {brand_voice.tone_description}
- **Formality:** {brand_voice.formality_level}
- **Technical Level:** {brand_voice.technical_jargon_level}

## Vocabulary Guidelines
**Preferred Terms (use frequently):**
{chr(10).join('- ' + term for term in brand_voice.preferred_terms)}

**Prohibited Terms (NEVER use):**
{chr(10).join('- ' + term for term in brand_voice.prohibited_terms)}

## Content Preferences
- Statistics: {'Include data and metrics' if brand_voice.use_statistics else 'Minimize'}
- Case Studies: {'Reference when relevant' if brand_voice.use_case_studies else 'Avoid'}
- Metaphors: {brand_voice.use_metaphors}
- Humor: {brand_voice.use_humor}
- Emojis: {'Allowed' if brand_voice.use_emojis else 'Never'}
- Questions: {brand_voice.use_questions}

## Key Value Propositions
{chr(10).join(f'- **{vp.title}**: {vp.description}' for vp in value_props)}

## Competitive Differentiators
{chr(10).join('- ' + diff for diff in differentiators)}
"""
        return prompt

    def to_dict(self) -> Dict[str, Any]:
        return self._config_data

    def save(self, path: Optional[Path] = None):
        save_path = path or self.config_path

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self._config_data, f, indent=2)

            logger.info(f"Saved brand voice to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save brand voice: {e}")
            raise

_brand_voice_config: Optional[BrandVoiceConfig] = None

def get_brand_voice_config() -> BrandVoiceConfig:
    global _brand_voice_config

    if _brand_voice_config is None:
        _brand_voice_config = BrandVoiceConfig()

    return _brand_voice_config

def reload_brand_voice_config():
    global _brand_voice_config
    _brand_voice_config = BrandVoiceConfig()
    logger.info("Brand voice configuration reloaded")
