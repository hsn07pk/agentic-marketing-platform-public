"""
Google Perspective API Integration for Toxicity Scoring

Per Research Plan Section 7.2:
- "Toxicity: Using a pre-trained model or an external API (like Google's Perspective API)
  to score for harmful language."

The Perspective API analyzes text for potentially harmful content across multiple attributes:
- TOXICITY: Rude, disrespectful, or unreasonable comment
- SEVERE_TOXICITY: Very hateful, aggressive, disrespectful
- IDENTITY_ATTACK: Negative or hateful comments targeting identity
- INSULT: Insulting, inflammatory, or negative comment
- PROFANITY: Obscene or vulgar language
- THREAT: Describes intention to inflict pain, injury, or violence
"""
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ...config.settings import settings

logger = logging.getLogger(__name__)

PERSPECTIVE_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

DEFAULT_ATTRIBUTES = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT"
]


@dataclass
class ToxicityResult:
    toxicity_score: float  # Main toxicity score (0-1)
    severe_toxicity_score: float
    identity_attack_score: float
    insult_score: float
    profanity_score: float
    threat_score: float
    summary_score: float  # Weighted average
    is_toxic: bool  # Above threshold
    details: Dict[str, float]  # All raw scores
    error: Optional[str] = None


class PerspectiveAPIClient:
    """
    Client for Google Perspective API toxicity analysis

    Provides automated toxicity scoring as part of the Governance Layer
    per Research Plan Section 7.2.
    """

    def __init__(self, api_key: Optional[str] = None, toxicity_threshold: float = None):
        self.api_key = api_key or getattr(settings, 'PERSPECTIVE_API_KEY', None)
        self.toxicity_threshold = toxicity_threshold or getattr(settings, 'TOXICITY_THRESHOLD', 0.1)
        self.enabled = bool(self.api_key)

        if not self.enabled:
            logger.info(
                "Perspective API not configured (PERSPECTIVE_API_KEY not set). "
                "Toxicity checks will fall back to LLM-as-Judge."
            )

    async def analyze_text(
        self,
        text: str,
        attributes: List[str] = None,
        language: str = "en"
    ) -> ToxicityResult:
        if not self.enabled:
            return ToxicityResult(
                toxicity_score=0.0,
                severe_toxicity_score=0.0,
                identity_attack_score=0.0,
                insult_score=0.0,
                profanity_score=0.0,
                threat_score=0.0,
                summary_score=0.0,
                is_toxic=False,
                details={},
                error="Perspective API not configured"
            )

        attributes = attributes or DEFAULT_ATTRIBUTES

        payload = {
            "comment": {
                "text": text
            },
            "languages": [language],
            "requestedAttributes": {attr: {} for attr in attributes}
        }

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{PERSPECTIVE_API_URL}?key={self.api_key}"

                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Perspective API error: {response.status} - {error_text}")
                        return ToxicityResult(
                            toxicity_score=0.0,
                            severe_toxicity_score=0.0,
                            identity_attack_score=0.0,
                            insult_score=0.0,
                            profanity_score=0.0,
                            threat_score=0.0,
                            summary_score=0.0,
                            is_toxic=False,
                            details={},
                            error=f"API error: {response.status}"
                        )

                    result = await response.json()
                    return self._parse_response(result)

        except aiohttp.ClientError as e:
            logger.error(f"Perspective API request failed: {e}")
            return ToxicityResult(
                toxicity_score=0.0,
                severe_toxicity_score=0.0,
                identity_attack_score=0.0,
                insult_score=0.0,
                profanity_score=0.0,
                threat_score=0.0,
                summary_score=0.0,
                is_toxic=False,
                details={},
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error calling Perspective API: {e}")
            return ToxicityResult(
                toxicity_score=0.0,
                severe_toxicity_score=0.0,
                identity_attack_score=0.0,
                insult_score=0.0,
                profanity_score=0.0,
                threat_score=0.0,
                summary_score=0.0,
                is_toxic=False,
                details={},
                error=str(e)
            )

    def _parse_response(self, response: Dict[str, Any]) -> ToxicityResult:
        scores = {}

        attribute_scores = response.get("attributeScores", {})

        for attr in DEFAULT_ATTRIBUTES:
            attr_data = attribute_scores.get(attr, {})
            summary_score = attr_data.get("summaryScore", {})
            scores[attr.lower()] = summary_score.get("value", 0.0)

        # Weight severe toxicity and threats higher
        weights = {
            "toxicity": 0.25,
            "severe_toxicity": 0.25,
            "identity_attack": 0.15,
            "insult": 0.1,
            "profanity": 0.1,
            "threat": 0.15
        }

        summary_score = sum(
            scores.get(attr, 0.0) * weight
            for attr, weight in weights.items()
        )

        toxicity_score = scores.get("toxicity", 0.0)
        is_toxic = toxicity_score > self.toxicity_threshold or summary_score > self.toxicity_threshold

        return ToxicityResult(
            toxicity_score=scores.get("toxicity", 0.0),
            severe_toxicity_score=scores.get("severe_toxicity", 0.0),
            identity_attack_score=scores.get("identity_attack", 0.0),
            insult_score=scores.get("insult", 0.0),
            profanity_score=scores.get("profanity", 0.0),
            threat_score=scores.get("threat", 0.0),
            summary_score=summary_score,
            is_toxic=is_toxic,
            details=scores
        )

    async def check_marketing_content(
        self,
        content: str,
        headline: Optional[str] = None
    ) -> Dict[str, Any]:
        full_text = f"{headline}\n\n{content}" if headline else content

        result = await self.analyze_text(full_text)

        # Build response compatible with safety validator
        issues = []

        if result.toxicity_score > 0.5:
            issues.append(f"High toxicity detected: {result.toxicity_score:.2f}")
        if result.severe_toxicity_score > 0.3:
            issues.append(f"Severe toxicity detected: {result.severe_toxicity_score:.2f}")
        if result.identity_attack_score > 0.3:
            issues.append(f"Identity attack language detected: {result.identity_attack_score:.2f}")
        if result.threat_score > 0.3:
            issues.append(f"Threatening language detected: {result.threat_score:.2f}")
        if result.profanity_score > 0.5:
            issues.append(f"Profanity detected: {result.profanity_score:.2f}")

        return {
            "score": result.toxicity_score,
            "summary_score": result.summary_score,
            "is_toxic": result.is_toxic,
            "issues": issues,
            "details": result.details,
            "source": "perspective_api" if not result.error else "fallback",
            "error": result.error
        }


_perspective_client: Optional[PerspectiveAPIClient] = None


def get_perspective_client() -> PerspectiveAPIClient:
    global _perspective_client
    if _perspective_client is None:
        _perspective_client = PerspectiveAPIClient()
    return _perspective_client


# Convenience function(s)
async def check_toxicity(
    content: str,
    headline: Optional[str] = None
) -> Dict[str, Any]:
    client = get_perspective_client()
    return await client.check_marketing_content(content, headline)
