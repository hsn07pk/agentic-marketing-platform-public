"""
Competitor Mention Validator
Ensures competitive references are factual, sourced, and avoid risky topics
"""
import csv
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class CompetitorValidator:
    """Validates competitor mentions against data/competitors/competitors.csv"""

    def __init__(self):
        self.competitors = self._load_competitors()

    def _load_competitors(self) -> Dict[str, Dict[str, Any]]:
        csv_file = Path("data/competitors/competitors.csv")

        if not csv_file.exists():
            logger.warning(f"Competitors CSV not found: {csv_file}")
            return {}

        try:
            competitors_dict = {}
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get('name'):
                        continue

                    name = row['name']
                    competitors_dict[name.lower()] = {
                        'name': name,
                        'category': row.get('category', ''),
                        'url': row.get('url', ''),
                        'key_features': row.get('key_features', ''),
                        'typical_claims': row.get('typical_claims', ''),
                        'differentiators_vs_us': row.get('differentiators_vs_us', ''),
                        'risky_topics': row.get('risky_topics', ''),
                        'last_checked': row.get('last_checked', '')
                    }

            logger.info(f"✅ Loaded {len(competitors_dict)} competitors from CSV")
            return competitors_dict

        except Exception as e:
            logger.error(f"Failed to load competitors CSV: {e}")
            return {}

    def validate_content(
        self,
        content: str,
        headline: Optional[str] = None
    ) -> Dict[str, Any]:
        full_text = f"{headline or ''} {content}".lower()

        result = {
            "valid": True,
            "competitors_mentioned": [],
            "risky_mentions": [],
            "warnings": [],
            "recommendations": []
        }

        for comp_key, comp_data in self.competitors.items():
            if comp_key in full_text or comp_data['name'] in full_text:
                result["competitors_mentioned"].append(comp_data['name'])

                risky_topics = comp_data.get('risky_topics', '').lower()
                if risky_topics:
                    risky_patterns = [
                        "guarantee",
                        "lack of science",
                        "pricing",
                        "accuracy claims",
                        "medical",
                        "psych claims",
                        "disparagement",
                        "privacy",
                        "fairness"
                    ]

                    for pattern in risky_patterns:
                        if pattern in risky_topics:
                            if self._check_risky_pattern(full_text, pattern, comp_key):
                                result["risky_mentions"].append({
                                    "competitor": comp_data['name'],
                                    "risky_topic": pattern,
                                    "guidance": risky_topics
                                })
                                result["valid"] = False

                if comp_data.get('differentiators_vs_us'):
                    result["recommendations"].append({
                        "competitor": comp_data['name'],
                        "differentiator": comp_data['differentiators_vs_us']
                    })

        if result["competitors_mentioned"]:
            result["warnings"].append(
                f"Content mentions competitors: {', '.join(result['competitors_mentioned'])}"
            )

        if result["risky_mentions"]:
            for risky in result["risky_mentions"]:
                result["warnings"].append(
                    f"⚠️  RISKY: Mentions {risky['competitor']} with risky topic '{risky['risky_topic']}'. "
                    f"Guidance: {risky['guidance']}"
                )

        if not result["valid"]:
            logger.warning(
                f"Competitor validation failed: {len(result['risky_mentions'])} risky mentions found"
            )

        return result

    def _check_risky_pattern(
        self,
        content: str,
        pattern: str,
        competitor_name: str
    ) -> bool:
        comp_pos = content.find(competitor_name)
        if comp_pos == -1:
            return False

        # 200-char window around mention to catch nearby risky language
        start = max(0, comp_pos - 200)
        end = min(len(content), comp_pos + 200)
        context = content[start:end]

        if pattern == "guarantee":
            return "guarantee" in context or "guaranteed" in context

        elif pattern == "lack of science":
            return "not science" in context or "lacks science" in context or "no science" in context

        elif pattern == "pricing":
            return "$" in context or "price" in context or "cost" in context or "cheaper" in context

        elif pattern == "accuracy claims":
            return "accurate" in context or "accuracy" in context or "90%" in context or "95%" in context

        elif pattern in ["medical", "psych claims"]:
            medical_terms = ["health", "medical", "psychological", "therapy", "treatment", "diagnosis"]
            return any(term in context for term in medical_terms)

        elif pattern == "disparagement":
            negative_terms = ["bad", "poor", "inferior", "worse", "fail", "failing", "inadequate", "subpar"]
            return any(term in context for term in negative_terms)

        elif pattern == "privacy":
            return "privacy" in context or "data breach" in context or "surveillance" in context

        elif pattern == "fairness":
            return "unfair" in context or "biased" in context or "discriminat" in context

        return False

    def get_competitor_info(self, competitor_name: str) -> Optional[Dict[str, Any]]:
        return self.competitors.get(competitor_name.lower())

    def get_all_competitors(self) -> List[Dict[str, Any]]:
        return list(self.competitors.values())

    def format_validation_report(
        self,
        validation_result: Dict[str, Any]
    ) -> str:
        lines = ["=== Competitor Mention Validation Report ===", ""]

        lines.append(f"Overall Status: {'✅ PASSED' if validation_result['valid'] else '❌ FAILED'}")
        lines.append("")

        if validation_result["competitors_mentioned"]:
            lines.append(f"Competitors Mentioned: {', '.join(validation_result['competitors_mentioned'])}")
            lines.append("")

        if validation_result["risky_mentions"]:
            lines.append("❌ Risky Mentions Found:")
            for risky in validation_result["risky_mentions"]:
                lines.append(f"  - {risky['competitor']}: {risky['risky_topic']}")
                lines.append(f"    Guidance: {risky['guidance']}")
            lines.append("")

        if validation_result["recommendations"]:
            lines.append("💡 Differentiator Recommendations:")
            for rec in validation_result["recommendations"]:
                lines.append(f"  - {rec['competitor']}:")
                lines.append(f"    {rec['differentiator']}")
            lines.append("")

        if validation_result["warnings"]:
            lines.append("⚠️  Warnings:")
            for warning in validation_result["warnings"]:
                lines.append(f"  - {warning}")
            lines.append("")

        return "\n".join(lines)
