"""
HITL Preflight Validator - Implements checklist from data/docs/governance/HITL_checklist.md
Validates content before it goes to the HITL queue
"""
import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class HITLPreflightValidator:
    """
    Validates content against HITL checklist before queuing for human review

    Based on data/docs/governance/HITL_checklist.md (2025-11-04)
    """

    # These phrases violate brand voice governance — they overpromise or
    # use hyperbolic language that cannot be substantiated with citations.
    FORBIDDEN_PHRASES = [
        "guarantee",
        "risk free",
        "risk-free",
        "proven roi",
        "unprecedented",
        "game changing",
        "game-changing",
        "disruptive"
    ]

    def __init__(self):
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[str, Any]:
        return {
            "english_only": True,
            "persona_required": True,
            "min_claims": 1,
            "min_confidence": 3,
            "no_forbidden_phrases": True,
            "no_dashes": True
        }

    def validate_content(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = {
            "valid": True,
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "score": 0.0
        }

        if self._check_english_only(content):
            result["checks_passed"].append("english_only")
        else:
            result["checks_failed"].append("english_only")
            result["valid"] = False

        persona = (metadata.get("persona") or "").lower().replace(" ", "_")
        known_personas = ["decision_maker", "practitioner", "researcher", "technical_buyer",
                          "influencer", "cto", "startup_founder", "vp_engineering"]
        if persona and (persona in known_personas or any(k in persona for k in ["maker", "engineer", "architect", "cto", "vp", "founder", "lead", "manager", "director"])):
            result["checks_passed"].append("persona_set")
        elif persona:
            # Accept any non-empty persona — the platform supports custom personas
            result["checks_passed"].append("persona_set")
        else:
            result["checks_failed"].append("persona_set")
            result["valid"] = False

        claim_ids = self._extract_claim_ids(content)
        if len(claim_ids) >= 1:
            result["checks_passed"].append("has_claims")
        else:
            result["checks_failed"].append("has_claims")
            result["valid"] = False

        if self._check_claims_for_quantitative(content, claim_ids):
            result["checks_passed"].append("claims_cited")
        else:
            result["checks_failed"].append("claims_cited")
            result["warnings"].append("Quantitative statements may lack claim citations")

        forbidden_found = self._check_forbidden_phrases(content)
        if not forbidden_found:
            result["checks_passed"].append("no_forbidden_phrases")
        else:
            result["checks_failed"].append("no_forbidden_phrases")
            result["warnings"].append(f"Found forbidden phrases: {', '.join(forbidden_found)}")
            result["valid"] = False

        if self._check_no_dashes(content):
            result["checks_passed"].append("no_dashes")
        else:
            result["checks_failed"].append("no_dashes")
            result["warnings"].append("Found en-dash (–) or em-dash (—). Use commas or parentheses instead.")

        total_checks = len(result["checks_passed"]) + len(result["checks_failed"])
        if total_checks > 0:
            result["score"] = len(result["checks_passed"]) / total_checks

        logger.info(
            f"HITL preflight validation: {len(result['checks_passed'])}/{total_checks} passed, "
            f"score: {result['score']:.2f}"
        )

        return result

    def _check_english_only(self, content: str) -> bool:
        ascii_chars = sum(1 for c in content if ord(c) < 128)
        return ascii_chars / len(content) > 0.9 if content else False

    def _extract_claim_ids(self, content: str) -> List[str]:
        """Extract claim IDs from content, supporting multiple citation formats:
        - [CLM_003] (standard)
        - [CLAIM_ID:CLM_003] (prefixed)
        - [CLAIM_ID: CLM_003] (prefixed with space)
        """
        found = set()
        # Standard format: [CLM_003]
        for m in re.findall(r'\[CLM_(\d{3})\]', content):
            found.add(f'CLM_{m}')
        # Prefixed format: [CLAIM_ID:CLM_003] or [CLAIM_ID: CLM_003]
        for m in re.findall(r'\[CLAIM_ID:\s*(CLM_\d{3})\]', content, re.IGNORECASE):
            found.add(m)
        return list(found)

    def _check_claims_for_quantitative(
        self,
        content: str,
        claim_ids: List[str]
    ) -> bool:
        # Heuristic: any number/percentage/dollar amount without a claim citation
        # within 100 chars is flagged as an unsupported quantitative claim.
        quant_patterns = [
            r'\d+%',
            r'\d+x',
            r'\$\d+',
            r'\d+\s*(times|fold)',
        ]
        # Match any citation format within context window
        citation_pattern = r'\[(?:CLAIM_ID:\s*)?CLM_\d{3}\]'

        for pattern in quant_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]

                if not re.search(citation_pattern, context, re.IGNORECASE):
                    return False

        return True

    def _check_forbidden_phrases(self, content: str) -> List[str]:
        content_lower = content.lower()
        found = []

        for phrase in self.FORBIDDEN_PHRASES:
            if phrase.lower() in content_lower:
                found.append(phrase)

        return found

    def _check_no_dashes(self, content: str) -> bool:
        # En-dash: \u2013, Em-dash: \u2014
        if '\u2013' in content or '\u2014' in content:
            return False
        return True

    def format_validation_report(
        self,
        validation_result: Dict[str, Any]
    ) -> str:
        lines = ["=== HITL Preflight Validation Report ===", ""]

        lines.append(f"Overall Status: {'✅ PASSED' if validation_result['valid'] else '❌ FAILED'}")
        lines.append(f"Score: {validation_result['score']:.0%}")
        lines.append("")

        if validation_result["checks_passed"]:
            lines.append("✅ Passed Checks:")
            for check in validation_result["checks_passed"]:
                lines.append(f"  - {check}")
            lines.append("")

        if validation_result["checks_failed"]:
            lines.append("❌ Failed Checks:")
            for check in validation_result["checks_failed"]:
                lines.append(f"  - {check}")
            lines.append("")

        if validation_result["warnings"]:
            lines.append("⚠️  Warnings:")
            for warning in validation_result["warnings"]:
                lines.append(f"  - {warning}")
            lines.append("")

        return "\n".join(lines)


def validate_before_hitl(
    content: str,
    metadata: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    validator = HITLPreflightValidator()
    result = validator.validate_content(content, metadata)
    return result["valid"], result
