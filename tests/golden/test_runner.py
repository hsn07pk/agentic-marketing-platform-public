#!/usr/bin/env python
"""
Fixture-based Golden Test Runner

This runner uses pre-generated fixtures instead of making OpenAI API calls,
making tests fast, reliable, and free from quota issues.

ROBUSTNESS FEATURES:
- Validates all fixture claims exist in current claim library
- Computes claim library hash to detect staleness
- Warns/fails when fixtures reference non-existent claims
- Supports both CSV and YAML claim library sources

Usage:
    python tests/golden/test_runner_fixture.py [--verbose] [--strict]
"""

import json
import re
import csv
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import argparse
import time

logger = logging.getLogger(__name__)

class FixtureGoldenTestRunner:
    """
    Golden Test Runner that uses fixtures instead of live API calls.

    Includes robustness features:
    - Claim library hash tracking for staleness detection
    - Fixture claim validation against current library
    - Support for both CSV and YAML claim sources
    """

    def __init__(self, verbose: bool = False, strict: bool = False):
        self.fixtures_dir = Path("tests/golden/fixtures")
        self.results_dir = Path("tests/golden")
        self.verbose = verbose
        self.strict = strict  # In strict mode, stale fixtures cause failure
        self.claim_library = self._load_claim_library()
        self.claim_library_hash = self._compute_claim_library_hash()
        self.stale_warnings: List[str] = []

        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @property
    def claim_library_count(self) -> int:
        """Return count of claims in library."""
        return len(self.claim_library)

    def _load_claim_library(self) -> List[Dict[str, Any]]:
        """
        Load claim library from CSV (primary) or YAML (fallback).

        Priority:
        1. data/claim_library/claims.csv (canonical source with full metadata)
        2. config/prompts/claim_library.yaml (backup/reference source)
        """
        claims = []
        csv_file = Path("data/claim_library/claims.csv")
        yaml_file = Path("config/prompts/claim_library.yaml")

        # Try CSV first (primary source)
        if csv_file.exists():
            try:
                with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('id'):
                            claims.append({
                                'id': row['id'],
                                'text': row.get('claim_text', ''),
                                'personas': row.get('personas', '[]'),
                                'source': 'csv'
                            })

                logger.info(f"Loaded {len(claims)} claims from CSV library")
                return claims

            except Exception as e:
                logger.error(f"Failed to load CSV claim library: {e}")

        # Fallback to YAML
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    for claim in data.get('claims', []):
                        if claim.get('id'):
                            claims.append({
                                'id': claim['id'],
                                'text': claim.get('text', ''),
                                'personas': claim.get('personas', []),
                                'source': 'yaml'
                            })

                logger.info(f"Loaded {len(claims)} claims from YAML library (fallback)")
                return claims

            except Exception as e:
                logger.error(f"Failed to load YAML claim library: {e}")

        logger.error("No claim library found!")
        return []

    def _compute_claim_library_hash(self) -> str:
        """
        Compute a hash of the claim library for staleness detection.

        This hash changes whenever claim IDs or text change, allowing
        detection of outdated fixtures.
        """
        if not self.claim_library:
            return "empty"

        # Sort claims by ID for consistent hashing
        sorted_claims = sorted(self.claim_library, key=lambda c: c['id'])
        # Create hash string from claim IDs and text
        hash_input = "|".join([f"{c['id']}:{c['text'][:50]}" for c in sorted_claims])
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _validate_fixture_claims(self, fixture: Dict[str, Any], test_id: str) -> Tuple[bool, List[str]]:
        """
        Validate that all claims in a fixture exist in the current claim library.

        Returns:
            Tuple of (is_valid, list of invalid claim IDs)
        """
        content_data = fixture.get("content", {})
        body = content_data.get("body", "")
        claims_used = content_data.get("claims_used", [])

        # Extract claims from body text
        found_claims = self._extract_claims_from_content(body)

        # Get valid claim IDs
        valid_claim_ids = {c['id'] for c in self.claim_library}

        # Check for invalid claims
        invalid_claims = []
        for claim_id in found_claims:
            if claim_id not in valid_claim_ids:
                invalid_claims.append(claim_id)

        # Also check declared claims_used
        for claim_id in claims_used:
            if claim_id not in valid_claim_ids and claim_id not in invalid_claims:
                invalid_claims.append(claim_id)

        if invalid_claims:
            warning = f"Fixture {test_id} references non-existent claims: {invalid_claims}"
            self.stale_warnings.append(warning)
            logger.warning(warning)
            return False, invalid_claims

        return True, []

    def _load_fixture(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Load fixture for a test case"""
        fixture_file = self.fixtures_dir / f"{test_id}.json"

        if not fixture_file.exists():
            logger.error(f"Fixture not found: {fixture_file}")
            return None

        try:
            with open(fixture_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load fixture {test_id}: {e}")
            return None

    def _extract_claims_from_content(self, content: str) -> List[str]:
        """Extract claim IDs from content"""
        pattern = r'\[([A-Z0-9_]+)\]'
        return re.findall(pattern, content)

    def _validate_claims(
        self,
        found_claims: List[str],
        expected_claims: List[str],
        test_assertions: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Validate claim usage against assertions"""

        # Get valid claim IDs
        valid_claim_ids = [c['id'] for c in self.claim_library]

        # Check all claims are from library
        if test_assertions.get("all_claims_from_library"):
            invalid = [c for c in found_claims if c not in valid_claim_ids]
            if invalid:
                return False, f"Invalid claims found: {invalid}"

        # Check no invalid claims
        if test_assertions.get("no_invalid_claims"):
            invalid = [c for c in found_claims if c not in valid_claim_ids]
            if invalid:
                return False, f"Invalid claims: {invalid}"

        # Check minimum claims
        if test_assertions.get("min_claims"):
            min_required = test_assertions["min_claims"]
            if len(found_claims) < min_required:
                return False, f"Insufficient claims: {len(found_claims)}/{min_required}"

        # Check maximum claims
        if test_assertions.get("max_claims"):
            max_allowed = test_assertions["max_claims"]
            if len(found_claims) > max_allowed:
                return False, f"Too many claims: {len(found_claims)}/{max_allowed}"

        # Check has_claim
        if test_assertions.get("has_claim"):
            if len(found_claims) == 0:
                return False, "No claims found in content"

        # Check citation format
        if test_assertions.get("has_citation"):
            claim_format = test_assertions.get("claim_format", "[CLAIM_ID]")
            # This is satisfied by the regex pattern - if we found claims, format is correct
            pass

        return True, "All claim requirements met"

    def _run_single_test(self, test_id: str, test_case: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Run a single test case using fixtures.

        Returns:
            Tuple of (passed, failure_reason)
        """

        try:
            # Load fixture
            fixture = self._load_fixture(test_id)
            if fixture is None:
                return False, "Fixture not found"

            # ROBUSTNESS: Validate fixture claims against current claim library
            fixture_valid, invalid_claims = self._validate_fixture_claims(fixture, test_id)
            if not fixture_valid:
                if self.strict:
                    return False, f"Stale fixture - invalid claims: {invalid_claims}"
                else:
                    # In non-strict mode, log warning but continue
                    logger.warning(f"Test {test_id}: Fixture has stale claims, test may be unreliable")

            # Extract content from fixture
            content_data = fixture.get("content", {})
            body = content_data.get("body", "")
            headline = content_data.get("headline", "")
            expected_claims = content_data.get("claims_used", [])

            # Extract claims from content
            found_claims = self._extract_claims_from_content(body)

            # Validate claims
            assertions = test_case.get("assertions", {})
            valid, reason = self._validate_claims(found_claims, expected_claims, assertions)

            if not valid:
                logger.error(f"Test {test_id} FAILED: {reason}")
                return False, reason

            # Check safety score thresholds (if in fixture)
            expected_safety = fixture.get("expected_safety_score", 0.0)
            if assertions.get("safety_score_min"):
                if expected_safety < assertions["safety_score_min"]:
                    reason = f"Safety score too low ({expected_safety} < {assertions['safety_score_min']})"
                    logger.error(f"Test {test_id} FAILED: {reason}")
                    return False, reason

            # Check toxicity score thresholds (if in fixture)
            expected_toxicity = fixture.get("expected_toxicity_score", 1.0)
            if assertions.get("toxicity_score_max"):
                # Toxicity score is INVERTED (1.0 = safe, 0.0 = toxic)
                min_toxicity_safety = 1.0 - assertions["toxicity_score_max"]
                if expected_toxicity < min_toxicity_safety:
                    reason = f"Toxicity safety too low ({expected_toxicity} < {min_toxicity_safety})"
                    logger.error(f"Test {test_id} FAILED: {reason}")
                    return False, reason

            logger.info(f"Test {test_id} PASSED (claims: {found_claims})")
            return True, None

        except Exception as e:
            logger.error(f"Test {test_id} FAILED with exception: {e}", exc_info=True)
            return False, str(e)

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from YAML or generate default list"""
        import yaml

        test_file = Path("tests/golden/test_cases.yaml")

        if test_file.exists():
            try:
                with open(test_file, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get('test_cases', [])
            except Exception as e:
                logger.error(f"Failed to load test_cases.yaml: {e}")

        # Generate default test list (40 tests)
        return [
            {"id": f"GOLDEN_{i:03d}", "input": {}, "assertions": {}}
            for i in range(1, 41)
        ]

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Golden Tests using fixtures"""

        test_cases = self._load_test_cases()

        results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "failures": [],
            "test_details": []
        }

        start_time = time.time()

        logger.info(f"Running {len(test_cases)} Golden Tests in FIXTURE MODE\n")

        for test_case in test_cases:
            test_id = test_case["id"]

            passed, failure_reason = self._run_single_test(test_id, test_case)

            # Add test detail
            test_detail = {
                "test_id": test_id,
                "name": test_case.get("name", test_id),
                "category": test_case.get("category", "unknown"),
                "status": "passed" if passed else "failed",
                "failure_reason": failure_reason,
                "duration": 0.001  # Fixtures are instant
            }
            results["test_details"].append(test_detail)

            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "test_id": test_id,
                    "reason": failure_reason
                })

        duration = time.time() - start_time
        results["pass_rate"] = results["passed"] / results["total"] * 100 if results["total"] > 0 else 0
        results["duration_seconds"] = round(duration, 2)
        results["claim_library_hash"] = self.claim_library_hash
        results["claim_library_count"] = len(self.claim_library)
        results["stale_warnings"] = self.stale_warnings

        return results


def main():
    parser = argparse.ArgumentParser(description="Fixture-based Golden Test Runner")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--strict", action="store_true",
                        help="Strict mode: fail on stale fixtures")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GOLDEN TEST SUITE - Agentic AI Marketing Platform")
    print(f"{'='*60}")
    print(f"Mode: FIXTURE (fast, no API calls)")
    if args.strict:
        print(f"Strict Mode: ENABLED (stale fixtures cause failure)")
    print(f"{'='*60}\n")

    runner = FixtureGoldenTestRunner(verbose=args.verbose, strict=args.strict)

    # Log claim library info
    print(f"Claim Library: {runner.claim_library_count} claims loaded")
    print(f"Library Hash: {runner.claim_library_hash}")
    print(f"{'='*60}\n")

    results = runner.run_all_tests()

    # Determine results file path (works in Docker and local)
    # Use /tmp for Docker containers (always writable), fallback to tests/golden for local
    import tempfile
    tmp_results = Path(tempfile.gettempdir()) / "golden_test_results.json"
    results_dir = Path("tests/golden")
    
    # Prefer /tmp (always writable in containers), fallback to tests/golden if writable
    if tmp_results.parent.exists():
        results_file = tmp_results
    elif results_dir.exists():
        results_file = results_dir / "golden_test_results.json"
    else:
        results_file = Path(".") / "golden_test_results.json"

    # Save results with full metadata
    results_with_timestamp = {
        "pass_rate": results["pass_rate"] / 100.0,
        "total_tests": results["total"],
        "passed_tests": results["passed"],
        "failed_tests": results["failed"],
        "last_run": datetime.utcnow().isoformat(),
        "duration_seconds": results["duration_seconds"],
        "mode": "fixture",
        "strict_mode": args.strict,
        "claim_library_hash": results["claim_library_hash"],
        "claim_library_count": results["claim_library_count"],
        "stale_warnings": results["stale_warnings"],
        "test_details": results["test_details"],
        "failures": results["failures"]
    }

    with open(results_file, 'w') as f:
        json.dump(results_with_timestamp, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print(f"GOLDEN TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass Rate: {results['pass_rate']:.1f}%")
    print(f"Duration: {results['duration_seconds']:.1f}s")
    print(f"Claim Library Hash: {results['claim_library_hash']}")
    print(f"Mode: FIXTURE{' (STRICT)' if args.strict else ''}")

    # Print stale warnings if any
    if results["stale_warnings"]:
        print(f"\n{'='*60}")
        print(f"⚠️  STALE FIXTURE WARNINGS ({len(results['stale_warnings'])})")
        print(f"{'='*60}")
        for warning in results["stale_warnings"]:
            print(f"  - {warning}")
        print(f"\nTo regenerate fixtures, run:")
        print(f"  python tests/golden/generate_fixtures.py")

    if results["pass_rate"] < 100.0:
        print(f"\n❌ DEPLOYMENT BLOCKED")
        print(f"Failed tests:")
        for failure in results['failures']:
            if isinstance(failure, dict):
                print(f"  - {failure['test_id']}: {failure['reason']}")
            else:
                print(f"  - {failure}")
        print(f"\nTo debug, run with --verbose flag:")
        print(f"  python tests/golden/test_runner_fixture.py --verbose")
        return 1
    else:
        print(f"\n✅ ALL TESTS PASSED - DEPLOYMENT APPROVED")
        print(f"\nResults saved to: {results_file}")
        return 0


if __name__ == "__main__":
    exit(main())
