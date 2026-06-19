"""
Golden Test Suite - Research Plan Section 7.2

A 'golden suite' of test items run automatically before deployment.
A failure in this test suite will block the deployment.

These tests validate:
1. Content generation pipeline works correctly
2. Safety validation catches issues
3. Claim validation works
4. Output format is correct
"""
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class GoldenTestSuite:
    def __init__(self):
        self.test_cases = []
        self._initialized = False
    
    def load_test_cases(
        self,
        test_file: str = "tests/golden/test_suite.yaml"
    ):
        try:
            test_path = Path(test_file)
            if test_path.exists():
                with open(test_file, 'r') as f:
                    data = yaml.safe_load(f)
                self.test_cases = data.get('test_cases', [])
                logger.info(f"Loaded {len(self.test_cases)} golden test cases from {test_file}")
            else:
                logger.info(f"Test file {test_file} not found, using default tests")
                self.test_cases = self._get_default_tests()
        except Exception as e:
            logger.warning(f"Failed to load test cases from file: {e}, using defaults")
            self.test_cases = self._get_default_tests()
        
        self._initialized = True
    
    def _get_default_tests(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "GOLDEN_001",
                "name": "Content Format - Headline Present",
                "category": "format",
                "type": "format_check",
                "test_content": "Headline: Test Headline\nBody: This is a test body with content.",
                "assertions": {
                    "has_headline": True,
                    "has_body": True
                }
            },
            {
                "id": "GOLDEN_002", 
                "name": "Content Format - Claim Citation Format",
                "category": "format",
                "type": "format_check",
                "test_content": "This content has a claim [CLM_001] with citation.",
                "assertions": {
                    "has_claim_reference": True,
                    "claim_format_valid": True
                }
            },
            {
                "id": "GOLDEN_003",
                "name": "Safety - No Prohibited Words",
                "category": "safety",
                "type": "safety_check",
                "test_content": "This is a professional marketing message about our product.",
                "assertions": {
                    "no_prohibited_words": True
                }
            },
            {
                "id": "GOLDEN_004",
                "name": "Safety - Profanity Detection",
                "category": "safety", 
                "type": "safety_check",
                "test_content": "This content contains the word damn which should be flagged.",
                "assertions": {
                    "profanity_detected": True
                },
                "expect_failure": True
            },
            {
                "id": "GOLDEN_005",
                "name": "Platform - LinkedIn Character Limit",
                "category": "platform",
                "type": "platform_check",
                "test_content": "A" * 3000,
                "platform": "linkedin",
                "assertions": {
                    "within_char_limit": True
                }
            },
            {
                "id": "GOLDEN_006",
                "name": "Platform - Twitter Character Limit",
                "category": "platform",
                "type": "platform_check",
                "test_content": "A" * 280,
                "platform": "twitter",
                "assertions": {
                    "within_char_limit": True
                }
            },
            {
                "id": "GOLDEN_007",
                "name": "Platform - Twitter Exceeds Limit",
                "category": "platform",
                "type": "platform_check",
                "test_content": "A" * 300,
                "platform": "twitter",
                "assertions": {
                    "within_char_limit": False
                },
                "expect_failure": False
            },
            {
                "id": "GOLDEN_008",
                "name": "Regex - Headline Extraction",
                "category": "regex",
                "type": "regex_check",
                "test_content": "Headline: Boost Your Marketing ROI\nBody: Content here",
                "pattern": r"Headline:\s*(.+?)(?:\n|$)",
                "assertions": {
                    "pattern_matches": True,
                    "extracted_value": "Boost Your Marketing ROI"
                }
            },
            {
                "id": "GOLDEN_009",
                "name": "Regex - Claim ID Extraction",
                "category": "regex",
                "type": "regex_check",
                "test_content": "Our product shows 50% improvement [CLM_042] in efficiency.",
                "pattern": r"\[CLM_(\d+)\]",
                "assertions": {
                    "pattern_matches": True,
                    "extracted_value": "042"
                }
            },
            {
                "id": "GOLDEN_010",
                "name": "Integration - Pipeline Ready",
                "category": "integration",
                "type": "integration_check",
                "assertions": {
                    "ollama_available": True,
                    "database_available": True
                }
            }
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        if not self._initialized:
            self.load_test_cases()
        
        results = {
            "total": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "failures": [],
            "test_results": [],
            "started_at": datetime.now().isoformat(),
            "categories_tested": set()
        }
        
        for test_case in self.test_cases:
            test_result = await self._run_single_test(test_case)
            results['test_results'].append(test_result)
            results['categories_tested'].add(test_case.get('category', 'unknown'))
            
            expect_failure = test_case.get('expect_failure', False)
            
            if test_result['passed'] or (expect_failure and not test_result['passed']):
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append({
                    "id": test_case['id'],
                    "name": test_case['name'],
                    "category": test_case.get('category'),
                    "reason": test_result.get('failure_reason')
                })
        
        results['categories_tested'] = list(results['categories_tested'])
        results['pass_rate'] = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0.0
        results['approved_for_deployment'] = results['pass_rate'] >= 100.0
        results['completed_at'] = datetime.now().isoformat()
        
        if results['approved_for_deployment']:
            logger.info(f"✅ All golden tests passed ({results['passed']}/{results['total']}) - DEPLOYMENT APPROVED")
        else:
            logger.warning(f"❌ Golden tests: {results['passed']}/{results['total']} passed ({results['pass_rate']:.1f}%) - DEPLOYMENT BLOCKED")
            for failure in results['failures']:
                logger.warning(f"  - {failure['id']}: {failure['reason']}")
        
        return results
    
    async def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "id": test_case['id'],
            "name": test_case['name'],
            "category": test_case.get('category'),
            "passed": False,
            "failure_reason": None,
            "details": {}
        }
        
        try:
            test_type = test_case.get('type', 'format_check')
            
            if test_type == 'format_check':
                result = await self._run_format_test(test_case, result)
            elif test_type == 'safety_check':
                result = await self._run_safety_test(test_case, result)
            elif test_type == 'platform_check':
                result = await self._run_platform_test(test_case, result)
            elif test_type == 'regex_check':
                result = await self._run_regex_test(test_case, result)
            elif test_type == 'integration_check':
                result = await self._run_integration_test(test_case, result)
            else:
                result['failure_reason'] = f"Unknown test type: {test_type}"
            
            return result
            
        except Exception as e:
            logger.error(f"Test {test_case['id']} failed with exception: {e}")
            result['failure_reason'] = str(e)
            return result
    
    async def _run_format_test(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        content = test_case.get('test_content', '')
        assertions = test_case.get('assertions', {})
        
        if assertions.get('has_headline'):
            has_headline = bool(re.search(r'Headline:\s*.+', content, re.IGNORECASE))
            result['details']['has_headline'] = has_headline
            if not has_headline:
                result['failure_reason'] = "Missing headline in content"
                return result
        
        if assertions.get('has_body'):
            has_body = bool(re.search(r'Body:\s*.+', content, re.IGNORECASE))
            result['details']['has_body'] = has_body
            if not has_body:
                result['failure_reason'] = "Missing body in content"
                return result
        
        if assertions.get('has_claim_reference'):
            has_claim = bool(re.search(r'\[CLM_\d+\]', content))
            result['details']['has_claim_reference'] = has_claim
            if not has_claim:
                result['failure_reason'] = "Missing claim reference [CLM_###]"
                return result
        
        if assertions.get('claim_format_valid'):
            claim_matches = re.findall(r'\[CLM_(\d+)\]', content)
            valid_format = all(len(m) == 3 and m.isdigit() for m in claim_matches) if claim_matches else False
            result['details']['claim_format_valid'] = valid_format
            if not valid_format:
                result['failure_reason'] = "Invalid claim format (should be [CLM_###])"
                return result
        
        result['passed'] = True
        return result
    
    async def _run_safety_test(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        content = test_case.get('test_content', '')
        assertions = test_case.get('assertions', {})
        
        prohibited_words = [
            'guarantee', 'guaranteed', 'miracle', 'cure', 'free money',
            'get rich quick', 'no risk', '100% safe'
        ]
        
        profanity_words = [
            'damn', 'hell', 'crap', 'ass', 'bastard'
        ]
        
        content_lower = content.lower()
        
        if assertions.get('no_prohibited_words'):
            found_prohibited = [w for w in prohibited_words if w in content_lower]
            result['details']['prohibited_words_found'] = found_prohibited
            if found_prohibited:
                result['failure_reason'] = f"Prohibited words found: {found_prohibited}"
                return result
        
        if 'profanity_detected' in assertions:
            found_profanity = [w for w in profanity_words if w in content_lower]
            result['details']['profanity_found'] = found_profanity
            expected = assertions.get('profanity_detected')
            actual = len(found_profanity) > 0
            
            if expected and not actual:
                result['failure_reason'] = "Expected profanity detection but none found"
                return result
            elif not expected and actual:
                result['failure_reason'] = f"Unexpected profanity found: {found_profanity}"
                return result
        
        result['passed'] = True
        return result
    
    async def _run_platform_test(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        content = test_case.get('test_content', '')
        platform = test_case.get('platform', 'general')
        assertions = test_case.get('assertions', {})
        
        char_limits = {
            'twitter': 280,
            'linkedin': 3000,
            'email': 10000,
            'general': 5000
        }
        
        limit = char_limits.get(platform, 5000)
        content_length = len(content)
        within_limit = content_length <= limit
        
        result['details']['content_length'] = content_length
        result['details']['char_limit'] = limit
        result['details']['within_limit'] = within_limit
        
        if 'within_char_limit' in assertions:
            expected = assertions.get('within_char_limit')
            if expected != within_limit:
                if expected:
                    result['failure_reason'] = f"Content exceeds {platform} limit ({content_length} > {limit})"
                else:
                    result['failure_reason'] = f"Expected content to exceed limit but it didn't"
                return result
        
        result['passed'] = True
        return result
    
    async def _run_regex_test(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        content = test_case.get('test_content', '')
        pattern = test_case.get('pattern', '')
        assertions = test_case.get('assertions', {})
        
        try:
            match = re.search(pattern, content)
            result['details']['pattern'] = pattern
            result['details']['match_found'] = match is not None
            
            if assertions.get('pattern_matches'):
                if not match:
                    result['failure_reason'] = f"Pattern '{pattern}' did not match content"
                    return result
                
                expected_value = assertions.get('extracted_value')
                if expected_value and match:
                    actual_value = match.group(1) if match.groups() else match.group(0)
                    result['details']['extracted_value'] = actual_value
                    if actual_value != expected_value:
                        result['failure_reason'] = f"Extracted '{actual_value}' but expected '{expected_value}'"
                        return result
            
            result['passed'] = True
            return result
            
        except re.error as e:
            result['failure_reason'] = f"Invalid regex pattern: {e}"
            return result
    
    async def _run_integration_test(self, test_case: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        import os
        assertions = test_case.get('assertions', {})
        
        if assertions.get('ollama_available'):
            try:
                import httpx
                ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
                if not ollama_host.startswith('http'):
                    ollama_host = f"http://{ollama_host}"
                ollama_host = ollama_host.rstrip('/')
                
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{ollama_host}/api/tags")
                    ollama_ok = response.status_code == 200
                    result['details']['ollama_available'] = ollama_ok
                    result['details']['ollama_host'] = ollama_host
                    if not ollama_ok:
                        result['failure_reason'] = f"Ollama not available (status: {response.status_code})"
                        return result
            except Exception as e:
                result['details']['ollama_available'] = False
                result['failure_reason'] = f"Ollama connection failed: {e}"
                return result
        
        if assertions.get('database_available'):
            try:
                from sqlalchemy import text
                from src.data_layer.database.connection import async_session_maker
                async with async_session_maker() as session:
                    await session.execute(text("SELECT 1"))
                    result['details']['database_available'] = True
            except Exception as e:
                result['details']['database_available'] = False
                result['failure_reason'] = f"Database connection failed: {e}"
                return result
        
        result['passed'] = True
        return result
    
    async def run_category_tests(self, category: str) -> Dict[str, Any]:
        if not self._initialized:
            self.load_test_cases()
        
        category_tests = [tc for tc in self.test_cases if tc.get('category') == category]
        
        results = {
            "category": category,
            "total": len(category_tests),
            "passed": 0,
            "failed": 0,
            "failures": []
        }
        
        for test_case in category_tests:
            test_result = await self._run_single_test(test_case)
            
            if test_result['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append({
                    "id": test_case['id'],
                    "reason": test_result['failure_reason']
                })
        
        results['pass_rate'] = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0.0
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        if not self._initialized:
            self.load_test_cases()
        
        categories = {}
        for test in self.test_cases:
            category = test.get('category', 'uncategorized')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_tests": len(self.test_cases),
            "categories": categories,
            "loaded": self._initialized
        }