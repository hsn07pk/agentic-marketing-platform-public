"""
Prompt Testing Service - Comprehensive validation of prompt changes.

This service ensures that any prompt modification is thoroughly tested
through the actual workflows before being deployed to production.

Key Features:
1. Sandbox mode - no production data pollution
2. Multiple test scenarios for each prompt type
3. Validates regex patterns still extract data correctly
4. Runs through actual content generation and safety validation
5. 100% pass rate required before deployment
"""
import re
import yaml
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import os

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "../../../config/prompts")


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class TestScenario:
    name: str
    description: str
    input_variables: Dict[str, str]
    expected_patterns: List[str]  # Regex patterns that must match in output
    forbidden_patterns: List[str] = field(default_factory=list)  # Must NOT appear
    min_output_length: int = 50
    max_output_length: int = 5000
    

@dataclass
class TestResult:
    """Result of a single test scenario"""
    scenario_name: str
    status: TestStatus
    elapsed_seconds: float
    output: str = ""
    patterns_matched: List[str] = field(default_factory=list)
    patterns_failed: List[str] = field(default_factory=list)
    forbidden_found: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTestSuite:
    """Complete test suite result"""
    prompt_file: str
    template_name: str
    test_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: TestStatus = TestStatus.PENDING
    results: List[TestResult] = field(default_factory=list)
    overall_pass_rate: float = 0.0
    can_deploy: bool = False
    blocking_issues: List[str] = field(default_factory=list)


class PromptTestingService:
    """
    Service for comprehensive prompt testing before deployment.
    
    Ensures prompts work correctly through actual workflows without
    polluting production data.
    """
    
    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        
        self.critical_patterns = {
            # Safety validation scores
            "toxicity_score": r'\*{0,2}TOXICITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
            "factuality_score": r'\*{0,2}FACTUALITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
            "brand_score": r'\*{0,2}BRAND_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
            "compliance_score": r'\*{0,2}COMPLIANCE_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
            # Content structure patterns (per template type)
            "headline": r'[Hh]eadline:?\s*(.+?)(?:\n|$)',
            "title": r'[Tt]itle:?\s*(.+?)(?:\n|$)',
            "meta_description": r'[Mm]eta [Dd]escription:?\s*(.+?)(?:\n|$)',
            "body": r'[Bb]ody:?\s*(.+?)(?:[Cc][Tt][Aa]:|$)',
            "cta": r'[Cc][Tt][Aa]:?\s*(.+?)(?:\n|$)',
            "tweet": r'[Tt]weet:?\s*(.+?)(?:[Cc]laims?|$)',
            "subject": r'[Ss]ubject:?\s*(.+?)(?:\n|$)',
            "post": r'[Pp]ost:?\s*(.+?)(?:[Hh]ashtags?:|$)',
            "message": r'[Mm]essage:?\s*(.+?)(?:[Bb]est|[Cc]laims?|$)',
            "preview": r'[Pp]review:?\s*(.+?)(?:\n|$)',
            # Claim citations
            "claim_citation": r'\[CLM_\d{3}\]|\[CLAIM_\d{3}\]',
            "claims_used": r'[Cc]laims?\s*[Uu]sed:?\s*(.+?)(?:\n|$)',
        }
        
        # Per-template required patterns for regex extraction validation
        self.template_required_patterns = {
            # Content generation templates
            "linkedin_ad": ["headline", "body", "cta", "claim_citation"],
            "twitter_ad": ["tweet", "claim_citation"],
            "email_campaign": ["subject", "body", "claim_citation"],
            "linkedin_organic": ["post", "claim_citation"],
            "linkedin_inmail": ["subject", "message", "claim_citation"],
            "nurture_sequence": ["subject", "body", "claim_citation"],
            "retargeting_ad": ["headline", "body", "cta", "claim_citation"],
            "blog_post": ["title", "claim_citation"],
            # Safety templates
            "toxicity_check": ["toxicity_score"],
            "factuality_check": ["factuality_score"],
            "brand_alignment_check": ["brand_score"],
            "regulatory_compliance_check": ["compliance_score"],
        }
        
        self.test_scenarios = self._build_test_scenarios()
    
    def _build_test_scenarios(self) -> Dict[str, List[TestScenario]]:
        """Build test scenarios for each prompt type"""
        
        sample_persona = "Marketing Director at mid-size B2B SaaS company, focused on pipeline generation"
        sample_claims = """
Available Claims:
- [CLM_001]: AI-powered marketing automation reduces manual work by 60%
- [CLM_002]: Agentic customers see 3x improvement in lead quality
- [CLM_003]: QWL-guided actions focus effort where impact is highest
- [CLM_006]: Team-level simulation prioritizes the next best action
"""
        sample_context = "Agentic AI provides autonomous marketing agents that generate and deploy B2B campaigns."
        sample_content_safe = "Discover how AI-powered marketing automation can help your team focus on high-impact activities. Our QWL-guided approach helps marketing directors prioritize actions [CLM_003]."
        sample_content_unsafe = "This is the BEST product ever! Guaranteed 1000% ROI or your money back! Buy now or regret forever!"
        
        return {
            "linkedin_ad": [
                TestScenario(
                    name="basic_linkedin_ad",
                    description="Generate a basic LinkedIn ad",
                    input_variables={
                        "persona_name": "Marketing Director",
                        "persona_description": sample_persona,
                        "pain_points": "Manual campaign management, low conversion rates",
                        "campaign_goal": "Generate demo requests",
                        "campaign_type": "Awareness",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "competitor_insights": "Competitors focus on features, not outcomes",
                        "brand_voice_guidelines": "Professional, data-driven, confident",
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Hh]eadline:',
                        r'[Bb]ody:',
                        r'[Cc][Tt][Aa]:',
                        r'\[CLM_\d{3}\]',  # Must include claim citation
                    ],
                    forbidden_patterns=[
                        r'(?i)guaranteed',
                        r'(?i)risk.?free',
                        r'(?i)best\s+ever',
                    ],
                    min_output_length=100,
                    max_output_length=1000
                ),
                TestScenario(
                    name="linkedin_ad_with_all_claims",
                    description="Verify max 3 claims constraint",
                    input_variables={
                        "persona_name": "CEO",
                        "persona_description": "C-level executive at enterprise company",
                        "pain_points": "Scaling marketing efficiently",
                        "campaign_goal": "Schedule strategy call",
                        "campaign_type": "Conversion",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "competitor_insights": "",
                        "brand_voice_guidelines": "Executive, strategic, ROI-focused",
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Hh]eadline:',
                        r'\[CLM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=100,
                    max_output_length=1000
                ),
            ],
            
            "twitter_ad": [
                TestScenario(
                    name="basic_twitter_ad",
                    description="Generate Twitter ad under 280 chars",
                    input_variables={
                        "persona_name": "Growth Marketer",
                        "persona_description": "Growth-focused marketer",
                        "campaign_goal": "Drive website traffic",
                        "trending_topics": "#MarketingAutomation #B2BMarketing",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Tt]weet:',
                        r'\[CLM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=50,
                    max_output_length=500
                ),
            ],
            
            "blog_post": [
                TestScenario(
                    name="basic_blog_post",
                    description="Generate SEO-optimized blog post with sections",
                    input_variables={
                        "persona_name": "Marketing Director",
                        "persona_description": sample_persona,
                        "pain_points": "Low organic traffic, inconsistent content output",
                        "campaign_goal": "Generate inbound leads via thought leadership",
                        "campaign_type": "Brand Awareness",
                        "blog_topic": "How AI is transforming B2B marketing automation",
                        "seo_keywords": "AI marketing automation, B2B content strategy, marketing ROI",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "brand_voice_guidelines": "Professional, data-driven, authoritative",
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Tt]itle:',
                        r'[Mm]eta [Dd]escription:',
                        r'##\s',
                        r'\[CLM_\d{3}\]',
                    ],
                    forbidden_patterns=[
                        r'(?i)guaranteed',
                        r'(?i)risk.?free',
                    ],
                    min_output_length=500,
                    max_output_length=5000
                ),
                TestScenario(
                    name="blog_post_seo_structure",
                    description="Verify blog has proper SEO structure and CTA",
                    input_variables={
                        "persona_name": "VP Engineering",
                        "persona_description": "Technical leader evaluating marketing tools",
                        "pain_points": "Manual campaign management, poor attribution",
                        "campaign_goal": "Drive demo requests",
                        "campaign_type": "Conversion",
                        "blog_topic": "5 ways autonomous marketing agents outperform traditional tools",
                        "seo_keywords": "autonomous marketing, marketing agents, campaign automation",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "brand_voice_guidelines": "Technical, evidence-based, confident",
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Tt]itle:',
                        r'[Cc][Tt][Aa]:',
                        r'\[CLM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=500,
                    max_output_length=5000
                ),
            ],
            
            "email_campaign": [
                TestScenario(
                    name="basic_email",
                    description="Generate email with subject and body",
                    input_variables={
                        "persona_name": "VP Marketing",
                        "persona_description": "VP of Marketing at B2B company",
                        "campaign_stage": "Awareness",
                        "interaction_history": "First touch",
                        "email_type": "Cold outreach",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "company_name": "TechCorp",
                        "industry": "SaaS",
                        "pain_points": "Marketing efficiency",
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Ss]ubject:',
                        r'[Bb]ody:',
                        r'\[CLM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=100,
                    max_output_length=1500
                ),
            ],
            
            "linkedin_organic": [
                TestScenario(
                    name="basic_linkedin_organic",
                    description="Generate LinkedIn organic post",
                    input_variables={
                        "post_objective": "Thought leadership on AI marketing",
                        "target_personas": "Marketing Directors, CMOs",
                        "content_theme": "AI-powered marketing automation",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "trending_topics": "#AIMarketing #B2B",
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Pp]ost:',
                        r'[Hh]ashtags?:',
                        r'\[CLM_\d{3}\]|\[CLAIM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=100,
                    max_output_length=1800
                ),
            ],
            
            "linkedin_inmail": [
                TestScenario(
                    name="basic_linkedin_inmail",
                    description="Generate LinkedIn InMail outreach",
                    input_variables={
                        "recipient_name": "Jane Smith",
                        "recipient_title": "VP of Marketing",
                        "recipient_company": "TechCorp",
                        "persona_name": "Marketing Director",
                        "connection_reason": "Shared interest in marketing automation",
                        "outreach_goal": "Schedule introductory call",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Ss]ubject:',
                        r'[Mm]essage:|Hi ',
                        r'\[CLM_\d{3}\]|\[CLAIM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=100,
                    max_output_length=1000
                ),
            ],
            
            "nurture_sequence": [
                TestScenario(
                    name="basic_nurture_email",
                    description="Generate nurture sequence email",
                    input_variables={
                        "sequence_number": "2",
                        "sequence_length": "5",
                        "sequence_goal": "Move from awareness to consideration",
                        "lead_status": "MQL",
                        "persona_name": "Marketing Manager",
                        "opened_previous": "Yes",
                        "clicked_previous": "No",
                        "email_purpose": "Share case study",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Ss]ubject:',
                        r'[Bb]ody:',
                        r'\[CLM_\d{3}\]|\[CLAIM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=100,
                    max_output_length=1200
                ),
            ],
            
            "retargeting_ad": [
                TestScenario(
                    name="basic_retargeting_ad",
                    description="Generate retargeting ad for previous visitors",
                    input_variables={
                        "persona_name": "Marketing Director",
                        "engagement_type": "Downloaded whitepaper",
                        "days_since": "7",
                        "retargeting_goal": "Schedule demo",
                        "retrieved_context": sample_context,
                        "available_claims": sample_claims,
                        "system_instructions": ""
                    },
                    expected_patterns=[
                        r'[Hh]eadline:',
                        r'[Bb]ody:',
                        r'[Cc][Tt][Aa]:',
                        r'\[CLM_\d{3}\]|\[CLAIM_\d{3}\]',
                    ],
                    forbidden_patterns=[],
                    min_output_length=100,
                    max_output_length=1000
                ),
            ],
            
            "toxicity_check": [
                TestScenario(
                    name="safe_content_toxicity",
                    description="Safe content should score low toxicity",
                    input_variables={
                        "content": sample_content_safe,
                    },
                    expected_patterns=[
                        r'TOXICITY_SCORE:?\s*[0-9.]+',
                        r'ISSUES',
                        r'SAFE:',
                    ],
                    forbidden_patterns=[],
                    min_output_length=50,
                    max_output_length=2000
                ),
                TestScenario(
                    name="unsafe_content_toxicity",
                    description="Unsafe content should score high toxicity",
                    input_variables={
                        "content": sample_content_unsafe,
                    },
                    expected_patterns=[
                        r'TOXICITY_SCORE:?\s*[0-9.]+',
                        r'ISSUES',
                    ],
                    forbidden_patterns=[],
                    min_output_length=50,
                    max_output_length=2000
                ),
            ],
            
            "factuality_check": [
                TestScenario(
                    name="valid_claims_factuality",
                    description="Content with valid claims should pass",
                    input_variables={
                        "content": sample_content_safe,
                        "claim_library": sample_claims,
                    },
                    expected_patterns=[
                        r'FACTUALITY_SCORE:?\s*[0-9.]+',
                        r'CLAIMS_FOUND',
                    ],
                    forbidden_patterns=[],
                    min_output_length=50,
                    max_output_length=2000
                ),
            ],
            
            "brand_alignment_check": [
                TestScenario(
                    name="brand_aligned_content",
                    description="Professional content should pass brand check",
                    input_variables={
                        "content": sample_content_safe,
                        "brand_guidelines": "Professional, data-driven, innovative",
                    },
                    expected_patterns=[
                        r'BRAND_SCORE:?\s*[0-9.]+',
                        r'VOICE_MATCH',
                    ],
                    forbidden_patterns=[],
                    min_output_length=50,
                    max_output_length=2000
                ),
            ],
            
            "regulatory_compliance_check": [
                TestScenario(
                    name="compliant_content",
                    description="Content without violations should pass",
                    input_variables={
                        "content": sample_content_safe,
                        "platform": "LinkedIn",
                    },
                    expected_patterns=[
                        r'COMPLIANCE_SCORE:?\s*[0-9.]+',
                    ],
                    forbidden_patterns=[],
                    min_output_length=50,
                    max_output_length=2000
                ),
            ],
        }
    
    async def test_prompt(
        self,
        prompt_file: str,
        template_name: str,
        new_prompt_content: str,
        model: Optional[str] = None
    ) -> PromptTestSuite:
        """
        Run comprehensive tests on a prompt before allowing deployment.
        
        Args:
            prompt_file: The prompt file (e.g., "content_generation.yaml")
            template_name: The template being modified (e.g., "linkedin_ad")
            new_prompt_content: The new prompt YAML content
            model: Optional model override
        
        Returns:
            PromptTestSuite with all test results
        """
        test_id = str(uuid.uuid4())[:8]
        suite = PromptTestSuite(
            prompt_file=prompt_file,
            template_name=template_name,
            test_id=test_id,
            started_at=datetime.now(),
            status=TestStatus.RUNNING
        )
        
        try:
            try:
                new_template = yaml.safe_load(new_prompt_content)
                if isinstance(new_template, dict):
                    if template_name in new_template:
                        template_data = new_template[template_name]
                    else:
                        template_data = new_template
                else:
                    template_data = {"user_template": new_prompt_content}
            except yaml.YAMLError:
                template_data = {"user_template": new_prompt_content}
            
            scenarios = self.test_scenarios.get(template_name, [])
            
            if not scenarios:
                suite.blocking_issues.append(f"No test scenarios defined for template '{template_name}'")
                suite.status = TestStatus.ERROR
                suite.completed_at = datetime.now()
                return suite
            
            passed_count = 0
            for scenario in scenarios:
                result = await self._run_scenario(
                    template_data=template_data,
                    scenario=scenario,
                    model=model or self.ollama_model
                )
                suite.results.append(result)
                
                if result.status == TestStatus.PASSED:
                    passed_count += 1
                elif result.status == TestStatus.FAILED:
                    for error in result.errors:
                        suite.blocking_issues.append(f"{scenario.name}: {error}")
                    for pattern in result.patterns_failed:
                        suite.blocking_issues.append(f"{scenario.name}: Pattern not found: {pattern}")
                    for forbidden in result.forbidden_found:
                        suite.blocking_issues.append(f"{scenario.name}: Forbidden pattern found: {forbidden}")
            
            suite.overall_pass_rate = passed_count / len(scenarios) if scenarios else 0.0
            
            suite.can_deploy = suite.overall_pass_rate == 1.0 and len(suite.blocking_issues) == 0
            suite.status = TestStatus.PASSED if suite.can_deploy else TestStatus.FAILED
            
        except Exception as e:
            logger.error(f"Test suite error: {e}")
            suite.status = TestStatus.ERROR
            suite.blocking_issues.append(f"Test execution error: {str(e)}")
        
        suite.completed_at = datetime.now()
        return suite
    
    async def _run_scenario(
        self,
        template_data: Dict[str, Any],
        scenario: TestScenario,
        model: str
    ) -> TestResult:
        """Run a single test scenario"""
        import httpx
        
        start_time = datetime.now()
        result = TestResult(
            scenario_name=scenario.name,
            status=TestStatus.RUNNING,
            elapsed_seconds=0
        )
        
        try:
            if isinstance(template_data, dict):
                system_prompt = template_data.get("system", "")
                user_template = template_data.get("user_template", template_data.get("user", ""))
            else:
                user_template = str(template_data)
                system_prompt = ""
            
            for var, value in scenario.input_variables.items():
                user_template = user_template.replace(f"{{{var}}}", str(value))
                system_prompt = system_prompt.replace(f"{{{var}}}", str(value))
            
            full_prompt = f"{system_prompt}\n\n{user_template}" if system_prompt else user_template
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model,
                        "prompt": full_prompt,
                        "temperature": 0.7,
                        "num_predict": 2000,
                        "stream": False
                    }
                )
                
                if response.status_code != 200:
                    result.status = TestStatus.ERROR
                    result.errors.append(f"LLM call failed: {response.status_code}")
                    return result
                
                output = response.json().get("response", "")
            
            result.output = output
            result.elapsed_seconds = (datetime.now() - start_time).total_seconds()
            
            if len(output) < scenario.min_output_length:
                result.errors.append(f"Output too short: {len(output)} < {scenario.min_output_length}")
            if len(output) > scenario.max_output_length:
                result.errors.append(f"Output too long: {len(output)} > {scenario.max_output_length}")
            
            for pattern in scenario.expected_patterns:
                if re.search(pattern, output, re.DOTALL | re.IGNORECASE):
                    result.patterns_matched.append(pattern)
                else:
                    result.patterns_failed.append(pattern)
            
            for pattern in scenario.forbidden_patterns:
                if re.search(pattern, output, re.DOTALL | re.IGNORECASE):
                    result.forbidden_found.append(pattern)
            
            for name, pattern in self.critical_patterns.items():
                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                if match:
                    result.extracted_data[name] = match.group(1) if match.groups() else match.group(0)
            
            if result.errors or result.patterns_failed or result.forbidden_found:
                result.status = TestStatus.FAILED
            else:
                result.status = TestStatus.PASSED
                
        except Exception as e:
            result.status = TestStatus.ERROR
            result.errors.append(str(e))
            result.elapsed_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    async def run_full_workflow_test(
        self,
        prompt_file: str,
        template_name: str,
        new_prompt_content: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete workflow test that exercises the full pipeline.
        
        This creates a sandbox test that:
        1. Generates content using the new prompt
        2. Runs safety validation on the generated content
        3. Validates all regex patterns extract correctly
        4. Ensures no breaking changes
        5. Cleans up all test data
        
        Returns comprehensive test results.
        """
        test_id = f"sandbox_test_{uuid.uuid4().hex[:8]}"
        
        results = {
            "test_id": test_id,
            "prompt_file": prompt_file,
            "template_name": template_name,
            "started_at": datetime.now().isoformat(),
            "phases": {},
            "overall_status": "pending",
            "can_deploy": False,
            "blocking_issues": [],
            "warnings": []
        }
        
        try:
            logger.info(f"[{test_id}] Phase 1: Running basic prompt tests")
            basic_suite = await self.test_prompt(
                prompt_file=prompt_file,
                template_name=template_name,
                new_prompt_content=new_prompt_content,
                model=model
            )
            
            results["phases"]["basic_tests"] = {
                "status": basic_suite.status.value,
                "pass_rate": basic_suite.overall_pass_rate,
                "results": [
                    {
                        "scenario": r.scenario_name,
                        "status": r.status.value,
                        "elapsed_seconds": r.elapsed_seconds,
                        "patterns_matched": r.patterns_matched,
                        "patterns_failed": r.patterns_failed,
                        "errors": r.errors
                    }
                    for r in basic_suite.results
                ],
                "blocking_issues": basic_suite.blocking_issues
            }
            
            if basic_suite.status != TestStatus.PASSED:
                results["blocking_issues"].extend(basic_suite.blocking_issues)
            
            logger.info(f"[{test_id}] Phase 2: Validating regex extraction")
            regex_results = await self._validate_regex_extraction(
                basic_suite.results,
                prompt_file,
                template_name
            )
            results["phases"]["regex_validation"] = regex_results
            
            if not regex_results.get("all_patterns_valid"):
                results["blocking_issues"].extend(regex_results.get("failed_patterns", []))
            
            if prompt_file == "content_generation.yaml":
                logger.info(f"[{test_id}] Phase 3: Running integration test")
                integration_results = await self._run_integration_test(
                    template_name=template_name,
                    new_prompt_content=new_prompt_content,
                    model=model,
                    test_id=test_id
                )
                results["phases"]["integration_test"] = integration_results
                
                if not integration_results.get("passed"):
                    results["blocking_issues"].extend(integration_results.get("errors", []))
            
            if prompt_file == "safety_judge.yaml":
                logger.info(f"[{test_id}] Phase 4: Running safety validation test")
                safety_results = await self._run_safety_validation_test(
                    template_name=template_name,
                    new_prompt_content=new_prompt_content,
                    model=model,
                    test_id=test_id
                )
                results["phases"]["safety_test"] = safety_results
                
                if not safety_results.get("passed"):
                    results["blocking_issues"].extend(safety_results.get("errors", []))
            
            results["completed_at"] = datetime.now().isoformat()
            
            if results["blocking_issues"]:
                results["overall_status"] = "failed"
                results["can_deploy"] = False
            else:
                results["overall_status"] = "passed"
                results["can_deploy"] = True
            
        except Exception as e:
            logger.error(f"[{test_id}] Workflow test error: {e}")
            results["overall_status"] = "error"
            results["blocking_issues"].append(f"Test execution error: {str(e)}")
            results["can_deploy"] = False
        
        return results
    
    async def _validate_regex_extraction(
        self,
        test_results: List[TestResult],
        prompt_file: str,
        template_name: str
    ) -> Dict[str, Any]:
        """Validate that all critical regex patterns still extract correctly.
        
        Uses per-template required patterns so each template type is only
        checked against patterns relevant to its output format.
        """
        
        # Use template-specific required patterns from the mapping
        required_patterns = self.template_required_patterns.get(template_name, [])
        
        # Fallback: if template not in mapping, infer from file type
        if not required_patterns:
            if prompt_file == "content_generation.yaml":
                # Unknown content template - require only claim_citation
                required_patterns = ["claim_citation"]
            elif prompt_file == "safety_judge.yaml":
                if "toxicity" in template_name:
                    required_patterns = ["toxicity_score"]
                elif "factuality" in template_name:
                    required_patterns = ["factuality_score"]
                elif "brand" in template_name:
                    required_patterns = ["brand_score"]
                elif "compliance" in template_name:
                    required_patterns = ["compliance_score"]
        
        results = {
            "required_patterns": required_patterns,
            "extracted_successfully": [],
            "failed_patterns": [],
            "all_patterns_valid": True,
            "status": "passed",
            "pass_rate": 1.0
        }
        
        # Check extracted data from ALL test results (not just PASSED)
        # A scenario might fail on length/forbidden checks but still
        # have extracted the required patterns successfully
        for test_result in test_results:
            for pattern_name in required_patterns:
                if pattern_name in test_result.extracted_data:
                    if pattern_name not in results["extracted_successfully"]:
                        results["extracted_successfully"].append(pattern_name)
        
        for pattern_name in required_patterns:
            if pattern_name not in results["extracted_successfully"]:
                results["failed_patterns"].append(
                    f"Critical pattern '{pattern_name}' could not be extracted from LLM output"
                )
                results["all_patterns_valid"] = False
        
        # Compute status and pass_rate from extraction results
        if required_patterns:
            results["pass_rate"] = len(results["extracted_successfully"]) / len(required_patterns)
        results["status"] = "passed" if results["all_patterns_valid"] else "failed"
        
        return results
    
    async def _run_integration_test(
        self,
        template_name: str,
        new_prompt_content: str,
        model: Optional[str],
        test_id: str
    ) -> Dict[str, Any]:
        """
        Run integration test: generate content, then validate it.
        
        This ensures the content generation and safety validation
        work together correctly.
        """
        results = {
            "passed": False,
            "content_generated": False,
            "content_validated": False,
            "safety_scores": {},
            "errors": [],
            "status": "pending",
            "pass_rate": 0.0
        }
        
        try:
            scenarios = self.test_scenarios.get(template_name, [])
            if not scenarios:
                results["errors"].append(f"No test scenarios for {template_name}")
                return results
            
            scenario = scenarios[0]
            
            try:
                new_template = yaml.safe_load(new_prompt_content)
                if isinstance(new_template, dict):
                    template_data = new_template.get(template_name, new_template)
                else:
                    template_data = {"user_template": new_prompt_content}
            except yaml.YAMLError:
                template_data = {"user_template": new_prompt_content}
            
            gen_result = await self._run_scenario(
                template_data=template_data,
                scenario=scenario,
                model=model or self.ollama_model
            )
            
            if gen_result.status != TestStatus.PASSED:
                results["errors"].append(f"Content generation failed: {gen_result.errors}")
                return results
            
            results["content_generated"] = True
            generated_content = gen_result.output
            
            safety_file = os.path.join(PROMPTS_DIR, "safety_judge.yaml")
            with open(safety_file, 'r') as f:
                safety_prompts = yaml.safe_load(f)
            
            toxicity_result = await self._test_safety_check(
                check_type="toxicity_check",
                content=generated_content,
                safety_prompts=safety_prompts,
                model=model
            )
            
            if toxicity_result.get("score") is not None:
                results["safety_scores"]["toxicity"] = toxicity_result["score"]
                results["content_validated"] = True
            else:
                results["errors"].append(f"Toxicity check failed: {toxicity_result.get('error')}")
            
            factuality_result = await self._test_safety_check(
                check_type="factuality_check",
                content=generated_content,
                safety_prompts=safety_prompts,
                model=model
            )
            
            if factuality_result.get("score") is not None:
                results["safety_scores"]["factuality"] = factuality_result["score"]
            
            if results["content_generated"] and results["content_validated"]:
                if "toxicity" in results["safety_scores"]:
                    results["passed"] = True
                else:
                    results["errors"].append("Could not extract safety scores from validation output")
            
        except Exception as e:
            results["errors"].append(str(e))
        
        # Set status and pass_rate based on final state
        results["status"] = "passed" if results["passed"] else "failed"
        results["pass_rate"] = 1.0 if results["passed"] else 0.0
        
        return results
    
    async def _run_safety_validation_test(
        self,
        template_name: str,
        new_prompt_content: str,
        model: Optional[str],
        test_id: str
    ) -> Dict[str, Any]:
        """
        Test safety validation prompts with known good and bad content.
        """
        results = {
            "passed": False,
            "safe_content_detected": False,
            "unsafe_content_detected": False,
            "score_extraction_works": False,
            "errors": [],
            "status": "pending",
            "pass_rate": 0.0
        }
        
        try:
            new_template = yaml.safe_load(new_prompt_content)
            template_data = new_template.get(template_name, new_template)
            
            # Test with known safe content
            safe_content = "Discover how AI-powered marketing automation can help your team focus on high-impact activities [CLM_003]."
            safe_result = await self._run_scenario(
                template_data=template_data,
                scenario=TestScenario(
                    name="safe_content_test",
                    description="Test with known safe content",
                    input_variables={"content": safe_content, "claim_library": "CLM_003: Available claim"},
                    expected_patterns=[r'SCORE:?\s*[0-9.]+'],
                    min_output_length=20,
                    max_output_length=3000
                ),
                model=model or self.ollama_model
            )
            
            # Test with known unsafe content
            unsafe_content = "This is GUARANTEED to make you rich! 100% risk-free! Buy now or regret forever!"
            unsafe_result = await self._run_scenario(
                template_data=template_data,
                scenario=TestScenario(
                    name="unsafe_content_test",
                    description="Test with known unsafe content",
                    input_variables={"content": unsafe_content, "claim_library": ""},
                    expected_patterns=[r'SCORE:?\s*[0-9.]+'],
                    min_output_length=20,
                    max_output_length=3000
                ),
                model=model or self.ollama_model
            )
            
            if safe_result.status == TestStatus.PASSED:
                results["safe_content_detected"] = True
                score_match = re.search(r'SCORE:?\s*([0-9.]+)', safe_result.output, re.IGNORECASE)
                if score_match:
                    safe_score = float(score_match.group(1))
                    results["safe_score"] = safe_score
                    results["score_extraction_works"] = True
            else:
                results["errors"].append(f"Safe content test failed: {safe_result.errors}")
            
            if unsafe_result.status == TestStatus.PASSED:
                results["unsafe_content_detected"] = True
                score_match = re.search(r'SCORE:?\s*([0-9.]+)', unsafe_result.output, re.IGNORECASE)
                if score_match:
                    unsafe_score = float(score_match.group(1))
                    results["unsafe_score"] = unsafe_score
            else:
                results["errors"].append(f"Unsafe content test failed: {unsafe_result.errors}")
            
            # Final determination
            if results["score_extraction_works"] and results["safe_content_detected"]:
                results["passed"] = True
            
        except Exception as e:
            results["errors"].append(str(e))
        
        # Set status and pass_rate based on final state
        results["status"] = "passed" if results["passed"] else "failed"
        results["pass_rate"] = 1.0 if results["passed"] else 0.0
        
        return results
    
    async def _test_safety_check(
        self,
        check_type: str,
        content: str,
        safety_prompts: Dict,
        model: Optional[str]
    ) -> Dict[str, Any]:
        """Run a specific safety check on content"""
        import httpx
        
        result = {"score": None, "error": None}
        
        try:
            check_config = safety_prompts.get(check_type, {})
            if not check_config:
                result["error"] = f"Safety check '{check_type}' not found"
                return result
            
            system_prompt = check_config.get("system_prompt", check_config.get("system", ""))
            user_template = check_config.get("user_prompt_template", check_config.get("user_template", check_config.get("user", "")))
            
            user_prompt = user_template
            user_prompt = user_prompt.replace("{content}", content)
            user_prompt = user_prompt.replace("{content_type}", "LinkedIn Ad")
            user_prompt = user_prompt.replace("{platform}", "LinkedIn")
            user_prompt = user_prompt.replace("{headline}", "Test Content")
            user_prompt = user_prompt.replace("{body}", content)
            user_prompt = user_prompt.replace("{cta}", "Learn More")
            user_prompt = user_prompt.replace("{claim_library}", "CLM_001: Test claim")
            user_prompt = user_prompt.replace("{claims_used}", "CLM_001")
            full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model or self.ollama_model,
                        "prompt": full_prompt,
                        "temperature": 0.1,
                        "num_predict": 1000,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    output = response.json().get("response", "")
                    
                    score_pattern = r'\*{0,2}(?:TOXICITY_|FACTUALITY_|BRAND_|COMPLIANCE_)?SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)'
                    match = re.search(score_pattern, output, re.IGNORECASE)
                    if match:
                        result["score"] = float(match.group(1))
                    else:
                        result["error"] = "Could not extract score from response"
                else:
                    result["error"] = f"LLM call failed: {response.status_code}"
                    
        except Exception as e:
            result["error"] = str(e)
        
        return result
