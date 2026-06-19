"""
Safety Validator Agent using LLM-as-a-Judge pattern
"""
import re
import yaml
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import hashlib
from pathlib import Path

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    from langchain.callbacks import get_openai_callback

from ...config.settings import settings
from ...governance.hitl_validator import HITLPreflightValidator
from ...governance.competitor_validator import CompetitorValidator
from ..memory.episodic_memory import EpisodicMemoryStore, AgentMemory, create_memory_from_task

logger = logging.getLogger(__name__)

class SafetyValidatorAgent:
    """
    Agent responsible for validating content safety and compliance
    """
    
    def __init__(self):
        safety_temp = getattr(settings, 'SAFETY_VALIDATOR_TEMPERATURE', 0.1)
        safety_max_tokens = getattr(settings, 'SAFETY_VALIDATOR_MAX_TOKENS', 1000)
        
        if settings.USE_LOCAL_LLM:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=settings.OLLAMA_MODEL,
                    temperature=safety_temp,
                    base_url=settings.OLLAMA_HOST
                )
                logger.info(f"SafetyValidator using local Ollama: {settings.OLLAMA_MODEL}")
                self.use_local_llm = True
            except ImportError:
                logger.warning("langchain_ollama not available, falling back to OpenAI")
                self.llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=safety_temp,
                    max_tokens=safety_max_tokens,
                    api_key=settings.OPENAI_API_KEY
                )
                self.use_local_llm = False
        else:
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=safety_temp,
                max_tokens=safety_max_tokens,
                api_key=settings.OPENAI_API_KEY
            )
            logger.info(f"SafetyValidator using OpenAI: {settings.OPENAI_MODEL}")
            self.use_local_llm = False

        self.rubrics = self._load_rubrics()

        self.claim_library = self._load_claim_library()

        self.hitl_validator = HITLPreflightValidator()

        self.competitor_validator = CompetitorValidator()

        self.memory = EpisodicMemoryStore(agent_name="safety_validator")

        self.total_cost = 0.0
        self.total_validations = 0
    
    def _load_rubrics(self) -> Dict[str, Any]:
        """Load safety validation rubrics"""
        rubrics_path = Path(settings.CONFIG_DIR) / "prompts" / "safety_judge.yaml"
        try:
            with open(rubrics_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('dimensions', self._get_default_rubrics())
        except Exception as e:
            logger.error(f"Failed to load rubrics: {e}")
            return self._get_default_rubrics()
    
    def _load_claim_library(self) -> Dict[str, Any]:
        """Load claim library for fact checking"""
        claim_path = settings.CLAIM_LIBRARY_PATH
        try:
            with open(claim_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load claim library: {e}")
            return {'claims': []}
    
    def _get_default_rubrics(self) -> Dict[str, Any]:
        """Default validation rubrics"""
        return {
            'toxicity': {
                'weight': 0.3,
                'criteria': [
                    'No hate speech or discriminatory language',
                    'No profanity or offensive content',
                    'No aggressive or threatening tone',
                    'Respectful of all groups and individuals'
                ]
            },
            'factuality': {
                'weight': 0.35,
                'criteria': [
                    'All claims must be from approved claim library',
                    'No unsupported statistics or data',
                    'No exaggerated or misleading statements',
                    'Accurate product/service descriptions'
                ]
            },
            'brand_alignment': {
                'weight': 0.25,
                'criteria': [
                    'Professional and appropriate tone',
                    'Consistent with Agentic brand voice',
                    'Focuses on value and benefits',
                    'Clear and compelling messaging'
                ]
            },
            'compliance': {
                'weight': 0.1,
                'criteria': [
                    'No medical or financial advice',
                    'Appropriate disclaimers included',
                    'Respects platform guidelines',
                    'GDPR/privacy compliant'
                ]
            }
        }
    
    async def validate_content(
        self,
        content_text: str,
        headline: Optional[str] = None,
        claims_used: Optional[List[str]] = None,
        platform: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate content for safety and compliance

        Args:
            content_text: Main content to validate
            headline: Optional headline
            claims_used: List of claim IDs supposedly used
            platform: Target platform
            context: Additional context

        Returns:
            Validation results with scores and issues
        """
        validation_id = hashlib.md5(content_text.encode()).hexdigest()[:8]
        start_time = datetime.now()
        task_id = f"{validation_id}_{platform}_{start_time.timestamp()}"
        actions_taken = []

        logger.info(
            "Safety validation started",
            extra={
                "event": "safety_validation_start",
                "validation_id": validation_id,
                "platform": platform,
                "content_length": len(content_text),
                "has_headline": headline is not None,
                "num_claims": len(claims_used) if claims_used else 0,
                "claims": claims_used
            }
        )

        memory_query = f"Validate {platform} content for safety and compliance"
        if claims_used:
            memory_query += f" with claims {claims_used}"
        relevant_memories = await self.memory.retrieve_relevant_memories(
            query=memory_query,
            k=3,
            outcome_filter=None
        )

        if relevant_memories:
            actions_taken.append(f"Retrieved {len(relevant_memories)} past validation experiences")
            logger.info(f"Using {len(relevant_memories)} past memories for safety validation")

        try:
            validation_results = {}
            issues = []

            hitl_metadata = {
                "persona": context.get("persona") if context else None,
                "platform": platform,
                "claims": claims_used
            }
            hitl_preflight = self.hitl_validator.validate_content(content_text, hitl_metadata)
            validation_results['hitl_preflight'] = hitl_preflight
            actions_taken.append("Ran HITL preflight checks")

            if not hitl_preflight['valid']:
                logger.warning(
                    f"HITL preflight checks failed: {len(hitl_preflight['checks_failed'])} checks failed",
                    extra={
                        "event": "hitl_preflight_failed",
                        "validation_id": validation_id,
                        "failed_checks": hitl_preflight['checks_failed'],
                        "warnings": hitl_preflight['warnings']
                    }
                )
                issues.extend(hitl_preflight['warnings'])
                actions_taken.append(f"HITL preflight failed: {len(hitl_preflight['checks_failed'])} checks")
            else:
                logger.info(f"HITL preflight checks passed: {hitl_preflight['score']:.0%}")
                actions_taken.append("HITL preflight passed")

            claim_validation = self._validate_claim_library_compliance(
                content_text,
                claims_used
            )
            validation_results['claim_validation'] = claim_validation
            actions_taken.append("Validated claim library compliance")

            # Blog/email content: claim failures are warnings, not critical blockers
            if not claim_validation['valid'] and platform not in ('blog', 'email'):
                logger.error(f"Claim validation failed: {claim_validation['reason']}")
                actions_taken.append(f"CRITICAL FAILURE: {claim_validation['reason']}")

                duration = (datetime.now() - start_time).total_seconds()
                failure_result = {
                    'success': False,
                    'cost': self.total_cost,
                    'duration': duration,
                    'quality_score': 0.0,
                    'error': claim_validation['reason']
                }

                failure_memory = create_memory_from_task(
                    agent_name="safety_validator",
                    task_id=task_id,
                    task_description=f"Validate {platform} content for safety and compliance",
                    actions=actions_taken,
                    result=failure_result
                )

                await self.memory.store_memory(failure_memory)

                result = {
                    'overall_score': 0.0,
                    'toxicity_score': 0.0,
                    'factuality_score': 0.0,
                    'brand_score': 0.0,
                    'compliance_score': 0.0,
                    'claim_validation': claim_validation,
                    'hitl_preflight': hitl_preflight,
                    'issues': [f"CRITICAL: {claim_validation['reason']}"] + claim_validation.get('issues', []),
                    'passed': False,
                    'requires_review': True
                }
                result['brand_alignment_score'] = result['brand_score']
                return result

            if not claim_validation['valid']:
                logger.warning(f"Claim validation warning for {platform}: {claim_validation['reason']}")
                actions_taken.append(f"Claim warning (non-blocking for {platform}): {claim_validation['reason']}")

            import asyncio
            
            toxicity_task = self._check_toxicity(content_text, headline)
            factuality_task = self._check_factuality(content_text, headline, claims_used)
            brand_task = self._check_brand_alignment(content_text, headline, platform)
            compliance_task = self._check_compliance(content_text, platform)
            
            toxicity_result, factuality_result, brand_result, compliance_result = await asyncio.gather(
                toxicity_task,
                factuality_task,
                brand_task,
                compliance_task
            )
            
            validation_results['toxicity_score'] = toxicity_result['score']
            if toxicity_result['issues']:
                issues.extend(toxicity_result['issues'])
            actions_taken.append(f"Toxicity check: score={toxicity_result['score']:.2f}")
            
            validation_results['factuality_score'] = factuality_result['score']
            if factuality_result['issues']:
                issues.extend(factuality_result['issues'])
            actions_taken.append(f"Factuality check: score={factuality_result['score']:.2f}")
            
            validation_results['brand_score'] = brand_result['score']
            if brand_result['issues']:
                issues.extend(brand_result['issues'])
            actions_taken.append(f"Brand alignment check: score={brand_result['score']:.2f}")
            
            validation_results['compliance_score'] = compliance_result['score']
            if compliance_result['issues']:
                issues.extend(compliance_result['issues'])
            actions_taken.append(f"Compliance check: score={compliance_result['score']:.2f}")

            competitor_result = self.competitor_validator.validate_content(
                content_text,
                headline
            )
            validation_results['competitor_validation'] = competitor_result
            if competitor_result['warnings']:
                issues.extend(competitor_result['warnings'])
            actions_taken.append("Competitor mention validation completed")

            if competitor_result['competitors_mentioned']:
                logger.info(
                    f"Competitors mentioned: {', '.join(competitor_result['competitors_mentioned'])}"
                )
            if competitor_result['risky_mentions']:
                logger.warning(
                    f"Risky competitor mentions found: {len(competitor_result['risky_mentions'])}"
                )

            # NOTE: toxicity_score is 0=safe, 1=toxic, so we invert it (1 - toxicity_score)
            # All other scores are 0=bad, 1=good
            rubrics = self.rubrics
            overall_score = (
                (1.0 - validation_results['toxicity_score']) * rubrics['toxicity']['weight'] +
                validation_results['factuality_score'] * rubrics['factuality']['weight'] +
                validation_results['brand_score'] * rubrics['brand_alignment']['weight'] +
                validation_results['compliance_score'] * rubrics.get('regulatory_compliance', rubrics.get('compliance', {})).get('weight', 0.1)
            )

            validation_results['overall_score'] = overall_score
            validation_results['issues'] = issues
            validation_results['passed'] = overall_score >= settings.SAFETY_SCORE_THRESHOLD
            validation_results['requires_review'] = overall_score < settings.AUTO_APPROVE_THRESHOLD

            self.total_validations += 1
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(
                "Safety validation completed",
                extra={
                    "event": "safety_validation_complete",
                    "validation_id": validation_id,
                    "platform": platform,
                    "overall_score": round(overall_score, 3),
                    "toxicity_score": round(validation_results['toxicity_score'], 3),
                    "factuality_score": round(validation_results['factuality_score'], 3),
                    "brand_score": round(validation_results['brand_score'], 3),
                    "compliance_score": round(validation_results['compliance_score'], 3),
                    "passed": validation_results['passed'],
                    "requires_review": validation_results['requires_review'],
                    "claim_validation_passed": claim_validation['valid'],
                    "num_issues": len(issues),
                    "issues": issues[:3] if issues else [],
                    "duration_seconds": round(duration, 2),
                    "total_validations": self.total_validations
                }
            )

            actions_taken.append(f"Validation completed: overall_score={overall_score:.2f}, passed={validation_results['passed']}")

            task_result = {
                'success': validation_results['passed'],
                'cost': self.total_cost,
                'duration': duration,
                'quality_score': overall_score,
                'passed': validation_results['passed'],
                'requires_review': validation_results['requires_review'],
                'num_issues': len(issues)
            }

            memory = create_memory_from_task(
                agent_name="safety_validator",
                task_id=task_id,
                task_description=f"Validate {platform} content for safety and compliance (score={overall_score:.2f})",
                actions=actions_taken,
                result=task_result,
                human_feedback=None
            )

            await self.memory.store_memory(memory)

            try:
                from ..utils.cost_tracker import track_llm_cost, estimate_tokens
                total_tokens = estimate_tokens(content_text) * 8  # Input repeated ~4x, outputs ~same
                model = settings.OLLAMA_MODEL if self.use_local_llm else "gpt-4"
                provider = "ollama" if self.use_local_llm else "openai"
                
                await track_llm_cost(
                    agent_type="safety_validator",
                    model=model,
                    tokens_prompt=total_tokens // 2,
                    tokens_completion=total_tokens // 2,
                    provider=provider,
                    campaign_id=context.get('campaign_id') if context else None,
                    action="safety_validation"
                )
            except Exception as cost_error:
                logger.warning(f"Failed to track validation costs: {cost_error}")

            validation_results['brand_alignment_score'] = validation_results['brand_score']

            return validation_results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            logger.error(
                "Safety validation failed with exception",
                extra={
                    "event": "safety_validation_error",
                    "validation_id": validation_id,
                    "platform": platform,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 2)
                },
                exc_info=True
            )

            actions_taken.append(f"Exception occurred: {str(e)}")

            exception_result = {
                'success': False,
                'cost': self.total_cost,
                'duration': duration,
                'quality_score': 0.0,
                'error': str(e)
            }

            exception_memory = create_memory_from_task(
                agent_name="safety_validator",
                task_id=task_id,
                task_description=f"Validate {platform} content for safety and compliance",
                actions=actions_taken,
                result=exception_result
            )

            await self.memory.store_memory(exception_memory)

            return {
                'overall_score': 0.0,
                'toxicity_score': 0.0,
                'factuality_score': 0.0,
                'brand_score': 0.0,
                'compliance_score': 0.0,
                'issues': [f"Validation error: {str(e)}"],
                'passed': False,
                'requires_review': True
            }
    
    async def _check_toxicity(
        self,
        content: str,
        headline: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check content for toxicity using Google Perspective API (Research Plan Section 7.2)

        Falls back to LLM-as-Judge if Perspective API is not configured.
        """
        try:
            from ..security.perspective_api import get_perspective_client

            perspective_client = get_perspective_client()
            if perspective_client.enabled:
                result = await perspective_client.check_marketing_content(content, headline)
                if not result.get("error"):
                    logger.info(f"Toxicity check via Perspective API: score={result['score']:.3f}")
                    return {
                        'score': result['score'],
                        'issues': result['issues'],
                        'source': 'perspective_api',
                        'details': result.get('details', {})
                    }
                else:
                    logger.warning(f"Perspective API error, falling back to LLM: {result['error']}")
        except Exception as e:
            logger.warning(f"Perspective API unavailable, falling back to LLM: {e}")

        return await self._check_toxicity_llm(content, headline)

    async def _check_toxicity_llm(
        self,
        content: str,
        headline: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check content for toxicity using LLM-as-Judge (fallback method)"""
        prompt = f"""You are a content moderation ANALYST reviewing marketing text for a compliance audit. Your task is to ANALYZE and SCORE the text below - you are NOT generating or endorsing this content, only evaluating it.

TEXT FOR TOXICITY ANALYSIS:
---
{content}
{f"HEADLINE: {headline}" if headline else ""}
---

SCORING GUIDE (0.0 = no issues, 1.0 = severe issues):
- 0.0-0.2: Professional, respectful language
- 0.3-0.5: Minor tone issues but acceptable
- 0.6-0.8: Contains problematic language
- 0.9-1.0: Contains hate speech, slurs, or threats

RESPOND ONLY WITH:
SCORE: [decimal]
ISSUES: [brief note or "None"]
SAFE: [YES/NO]

SCORE: """

        try:
            if self.use_local_llm:
                response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            else:
                with get_openai_callback() as cb:
                    response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
                    self.total_cost += cb.total_cost

            response_text = response.generations[0][0].text

            score_match = re.search(r'\*{0,2}SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)', response_text, re.IGNORECASE)

            if not score_match:
                refusal_indicators = [
                    "cannot", "can't", "won't", "will not", "refuse",
                    "inappropriate", "harmful", "offensive", "hate speech",
                    "discrimination", "not able to"
                ]
                response_lower = response_text.lower()

                if any(indicator in response_lower for indicator in refusal_indicators):
                    logger.warning(f"Toxicity check: LLM refused to score content (likely highly toxic): {response_text[:100]}")
                    return {
                        'score': 0.95,
                        'issues': ["Content flagged as highly toxic - LLM safety refusal triggered",
                                   f"LLM response: {response_text[:150]}..."],
                        'source': 'llm_as_judge'
                    }

                logger.error(f"Toxicity check: Failed to parse SCORE from LLM response: {response_text[:200]}")
                raise ValueError(f"LLM response missing required SCORE field. Response: {response_text[:100]}")

            score = float(score_match.group(1))

            if score < 0.0 or score > 1.0:
                logger.error(f"Toxicity check: Invalid score {score}, must be 0.0-1.0")
                raise ValueError(f"Invalid toxicity score {score}, must be between 0.0 and 1.0")

            issues_match = re.search(r'ISSUES:\s*(.+?)(?:SAFE:|$)', response_text, re.DOTALL)
            issues_text = issues_match.group(1).strip() if issues_match else ""

            issues = []
            if issues_text and issues_text.lower() != "none":
                issues = [i.strip() for i in issues_text.split('\n') if i.strip()]

            return {
                'score': score,
                'issues': issues,
                'source': 'llm_as_judge'
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Toxicity check failed with exception: {e}", exc_info=True)
            raise RuntimeError(f"Toxicity validation failed: {str(e)}")
    
    async def _check_factuality(
        self,
        content: str,
        headline: Optional[str],
        claims_used: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Check factual accuracy and claim citation"""
        approved_claims = {c['id']: c['text'] for c in self.claim_library.get('claims', [])}
        
        claim_pattern = r'\[([A-Z0-9_]+)\]'
        found_citations = re.findall(claim_pattern, content)
        valid_citations = [c for c in found_citations if c in approved_claims]
        invalid_citations = [c for c in found_citations if c not in approved_claims]
        
        cited_claims_info = ""
        if valid_citations:
            cited_claims_info = "PROPERLY CITED CLAIMS IN CONTENT:\n"
            for cid in set(valid_citations):
                cited_claims_info += f"  - [{cid}]: \"{approved_claims[cid]}\"\n"
        
        relevant_claims = {k: v for k, v in approved_claims.items() if k in valid_citations}

        prompt = f"""You are a fact-checking auditor. Your job is to EVALUATE and SCORE marketing content for factual accuracy - NOT to refuse or censor it.

IMPORTANT: You MUST provide a numerical score even for problematic content. This is an audit, not content generation.

CONTENT TO EVALUATE:
{content}

{f"HEADLINE: {headline}" if headline else ""}

{cited_claims_info}

EVALUATION CRITERIA (rate factuality 0.0 = all false/unverified, 1.0 = fully accurate):
1. Check if statements in content match the cited claims
2. Check for unsupported statistics or exaggerations NOT backed by citations
3. Check for misleading statements

SCORING GUIDE:
- 1.0: All factual statements are properly cited with valid claim IDs
- 0.8: Minor uncited statements but main claims are valid
- 0.5: Some unsupported claims or misleading statements
- 0.0: Major false or unverified claims

YOU MUST RESPOND IN THIS EXACT FORMAT:
SCORE: [number between 0.0 and 1.0]
UNCITED_CLAIMS: [list any statements that make claims without citation, or "None"]
ISSUES: [list or "None"]

Now evaluate the content and provide your numerical assessment:
SCORE: """

        try:
            if self.use_local_llm:
                response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            else:
                with get_openai_callback() as cb:
                    response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
                    self.total_cost += cb.total_cost

            response_text = response.generations[0][0].text

            score_match = re.search(r'\*{0,2}SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)', response_text, re.IGNORECASE)

            if not score_match:
                refusal_indicators = ["cannot", "can't", "won't", "refuse", "inappropriate", "harmful"]
                if any(ind in response_text.lower() for ind in refusal_indicators):
                    logger.warning(f"Factuality check: LLM refused to score content: {response_text[:100]}")
                    return {'score': 0.0, 'issues': ["Content flagged - LLM refused to evaluate"]}

                logger.error(f"Factuality check: Failed to parse SCORE from LLM response: {response_text[:200]}")
                raise ValueError(f"LLM response missing required SCORE field. Response: {response_text[:100]}")

            score = float(score_match.group(1))

            if score < 0.0 or score > 1.0:
                logger.error(f"Factuality check: Invalid score {score}, must be 0.0-1.0")
                raise ValueError(f"Invalid factuality score {score}, must be between 0.0 and 1.0")

            issues = []
            
            if invalid_citations:
                issues.append(f"Invalid claim IDs not in library: {invalid_citations}")

            uncited_match = re.search(r'UNCITED_CLAIMS:\s*(.+?)(?:ISSUES:|$)', response_text, re.DOTALL)
            if uncited_match and uncited_match.group(1).strip().lower() not in ["none", "n/a", "[]", "null"]:
                uncited = uncited_match.group(1).strip()
                # Only add if it's not just citing the valid claim IDs
                if not all(cid in uncited for cid in valid_citations):
                    issues.append(f"Uncited statements: {uncited}")

            return {
                'score': score,
                'issues': issues,
                'valid_citations': valid_citations,
                'invalid_citations': invalid_citations
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Factuality check failed with exception: {e}", exc_info=True)
            raise RuntimeError(f"Factuality validation failed: {str(e)}")
    
    async def _check_brand_alignment(
        self,
        content: str,
        headline: Optional[str],
        platform: str
    ) -> Dict[str, Any]:
        """Check brand voice and messaging alignment"""
        prompt = f"""You are a brand auditor for Agentic AI. Your job is to EVALUATE and SCORE marketing content for brand alignment - NOT to refuse or censor it.

IMPORTANT: You MUST provide a numerical score even for off-brand content. This is an audit, not content generation.

PLATFORM: {platform}

CONTENT TO EVALUATE:
{content}

{f"HEADLINE: {headline}" if headline else ""}

AGENTIC BRAND GUIDELINES:
- Professional and innovative tone
- Focus on AI-driven value and ROI
- Data-driven but accessible language
- Confident without being arrogant
- Customer-centric messaging

EVALUATION: Rate brand alignment 0.0 = completely off-brand, 1.0 = perfect alignment

YOU MUST RESPOND IN THIS EXACT FORMAT:
SCORE: [number between 0.0 and 1.0]
TONE_ISSUES: [list or "None"]
MESSAGING_ISSUES: [list or "None"]
IMPROVEMENTS: [list or "None"]

Now evaluate the content and provide your numerical assessment:
SCORE: """

        try:
            if self.use_local_llm:
                response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            else:
                with get_openai_callback() as cb:
                    response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
                    self.total_cost += cb.total_cost

            response_text = response.generations[0][0].text

            score_match = re.search(r'\*{0,2}SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)', response_text, re.IGNORECASE)

            if not score_match:
                refusal_indicators = ["cannot", "can't", "won't", "refuse", "inappropriate", "harmful"]
                if any(ind in response_text.lower() for ind in refusal_indicators):
                    logger.warning(f"Brand alignment check: LLM refused to score content: {response_text[:100]}")
                    return {'score': 0.0, 'issues': ["Content flagged - completely off-brand"]}

                logger.error(f"Brand alignment check: Failed to parse SCORE from LLM response: {response_text[:200]}")
                raise ValueError(f"LLM response missing required SCORE field. Response: {response_text[:100]}")

            score = float(score_match.group(1))

            if score < 0.0 or score > 1.0:
                logger.error(f"Brand alignment check: Invalid score {score}, must be 0.0-1.0")
                raise ValueError(f"Invalid brand alignment score {score}, must be between 0.0 and 1.0")

            issues = []

            tone_match = re.search(r'TONE_ISSUES:\s*(.+?)(?:MESSAGING_ISSUES:|$)', response_text, re.DOTALL)
            if tone_match and tone_match.group(1).strip().lower() != "none":
                issues.append(f"Tone: {tone_match.group(1).strip()}")

            msg_match = re.search(r'MESSAGING_ISSUES:\s*(.+?)(?:IMPROVEMENTS:|$)', response_text, re.DOTALL)
            if msg_match and msg_match.group(1).strip().lower() != "none":
                issues.append(f"Messaging: {msg_match.group(1).strip()}")

            return {
                'score': score,
                'issues': issues
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Brand alignment check failed with exception: {e}", exc_info=True)
            raise RuntimeError(f"Brand alignment validation failed: {str(e)}")
    
    async def _check_compliance(
        self,
        content: str,
        platform: str
    ) -> Dict[str, Any]:
        """Check legal and platform compliance"""
        prompt = f"""You are a marketing compliance auditor. EVALUATE and SCORE this B2B marketing content.

PLATFORM: {platform}

CONTENT:
{content}

SCORING GUIDE:
- 1.0: Compliant B2B marketing (standard product claims, no regulated advice)
- 0.7-0.9: Minor issues (could add optional disclaimers)
- 0.5-0.7: Needs attention (missing important disclosures)
- Below 0.5: Serious violations (medical/financial advice without disclaimers, illegal claims)

IMPORTANT CONTEXT:
- This is B2B SaaS/AI marketing content, NOT medical or financial advice
- General product benefit claims (e.g., "40% better engagement") are NORMAL marketing
- Claims with [CLM_XXX] citations are pre-approved and verified
- GDPR only applies if collecting personal data (this is marketing copy, not a data form)
- Standard B2B marketing does NOT require special disclaimers

RESPOND EXACTLY:
SCORE: [0.0-1.0]
VIOLATIONS: [specific issues or "None"]

SCORE: """

        try:
            if self.use_local_llm:
                response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
            else:
                with get_openai_callback() as cb:
                    response = await self.llm.agenerate([[HumanMessage(content=prompt)]])
                    self.total_cost += cb.total_cost

            response_text = response.generations[0][0].text

            score_match = re.search(r'\*{0,2}SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)', response_text, re.IGNORECASE)

            if not score_match:
                refusal_indicators = ["cannot", "can't", "won't", "refuse", "inappropriate", "harmful"]
                if any(ind in response_text.lower() for ind in refusal_indicators):
                    logger.warning(f"Compliance check: LLM refused to score content: {response_text[:100]}")
                    return {'score': 0.0, 'issues': ["Content flagged - compliance violations detected"]}

                logger.error(f"Compliance check: Failed to parse SCORE from LLM response: {response_text[:200]}")
                raise ValueError(f"LLM response missing required SCORE field. Response: {response_text[:100]}")

            score = float(score_match.group(1))

            if score < 0.0 or score > 1.0:
                logger.error(f"Compliance check: Invalid score {score}, must be 0.0-1.0")
                raise ValueError(f"Invalid compliance score {score}, must be between 0.0 and 1.0")

            issues = []

            violations_match = re.search(r'VIOLATIONS:\s*(.+?)(?:\n|$)', response_text, re.DOTALL)
            if violations_match:
                violations_text = violations_match.group(1).strip()
                if violations_text.lower() not in ["none", "n/a", "no violations", "none found", ""]:
                    # Don't flag GDPR for standard marketing copy without data collection
                    if "gdpr" in violations_text.lower() and "data" not in content.lower():
                        pass
                    else:
                        issues.append(f"Violations: {violations_text}")

            return {
                'score': score,
                'issues': issues
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Compliance check failed with exception: {e}", exc_info=True)
            raise RuntimeError(f"Compliance validation failed: {str(e)}")

    def _validate_claim_library_compliance(
        self,
        content_text: str,
        claims_used: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Validate content against claim library requirements (MANDATORY)

        Args:
            content_text: Content to validate
            claims_used: List of claim IDs that should be in content

        Returns:
            Dict with 'valid' bool, 'reason' string, and 'issues' list
        """
        validation_rules = self.claim_library.get('validation_rules', {})
        min_claims = validation_rules.get('min_claims_per_content', 1)
        max_claims = validation_rules.get('max_claims_per_content', 3)
        require_citation = validation_rules.get('require_citation', True)
        citation_format = validation_rules.get('citation_format', '[CLAIM_ID]')

        issues = []

        claim_pattern = r'\[([A-Z0-9_]+)\]'
        found_citations = re.findall(claim_pattern, content_text)
        
        unique_citations = list(set(found_citations))

        all_claims = self.claim_library.get('claims', [])
        valid_claim_ids = [c['id'] for c in all_claims]

        valid_found_claims = [c for c in unique_citations if c in valid_claim_ids]
        invalid_citations = [c for c in unique_citations if c not in valid_claim_ids]

        if invalid_citations:
            issues.append(f"Invalid claim citations found: {invalid_citations}")

        if len(valid_found_claims) < min_claims:
            return {
                'valid': False,
                'reason': f"Insufficient claims: {len(valid_found_claims)}/{min_claims} required",
                'issues': issues + [f"Content must include at least {min_claims} claims from the approved library"]
            }

        if len(valid_found_claims) > max_claims:
            return {
                'valid': False,
                'reason': f"Too many claims: {len(valid_found_claims)}/{max_claims} maximum",
                'issues': issues + [f"Content exceeds maximum of {max_claims} claims"]
            }

        if claims_used is not None:
            missing_claims = [c for c in claims_used if c not in valid_found_claims]
            extra_claims = [c for c in valid_found_claims if c not in claims_used]

            if missing_claims:
                issues.append(f"Declared claims not found in content: {missing_claims}")
            if extra_claims:
                issues.append(f"Content has claims not declared: {extra_claims}")

        if require_citation:
            for claim_id in valid_found_claims:
                expected_citation = citation_format.replace('CLAIM_ID', claim_id)
                if expected_citation not in content_text:
                    issues.append(f"Claim {claim_id} improperly cited (expected: {expected_citation})")

        current_date = datetime.now()
        for claim_id in valid_found_claims:
            claim = next((c for c in all_claims if c['id'] == claim_id), None)
            if claim and claim.get('expiry_date'):
                try:
                    from dateutil import parser as date_parser
                    expiry_date = date_parser.parse(claim['expiry_date'])
                    if current_date > expiry_date:
                        issues.append(f"Claim {claim_id} has expired (expiry: {claim['expiry_date']})")
                        return {
                            'valid': False,
                            'reason': f"Expired claim used: {claim_id}",
                            'issues': issues
                        }
                except Exception as e:
                    logger.warning(f"Failed to parse expiry date for {claim_id}: {e}")

        if issues:
            return {
                'valid': False,
                'reason': f"Claim validation issues: {'; '.join(issues[:2])}",
                'issues': issues
            }

        return {
            'valid': True,
            'reason': f"All {len(valid_found_claims)} claims properly validated",
            'issues': [],
            'claims_found': valid_found_claims
        }

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'total_validations': self.total_validations,
            'total_cost': self.total_cost,
            'average_cost_per_validation': self.total_cost / max(1, self.total_validations)
        }