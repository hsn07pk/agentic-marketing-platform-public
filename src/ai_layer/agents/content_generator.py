"""
Content Generation Agent with RAG, Claim Library, and Semantic Cache support
"""
import json
import yaml
import csv
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import hashlib
from pathlib import Path

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
import redis

from ...config.settings import settings
from ...config.brand_voice import get_brand_voice_config
from ...data_layer.database.models import Content, ContentStatus, VectorStore
from ...data_layer.vector_store.pgvector_store import PgVectorStore
from ...data_layer.vector_store.semantic_cache import SemanticCache
from ..memory.episodic_memory import EpisodicMemoryStore, AgentMemory, create_memory_from_task
from ..security.prompt_shield import get_prompt_shield, ShieldLevel
from .market_scraper import MarketScraperAgent

logger = logging.getLogger(__name__)

class ContentGeneratorAgent:
    """
    Agent responsible for generating marketing content with RAG and episodic memory
    """

    def __init__(self):
        self.use_local_llm = settings.get('USE_LOCAL_LLM', False)

        if self.use_local_llm:
            from .ollama_integration import OllamaClient
            self.ollama_client = OllamaClient(host=settings.get('OLLAMA_HOST', 'http://localhost:11434'))
            self.ollama_client.set_model(settings.get('OLLAMA_MODEL', 'mixtral:8x7b'))
            self.llm = None
            logger.info(f"Using Ollama local LLM: {settings.get('OLLAMA_MODEL', 'mixtral:8x7b')}")
        else:
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                api_key=settings.OPENAI_API_KEY
            )
            self.ollama_client = None
            logger.info(f"Using OpenAI: {settings.OPENAI_MODEL}")

        self.semantic_cache = SemanticCache()
        self._cache_initialized = False
        
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
        except Exception as e:
            logger.warning(f"Redis connection failed for cache tracking: {e}")
            self.redis_client = None

        self.embeddings = None
        self.embeddings = None

        self.vector_store = PgVectorStore(collection_name="documents")

        self.memory = EpisodicMemoryStore(agent_name="content_generator")

        try:
            self.market_scraper = MarketScraperAgent()
            logger.info("MarketScraperAgent initialized successfully")
        except Exception as e:
            logger.warning(f"MarketScraperAgent initialization failed: {e}. Competitive insights will be limited.")
            self.market_scraper = None

        self.brand_voice_config = get_brand_voice_config()
        self.company = self.brand_voice_config.get_company()
        self.brand_voice = self.brand_voice_config.get_brand_voice()

        self.claim_library = self._load_claim_library()

        self.prompts = self._load_prompts()

        self.total_cost = 0.0
        self.total_tokens = 0
        self.task_start_time = None

        self.prompt_shield = get_prompt_shield(ShieldLevel.ENHANCED)
        logger.info("Prompt shielding initialized with ENHANCED security level")
    
    def _load_claim_library(self) -> Dict[str, Any]:
        """
        Load the versioned claim library from CSV (preferred) or YAML (fallback)

        Priority:
        1. data/claim_library/claims.csv
        2. config/prompts/claim_library.yaml (from settings.CLAIM_LIBRARY_PATH)
        """
        csv_file = Path("data/claim_library/claims.csv")
        if csv_file.exists():
            try:
                claims_list = []
                with open(csv_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row.get('id'):
                            continue

                        personas_str = row.get('personas', '[]')
                        tags_str = row.get('tags', '[]')

                        if not personas_str:
                            personas_str = '[]'
                        if not tags_str:
                            tags_str = '[]'

                        if personas_str.startswith('[') and not personas_str.startswith('["'):
                            personas_str = personas_str.replace('[', '["').replace(']', '"]').replace(', ', '", "')
                        if tags_str.startswith('[') and not tags_str.startswith('["'):
                            tags_str = tags_str.replace('[', '["').replace(']', '"]').replace(', ', '", "')

                        try:
                            personas = json.loads(personas_str)
                            tags = json.loads(tags_str)
                        except json.JSONDecodeError:
                            personas = []
                            tags = []

                        claims_list.append({
                            'id': row['id'],
                            'text': row.get('claim_text', ''),
                            'type': row.get('claim_type', ''),
                            'personas': personas,
                            'tags': tags,
                            'source': row.get('source_title', ''),
                            'source_url': row.get('source_url', ''),
                            'evidence_url': row.get('source_url', ''),
                            'evidence_excerpt': row.get('evidence_excerpt', ''),
                            'confidence': int(row.get('confidence', 3)),
                            'priority': int(row.get('confidence', 3)) * 2
                        })

                logger.info(f"✅ Loaded {len(claims_list)} claims from CSV: {csv_file}")
                return {
                    'claims': claims_list,
                    'version': '1.0.0',
                    'source': 'csv'
                }

            except Exception as e:
                logger.error(f"Failed to load claims from CSV, falling back to YAML: {e}")

        claim_path = settings.CLAIM_LIBRARY_PATH
        try:
            with open(claim_path, 'r') as f:
                claims = yaml.safe_load(f)
                logger.info(f"✅ Loaded {len(claims.get('claims', []))} claims from YAML: {claim_path}")
                claims['source'] = 'yaml'
                return claims
        except Exception as e:
            logger.error(f"❌ Failed to load claim library: {e}")
            return {'claims': [], 'version': '1.0.0', 'source': 'none'}
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates"""
        prompts_path = Path(settings.CONFIG_DIR) / "prompts" / "content_generation.yaml"
        try:
            with open(prompts_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """Default prompt templates with brand voice"""
        company_name = self.company.name if hasattr(self, 'company') else 'Agentic AI'

        return {
            'linkedin_ad': f"""
You are an expert B2B marketing copywriter for {company_name}.

{{brand_voice_guidelines}}

Target Persona: {{persona}}
Campaign Goal: {{goal}}
Key Message: {{message}}

Context from our knowledge base:
{{context}}

Available claims (you MUST use 1-3 of these):
{{claims}}

Generate engaging LinkedIn ad copy that:
1. Hooks the reader immediately
2. Speaks directly to the persona's pain points
3. Includes 1-3 claims from the provided list (cite with [CLAIM_ID])
4. Has a clear, compelling CTA
5. Follows LinkedIn best practices

Format:
Headline: [compelling headline]
Body: [main copy with claims]
CTA: [call to action]
""",
            'twitter_ad': """
You are a social media marketing expert for Agentic AI.

Target: {persona}
Goal: {goal}
Keywords: {keywords}

Context:
{context}

Available claims:
{claims}

Create a Twitter ad that:
1. Grabs attention in <280 characters
2. Uses 1-2 relevant claims [CLAIM_ID]
3. Includes trending elements if relevant
4. Has a strong CTA

Output:
Tweet: [your tweet with claims]
""",
            'email': """
You are an email marketing specialist for Agentic AI.

Recipient Persona: {persona}
Campaign Stage: {stage}
Previous Interactions: {history}

Context:
{context}

Claims to use:
{claims}

Write a personalized email that:
1. Has a subject line with >30% open rate potential
2. Personalizes based on persona and stage
3. Incorporates 2-3 claims naturally [CLAIM_ID]
4. Guides toward booking a demo

Format:
Subject: [subject line]
Preview: [preview text]
Body: [full email with claims]
"""
        }
    
    async def generate_content(
        self,
        platform: str,
        persona: str,
        campaign_config: Dict[str, Any],
        context_query: Optional[str] = None,
        previous_feedback: Optional[str] = None
    ) -> Tuple[Content, Dict[str, Any]]:
        """
        Generate marketing content with RAG and claim validation

        Args:
            platform: Target platform (linkedin, twitter, email)
            persona: Target persona identifier
            campaign_config: Campaign configuration
            context_query: Optional query for retrieving context
            previous_feedback: Optional feedback from previous rejection/regeneration

        Returns:
            Generated content object and metadata
        """
        max_retries = 3
        retry_count = 0
        self.task_start_time = datetime.now()
        task_id = f"{platform}_{persona}_{self.task_start_time.timestamp()}"
        actions_taken = []

        logger.info(
            "Content generation started",
            extra={
                "event": "content_generation_start",
                "task_id": task_id,
                "platform": platform,
                "persona": persona,
                "goal": campaign_config.get('goal', 'unknown'),
                "campaign_id": campaign_config.get('campaign_id'),
                "max_retries": max_retries
            }
        )

        memory_query = f"Generate {platform} content for {persona} with goal {campaign_config.get('goal', '')}"
        relevant_memories = await self.memory.retrieve_relevant_memories(
            query=memory_query,
            k=3,
            outcome_filter=None
        )
        memory_context = await self.memory.format_memories_for_prompt(relevant_memories)

        while retry_count < max_retries:
            try:
                context = await self._retrieve_context(
                    query=context_query or f"{platform} {persona} {campaign_config.get('goal', '')}",
                    k=5
                )
                actions_taken.append(f"Retrieved {len(context)} context documents")

                logger.info(
                    "Context retrieved from RAG",
                    extra={
                        "event": "rag_context_retrieved",
                        "task_id": task_id,
                        "num_documents": len(context),
                        "sources": [c.get('source', 'unknown') for c in context[:3]],
                        "query": context_query or f"{platform} {persona}"
                    }
                )

                selected_claims = self._select_claims(persona, campaign_config)
                actions_taken.append(f"Selected {len(selected_claims)} relevant claims")

                logger.info(
                    "Claims selected from library",
                    extra={
                        "event": "claims_selected",
                        "task_id": task_id,
                        "num_claims": len(selected_claims),
                        "claim_ids": [c['id'] for c in selected_claims],
                        "persona": persona
                    }
                )

                validation_rules = self.claim_library.get('validation_rules', {})
                min_claims = validation_rules.get('min_claims_per_content', 1)

                if len(selected_claims) < min_claims:
                    raise ValueError(f"Insufficient relevant claims: found {len(selected_claims)}, need {min_claims}")

                # Blog uses 'blog_post' template key, others use '{platform}_ad'
                if platform == "blog":
                    prompt_template = self.prompts.get('blog_post', self.prompts.get('linkedin_ad'))
                else:
                    prompt_template = self.prompts.get(f"{platform}_ad", self.prompts.get('linkedin_ad'))

                if isinstance(prompt_template, dict):
                    template_str = prompt_template.get('user_template', '')
                elif isinstance(prompt_template, str):
                    template_str = prompt_template
                else:
                    raise ValueError(f"Invalid prompt template format: {type(prompt_template)}")

                brand_voice_guidelines = self.brand_voice_config.format_for_prompt(persona=persona)

                trending_topics = self._get_trending_topics(campaign_config)

                competitor_insights = campaign_config.get('competitor_insights', 'N/A')
                if self.market_scraper is not None and competitor_insights == 'N/A':
                    try:
                        content_patterns = campaign_config.get('content_patterns', None)
                        competitor_insights = self.market_scraper.format_competitive_insights_for_content(
                            persona=persona,
                            content_patterns=content_patterns
                        )
                        logger.info("Competitive insights + market patterns added from MarketScraper")
                    except Exception as e:
                        logger.warning(f"Failed to get competitive insights from MarketScraper: {e}")
                        competitor_insights = 'N/A'

                strategy = campaign_config.get('strategy', {})

                prompt_params = {
                    'persona_name': persona,
                    'persona_description': campaign_config.get('persona_description', f'{persona} persona'),
                    'pain_points': campaign_config.get('pain_points', 'Common industry challenges'),
                    'campaign_goal': campaign_config.get('goal', 'generate leads'),
                    'campaign_type': campaign_config.get('type', 'lead generation'),
                    'retrieved_context': self._format_context(context),
                    'available_claims': self._format_claims(selected_claims),
                    'competitor_insights': competitor_insights,
                    'brand_voice_guidelines': brand_voice_guidelines,
                    'trending_topics': trending_topics,
                    'strategy_hook': strategy.get('hook', 'Transform your business'),
                    'strategy_cta': strategy.get('cta', 'Learn More'),
                    'strategy_tone': strategy.get('tone', 'professional'),
                    'strategy_angle': strategy.get('angle', 'innovation'),
                    'strategy_name': strategy.get('strategy_name', 'Default'),
                    'campaign_stage': campaign_config.get('stage', 'awareness'),
                    'interaction_history': campaign_config.get('interaction_history', 'First contact'),
                    'email_type': campaign_config.get('email_type', 'outreach'),
                    'company_name': campaign_config.get('company_name', 'Valued Company'),
                    'industry': campaign_config.get('industry', 'Technology'),
                    'blog_topic': campaign_config.get('blog_topic', campaign_config.get('goal', 'Industry insights and best practices')),
                    'seo_keywords': campaign_config.get('seo_keywords', campaign_config.get('target_keywords', 'B2B, marketing, AI')),
                    'post_objective': campaign_config.get('post_objective', 'Share insights'),
                    'target_personas': campaign_config.get('target_personas', persona),
                    'content_theme': campaign_config.get('content_theme', 'Product innovation')
                }

                if strategy:
                    logger.info(
                        "Using optimized strategy for content generation",
                        extra={
                            "event": "strategy_applied",
                            "task_id": task_id,
                            "strategy_name": strategy.get('strategy_name'),
                            "strategy_action": strategy.get('action'),
                            "confidence": strategy.get('confidence'),
                            "hook": strategy.get('hook'),
                            "cta": strategy.get('cta'),
                            "tone": strategy.get('tone')
                        }
                    )
                    actions_taken.append(f"Applied strategy: {strategy.get('strategy_name')} (confidence: {strategy.get('confidence', 0):.2f})")

                try:
                    prompt = template_str.format(**prompt_params)
                except KeyError as e:
                    logger.warning(f"Missing template parameter {e}, using partial template")
                    prompt = template_str
                    for key, value in prompt_params.items():
                        placeholder = '{' + key + '}'
                        if placeholder in prompt:
                            prompt = prompt.replace(placeholder, str(value))

                enforcement_note = f"\n\nIMPORTANT: You MUST include between {min_claims} and {validation_rules.get('max_claims_per_content', 3)} claims from the provided list. Each claim MUST be cited using the format [CLAIM_ID]. Content without proper claim citations will be rejected."
                prompt = prompt + enforcement_note

                shielded = self.prompt_shield.shield_prompt(
                    system_prompt=prompt,
                    user_input=f"Generate {platform} content for persona: {persona}",
                    output_format="Headline: [headline]\nBody: [body with claims]\nCTA: [call to action]",
                    context={
                        "rag_context": self._format_context(context),
                        "available_claims": self._format_claims(selected_claims),
                        "campaign_goal": campaign_config.get('goal', '')
                    }
                )
                prompt = shielded.full_prompt

                if shielded.detected_risks:
                    logger.warning(
                        "Prompt shielding detected potential security risks",
                        extra={
                            "event": "prompt_shield_risks",
                            "task_id": task_id,
                            "risks": shielded.detected_risks,
                            "input_sanitized": shielded.input_sanitized
                        }
                    )
                    actions_taken.append(f"Prompt shielding: {len(shielded.detected_risks)} risks detected and mitigated")

                if memory_context and "No relevant past experiences" not in memory_context:
                    prompt = prompt + f"\n\n{memory_context}\n\nUse these past experiences to improve your content generation."
                    actions_taken.append("Incorporated past experience memories into context")

                if previous_feedback:
                    feedback_note = f"\n\n**PREVIOUS HUMAN FEEDBACK (MUST ADDRESS):**\n{previous_feedback}\n\nThis content was previously rejected or required regeneration. You MUST carefully address all points in the feedback above. Make substantial improvements based on the specific issues raised."
                    prompt = prompt + feedback_note
                    actions_taken.append(f"Incorporating human feedback: {previous_feedback[:100]}...")
                    logger.info(
                        "Human feedback incorporated into generation",
                        extra={
                            "event": "feedback_incorporated",
                            "task_id": task_id,
                            "feedback_length": len(previous_feedback),
                            "feedback_preview": previous_feedback[:200]
                        }
                    )

                logger.info(
                    "Calling LLM for content generation",
                    extra={
                        "event": "llm_call_start",
                        "task_id": task_id,
                        "model": settings.OPENAI_MODEL,
                        "prompt_length": len(prompt),
                        "retry_attempt": retry_count + 1
                    }
                )

                # Skip cache on retries to avoid returning the same non-compliant content
                cached_response = None
                cache_hit = False
                start_time = time.time()
                
                if settings.ENABLE_SEMANTIC_CACHE and retry_count == 0:
                    try:
                        if not self._cache_initialized:
                            await self.semantic_cache.initialize()
                            self._cache_initialized = True
                        
                        model_name = settings.get('OLLAMA_MODEL', 'mixtral:8x7b') if self.use_local_llm else settings.OPENAI_MODEL
                        campaign_id_for_cache = campaign_config.get('campaign_id')
                        cached_response = await self.semantic_cache.get(
                            prompt, model=model_name, campaign_id=campaign_id_for_cache
                        )
                        
                        if cached_response:
                            cache_hit = True
                            content_text = cached_response.get('response', '')
                            cost = 0.0
                            
                            cached_claims = self._extract_claim_citations(content_text)
                            if len(cached_claims) < 1:
                                logger.info(
                                    "Semantic cache HIT but content has no claims - bypassing cache",
                                    extra={
                                        "event": "cache_bypass",
                                        "task_id": task_id,
                                        "reason": "no_claims_in_cached_content"
                                    }
                                )
                                cache_hit = False
                                cached_response = None
                            else:
                                if self.redis_client:
                                    self.redis_client.incr("semantic_cache:hits")
                                    latency_ms = (time.time() - start_time) * 1000
                                    self.redis_client.set("semantic_cache:avg_cached_latency", latency_ms)
                                
                                logger.info(
                                    "Semantic cache HIT - avoiding LLM call",
                                    extra={
                                        "event": "cache_hit",
                                        "task_id": task_id,
                                        "similarity": cached_response.get('similarity', 0),
                                        "latency_ms": latency_ms
                                    }
                                )
                    except Exception as cache_error:
                        logger.warning(f"Semantic cache lookup failed: {cache_error}")
                elif retry_count > 0:
                    logger.info(f"Skipping cache on retry attempt {retry_count + 1}")

                if not cache_hit:
                    if self.redis_client and settings.ENABLE_SEMANTIC_CACHE:
                        self.redis_client.incr("semantic_cache:misses")

                    if self.use_local_llm and self.ollama_client:
                        content_text = await self.ollama_client.generate(
                            prompt=prompt,
                            temperature=settings.get('OLLAMA_TEMPERATURE', 0.7),
                            max_tokens=settings.get('OLLAMA_MAX_TOKENS', 2000)
                        )

                        estimated_tokens = len(prompt.split()) + len(content_text.split())
                        self.total_tokens += estimated_tokens
                        cost = 0.0

                        logger.info(
                            "LLM response received (Ollama)",
                            extra={
                                "event": "llm_call_complete",
                                "task_id": task_id,
                                "tokens_used": estimated_tokens,
                                "cost_eur": 0.0,
                                "model": settings.get('OLLAMA_MODEL', 'mixtral:8x7b'),
                                "llm_provider": "ollama"
                            }
                        )
                    else:
                        with get_openai_callback() as cb:
                            response = await self.llm.agenerate([[HumanMessage(content=prompt)]])

                            self.total_cost += cb.total_cost
                            self.total_tokens += cb.total_tokens
                            cost = cb.total_cost

                        logger.info(
                            "LLM response received (OpenAI)",
                            extra={
                                "event": "llm_call_complete",
                                "task_id": task_id,
                                "tokens_used": cb.total_tokens,
                                "cost_eur": round(cb.total_cost, 4),
                                "prompt_tokens": cb.prompt_tokens,
                                "completion_tokens": cb.completion_tokens,
                                "llm_provider": "openai"
                            }
                        )

                        content_text = response.generations[0][0].text
                    
                    # Don't cache here — only cache AFTER safety validation passes

                parsed_content = self._parse_generated_content(content_text, platform)

                # Extract claims from PARSED content (not raw LLM output) so the
                # "Claims Used:" footer doesn't inflate claims_used with unembedded IDs
                parsed_full_text = ' '.join(filter(None, [
                    parsed_content.get('headline', ''),
                    parsed_content.get('body', ''),
                    parsed_content.get('cta', '')
                ]))
                used_claims = self._extract_claim_citations(parsed_full_text)

                # Normalize [CLAIM_ID:CLM_nnn] → [CLM_nnn] for consistency
                import re
                content_text = re.sub(
                    r'\[CLAIM_ID:\s*(CLM_\d{3})\]', r'[\1]', content_text, flags=re.IGNORECASE
                )

                # Strip hallucinated claim IDs (e.g., [CLM_5]) that aren't in the library —
                # leaving them causes safety validation hard-fail (all scores = 0)
                import re
                all_bracket_tokens = re.findall(r'\[([A-Z0-9_]+)\]', content_text)
                claim_ids = [c['id'] for c in self.claim_library.get('claims', [])]
                hallucinated_claims = [t for t in all_bracket_tokens if t not in claim_ids and t not in used_claims]
                if hallucinated_claims:
                    for invalid_claim in set(hallucinated_claims):
                        pattern = r'\[' + re.escape(invalid_claim) + r'\]'
                        content_text = re.sub(pattern, '', content_text)
                    content_text = re.sub(r'  +', ' ', content_text)
                    parsed_content = self._parse_generated_content(content_text, platform)
                    logger.warning(
                        f"Removed {len(set(hallucinated_claims))} hallucinated claim citations from content: {list(set(hallucinated_claims))}",
                        extra={
                            "event": "hallucinated_claims_removed",
                            "task_id": task_id,
                            "invalid_claims": list(set(hallucinated_claims)),
                            "valid_claims": used_claims
                        }
                    )
                    actions_taken.append(f"Removed hallucinated claims: {list(set(hallucinated_claims))}")

                # Auto-fix excess claims to handle LLM instruction-following variance
                max_claims_allowed = validation_rules.get('max_claims_per_content', 3)
                if len(used_claims) > max_claims_allowed:
                    logger.warning(
                        f"LLM generated {len(used_claims)} claims (max: {max_claims_allowed}). "
                        f"Auto-fixing by truncating to first {max_claims_allowed} claims."
                    )

                    excess_claims = used_claims[max_claims_allowed:]
                    used_claims = used_claims[:max_claims_allowed]

                    import re
                    for excess_claim in excess_claims:
                        pattern = r'\[' + re.escape(excess_claim) + r'\]'
                        content_text = re.sub(pattern, '', content_text)

                    parsed_content = self._parse_generated_content(content_text, platform)

                    actions_taken.append(f"Auto-truncated claims from {len(used_claims) + len(excess_claims)} to {len(used_claims)}")

                # Sanitize forbidden phrases from LLM output — the model sometimes
                # uses superlatives despite prompt constraints (especially smaller models).
                forbidden_replacements = {
                    "unprecedented": "significant",
                    "game-changing": "impactful",
                    "game changing": "impactful",
                    "disruptive": "innovative",
                    "guaranteed": "designed to",
                    "guarantee": "commitment",
                    "risk-free": "low-risk",
                    "risk free": "low-risk",
                    "proven roi": "demonstrated returns",
                }
                sanitized_phrases = []
                body_lower = content_text.lower()
                for phrase, replacement in forbidden_replacements.items():
                    if phrase in body_lower:
                        content_text = re.sub(
                            re.escape(phrase), replacement, content_text, flags=re.IGNORECASE
                        )
                        sanitized_phrases.append(phrase)
                if sanitized_phrases:
                    parsed_content = self._parse_generated_content(content_text, platform)
                    logger.info(
                        f"Auto-sanitized {len(sanitized_phrases)} forbidden phrases: {sanitized_phrases}"
                    )
                    actions_taken.append(f"Sanitized forbidden phrases: {sanitized_phrases}")

                # Normalize em-dashes (—) and en-dashes (–) to dashes for brand consistency
                if '\u2014' in content_text or '\u2013' in content_text:
                    content_text = content_text.replace('\u2014', ' - ').replace('\u2013', '-')
                    parsed_content = self._parse_generated_content(content_text, platform)

                validation_result = self._validate_claim_usage(
                    content_text,
                    used_claims,
                    selected_claims
                )

                if not validation_result['valid']:
                    retry_count += 1
                    logger.warning(f"Claim validation failed (attempt {retry_count}/{max_retries}): {validation_result['reason']}")

                    if retry_count >= max_retries:
                        raise ValueError(f"Failed to generate compliant content after {max_retries} attempts: {validation_result['reason']}")

                    prompt = prompt + f"\n\nPREVIOUS ATTEMPT FAILED: {validation_result['reason']}. Please correct this issue."
                    continue
                
                if settings.ENABLE_SEMANTIC_CACHE and not cache_hit:
                    try:
                        model_name = settings.get('OLLAMA_MODEL', 'mixtral:8x7b') if self.use_local_llm else settings.OPENAI_MODEL
                        original_prompt = prompt.split("\n\nPREVIOUS ATTEMPT FAILED:")[0]
                        await self.semantic_cache.set(
                            original_prompt, content_text, model=model_name,
                            campaign_id=campaign_config.get('campaign_id')
                        )
                        
                        if self.redis_client:
                            latency_ms = (time.time() - start_time) * 1000
                            self.redis_client.set("semantic_cache:avg_uncached_latency", latency_ms)
                        
                        logger.info(
                            "Cached validated content for future use",
                            extra={
                                "event": "cache_set",
                                "task_id": task_id,
                                "claims_count": len(used_claims)
                            }
                        )
                    except Exception as cache_set_error:
                        logger.warning(f"Failed to cache response: {cache_set_error}")

                model_used = settings.get('OLLAMA_MODEL', 'mixtral:8x7b') if self.use_local_llm else settings.OPENAI_MODEL

                # Map content_type correctly per platform
                content_type_map = {
                    'blog': 'blog_post',
                    'linkedin': 'linkedin_ad',
                    'twitter': 'twitter_ad',
                    'email': 'email',
                }
                content_type = content_type_map.get(platform, f"{platform}_ad")

                # Sanitize fields for DB column limits
                def _truncate(value: str, max_len: int) -> str:
                    """Safely truncate a string to fit DB column, preserving word boundaries."""
                    if not value or len(value) <= max_len:
                        return value
                    truncated = value[:max_len - 3].rsplit(' ', 1)[0]
                    return truncated + '...'

                content = Content(
                    content_type=_truncate(content_type, 50),
                    headline=parsed_content.get('headline', ''),
                    body=parsed_content.get('body', content_text),
                    cta=parsed_content.get('cta', ''),
                    generated_by='content_generator',
                    prompt_used=prompt,
                    model_used=_truncate(model_used, 50),
                    claims_used=used_claims,
                    generation_cost=cost,
                    status=ContentStatus.GENERATED
                )

                if self.use_local_llm:
                    tokens_used = len(prompt.split()) + len(content_text.split())
                else:
                    tokens_used = self.total_tokens

                metadata = {
                    'model': model_used,
                    'tokens_used': tokens_used,
                    'cost': cost,
                    'context_sources': [c['source'] for c in context],
                    'claims_validated': True,
                    'retry_count': retry_count,
                    'selected_claims': [c['id'] for c in selected_claims],
                    'llm_provider': 'ollama' if self.use_local_llm else 'openai'
                }

                duration = (datetime.now() - self.task_start_time).total_seconds()
                actions_taken.append(f"Generated content with {len(used_claims)} claims")
                actions_taken.append(f"Content validation passed")

                task_result = {
                    'success': True,
                    'cost': cost,
                    'duration': duration,
                    'quality_score': 1.0,
                    'retry_count': retry_count
                }

                memory = create_memory_from_task(
                    agent_name="content_generator",
                    task_id=task_id,
                    task_description=f"Generate {platform} content for {persona}: {campaign_config.get('goal', '')}",
                    actions=actions_taken,
                    result=task_result,
                    human_feedback=None
                )

                await self.memory.store_memory(memory)

                if not cache_hit:
                    try:
                        from ..utils.cost_tracker import track_llm_cost
                        model_used_for_tracking = model_used
                        provider = "ollama" if self.use_local_llm else "openai"
                        
                        prompt_tokens = tokens_used // 2
                        completion_tokens = tokens_used // 2
                        
                        await track_llm_cost(
                            agent_type="content_generator",
                            model=model_used_for_tracking,
                            tokens_prompt=prompt_tokens,
                            tokens_completion=completion_tokens,
                            provider=provider,
                            campaign_id=str(campaign_config.get('campaign_id')) if campaign_config.get('campaign_id') else None,
                            action="content_generation"
                        )
                    except Exception as cost_error:
                        logger.warning(f"Failed to track generation costs: {cost_error}")

                logger.info(
                    "Content generation completed successfully",
                    extra={
                        "event": "content_generation_success",
                        "task_id": task_id,
                        "platform": platform,
                        "persona": persona,
                        "duration_seconds": round(duration, 2),
                        "cost_eur": round(cost, 4),
                        "tokens_used": tokens_used,
                        "claims_used": used_claims,
                        "num_claims": len(used_claims),
                        "retry_count": retry_count,
                        "content_length": len(content_text),
                        "headline": parsed_content.get('headline', '')[:50],
                        "llm_provider": 'ollama' if self.use_local_llm else 'openai',
                        "model": model_used
                    }
                )

                return content, metadata

            except Exception as e:
                logger.error(
                    "Content generation failed",
                    extra={
                        "event": "content_generation_error",
                        "task_id": task_id,
                        "platform": platform,
                        "persona": persona,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "retry_count": retry_count,
                        "max_retries": max_retries
                    },
                    exc_info=True
                )

                if self.task_start_time:
                    duration = (datetime.now() - self.task_start_time).total_seconds()
                    actions_taken.append(f"Failed: {str(e)}")

                    failure_result = {
                        'success': False,
                        'cost': self.total_cost,
                        'duration': duration,
                        'quality_score': 0.0,
                        'error': str(e)
                    }

                    failure_memory = create_memory_from_task(
                        agent_name="content_generator",
                        task_id=task_id,
                        task_description=f"Generate {platform} content for {persona}: {campaign_config.get('goal', '')}",
                        actions=actions_taken,
                        result=failure_result
                    )

                    await self.memory.store_memory(failure_memory)

                raise
    
    async def _retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector store"""
        try:
            results = await self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

            context = []
            for doc, score in results:
                context.append({
                    'text': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'score': score
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def _select_claims(self, persona: str, config: Dict) -> List[Dict]:
        """Select relevant claims for the persona and campaign"""
        all_claims = self.claim_library.get('claims', [])

        relevant_claims = []
        for claim in all_claims:
            if persona.lower() in claim.get('personas', []):
                relevant_claims.append(claim)
            elif any(goal in claim.get('goals', []) for goal in config.get('goals', [])):
                relevant_claims.append(claim)

        if not relevant_claims:
            logger.warning(f"No persona-specific claims for '{persona}', using all {len(all_claims)} claims")
            relevant_claims = all_claims

        max_claims = min(settings.MAX_CLAIMS_PER_CONTENT, len(relevant_claims))

        sorted_claims = sorted(
            relevant_claims,
            key=lambda x: x.get('priority', 0),
            reverse=True
        )

        return sorted_claims[:max_claims]
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format context for prompt"""
        if not context:
            return "No specific context available."
        
        formatted = []
        for i, ctx in enumerate(context, 1):
            formatted.append(f"{i}. {ctx['text'][:200]}...")
        
        return "\n".join(formatted)
    
    def _format_claims(self, claims: List[Dict]) -> str:
        """Format claims for prompt"""
        if not claims:
            return "No specific claims required."

        formatted = []
        for claim in claims:
            formatted.append(
                f"[{claim['id']}] {claim['text']} (Source: {claim['source']})"
            )

        return "\n".join(formatted)

    def _get_trending_topics(self, campaign_config: Dict[str, Any]) -> str:
        """
        Get trending topics from campaign config, market patterns, or competitor insights.
        No hardcoded fallback — derives topics from available data.
        """
        if 'trending_topics' in campaign_config and campaign_config['trending_topics']:
            topics = campaign_config['trending_topics']
            if isinstance(topics, list):
                return ", ".join(topics)
            return str(topics)

        # Extract from scraped content patterns if available
        content_patterns = campaign_config.get('content_patterns')
        if content_patterns:
            common_themes = content_patterns.get('common_themes', [])
            if common_themes:
                theme_names = [
                    t.get('theme', '') if isinstance(t, dict) else str(t)
                    for t in common_themes[:5]
                ]
                theme_names = [t for t in theme_names if t]
                if theme_names:
                    logger.info(f"Trending topics derived from scraped content patterns: {theme_names}")
                    return ", ".join(theme_names)

        # Extract from competitor differentiation opportunities
        if self.market_scraper is not None:
            try:
                diff_opps = self.market_scraper.get_differentiation_opportunities()
                if diff_opps:
                    topics = [" ".join(opp.split()[:5]) for opp in diff_opps[:3]]
                    if topics:
                        logger.info(f"Trending topics derived from competitive insights: {topics}")
                        return ", ".join(topics)
            except Exception as e:
                logger.warning(f"Failed to get trending topics from MarketScraper: {e}")

        return "Employee experience, People analytics, QWL-driven leadership"

    def _parse_generated_content(self, content: str, platform: str) -> Dict[str, str]:
        """Parse generated content into structured format.
        
        Uses case-insensitive matching and handles LLM output variations
        (e.g. 'Call to Action:' vs 'CTA:', 'Title:' mapped to 'headline').
        """
        import re

        parsed = {}
        
        lines = content.split('\n')
        
        current_section = None
        current_content = []

        # Case-insensitive section marker detection
        SECTION_MARKERS = [
            (r'^headline:\s*', 'headline'),
            (r'^body:\s*', 'body'),
            (r'^cta:\s*', 'cta'),
            (r'^call[\s\-]*to[\s\-]*action:\s*', 'cta'),
            (r'^subject:\s*', 'subject'),
            (r'^tweet:\s*', 'body'),
            (r'^title:\s*', 'headline'),
            (r'^meta\s*description:\s*', 'meta_description'),
            (r'^introduction:\s*', 'body'),
            (r'^seo\s*keywords:\s*', '_seo_keywords'),
            (r'^claims?\s*used:\s*', '_claims_terminal'),
        ]

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_section and current_section not in ('_seo_keywords', '_claims_terminal'):
                    current_content.append('')
                continue

            matched = False
            for pattern, section_key in SECTION_MARKERS:
                m = re.match(pattern, line_stripped, re.IGNORECASE)
                if m:
                    if current_section and current_content and current_section not in ('_seo_keywords', '_claims_terminal'):
                        parsed[current_section] = ' '.join(c for c in current_content if c).strip()
                    remainder = line_stripped[m.end():].strip()
                    if section_key == '_seo_keywords':
                        parsed['seo_keywords'] = remainder
                        current_section = None
                        current_content = []
                    elif section_key == '_claims_terminal':
                        current_section = None
                        current_content = []
                    else:
                        current_section = section_key
                        current_content = [remainder] if remainder else []
                    matched = True
                    break

            if matched:
                continue

            if line_stripped.startswith('## Conclusion'):
                if current_section and current_content:
                    parsed[current_section] = ' '.join(c for c in current_content if c).strip()
                current_section = 'conclusion'
                current_content = []
            elif line_stripped.startswith('## '):
                if current_section and current_content:
                    existing_body = parsed.get('body', '')
                    new_part = ' '.join(c for c in current_content if c).strip()
                    if current_section == 'body' and existing_body:
                        parsed['body'] = existing_body + ' ' + new_part
                    else:
                        parsed[current_section] = new_part
                current_section = 'body'
                current_content = [line_stripped]
            elif current_section and current_section not in ('_seo_keywords', '_claims_terminal'):
                current_content.append(line_stripped)
        
        if current_section and current_content and current_section not in ('_seo_keywords', '_claims_terminal'):
            existing = parsed.get(current_section, '')
            new_part = ' '.join(c for c in current_content if c).strip()
            if current_section == 'body' and existing:
                parsed[current_section] = existing + ' ' + new_part
            else:
                parsed[current_section] = new_part
        
        if not parsed:
            parsed['body'] = content

        # Merge conclusion into body for blog posts
        if 'conclusion' in parsed and parsed['conclusion']:
            body = parsed.get('body', '')
            parsed['body'] = (body + ' ' + parsed['conclusion']).strip() if body else parsed['conclusion']
        
        # Clean trailing claims references from all fields
        claims_pattern = r'\s*Claims?\s*Used:.*$'
        for key in list(parsed.keys()):
            if parsed[key] and isinstance(parsed[key], str):
                parsed[key] = re.sub(claims_pattern, '', parsed[key], flags=re.IGNORECASE).strip()
        
        return parsed
    
    def _extract_claim_citations(self, content: str) -> List[str]:
        """Extract UNIQUE claim IDs from content"""
        import re

        valid_claims = []
        claim_ids = [c['id'] for c in self.claim_library.get('claims', [])]

        pattern_prefixed = r'\[CLAIM_ID:\s*([A-Z0-9_]+)\]'
        matches = re.findall(pattern_prefixed, content, re.IGNORECASE)
        for match in matches:
            if match in claim_ids and match not in valid_claims:
                valid_claims.append(match)

        pattern_direct = r'\[([A-Z0-9_]+)\]'
        matches = re.findall(pattern_direct, content)
        for match in matches:
            if match in claim_ids and match not in valid_claims:
                valid_claims.append(match)

        return valid_claims

    def _validate_claim_usage(
        self,
        content: str,
        used_claims: List[str],
        selected_claims: List[Dict]
    ) -> Dict[str, Any]:
        """
        Validate that claim usage meets requirements

        Args:
            content: Generated content text
            used_claims: List of claim IDs found in content
            selected_claims: List of claims that were provided to LLM

        Returns:
            Validation result with 'valid' bool and 'reason' string
        """
        validation_rules = self.claim_library.get('validation_rules', {})
        min_claims = validation_rules.get('min_claims_per_content', 1)
        max_claims = validation_rules.get('max_claims_per_content', 3)
        require_citation = validation_rules.get('require_citation', True)

        if len(used_claims) < min_claims:
            return {
                'valid': False,
                'reason': f"Insufficient claims used: {len(used_claims)}/{min_claims} minimum required. Found: {used_claims}"
            }

        if len(used_claims) > max_claims:
            return {
                'valid': False,
                'reason': f"Too many claims used: {len(used_claims)}/{max_claims} maximum allowed"
            }

        selected_claim_ids = [c['id'] for c in selected_claims]
        invalid_claims = [c for c in used_claims if c not in selected_claim_ids]
        if invalid_claims:
            return {
                'valid': False,
                'reason': f"Claims used that were not provided: {invalid_claims}"
            }

        if require_citation:
            citation_format = validation_rules.get('citation_format', '[CLAIM_ID]')
            for claim_id in used_claims:
                expected_direct = citation_format.replace('CLAIM_ID', claim_id)
                expected_prefixed = f"[CLAIM_ID:{claim_id}]"
                expected_prefixed_space = f"[CLAIM_ID: {claim_id}]"
                
                if (expected_direct not in content and 
                    expected_prefixed not in content and 
                    expected_prefixed_space not in content):
                    return {
                        'valid': False,
                        'reason': f"Claim {claim_id} not properly cited with format {citation_format}"
                    }

        current_date = datetime.now()
        for claim_id in used_claims:
            claim = next((c for c in selected_claims if c['id'] == claim_id), None)
            if claim and claim.get('expiry_date'):
                from dateutil import parser
                try:
                    expiry_date = parser.parse(claim['expiry_date'])
                    if current_date > expiry_date:
                        return {
                            'valid': False,
                            'reason': f"Claim {claim_id} has expired (expiry: {claim['expiry_date']})"
                        }
                except:
                    pass  # If date parsing fails, skip expiry check

        return {
            'valid': True,
            'reason': 'All claim requirements met'
        }

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'total_cost': self.total_cost,
            'total_tokens': self.total_tokens,
            'average_cost_per_generation': self.total_cost / max(1, self.total_tokens),
            'claim_library_version': self.claim_library.get('version', 'unknown')
        }