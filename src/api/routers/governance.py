"""
Governance and safety API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import logging

from ...data_layer.database.models import Content, HITLQueue, ContentStatus, Campaign
from ...governance.hitl_queue import HITLQueueManager
from ...governance.safety_scorer import SafetyScorer
from ...governance.claim_validator import ClaimValidator
from ..dependencies import get_db, get_current_user
from pydantic import BaseModel, Field
import csv
import json
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/hitl-queue", response_model=List[Dict[str, Any]])
async def get_hitl_queue(
    status: Optional[str] = Query(default="pending"),
    limit: int = Query(default=50, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get items in HITL review queue"""
    try:
        query = select(HITLQueue).join(Content)

        if status and status != "all":
            query = query.where(HITLQueue.status == status)
        
        query = query.order_by(HITLQueue.priority.desc(), HITLQueue.created_at)
        query = query.limit(limit)
        
        result = await db.execute(query)
        items = result.scalars().all()
        
        queue_items = []
        for item in items:
            content_result = await db.execute(
                select(Content).where(Content.id == item.content_id)
            )
            content = content_result.scalar_one_or_none()
            
            if content:
                campaign_name = None
                campaign_platform = None
                if content.campaign_id:
                    campaign_result = await db.execute(
                        select(Campaign).where(Campaign.id == content.campaign_id)
                    )
                    campaign = campaign_result.scalar_one_or_none()
                    if campaign:
                        campaign_name = campaign.name
                        campaign_platform = campaign.platform.value if campaign.platform else None
                
                queue_items.append({
                    "id": str(item.id),
                    "content_id": str(item.content_id),
                    "campaign_id": str(content.campaign_id) if content.campaign_id else None,
                    "campaign_name": campaign_name or "Unknown Campaign",
                    "platform": campaign_platform or "linkedin",
                    "priority": item.priority,
                    "reason": item.reason,
                    "review_reason": item.reason,
                    "status": item.status,
                    "headline": content.headline,
                    "body": content.body,
                    "cta": content.cta,
                    "claims_used": content.claims_used or [],
                    "safety_score": content.safety_score,
                    "toxicity_score": content.toxicity_score,
                    "factuality_score": content.factuality_score,
                    "brand_score": content.brand_alignment_score,
                    "created_at": item.created_at.isoformat(),
                    "decision": item.decision,
                    "feedback": item.feedback,
                    "reviewed_by": content.reviewed_by or item.assigned_to,
                    "reviewed_at": content.reviewed_at.isoformat() if content.reviewed_at else (item.completed_at.isoformat() if item.completed_at else None)
                })
        
        return queue_items
    except Exception as e:
        logger.error(f"Failed to get HITL queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/review", response_model=Dict[str, Any])
async def submit_review(
    review_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Submit content review decision and resume workflow if approved"""
    try:
        content_id = review_data.get("content_id")
        raw_decision = (review_data.get("decision") or "").strip().lower()
        # Normalize: accept both "approve"/"approved" and "reject"/"rejected"
        if raw_decision in ("approve", "approved"):
            decision = "approve"
        elif raw_decision in ("reject", "rejected"):
            decision = "reject"
        else:
            decision = raw_decision
        feedback = review_data.get("feedback", "")
        modifications = review_data.get("modifications", {})
        reviewer_email = review_data.get("reviewer_email", "reviewer@example.com")

        content_query = await db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = content_query.scalar_one_or_none()

        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        campaign_id = str(content.campaign_id)

        await db.execute(
            update(HITLQueue)
            .where(HITLQueue.content_id == content_id)
            .values(
                status="completed",
                decision=decision,
                feedback=feedback,
                modifications=modifications,
                completed_at=datetime.utcnow()
            )
        )

        if decision == "approve":
            content_status = ContentStatus.APPROVED
        elif decision == "reject":
            content_status = ContentStatus.REJECTED
        else:
            content_status = ContentStatus.GENERATED

        await db.execute(
            update(Content)
            .where(Content.id == content_id)
            .values(
                status=content_status,
                review_notes=feedback if feedback else None,
                reviewed_by=reviewer_email,
                reviewed_at=datetime.utcnow()
            )
        )

        await db.commit()

        try:
            from ...ai_layer.learning.governance_metrics_tracker import GovernanceMetricsTracker
            tracker = GovernanceMetricsTracker()
            await tracker.save_period_metrics("daily")
        except Exception as e:
            logger.warning(f"Failed to snapshot governance metrics: {e}")

        try:
            from ...data_layer.database.models import WorkflowEvent, WorkflowEventType
            
            await db.execute(
                update(WorkflowEvent)
                .where(and_(
                    WorkflowEvent.content_id == content_id,
                    WorkflowEvent.event_type == WorkflowEventType.HITL_QUEUE_ADDED,
                    WorkflowEvent.is_dismissed == False
                ))
                .values(
                    is_dismissed=True,
                    dismissed_at=datetime.utcnow()
                )
            )
            await db.commit()
            logger.info(f"Dismissed HITL workflow events for content {content_id}")
        except Exception as e:
            logger.warning(f"Failed to dismiss workflow events: {e}")


        logger.info(
            f"Review recorded: decision={decision}, feedback={'Yes' if feedback else 'No'}",
            extra={
                "content_id": content_id,
                "decision": decision,
                "has_feedback": bool(feedback),
                "reviewer": reviewer_email
            }
        )

        # Collect content info for episodic memory (fast DB read, no embedding)
        content_result = await db.execute(
            select(Content).where(Content.id == content_id)
        )
        content_obj = content_result.scalar_one_or_none()
        _content_headline = content_obj.headline if content_obj else "N/A"
        _content_body_len = len(content_obj.body or "") if content_obj else 0
        _content_type = content_obj.content_type if content_obj else "unknown"
        _safety_score = float(content_obj.safety_score) if content_obj and content_obj.safety_score else 0.0

        import asyncio

        async def _store_episodic_memory(cid, c_id, dec, fb, headline, body_len, ctype, sscore):
            """Store review feedback in episodic memory (background)."""
            try:
                from ...ai_layer.memory.episodic_memory import EpisodicMemoryStore, AgentMemory
                memory_store = EpisodicMemoryStore(agent_name="content_generator")
                outcome = "success" if dec == "approve" else "failure"
                agent_memory = AgentMemory(
                    agent_name="content_generator",
                    task_id=str(c_id),
                    task_description=f"Generated {ctype} for campaign {cid}",
                    actions_taken=[f"Generated headline: {headline}", f"Body ({body_len} chars)"],
                    outcome=outcome,
                    metrics={"safety_score": sscore, "review_decision": dec},
                    human_feedback=fb if fb else None,
                    lessons_learned=f"Content {outcome}. " + (fb if fb else "No specific feedback provided.")
                )
                await memory_store.store_memory(agent_memory)
                logger.info(f"Stored review feedback in episodic memory")
            except Exception as e:
                logger.warning(f"Failed to store feedback in episodic memory: {e}")

        if decision == "reject":
            logger.info(f"Content rejected, triggering auto-regeneration with feedback")

            async def _reject_background(cid: str, c_id, fb: str, headline, body_len, ctype, sscore):
                """Cache invalidation + episodic memory + regeneration — all background."""
                # 1. Store episodic memory
                await _store_episodic_memory(cid, c_id, "reject", fb, headline, body_len, ctype, sscore)
                # 2. Invalidate semantic cache
                try:
                    from ...data_layer.vector_store.semantic_cache import SemanticCache
                    cache = SemanticCache()
                    await cache.initialize()
                    await cache.invalidate_campaign(cid)
                except Exception as e:
                    logger.warning(f"Cache invalidation on rejection failed: {e}")
                # 3. Regenerate
                try:
                    from ...ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator
                    orch = MarketingOrchestrator()
                    await orch.run_campaign_workflow(
                        campaign_id=cid,
                        previous_feedback=fb or "Content rejected - needs improvement"
                    )
                    logger.info(f"Auto-regeneration completed for campaign {cid}")
                except Exception as e:
                    logger.error(f"Auto-regeneration failed for campaign {cid}: {e}")

            asyncio.create_task(
                _reject_background(campaign_id, content_id, feedback,
                                   _content_headline, _content_body_len, _content_type, _safety_score)
            )

            return {
                "status": "success",
                "content_id": content_id,
                "decision": decision,
                "message": f"Content rejected and regeneration started in background",
                "regeneration_initiated": True
            }

        # For approve/other: store memory in background, but continue with workflow
        asyncio.create_task(
            _store_episodic_memory(campaign_id, content_id, decision, feedback,
                                  _content_headline, _content_body_len, _content_type, _safety_score)
        )

        if decision == "approve":
            logger.info(f"Content approved, resuming workflow for campaign {campaign_id}")

            from ...ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator

            orchestrator = MarketingOrchestrator()

            try:
                result = await orchestrator.resume_workflow_after_approval(
                    campaign_id=campaign_id,
                    content_id=content_id
                )

                logger.info(
                    f"Workflow resumption completed",
                    extra={
                        "campaign_id": campaign_id,
                        "content_id": content_id,
                        "success": result.get('success'),
                        "deployment_status": result.get('deployment_status')
                    }
                )

                return {
                    "status": "success",
                    "content_id": content_id,
                    "decision": decision,
                    "message": f"Content approved and workflow resumed",
                    "workflow_status": result.get('deployment_status', 'in_progress')
                }
            except Exception as e:
                logger.error(f"Workflow resumption failed: {e}", exc_info=True)
                return {
                    "status": "success",
                    "content_id": content_id,
                    "decision": decision,
                    "message": f"Content approved but workflow resumption failed: {str(e)}",
                    "workflow_error": str(e)
                }

        return {
            "status": "success",
            "content_id": content_id,
            "decision": decision,
            "message": f"Content {decision}ed successfully"
        }
    except Exception as e:
        logger.error(f"Failed to submit review: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/regenerate", response_model=Dict[str, Any])
async def regenerate_content(
    regenerate_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """Regenerate content with low safety score"""
    try:
        content_id = regenerate_data.get("content_id")
        feedback = regenerate_data.get("feedback", "Safety score too low - regenerating")

        content_query = await db.execute(
            select(Content).where(Content.id == content_id)
        )
        content = content_query.scalar_one_or_none()

        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        campaign_id = str(content.campaign_id)

        await db.execute(
            update(HITLQueue)
            .where(HITLQueue.content_id == content_id)
            .values(
                status="completed",
                decision="regenerate",
                feedback=feedback,
                completed_at=datetime.utcnow()
            )
        )

        await db.execute(
            update(Content)
            .where(Content.id == content_id)
            .values(
                status=ContentStatus.REJECTED,
                review_notes=feedback,
                reviewed_at=datetime.utcnow()
            )
        )

        await db.commit()

        try:
            from ...data_layer.database.models import WorkflowEvent, WorkflowEventType
            
            await db.execute(
                update(WorkflowEvent)
                .where(and_(
                    WorkflowEvent.content_id == content_id,
                    WorkflowEvent.event_type == WorkflowEventType.HITL_QUEUE_ADDED,
                    WorkflowEvent.is_dismissed == False
                ))
                .values(
                    is_dismissed=True,
                    dismissed_at=datetime.utcnow()
                )
            )
            await db.commit()
            logger.info(f"Dismissed HITL workflow events for regenerated content {content_id}")
        except Exception as e:
            logger.warning(f"Failed to dismiss workflow events: {e}")

        logger.info(f"Regenerating content for campaign {campaign_id}")

        from ...ai_layer.orchestration.langgraph_supervisor import MarketingOrchestrator

        try:
            orchestrator = MarketingOrchestrator()

            result = await orchestrator.run_campaign_workflow(
                campaign_id=campaign_id,
                previous_feedback=feedback
            )

            logger.info(
                f"Content regeneration workflow started",
                extra={
                    "campaign_id": campaign_id,
                    "old_content_id": content_id,
                    "success": result.get('success')
                }
            )

            return {
                "status": "success",
                "content_id": content_id,
                "campaign_id": campaign_id,
                "message": "Content marked for regeneration and new workflow started",
                "workflow_initiated": True
            }
        except Exception as e:
            logger.error(f"Workflow regeneration failed: {e}", exc_info=True)
            return {
                "status": "success",
                "content_id": content_id,
                "campaign_id": campaign_id,
                "message": f"Content rejected but workflow regeneration failed: {str(e)}",
                "workflow_error": str(e)
            }
    except Exception as e:
        logger.error(f"Failed to regenerate content: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate", response_model=Dict[str, Any])
async def validate_content(
    content_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """Validate content safety"""
    try:
        scorer = SafetyScorer()

        content_text = content_data.get("body") or content_data.get("content", "")
        
        claims = content_data.get("claims_used") or content_data.get("claims", [])
        if not claims and content_text:
            import re
            pattern = r'\[([A-Z0-9_]+)\]'
            claims = list(set(re.findall(pattern, content_text)))

        validation_result = await scorer.validate(
            content=content_text,
            headline=content_data.get("headline", ""),
            claims_used=claims
        )

        return validation_result
    except Exception as e:
        logger.error(f"Failed to validate content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety-stats", response_model=Dict[str, Any])
async def get_safety_statistics(
    days: int = Query(default=30),
    db: AsyncSession = Depends(get_db)
):
    """Get safety validation statistics"""
    try:
        since_date = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(Content)
            .where(Content.created_at >= since_date)
        )
        contents = result.scalars().all()

        total_content = len(contents)
        # DEPLOYED content is considered approved (went through approval and was deployed)
        approved = sum(1 for c in contents if c.status in [ContentStatus.APPROVED, ContentStatus.DEPLOYED])
        rejected = sum(1 for c in contents if c.status == ContentStatus.REJECTED)
        pending = sum(1 for c in contents if c.status == ContentStatus.PENDING_REVIEW)

        safety_scores = [c.safety_score for c in contents if c.safety_score is not None]
        toxicity_scores = [c.toxicity_score for c in contents if c.toxicity_score is not None]
        factuality_scores = [c.factuality_score for c in contents if c.factuality_score is not None]

        avg_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else 0.0
        avg_toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0.0
        avg_factuality = sum(factuality_scores) / len(factuality_scores) if factuality_scores else 0.0

        return {
            "period_days": days,
            "total_content": total_content,
            "approved": approved,
            "rejected": rejected,
            "pending_review": pending,
            "approval_rate": approved / max(total_content, 1),
            "average_safety_score": avg_safety_score,
            "average_toxicity_score": avg_toxicity,
            "average_factuality_score": avg_factuality,
            "high_risk_content": sum(1 for c in contents if (c.safety_score or 0) < 0.5)
        }
    except Exception as e:
        logger.error(f"Failed to get safety stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_review_history(
    limit: int = Query(default=50, le=200),
    days: int = Query(default=30),
    db: AsyncSession = Depends(get_db)
):
    """Get content review history"""
    try:
        since_date = datetime.utcnow() - timedelta(days=days)

        result = await db.execute(
            select(HITLQueue)
            .where(and_(
                HITLQueue.status == "completed",
                HITLQueue.completed_at >= since_date
            ))
            .order_by(HITLQueue.completed_at.desc())
            .limit(limit)
        )
        items = result.scalars().all()

        history = []
        for item in items:
            content_result = await db.execute(
                select(Content).where(Content.id == item.content_id)
            )
            content = content_result.scalar_one_or_none()

            if content:
                campaign_name = None
                campaign_platform = None
                if content.campaign_id:
                    camp_result = await db.execute(
                        select(Campaign).where(Campaign.id == content.campaign_id)
                    )
                    campaign_obj = camp_result.scalar_one_or_none()
                    if campaign_obj:
                        campaign_name = campaign_obj.name
                        campaign_platform = campaign_obj.platform.value if campaign_obj.platform else None

                history.append({
                    "id": str(item.id),
                    "content_id": str(item.content_id),
                    "campaign_id": str(content.campaign_id),
                    "campaign_name": campaign_name or "Unknown Campaign",
                    "platform": campaign_platform or "linkedin",
                    "headline": content.headline,
                    "body": content.body,
                    "content_type": content.content_type,
                    "decision": item.decision,
                    # Use 'is not None' to preserve 0.0 values (0.0 is a valid score!)
                    "safety_score": float(content.safety_score) if content.safety_score is not None else None,
                    "toxicity_score": float(content.toxicity_score) if content.toxicity_score is not None else None,
                    "factuality_score": float(content.factuality_score) if content.factuality_score is not None else None,
                    "brand_alignment_score": float(content.brand_alignment_score) if content.brand_alignment_score is not None else None,
                    "reviewed_by": content.reviewed_by or "system",
                    "reviewed_at": item.completed_at.isoformat() if item.completed_at else None,
                    "feedback": item.feedback,
                    "review_notes": content.review_notes,
                    "priority": item.priority,
                    "status": content.status.value if content.status else None,
                    "created_at": content.created_at.isoformat() if content.created_at else None
                })

        return history
    except Exception as e:
        logger.error(f"Failed to get review history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/golden-tests", response_model=Dict[str, Any])
async def get_golden_test_results(
    db: AsyncSession = Depends(get_db)
):
    """Get latest golden test results"""
    try:
        import os
        import json
        from pathlib import Path

        results_file = Path("tests/golden/golden_test_results.json")
        if not results_file.exists():
            results_file = Path("golden_test_results.json")

        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                return results
        else:
            return {
                "pass_rate": 0.0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "last_run": None,
                "test_details": []
            }
    except Exception as e:
        logger.error(f"Failed to get golden test results: {e}")
        return {
            "pass_rate": 0.0,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "last_run": None,
            "test_details": []
        }

@router.post("/run-golden-tests", response_model=Dict[str, Any])
async def run_golden_tests(
    db: AsyncSession = Depends(get_db)
):
    """Trigger golden test suite run"""
    try:
        import subprocess
        import json
        from pathlib import Path

        import os
        result = subprocess.run(
            ["python", "tests/golden/test_runner.py"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONPATH": "/app"}
        )

        if result.returncode != 0:
            logger.error(f"Golden tests failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Tests failed: {result.stderr}")

        import tempfile
        results_file = Path(tempfile.gettempdir()) / "golden_test_results.json"
        if not results_file.exists():
            results_file = Path("tests/golden/golden_test_results.json")
        if not results_file.exists():
            results_file = Path("golden_test_results.json")
        if not results_file.exists():
            raise HTTPException(status_code=500, detail="Test results file not found")

        with open(results_file, 'r') as f:
            test_results = json.load(f)

        return {"status": "completed", "results": test_results}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Test execution timed out")
    except Exception as e:
        logger.error(f"Failed to run golden tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/contents", response_model=List[Dict[str, Any]])
async def list_contents(
    limit: int = Query(default=100, le=2000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List all content items with safety scores"""
    try:
        result = await db.execute(
            select(Content)
            .order_by(Content.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        contents = result.scalars().all()

        return [{
            "id": str(c.id),
            "campaign_id": str(c.campaign_id) if c.campaign_id else None,
            "content_type": c.content_type,
            "headline": c.headline,
            "body": c.body,
            "cta": c.cta,
            "status": c.status.value if hasattr(c.status, 'value') else c.status,
            "safety_score": c.safety_score,
            "toxicity_score": c.toxicity_score,
            "factuality_score": c.factuality_score,
            "brand_alignment_score": c.brand_alignment_score,
            "impressions": c.impressions or 0,
            "clicks": c.clicks or 0,
            "conversions": c.conversions or 0,
            "created_at": c.created_at.isoformat() if c.created_at else None
        } for c in contents]
    except Exception as e:
        logger.error(f"Failed to list contents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Research Plan Reference: Section 6.1 - "10-20 canonical, sourced claims"

class ClaimCreate(BaseModel):
    """Schema for creating a new claim"""
    claim_text: str = Field(..., description="The claim statement text")
    claim_type: str = Field(default="product", description="Type: product, impact, testimonial")
    personas: List[str] = Field(default=[], description="Target personas")
    tags: List[str] = Field(default=[], description="Categorization tags")
    source_title: str = Field(default="", description="Source document title")
    source_url: str = Field(default="", description="URL to source evidence")
    source_date: str = Field(default="", description="Date of source")
    evidence_excerpt: str = Field(default="", description="Excerpt from source")
    confidence: int = Field(default=3, ge=1, le=5, description="Confidence level 1-5")


class ClaimUpdate(BaseModel):
    """Schema for updating an existing claim"""
    claim_text: Optional[str] = None
    claim_type: Optional[str] = None
    personas: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    source_date: Optional[str] = None
    evidence_excerpt: Optional[str] = None
    confidence: Optional[int] = None


@router.get("/claims", response_model=List[Dict[str, Any]])
async def list_claims(
    persona: Optional[str] = Query(None, description="Filter by persona"),
    claim_type: Optional[str] = Query(None, description="Filter by claim type")
):
    """
    List all claims from the Claim Library.

    Research Plan Reference: Section 6.1 - "version-controlled set of 10-20 canonical, sourced claims"
    """
    try:
        validator = ClaimValidator()
        claims = list(validator.claims_library.values())

        if persona:
            claims = [c for c in claims if persona in c.get('personas', [])]
        if claim_type:
            claims = [c for c in claims if c.get('type', '') == claim_type]

        return claims
    except Exception as e:
        logger.error(f"Failed to list claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims/stats", response_model=Dict[str, Any])
async def get_claims_stats():
    """Get statistics about the Claim Library"""
    try:
        validator = ClaimValidator()
        claims = list(validator.claims_library.values())

        by_type = {}
        by_persona = {}
        confidence_dist = {}

        for claim in claims:
            ctype = claim.get('type', 'unknown')
            by_type[ctype] = by_type.get(ctype, 0) + 1

            for persona in claim.get('personas', []):
                by_persona[persona] = by_persona.get(persona, 0) + 1

            conf = str(claim.get('confidence', 3))
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

        return {
            "total_claims": len(claims),
            "by_type": by_type,
            "by_persona": by_persona,
            "confidence_distribution": confidence_dist,
            "data_source": "CSV" if Path("data/claim_library/claims.csv").exists() else "YAML"
        }
    except Exception as e:
        logger.error(f"Failed to get claims stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims/{claim_id}", response_model=Dict[str, Any])
async def get_claim(claim_id: str):
    """Get a single claim by ID"""
    try:
        validator = ClaimValidator()
        claim = validator.get_claim_by_id(claim_id)

        if not claim:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

        return claim
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get claim {claim_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims/for-persona/{persona}", response_model=List[Dict[str, Any]])
async def get_claims_for_persona(persona: str):
    """Get all claims relevant to a specific persona"""
    try:
        validator = ClaimValidator()
        return validator.get_claims_for_persona(persona)
    except Exception as e:
        logger.error(f"Failed to get claims for persona {persona}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/claims", response_model=Dict[str, Any])
async def create_claim(claim_data: ClaimCreate):
    """
    Create a new claim and add it to the Claim Library CSV.

    Research Plan Reference: Section 6.1 - Claim Library management
    """
    try:
        csv_path = Path("data/claim_library/claims.csv")

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        validator = ClaimValidator()
        existing_ids = list(validator.claims_library.keys())

        max_num = 0
        for cid in existing_ids:
            try:
                num = int(cid.split('_')[-1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass

        new_id = f"CLM_{max_num + 1:03d}"

        new_claim = {
            'id': new_id,
            'claim_text': claim_data.claim_text,
            'claim_type': claim_data.claim_type,
            'personas': json.dumps(claim_data.personas),
            'tags': json.dumps(claim_data.tags),
            'source_title': claim_data.source_title,
            'source_url': claim_data.source_url,
            'source_date': claim_data.source_date,
            'evidence_excerpt': claim_data.evidence_excerpt,
            'confidence': claim_data.confidence
        }

        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'claim_text', 'claim_type', 'personas', 'tags',
                         'source_title', 'source_url', 'source_date', 'evidence_excerpt', 'confidence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(new_claim)

        logger.info(f"Created new claim: {new_id}")

        return {
            "success": True,
            "claim_id": new_id,
            "message": f"Claim {new_id} created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create claim: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/claims/{claim_id}", response_model=Dict[str, Any])
async def update_claim(claim_id: str, claim_data: ClaimUpdate):
    """
    Update an existing claim in the Claim Library.
    """
    try:
        csv_path = Path("data/claim_library/claims.csv")

        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Claims CSV file not found")

        claims = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                claims.append(row)

        found = False
        for i, claim in enumerate(claims):
            if claim.get('id') == claim_id:
                found = True
                if claim_data.claim_text is not None:
                    claims[i]['claim_text'] = claim_data.claim_text
                if claim_data.claim_type is not None:
                    claims[i]['claim_type'] = claim_data.claim_type
                if claim_data.personas is not None:
                    claims[i]['personas'] = json.dumps(claim_data.personas)
                if claim_data.tags is not None:
                    claims[i]['tags'] = json.dumps(claim_data.tags)
                if claim_data.source_title is not None:
                    claims[i]['source_title'] = claim_data.source_title
                if claim_data.source_url is not None:
                    claims[i]['source_url'] = claim_data.source_url
                if claim_data.source_date is not None:
                    claims[i]['source_date'] = claim_data.source_date
                if claim_data.evidence_excerpt is not None:
                    claims[i]['evidence_excerpt'] = claim_data.evidence_excerpt
                if claim_data.confidence is not None:
                    claims[i]['confidence'] = str(claim_data.confidence)
                break

        if not found:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

        fieldnames = ['id', 'claim_text', 'claim_type', 'personas', 'tags',
                     'source_title', 'source_url', 'source_date', 'evidence_excerpt', 'confidence']

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(claims)

        logger.info(f"Updated claim: {claim_id}")

        return {
            "success": True,
            "claim_id": claim_id,
            "message": f"Claim {claim_id} updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update claim {claim_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/claims/{claim_id}", response_model=Dict[str, Any])
async def delete_claim(claim_id: str):
    """
    Delete a claim from the Claim Library.
    """
    try:
        csv_path = Path("data/claim_library/claims.csv")

        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Claims CSV file not found")

        claims = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                claims.append(row)

        original_count = len(claims)
        claims = [c for c in claims if c.get('id') != claim_id]

        if len(claims) == original_count:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

        fieldnames = ['id', 'claim_text', 'claim_type', 'personas', 'tags',
                     'source_title', 'source_url', 'source_date', 'evidence_excerpt', 'confidence']

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(claims)

        logger.info(f"Deleted claim: {claim_id}")

        return {
            "success": True,
            "claim_id": claim_id,
            "message": f"Claim {claim_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete claim {claim_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims/validate/library", response_model=Dict[str, Any])
async def validate_claim_library():
    """
    Validate the entire Claim Library for completeness and correctness.

    Research Plan Reference: Section 6.1 - ensures all claims have proper sources
    """
    try:
        validator = ClaimValidator()
        validation_result = validator.validate_claim_library()

        return {
            "valid": validation_result.get("valid", False),
            "total_claims": validation_result.get("total_claims", 0),
            "issues": validation_result.get("issues", []),
            "message": "Library is valid" if validation_result.get("valid") else f"Found {len(validation_result.get('issues', []))} issues"
        }
    except Exception as e:
        logger.error(f"Failed to validate claim library: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/claims/validate/content", response_model=Dict[str, Any])
async def validate_content_claims(
    content_data: Dict[str, Any]
):
    """
    Validate that content properly cites claims from the Claim Library.

    Research Plan Reference: Section 10.2 - "100% of claims must be cited"
    """
    try:
        validator = ClaimValidator()

        content_text = content_data.get("content", "")
        claims_used = content_data.get("claims_used", [])

        result = validator.validate_content(content_text, claims_used)

        return {
            "is_valid": result.get("is_valid", False),
            "all_claims_cited": result.get("all_claims_cited", False),
            "score": result.get("score", 0.0),
            "claims_found": result.get("claims_found", []),
            "citations_found": result.get("citations_found", []),
            "missing_citations": result.get("missing_citations", []),
            "hallucinated_claims": result.get("hallucinated_claims", [])
        }
    except Exception as e:
        logger.error(f"Failed to validate content claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))