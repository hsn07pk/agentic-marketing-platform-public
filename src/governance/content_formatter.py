"""
Content Formatter for Publishing

Handles expansion of internal citation formats [CLM_XXX] to human-readable
text before content is published to external platforms.

Industry Standard: Internal systems use citation IDs for tracking and validation,
but published content shows the actual claim text or removes citations entirely.
"""
import re
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from functools import lru_cache

from ..config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_claim_library() -> Dict[str, str]:
    """
    Load claim library and cache it for claim text expansion.
    
    Returns:
        Dict mapping claim IDs to claim text
    """
    try:
        claim_path = Path(settings.CLAIM_LIBRARY_PATH)
        
        if claim_path.exists():
            with open(claim_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return {
                    c['id']: c.get('text', c.get('claim_text', ''))
                    for c in data.get('claims', [])
                }
    except Exception as e:
        logger.error(f"Failed to load claim library for formatting: {e}")
    
    return {}


def expand_claim_citations(
    content: str,
    format_style: str = "inline",
    claim_library: Optional[Dict[str, str]] = None
) -> str:
    """
    Expand claim citations [CLM_XXX] to human-readable text.
    
    This is the canonical function for preparing content for external publishing.
    All content going to social platforms, emails, or external systems should
    pass through this function.
    
    Args:
        content: Content with [CLM_XXX] or [CLAIM_ID:CLM_XXX] citations
        format_style: 
            - "inline": Replace [CLM_XXX] with claim text inline (default for social)
            - "footnote": Keep [1], [2] and add footnotes at end (good for email)
            - "remove": Remove citations entirely (clean copy)
            - "keep": Keep original [CLM_XXX] format (for internal use only)
        claim_library: Optional pre-loaded claim library dict
    
    Returns:
        Content with expanded/formatted citations
    """
    if not content:
        return content
    
    if format_style == "keep":
        return content
    
    if claim_library is None:
        claim_library = _load_claim_library()
    
    # Normalize [CLAIM_ID:CLM_XXX] and [CLAIM_ID: CLM_XXX] to [CLM_XXX]
    content = re.sub(r'\[CLAIM_ID:\s*([A-Z0-9_]+)\]', r'[\1]', content, flags=re.IGNORECASE)
    
    pattern = r'\[([A-Z0-9_]+)\]'
    citations = re.findall(pattern, content)
    
    if not citations:
        return content
    
    if format_style == "inline":
        # Citation markers are for internal tracking; content already conveys the claim
        result = content
        for claim_id in set(citations):
            result = result.replace(f"[{claim_id}]", "")
        
        result = re.sub(r' +', ' ', result)
        result = re.sub(r' \.', '.', result)
        result = re.sub(r' ,', ',', result)
        return result.strip()
    
    elif format_style == "footnote":
        result = content
        footnotes = []
        seen = {}
        counter = 1
        
        for claim_id in citations:
            if claim_id not in seen:
                seen[claim_id] = counter
                claim_text = claim_library.get(claim_id, f"Reference: {claim_id}")
                footnotes.append(f"[{counter}] {claim_text}")
                counter += 1
            result = result.replace(f"[{claim_id}]", f"[{seen[claim_id]}]", 1)
        
        if footnotes:
            result += "\n\n---\nReferences:\n" + "\n".join(footnotes)
        return result
    
    elif format_style == "remove":
        result = re.sub(pattern, '', content)
        result = re.sub(r' +', ' ', result)
        return result.strip()
    
    return content


def format_content_for_platform(
    content: Dict[str, Any],
    platform: str
) -> Dict[str, Any]:
    """
    Format content for a specific platform, expanding citations appropriately.
    
    This is the main entry point for preparing content for deployment.
    
    Args:
        content: Content dict with headline, body, cta, etc.
        platform: Target platform (linkedin, twitter, email, etc.)
    
    Returns:
        Formatted content dict ready for platform deployment
    """
    platform_formats = {
        'linkedin': 'inline',
        'twitter': 'inline',
        'x': 'inline',
        'facebook': 'inline',
        'instagram': 'inline',
        'email': 'footnote',
        'mailchimp': 'footnote',
        'sendgrid': 'footnote',
        'blog': 'footnote',
        'internal': 'keep',
        'preview': 'keep'
    }
    
    format_style = platform_formats.get(platform.lower(), 'inline')
    
    formatted = content.copy()
    text_fields = ['headline', 'body', 'cta', 'subject', 'text', 'content']
    
    for field in text_fields:
        if field in formatted and formatted[field]:
            formatted[field] = expand_claim_citations(
                formatted[field],
                format_style=format_style
            )
    
    formatted['_formatting'] = {
        'platform': platform,
        'format_style': format_style,
        'citations_expanded': True
    }
    
    logger.debug(
        f"Formatted content for {platform}",
        extra={
            "event": "content_formatted",
            "platform": platform,
            "format_style": format_style
        }
    )
    
    return formatted


def get_claim_details(claim_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get full details for a list of claim IDs.
    
    Useful for displaying claim information in UI or generating references.
    
    Args:
        claim_ids: List of claim IDs (e.g., ['CLM_003', 'CLM_006'])
    
    Returns:
        List of claim detail dicts
    """
    try:
        claim_path = Path(settings.CLAIM_LIBRARY_PATH)
        
        if claim_path.exists():
            with open(claim_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                claims_by_id = {c['id']: c for c in data.get('claims', [])}
                
                return [
                    claims_by_id.get(cid, {'id': cid, 'text': 'Unknown claim'})
                    for cid in claim_ids
                ]
    except Exception as e:
        logger.error(f"Failed to get claim details: {e}")
    
    return [{'id': cid, 'text': 'Unknown claim'} for cid in claim_ids]


def extract_citations_from_content(content: str) -> List[str]:
    """
    Extract all claim citation IDs from content.
    
    Args:
        content: Content text with [CLM_XXX] or [CLAIM_ID:CLM_XXX] citations
    
    Returns:
        List of unique claim IDs found
    """
    if not content:
        return []
    
    valid_claims = []
    
    pattern_prefixed = r'\[CLAIM_ID:\s*([A-Z0-9_]+)\]'
    matches = re.findall(pattern_prefixed, content, re.IGNORECASE)
    valid_claims.extend(matches)
    
    pattern_direct = r'\[([A-Z0-9_]+)\]'
    matches = re.findall(pattern_direct, content)
    for match in matches:
        if match.startswith('CLM_') and match not in valid_claims:
            valid_claims.append(match)
    
    return list(set(valid_claims))
