"""
Governance module for content safety, validation, and formatting.
"""
from .content_formatter import (
    expand_claim_citations,
    format_content_for_platform,
    extract_citations_from_content,
    get_claim_details
)

__all__ = [
    'expand_claim_citations',
    'format_content_for_platform',
    'extract_citations_from_content',
    'get_claim_details'
]
