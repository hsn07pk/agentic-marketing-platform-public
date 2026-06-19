"""
Platform Simulation Components

Per Research Plan Section 5.2 - Platform Simulations
Provides realistic simulations for LinkedIn, Twitter, Email, and Blog marketing platforms.
"""
from .base_platform import BasePlatform
from .linkedin_platform import LinkedInPlatform
from .email_platform import EmailPlatform
from .blog_platform import BlogPlatform

# Twitter platform may exist as twitter_platform.py or x_platform.py
try:
    from .twitter_platform import TwitterPlatform
except ImportError:
    try:
        from .x_platform import XPlatform as TwitterPlatform
    except ImportError:
        TwitterPlatform = None

__all__ = [
    'BasePlatform',
    'LinkedInPlatform',
    'EmailPlatform',
    'BlogPlatform',
    'TwitterPlatform'
]
