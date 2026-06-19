"""
Automation Layer Connectors

External platform API connectors for the Automation Layer (ACT phase).
Per Research Plan Section 3 and Section 6.3 - Deployer Agent Tools.
"""
from .base_connector import BaseConnector, PlatformResponse
from .email_api import EmailConnector
from .mailchimp_api import MailchimpConnector
from .blog_api import BlogConnector

__all__ = [
    'BaseConnector',
    'PlatformResponse',
    'EmailConnector',
    'MailchimpConnector',
    'BlogConnector',
]
