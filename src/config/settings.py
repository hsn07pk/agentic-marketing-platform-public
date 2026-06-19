"""
Settings module wrapper
Re-exports settings from the main config directory
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from actual settings location
from config.settings import (
    Settings,
    settings,
    DEBUG,
    DATABASE_URL,
    REDIS_URL,
)

__all__ = [
    'Settings',
    'settings',
    'DEBUG',
    'DATABASE_URL',
    'REDIS_URL',
]
