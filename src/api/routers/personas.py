"""
Persona management API endpoints
Provides dynamic access to available personas from config/personas/*.yaml
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging
from pathlib import Path

from ...simulation.agents.persona_factory import PersonaFactory
from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize persona factory
persona_factory = PersonaFactory()

@router.get("/list", response_model=List[str])
async def list_available_personas():
    """
    List all available persona IDs from config/personas/*.yaml

    This endpoint dynamically scans the personas directory and returns
    all available personas, including any custom personas added by users.

    Returns:
        List of persona IDs (e.g., ["decision_maker", "practitioner", "researcher"])
    """
    try:
        personas = persona_factory.list_available_personas()
        logger.info(f"Listed {len(personas)} available personas")
        return personas
    except Exception as e:
        logger.error(f"Failed to list personas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list personas: {str(e)}")


@router.get("/{persona_id}", response_model=Dict[str, Any])
async def get_persona_details(persona_id: str):
    """
    Get detailed configuration for a specific persona

    Args:
        persona_id: Persona identifier (e.g., "decision_maker")

    Returns:
        Full persona configuration including demographics, behavior, content preferences
    """
    try:
        config = persona_factory.get_persona_config(persona_id)
        return config
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get persona {persona_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get persona: {str(e)}")


@router.get("/{persona_id}/summary")
async def get_persona_summary(persona_id: str):
    """
    Get human-readable summary of a persona

    Args:
        persona_id: Persona identifier

    Returns:
        Human-readable text summary with key metrics and platform activity
    """
    try:
        summary = persona_factory.get_persona_summary(persona_id)
        return {"persona_id": persona_id, "summary": summary}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get persona summary {persona_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.get("/{persona_id}/preferences")
async def get_persona_preferences(persona_id: str):
    """
    Get content preferences for a persona

    Args:
        persona_id: Persona identifier

    Returns:
        Content preferences including formality, tone, required citations, etc.
    """
    try:
        preferences = persona_factory.get_persona_preferences(persona_id)
        if not preferences:
            raise HTTPException(status_code=404, detail=f"Persona {persona_id} not found")
        return {"persona_id": persona_id, "preferences": preferences}
    except Exception as e:
        logger.error(f"Failed to get preferences for {persona_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")


@router.get("/{persona_id}/platform/{platform}")
async def get_persona_platform_behavior(persona_id: str, platform: str):
    """
    Get platform-specific behavior for a persona

    Args:
        persona_id: Persona identifier
        platform: Platform name (e.g., "linkedin", "x", "email")

    Returns:
        Platform-specific configuration (length, structure, CTAs, etc.)
    """
    try:
        behavior = persona_factory.get_platform_behavior(persona_id, platform)
        if not behavior:
            raise HTTPException(
                status_code=404,
                detail=f"Platform '{platform}' not found for persona '{persona_id}'"
            )
        return {
            "persona_id": persona_id,
            "platform": platform,
            "behavior": behavior
        }
    except Exception as e:
        logger.error(f"Failed to get platform behavior: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get platform behavior: {str(e)}")


@router.get("/{persona_id}/validate")
async def validate_persona(persona_id: str):
    """
    Validate that a persona configuration is complete and correct

    Args:
        persona_id: Persona identifier

    Returns:
        Validation result with status and any errors
    """
    try:
        is_valid = persona_factory.validate_persona_config(persona_id)
        return {
            "persona_id": persona_id,
            "valid": is_valid,
            "message": "Persona configuration is valid" if is_valid else "Persona configuration is invalid or incomplete"
        }
    except Exception as e:
        logger.error(f"Failed to validate persona {persona_id}: {e}", exc_info=True)
        return {
            "persona_id": persona_id,
            "valid": False,
            "message": f"Validation error: {str(e)}"
        }


@router.post("/reload")
async def reload_personas():
    """
    Reload all personas from disk

    This endpoint forces a rescan of the config/personas/ directory,
    useful after adding new persona YAML files without restarting the server.

    Returns:
        List of loaded persona IDs
    """
    try:
        global persona_factory
        persona_factory = PersonaFactory()
        personas = persona_factory.list_available_personas()
        logger.info(f"Reloaded {len(personas)} personas from disk")
        return {
            "status": "success",
            "message": f"Reloaded {len(personas)} personas",
            "personas": personas
        }
    except Exception as e:
        logger.error(f"Failed to reload personas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload personas: {str(e)}")
