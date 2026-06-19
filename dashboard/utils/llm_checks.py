"""
LLM readiness checks used across dashboard pages.

Provides a single function that queries the API and returns a structured
diagnostic result so every page can display consistent warnings/guidance.
"""
from typing import Dict, Any, Optional
import streamlit as st


def check_llm_readiness(api) -> Dict[str, Any]:
    """Check whether a valid LLM model is installed and active.

    Strategy: Try the /llm/models endpoint first (the source of truth for
    model availability).  This implicitly proves Ollama is reachable because
    the API server queries Ollama to build the model list.  Only fall back to
    /health/detailed for diagnostics when /llm/models itself fails.

    Returns a dict with:
      - ready (bool): True if the configured model is actually installed.
      - ollama_available (bool): True if Ollama is reachable.
      - configured_model (str|None): Model name in config.
      - installed_models (list[str]): Models currently downloaded.
      - active_model_installed (bool): configured model ∈ installed list.
      - message (str): Human-readable summary.
      - fix_steps (list[str]): Ordered steps to resolve the issue.
    """
    result: Dict[str, Any] = {
        "ready": False,
        "ollama_available": False,
        "configured_model": None,
        "installed_models": [],
        "active_model_installed": False,
        "message": "",
        "fix_steps": [],
    }

    # 1. Try the models endpoint directly — this is the source of truth
    #    Uses direct_get_json to bypass the resilient session's circuit
    #    breaker so that a transient timeout on an unrelated page never
    #    causes a false "Ollama unreachable" error here.
    models_data = api.direct_get_json("/llm/models")

    if models_data and models_data.get("models") is not None:
        # Models endpoint responded → Ollama is reachable
        result["ollama_available"] = True
        result["installed_models"] = [
            m.get("name") for m in models_data.get("models", [])
        ]
        result["configured_model"] = models_data.get("configured_model")
    else:
        # Models endpoint failed — use health endpoint for diagnostics
        health = api.direct_get_json("/health/detailed")
        ollama_info = (health or {}).get("components", {}).get("ollama", {})
        result["ollama_available"] = ollama_info.get("status") == "healthy"

        if not result["ollama_available"]:
            result["message"] = "Ollama is not running or unreachable."
            result["fix_steps"] = [
                "Ensure the Ollama service is running on the server.",
                "Check OLLAMA_HOST in ⚙️ Operations → System Settings → llm.",
            ]
            return result

        # Ollama healthy but models endpoint failed
        result["message"] = "Could not retrieve model list from the API."
        result["fix_steps"] = [
            "Check the API server logs for errors.",
            "Try refreshing the page.",
        ]
        return result

    configured = result["configured_model"]
    installed = result["installed_models"]

    # 2. No models installed at all
    if not installed:
        result["message"] = "No LLM models are installed."
        result["fix_steps"] = [
            "Go to 🤖 LLM Management → Model Management.",
            "Download a model (e.g. llama3:8b or mistral:7b).",
            "Click 'Set Active' on the downloaded model.",
        ]
        return result

    # 3. Configured model not among installed models
    if configured and configured not in installed:
        result["active_model_installed"] = False
        result["message"] = (
            f"The configured model '{configured}' is not installed. "
            f"Installed models: {', '.join(installed)}."
        )
        result["fix_steps"] = [
            "Go to 🤖 LLM Management → Model Management.",
            f"Either download '{configured}' or click 'Set Active' on one of the installed models ({', '.join(installed)}).",
            "After setting an active model, restart any failed campaigns.",
        ]
        return result

    # 4. No configured model (empty / None)
    if not configured:
        result["active_model_installed"] = False
        result["message"] = "No active LLM model is configured."
        result["fix_steps"] = [
            "Go to 🤖 LLM Management → Model Management.",
            f"Click 'Set Active' on one of the installed models ({', '.join(installed)}).",
        ]
        return result

    # All good
    result["ready"] = True
    result["active_model_installed"] = True
    result["message"] = f"LLM ready — active model: {configured}"
    return result


def render_llm_status_banner(api, *, context: str = "general") -> Dict[str, Any]:
    """Render an inline Streamlit banner if the LLM is not ready.

    Args:
        api: AgenticAPIClient instance.
        context: 'campaign_start' | 'workflow_progress' | 'general'
                 — adjusts the wording slightly.

    Returns:
        The readiness dict (callers can check result['ready']).
    """
    status = check_llm_readiness(api)

    if status["ready"]:
        return status

    # Build the warning block
    if context == "campaign_start":
        header = "⚠️ Cannot Start Campaign — No Active LLM Model"
    elif context == "workflow_progress":
        header = "⚠️ LLM Model Issue Detected — Content Generation Will Fail"
    else:
        header = "⚠️ No Active LLM Model — Action Required"

    steps_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(status["fix_steps"]))

    st.error(
        f"**{header}**\n\n"
        f"{status['message']}\n\n"
        f"**How to fix:**\n{steps_md}"
    )

    return status
