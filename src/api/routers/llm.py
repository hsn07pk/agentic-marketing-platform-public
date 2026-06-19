"""
LLM Management API Router
Provides endpoints for managing Ollama models and prompts.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import httpx
import yaml
import os
import re
import asyncio

from ...config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "../../../config/prompts")

_DEFAULT_PROMPTS: Dict[str, Dict[str, Any]] = {}

def _load_defaults():
    try:
        for fname in os.listdir(PROMPTS_DIR):
            if fname.endswith(('.yaml', '.yml')):
                filepath = os.path.join(PROMPTS_DIR, fname)
                with open(filepath, 'r') as f:
                    _DEFAULT_PROMPTS[fname] = yaml.safe_load(f)
        logger.info(f"Loaded {len(_DEFAULT_PROMPTS)} default prompt files for reset")
    except Exception as e:
        logger.warning(f"Could not load default prompts: {e}")

_load_defaults()


class ModelPullRequest(BaseModel):
    model_name: str = Field(..., description="Model name to pull (e.g., 'qwen3:8b', 'mistral:7b')")


class ModelDeleteRequest(BaseModel):
    model_name: str = Field(..., description="Model name to delete")


class ModelTestRequest(BaseModel):
    """Request to test a model with a prompt"""
    prompt: str = Field(..., description="Test prompt to send to the model")
    model: Optional[str] = Field(None, description="Model to use (defaults to configured model)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(500, ge=1, le=4000, description="Maximum tokens to generate")


class SetActiveModelRequest(BaseModel):
    """Request to set the active model"""
    model_name: str = Field(..., description="Model name to set as active")


class PromptUpdateRequest(BaseModel):
    """Request to update a prompt template"""
    content: str = Field(..., description="New prompt content")
    validate_first: bool = Field(True, description="Run validation before saving")


class PromptTestRequest(BaseModel):
    """Request to test a prompt template"""
    prompt_name: str = Field(..., description="Name of the prompt template")
    test_variables: Dict[str, str] = Field(default_factory=dict, description="Variables to substitute")
    model: Optional[str] = Field(None, description="Model to use for testing")


async def get_ollama_host() -> str:
    """Get the working Ollama host"""
    configured_host = settings.OLLAMA_HOST
    
    port_match = re.search(r':(\d+)$', configured_host.rstrip('/'))
    port = port_match.group(1) if port_match else "11434"
    
    hosts_to_try = [
        configured_host,
        f"http://localhost:{port}",
        f"http://host.docker.internal:{port}",
    ]
    
    timeout = 3.0
    for host in hosts_to_try:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{host}/api/tags")
                if response.status_code == 200:
                    return host
        except Exception:
            continue
    
    return configured_host  # Fallback to configured


def load_prompt_file(filename: str) -> Dict[str, Any]:
    """Load a prompt YAML file"""
    filepath = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filename}")
    
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_prompt_file(filename: str, content: Dict[str, Any]) -> None:
    """Save a prompt YAML file"""
    filepath = os.path.join(PROMPTS_DIR, filename)
    with open(filepath, 'w') as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


RESPONSE_PATTERNS = {
    "toxicity_score": r'\*{0,2}TOXICITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
    "factuality_score": r'\*{0,2}FACTUALITY_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
    "brand_score": r'\*{0,2}BRAND_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
    "compliance_score": r'\*{0,2}COMPLIANCE_SCORE\*{0,2}:?\*{0,2}\s*([0-9.]+)',
    "claim_citation": r'\[CLM_\d{3}\]',
    "headline": r'Headline:\s*(.+?)(?:\n|$)',
    "body": r'Body:\s*(.+?)(?:CTA:|$)',
}


@router.get("/patterns")
async def get_response_patterns() -> Dict[str, Any]:
    """
    Get the response parsing patterns used by the system.
    These patterns are used to extract scores and content from LLM responses.
    """
    return {
        "patterns": {
            "Toxicity Score": RESPONSE_PATTERNS["toxicity_score"],
            "Factuality Score": RESPONSE_PATTERNS["factuality_score"],
            "Brand Score": RESPONSE_PATTERNS["brand_score"],
            "Compliance Score": RESPONSE_PATTERNS["compliance_score"],
            "Claim Citation": RESPONSE_PATTERNS["claim_citation"],
            "Headline Extract": RESPONSE_PATTERNS["headline"],
            "Body Extract": RESPONSE_PATTERNS["body"],
        },
        "description": "Regex patterns that LLM responses must match for score extraction"
    }


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """
    List all available Ollama models with details.
    """
    try:
        ollama_host = await get_ollama_host()
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail="Failed to get models from Ollama")
            
            models = response.json().get("models", [])
            
            ps_response = await client.get(f"{ollama_host}/api/ps")
            running_models = []
            if ps_response.status_code == 200:
                running_models = [m.get("name") for m in ps_response.json().get("models", [])]
            
            model_list = []
            for m in models:
                model_info = {
                    "name": m.get("name", "unknown"),
                    "size_bytes": m.get("size", 0),
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                    "modified_at": m.get("modified_at", ""),
                    "digest": m.get("digest", "")[:12] if m.get("digest") else "",
                    "is_loaded": m.get("name") in running_models,
                    "is_active": m.get("name") == getattr(settings, 'OLLAMA_MODEL', 'qwen3:8b')
                }
                model_list.append(model_info)
            
            return {
                "models": model_list,
                "total": len(model_list),
                "loaded_count": len(running_models),
                "configured_model": getattr(settings, 'OLLAMA_MODEL', 'qwen3:8b'),
                "ollama_host": ollama_host
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/pull")
async def pull_model(request: ModelPullRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Pull/download a model from Ollama library.
    Returns immediately and downloads in background.
    """
    try:
        ollama_host = await get_ollama_host()
        
        async def do_pull():
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(
                        f"{ollama_host}/api/pull",
                        json={"name": request.model_name},
                        timeout=600.0  # 10 minute timeout for large models
                    )
                    logger.info(f"Model pull completed for {request.model_name}: {response.status_code}")
            except Exception as e:
                logger.error(f"Model pull failed for {request.model_name}: {e}")
        
        background_tasks.add_task(do_pull)
        
        return {
            "status": "started",
            "message": f"Download started for model '{request.model_name}'",
            "model": request.model_name
        }
        
    except Exception as e:
        logger.error(f"Failed to start model pull: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/pull/{model_name}/status")
async def get_pull_status(model_name: str) -> Dict[str, Any]:
    """
    Get the status of a model pull operation by checking if model exists.
    """
    try:
        ollama_host = await get_ollama_host()
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if model_name in model_names:
                    return {"status": "completed", "model": model_name, "exists": True}
                else:
                    return {"status": "in_progress", "model": model_name, "exists": False}
        
        return {"status": "unknown", "model": model_name}
        
    except Exception as e:
        logger.error(f"Failed to check pull status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_model(model_name: str) -> Dict[str, Any]:
    """
    Delete a model from Ollama.
    """
    try:
        ollama_host = await get_ollama_host()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{ollama_host}/api/delete",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                return {
                    "status": "deleted",
                    "message": f"Model '{model_name}' deleted successfully",
                    "model": model_name
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to delete model: {response.text}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/test")
async def test_model(request: ModelTestRequest) -> Dict[str, Any]:
    """
    Test a model with a sample prompt.
    """
    try:
        ollama_host = await get_ollama_host()
        model = request.model or getattr(settings, 'OLLAMA_MODEL', 'qwen3:8b')
        
        start_time = datetime.now()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": request.prompt,
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Model generation failed: {response.text}"
                )
            
            result = response.json()
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return {
                "model": model,
                "prompt": request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt,
                "response": result.get("response", ""),
                "elapsed_seconds": round(elapsed, 2),
                "tokens_generated": result.get("eval_count", 0),
                "tokens_per_second": round(result.get("eval_count", 0) / elapsed, 1) if elapsed > 0 else 0
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/models/active")
async def set_active_model(request: SetActiveModelRequest) -> Dict[str, Any]:
    """
    Set the active model for content generation.
    Updates the OLLAMA_MODEL configuration.
    """
    try:
        ollama_host = await get_ollama_host()
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if request.model_name not in model_names:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{request.model_name}' not found. Available: {', '.join(model_names)}"
                    )
        
        from sqlalchemy.orm import Session
        from src.data_layer.database.connection import sync_session_maker
        from src.data_layer.database.models import SystemConfiguration
        
        with sync_session_maker() as session:
            config = session.query(SystemConfiguration).filter(
                SystemConfiguration.key == "OLLAMA_MODEL"
            ).first()
            
            if config:
                old_value = config.value
                config.value = request.model_name
                config.updated_at = datetime.utcnow()
            else:
                from src.data_layer.database.models import ConfigurationCategory
                config = SystemConfiguration(
                    key="OLLAMA_MODEL",
                    value=request.model_name,
                    category=ConfigurationCategory.LLM,
                    description="Active Ollama model",
                    is_secret=False
                )
                old_value = None
                session.add(config)
            
            session.commit()
        
        # Invalidate the in-process config cache so subsequent reads
        # pick up the new OLLAMA_MODEL value immediately.
        try:
            from src.config.configuration_service import refresh_runtime_config_cache
            refresh_runtime_config_cache()
        except Exception as cache_err:
            logger.warning(f"Could not refresh config cache: {cache_err}")
        
        # Pre-warm: send a minimal generate request so Ollama loads the
        # model into VRAM immediately instead of on the first real request.
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post(
                    f"{ollama_host}/api/generate",
                    json={
                        "model": request.model_name,
                        "prompt": "hi",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
                logger.info(f"Pre-warmed model '{request.model_name}' into VRAM")
        except Exception as warm_err:
            logger.warning(f"Model pre-warm failed (non-critical): {warm_err}")
        
        return {
            "status": "updated",
            "message": f"Active model changed to '{request.model_name}'",
            "model": request.model_name,
            "previous_model": old_value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set active model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts")
async def list_prompts() -> Dict[str, Any]:
    """
    List all prompt templates with their metadata.
    """
    try:
        prompt_files = {
            "content_generation.yaml": "Content Generation",
            "safety_judge.yaml": "Safety Validation",
            "claim_library.yaml": "Claim Library"
        }
        
        prompts = []
        
        for filename, category in prompt_files.items():
            try:
                filepath = os.path.join(PROMPTS_DIR, filename)
                if os.path.exists(filepath):
                    data = load_prompt_file(filename)
                    
                    if filename == "content_generation.yaml":
                        templates = [k for k in data.keys() if k not in ['version', 'last_updated', 'system_instructions', 'validation_rules']]
                    elif filename == "safety_judge.yaml":
                        templates = [k for k in data.keys() if k not in ['version', 'last_updated', 'final_assessment']]
                    else:
                        templates = list(data.keys()) if isinstance(data, dict) else []
                    
                    prompts.append({
                        "file": filename,
                        "category": category,
                        "version": data.get("version", "1.0.0"),
                        "last_updated": data.get("last_updated", ""),
                        "templates": templates,
                        "template_count": len(templates)
                    })
            except Exception as e:
                logger.warning(f"Failed to load prompt file {filename}: {e}")
        
        return {
            "prompts": prompts,
            "total_files": len(prompts),
            "prompts_dir": PROMPTS_DIR
        }
        
    except Exception as e:
        logger.error(f"Failed to list prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts/{filename}")
async def get_prompt(filename: str) -> Dict[str, Any]:
    """
    Get a specific prompt file's content.
    """
    try:
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        data = load_prompt_file(filename)
        
        return {
            "file": filename,
            "content": data,
            "raw_yaml": yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prompt file not found: {filename}")
    except Exception as e:
        logger.error(f"Failed to get prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts/{filename}/{template_name}")
async def get_prompt_template(filename: str, template_name: str) -> Dict[str, Any]:
    """
    Get a specific template from a prompt file.
    """
    try:
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        data = load_prompt_file(filename)
        
        if template_name not in data:
            raise HTTPException(
                status_code=404,
                detail=f"Template '{template_name}' not found in {filename}"
            )
        
        template = data[template_name]
        
        variables = []
        if isinstance(template, dict):
            template_str = str(template)
            variables = re.findall(r'\{(\w+)\}', template_str)
            variables = list(set(variables))
        
        return {
            "file": filename,
            "template_name": template_name,
            "template": template,
            "variables": variables,
            "raw_yaml": yaml.dump({template_name: template}, default_flow_style=False, allow_unicode=True)
        }
        
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prompt file not found: {filename}")
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/prompts/{filename}/{template_name}")
async def update_prompt_template(
    filename: str,
    template_name: str,
    request: PromptUpdateRequest
) -> Dict[str, Any]:
    """
    Update a specific template in a prompt file.
    Validates the new content before saving.
    
    ⚠️ IMPORTANT: Direct saves are BLOCKED for production safety.
    Use POST /prompts/{filename}/{template_name}/deploy for validated deployment.
    """
    raise HTTPException(
        status_code=403,
        detail={
            "message": "Direct prompt updates are blocked for production safety",
            "action_required": "Use the /deploy endpoint for validated deployment",
            "validate_endpoint": f"POST /api/v1/llm/prompts/{filename}/{template_name}/validate",
            "deploy_endpoint": f"POST /api/v1/llm/prompts/{filename}/{template_name}/deploy",
            "reason": "All prompt changes must pass comprehensive validation before deployment"
        }
    )


@router.post("/prompts/test")
async def test_prompt(request: PromptTestRequest) -> Dict[str, Any]:
    """
    Test a prompt template with sample variables.
    """
    try:
        prompt_files = ["content_generation.yaml", "safety_judge.yaml"]
        template_data = None
        source_file = None
        
        for filename in prompt_files:
            try:
                data = load_prompt_file(filename)
                if request.prompt_name in data:
                    template_data = data[request.prompt_name]
                    source_file = filename
                    break
            except Exception:
                continue
        
        if not template_data:
            raise HTTPException(
                status_code=404,
                detail=f"Template '{request.prompt_name}' not found"
            )
        
        if isinstance(template_data, dict):
            system_prompt = template_data.get("system", "")
            user_template = template_data.get("user_template", template_data.get("user", ""))
        else:
            user_template = str(template_data)
            system_prompt = ""
        
        for var, value in request.test_variables.items():
            user_template = user_template.replace(f"{{{var}}}", value)
            system_prompt = system_prompt.replace(f"{{{var}}}", value)
        
        full_prompt = f"{system_prompt}\n\n{user_template}" if system_prompt else user_template
        
        ollama_host = await get_ollama_host()
        model = request.model or getattr(settings, 'OLLAMA_MODEL', 'qwen3:8b')
        
        start_time = datetime.now()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "temperature": 0.7,
                    "num_predict": 1000,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Generation failed: {response.text}"
                )
            
            result = response.json()
            elapsed = (datetime.now() - start_time).total_seconds()
        
        response_text = result.get("response", "")
        
        validation_results = {}
        for pattern_name, pattern in RESPONSE_PATTERNS.items():
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            validation_results[pattern_name] = {
                "found": len(matches) > 0,
                "matches": matches[:3] if matches else []
            }
        
        return {
            "source_file": source_file,
            "template_name": request.prompt_name,
            "model": model,
            "prompt_preview": full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt,
            "response": response_text,
            "elapsed_seconds": round(elapsed, 2),
            "validation_results": validation_results,
            "variables_used": list(request.test_variables.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/{filename}/{template_name}/reset")
async def reset_prompt_template(filename: str, template_name: str) -> Dict[str, Any]:
    """
    Reset a template to its default value (loaded at application startup).
    """
    filepath = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Prompt file not found: {filename}")

    if filename not in _DEFAULT_PROMPTS:
        raise HTTPException(
            status_code=500,
            detail=f"No default stored for {filename}. Restart the API to reload defaults."
        )

    default_content = _DEFAULT_PROMPTS[filename]

    default_template = None
    for section in ["templates", "rubrics", "claims"]:
        if section in default_content and template_name in default_content[section]:
            default_template = default_content[section][template_name]
            break

    if default_template is None and template_name in default_content:
        default_template = default_content[template_name]

    if default_template is None:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{template_name}' not found in default version of {filename}"
        )

    try:
        current_content = load_prompt_file(filename)

        replaced = False
        for section in ["templates", "rubrics", "claims"]:
            if section in current_content and template_name in current_content[section]:
                current_content[section][template_name] = default_template
                replaced = True
                break

        if not replaced and template_name in current_content:
            current_content[template_name] = default_template
            replaced = True

        if not replaced:
            raise HTTPException(
                status_code=404,
                detail=f"Template '{template_name}' not found in current {filename}"
            )

        save_prompt_file(filename, current_content)

        return {
            "success": True,
            "message": f"Template '{template_name}' in '{filename}' reset to default",
            "template_name": template_name,
            "filename": filename
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset template error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class PromptValidationRequest(BaseModel):
    """Request for comprehensive prompt validation before deployment"""
    prompt_content: str = Field(..., description="The new prompt YAML content")
    model: Optional[str] = Field(None, description="Model to use for testing")


@router.post("/prompts/{filename}/{template_name}/validate")
async def validate_prompt_for_deployment(
    filename: str,
    template_name: str,
    request: PromptValidationRequest
) -> Dict[str, Any]:
    """
    Run comprehensive validation before allowing prompt deployment.
    
    This endpoint:
    1. Runs multiple test scenarios with the new prompt
    2. Validates all regex patterns extract correctly
    3. Runs integration tests (content + safety validation)
    4. Requires 100% pass rate before allowing deployment
    
    NO changes are saved - this only tests the prompt.
    """
    try:
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        from src.ai_layer.testing.prompt_testing_service import PromptTestingService
        
        testing_service = PromptTestingService()
        
        results = await testing_service.run_full_workflow_test(
            prompt_file=filename,
            template_name=template_name,
            new_prompt_content=request.prompt_content,
            model=request.model
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Prompt validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/{filename}/{template_name}/deploy")
async def deploy_validated_prompt(
    filename: str,
    template_name: str,
    request: PromptValidationRequest
) -> Dict[str, Any]:
    """
    Deploy a prompt after comprehensive validation.
    
    This endpoint:
    1. First runs full validation (same as /validate)
    2. Only if ALL tests pass (100%), saves the prompt
    3. Returns detailed results showing why deployment was blocked (if any)
    
    This is the ONLY way to update production prompts.
    """
    try:
        if not filename.endswith('.yaml'):
            filename += '.yaml'
        
        from src.ai_layer.testing.prompt_testing_service import PromptTestingService
        
        testing_service = PromptTestingService()
        
        results = await testing_service.run_full_workflow_test(
            prompt_file=filename,
            template_name=template_name,
            new_prompt_content=request.prompt_content,
            model=request.model
        )
        
        if not results.get("can_deploy"):
            return {
                "deployed": False,
                "message": "Deployment blocked - validation failed",
                "validation_results": results,
                "blocking_issues": results.get("blocking_issues", [])
            }
        
        try:
            data = load_prompt_file(filename)
            
            if template_name not in data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Template '{template_name}' not found in {filename}"
                )
            
            if isinstance(data[template_name], dict):
                data[template_name]["user_template"] = request.prompt_content
            else:
                try:
                    new_template = yaml.safe_load(request.prompt_content)
                    if isinstance(new_template, dict) and template_name in new_template:
                        data[template_name] = new_template[template_name]
                    elif isinstance(new_template, dict):
                        data[template_name] = new_template
                    else:
                        if not isinstance(data[template_name], dict):
                            data[template_name] = {"user_template": request.prompt_content}
                        else:
                            data[template_name]["user_template"] = request.prompt_content
                except yaml.YAMLError:
                    if not isinstance(data[template_name], dict):
                        data[template_name] = {"user_template": request.prompt_content}
                    else:
                        data[template_name]["user_template"] = request.prompt_content
            
            data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
            save_prompt_file(filename, data)
            
            return {
                "deployed": True,
                "message": f"Prompt '{template_name}' deployed successfully after passing all {len(results.get('phases', {}))} validation phases",
                "validation_results": results,
                "file": filename,
                "template_name": template_name
            }
            
        except Exception as save_error:
            logger.error(f"Failed to save prompt after validation: {save_error}")
            return {
                "deployed": False,
                "message": f"Validation passed but save failed: {str(save_error)}",
                "validation_results": results
            }
        
    except Exception as e:
        logger.error(f"Prompt deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/popular")
async def get_popular_models() -> Dict[str, Any]:
    """
    Get a list of popular Ollama models that can be downloaded.
    """
    popular_models = [
        {
            "name": "qwen3:8b",
            "description": "Meta's Llama 3 8B - General purpose, good balance of speed and quality",
            "size_gb": 4.7,
            "parameters": "8B",
            "use_case": "General content generation"
        },
        {
            "name": "llama3:70b",
            "description": "Meta's Llama 3 70B - High quality, requires significant VRAM",
            "size_gb": 40,
            "parameters": "70B",
            "use_case": "High-quality content, complex tasks"
        },
        {
            "name": "qwen3:8b",
            "description": "Alibaba's Qwen 3 8B - Fast, multilingual support",
            "size_gb": 4.7,
            "parameters": "8B",
            "use_case": "Fast generation, multilingual"
        },
        {
            "name": "mistral:7b",
            "description": "Mistral 7B - Efficient, good for quick tasks",
            "size_gb": 4.1,
            "parameters": "7B",
            "use_case": "Quick drafts, validation"
        },
        {
            "name": "mixtral:8x7b",
            "description": "Mixtral 8x7B MoE - High quality mixture of experts",
            "size_gb": 26,
            "parameters": "46.7B (8x7B MoE)",
            "use_case": "Complex reasoning, long content"
        },
        {
            "name": "codellama:7b",
            "description": "Code Llama 7B - Specialized for code generation",
            "size_gb": 3.8,
            "parameters": "7B",
            "use_case": "Code generation, technical content"
        },
        {
            "name": "phi3:mini",
            "description": "Microsoft Phi-3 Mini - Small but capable",
            "size_gb": 2.3,
            "parameters": "3.8B",
            "use_case": "Light tasks, low resource environments"
        },
        {
            "name": "gemma:7b",
            "description": "Google Gemma 7B - Google's open model",
            "size_gb": 5.0,
            "parameters": "7B",
            "use_case": "General purpose"
        }
    ]
    
    return {
        "models": popular_models,
        "note": "Sizes are approximate. Actual download may vary."
    }
