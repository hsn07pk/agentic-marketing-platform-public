from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import httpx
import redis
import logging
import os

from src.data_layer.database.connection import get_db
from src.data_layer.database.models import ConfigurationCategory
from src.config.configuration_service import ConfigurationService, get_configuration_service
from src.config.encryption import mask_sensitive_value
from src.shared.constants import get_all_constants_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["Configuration"])


class ConfigurationValue(BaseModel):
    value: str = Field(..., description="New value for the configuration")


class BulkConfigurationUpdate(BaseModel):
    configurations: Dict[str, str] = Field(..., description="Key-value pairs to update")


class ConnectionTestRequest(BaseModel):
    service: str = Field(..., description="Service to test: database, redis, ollama, openai")
    connection_string: Optional[str] = Field(None, description="Optional custom connection string to test")


class ConfigurationResponse(BaseModel):
    key: str
    display_value: str
    category: str
    is_secret: bool
    description: Optional[str]
    default_value: Optional[str]
    value_type: str
    updated_at: Optional[str]


class CategoryResponse(BaseModel):
    category: str
    display_name: str
    total_settings: int
    configured_count: int


class StatusResponse(BaseModel):
    is_configured: bool
    total_settings: int
    configured_count: int
    categories: List[CategoryResponse]


@router.get("/constants")
async def get_all_constants():
    """
    Get all shared constants.
    
    This endpoint provides the SINGLE SOURCE OF TRUTH for all constants
    that are used by both backend and frontend. Frontend should fetch
    these on startup rather than maintaining duplicate definitions.
    
    Returns:
        Dictionary containing all platform, status, workflow, threshold,
        and display constants.
    """
    return get_all_constants_dict()


@router.get("/status", response_model=StatusResponse)
async def get_configuration_status(db: Session = Depends(get_db)):
    """
    Get overall configuration status.
    Used to determine if setup wizard should be shown.
    """
    service = get_configuration_service(db)
    return service.get_status()


@router.get("/categories", response_model=List[CategoryResponse])
async def get_all_categories(db: Session = Depends(get_db)):
    """
    Get list of all configuration categories with counts.
    """
    service = get_configuration_service(db)
    return service.get_all_categories()


@router.get("/category/{category}", response_model=List[ConfigurationResponse])
async def get_category_configurations(
    category: str,
    db: Session = Depends(get_db)
):
    """
    Get all configurations for a specific category.
    Secret values are masked in the response.
    """
    try:
        cat_enum = ConfigurationCategory(category)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category: {category}. Valid categories: {[c.value for c in ConfigurationCategory]}"
        )
    
    service = get_configuration_service(db)
    return service.get_by_category(cat_enum)


@router.get("/value/{key}")
async def get_configuration_value(
    key: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific configuration value.
    Secret values are masked.
    """
    service = get_configuration_service(db)
    value = service.get_value(key)
    
    from src.data_layer.database.models import SystemConfiguration
    config = db.query(SystemConfiguration).filter_by(key=key).first()
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration not found: {key}"
        )
    
    display_value = str(value) if value else ""
    if config.is_secret and value:
        display_value = mask_sensitive_value(str(value))
    
    return {
        "key": key,
        "value": value if not config.is_secret else None,
        "display_value": display_value,
        "is_secret": config.is_secret,
        "category": config.category.value,
    }


@router.put("/value/{key}")
async def update_configuration_value(
    key: str,
    update: ConfigurationValue,
    db: Session = Depends(get_db)
):
    """
    Update a configuration value.
    Values for secret fields will be encrypted.
    """
    service = get_configuration_service(db)
    success = service.set_value(key, update.value)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {key}"
        )
    
    return {"success": True, "key": key, "message": "Configuration updated"}


@router.post("/bulk-update")
async def bulk_update_configurations(
    updates: BulkConfigurationUpdate,
    db: Session = Depends(get_db)
):
    """
    Update multiple configurations at once.
    Used by the setup wizard.
    """
    service = get_configuration_service(db)
    results = service.bulk_update(updates.configurations)
    
    failed = [k for k, v in results.items() if not v]
    if failed:
        return {
            "success": False,
            "updated": [k for k, v in results.items() if v],
            "failed": failed,
            "message": f"Some configurations failed to update: {failed}"
        }
    
    return {
        "success": True,
        "updated": list(results.keys()),
        "message": f"Updated {len(results)} configurations"
    }


@router.post("/initialize")
async def initialize_configurations(db: Session = Depends(get_db)):
    """
    Initialize the configuration database with defaults.
    Called on first startup.
    """
    service = get_configuration_service(db)
    count = service.initialize_defaults()
    
    return {
        "success": True,
        "created": count,
        "message": f"Initialized {count} default configurations"
    }


@router.post("/test-connection")
async def test_connection(
    request: ConnectionTestRequest,
    db: Session = Depends(get_db)
):
    """
    Test connection to a service.
    Supports: database, redis, ollama, openai
    """
    service = get_configuration_service(db)
    
    if request.service == "database":
        return {"success": True, "service": "database", "message": "Database connection successful"}
    
    elif request.service == "redis":
        try:
            redis_url = request.connection_string or service.get_value("REDIS_URL", "redis://localhost:6379")
            # API runs in host network mode — map docker service names to localhost
            redis_url = redis_url.replace("redis://redis:", "redis://localhost:")
            r = redis.from_url(redis_url)
            r.ping()
            return {"success": True, "service": "redis", "message": "Redis connection successful"}
        except Exception as e:
            return {"success": False, "service": "redis", "message": f"Redis connection failed: {str(e)}"}
    
    elif request.service == "ollama":
        try:
            ollama_host = request.connection_string or service.get_value("OLLAMA_HOST", "http://localhost:11434")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return {
                        "success": True,
                        "service": "ollama",
                        "message": f"Ollama connected. Available models: {len(models)}",
                        "models": [m.get("name") for m in models[:10]]  # Return first 10
                    }
                else:
                    return {"success": False, "service": "ollama", "message": f"Ollama returned status {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "ollama", "message": f"Ollama connection failed: {str(e)}"}
    
    elif request.service == "openai":
        try:
            api_key = request.connection_string or service.get_value("OPENAI_API_KEY")
            if not api_key:
                return {"success": False, "service": "openai", "message": "OpenAI API key not configured"}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                if response.status_code == 200:
                    return {"success": True, "service": "openai", "message": "OpenAI API key is valid"}
                elif response.status_code == 401:
                    return {"success": False, "service": "openai", "message": "Invalid OpenAI API key"}
                else:
                    return {"success": False, "service": "openai", "message": f"OpenAI API returned status {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "openai", "message": f"OpenAI connection failed: {str(e)}"}
    
    elif request.service == "slack":
        try:
            webhook_url = service.get_value("SLACK_WEBHOOK_URL", "")
            if not webhook_url:
                return {"success": False, "service": "slack", "message": "SLACK_WEBHOOK_URL not configured"}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    webhook_url,
                    json={"text": "✅ Agentic AI — Slack integration test successful!"}
                )
                if response.status_code == 200:
                    return {"success": True, "service": "slack", "message": "Slack test message sent successfully"}
                else:
                    return {"success": False, "service": "slack", "message": f"Slack webhook returned HTTP {response.status_code}: {response.text[:200]}"}
        except Exception as e:
            return {"success": False, "service": "slack", "message": f"Slack connection failed: {str(e)}"}

    elif request.service == "mailgun":
        try:
            mailgun_key = service.get_value("MAILGUN_API_KEY", "")
            mailgun_domain = service.get_value("MAILGUN_DOMAIN", "")
            from_email = service.get_value("MAILGUN_FROM_EMAIL", f"mailgun@{mailgun_domain}" if mailgun_domain else "alerts@example.com")
            
            if not mailgun_key:
                return {"success": False, "service": "mailgun", "message": "MAILGUN_API_KEY not configured"}
            if not mailgun_domain:
                return {"success": False, "service": "mailgun", "message": "MAILGUN_DOMAIN not configured"}
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Step 1: Verify domain
                response = await client.get(
                    f"https://api.mailgun.net/v3/domains/{mailgun_domain}",
                    auth=("api", mailgun_key)
                )
                if response.status_code == 401:
                    return {"success": False, "service": "mailgun", "message": "Mailgun API key is invalid or unauthorized. Use the Private API Key from Mailgun dashboard (Settings → API Keys)."}
                if response.status_code != 200:
                    return {"success": False, "service": "mailgun", "message": f"Mailgun domain '{mailgun_domain}' not found (HTTP {response.status_code}). Check MAILGUN_DOMAIN in System Settings → Email."}
                
                domain_state = response.json().get("domain", {}).get("state", "unknown")
                
                # Step 2: Try sending test email
                response = await client.post(
                    f"https://api.mailgun.net/v3/{mailgun_domain}/messages",
                    auth=("api", mailgun_key),
                    data={
                        "from": f"Agentic AI <{from_email}>",
                        "to": from_email,
                        "subject": "✅ Agentic AI — Mailgun Integration Test",
                        "text": "This is a test email from Agentic AI Platform. Mailgun integration is working correctly."
                    }
                )
                if response.status_code == 200:
                    return {"success": True, "service": "mailgun", "message": f"Mailgun verified (domain: {domain_state}) and test email sent to {from_email}"}
                elif response.status_code == 401:
                    return {"success": False, "service": "mailgun", "message": f"Domain verified ({domain_state}) but sending is forbidden. Your API key lacks sending permissions — use the Private API Key (starts with 'key-') from Mailgun Settings → API Keys, not a restricted/domain key."}
                else:
                    return {"success": False, "service": "mailgun", "message": f"Domain verified ({domain_state}) but send failed (HTTP {response.status_code}): {response.text[:200]}"}
        except Exception as e:
            return {"success": False, "service": "mailgun", "message": f"Mailgun connection failed: {str(e)}"}

    elif request.service == "linkedin":
        try:
            token = service.get_value("LINKEDIN_ACCESS_TOKEN", "") or os.environ.get("LINKEDIN_ACCESS_TOKEN", "")
            if not token:
                return {"success": False, "service": "linkedin", "message": "LINKEDIN_ACCESS_TOKEN not configured"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://api.linkedin.com/v2/me", headers={"Authorization": f"Bearer {token}"})
                if response.status_code == 200:
                    profile = response.json()
                    name = f"{profile.get('localizedFirstName', '')} {profile.get('localizedLastName', '')}".strip()
                    return {"success": True, "service": "linkedin", "message": f"LinkedIn connected as {name}"}
                return {"success": False, "service": "linkedin", "message": f"LinkedIn returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "linkedin", "message": f"LinkedIn connection failed: {str(e)}"}

    elif request.service == "twitter":
        try:
            token = service.get_value("TWITTER_ACCESS_TOKEN", "") or os.environ.get("TWITTER_ACCESS_TOKEN", "")
            if not token:
                return {"success": False, "service": "twitter", "message": "TWITTER_ACCESS_TOKEN not configured"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://api.twitter.com/2/users/me", headers={"Authorization": f"Bearer {token}"})
                if response.status_code == 200:
                    username = response.json().get("data", {}).get("username", "N/A")
                    return {"success": True, "service": "twitter", "message": f"Twitter connected as @{username}"}
                return {"success": False, "service": "twitter", "message": f"Twitter returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "twitter", "message": f"Twitter connection failed: {str(e)}"}

    elif request.service == "wordpress":
        try:
            blog_url = service.get_value("BLOG_CMS_URL", "") or os.environ.get("BLOG_CMS_URL", "")
            if not blog_url:
                return {"success": False, "service": "wordpress", "message": "BLOG_CMS_URL not configured"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{blog_url}/wp-json/wp/v2/posts?per_page=1")
                if response.status_code == 200:
                    return {"success": True, "service": "wordpress", "message": f"WordPress connected at {blog_url}"}
                return {"success": False, "service": "wordpress", "message": f"WordPress returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "wordpress", "message": f"WordPress connection failed: {str(e)}"}

    elif request.service == "mailchimp":
        try:
            mc_key = service.get_value("MAILCHIMP_API_KEY", "") or os.environ.get("MAILCHIMP_API_KEY", "")
            if not mc_key:
                return {"success": False, "service": "mailchimp", "message": "MAILCHIMP_API_KEY not configured"}
            dc = mc_key.split("-")[-1] if "-" in mc_key else "us1"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"https://{dc}.api.mailchimp.com/3.0/", headers={"Authorization": f"Bearer {mc_key}"})
                if response.status_code == 200:
                    account = response.json().get("account_name", "N/A")
                    return {"success": True, "service": "mailchimp", "message": f"Mailchimp connected: {account}"}
                return {"success": False, "service": "mailchimp", "message": f"Mailchimp returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "mailchimp", "message": f"Mailchimp connection failed: {str(e)}"}

    elif request.service == "perspective":
        try:
            key = service.get_value("PERSPECTIVE_API_KEY", "") or os.environ.get("PERSPECTIVE_API_KEY", "")
            if not key:
                return {"success": False, "service": "perspective", "message": "PERSPECTIVE_API_KEY not configured"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={key}",
                    json={"comment": {"text": "hello"}, "languages": ["en"], "requestedAttributes": {"TOXICITY": {}}}
                )
                if response.status_code in (200, 400):
                    return {"success": True, "service": "perspective", "message": "Perspective API connected"}
                return {"success": False, "service": "perspective", "message": f"Perspective returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "perspective", "message": f"Perspective connection failed: {str(e)}"}

    elif request.service == "apify":
        try:
            token = service.get_value("APIFY_API_TOKEN", "") or os.environ.get("APIFY_API_TOKEN", "")
            if not token:
                return {"success": False, "service": "apify", "message": "APIFY_API_TOKEN not configured"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://api.apify.com/v2/users/me", params={"token": token})
                if response.status_code == 200:
                    username = response.json().get("data", {}).get("username", "N/A")
                    return {"success": True, "service": "apify", "message": f"Apify connected: {username}"}
                return {"success": False, "service": "apify", "message": f"Apify returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "apify", "message": f"Apify connection failed: {str(e)}"}

    elif request.service == "hubspot":
        try:
            key = service.get_value("HUBSPOT_API_KEY", "") or os.environ.get("HUBSPOT_API_KEY", "")
            if not key:
                return {"success": False, "service": "hubspot", "message": "HUBSPOT_API_KEY not configured"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://api.hubapi.com/crm/v3/objects/contacts?limit=1", headers={"Authorization": f"Bearer {key}"})
                if response.status_code == 200:
                    return {"success": True, "service": "hubspot", "message": "HubSpot CRM connected"}
                return {"success": False, "service": "hubspot", "message": f"HubSpot returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "hubspot", "message": f"HubSpot connection failed: {str(e)}"}

    elif request.service == "calcom":
        try:
            key = service.get_value("CALENDAR_API_KEY", "") or os.environ.get("CALENDAR_API_KEY", "")
            if not key:
                return {"success": False, "service": "calcom", "message": "CALENDAR_API_KEY not configured"}
            api_url = service.get_value("CALENDAR_API_URL", "https://api.cal.com/v1")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{api_url}/event-types", params={"apiKey": key})
                if response.status_code == 200:
                    types = response.json().get("event_types", [])
                    return {"success": True, "service": "calcom", "message": f"Cal.com connected ({len(types)} event types)"}
                body = response.json() if 'json' in response.headers.get('content-type', '') else {}
                err = body.get('message') or body.get('error') or f"HTTP {response.status_code}"
                return {"success": False, "service": "calcom", "message": f"Cal.com: {err}"}
        except Exception as e:
            return {"success": False, "service": "calcom", "message": f"Cal.com connection failed: {str(e)}"}

    elif request.service == "sendgrid":
        try:
            key = service.get_value("SENDGRID_API_KEY", "") or os.environ.get("SENDGRID_API_KEY", "")
            if not key:
                return {"success": False, "service": "sendgrid", "message": "SENDGRID_API_KEY not configured"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://api.sendgrid.com/v3/user/profile", headers={"Authorization": f"Bearer {key}"})
                if response.status_code == 200:
                    return {"success": True, "service": "sendgrid", "message": "SendGrid connected"}
                return {"success": False, "service": "sendgrid", "message": f"SendGrid returned HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "service": "sendgrid", "message": f"SendGrid connection failed: {str(e)}"}

    else:
        valid = "database, redis, ollama, openai, slack, mailgun, linkedin, twitter, wordpress, mailchimp, perspective, apify, hubspot, calcom, sendgrid"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown service: {request.service}. Valid services: {valid}"
        )


@router.get("/export")
async def export_configurations(
    db: Session = Depends(get_db)
):
    """
    Export all configurations (secrets are excluded).
    Useful for backup or debugging.
    """
    service = get_configuration_service(db)
    configs = service.export_to_dict(include_secrets=False)
    
    return {
        "configurations": configs,
        "total": len(configs),
        "note": "Secret values are excluded from export"
    }


@router.get("/mock-mode")
async def get_mock_mode_status(db: Session = Depends(get_db)):
    """
    Get current mock mode status.
    Used by dashboard to show mock mode indicator and filter KPIs accordingly.
    """
    service = get_configuration_service(db)
    
    mock_mode_enabled = service.get_value("MOCK_MODE_ENABLED", "True")
    enable_mock_deployment = service.get_value("ENABLE_MOCK_DEPLOYMENT", "True")
    enable_mock_experiments = service.get_value("ENABLE_MOCK_EXPERIMENTS", "True")
    
    def to_bool(val):
        return str(val).lower() in ('true', '1', 'yes', 'on')
    
    mock_enabled = to_bool(mock_mode_enabled)
    include_mock_in_metrics = to_bool(service.get_value("INCLUDE_MOCK_IN_METRICS", "True"))
    
    return {
        "mock_mode_enabled": mock_enabled,
        "include_mock_in_metrics": include_mock_in_metrics,
        "settings": {
            "MOCK_MODE_ENABLED": to_bool(mock_mode_enabled),
            "ENABLE_MOCK_DEPLOYMENT": to_bool(enable_mock_deployment),
            "ENABLE_MOCK_EXPERIMENTS": to_bool(enable_mock_experiments),
            "INCLUDE_MOCK_IN_METRICS": include_mock_in_metrics
        },
        "description": "When mock mode is ON, test data and simulated metrics are used. When OFF, only real authenticated data is shown.",
        "kpi_filter_hint": "When INCLUDE_MOCK_IN_METRICS is ON, mock campaign data is included in all KPIs (clearly labeled). When OFF, only production data is shown."
    }


@router.get("/integrations/status")
async def get_integrations_status(db: Session = Depends(get_db)):
    """
    Check connectivity status of all external integrations.

    Research Plan Reference: Section 5.3 - External service connectivity

    Returns status for:
    - Cal.com (Calendar booking integration)
    - HubSpot (CRM integration)
    - Ollama (Local LLM backend)
    - OpenAI (Cloud LLM backend)
    """
    service = get_configuration_service(db)

    results = {
        "calcom": {
            "connected": False,
            "configured": False,
            "error": None,
            "details": {}
        },
        "hubspot": {
            "connected": False,
            "configured": False,
            "error": None,
            "details": {}
        },
        "ollama": {
            "connected": False,
            "configured": False,
            "error": None,
            "details": {}
        },
        "openai": {
            "connected": False,
            "configured": False,
            "error": None,
            "details": {}
        },
        "slack": {
            "connected": False,
            "configured": False,
            "error": None,
            "details": {}
        },
        "sendgrid": {
            "connected": False,
            "configured": False,
            "error": None,
            "details": {}
        }
    }

    try:
        calcom_api_key = service.get_value("CALCOM_API_KEY", "") or service.get_value("CALENDAR_API_KEY", "")
        if not calcom_api_key:
            calcom_api_key = os.environ.get("CALENDAR_API_KEY", "")

        results["calcom"]["configured"] = bool(calcom_api_key)

        if calcom_api_key:
            from ...automation_layer.connectors.calendar_api import CalendarAPIConnector
            connector = CalendarAPIConnector()
            connector.api_key = calcom_api_key
            result = await connector.get_event_types()
            results["calcom"]["connected"] = result.success
            if result.success:
                results["calcom"]["details"] = {
                    "event_types_count": len(result.data.get("event_types", []))
                }
            else:
                results["calcom"]["error"] = result.error
        else:
            results["calcom"]["error"] = "API key not configured. Set CALENDAR_API_KEY in environment or Operations → System Settings."
    except Exception as e:
        results["calcom"]["error"] = str(e)

    try:
        hubspot_api_key = service.get_value("HUBSPOT_API_KEY", "")

        results["hubspot"]["configured"] = bool(hubspot_api_key)

        if hubspot_api_key:
            from ...automation_layer.connectors.hubspot_api import HubSpotAPIConnector
            connector = HubSpotAPIConnector()
            result = await connector.count_contacts()
            results["hubspot"]["connected"] = result.success
            if result.success:
                results["hubspot"]["details"] = {
                    "contacts_count": result.data.get("count", 0)
                }
            else:
                results["hubspot"]["error"] = result.error
        else:
            results["hubspot"]["error"] = "API key not configured"
    except Exception as e:
        results["hubspot"]["error"] = str(e)

    try:
        ollama_host = service.get_value("OLLAMA_HOST", "http://localhost:11434")
        use_ollama = service.get_value("USE_LOCAL_LLM", "True")  # Fixed key

        results["ollama"]["configured"] = str(use_ollama).lower() in ('true', '1', 'yes')

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    results["ollama"]["connected"] = True
                    results["ollama"]["details"] = {
                        "available_models": len(models),
                        "models": [m.get("name") for m in models[:5]]
                    }
                else:
                    results["ollama"]["error"] = f"HTTP {response.status_code}"
            except httpx.ConnectError:
                results["ollama"]["error"] = "Connection refused - Ollama not running"
            except httpx.TimeoutException:
                results["ollama"]["error"] = "Connection timeout"
    except Exception as e:
        results["ollama"]["error"] = str(e)

    try:
        openai_api_key = service.get_value("OPENAI_API_KEY", "")

        results["openai"]["configured"] = bool(openai_api_key)

        if openai_api_key:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {openai_api_key}"}
                )
                if response.status_code == 200:
                    results["openai"]["connected"] = True
                    models_data = response.json().get("data", [])
                    gpt4_models = [m["id"] for m in models_data if "gpt-4" in m["id"]][:5]
                    results["openai"]["details"] = {
                        "total_models": len(models_data),
                        "gpt4_models": gpt4_models
                    }
                elif response.status_code == 401:
                    results["openai"]["error"] = "Invalid API key"
                else:
                    results["openai"]["error"] = f"HTTP {response.status_code}"
        else:
            results["openai"]["error"] = "API key not configured"
    except Exception as e:
        results["openai"]["error"] = str(e)

    try:
        slack_webhook = service.get_value("SLACK_WEBHOOK_URL", "")
        results["slack"]["configured"] = bool(slack_webhook and slack_webhook.strip())

        if slack_webhook and slack_webhook.strip():
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test with a dry-run (empty payload returns invalid_payload but confirms connectivity)
                response = await client.post(
                    slack_webhook,
                    json={"text": ""},
                    timeout=10.0
                )
                # Slack returns 200 for valid webhook even with empty text (or 400 for invalid payload)
                if response.status_code in (200, 400):
                    results["slack"]["connected"] = True
                    results["slack"]["details"] = {"webhook": "configured"}
                else:
                    results["slack"]["error"] = f"HTTP {response.status_code}"
        else:
            results["slack"]["error"] = "SLACK_WEBHOOK_URL not configured"
    except Exception as e:
        results["slack"]["error"] = str(e)

    try:
        sendgrid_key = service.get_value("SENDGRID_API_KEY", "")
        sendgrid_from = service.get_value("SENDGRID_FROM_EMAIL", "")
        results["sendgrid"]["configured"] = bool(sendgrid_key and sendgrid_key.strip())

        if sendgrid_key and sendgrid_key.strip():
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.sendgrid.com/v3/user/profile",
                    headers={"Authorization": f"Bearer {sendgrid_key}"}
                )
                if response.status_code == 200:
                    results["sendgrid"]["connected"] = True
                    profile = response.json()
                    results["sendgrid"]["details"] = {
                        "from_email": sendgrid_from or "Not set",
                        "username": profile.get("username", "")
                    }
                elif response.status_code == 401:
                    results["sendgrid"]["error"] = "Invalid API key"
                else:
                    results["sendgrid"]["error"] = f"HTTP {response.status_code}"
        else:
            results["sendgrid"]["error"] = "SENDGRID_API_KEY not configured"
    except Exception as e:
        results["sendgrid"]["error"] = str(e)

    results["mailgun"] = {"connected": False, "configured": False, "error": None, "details": {}}
    try:
        mailgun_key = service.get_value("MAILGUN_API_KEY", "") or os.environ.get("MAILGUN_API_KEY", "")
        mailgun_domain = service.get_value("MAILGUN_DOMAIN", "") or os.environ.get("MAILGUN_DOMAIN", "")
        results["mailgun"]["configured"] = bool(mailgun_key and mailgun_domain)

        if mailgun_key and mailgun_domain:
            mailgun_region = service.get_value("MAILGUN_REGION", "eu")
            api_host = "api.eu.mailgun.net" if mailgun_region == "eu" else "api.mailgun.net"
            import base64
            auth = base64.b64encode(f"api:{mailgun_key}".encode()).decode()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://{api_host}/v3/domains/{mailgun_domain}",
                    headers={"Authorization": f"Basic {auth}"}
                )
                if response.status_code == 200:
                    results["mailgun"]["connected"] = True
                    results["mailgun"]["details"] = {"domain": mailgun_domain}
                else:
                    results["mailgun"]["error"] = f"HTTP {response.status_code}"
        else:
            missing = []
            if not mailgun_key:
                missing.append("MAILGUN_API_KEY")
            if not mailgun_domain:
                missing.append("MAILGUN_DOMAIN")
            results["mailgun"]["error"] = f"Missing: {', '.join(missing)}"
    except Exception as e:
        results["mailgun"]["error"] = str(e)

    # LinkedIn
    results["linkedin"] = {"connected": False, "configured": False, "error": None, "details": {}}
    try:
        linkedin_token = service.get_value("LINKEDIN_ACCESS_TOKEN", "") or os.environ.get("LINKEDIN_ACCESS_TOKEN", "")
        results["linkedin"]["configured"] = bool(linkedin_token)
        if linkedin_token:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.linkedin.com/v2/me",
                    headers={"Authorization": f"Bearer {linkedin_token}"}
                )
                if response.status_code == 200:
                    results["linkedin"]["connected"] = True
                    profile = response.json()
                    first = profile.get("localizedFirstName", "")
                    last = profile.get("localizedLastName", "")
                    results["linkedin"]["details"] = {"profile": f"{first} {last}".strip()}
                elif response.status_code == 401:
                    results["linkedin"]["error"] = "Access token expired or invalid"
                else:
                    results["linkedin"]["error"] = f"HTTP {response.status_code}"
        else:
            results["linkedin"]["error"] = "LINKEDIN_ACCESS_TOKEN not configured"
    except Exception as e:
        results["linkedin"]["error"] = str(e)

    # Twitter/X
    results["twitter"] = {"connected": False, "configured": False, "error": None, "details": {}}
    try:
        twitter_key = service.get_value("TWITTER_API_KEY", "") or os.environ.get("TWITTER_API_KEY", "")
        twitter_secret = service.get_value("TWITTER_API_SECRET", "") or os.environ.get("TWITTER_API_SECRET", "")
        results["twitter"]["configured"] = bool(twitter_key and twitter_secret)
        if twitter_key and twitter_secret:
            twitter_token = service.get_value("TWITTER_ACCESS_TOKEN", "") or os.environ.get("TWITTER_ACCESS_TOKEN", "")
            if twitter_token:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        "https://api.twitter.com/2/users/me",
                        headers={"Authorization": f"Bearer {twitter_token}"}
                    )
                    if response.status_code == 200:
                        results["twitter"]["connected"] = True
                        data = response.json().get("data", {})
                        results["twitter"]["details"] = {"username": f"@{data.get('username', 'N/A')}"}
                    elif response.status_code == 401:
                        results["twitter"]["error"] = "Token expired or invalid"
                    else:
                        results["twitter"]["error"] = f"HTTP {response.status_code}"
            else:
                results["twitter"]["error"] = "TWITTER_ACCESS_TOKEN not configured"
        else:
            results["twitter"]["error"] = "API credentials not configured"
    except Exception as e:
        results["twitter"]["error"] = str(e)

    # WordPress/Blog CMS
    results["wordpress"] = {"connected": False, "configured": False, "error": None, "details": {}}
    try:
        blog_url = service.get_value("BLOG_CMS_URL", "") or os.environ.get("BLOG_CMS_URL", "")
        blog_user = service.get_value("BLOG_USERNAME", "")
        blog_pass = service.get_value("BLOG_APP_PASSWORD", "")
        has_auth = bool(blog_user and blog_pass)
        results["wordpress"]["configured"] = bool(blog_url and has_auth)
        if blog_url:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{blog_url}/wp-json/wp/v2/posts?per_page=1")
                if response.status_code == 200:
                    results["wordpress"]["connected"] = has_auth
                    results["wordpress"]["details"] = {"url": blog_url}
                    if not has_auth:
                        results["wordpress"]["error"] = "Site reachable but BLOG_USERNAME / BLOG_APP_PASSWORD not set — cannot publish"
                elif response.status_code == 401:
                    results["wordpress"]["error"] = "Authentication required"
                else:
                    results["wordpress"]["error"] = f"HTTP {response.status_code}"
        else:
            results["wordpress"]["error"] = "BLOG_CMS_URL not configured"
    except Exception as e:
        results["wordpress"]["error"] = str(e)

    # Mailchimp
    results["mailchimp"] = {"connected": False, "configured": False, "error": None, "details": {}}
    try:
        mc_key = service.get_value("MAILCHIMP_API_KEY", "") or os.environ.get("MAILCHIMP_API_KEY", "")
        results["mailchimp"]["configured"] = bool(mc_key)
        if mc_key:
            dc = mc_key.split("-")[-1] if "-" in mc_key else "us1"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://{dc}.api.mailchimp.com/3.0/",
                    headers={"Authorization": f"Bearer {mc_key}"}
                )
                if response.status_code == 200:
                    results["mailchimp"]["connected"] = True
                    data = response.json()
                    results["mailchimp"]["details"] = {"account": data.get("account_name", "N/A")}
                elif response.status_code == 401:
                    results["mailchimp"]["error"] = "Invalid API key"
                else:
                    results["mailchimp"]["error"] = f"HTTP {response.status_code}"
        else:
            results["mailchimp"]["error"] = "MAILCHIMP_API_KEY not configured"
    except Exception as e:
        results["mailchimp"]["error"] = str(e)

    # Google Perspective API
    results["perspective"] = {"connected": False, "configured": False, "error": None, "details": {}}
    try:
        perspective_key = service.get_value("PERSPECTIVE_API_KEY", "") or os.environ.get("PERSPECTIVE_API_KEY", "")
        results["perspective"]["configured"] = bool(perspective_key)
        if perspective_key:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={perspective_key}",
                    json={
                        "comment": {"text": "test"},
                        "languages": ["en"],
                        "requestedAttributes": {"TOXICITY": {}}
                    }
                )
                if response.status_code == 200:
                    results["perspective"]["connected"] = True
                    results["perspective"]["details"] = {"service": "Toxicity Analysis"}
                elif response.status_code == 400:
                    results["perspective"]["connected"] = True
                    results["perspective"]["details"] = {"service": "Toxicity Analysis"}
                elif response.status_code == 403:
                    results["perspective"]["error"] = "API key invalid or quota exceeded"
                else:
                    results["perspective"]["error"] = f"HTTP {response.status_code}"
        else:
            results["perspective"]["error"] = "PERSPECTIVE_API_KEY not configured"
    except Exception as e:
        results["perspective"]["error"] = str(e)

    # Apify
    results["apify"] = {"connected": False, "configured": False, "error": None, "details": {}}
    try:
        apify_token = service.get_value("APIFY_API_TOKEN", "") or os.environ.get("APIFY_API_TOKEN", "")
        results["apify"]["configured"] = bool(apify_token)
        if apify_token:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.apify.com/v2/users/me",
                    params={"token": apify_token}
                )
                if response.status_code == 200:
                    results["apify"]["connected"] = True
                    data = response.json().get("data", {})
                    results["apify"]["details"] = {"username": data.get("username", "N/A")}
                elif response.status_code == 401:
                    results["apify"]["error"] = "Invalid API token"
                else:
                    results["apify"]["error"] = f"HTTP {response.status_code}"
        else:
            results["apify"]["error"] = "APIFY_API_TOKEN not configured"
    except Exception as e:
        results["apify"]["error"] = str(e)

    return results
