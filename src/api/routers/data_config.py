"""
API router for managing data configuration files.
Provides CRUD endpoints for claims, brand voice, competitors, and product catalog.

These files are the core configuration that drives content generation:
- data/claim_library/claims.csv - Verified claims for content grounding
- data/company/brand_voice.json - Brand voice and style guidelines  
- data/competitors/competitors.csv - Competitor intelligence
- data/products/catalog.json - Product/module catalog
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
import logging
import csv
import json
import os
from pathlib import Path
from datetime import datetime
import io
import shutil

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-config", tags=["Data Configuration"])
DATA_DIR = Path("/app/data")
CLAIMS_PATH = DATA_DIR / "claim_library" / "claims.csv"
BRAND_VOICE_PATH = DATA_DIR / "company" / "brand_voice.json"
COMPETITORS_PATH = DATA_DIR / "competitors" / "competitors.csv"
PRODUCTS_PATH = DATA_DIR / "products" / "catalog.json"
VERSIONS_DIR = DATA_DIR / ".versions"
CONFIG_FILES = {
    "claims": CLAIMS_PATH,
    "brand_voice": BRAND_VOICE_PATH,
    "competitors": COMPETITORS_PATH,
    "products": PRODUCTS_PATH,
}


def _ensure_versions_dir():
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _create_version(config_name: str, reason: str = "manual") -> Optional[str]:
    """
    Create a versioned backup of a config file before modifying it.
    Returns the version ID or None if the source doesn't exist.
    """
    source = CONFIG_FILES.get(config_name)
    if not source or not source.exists():
        return None
    _ensure_versions_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    version_id = f"{config_name}__{ts}__{reason}"
    dest = VERSIONS_DIR / (version_id + source.suffix)
    shutil.copy2(str(source), str(dest))
    meta = {
        "version_id": version_id,
        "config_name": config_name,
        "created_at": datetime.utcnow().isoformat(),
        "reason": reason,
        "file_size": source.stat().st_size,
        "source_file": str(source),
    }
    meta_path = VERSIONS_DIR / (version_id + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Created version {version_id} for {config_name}")
    return version_id


def _list_versions(config_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all available versions, optionally filtered by config name."""
    _ensure_versions_dir()
    versions = []
    for meta_file in sorted(VERSIONS_DIR.glob("*.meta.json"), reverse=True):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if config_name and meta.get("config_name") != config_name:
                continue
            versions.append(meta)
        except Exception:
            continue
    return versions


def _restore_version(version_id: str) -> bool:
    """Restore a config file from a versioned backup. Creates a backup of current first."""
    _ensure_versions_dir()
    meta_path = VERSIONS_DIR / (version_id + ".meta.json")
    if not meta_path.exists():
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    config_name = meta["config_name"]
    target = CONFIG_FILES.get(config_name)
    if not target:
        return False
    suffix = target.suffix
    backup_path = VERSIONS_DIR / (version_id + suffix)
    if not backup_path.exists():
        return False
    _create_version(config_name, reason="pre_restore")
    shutil.copy2(str(backup_path), str(target))
    logger.info(f"Restored {config_name} from version {version_id}")
    return True


class ClaimCreate(BaseModel):
    claim_text: str = Field(..., description="The claim text")
    claim_type: str = Field("qualitative", description="Type: quantitative, qualitative, methodological, process")
    personas: List[str] = Field(default_factory=list, description="Target personas")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    source_url: str = Field("", description="URL of the source")
    source_title: str = Field("", description="Title of the source")
    date: str = Field("", description="Date of the source")
    evidence_excerpt: str = Field("", description="Evidence excerpt from source")
    confidence: int = Field(3, ge=1, le=5, description="Confidence level 1-5")


class ClaimUpdate(BaseModel):
    claim_text: Optional[str] = None
    claim_type: Optional[str] = None
    personas: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    date: Optional[str] = None
    evidence_excerpt: Optional[str] = None
    confidence: Optional[int] = Field(None, ge=1, le=5)


class CompetitorCreate(BaseModel):
    name: str = Field(..., description="Competitor name")
    category: str = Field("", description="Market category")
    url: str = Field("", description="Website URL")
    key_features: str = Field("", description="Key features")
    typical_claims: str = Field("", description="Typical marketing claims")
    differentiators_vs_us: str = Field("", description="How we differentiate")
    risky_topics: str = Field("", description="Topics to avoid")


class CompetitorUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    url: Optional[str] = None
    key_features: Optional[str] = None
    typical_claims: Optional[str] = None
    differentiators_vs_us: Optional[str] = None
    risky_topics: Optional[str] = None


def read_csv_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_file(path: Path, data: List[Dict[str, Any]], fieldnames: List[str]):
    """Auto-versions before write."""
    for name, p in CONFIG_FILES.items():
        if p == path:
            _create_version(name, reason="auto")
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path: Path, data: Dict[str, Any]):
    """Auto-versions before write."""
    for name, p in CONFIG_FILES.items():
        if p == path:
            _create_version(name, reason="auto")
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@router.get("/claims")
async def list_claims(
    limit: int = 100,
    offset: int = 0,
    claim_type: Optional[str] = None,
    confidence_min: Optional[int] = None,
    search: Optional[str] = None
):
    """List all claims with optional filtering."""
    try:
        claims = read_csv_file(CLAIMS_PATH)
        
        if claim_type:
            claims = [c for c in claims if c.get('claim_type', '').lower() == claim_type.lower()]
        
        if confidence_min:
            claims = [c for c in claims if int(c.get('confidence', 0)) >= confidence_min]
        
        if search:
            search_lower = search.lower()
            claims = [c for c in claims if search_lower in c.get('claim_text', '').lower()]
        
        total = len(claims)
        claims = claims[offset:offset + limit]
        
        return {
            "claims": claims,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to list claims: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims/{claim_id}")
async def get_claim(claim_id: str):
    try:
        claims = read_csv_file(CLAIMS_PATH)
        claim = next((c for c in claims if c.get('id') == claim_id), None)
        
        if not claim:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
        
        return claim
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get claim: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/claims")
async def create_claim(claim: ClaimCreate):
    try:
        claims = read_csv_file(CLAIMS_PATH)
        
        existing_ids = [c.get('id', '') for c in claims]
        max_num = 0
        for cid in existing_ids:
            if cid.startswith('CLM_'):
                try:
                    num = int(cid[4:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        
        new_id = f"CLM_{max_num + 1:03d}"
        
        new_claim = {
            "id": new_id,
            "claim_text": claim.claim_text,
            "claim_type": claim.claim_type,
            "personas": str(claim.personas),
            "tags": str(claim.tags),
            "source_url": claim.source_url,
            "source_title": claim.source_title,
            "date": claim.date or datetime.now().strftime("%Y-%m-%d"),
            "evidence_excerpt": claim.evidence_excerpt,
            "confidence": str(claim.confidence)
        }
        
        claims.append(new_claim)
        
        fieldnames = ["id", "claim_text", "claim_type", "personas", "tags", 
                      "source_url", "source_title", "date", "evidence_excerpt", "confidence"]
        write_csv_file(CLAIMS_PATH, claims, fieldnames)
        
        return {"success": True, "claim_id": new_id, "message": "Claim created"}
    except Exception as e:
        logger.error(f"Failed to create claim: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/claims/{claim_id}")
async def update_claim(claim_id: str, update: ClaimUpdate):
    try:
        claims = read_csv_file(CLAIMS_PATH)
        claim_idx = next((i for i, c in enumerate(claims) if c.get('id') == claim_id), None)
        
        if claim_idx is None:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
        
        update_dict = update.model_dump(exclude_none=True)
        for key, value in update_dict.items():
            if key in ['personas', 'tags']:
                claims[claim_idx][key] = str(value)
            elif key == 'confidence':
                claims[claim_idx][key] = str(value)
            else:
                claims[claim_idx][key] = value
        
        fieldnames = ["id", "claim_text", "claim_type", "personas", "tags",
                      "source_url", "source_title", "date", "evidence_excerpt", "confidence"]
        write_csv_file(CLAIMS_PATH, claims, fieldnames)
        
        return {"success": True, "message": f"Claim {claim_id} updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update claim: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/claims/{claim_id}")
async def delete_claim(claim_id: str):
    try:
        claims = read_csv_file(CLAIMS_PATH)
        original_count = len(claims)
        claims = [c for c in claims if c.get('id') != claim_id]
        
        if len(claims) == original_count:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
        
        fieldnames = ["id", "claim_text", "claim_type", "personas", "tags",
                      "source_url", "source_title", "date", "evidence_excerpt", "confidence"]
        write_csv_file(CLAIMS_PATH, claims, fieldnames)
        
        return {"success": True, "message": f"Claim {claim_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete claim: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claims/types/list")
async def list_claim_types():
    try:
        claims = read_csv_file(CLAIMS_PATH)
        types = list(set(c.get('claim_type', '') for c in claims if c.get('claim_type')))
        return {"types": sorted(types)}
    except Exception as e:
        logger.error(f"Failed to list claim types: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brand-voice")
async def get_brand_voice():
    try:
        brand_voice = read_json_file(BRAND_VOICE_PATH)
        return brand_voice
    except Exception as e:
        logger.error(f"Failed to get brand voice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/brand-voice")
async def update_brand_voice(brand_voice: Dict[str, Any]):
    try:
        required_sections = ["profile", "language", "goals", "audience", "tone", "style"]
        missing = [s for s in required_sections if s not in brand_voice]
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required sections: {missing}"
            )
        
        write_json_file(BRAND_VOICE_PATH, brand_voice)
        return {"success": True, "message": "Brand voice updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update brand voice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brand-voice/{section}")
async def get_brand_voice_section(section: str):
    try:
        brand_voice = read_json_file(BRAND_VOICE_PATH)
        
        if section not in brand_voice:
            raise HTTPException(status_code=404, detail=f"Section '{section}' not found")
        
        return {section: brand_voice[section]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get brand voice section: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/brand-voice/{section}")
async def update_brand_voice_section(section: str, data: Dict[str, Any]):
    try:
        brand_voice = read_json_file(BRAND_VOICE_PATH)
        brand_voice[section] = data
        write_json_file(BRAND_VOICE_PATH, brand_voice)
        return {"success": True, "message": f"Section '{section}' updated"}
    except Exception as e:
        logger.error(f"Failed to update brand voice section: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitors")
async def list_competitors(
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """List all competitors with optional filtering."""
    try:
        competitors = read_csv_file(COMPETITORS_PATH)
        
        if category:
            competitors = [c for c in competitors if category.lower() in c.get('category', '').lower()]
        
        if search:
            search_lower = search.lower()
            competitors = [c for c in competitors if 
                          search_lower in c.get('name', '').lower() or
                          search_lower in c.get('key_features', '').lower()]
        
        return {
            "competitors": competitors,
            "total": len(competitors)
        }
    except Exception as e:
        logger.error(f"Failed to list competitors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitors/{name}")
async def get_competitor(name: str):
    try:
        competitors = read_csv_file(COMPETITORS_PATH)
        competitor = next((c for c in competitors if c.get('name', '').lower() == name.lower()), None)
        
        if not competitor:
            raise HTTPException(status_code=404, detail=f"Competitor '{name}' not found")
        
        return competitor
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get competitor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitors")
async def create_competitor(competitor: CompetitorCreate):
    try:
        competitors = read_csv_file(COMPETITORS_PATH)
        
        if any(c.get('name', '').lower() == competitor.name.lower() for c in competitors):
            raise HTTPException(status_code=400, detail=f"Competitor '{competitor.name}' already exists")
        
        new_competitor = {
            "name": competitor.name,
            "category": competitor.category,
            "url": competitor.url,
            "key_features": competitor.key_features,
            "typical_claims": competitor.typical_claims,
            "differentiators_vs_us": competitor.differentiators_vs_us,
            "risky_topics": competitor.risky_topics,
            "last_checked": datetime.now().strftime("%Y-%m-%d")
        }
        
        competitors.append(new_competitor)
        
        fieldnames = ["name", "category", "url", "key_features", "typical_claims", 
                      "differentiators_vs_us", "risky_topics", "last_checked"]
        write_csv_file(COMPETITORS_PATH, competitors, fieldnames)
        
        return {"success": True, "message": f"Competitor '{competitor.name}' created"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create competitor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/competitors/{name}")
async def update_competitor(name: str, update: CompetitorUpdate):
    try:
        competitors = read_csv_file(COMPETITORS_PATH)
        comp_idx = next((i for i, c in enumerate(competitors) if c.get('name', '').lower() == name.lower()), None)
        
        if comp_idx is None:
            raise HTTPException(status_code=404, detail=f"Competitor '{name}' not found")
        
        update_dict = update.model_dump(exclude_none=True)
        for key, value in update_dict.items():
            competitors[comp_idx][key] = value
        competitors[comp_idx]["last_checked"] = datetime.now().strftime("%Y-%m-%d")
        
        fieldnames = ["name", "category", "url", "key_features", "typical_claims",
                      "differentiators_vs_us", "risky_topics", "last_checked"]
        write_csv_file(COMPETITORS_PATH, competitors, fieldnames)
        
        return {"success": True, "message": f"Competitor '{name}' updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update competitor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/competitors/{name}")
async def delete_competitor(name: str):
    try:
        competitors = read_csv_file(COMPETITORS_PATH)
        original_count = len(competitors)
        competitors = [c for c in competitors if c.get('name', '').lower() != name.lower()]
        
        if len(competitors) == original_count:
            raise HTTPException(status_code=404, detail=f"Competitor '{name}' not found")
        
        fieldnames = ["name", "category", "url", "key_features", "typical_claims",
                      "differentiators_vs_us", "risky_topics", "last_checked"]
        write_csv_file(COMPETITORS_PATH, competitors, fieldnames)
        
        return {"success": True, "message": f"Competitor '{name}' deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete competitor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Product Catalog Endpoints

@router.get("/products")
async def get_product_catalog():
    try:
        catalog = read_json_file(PRODUCTS_PATH)
        return catalog
    except Exception as e:
        logger.error(f"Failed to get product catalog: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/products")
async def update_product_catalog(catalog: Dict[str, Any]):
    try:
        write_json_file(PRODUCTS_PATH, catalog)
        return {"success": True, "message": "Product catalog updated"}
    except Exception as e:
        logger.error(f"Failed to update product catalog: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/modules")
async def list_product_modules():
    try:
        catalog = read_json_file(PRODUCTS_PATH)
        modules = catalog.get("modules", [])
        return {
            "modules": modules,
            "total": len(modules)
        }
    except Exception as e:
        logger.error(f"Failed to list modules: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/modules/{module_id}")
async def get_product_module(module_id: str):
    try:
        catalog = read_json_file(PRODUCTS_PATH)
        module = next((m for m in catalog.get("modules", []) if m.get("id") == module_id), None)
        
        if not module:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
        
        return module
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get module: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/products/modules/{module_id}")
async def update_product_module(module_id: str, module_data: Dict[str, Any]):
    try:
        catalog = read_json_file(PRODUCTS_PATH)
        modules = catalog.get("modules", [])
        
        module_idx = next((i for i, m in enumerate(modules) if m.get("id") == module_id), None)
        
        if module_idx is None:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
        
        module_data["id"] = module_id
        catalog["modules"][module_idx] = module_data
        
        write_json_file(PRODUCTS_PATH, catalog)
        return {"success": True, "message": f"Module '{module_id}' updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update module: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/packages")
async def list_product_packages():
    try:
        catalog = read_json_file(PRODUCTS_PATH)
        packages = catalog.get("packages", [])
        return {
            "packages": packages,
            "total": len(packages)
        }
    except Exception as e:
        logger.error(f"Failed to list packages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/governance")
async def get_product_governance():
    try:
        catalog = read_json_file(PRODUCTS_PATH)
        return catalog.get("governance", {})
    except Exception as e:
        logger.error(f"Failed to get governance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/products/governance")
async def update_product_governance(governance: Dict[str, Any]):
    try:
        catalog = read_json_file(PRODUCTS_PATH)
        catalog["governance"] = governance
        write_json_file(PRODUCTS_PATH, catalog)
        return {"success": True, "message": "Governance rules updated"}
    except Exception as e:
        logger.error(f"Failed to update governance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_data_config_summary():
    try:
        claims = read_csv_file(CLAIMS_PATH)
        brand_voice = read_json_file(BRAND_VOICE_PATH)
        competitors = read_csv_file(COMPETITORS_PATH)
        catalog = read_json_file(PRODUCTS_PATH)
        
        return {
            "claims": {
                "total": len(claims),
                "types": list(set(c.get('claim_type', '') for c in claims)),
                "path": str(CLAIMS_PATH)
            },
            "brand_voice": {
                "sections": list(brand_voice.keys()),
                "profile_name": brand_voice.get("profile", {}).get("name", ""),
                "path": str(BRAND_VOICE_PATH)
            },
            "competitors": {
                "total": len(competitors),
                "categories": list(set(c.get('category', '') for c in competitors)),
                "path": str(COMPETITORS_PATH)
            },
            "products": {
                "modules": len(catalog.get("modules", [])),
                "packages": len(catalog.get("packages", [])),
                "path": str(PRODUCTS_PATH)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get data config summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload-all")
async def reload_all_data_configs():
    """
    Force reload of all data configuration files.
    This clears any caches and ensures fresh data is loaded.
    """
    try:
        claims = read_csv_file(CLAIMS_PATH)
        brand_voice = read_json_file(BRAND_VOICE_PATH)
        competitors = read_csv_file(COMPETITORS_PATH)
        catalog = read_json_file(PRODUCTS_PATH)
        
        return {
            "success": True,
            "message": "All data configuration files reloaded",
            "counts": {
                "claims": len(claims),
                "brand_voice_sections": len(brand_voice),
                "competitors": len(competitors),
                "modules": len(catalog.get("modules", []))
            }
        }
    except Exception as e:
        logger.error(f"Failed to reload data configs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions")
async def list_config_versions(
    config_name: Optional[str] = None,
    limit: int = 50
):
    """List available versions/backups for data config files."""
    try:
        versions = _list_versions(config_name)
        return {"versions": versions[:limit], "total": len(versions)}
    except Exception as e:
        logger.error(f"Failed to list versions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/versions/{version_id}/restore")
async def restore_config_version(version_id: str):
    """Restore a data config file from a previous version."""
    try:
        success = _restore_version(version_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")
        return {"success": True, "message": f"Restored from version {version_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restore version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/versions/{version_id}")
async def delete_config_version(version_id: str):
    """Delete a specific version backup."""
    try:
        _ensure_versions_dir()
        meta_path = VERSIONS_DIR / (version_id + ".meta.json")
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")
        with open(meta_path) as f:
            meta = json.load(f)
        config_name = meta["config_name"]
        target = CONFIG_FILES.get(config_name)
        suffix = target.suffix if target else ""
        backup_path = VERSIONS_DIR / (version_id + suffix)
        if backup_path.exists():
            backup_path.unlink()
        meta_path.unlink()
        return {"success": True, "message": f"Deleted version {version_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete version: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup")
async def create_full_backup():
    """Create a manual backup of all data config files."""
    try:
        version_ids = []
        for name in CONFIG_FILES:
            vid = _create_version(name, reason="manual_backup")
            if vid:
                version_ids.append(vid)
        return {
            "success": True,
            "message": f"Created {len(version_ids)} backups",
            "versions": version_ids
        }
    except Exception as e:
        logger.error(f"Failed to create backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup/{config_name}")
async def create_config_backup(config_name: str):
    """Create a manual backup of a specific data config file."""
    if config_name not in CONFIG_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown config: {config_name}")
    try:
        vid = _create_version(config_name, reason="manual_backup")
        if not vid:
            raise HTTPException(status_code=404, detail=f"Config file for '{config_name}' not found")
        return {"success": True, "version_id": vid}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
