"""
Health check and system status endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any, Optional
import psutil
import httpx
import logging
from datetime import datetime

from ..dependencies import get_db, get_redis
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

_UPTIME_REDIS_KEY = "agentic:api:start_time"

async def get_or_set_start_time(redis) -> datetime:
    """Get API start time from Redis, or set it if not exists.
    
    This persists across code reloads (uvicorn --reload).
    """
    try:
        stored_time = await redis.get(_UPTIME_REDIS_KEY)
        if stored_time:
            return datetime.fromisoformat(stored_time)
        else:
            # First startup - set the time
            now = datetime.utcnow()
            await redis.set(_UPTIME_REDIS_KEY, now.isoformat())
            # Expire after 7 days to auto-reset on long restarts
            await redis.expire(_UPTIME_REDIS_KEY, 7 * 24 * 3600)
            return now
    except Exception as e:
        logger.warning(f"Redis uptime storage failed, using current time: {e}")
        return datetime.utcnow()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


async def check_ollama_gpu() -> Dict[str, Any]:
    """Check Ollama service and its GPU status.
    
    Tries multiple host endpoints for Docker/native compatibility:
    1. Configured OLLAMA_HOST from settings
    2. host.docker.internal (Docker on Linux/Mac)
    3. localhost fallback
    """
    configured_host = settings.OLLAMA_HOST
    
    import re
    port_match = re.search(r':(\d+)$', configured_host.rstrip('/'))
    port = port_match.group(1) if port_match else "11434"
    
    hosts_to_try = [
        configured_host,
        f"http://host.docker.internal:{port}",
        f"http://172.17.0.1:{port}",  # Docker bridge gateway
        f"http://localhost:{port}",
    ]
    seen = set()
    hosts_to_try = [h for h in hosts_to_try if not (h in seen or seen.add(h))]
    
    health_timeout = getattr(settings, 'HEALTH_CHECK_TIMEOUT', 5.0)
    last_error = None
    
    for ollama_host in hosts_to_try:
        try:
            async with httpx.AsyncClient(timeout=health_timeout) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    
                    model_list = []
                    for m in models:
                        model_info = {
                            "name": m.get("name", "unknown"),
                            "size": f"{m.get('size', 0) / 1e9:.1f} GB",
                            "modified": m.get("modified_at", "")[:10] if m.get("modified_at") else ""
                        }
                        model_list.append(model_info)

                    ps_response = await client.get(f"{ollama_host}/api/ps")
                    running_models = []
                    gpu_info = None
                    loaded_model_names = []

                    if ps_response.status_code == 200:
                        ps_data = ps_response.json()
                        running_models = ps_data.get("models", [])
                        loaded_model_names = [m.get("name", "unknown") for m in running_models]

                        for model in running_models:
                            if model.get("size_vram", 0) > 0:
                                gpu_info = {
                                    "model": model.get("name", "Unknown"),
                                    "vram_used": f"{model.get('size_vram', 0) / 1e9:.2f} GB",
                                    "size": f"{model.get('size', 0) / 1e9:.2f} GB"
                                }
                                break

                    configured_model = getattr(settings, 'OLLAMA_MODEL', 'qwen3:8b')

                    return {
                        "status": "healthy",
                        "message": f"Ollama GPU inference available (via {ollama_host})",
                        "models_available": len(models),
                        "models_loaded": len(running_models),
                        "model_list": model_list,
                        "loaded_models": loaded_model_names,
                        "configured_model": configured_model,
                        "gpu_info": gpu_info,
                        "model": "NVIDIA GPU (via Ollama)",
                        "host": ollama_host
                    }
                else:
                    last_error = f"Ollama returned status {response.status_code}"
                    continue
        except httpx.ConnectError as e:
            last_error = f"Connect error to {ollama_host}"
            logger.debug(f"Ollama not reachable at {ollama_host}: {e}")
            continue
        except Exception as e:
            last_error = str(e)
            logger.debug(f"Error checking Ollama at {ollama_host}: {e}")
            continue
    
    # All hosts failed
    return {
        "status": "unavailable",
        "message": f"Ollama service not reachable. Last error: {last_error}. "
                   f"Ensure Ollama is bound to 0.0.0.0 (set OLLAMA_HOST=0.0.0.0 in ollama.service)"
    }

@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis)
):
    start_time = await get_or_set_start_time(redis)
    uptime_seconds = (datetime.utcnow() - start_time).total_seconds()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime_seconds,
        "version": getattr(settings, 'API_VERSION', '1.0.0'),
        "environment": settings.ENVIRONMENT,
        "components": {}
    }

    try:
        import time as time_module
        db_start = time_module.time()
        await db.execute(text("SELECT 1"))
        db_latency = (time_module.time() - db_start) * 1000  # Convert to ms
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Database connection OK",
            "latency_ms": round(db_latency, 2)
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": str(e),
            "latency_ms": None
        }
        health_status["status"] = "degraded"

    try:
        import time as time_module
        redis_start = time_module.time()
        await redis.ping()
        redis_latency = (time_module.time() - redis_start) * 1000
        
        redis_info = await redis.info("memory")
        used_memory_mb = redis_info.get("used_memory", 0) / (1024 * 1024)
        
        health_status["components"]["redis"] = {
            "status": "healthy",
            "message": "Redis connection OK",
            "latency_ms": round(redis_latency, 2),
            "used_memory_mb": round(used_memory_mb, 2)
        }
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "message": str(e),
            "latency_ms": None
        }
        health_status["status"] = "degraded"

    # GPU check via torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            health_status["components"]["gpu"] = {
                "status": "healthy",
                "message": f"GPU available: {gpu_name}",
                "model": gpu_name,
                "memory_gb": f"{gpu_memory:.1f} GB",
                "cuda_version": torch.version.cuda
            }
        else:
            health_status["components"]["gpu"] = {
                "status": "unavailable",
                "message": "No GPU detected (CUDA not available)"
            }
    except Exception as e:
        health_status["components"]["gpu"] = {
            "status": "unavailable",
            "message": f"Error checking GPU: {str(e)}"
        }

    ollama_status = await check_ollama_gpu()
    health_status["components"]["ollama"] = {
        "status": ollama_status["status"],
        "message": ollama_status.get("message", ""),
        "models_available": ollama_status.get("models_available", 0),
        "models_loaded": ollama_status.get("models_loaded", 0),
        "model_list": ollama_status.get("model_list", []),
        "loaded_models": ollama_status.get("loaded_models", []),
        "configured_model": ollama_status.get("configured_model", ""),
        "host": settings.OLLAMA_HOST
    }

    net_io = psutil.net_io_counters()
    health_status["system"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv
        }
    }

    return health_status

@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis)
):
    try:
        await db.execute(text("SELECT 1"))
        await redis.ping()

        return {"ready": True}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"ready": False, "error": str(e)}

@router.get("/live")
async def liveness_check():
    return {"alive": True}


@router.get("/health/metrics")
async def get_system_metrics():
    """
    Get comprehensive system metrics including network, CPU, memory, disk.
    Uses psutil for real-time system metrics.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        memory = psutil.virtual_memory()
        
        disk = psutil.disk_usage('/')
        
        net_io = psutil.net_io_counters()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None
            },
            "memory": {
                "percent": memory.percent,
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2)
            },
            "disk": {
                "percent": disk.percent,
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2)
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errors_in": net_io.errin,
                "errors_out": net_io.errout
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {"error": str(e)}


@router.get("/health/prometheus/query")
async def query_prometheus(query: str, time_range: str = "1h"):
    """
    Query Prometheus for metrics.
    
    Args:
        query: PromQL query string
        time_range: Time range for range queries (e.g., "1h", "30m", "24h")
    """
    prometheus_url = getattr(settings, 'PROMETHEUS_URL', 'http://localhost:9090')
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            import re
            match = re.match(r'(\d+)([mhd])', time_range)
            if match:
                value, unit = int(match.group(1)), match.group(2)
                seconds = value * {'m': 60, 'h': 3600, 'd': 86400}.get(unit, 3600)
            else:
                seconds = 3600
            
            end_time = datetime.utcnow()
            start_time = end_time - __import__('datetime').timedelta(seconds=seconds)
            
            response = await client.get(
                f"{prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start_time.timestamp(),
                    "end": end_time.timestamp(),
                    "step": max(seconds // 60, 15)  # At least 15s step
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": f"Prometheus returned {response.status_code}"}
                
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/health/metrics/history")
async def get_metrics_history(metric: str = "cpu", time_range: str = "1h"):
    """
    Get historical metrics from Prometheus.
    
    Args:
        metric: One of "cpu", "memory", "disk", "network"
        time_range: Time range (e.g., "1h", "6h", "24h")
    """
    prometheus_url = getattr(settings, 'PROMETHEUS_URL', 'http://localhost:9090')
    
    queries = {
        "cpu": "100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)",
        "memory": "100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
        "disk": "100 * (1 - node_filesystem_avail_bytes{mountpoint='/'} / node_filesystem_size_bytes{mountpoint='/'})",
        "network_in": "rate(node_network_receive_bytes_total{device='eth0'}[5m])",
        "network_out": "rate(node_network_transmit_bytes_total{device='eth0'}[5m])"
    }
    
    if metric not in queries:
        return {"error": f"Unknown metric: {metric}. Available: {list(queries.keys())}"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            import re
            match = re.match(r'(\d+)([mhd])', time_range)
            if match:
                value, unit = int(match.group(1)), match.group(2)
                seconds = value * {'m': 60, 'h': 3600, 'd': 86400}.get(unit, 3600)
            else:
                seconds = 3600
            
            end_time = datetime.utcnow()
            start_time = end_time - __import__('datetime').timedelta(seconds=seconds)
            step = max(seconds // 60, 15)  # At least 60 data points
            
            response = await client.get(
                f"{prometheus_url}/api/v1/query_range",
                params={
                    "query": queries[metric],
                    "start": start_time.timestamp(),
                    "end": end_time.timestamp(),
                    "step": step
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and data.get("data", {}).get("result"):
                    result = data["data"]["result"][0] if data["data"]["result"] else {}
                    values = result.get("values", [])
                    
                    return {
                        "metric": metric,
                        "time_range": time_range,
                        "data_points": len(values),
                        "values": [
                            {
                                "timestamp": datetime.fromtimestamp(v[0]).isoformat(),
                                "value": float(v[1]) if v[1] != "NaN" else None
                            }
                            for v in values
                        ]
                    }
                else:
                    return {"metric": metric, "data_points": 0, "values": [], "message": "No data available"}
            else:
                # Prometheus returned error - return clear error
                logger.warning(f"Prometheus returned {response.status_code}")
                return {
                    "error": f"Prometheus unavailable (HTTP {response.status_code})",
                    "message": "Historical metrics require Prometheus. Please ensure node_exporter is running.",
                    "values": []
                }
                
    except Exception as e:
        logger.warning(f"Prometheus connection failed: {e}")
        return {
            "error": "Prometheus connection failed",
            "message": f"Unable to fetch historical metrics: {str(e)[:100]}. Ensure Prometheus is running.",
            "values": []
        }


@router.get("/health/containers")
async def get_container_metrics():
    """
    Get container metrics from cAdvisor via Prometheus.
    Returns CPU, memory, and network usage per container.
    Resolves container names via docker inspect output.
    """
    prometheus_url = getattr(settings, 'PROMETHEUS_URL', 'http://localhost:9090')
    
    containers = {}
    
    id_to_name = {}
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "\t" in line:
                    full_id, name = line.split("\t", 1)
                    id_to_name[full_id] = name
    except Exception:
        pass  # Docker CLI not available in container
    
    # Fallback: try Docker Engine API via unix socket
    if not id_to_name:
        try:
            async with httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds="/var/run/docker.sock"),
                timeout=5.0
            ) as docker_client:
                resp = await docker_client.get("http://localhost/containers/json")
                if resp.status_code == 200:
                    for c in resp.json():
                        full_id = c.get("Id", "")
                        names = c.get("Names", [])
                        name = names[0].lstrip("/") if names else full_id[:12]
                        id_to_name[full_id] = name
        except Exception:
            pass
    
    def resolve_name(docker_scope_id: str) -> str:
        """Extract Docker container ID from cgroup scope path and resolve name."""
        if "docker-" in docker_scope_id:
            full_id = docker_scope_id.split("docker-")[-1].replace(".scope", "")
            if full_id in id_to_name:
                return id_to_name[full_id]
            for stored_id, name in id_to_name.items():
                if stored_id.startswith(full_id[:12]) or full_id.startswith(stored_id[:12]):
                    return name
            return full_id[:12]
        return docker_scope_id.split("/")[-1][:20]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            docker_filter = 'id=~"/system.slice/docker-.*"'
            
            cpu_response = await client.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": f'rate(container_cpu_usage_seconds_total{{{docker_filter},cpu="total"}}[5m]) * 100'}
            )
            
            if cpu_response.status_code == 200:
                cpu_data = cpu_response.json()
                for result in cpu_data.get("data", {}).get("result", []):
                    scope_id = result["metric"].get("id", "")
                    name = resolve_name(scope_id)
                    if name not in containers:
                        containers[name] = {"name": name}
                    containers[name]["cpu_percent"] = round(float(result["value"][1]), 2)
            
            mem_response = await client.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": f'container_memory_usage_bytes{{{docker_filter}}}'}
            )
            
            if mem_response.status_code == 200:
                mem_data = mem_response.json()
                for result in mem_data.get("data", {}).get("result", []):
                    scope_id = result["metric"].get("id", "")
                    name = resolve_name(scope_id)
                    if name not in containers:
                        containers[name] = {"name": name}
                    memory_bytes = float(result["value"][1])
                    containers[name]["memory_mb"] = round(memory_bytes / (1024**2), 1)
            
            net_rx_response = await client.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": f'rate(container_network_receive_bytes_total{{{docker_filter}}}[5m])'}
            )
            
            if net_rx_response.status_code == 200:
                net_data = net_rx_response.json()
                for result in net_data.get("data", {}).get("result", []):
                    scope_id = result["metric"].get("id", "")
                    name = resolve_name(scope_id)
                    if name not in containers:
                        containers[name] = {"name": name}
                    containers[name]["network_rx_kbps"] = round(float(result["value"][1]) / 1024, 2)
            
            net_tx_response = await client.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": f'rate(container_network_transmit_bytes_total{{{docker_filter}}}[5m])'}
            )
            
            if net_tx_response.status_code == 200:
                net_data = net_tx_response.json()
                for result in net_data.get("data", {}).get("result", []):
                    scope_id = result["metric"].get("id", "")
                    name = resolve_name(scope_id)
                    if name not in containers:
                        containers[name] = {"name": name}
                    containers[name]["network_tx_kbps"] = round(float(result["value"][1]) / 1024, 2)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "containers": list(containers.values())
        }
        
    except Exception as e:
        logger.error(f"Failed to get container metrics: {e}")
        return {"error": str(e), "containers": []}