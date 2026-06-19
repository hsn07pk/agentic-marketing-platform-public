"""
Centralized API Client for Agentic Dashboard
ALL backend communication - NO MOCK DATA
"""
import requests
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any
import logging
import os

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ─── Resilient Session ────────────────────────────────────────────────────
# Drop-in replacement for requests.Session that prevents the Streamlit
# dashboard from freezing ("Connecting..." / "Running...") even when the
# FastAPI backend is slow or temporarily unreachable.
#
#   CIRCUIT BREAKER  — First failed GET trips a 5-second breaker.  All
#                      subsequent GETs fail instantly (0 ms) instead of
#                      each blocking for the full timeout.
#
#   RENDER-CYCLE DEDUP — Same URL+params within 3 s returns the cached
#                        response object.  Cleared on every mutation.
#
#   SHORT TIMEOUTS   — GET: (connect=1s, read=3s).  Mutations: (2s, 15s).
#                      No retries.  Fail fast, let the user hit Refresh.
# ──────────────────────────────────────────────────────────────────────────

_circuit_open_until: float = 0.0
_circuit_lock = threading.Lock()
_dedup: Dict[str, tuple] = {}

def _trip_circuit(seconds: float = 5.0):
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = time.time() + seconds

def _reset_circuit():
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = 0.0

def _dedup_clear():
    _dedup.clear()


class _ResilientSession(requests.Session):
    """Drop-in Session with circuit breaker + dedup for GETs."""

    _GET_TIMEOUT   = (1, 3)
    _WRITE_TIMEOUT = (2, 15)

    def get(self, url, **kwargs):
        # Enforce short timeout — override any hardcoded timeout= in callers
        kwargs['timeout'] = self._GET_TIMEOUT

        # Dedup: same GET within 3 s → return previous response
        raw_key = f"{url}|{sorted((kwargs.get('params') or {}).items())}"
        key = hashlib.md5(raw_key.encode()).hexdigest()
        entry = _dedup.get(key)
        if entry and time.time() < entry[0]:
            return entry[1]
        _dedup.pop(key, None)

        # Circuit breaker: if open, raise immediately
        if time.time() < _circuit_open_until:
            raise requests.ConnectionError("Circuit breaker open")

        try:
            resp = super().get(url, **kwargs)
            _reset_circuit()
            _dedup[key] = (time.time() + 3.0, resp)
            return resp
        except (requests.ConnectionError, requests.Timeout):
            _trip_circuit(5.0)
            raise

    def post(self, url, **kwargs):
        kwargs['timeout'] = self._WRITE_TIMEOUT
        _dedup_clear()
        return super().post(url, **kwargs)

    def put(self, url, **kwargs):
        kwargs['timeout'] = self._WRITE_TIMEOUT
        _dedup_clear()
        return super().put(url, **kwargs)

    def patch(self, url, **kwargs):
        kwargs['timeout'] = self._WRITE_TIMEOUT
        _dedup_clear()
        return super().patch(url, **kwargs)

    def delete(self, url, **kwargs):
        kwargs['timeout'] = self._WRITE_TIMEOUT
        _dedup_clear()
        return super().delete(url, **kwargs)


class AgenticAPIClient:
    
    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = os.getenv('API_URL', 'http://localhost:8000')
        self.base_url = base_url.rstrip('/')
        
        self.session = _ResilientSession()
        
        # Zero retries — circuit breaker handles the failure fast-path
        adapter = HTTPAdapter(
            max_retries=Retry(total=0),
            pool_connections=10,
            pool_maxsize=10,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _handle_error(self, error: Exception, endpoint: str, default_return=None):
        error_msg = f"API Error ({endpoint}): {str(error)}"
        logger.error(error_msg)
        if default_return is not None:
            return default_return
        raise error

    def direct_get_json(self, path: str, timeout: float = 5.0) -> Optional[Dict]:
        """GET that bypasses the circuit breaker / dedup cache.

        Used for diagnostic / readiness checks that must always attempt a
        fresh request regardless of the session's circuit-breaker state.
        Returns parsed JSON on success, None on any failure.
        """
        if path.startswith("/api") or path.startswith("/health"):
            url = f"{self.base_url}{path}"
        else:
            url = f"{self.base_url}/api/v1{path}"
        try:
            resp = requests.get(url, timeout=timeout, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def get(self, endpoint: str, params: Dict = None, timeout: int = 10):
        """Generic GET request"""
        try:
            url = f"{self.base_url}/api/v1{endpoint}" if not endpoint.startswith('/api') else f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=timeout)
            return response
        except Exception as e:
            logger.error(f"GET {endpoint} failed: {e}")
            raise e

    def post(self, endpoint: str, json: Dict = None, params: Dict = None, timeout: int = 10):
        """Generic POST request"""
        try:
            url = f"{self.base_url}/api/v1{endpoint}" if not endpoint.startswith('/api') else f"{self.base_url}{endpoint}"
            response = self.session.post(url, json=json, params=params, timeout=timeout)
            return response
        except Exception as e:
            logger.error(f"POST {endpoint} failed: {e}")
            raise e

    def request(self, method: str, endpoint: str, json: Dict = None, params: Dict = None, timeout: int = 30) -> Any:
        """
        Generic request method that returns JSON response directly.
        Supports GET, POST, PUT, DELETE, PATCH methods.
        Used by the System Settings configuration UI.
        """
        try:
            url = f"{self.base_url}/api/v1{endpoint}" if not endpoint.startswith('/api') else f"{self.base_url}{endpoint}"
            
            method = method.upper()
            if method == "GET":
                response = self.session.get(url, params=params, timeout=timeout)
            elif method == "POST":
                response = self.session.post(url, json=json, params=params, timeout=timeout)
            elif method == "PUT":
                response = self.session.put(url, json=json, params=params, timeout=timeout)
            elif method == "PATCH":
                response = self.session.patch(url, json=json, params=params, timeout=timeout)
            elif method == "DELETE":
                response = self.session.delete(url, params=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"{method} {endpoint} failed: {e}")
            raise e

    
    def get_campaigns(self, status: str = None, platform: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all campaigns with optional filters"""
        try:
            params = {'limit': limit, 'offset': offset}
            if status:
                params['status'] = status
            if platform:
                params['platform'] = platform
            
            response = self.session.get(
                f"{self.base_url}/api/v1/campaigns",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_campaigns', [])
    
    def get_campaign(self, campaign_id: str) -> Optional[Dict]:
        """Get single campaign by ID"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_campaign/{campaign_id}', None)
    
    def create_campaign(self, campaign_data: Dict) -> Optional[Dict]:
        """Create new campaign"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/campaigns/",
                json=campaign_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'create_campaign', None)
    
    def start_campaign(self, campaign_id: str) -> Dict:
        """Start campaign workflow"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}/start",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'start_campaign/{campaign_id}', {'status': 'error'})
    
    def pause_campaign(self, campaign_id: str) -> Dict:
        """Pause campaign"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}/pause",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'pause_campaign/{campaign_id}', {'status': 'error'})
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict:
        """Get metrics for specific campaign"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}/metrics",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_campaign_metrics/{campaign_id}', {})

    def get_campaign_simulation(self, campaign_id: str) -> Dict:
        """Get simulation results for campaign"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}/simulation",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_campaign_simulation/{campaign_id}', {
                "has_simulation": False,
                "message": "Simulation results not available"
            })
    
    def update_campaign(self, campaign_id: str, updates: Dict) -> Dict:
        """Update campaign"""
        try:
            response = self.session.patch(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}",
                json=updates,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'update_campaign/{campaign_id}', {'status': 'error'})
    
    def delete_campaign(self, campaign_id: str) -> Dict:
        """Delete campaign"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'delete_campaign/{campaign_id}', {'status': 'error'})

    def check_campaign_completion(self, campaign_id: str) -> Dict:
        """Check if campaign meets completion criteria (budget/end date)"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}/check-completion",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'check_campaign_completion/{campaign_id}', {'status': 'error'})
    
    
    def get_metrics_overview(self, days: int = 30, include_mock: Optional[bool] = None) -> Dict:
        """Get platform-wide metrics overview"""
        try:
            params = {'days': days}
            if include_mock is not None:
                params['include_mock'] = include_mock
            response = self.session.get(
                f"{self.base_url}/api/v1/metrics/overview",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_metrics_overview', {
                "period_days": days,
                "total_campaigns": 0,
                "total_impressions": 0,
                "total_clicks": 0,
                "total_conversions": 0,
                "average_ctr": 0.0,
                "average_cpl": 0.0,
                "total_spent": 0.0,
                "roi": 0.0,
                "includes_mock_data": False
            })
    
    def get_time_series_metrics(self, metric_name: str, days: int = 30, granularity: str = "daily") -> List[Dict]:
        """Get time series data for specific metric"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/metrics/time-series",
                params={
                    'metric_name': metric_name,
                    'days': days,
                    'granularity': granularity
                },
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_time_series_metrics', [])
    
    def get_experiment_metrics(self, active_only: bool = True) -> List[Dict]:
        """Get experiment performance metrics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/metrics/experiments",
                params={'active_only': active_only},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_experiment_metrics', [])
    
    
    def get_hitl_queue(self, status: str = "pending", limit: int = 50) -> List[Dict]:
        """Get HITL review queue"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/governance/hitl-queue",
                params={'status': status, 'limit': limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_hitl_queue', [])
    
    def submit_review(self, review_data: Dict) -> Dict:
        """Submit content review decision"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/governance/review",
                json=review_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'submit_review', {'status': 'error'})
    
    def validate_content(self, content_data: Dict) -> Dict:
        """Validate content safety"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/governance/validate",
                json=content_data,
                timeout=120  # 120 seconds for local Ollama LLM (4 validation checks)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'validate_content', {})
    
    def get_safety_stats(self, days: int = 30) -> Dict:
        """Get safety validation statistics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/governance/safety-stats",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_safety_stats', {
                "period_days": days,
                "total_content": 0,
                "approved": 0,
                "rejected": 0,
                "pending_review": 0,
                "approval_rate": 0.0,
                "average_safety_score": 0.0,
                "average_toxicity_score": 0.0,
                "average_factuality_score": 0.0,
                "high_risk_content": 0
            })
    
    def get_review_history(self, limit: int = 50, days: int = 30) -> List[Dict]:
        """Get content review history"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/governance/history",
                params={'limit': limit, 'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_review_history', [])

    def list_contents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all content items with safety scores"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/governance/contents",
                params={'limit': limit, 'offset': offset},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_contents', [])

    def regenerate_content(self, content_id: str, feedback: str = "") -> Dict:
        """Request content regeneration for low safety score"""
        try:
            # Longer timeout (120s) because regeneration runs full workflow with LLM calls
            response = self.session.post(
                f"{self.base_url}/api/v1/governance/regenerate",
                json={
                    'content_id': content_id,
                    'feedback': feedback or "Safety score too low - regenerating"
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'regenerate_content', {
                'status': 'error',
                'message': str(e)
            })
    
    def get_golden_test_results(self) -> Dict:
        """Get latest golden test results"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/governance/golden-tests",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_golden_test_results', {
                "pass_rate": 0.0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "last_run": None,
                "test_details": []
            })
    
    def run_golden_tests(self) -> Dict:
        """Trigger golden test suite run"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/governance/run-golden-tests",
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'run_golden_tests', {'status': 'error', 'message': str(e)})
    
    
    def list_experiments(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List all experiments"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/experiments",
                params={'limit': limit, 'offset': offset},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_experiments', [])
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get specific experiment"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/experiments/{experiment_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_experiment/{experiment_id}', None)
    
    def get_experiment_status(self, experiment_id: str) -> Dict:
        """Get experiment status and recommendations"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/experiments/{experiment_id}/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_experiment_status/{experiment_id}', {})
    
    def create_experiment(self, experiment_data: Dict) -> Optional[Dict]:
        """Create new experiment"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/experiments",
                json=experiment_data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'create_experiment', None)

    def get_bandit_arms(self, experiment_id: str) -> List[Dict]:
        """Get bandit arms for an experiment"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/bandit-arms/experiment/{experiment_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_bandit_arms/{experiment_id}', [])

    def simulate_experiment(self, experiment_id: str, num_pulls: int = 100) -> Optional[Dict]:
        """Simulate traffic for an experiment"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/experiments/{experiment_id}/simulate",
                params={"num_pulls": num_pulls},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'simulate_experiment/{experiment_id}', None)
    
    
    def get_pending_rewards(self) -> Dict:
        """Get all pending delayed rewards"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/rewards/pending",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_pending_rewards', {"total": 0, "pending_rewards": []})
    
    def process_pending_rewards(self) -> Dict:
        """Process pending rewards"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/rewards/process",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'process_pending_rewards', {"checked": 0, "booked": 0, "expired": 0})
    
    def get_campaign_rewards(self, campaign_id: str) -> Dict:
        """Get delayed rewards for campaign"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/rewards/campaign/{campaign_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_campaign_rewards/{campaign_id}', {})
    
    def register_conversion(self, campaign_id: str, lead_email: str, lead_data: Dict) -> Dict:
        """Register conversion"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/rewards/register",
                json={
                    'campaign_id': campaign_id,
                    'lead_email': lead_email,
                    'lead_data': lead_data
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'register_conversion', {'status': 'error'})


    def get_available_personas(self) -> List[str]:
        """Get list of available persona IDs from config/personas/*.yaml"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/personas/list",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch personas from API, using defaults: {e}")
            return ["decision_maker", "practitioner", "researcher"]

    def get_persona_details(self, persona_id: str) -> Optional[Dict]:
        """Get detailed configuration for a specific persona"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/personas/{persona_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_persona_details/{persona_id}', None)

    def reload_personas(self) -> Dict:
        """Reload personas from disk (useful after adding new persona YAML files)"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/personas/reload",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'reload_personas', {'status': 'error'})

    
    def scrape_content(self, keywords: List[str], limit: int = 20, platform: str = "linkedin") -> Dict:
        """Scrape market content for inspiration"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/scraper/scrape",
                json={'keywords': keywords, 'limit': limit, 'platform': platform},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'scrape_content', {})
    
    def analyze_posts(self, posts: List[Dict]) -> Dict:
        """Analyze scraped posts"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/scraper/analyze",
                json={'posts': posts},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'analyze_posts', {})
    
    
    def get_optimal_strategy(self, strategy_request: Dict) -> Dict:
        """Get optimal strategy using contextual bandits"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/strategy/optimize",
                json=strategy_request,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_optimal_strategy', {})
    
    def update_strategy_performance(self, update_request: Dict) -> Dict:
        """Update strategy performance"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/strategy/update-performance",
                json=update_request,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_strategy_performance', {'status': 'error'})
    
    def get_strategy_performance(self, campaign_id: str) -> Dict:
        """Get strategy performance report"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/strategy/performance/{campaign_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_strategy_performance/{campaign_id}', {})
    
    
    def get_ope_status(self) -> Dict:
        """Get OPE system status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/ope/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_ope_status', {"ope_available": False})
    
    def evaluate_policy(self, ope_request: Dict) -> Dict:
        """Perform offline policy evaluation"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/ope/evaluate",
                json=ope_request,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'evaluate_policy', {})
    
    def compare_policies(self, policy_a: str, policy_b: str, campaign_id: str = None) -> Dict:
        """Compare two policies"""
        try:
            params = {'policy_a': policy_a, 'policy_b': policy_b}
            if campaign_id:
                params['campaign_id'] = campaign_id
            response = self.session.post(
                f"{self.base_url}/api/v1/ope/compare-policies",
                params=params,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'compare_policies', {})
    
    def evaluate_marl_promotion(self, request: Dict) -> Dict:
        """Evaluate MARL policy for promotion"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/ope/marl-promotion/evaluate",
                json=request,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'evaluate_marl_promotion', {})
    
    def get_marl_promotion_history(self, limit: int = 10) -> List[Dict]:
        """Get MARL promotion history"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/ope/marl-promotion/history",
                params={'limit': limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_marl_promotion_history', [])
    
    
    def start_canary_deployment(self, deployment_request: Dict) -> Dict:
        """Start canary deployment"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/canary/start",
                json=deployment_request,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'start_canary_deployment', {})
    
    def get_deployment_status(self, deployment_id: str) -> Dict:
        """Get canary deployment status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/canary/status/{deployment_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_deployment_status/{deployment_id}', {})
    
    def list_active_deployments(self) -> List[Dict]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/canary/active",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_active_deployments', [])
    
    def get_deployment_history(self, limit: int = 20) -> List[Dict]:
        """Get deployment history"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/canary/history",
                params={'limit': limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_deployment_history', [])
    
    def rollback_deployment(self, deployment_id: str, reason: str) -> Dict:
        """Rollback canary deployment"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/canary/rollback",
                json={'deployment_id': deployment_id, 'reason': reason},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'rollback_deployment', {})
    
    
    def list_agents_with_memory(self) -> List[str]:
        """List agents with episodic memory"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/memory/agents",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_agents_with_memory', [])
    
    def get_agent_memory_stats(self, agent_name: str) -> Dict:
        """Get agent memory statistics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/memory/{agent_name}/stats",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_agent_memory_stats/{agent_name}', {
                "agent_name": agent_name,
                "total_memories": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "avg_cost": 0.0,
                "avg_duration": 0.0
            })
    
    def query_agent_memory(self, agent_name: str, query: str, k: int = 5) -> List[Dict]:
        """Query agent's episodic memory"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/memory/query",
                json={'agent_name': agent_name, 'query': query, 'k': k},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'query_agent_memory/{agent_name}', [])
    
    def get_recent_memories(self, agent_name: str, limit: int = 10) -> List[Dict]:
        """Get recent memories for agent"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/memory/{agent_name}/recent",
                params={'limit': limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_recent_memories/{agent_name}', [])
    
    def get_failure_patterns(self, agent_name: str, limit: int = 10) -> List[Dict]:
        """Get failure patterns for agent"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/memory/{agent_name}/failures",
                params={'limit': limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_failure_patterns/{agent_name}', [])
    
    
    def get_knowledge_base_stats(self, collection_name: str = "documents") -> Dict:
        """Get knowledge base statistics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/stats",
                params={"collection_name": collection_name},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_knowledge_base_stats', {
                "collection_name": "documents",
                "total_documents": 0,
                "last_updated": None,
                "status": "unknown"
            })
    
    def get_kb_categories(self, collection_name: str = "documents") -> Dict:
        """Get list of categories in knowledge base"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/categories",
                params={"collection_name": collection_name},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_kb_categories', {"categories": []})
    
    def ingest_knowledge_base(self, validate: bool = True) -> Dict:
        """Ingest default knowledge base"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/knowledge-base/ingest-knowledge-base",
                params={'validate': validate},
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'ingest_knowledge_base', {})
    
    def ingest_directory(self, directory_path: str, collection_name: str = "documents", validate: bool = True) -> Dict:
        """Ingest documents from directory"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/knowledge-base/ingest-directory",
                json={
                    'directory_path': directory_path,
                    'collection_name': collection_name,
                    'validate_content': validate
                },
                timeout=180
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'ingest_directory', {})


    def clear_cache(self) -> Dict:
        """Clear the system cache"""
        try:
            response = self.session.post(f"{self.base_url}/api/v1/operations/cache/clear", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'clear_cache', {'success': False, 'message': str(e)})

    def trigger_backup(self) -> Dict:
        """Trigger a database backup"""
        try:
            response = self.session.post(f"{self.base_url}/api/v1/operations/backup/trigger", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'trigger_backup', {'success': False, 'message': str(e)})

    def rotate_logs(self) -> Dict:
        """Trigger log rotation"""
        try:
            response = self.session.post(f"{self.base_url}/api/v1/operations/logs/rotate", timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'rotate_logs', {'success': False, 'message': str(e)})
    
    
    def get_health(self) -> Dict:
        """Get system health status"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_health', {"status": "error", "message": str(e)})
    
    def get_detailed_health(self) -> Dict:
        """Get detailed system health"""
        try:
            response = self.session.get(
                f"{self.base_url}/health/detailed",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_detailed_health', {"status": "error"})
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics including network"""
        try:
            response = self.session.get(
                f"{self.base_url}/health/metrics",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_system_metrics', {"error": str(e)})
    
    def get_metrics_history(self, metric: str = "cpu", time_range: str = "1h") -> Dict:
        """Get historical metrics from Prometheus"""
        try:
            response = self.session.get(
                f"{self.base_url}/health/metrics/history",
                params={"metric": metric, "time_range": time_range},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_metrics_history', {"error": str(e), "values": []})
    
    def get_container_metrics(self) -> Dict:
        """Get container metrics from cAdvisor via Prometheus"""
        try:
            response = self.session.get(
                f"{self.base_url}/health/containers",
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_container_metrics', {"error": str(e), "containers": []})
    
    def get_readiness(self) -> Dict:
        """Check if system is ready"""
        try:
            response = self.session.get(
                f"{self.base_url}/ready",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_readiness', {"ready": False})
    
    
    def get_research_mode_status(self) -> Dict:
        """Get research mode status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/advanced-experiments/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_research_mode_status', {"research_mode_enabled": False})
    
    def run_advanced_experiment(self, experiment_request: Dict) -> Dict:
        """Run advanced research experiment"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/advanced-experiments/run",
                json=experiment_request,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'run_advanced_experiment', {})
    
    def compare_experiment_methods(self, experiment_types: List[str], n_iterations: int = 100) -> Dict:
        """Compare multiple experimental methods"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/advanced-experiments/compare-methods",
                params={'n_iterations': n_iterations},
                json={'experiment_types': experiment_types},
                timeout=600
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'compare_experiment_methods', {})
    
    def get_experiment_history(self, limit: int = 50) -> Dict:
        """Get research experiment history"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/advanced-experiments/experiment-history",
                params={'limit': limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_experiment_history', {"experiments": [], "total": 0})
    
    
    def get_api_info(self) -> Dict:
        """Get API information"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/info",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_api_info', {})
    
    
    def get_campaign_experiments(self, campaign_id: str) -> List[Dict]:
        """Get experiments for a specific campaign"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/experiments/campaign/{campaign_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_campaign_experiments', [])
    
    def post_experiment_results(self, experiment_id: str, results: Dict) -> Dict:
        """Post results for an experiment"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/experiments/{experiment_id}/results",
                json=results,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'post_experiment_results', {'status': 'error'})
    
    def get_campaign_experiment_summary(self, campaign_id: str) -> Dict:
        """Get experiment summary for a campaign"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/experiments/campaign/{campaign_id}/summary",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_campaign_experiment_summary', {})
    
    def evaluate_campaign_experiments(self, campaign_id: str) -> Dict:
        """Evaluate experiments for a campaign"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/experiments/evaluate/{campaign_id}",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'evaluate_campaign_experiments', {'status': 'error'})
    
    
    def get_booking_status(self, lead_email: str) -> Dict:
        """Get booking status for a lead"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/rewards/booking/{lead_email}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_booking_status', {})
    
    
    def clear_agent_memory(self, agent_name: str) -> Dict:
        """Clear all memories for an agent"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/memory/{agent_name}/clear",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'clear_agent_memory', {'status': 'error'})
    
    
    def ingest_directory(self, directory_path: str, collection_name: str = "agentic_kb", validate: bool = True) -> Dict:
        """Ingest a directory into knowledge base"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/knowledge-base/ingest-directory",
                json={
                    "directory_path": directory_path,
                    "collection_name": collection_name,
                    "validate": validate
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'ingest_directory', {'status': 'error'})
    
    def upload_document(self, file_data: bytes, filename: str, collection_name: str = "agentic_kb") -> Dict:
        """Upload a document to knowledge base"""
        try:
            files = {'file': (filename, file_data)}
            response = self.session.post(
                f"{self.base_url}/api/v1/knowledge-base/upload",
                files=files,
                data={'collection_name': collection_name},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'upload_document', {'status': 'error'})
    
    def list_kb_documents(self, collection_name: str = "agentic_kb", limit: int = 100, offset: int = 0) -> Dict:
        """List documents in knowledge base"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/documents",
                params={
                    'collection_name': collection_name,
                    'limit': limit,
                    'offset': offset
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_kb_documents', {'documents': [], 'total': 0})
    
    def delete_kb_document(self, document_id: str, collection_name: str = "documents") -> Dict:
        """Delete a document from knowledge base"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/knowledge-base/documents/{document_id}",
                params={'collection_name': collection_name},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'delete_kb_document', {'status': 'error'})

    def get_kb_document(self, document_id: int, include_vector: bool = False) -> Dict:
        """Get a single document by ID"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/documents/{document_id}",
                params={'include_vector': include_vector},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_kb_document', {})

    def create_kb_document(self, content: str, title: str, category: str = "general", tags: List[str] = None, collection: str = "documents") -> Dict:
        """Create a new document"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/knowledge-base/documents",
                json={
                    "content": content,
                    "title": title,
                    "category": category,
                    "tags": tags or [],
                    "collection_name": collection
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'create_kb_document', {'success': False, 'message': str(e)})

    def update_kb_document(self, document_id: int, content: str = None, metadata: Dict = None) -> Dict:
        """Update a document"""
        try:
            payload = {}
            if content is not None:
                payload["content"] = content
            if metadata is not None:
                payload["metadata"] = metadata
            
            response = self.session.put(
                f"{self.base_url}/api/v1/knowledge-base/documents/{document_id}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_kb_document', {'success': False, 'message': str(e)})
    
    def scrape_url_to_kb(self, url: str, category: str = "general", tags: List[str] = None, collection: str = "documents") -> Dict:
        """Scrape a URL and add content to knowledge base"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/knowledge-base/scrape-url",
                json={
                    "url": url,
                    "category": category,
                    "tags": tags or [],
                    "collection_name": collection
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'scrape_url_to_kb', {'success': False, 'error': str(e)})

    def update_kb_document_metadata(self, document_id: int, title: str = None, category: str = None, tags: List[str] = None) -> Dict:
        """Update document metadata only"""
        try:
            payload = {}
            if title is not None:
                payload["title"] = title
            if category is not None:
                payload["category"] = category
            if tags is not None:
                payload["tags"] = tags
            
            response = self.session.patch(
                f"{self.base_url}/api/v1/knowledge-base/documents/{document_id}/metadata",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_kb_document_metadata', {'success': False, 'message': str(e)})

    def find_similar_documents(self, document_id: int, limit: int = 5) -> Dict:
        """Find similar documents"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/documents/{document_id}/similar",
                params={'limit': limit},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'find_similar_documents', {'documents': []})

    def get_kb_collections(self) -> Dict:
        """List all collections"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/collections",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_kb_collections', {'collections': []})

    def clear_kb_collection(self, collection_name: str) -> Dict:
        """Clear a collection"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/knowledge-base/collections/{collection_name}",
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'clear_kb_collection', {'success': False, 'message': str(e)})

    def get_kb_embeddings_for_viz(self, collection: str = "documents", limit: int = 100) -> Dict:
        """Get embeddings for visualization"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/embeddings/visualization",
                params={'collection_name': collection, 'limit': limit},
                timeout=20
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_kb_embeddings_for_viz', {'points': []})

    def list_kb_documents_paginated(self, collection: str = "documents", limit: int = 20, offset: int = 0, category: str = None, sort_by: str = "created_at", sort_order: str = "desc") -> Dict:
        """List documents with pagination"""
        try:
            params = {
                'collection_name': collection,
                'limit': limit,
                'offset': offset,
                'sort_by': sort_by,
                'sort_order': sort_order
            }
            if category:
                params['category'] = category
                
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/documents-paginated",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_kb_documents_paginated', {'documents': [], 'total': 0})
    

    def list_calibrations(self, limit: int = 20) -> List[Dict]:
        """List all calibration runs"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/calibration/",
                params={'limit': limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_calibrations', [])

    def get_calibration(self, calibration_run_id: str) -> Dict:
        """Get specific calibration run details"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/calibration/{calibration_run_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_calibration', {})

    def get_active_calibrations(self) -> Dict:
        """Get currently active persona calibrations"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/calibration/personas/active",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_active_calibrations', {'has_calibrations': False, 'personas': []})

    def upload_calibration_data(self, file_content: bytes, filename: str, name: str = "Historical Data Calibration") -> Dict:
        """Upload historical campaign data CSV for calibration"""
        try:
            files = {'file': (filename, file_content, 'text/csv')}
            data = {'name': name}
            response = self.session.post(
                f"{self.base_url}/api/v1/calibration/upload",
                files=files,
                data=data,
                timeout=300  # Calibration can take time
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'upload_calibration_data', {'status': 'error', 'message': str(e)})


    def get_liveness(self) -> Dict:
        """Get liveness probe status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/health/live",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_liveness', {'status': 'unhealthy'})


    def get_campaign_events(
        self,
        campaign_id: str,
        limit: int = 50,
        severity: str = None,
        actionable_only: bool = False,
        include_dismissed: bool = False
    ) -> List[Dict]:
        """Get workflow events for a campaign (timeline/activity feed)"""
        try:
            params = {'limit': limit}
            if severity:
                params['severity'] = severity
            if actionable_only:
                params['actionable_only'] = actionable_only
            if include_dismissed:
                params['include_dismissed'] = include_dismissed

            response = self.session.get(
                f"{self.base_url}/api/v1/events/campaign/{campaign_id}",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'get_campaign_events/{campaign_id}', [])

    def get_active_alerts(self, campaign_id: str = None, severity: str = None) -> Dict:
        """Get active alerts (actionable events that haven't been dismissed)"""
        try:
            params = {}
            if campaign_id:
                params['campaign_id'] = campaign_id
            if severity:
                params['severity'] = severity

            response = self.session.get(
                f"{self.base_url}/api/v1/events/alerts",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_active_alerts', {
                "total_count": 0,
                "critical_count": 0,
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "alerts": {"critical": [], "error": [], "warning": [], "info": []}
            })

    def dismiss_event(self, event_id: str) -> Dict:
        """Dismiss an event/alert"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/events/{event_id}/dismiss",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, f'dismiss_event/{event_id}', {'status': 'error'})

    def get_events_summary(self, days: int = 7) -> Dict:
        """Get summary of events over time period"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/events/summary",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_events_summary', {
                "period_days": days,
                "total_events": 0,
                "actionable_pending": 0,
                "by_severity": {"critical": 0, "error": 0, "warning": 0, "info": 0},
                "by_type": {},
                "most_recent": None
            })

    def cleanup_stale_hitl_alerts(self) -> Dict:
        """Cleanup stale HITL alerts - dismiss resolved but undismissed alerts"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/events/cleanup-stale-hitl-alerts",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'cleanup_stale_hitl_alerts', {'status': 'error'})

    def cleanup_marl_alerts(self) -> Dict:
        """Cleanup stale MARL gating alerts older than 24 hours"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/events/cleanup-marl-alerts",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'cleanup_marl_alerts', {'status': 'error'})

    
    def get_cost_summary(self, days: int = 30, include_mock: Optional[bool] = None) -> Dict:
        """Get cost summary for the specified time period"""
        try:
            params = {'days': days}
            if include_mock is not None:
                params['include_mock'] = include_mock
            response = self.session.get(
                f"{self.base_url}/api/v1/costs/summary",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_cost_summary', {
                "total_cost": 0,
                "total_calls": 0,
                "by_provider": [],
                "by_agent": []
            })
    
    def get_costs_by_model(self, days: int = 30, include_mock: Optional[bool] = None) -> Dict:
        """Get cost breakdown by AI model"""
        try:
            params = {'days': days}
            if include_mock is not None:
                params['include_mock'] = include_mock
            response = self.session.get(
                f"{self.base_url}/api/v1/costs/by-model",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_costs_by_model', {
                "total_cost": 0,
                "models": []
            })
    
    def get_costs_by_agent(self, days: int = 30, include_mock: Optional[bool] = None) -> Dict:
        """Get cost breakdown by agent type"""
        try:
            params = {'days': days}
            if include_mock is not None:
                params['include_mock'] = include_mock
            response = self.session.get(
                f"{self.base_url}/api/v1/costs/by-agent",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_costs_by_agent', {
                "total_cost": 0,
                "agents": []
            })
    
    def get_daily_costs(self, days: int = 30, include_mock: Optional[bool] = None) -> Dict:
        """Get daily cost time series"""
        try:
            params = {'days': days}
            if include_mock is not None:
                params['include_mock'] = include_mock
            response = self.session.get(
                f"{self.base_url}/api/v1/costs/daily",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_daily_costs', {
                "total_cost": 0,
                "daily": [],
                "avg_daily_cost": 0
            })
    
    def get_cost_forecast(self, days: int = 30) -> Dict:
        """Get cost forecast based on historical burn rate"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/costs/forecast",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_cost_forecast', {
                "daily_burn_rate": 0,
                "weekly_forecast": 0,
                "monthly_forecast": 0,
                "forecast": []
            })


    def get_funnel_overview(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get full funnel attribution overview"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = self.session.get(
                f"{self.base_url}/api/v1/funnel/attribution/overview",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_funnel_overview', {
                "funnel_stages": {},
                "metrics": {},
                "campaigns": []
            })

    def get_calendar_bookings(self, start_date: str = None, end_date: str = None, limit: int = 50) -> Dict:
        """Get Cal.com bookings"""
        try:
            params = {'limit': limit}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = self.session.get(
                f"{self.base_url}/api/v1/funnel/calendar/bookings",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_calendar_bookings', {"bookings": [], "count": 0})

    def get_calendar_metrics(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get Cal.com booking metrics"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = self.session.get(
                f"{self.base_url}/api/v1/funnel/calendar/metrics",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_calendar_metrics', {"metrics": {}})

    def get_hubspot_deals(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get HubSpot deals and pipeline status"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = self.session.get(
                f"{self.base_url}/api/v1/funnel/hubspot/deals",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_hubspot_deals', {"deals": [], "by_stage": {}})

    def get_hubspot_lead_quality(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get HubSpot lead quality metrics"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = self.session.get(
                f"{self.base_url}/api/v1/funnel/hubspot/lead-quality",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_hubspot_lead_quality', {"avg_score": 0, "distribution": {}})

    def get_attribution_by_campaign(self, start_date: str = None, end_date: str = None) -> Dict:
        """Get attribution data by campaign"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = self.session.get(
                f"{self.base_url}/api/v1/funnel/attribution/by-campaign",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_attribution_by_campaign', {"campaigns": []})


    def get_simulation_validation_overview(self) -> Dict:
        """Get simulation validation overview"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/simulation/validation/overview",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_simulation_validation_overview', {
                "overall_mape": 0,
                "accuracy_percentage": 0,
                "meets_target": False,
                "total_validations": 0,
                "campaigns_validated": 0
            })

    def get_simulation_validation_by_metric(self) -> Dict:
        """Get simulation validation by metric"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/simulation/validation/by-metric",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_simulation_validation_by_metric', {"metrics": []})

    def get_campaign_validations(self) -> Dict:
        """Get campaign-level validation results"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/simulation/validation/campaigns",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_campaign_validations', {"campaigns": []})

    def run_simulation_validation(self) -> Dict:
        """Run new simulation validation"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/simulation/validation/run",
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'run_simulation_validation', {'status': 'error'})

    def get_persona_calibrations(self) -> Dict:
        """Get persona calibration status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/calibration/personas/active",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_persona_calibrations', {"has_calibrations": False, "personas": []})

    def get_weekly_report(self) -> Dict:
        """Get the latest weekly learning report including best/worst hooks and recommendations"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/weekly-report/latest",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_weekly_report', {
                "report_id": None,
                "week_start": None,
                "week_end": None,
                "best_hooks": [],
                "worst_hooks": [],
                "platform_performance": [],
                "persona_performance": [],
                "bandit_insights": {},
                "recommendations": [],
                "generated_at": None
            })

    def generate_weekly_report(self) -> Dict:
        """Trigger generation of a new weekly learning report"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/analytics/weekly-report/generate",
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'generate_weekly_report', {'status': 'error', 'message': str(e)})

    def generate_weekly_report_custom(self, start_date: str, end_date: str) -> Dict:
        """Generate a weekly report for a custom date range"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/analytics/weekly-report/generate-custom",
                json={"start_date": start_date, "end_date": end_date},
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'generate_weekly_report_custom', {'status': 'error', 'message': str(e)})

    def get_weekly_report_history(self, limit: int = 10) -> List:
        """Get historical weekly reports"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/weekly-report/history",
                params={"limit": limit},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_error(e, 'get_weekly_report_history', [])
            return []

    def get_weekly_report_scheduler_status(self) -> Dict:
        """Get weekly report scheduler status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/operations/scheduler-status",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return {
                "running": data.get("status") == "running",
                "next_scheduled_run": data.get("next_weekly_report", "Unknown")
            }
        except Exception as e:
            return {"running": False, "next_scheduled_run": "Unknown"}


    def get_override_rate(self) -> Dict:
        """Get current human override rate metric and trend"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/governance-metrics/override-rate",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return {
                "current_rate": data.get("override_rate", 0.0),
                "target_rate": data.get("target", 5.0),
                "total_reviews": data.get("total_reviews", 0),
                "total_overrides": data.get("override_count", 0),
                "passes_target": data.get("meets_target", True),
                "approved_count": data.get("approved_count", 0),
                "rejected_count": data.get("rejected_count", 0),
                "modified_count": data.get("modified_count", 0),
                "status": data.get("status", "N/A"),
                "gap_to_target": data.get("gap_to_target", 0.0)
            }
        except Exception as e:
            return self._handle_error(e, 'get_override_rate', {
                "current_rate": 0.0,
                "target_rate": 5.0,
                "total_reviews": 0,
                "total_overrides": 0,
                "passes_target": True,
                "trend": "stable"
            })

    def get_override_rate_trend(self, days: int = 30) -> Dict:
        """Get human override rate trend over time"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/governance-metrics/override-trend",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return {"trend_data": data, "days": days}
            return data
        except Exception as e:
            return self._handle_error(e, 'get_override_rate_trend', {
                "days": days,
                "trend_data": [],
                "avg_rate": 0.0
            })

    def get_semantic_cache_metrics(self) -> Dict:
        """Get semantic cache hit rate and related metrics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/costs/cache-metrics",
                timeout=30  # Increased timeout for cache initialization
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_semantic_cache_metrics', {
                "hit_rate": 0.0,
                "hit_rate_target": 20.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_queries": 0,
                "meets_target": False,
                "estimated_cost_savings": 0.0
            })

    def get_simulation_accuracy(self, days: int = 30) -> Dict:
        """Get simulation-to-live accuracy metrics (MAPE)"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/simulation-accuracy/summary",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_simulation_accuracy', {
                "average_mape": None,
                "target_mape": 10.0,
                "passes_target": False,
                "campaigns_analyzed": 0
            })

    def get_simulation_accuracy_trend(self, days: int = 30) -> List:
        """Get simulation accuracy trend over time"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/simulation-accuracy/trend",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_simulation_accuracy_trend', [])

    def get_costs_by_campaign(self, days: int = 30) -> List:
        """Get cost breakdown by campaign"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/costs/by-campaign",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_costs_by_campaign', [])

    def get_experiment_regret(self, experiment_id: str) -> Dict:
        """Get cumulative regret for an experiment"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/experiments/{experiment_id}/regret",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_experiment_regret', {
                "cumulative_regret": 0.0,
                "regret_history": [],
                "optimal_arm": None
            })

    def get_show_rate(self, days: int = 30) -> Dict:
        """Get meeting show rate from Cal.com data"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/funnel/calcom/show-rate",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_show_rate', {
                "show_rate": 0.0,
                "total_bookings": 0,
                "shows": 0,
                "no_shows": 0
            })

    def get_review_time_stats(self, days: int = 30) -> Dict:
        """Get human review time statistics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/analytics/governance-metrics/review-time",
                params={'days': days},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_review_time_stats', {
                "avg_review_duration_seconds": 0,
                "avg_review_duration_minutes": 0,
                "total_reviews": 0,
                "estimated_time_saved_hours": 0,
                "efficiency_gain_pct": 0
            })

    def get_mlflow_models(self) -> List:
        """Get registered models from MLflow"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/mlflow/models",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_mlflow_models', [])

    def get_mlflow_status(self) -> Dict:
        """Get MLflow connection status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/mlflow/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_mlflow_status', {
                "connected": False,
                "tracking_uri": None
            })

    def get_claim_library(self) -> List:
        """Get all claims from the claim library"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/governance/claims",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_claim_library', [])

    def create_claim(self, claim_data: Dict) -> Dict:
        """Create a new claim in the library"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/governance/claims",
                json=claim_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'create_claim', {'error': str(e)})

    def delete_claim(self, claim_id: str) -> Dict:
        """Delete a claim from the library"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/governance/claims/{claim_id}",
                timeout=10
            )
            response.raise_for_status()
            return {"status": "deleted"}
        except Exception as e:
            return self._handle_error(e, 'delete_claim', {'error': str(e)})


    def get_integration_status(self) -> Dict:
        """Get status of external integrations (Cal.com, HubSpot, Ollama, OpenAI)"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/config/integrations/status",
                timeout=15  # Longer timeout for multiple integration checks
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_integration_status', {
                "calcom": {"connected": False, "configured": False, "error": str(e)},
                "hubspot": {"connected": False, "configured": False, "error": str(e)},
                "ollama": {"connected": False, "configured": False, "error": str(e)},
                "openai": {"connected": False, "configured": False, "error": str(e)}
            })

    def clone_campaign(self, campaign_id: str, new_name: str = None) -> Dict:
        """Clone an existing campaign"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/campaigns/{campaign_id}/clone",
                json={"new_name": new_name} if new_name else {},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'clone_campaign', {'error': str(e)})

    def search_knowledge_base(self, query: str, limit: int = 10) -> List:
        """Search the knowledge base using semantic search"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/knowledge-base/search",
                params={'query': query, 'limit': limit},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'search_knowledge_base', [])


    def get_mock_mode_status(self) -> Dict:
        """Get current mock mode status for displaying indicators in the dashboard"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/config/mock-mode",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_mock_mode_status', {
                'mock_mode_enabled': False,
                'settings': {},
                'description': 'Unable to fetch mock mode status'
            })


    def get_configurations(self, category: str = None) -> List[Dict]:
        """Get system configurations from database
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of configuration dicts with key, value, description
        """
        try:
            if category:
                url = f"{self.base_url}/api/v1/config/category/{category}"
            else:
                all_configs = []
                categories_resp = self.session.get(
                    f"{self.base_url}/api/v1/config/categories",
                    timeout=10
                )
                if categories_resp.status_code == 200:
                    categories = categories_resp.json()
                    for cat in categories:
                        cat_name = cat.get('name') if isinstance(cat, dict) else cat
                        cat_resp = self.session.get(
                            f"{self.base_url}/api/v1/config/category/{cat_name}",
                            timeout=10
                        )
                        if cat_resp.status_code == 200:
                            all_configs.extend(cat_resp.json())
                return all_configs
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_configurations', [])
    
    def get_config_value(self, key: str) -> Optional[str]:
        """Get a single configuration value by key"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/config/value/{key}",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('value')
            return None
        except Exception:
            return None

    def is_config_key_set(self, key: str) -> bool:
        """Check if a configuration key has a non-empty value (works for secrets too)"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/config/value/{key}",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('is_secret'):
                    return bool(data.get('display_value', ''))
                return data.get('value') is not None and data.get('value') != ''
            return False
        except Exception:
            return False

    
    def get_data_config_summary(self) -> Dict:
        """Get summary of all data configuration files"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/summary",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_data_config_summary', {})
    
    def list_claims(self, limit: int = 100, offset: int = 0, claim_type: str = None, 
                    confidence_min: int = None, search: str = None) -> Dict:
        """List claims with optional filtering"""
        try:
            params = {"limit": limit, "offset": offset}
            if claim_type:
                params["claim_type"] = claim_type
            if confidence_min:
                params["confidence_min"] = confidence_min
            if search:
                params["search"] = search
            
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/claims",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_claims', {"claims": [], "total": 0})
    
    def get_claim(self, claim_id: str) -> Dict:
        """Get a specific claim by ID"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/claims/{claim_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_claim', {})
    
    def create_claim(self, claim_data: Dict) -> Dict:
        """Create a new claim"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/data-config/claims",
                json=claim_data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'create_claim', {"success": False, "error": str(e)})
    
    def update_claim(self, claim_id: str, claim_data: Dict) -> Dict:
        """Update an existing claim"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/v1/data-config/claims/{claim_id}",
                json=claim_data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_claim', {"success": False, "error": str(e)})
    
    def delete_claim(self, claim_id: str) -> Dict:
        """Delete a claim"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/data-config/claims/{claim_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'delete_claim', {"success": False, "error": str(e)})
    
    def get_claim_types(self) -> List[str]:
        """Get list of claim types"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/claims/types/list",
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("types", [])
        except Exception as e:
            return self._handle_error(e, 'get_claim_types', [])
    
    def get_brand_voice(self) -> Dict:
        """Get the complete brand voice configuration"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/brand-voice",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_brand_voice', {})
    
    def update_brand_voice(self, brand_voice: Dict) -> Dict:
        """Update the complete brand voice configuration"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/v1/data-config/brand-voice",
                json=brand_voice,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_brand_voice', {"success": False, "error": str(e)})
    
    def get_brand_voice_section(self, section: str) -> Dict:
        """Get a specific section of brand voice"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/brand-voice/{section}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_brand_voice_section', {})
    
    def update_brand_voice_section(self, section: str, data: Dict) -> Dict:
        """Update a specific section of brand voice"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/v1/data-config/brand-voice/{section}",
                json=data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_brand_voice_section', {"success": False, "error": str(e)})
    
    def list_competitors(self, category: str = None, search: str = None) -> Dict:
        """List competitors with optional filtering"""
        try:
            params = {}
            if category:
                params["category"] = category
            if search:
                params["search"] = search
            
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/competitors",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_competitors', {"competitors": [], "total": 0})
    
    def get_competitor(self, name: str) -> Dict:
        """Get a specific competitor by name"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/competitors/{name}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_competitor', {})
    
    def create_competitor(self, competitor_data: Dict) -> Dict:
        """Create a new competitor"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/data-config/competitors",
                json=competitor_data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'create_competitor', {"success": False, "error": str(e)})
    
    def update_competitor(self, name: str, competitor_data: Dict) -> Dict:
        """Update an existing competitor"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/v1/data-config/competitors/{name}",
                json=competitor_data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_competitor', {"success": False, "error": str(e)})
    
    def delete_competitor(self, name: str) -> Dict:
        """Delete a competitor"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/data-config/competitors/{name}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'delete_competitor', {"success": False, "error": str(e)})
    
    def get_product_catalog(self) -> Dict:
        """Get the complete product catalog"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/products",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_product_catalog', {})
    
    def update_product_catalog(self, catalog: Dict) -> Dict:
        """Update the complete product catalog"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/v1/data-config/products",
                json=catalog,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_product_catalog', {"success": False, "error": str(e)})
    
    def list_product_modules(self) -> Dict:
        """Get list of product modules"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/products/modules",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_product_modules', {"modules": [], "total": 0})
    
    def get_product_module(self, module_id: str) -> Dict:
        """Get a specific product module"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/products/modules/{module_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_product_module', {})
    
    def update_product_module(self, module_id: str, module_data: Dict) -> Dict:
        """Update a specific product module"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/v1/data-config/products/modules/{module_id}",
                json=module_data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_product_module', {"success": False, "error": str(e)})
    
    def list_product_packages(self) -> Dict:
        """Get list of product packages"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/products/packages",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_product_packages', {"packages": [], "total": 0})
    
    def get_product_governance(self) -> Dict:
        """Get product governance rules"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/products/governance",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'get_product_governance', {})
    
    def update_product_governance(self, governance: Dict) -> Dict:
        """Update product governance rules"""
        try:
            response = self.session.put(
                f"{self.base_url}/api/v1/data-config/products/governance",
                json=governance,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'update_product_governance', {"success": False, "error": str(e)})


    def list_data_config_versions(self, config_name: str = None, limit: int = 50) -> Dict:
        """List available data config versions"""
        try:
            params = {"limit": limit}
            if config_name:
                params["config_name"] = config_name
            response = self.session.get(
                f"{self.base_url}/api/v1/data-config/versions",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'list_data_config_versions', {"versions": [], "total": 0})

    def restore_data_config_version(self, version_id: str) -> Dict:
        """Restore a data config from a previous version"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/data-config/versions/{version_id}/restore"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'restore_data_config_version', {"success": False, "error": str(e)})

    def delete_data_config_version(self, version_id: str) -> Dict:
        """Delete a data config version"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/data-config/versions/{version_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'delete_data_config_version', {"success": False, "error": str(e)})

    def create_data_config_backup(self, config_name: str = None) -> Dict:
        """Create a backup of data config files"""
        try:
            if config_name:
                url = f"{self.base_url}/api/v1/data-config/backup/{config_name}"
            else:
                url = f"{self.base_url}/api/v1/data-config/backup"
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return self._handle_error(e, 'create_data_config_backup', {"success": False, "error": str(e)})
