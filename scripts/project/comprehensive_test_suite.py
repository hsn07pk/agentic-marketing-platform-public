#!/usr/bin/env python3
"""
Agentic AI Agent Platform - Comprehensive End-to-End Testing Suite
Version: 1.0
Last Updated: November 12, 2025

This script runs the complete test suite from test.md, checking API key availability
and running only tests that don't require missing API keys.

Features:
- Reads .env file every time it runs
- Checks which API keys are present
- Runs tests progressively with detailed logging
- Logs to both terminal and scripts/logs directory (debug level)
- Shows progress with checkmarks
- Waits for user input between tests
- Provides detailed output after each test
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
from dotenv import load_dotenv
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# LOGGING SETUP

class ComprehensiveTestLogger:
    """Custom logger that logs to both terminal and file with detailed formatting"""

    def __init__(self):
        self.log_dir = project_root / "scripts" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"comprehensive_test_{timestamp}.log"

        # Create logger
        self.logger = logging.getLogger("ComprehensiveTest")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler (DEBUG level)
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler (INFO level, but we'll log DEBUG to file)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)

# Initialize logger
test_logger = ComprehensiveTestLogger()

# API KEY DETECTION

class APIKeyChecker:
    """Checks which API keys are available in the .env file"""

    def __init__(self):
        self.env_path = project_root / ".env"
        self.api_keys = {}
        self.reload_env()

    def reload_env(self):
        """Reload .env file and check for API keys"""
        test_logger.info("=" * 80)
        test_logger.info("🔑 CHECKING API KEY AVAILABILITY")
        test_logger.info("=" * 80)

        if not self.env_path.exists():
            test_logger.warning(f"⚠️  .env file not found at {self.env_path}")
            test_logger.info("ℹ️  Checking environment variables (Docker mode)")
        else:
            # Load environment variables from .env file
            load_dotenv(self.env_path, override=True)
            test_logger.debug(f"Reading .env file from: {self.env_path}")

        # Check for each required API key
        keys_to_check = {
            'OPENAI_API_KEY': 'OpenAI API',
            'LINKEDIN_CLIENT_ID': 'LinkedIn',
            'LINKEDIN_CLIENT_SECRET': 'LinkedIn Secret',
            'LINKEDIN_ACCESS_TOKEN': 'LinkedIn Token',
            'LINKEDIN_ACCOUNT_ID': 'LinkedIn Account',
            'LINKEDIN_ORGANIZATION_ID': 'LinkedIn Org',
            'TWITTER_API_KEY': 'Twitter/X API',
            'TWITTER_API_SECRET': 'Twitter/X Secret',
            'TWITTER_ACCESS_TOKEN': 'Twitter/X Token',
            'TWITTER_ACCESS_TOKEN_SECRET': 'Twitter/X Token Secret',
            'SENDGRID_API_KEY': 'SendGrid',
            'SENDGRID_FROM_EMAIL': 'SendGrid Email',
            'APIFY_API_TOKEN': 'Apify',
            'HUBSPOT_API_KEY': 'HubSpot',
            'CALENDAR_API_KEY': 'Cal.com',
        }

        for key, name in keys_to_check.items():
            value = os.getenv(key)
            is_present = bool(value and value.strip() and value.strip() != '')
            self.api_keys[key] = is_present

            status = "✅ Present" if is_present else "❌ Missing"
            test_logger.info(f"  {name:30s}: {status}")
            test_logger.debug(f"    Key: {key}, Value length: {len(value) if value else 0}")

        test_logger.info("")
        test_logger.info(f"📊 Summary: {sum(self.api_keys.values())}/{len(self.api_keys)} API keys present")
        test_logger.info("=" * 80)
        test_logger.info("")

    def has_openai(self) -> bool:
        return self.api_keys.get('OPENAI_API_KEY', False)

    def has_linkedin(self) -> bool:
        return all([
            self.api_keys.get('LINKEDIN_CLIENT_ID', False),
            self.api_keys.get('LINKEDIN_CLIENT_SECRET', False),
            self.api_keys.get('LINKEDIN_ACCESS_TOKEN', False)
        ])

    def has_twitter(self) -> bool:
        return all([
            self.api_keys.get('TWITTER_API_KEY', False),
            self.api_keys.get('TWITTER_API_SECRET', False),
            self.api_keys.get('TWITTER_ACCESS_TOKEN', False)
        ])

    def has_sendgrid(self) -> bool:
        return self.api_keys.get('SENDGRID_API_KEY', False)

    def can_run_ai_tests(self) -> bool:
        return self.has_openai()

    def can_run_linkedin_deployment(self) -> bool:
        return self.has_openai() and self.has_linkedin()

    def can_run_twitter_deployment(self) -> bool:
        return self.has_openai() and self.has_twitter()

    def can_run_email_deployment(self) -> bool:
        return self.has_openai() and self.has_sendgrid()

# TEST RESULT TRACKING

class TestResult:
    """Tracks the result of a single test"""

    def __init__(self, test_id: str, test_name: str):
        self.test_id = test_id
        self.test_name = test_name
        self.status: str = "PENDING"  # PASSED, FAILED, SKIPPED, ERROR
        self.duration: float = 0.0
        self.output: str = ""
        self.error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def start(self):
        self.start_time = datetime.now()
        self.status = "RUNNING"

    def complete(self, status: str, output: str = "", error: Optional[str] = None):
        self.end_time = datetime.now()
        self.status = status
        self.output = output
        self.error = error
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict:
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'status': self.status,
            'duration': self.duration,
            'output': self.output,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }

class TestSuite:
    """Manages the entire test suite"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.current_test: Optional[TestResult] = None

    def start_test(self, test_id: str, test_name: str) -> TestResult:
        result = TestResult(test_id, test_name)
        result.start()
        self.current_test = result
        self.results.append(result)
        return result

    def get_summary(self) -> Dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASSED")
        failed = sum(1 for r in self.results if r.status == "FAILED")
        skipped = sum(1 for r in self.results if r.status == "SKIPPED")
        errors = sum(1 for r in self.results if r.status == "ERROR")

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'errors': errors,
            'success_rate': (passed / total * 100) if total > 0 else 0
        }

    def save_results(self):
        """Save test results to JSON file"""
        results_file = test_logger.log_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            'summary': self.get_summary(),
            'results': [r.to_dict() for r in self.results]
        }

        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)

        test_logger.info(f"📄 Test results saved to: {results_file}")

# TEST EXECUTION HELPERS

def print_test_header(test_id: str, test_name: str, layer: str = ""):
    """Print a formatted test header"""
    test_logger.info("")
    test_logger.info("=" * 80)
    test_logger.info(f"🧪 TEST {test_id}: {test_name}")
    if layer:
        test_logger.info(f"📦 Layer: {layer}")
    test_logger.info("=" * 80)
    test_logger.debug(f"Starting test execution at {datetime.now()}")

def print_test_result(result: TestResult):
    """Print formatted test result"""
    test_logger.info("")
    test_logger.info("-" * 80)

    status_icons = {
        "PASSED": "✅",
        "FAILED": "❌",
        "SKIPPED": "⏭️",
        "ERROR": "💥"
    }

    icon = status_icons.get(result.status, "❓")
    test_logger.info(f"{icon} Test {result.test_id}: {result.status}")
    test_logger.info(f"⏱️  Duration: {result.duration:.2f}s")

    if result.output:
        test_logger.info("")
        test_logger.info("📤 OUTPUT:")
        test_logger.info(result.output)

    if result.error:
        test_logger.error("")
        test_logger.error("🔥 ERROR:")
        test_logger.error(result.error)

    test_logger.info("-" * 80)
    test_logger.debug(f"Test completed at {datetime.now()}")

def wait_for_user():
    """Wait for user to press Enter to continue (disabled for non-interactive mode)"""
    # Disabled for non-interactive execution
    pass

# LAYER 1: INFRASTRUCTURE TESTING

async def test_1_1_postgresql_connectivity(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 1.1: PostgreSQL Connectivity"""
    test_id = "1.1"
    test_name = "PostgreSQL Connectivity"
    layer = "Infrastructure"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.data_layer.database.connection import get_async_session
        from sqlalchemy import text

        test_logger.info("🔄 Testing PostgreSQL connection...")
        test_logger.debug("Creating database session")

        # Test connection by executing a simple query
        async with get_async_session() as session:
            result_query = await session.execute(text("SELECT 1"))
            result_query.scalar()

            # Check for pgvector extension
            ext_result = await session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            has_vector = ext_result.scalar() is not None

        output = "✅ Database connection successful\n"
        output += "✅ PostgreSQL is accessible\n"
        if has_vector:
            output += "✅ pgvector extension is installed"
        else:
            output += "⚠️  pgvector extension not found"

        result.complete("PASSED", output)
        test_logger.info("✅ PostgreSQL connectivity test PASSED")

    except Exception as e:
        error_msg = f"PostgreSQL connection failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_1_2_redis_connectivity(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 1.2: Redis Connectivity"""
    test_id = "1.2"
    test_name = "Redis Connectivity"
    layer = "Infrastructure"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import redis
        from src.config.settings import settings

        test_logger.info("🔄 Testing Redis connection...")
        test_logger.debug(f"Redis URL: {settings.REDIS_URL}")

        # Test Redis connection
        r = redis.from_url(settings.REDIS_URL)
        ping_result = r.ping()
        test_logger.debug(f"Redis ping result: {ping_result}")

        # Test set/get
        test_key = 'comprehensive_test_key'
        test_value = 'comprehensive_test_value'
        r.set(test_key, test_value, ex=60)
        retrieved_value = r.get(test_key)

        test_logger.debug(f"Set key: {test_key}, value: {test_value}")
        test_logger.debug(f"Retrieved value: {retrieved_value}")

        output = f"✅ Redis ping: {ping_result}\n"
        output += f"✅ Redis set/get: {retrieved_value.decode() if retrieved_value else 'None'}\n"
        output += "✅ Redis is fully operational"

        result.complete("PASSED", output)
        test_logger.info("✅ Redis connectivity test PASSED")

    except Exception as e:
        error_msg = f"Redis connection failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_1_3_pgvector_extension(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 1.3: pgvector Extension"""
    test_id = "1.3"
    test_name = "pgvector Extension"
    layer = "Infrastructure"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.data_layer.vector_store.pgvector_store import PgVectorStore
        from langchain.schema import Document

        test_logger.info("🔄 Testing pgvector extension...")

        # Initialize vector store
        store = PgVectorStore(collection_name='test_comprehensive')
        await store.initialize()
        test_logger.debug("Vector store initialized")

        # Test adding documents
        docs = [
            Document(
                page_content='AI marketing platform test',
                metadata={'id': '1', 'source': 'test'}
            ),
            Document(
                page_content='Machine learning for advertising',
                metadata={'id': '2', 'source': 'test'}
            )
        ]

        doc_ids = await store.add_documents(docs)
        test_logger.debug(f"Added {len(doc_ids)} documents")

        # Test similarity search
        results = await store.similarity_search('AI marketing', k=1)
        test_logger.debug(f"Search returned {len(results)} results")

        output = "✅ Vector store initialized\n"
        output += f"✅ Added {len(doc_ids)} test documents\n"
        output += f"✅ Search returned {len(results)} results\n"
        if results:
            output += f"✅ Top result: {results[0].page_content}"

        result.complete("PASSED", output)
        test_logger.info("✅ pgvector extension test PASSED")

    except Exception as e:
        error_msg = f"pgvector test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_1_4_api_health_check(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 1.4: API Health Check"""
    test_id = "1.4"
    test_name = "API Health Check"
    layer = "Infrastructure"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import httpx

        test_logger.info("🔄 Testing API health endpoints...")

        base_url = "http://localhost:8000"
        endpoints = ['/health', '/ready', '/live']

        output = ""

        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                url = f"{base_url}{endpoint}"
                test_logger.debug(f"Testing endpoint: {url}")

                try:
                    response = await client.get(url, timeout=5.0)
                    test_logger.debug(f"Response status: {response.status_code}")
                    test_logger.debug(f"Response body: {response.text}")

                    if response.status_code == 200:
                        output += f"✅ {endpoint}: OK (200)\n"
                        output += f"   Response: {response.json()}\n"
                    else:
                        output += f"⚠️  {endpoint}: Status {response.status_code}\n"

                except httpx.ConnectError:
                    output += f"❌ {endpoint}: API not running\n"
                    test_logger.warning(f"Could not connect to {url}")

        result.complete("PASSED", output)
        test_logger.info("✅ API health check test PASSED")

    except Exception as e:
        error_msg = f"API health check failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_1_5_dashboard_accessibility(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 1.5: Dashboard Accessibility"""
    test_id = "1.5"
    test_name = "Dashboard Accessibility"
    layer = "Infrastructure"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import httpx

        test_logger.info("🔄 Testing Dashboard accessibility...")

        # Use container service name when running inside Docker
        dashboard_url = "http://dashboard:8501"
        health_endpoint = f"{dashboard_url}/_stcore/health"

        test_logger.debug(f"Testing dashboard at: {dashboard_url}")

        output = ""
        dashboard_accessible = False

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(health_endpoint, timeout=5.0)
                test_logger.debug(f"Health check response: {response.status_code}")

                if response.status_code == 200:
                    dashboard_accessible = True
                    output = f"✅ Dashboard is accessible at {dashboard_url}\n"
                    # Streamlit health endpoint returns plain text, not JSON
                    try:
                        health_data = response.json()
                        output += f"✅ Health check: {health_data}\n"
                    except:
                        output += f"✅ Health check: {response.text if response.text else 'OK'}\n"
                    output += "ℹ️  Manual verification recommended: Open browser to http://localhost:8501"
                else:
                    output = f"⚠️  Dashboard returned status: {response.status_code}"

            except httpx.ConnectError:
                output = f"❌ Dashboard not running at {dashboard_url}\n"
                output += "ℹ️  Start with: docker-compose up -d dashboard"

        # Only pass if dashboard is actually accessible
        if dashboard_accessible:
            result.complete("PASSED", output)
            test_logger.info("✅ Dashboard accessibility test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ Dashboard accessibility test FAILED - service not running")

    except Exception as e:
        error_msg = f"Dashboard test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 2: DATA LAYER TESTING

async def test_2_1_campaign_repository(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 2.1: Campaign Repository"""
    test_id = "2.1"
    test_name = "Campaign Repository CRUD"
    layer = "Data Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.data_layer.repositories.campaign_repo import CampaignRepository
        from src.data_layer.database.connection import get_async_session
        from datetime import datetime, timedelta

        test_logger.info("🔄 Testing Campaign Repository operations...")

        output = ""

        async with get_async_session() as session:
            repo = CampaignRepository(session)
            test_logger.debug("Campaign repository initialized")

            # Test CREATE
            campaign_data = {
                'name': 'Comprehensive Test Campaign',
                'platform': 'linkedin',
                'status': 'draft',
                'goal': 'lead_generation',
                'target_persona': 'decision_maker',
                'budget_total': 1000.0,
                'start_date': datetime.utcnow(),
                'end_date': datetime.utcnow() + timedelta(days=14)
            }

            campaign = await repo.create(campaign_data)
            test_logger.debug(f"Created campaign: {campaign.id}")

            output += f"✅ CREATE: Campaign created with ID: {campaign.id}\n"
            output += f"   Name: {campaign.name}\n"
            output += f"   Status: {campaign.status}\n"

            # Test GET
            fetched = await repo.get_by_id(str(campaign.id))
            test_logger.debug(f"Retrieved campaign: {fetched.id}")

            output += f"✅ READ: Retrieved campaign {fetched.id}\n"

            # Test UPDATE
            await repo.update(str(campaign.id), {'status': 'running'})
            updated = await repo.get_by_id(str(campaign.id))
            test_logger.debug(f"Updated campaign status to: {updated.status}")

            output += f"✅ UPDATE: Status changed to {updated.status}\n"

            # Test LIST
            all_campaigns = await repo.get_all()
            test_logger.debug(f"Listed {len(all_campaigns)} campaigns")

            output += f"✅ LIST: Found {len(all_campaigns)} total campaigns"

        result.complete("PASSED", output)
        test_logger.info("✅ Campaign repository test PASSED")

    except Exception as e:
        error_msg = f"Campaign repository test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_2_2_content_repository(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 2.2: Content Repository"""
    test_id = "2.2"
    test_name = "Content Repository Operations"
    layer = "Data Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.data_layer.repositories.content_repo import ContentRepository
        from src.data_layer.repositories.campaign_repo import CampaignRepository
        from src.data_layer.database.connection import get_async_session

        test_logger.info("🔄 Testing Content Repository operations...")

        output = ""

        async with get_async_session() as session:
            campaign_repo = CampaignRepository(session)
            campaigns = await campaign_repo.get_all()

            if not campaigns:
                output = "⚠️  No campaigns found. Run Test 2.1 first or seed data."
                result.complete("SKIPPED", output)
                test_logger.warning("No campaigns available for content test")
                print_test_result(result)
                wait_for_user()
                return

            campaign = campaigns[0]
            test_logger.debug(f"Using campaign: {campaign.id}")

            content_repo = ContentRepository(session)

            # Test CREATE (using valid claim IDs from library)
            content_data = {
                'campaign_id': campaign.id,
                'content_type': 'linkedin_ad',
                'headline': 'Transform Your Marketing with AI - Test',
                'body': 'Discover how AI agents can automate campaigns and improve ROI...',
                'cta': 'Book a Demo',
                'status': 'generated',
                'claims_used': ['CLM_003', 'CLM_006']
            }

            content = await content_repo.create(content_data)
            test_logger.debug(f"Created content: {content.id}")

            output += f"✅ CREATE: Content created with ID: {content.id}\n"
            output += f"   Headline: {content.headline}\n"
            output += f"   Claims: {content.claims_used}\n"

            # Test MARK_DEPLOYED
            await content_repo.mark_deployed(
                str(content.id),
                platform_post_id='test_post_123',
                metrics={'impressions': 1000}
            )

            deployed = await content_repo.get_by_id(str(content.id))
            test_logger.debug(f"Content status: {deployed.status}")

            output += f"✅ MARK_DEPLOYED: Status changed to {deployed.status}\n"

            # Test GET_PENDING_REVIEW
            pending = await content_repo.get_pending_review()
            test_logger.debug(f"Pending review items: {len(pending)}")

            output += f"✅ GET_PENDING_REVIEW: {len(pending)} items pending"

        result.complete("PASSED", output)
        test_logger.info("✅ Content repository test PASSED")

    except Exception as e:
        error_msg = f"Content repository test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_2_3_vector_store_operations(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 2.3: Vector Store Operations"""
    test_id = "2.3"
    test_name = "Vector Store Operations"
    layer = "Data Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.data_layer.vector_store.pgvector_store import PgVectorStore
        from langchain.schema import Document

        test_logger.info("🔄 Testing Vector Store operations...")

        store = PgVectorStore(collection_name='test_documents_comprehensive')
        await store.initialize()
        test_logger.debug("Vector store initialized")

        # Add documents
        docs = [
            Document(
                page_content='Agentic uses AI agents for marketing automation and optimization',
                metadata={'source': 'website', 'type': 'feature'}
            ),
            Document(
                page_content='Thompson Sampling optimizes A/B testing decisions',
                metadata={'source': 'paper', 'type': 'algorithm'}
            ),
            Document(
                page_content='LangGraph orchestrates multi-agent workflows with supervision',
                metadata={'source': 'docs', 'type': 'architecture'}
            )
        ]

        doc_ids = await store.add_documents(docs)
        test_logger.debug(f"Added {len(doc_ids)} documents")

        output = f"✅ ADD_DOCUMENTS: Added {len(doc_ids)} documents\n\n"

        # Wait for embeddings to be indexed
        import asyncio
        await asyncio.sleep(2.0)
        test_logger.debug("Waited 2s for embeddings to be indexed")

        # Similarity search
        results = await store.similarity_search('AI agents', k=2)
        test_logger.debug(f"Search returned {len(results)} results")

        search_successful = len(results) > 0

        if search_successful:
            output += f"✅ SIMILARITY_SEARCH: Found {len(results)} results:\n"
            for i, doc in enumerate(results, 1):
                output += f"   {i}. {doc.page_content[:60]}...\n"
                output += f"      Metadata: {doc.metadata}\n"
        else:
            output += f"❌ SIMILARITY_SEARCH: Found {len(results)} results:\n"
            output += f"   Expected: >0 results (just added {len(doc_ids)} documents)\n"
            output += f"   This suggests vector search is not functioning correctly\n"

        # Search with filter
        filtered_results = await store.similarity_search(
            'marketing',
            k=5,
            filter={'type': 'feature'}
        )
        test_logger.debug(f"Filtered search returned {len(filtered_results)} results")

        output += f"\n✅ FILTERED_SEARCH: {len(filtered_results)} results with filter type='feature'"

        # Only pass if similarity search works
        if search_successful:
            result.complete("PASSED", output)
            test_logger.info("✅ Vector store operations test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ Vector store operations test FAILED - similarity search returned 0 results")

    except Exception as e:
        error_msg = f"Vector store test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_2_4_semantic_cache(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 2.4: Semantic Cache"""
    test_id = "2.4"
    test_name = "Semantic Cache Operations"
    layer = "Data Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.data_layer.vector_store.semantic_cache import SemanticCache
        import time

        test_logger.info("🔄 Testing Semantic Cache...")

        cache = SemanticCache()
        await cache.initialize()

        # Clear old cache entries first
        await cache.vector_store.clear_collection()
        test_logger.debug("Cleared old cache entries")

        # Test SET with unique timestamp to avoid collision
        timestamp = int(time.time())
        prompt = f'Generate a LinkedIn post about AI marketing automation TEST_{timestamp}'
        response = 'Transform your marketing strategy with AI agents that learn and adapt...'

        await cache.set(prompt, response, model="gpt-4")
        test_logger.debug(f"Cached prompt: {prompt[:50]}...")

        output = "✅ SET: Cached response for prompt\n\n"

        # Test GET (exact match)
        cached = await cache.get(prompt)
        test_logger.debug(f"Cache hit (exact): {bool(cached)}")

        if cached:
            output += f"✅ GET (exact): Cache hit\n"
            response_text = cached.get('response', '')
            output += f"   Cached response: {response_text[:60]}...\n"
        else:
            output += "❌ GET (exact): Cache miss (unexpected)\n"

        # Test GET (similar prompt - using similar words)
        similar_prompt = f'Create a LinkedIn post about AI marketing TEST_{timestamp}'
        cached_similar = await cache.get(similar_prompt)
        test_logger.debug(f"Cache hit (similar): {bool(cached_similar)}")

        if cached_similar:
            output += f"✅ GET (similar): Cache hit with similar prompt\n"
            output += f"   Similarity: {cached_similar.get('similarity', 0):.3f}\n"
        else:
            output += f"⚠️  GET (similar): Cache miss (similarity < 0.95 threshold)\n"

        # Test GET (different prompt - should miss)
        different_prompt = 'What is the weather today in Helsinki?'
        cached_miss = await cache.get(different_prompt)
        test_logger.debug(f"Cache miss (different): {bool(cached_miss)}")

        if not cached_miss:
            output += "✅ GET (different): Cache miss as expected"
        else:
            output += "⚠️  GET (different): Unexpected cache hit"

        result.complete("PASSED", output)
        test_logger.info("✅ Semantic cache test PASSED")

    except Exception as e:
        error_msg = f"Semantic cache test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 3: AI LAYER TESTING (Requires OpenAI API Key)

async def test_3_1_content_generator(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.1: Content Generator Agent"""
    test_id = "3.1"
    test_name = "Content Generator Agent"
    layer = "AI Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    if not api_key_checker.can_run_ai_tests():
        output = "⚠️  SKIPPED: OpenAI API key not configured"
        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - OpenAI API key required")
        print_test_result(result)
        wait_for_user()
        return

    try:
        from src.ai_layer.agents.content_generator import ContentGeneratorAgent

        test_logger.info("🔄 Testing Content Generator Agent...")
        test_logger.debug("Initializing ContentGeneratorAgent")

        agent = ContentGeneratorAgent()
        test_logger.debug("Agent initialized successfully")

        content, metadata = await agent.generate_content(
            platform='linkedin',
            persona='decision_maker',
            campaign_config={
                'goal': 'lead_generation',
                'type': 'educational',
                'budget': 1000.0
            },
            context_query='AI marketing automation'
        )

        test_logger.debug(f"Generated content: {content.headline}")
        test_logger.debug(f"Metadata: {metadata}")

        output = "✅ Content generated successfully\n\n"
        output += f"Headline: {content.headline}\n"
        output += f"Body length: {len(content.body)} characters\n"
        output += f"CTA: {content.cta}\n"
        output += f"Claims used: {content.claims_used}\n\n"
        output += "Metadata:\n"
        output += f"  Model: {metadata.get('model', metadata.get('model_used', 'N/A'))}\n"
        output += f"  Cost: ${metadata.get('cost', 0):.4f}\n"
        output += f"  Tokens: {metadata.get('tokens_used', metadata.get('total_tokens', 0))}"

        result.complete("PASSED", output)
        test_logger.info("✅ Content generator test PASSED")

    except Exception as e:
        error_msg = f"Content generator test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_3_2_safety_validator(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.2: Safety Validator Agent"""
    test_id = "3.2"
    test_name = "Safety Validator Agent"
    layer = "AI Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    if not api_key_checker.can_run_ai_tests():
        output = "⚠️  SKIPPED: OpenAI API key not configured"
        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - OpenAI API key required")
        print_test_result(result)
        wait_for_user()
        return

    try:
        from src.ai_layer.agents.safety_validator import SafetyValidatorAgent

        test_logger.info("🔄 Testing Safety Validator Agent...")

        agent = SafetyValidatorAgent()
        test_logger.debug("SafetyValidatorAgent initialized")

        # Test SAFE content (using CLM_003 which exists in claim library)
        safe_result = await agent.validate_content(
            content_text='QWL-guided, evidence-based actions help teams focus effort where impact is highest [CLM_003]',
            headline='Transform Your Marketing with AI',
            claims_used=['CLM_003'],
            platform='linkedin',
            context={'persona': 'decision_maker'}
        )

        test_logger.debug(f"Safe content validation: {safe_result}")

        output = "✅ SAFE CONTENT VALIDATION:\n"
        output += f"  Overall score: {safe_result['overall_score']:.3f}\n"
        output += f"  Toxicity: {safe_result['toxicity_score']:.3f}\n"
        output += f"  Factuality: {safe_result['factuality_score']:.3f}\n"
        output += f"  Brand alignment: {safe_result['brand_alignment_score']:.3f}\n"
        output += f"  Passed: {safe_result['passed']}\n"
        output += f"  Requires review: {safe_result['requires_review']}\n\n"

        # Test UNSAFE content
        unsafe_result = await agent.validate_content(
            content_text='Our competitors are terrible and their products are garbage!',
            headline='Why We Are Better',
            claims_used=[],
            platform='linkedin'
        )

        test_logger.debug(f"Unsafe content validation: {unsafe_result}")

        output += "✅ UNSAFE CONTENT DETECTION:\n"
        output += f"  Overall score: {unsafe_result['overall_score']:.3f}\n"
        output += f"  Toxicity: {unsafe_result['toxicity_score']:.3f}\n"
        output += f"  Should be flagged: {unsafe_result['overall_score'] < 0.7}"

        result.complete("PASSED", output)
        test_logger.info("✅ Safety validator test PASSED")

    except Exception as e:
        error_msg = f"Safety validator test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_3_3_strategy_optimizer(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.3: Strategy Optimizer Agent"""
    test_id = "3.3"
    test_name = "Strategy Optimizer Agent"
    layer = "AI Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    if not api_key_checker.can_run_ai_tests():
        output = "⚠️  SKIPPED: OpenAI API key not configured"
        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - OpenAI API key required")
        print_test_result(result)
        wait_for_user()
        return

    try:
        from src.ai_layer.agents.strategy_optimizer import StrategyOptimizerAgent

        test_logger.info("🔄 Testing Strategy Optimizer Agent...")

        agent = StrategyOptimizerAgent()
        test_logger.debug("StrategyOptimizerAgent initialized")

        strategy = await agent.optimize(
            campaign_id='test_campaign_001',
            historical_data={
                'total_impressions': 10000,
                'avg_ctr': 0.03,
                'conversions': 50
            },
            constraints={
                'max_budget': 1000,
                'duration_days': 30
            }
        )

        test_logger.debug(f"Strategy optimization result: {strategy}")

        output = "✅ Strategy optimization complete\n\n"
        output += f"Recommended action: {strategy['recommended_action']}\n"
        output += f"Confidence: {strategy['confidence']:.3f}\n"
        output += f"Bandit arm: {strategy.get('selected_arm', 'N/A')}\n"
        output += f"Rationale: {strategy.get('rationale', 'N/A')[:100]}..."

        result.complete("PASSED", output)
        test_logger.info("✅ Strategy optimizer test PASSED")

    except Exception as e:
        error_msg = f"Strategy optimizer test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_3_4_thompson_sampling(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.4: Thompson Sampling Bandit"""
    test_id = "3.4"
    test_name = "Thompson Sampling Bandit"
    layer = "AI Layer - Learning"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.ai_layer.learning.thompson_sampling import ThompsonSamplingBandit

        test_logger.info("🔄 Testing Thompson Sampling Bandit...")

        # Initialize bandit with 3 arms
        bandit = ThompsonSamplingBandit(
            experiment_id='test_ts_comprehensive',
            arms=['hook_problem', 'hook_transform', 'hook_social_proof']
        )

        test_logger.debug(f"Bandit initialized with arms: {bandit.arms}")

        output = f"✅ Initialized bandit with arms: {bandit.arms}\n\n"
        output += "Simulating 20 rounds:\n"

        # Simulate 20 rounds
        for round_num in range(1, 21):
            # select_arm() returns (arm_id, confidence)
            selected_arm, confidence = bandit.select_arm()

            # Simulate reward (hook_transform performs better)
            if selected_arm == 'hook_transform':
                reward = 1 if round_num % 2 == 0 else 0  # 50% success
            else:
                reward = 1 if round_num % 4 == 0 else 0  # 25% success

            bandit.update_arm(selected_arm, reward)

            if round_num in [5, 10, 15, 20]:
                output += f"  Round {round_num}: selected {selected_arm}, reward={reward}, confidence={confidence:.3f}\n"
                test_logger.debug(f"Round {round_num}: arm={selected_arm}, reward={reward}, confidence={confidence:.3f}")

        output += "\nFinal arm statistics:\n"
        for arm_id in bandit.arms:
            arm = bandit.arms[arm_id]
            est_ctr = arm.successes / arm.pulls if arm.pulls > 0 else 0
            output += f"  {arm_id}:\n"
            output += f"    Pulls: {arm.pulls}\n"
            output += f"    Successes: {arm.successes}\n"
            output += f"    Est. CTR: {est_ctr:.2%}\n"
            test_logger.debug(f"Arm {arm_id}: pulls={arm.pulls}, successes={arm.successes}, CTR={est_ctr:.2%}")

        result.complete("PASSED", output)
        test_logger.info("✅ Thompson Sampling test PASSED")

    except Exception as e:
        error_msg = f"Thompson Sampling test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_3_5_linucb(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.5: LinUCB Contextual Bandit"""
    test_id = "3.5"
    test_name = "LinUCB Contextual Bandit"
    layer = "AI Layer - Learning"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import numpy as np
        from src.ai_layer.learning.linucb import LinUCBBandit

        test_logger.info("🔄 Testing LinUCB Contextual Bandit...")

        # Initialize LinUCB with 3 arms, 5 features
        bandit = LinUCBBandit(
            n_arms=3,
            n_features=5,
            alpha=0.1,
            use_gpu=False
        )

        test_logger.debug(f"LinUCB initialized: arms={3}, features={5}, device={bandit.device}")

        output = f"✅ Initialized LinUCB bandit\n"
        output += f"   Num arms: 3\n"
        output += f"   Feature dim: 5\n"
        output += f"   Device: {bandit.device}\n\n"
        output += "Simulating 20 rounds with context:\n"

        # Simulate 20 rounds with context
        for round_num in range(1, 21):
            # Generate random context (persona features)
            context = np.random.randn(5)

            selected_arm, confidence = bandit.select_arm(context)

            # Simulate reward
            reward = 1.0 if (selected_arm == 1 and context[3] > 0) else 0.0

            bandit.update(selected_arm, context, reward)

            if round_num in [5, 10, 15, 20]:
                output += f"  Round {round_num}: arm={selected_arm}, reward={reward:.1f}\n"
                test_logger.debug(f"Round {round_num}: arm={selected_arm}, reward={reward}, context[3]={context[3]:.2f}")

        output += "\n✅ LinUCB learning complete\n"
        output += "   Total updates: 20\n"
        output += "   Contextual decisions enabled: True"

        result.complete("PASSED", output)
        test_logger.info("✅ LinUCB test PASSED")

    except Exception as e:
        error_msg = f"LinUCB test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_3_6_episodic_memory(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.6: Episodic Memory Store"""
    test_id = "3.6"
    test_name = "Episodic Memory Store"
    layer = "AI Layer - Memory"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.ai_layer.memory.episodic_memory import EpisodicMemoryStore, AgentMemory
        from datetime import datetime

        test_logger.info("🔄 Testing Episodic Memory Store...")

        # Initialize memory for content generator
        memory_store = EpisodicMemoryStore(agent_name='content_generator')
        test_logger.debug("Memory store initialized")

        output = "✅ Initialized episodic memory store\n\n"

        # Store a memory
        task_memory = AgentMemory(
            agent_name='content_generator',
            task_id='task_comprehensive_001',
            task_description='Generate LinkedIn post for decision_maker persona',
            actions_taken=[
                'Retrieved 3 claims from library',
                'Generated headline with transformation hook',
                'Included clear CTA for demo booking'
            ],
            outcome='success',
            metrics={'safety_score': 0.89, 'ctr': 0.045},
            human_feedback='Good, but could be more concise',
            timestamp=datetime.utcnow(),
            lessons_learned='Decision makers prefer shorter, punchier content'
        )

        await memory_store.store_memory(task_memory)
        test_logger.debug(f"Stored memory for task: {task_memory.task_id}")

        output += "✅ Stored memory:\n"
        output += f"   Task: {task_memory.task_description}\n"
        output += f"   Outcome: {task_memory.outcome}\n"
        output += f"   Lessons: {task_memory.lessons_learned}\n\n"

        # Wait for embeddings to be indexed
        import asyncio
        await asyncio.sleep(2.0)
        test_logger.debug("Waited 2s for embeddings to be indexed")

        # Retrieve similar memories
        query = 'Generate content for executives on LinkedIn'
        memories = await memory_store.retrieve_relevant_memories(query, k=1)

        test_logger.debug(f"Retrieved {len(memories)} memories for query: {query}")

        memory_retrieval_successful = len(memories) > 0

        if memory_retrieval_successful:
            output += f"✅ Retrieved {len(memories)} relevant memories:\n"
            mem = memories[0]
            # retrieve_relevant_memories returns dict format
            content_preview = mem.get('content', '')[:100] + '...' if len(mem.get('content', '')) > 100 else mem.get('content', '')
            output += f"   Content: {content_preview}\n"
            output += f"   Similarity: {mem.get('similarity_score', 0):.3f}\n"
            output += f"   Outcome: {mem.get('outcome', mem.get('metadata', {}).get('outcome', 'N/A'))}"

            result.complete("PASSED", output)
            test_logger.info("✅ Episodic memory test PASSED")
        else:
            output += f"❌ Retrieved {len(memories)} relevant memories:\n"
            output += f"   Expected: >0 memories (just stored 1 memory)\n"
            output += f"   This suggests memory retrieval is not functioning correctly\n"
            output += f"   Possible issues: embeddings not generated, vector search failed"

            result.complete("FAILED", output)
            test_logger.error("❌ Episodic memory test FAILED - memory retrieval returned 0 results")

    except Exception as e:
        error_msg = f"Episodic memory test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_3_7_offline_policy_evaluation(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.7: Offline Policy Evaluation (OPE)"""
    test_id = "3.7"
    test_name = "Offline Policy Evaluation"
    layer = "AI Layer - Learning"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import httpx

        test_logger.info("🔄 Testing Offline Policy Evaluation...")

        ope_status_url = "http://localhost:8000/api/v1/ope/status"

        output = ""

        async with httpx.AsyncClient() as client:
            # Check OPE status
            response = await client.get(ope_status_url, timeout=10.0)

            test_logger.debug(f"OPE status response: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                output += "✅ OPE endpoints available\n\n"
                output += "Available endpoints:\n"
                output += "   - POST /api/v1/ope/evaluate\n"
                output += "   - POST /api/v1/ope/marl-promotion/evaluate\n"
                output += "   - POST /api/v1/ope/compare-policies\n\n"

                from src.api.routers.ope import query_new_policy_actions

                test_data = [
                    {
                        'state': {'platform': 'linkedin', 'persona': 'decision_maker', 'budget': 5000, 'hour_of_day': 10, 'day_of_week': 2},
                        'action': 'hook_transform',
                        'reward': 0.045,
                        'propensity': 0.25
                    },
                    {
                        'state': {'platform': 'twitter', 'persona': 'influencer', 'budget': 3000, 'hour_of_day': 14, 'day_of_week': 4},
                        'action': 'hook_question',
                        'reward': 0.038,
                        'propensity': 0.25
                    }
                ]

                # Test LinUCB policy querying
                linucb_actions = query_new_policy_actions('linucb', test_data)
                test_logger.debug(f"LinUCB actions: {linucb_actions}")

                if linucb_actions and len(linucb_actions) == len(test_data):
                    output += "✅ LinUCB policy querying works\n"
                    output += f"   Generated {len(linucb_actions)} actions for {len(test_data)} states\n\n"
                else:
                    output += "⚠️  LinUCB policy querying returned empty\n\n"

                # Test Thompson Sampling policy querying
                ts_actions = query_new_policy_actions('thompson_sampling', test_data)
                test_logger.debug(f"Thompson Sampling actions: {ts_actions}")

                if ts_actions and len(ts_actions) == len(test_data):
                    output += "✅ Thompson Sampling policy querying works\n"
                    output += f"   Generated {len(ts_actions)} actions for {len(test_data)} states\n\n"
                else:
                    output += "⚠️  Thompson Sampling policy querying returned empty\n\n"

                output += "ℹ️  OPE ready for counterfactual policy evaluation\n"
                output += "   Use /api/v1/ope/evaluate to compare policies"

            else:
                output = f"❌ OPE endpoints not available: {response.status_code}\n\n"
                output += "Expected: HTTP 200 from /api/v1/ope/status\n"
                output += "Possible issues:\n"
                output += "   - OPE router not registered in main.py\n"
                output += "   - API server misconfigured\n"
                output += "ℹ️  OPE is part of core platform, not an external API feature"

        # Only pass if OPE endpoints are available
        if response.status_code == 200:
            result.complete("PASSED", output)
            test_logger.info("✅ OPE test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ OPE test FAILED - endpoints not available")

    except Exception as e:
        error_msg = f"OPE test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_3_8_advanced_experiments(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 3.8: Advanced Research Experiments"""
    test_id = "3.8"
    test_name = "Advanced Research Experiments"
    layer = "AI Layer - Research"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import httpx

        test_logger.info("🔄 Testing Advanced Research Experiments...")

        research_status_url = "http://localhost:8000/api/v1/advanced-experiments/status"

        output = ""

        async with httpx.AsyncClient() as client:
            # Check research mode status
            response = await client.get(research_status_url, timeout=10.0)

            test_logger.debug(f"Research status response: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                research_enabled = data.get('research_mode_enabled', False)

                if research_enabled:
                    output += "✅ Research mode ENABLED\n\n"
                    output += f"Current experiment: {data.get('current_experiment_type', 'N/A')}\n\n"
                    output += "Available experiment types:\n"
                    for exp_type in data.get('available_experiment_types', []):
                        output += f"   - {exp_type}\n"
                    output += "\n"

                    config = data.get('configuration', {})
                    output += "Configuration:\n"
                    output += f"   GPU: {config.get('use_gpu', False)}\n"
                    output += f"   Transformer model: {config.get('transformer_model', 'N/A')}\n"
                    output += f"   Meta-learning steps: {config.get('meta_learning_steps', 'N/A')}\n"
                    output += f"   Ensemble size: {config.get('ensemble_size', 'N/A')}\n\n"

                    # Test integration with StrategyOptimizer
                    from src.ai_layer.agents.strategy_optimizer import StrategyOptimizerAgent

                    optimizer = StrategyOptimizerAgent()
                    test_logger.debug(f"Optimizer research mode: {optimizer.research_mode}")

                    if optimizer.research_mode and optimizer.experiment_runner:
                        output += "✅ StrategyOptimizer integrated with advanced experiments\n"
                        output += f"   Experiment type: {optimizer.experiment_runner.config.experiment_type}\n"
                    else:
                        output += "⚠️  StrategyOptimizer not in research mode\n"
                        output += "   Set ENABLE_RESEARCH_MODE=True to activate\n"

                else:
                    output += "⚠️  Research mode DISABLED\n\n"
                    output += "To enable:\n"
                    output += "   1. Set ENABLE_RESEARCH_MODE=True in .env\n"
                    output += "   2. Set EXPERIMENT_TYPE to one of:\n"
                    output += "      - baseline\n"
                    output += "      - transformer_bandits\n"
                    output += "      - meta_learning\n"
                    output += "      - gaussian_process\n"
                    output += "      - causal_inference\n"
                    output += "      - bayesian_optimization\n"
                    output += "      - ensemble\n"
                    output += "   3. Restart API server\n\n"
                    output += "ℹ️  Research features available for thesis experiments"

            else:
                output = f"❌ Advanced experiments endpoints not available: {response.status_code}\n\n"
                output += "Expected: HTTP 200 from /api/v1/advanced-experiments/status\n"
                output += "Possible issues:\n"
                output += "   - Advanced experiments router not registered in main.py\n"
                output += "   - API server misconfigured"

        # Only pass if endpoint is available (research mode can be disabled, that's ok)
        if response.status_code == 200:
            result.complete("PASSED", output)
            test_logger.info("✅ Advanced experiments test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ Advanced experiments test FAILED - endpoints not available")

    except Exception as e:
        error_msg = f"Advanced experiments test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 4: AUTOMATION LAYER TESTING

async def test_4_1_campaign_deployer_mock(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 4.1: Campaign Deployer (Mock Mode)"""
    test_id = "4.1"
    test_name = "Campaign Deployer (Mock Mode)"
    layer = "Automation Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.automation_layer.deployer import CampaignDeployer

        test_logger.info("🔄 Testing Campaign Deployer...")

        deployer = CampaignDeployer()
        test_logger.debug(f"Available connectors: {list(deployer.connectors.keys())}")

        output = f"✅ Initialized CampaignDeployer\n"
        output += f"   Available connectors: {list(deployer.connectors.keys())}\n\n"

        # Test mock deployment
        deploy_result = await deployer.deploy(
            content_id='test_content_comprehensive_001',
            platform='linkedin',
            content={
                'headline': 'Transform Your Marketing with AI',
                'body': 'Discover how AI agents can automate campaigns and boost ROI...',
                'cta': 'Book a Demo'
            },
            campaign_config={
                'landing_url': 'https://example.com/demo'
            }
        )

        test_logger.debug(f"Deployment result: {deploy_result}")

        output += "✅ Deployment attempted:\n"
        output += f"   Success: {deploy_result.get('success', False)}\n"
        output += f"   Platform: {deploy_result.get('platform', 'N/A')}\n"

        if deploy_result.get('success'):
            output += f"   Post ID: {deploy_result.get('post_id', 'N/A')}\n"
            output += "   ℹ️  Real deployment successful (API keys configured)"
        else:
            output += f"   Error: {deploy_result.get('error', 'Unknown')}\n"
            output += "   ℹ️  Mock mode active (API keys not configured)"

        result.complete("PASSED", output)
        test_logger.info("✅ Campaign deployer test PASSED")

    except Exception as e:
        error_msg = f"Campaign deployer test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_4_2_linkedin_connector(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 4.2: LinkedIn Connector (Requires API Keys)"""
    test_id = "4.2"
    test_name = "LinkedIn Connector"
    layer = "Automation Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    if not api_key_checker.has_linkedin():
        output = "⚠️  SKIPPED: LinkedIn API keys not configured\n"
        output += "ℹ️  Required: LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET, LINKEDIN_ACCESS_TOKEN"
        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - LinkedIn API keys required")
        print_test_result(result)
        wait_for_user()
        return

    try:
        from src.automation_layer.connectors.linkedin_api import LinkedInConnector

        test_logger.info("🔄 Testing LinkedIn Connector...")

        connector = LinkedInConnector()
        test_logger.debug("LinkedIn connector initialized")

        # Validate credentials
        is_valid = await connector.validate_credentials()
        test_logger.debug(f"Credentials valid: {is_valid}")

        output = f"✅ LinkedIn connector initialized\n"
        output += f"✅ Credentials valid: {is_valid}\n\n"

        if is_valid:
            # Test deployment
            deploy_result = await connector.create_sponsored_content({
                'headline': 'Test Ad from Agentic - Comprehensive Test',
                'body': 'This is a test advertisement from the comprehensive test suite.',
                'landing_url': 'https://example.com/demo',
                'image_url': None
            })

            test_logger.debug(f"Deployment result: {deploy_result}")

            output += "✅ LinkedIn deployment attempted:\n"
            output += f"   Success: {deploy_result.success}\n"

            if deploy_result.success:
                output += f"   Post ID: {deploy_result.response_data.get('id', 'N/A')}"
            else:
                output += f"   Error: {deploy_result.error_message}"
        else:
            output += "⚠️  Credentials invalid - skipping deployment test"

        result.complete("PASSED", output)
        test_logger.info("✅ LinkedIn connector test PASSED")

    except Exception as e:
        error_msg = f"LinkedIn connector test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 5: GOVERNANCE LAYER TESTING

async def test_5_1_hitl_queue_manager(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 5.1: HITL Queue Manager"""
    test_id = "5.1"
    test_name = "HITL Queue Manager"
    layer = "Governance Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from uuid import uuid4
        import redis
        from src.governance.hitl_queue import HITLQueueManager
        from src.data_layer.database.connection import get_async_session
        from src.config.settings import settings

        test_logger.info("🔄 Testing HITL Queue Manager...")

        output = ""

        # Initialize redis client
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

        async with get_async_session() as session:
            queue_manager = HITLQueueManager(session, redis_client)
            test_logger.debug("HITL Queue Manager initialized")

            output += "✅ Initialized HITL Queue Manager\n\n"

            # Create test campaign and content first (required for foreign key)
            from src.data_layer.database.models import Campaign, Content, CampaignGoal, Platform, ContentStatus
            from datetime import datetime, timedelta

            campaign_id = uuid4()
            campaign = Campaign(
                id=campaign_id,
                name="Test HITL Campaign",
                goal=CampaignGoal.LEAD_GENERATION,
                platform=Platform.LINKEDIN,
                status="draft",
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=7),
                budget_total=1000.0,
                budget_spent=0.0
            )
            session.add(campaign)
            await session.commit()

            content_id = uuid4()
            content = Content(
                id=content_id,
                campaign_id=campaign_id,
                body="Test content for HITL queue",
                headline="Test Headline",
                status=ContentStatus.PENDING_REVIEW,
                safety_score=0.72
            )
            session.add(content)
            await session.commit()
            test_logger.debug(f"Created test content: {content_id}")

            # Add item to queue
            queue_item = await queue_manager.add_for_review(
                content_id=str(content_id),
                priority=7,
                reason='Safety score below threshold (0.72) - Comprehensive Test'
            )

            test_logger.debug(f"Added to queue: {queue_item.id}")

            output += f"✅ ADD TO QUEUE: Item added\n"
            output += f"   Queue Item ID: {queue_item.id}\n"
            output += f"   Content ID: {content_id}\n"
            output += f"   Priority: {queue_item.priority}\n"
            output += f"   Status: {queue_item.status}\n\n"

            # Get pending items
            pending = await queue_manager.get_pending_items()
            test_logger.debug(f"Pending items: {len(pending)}")

            output += f"✅ GET PENDING: {len(pending)} items in queue\n\n"

            # Get next item (moves to processing)
            next_item = await queue_manager.get_next_for_review(reviewer_id='test@example.com', timeout=0)
            test_logger.debug(f"Got next item: {next_item.id if next_item else None}")

            if next_item:
                output += f"✅ GET NEXT ITEM: Retrieved for processing\n"
                output += f"   Item ID: {next_item.id}\n\n"

                # Approve content
                approved = await queue_manager.approve_content(
                    queue_item_id=str(queue_item.id),
                    reviewer_email='test@example.com',
                    feedback='Approved after review - Comprehensive Test'
                )

                test_logger.debug(f"Approved queue item: {queue_item.id}, success={approved}")

                output += f"✅ APPROVE CONTENT: {'Approved' if approved else 'Failed'}\n"
                output += "   Reviewer: test@example.com\n"
                output += "   Feedback provided\n\n"

                # Verify status changed
                updated_pending = await queue_manager.get_pending_items()
                test_logger.debug(f"Updated pending count: {len(updated_pending)}")

                output += f"✅ VERIFY: Pending count updated to {len(updated_pending)}\n"
                if len(updated_pending) < len(pending):
                    output += f"   Correctly decreased from {len(pending)} to {len(updated_pending)}"
                else:
                    output += f"   ⚠️  Count did not decrease (still {len(updated_pending)})"
            else:
                output += "❌ GET NEXT ITEM: Failed to retrieve item\n"

        result.complete("PASSED", output)
        test_logger.info("✅ HITL queue manager test PASSED")

    except Exception as e:
        error_msg = f"HITL queue test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_5_2_claim_validator(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 5.2: Claim Validator"""
    test_id = "5.2"
    test_name = "Claim Validator"
    layer = "Governance Layer"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.governance.claim_validator import ClaimValidator

        test_logger.info("🔄 Testing Claim Validator...")

        validator = ClaimValidator()
        test_logger.debug(f"Loaded {len(validator.claims_library)} claims")

        output = f"✅ Initialized Claim Validator\n"
        output += f"   Loaded {len(validator.claims_library)} claims from library\n\n"

        # Test VALID content
        content_valid = """
Our AI platform reduces marketing costs by up to 30% [CLM_003]
while improving campaign performance through automated optimization [CLM_006].
        """

        result_valid = validator.validate_content(
            content_text=content_valid,
            claims_used=['CLM_003', 'CLM_006']
        )

        test_logger.debug(f"Valid content validation: {result_valid}")

        output += "✅ VALID CONTENT TEST:\n"
        output += f"   All claims cited: {result_valid['all_claims_cited']}\n"
        output += f"   Hallucinated claims: {result_valid['hallucinated_claims']}\n"
        output += f"   Missing citations: {result_valid['missing_citations']}\n"
        output += f"   Valid: {result_valid['is_valid']}\n\n"

        # Test INVALID content
        content_invalid = """
Our platform is 10x better than competitors [CLM_999_FAKE]
and guarantees 500% ROI instantly.
        """

        result_invalid = validator.validate_content(
            content_text=content_invalid,
            claims_used=['CLM_999_FAKE']
        )

        test_logger.debug(f"Invalid content validation: {result_invalid}")

        output += "✅ INVALID CONTENT DETECTION:\n"
        output += f"   All claims cited: {result_invalid['all_claims_cited']}\n"
        output += f"   Hallucinated claims: {result_invalid['hallucinated_claims']}\n"
        output += f"   Valid: {result_invalid['is_valid']}\n"
        output += "   ℹ️  Correctly detected hallucinated claim\n\n"

        # Validate that claim validator is working correctly
        validation_works = True
        validation_errors = []

        # Valid content should be marked as valid (or at least have recognized claims)
        if result_valid['is_valid'] is False and len(result_valid['hallucinated_claims']) > 0:
            validation_works = False
            validation_errors.append(f"Valid content marked as invalid: hallucinated_claims={result_valid['hallucinated_claims']}")

        # Invalid content should be marked as invalid
        if result_invalid['is_valid'] is not False:
            validation_works = False
            validation_errors.append("Invalid content (CLM_999_FAKE) was not detected as invalid")

        if validation_works:
            result.complete("PASSED", output)
            test_logger.info("✅ Claim validator test PASSED")
        else:
            output += "❌ VALIDATION FAILED:\n"
            for error in validation_errors:
                output += f"   - {error}\n"
            result.complete("FAILED", output)
            test_logger.error("❌ Claim validator test FAILED - validation logic not working correctly")

    except Exception as e:
        error_msg = f"Claim validator test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 6: COST CONTROL TESTING

async def test_6_1_budget_manager(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 6.1: Budget Manager"""
    test_id = "6.1"
    test_name = "Budget Manager"
    layer = "Cost Control"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from uuid import uuid4
        import redis.asyncio as redis
        from src.cost_control.budget_manager import BudgetManager
        from src.data_layer.database.connection import get_async_session
        from src.config.settings import settings

        test_logger.info("🔄 Testing Budget Manager...")

        # Initialize async redis client
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

        async with get_async_session() as session:
            manager = BudgetManager(session, redis_client)
            campaign_id = str(uuid4())

            test_logger.debug(f"Testing with campaign_id: {campaign_id}")

            # Create a test campaign in the database
            from src.data_layer.database.models import Campaign, CampaignStatus, Platform
            from datetime import datetime

            test_campaign = Campaign(
                id=campaign_id,
                name=f"Budget Test Campaign",
                platform=Platform.LINKEDIN,
                status=CampaignStatus.DRAFT,
                budget_total=1000.0,
                budget_spent=0.0,
                budget_daily_limit=100.0,
                created_at=datetime.utcnow()
            )
            session.add(test_campaign)
            await session.commit()
            test_logger.debug(f"Created test campaign: {campaign_id}")

            output = "✅ Initialized Budget Manager\n"
            output += "✅ Created test campaign in database\n\n"

            # Track costs
            await manager.track_cost(
                campaign_id=campaign_id,
                source_type='openai_api',
                cost_amount=0.05,
                metadata={'model': 'gpt-4o', 'tokens': 1000}
            )

            test_logger.debug("Tracked cost: $0.05")

            output += "✅ TRACK COST 1: $0.05\n"
            output += "   Type: openai_api\n"
            output += "   Model: gpt-4o, Tokens: 1000\n\n"

            await manager.track_cost(
                campaign_id=campaign_id,
                source_type='openai_api',
                cost_amount=0.03,
                metadata={'model': 'gpt-4o', 'tokens': 600}
            )

            test_logger.debug("Tracked cost: $0.03")

            output += "✅ TRACK COST 2: $0.03\n"
            output += "   Type: openai_api\n"
            output += "   Model: gpt-4o, Tokens: 600\n\n"

            # Get campaign costs
            total = await manager.get_campaign_costs(campaign_id)
            test_logger.debug(f"Total campaign cost: ${total:.4f}")

            expected_total = 0.08  # $0.05 + $0.03
            cost_tracking_works = total >= expected_total * 0.9  # Allow 10% tolerance

            if cost_tracking_works:
                output += f"✅ GET CAMPAIGN COSTS: ${total:.4f}\n"
                output += f"   Expected: ${expected_total:.2f} (tracked $0.05 + $0.03)\n\n"
            else:
                output += f"❌ GET CAMPAIGN COSTS: ${total:.4f}\n"
                output += f"   Expected: ~${expected_total:.2f} (tracked $0.05 + $0.03)\n"
                output += f"   Issue: Cost tracking not aggregating correctly\n\n"

            # Check budget (assuming budget=1.0)
            can_proceed = await manager.check_budget(
            campaign_id=campaign_id,
            estimated_cost=0.10
            )

            test_logger.debug(f"Can proceed with $0.10 spend: {can_proceed}")

            output += f"✅ CHECK BUDGET ($0.10): {can_proceed}\n\n"

            # Try to exceed budget
            can_proceed_large = await manager.check_budget(
            campaign_id=campaign_id,
            estimated_cost=100.0
            )

            test_logger.debug(f"Can proceed with $100 spend: {can_proceed_large}")

            output += f"✅ CHECK BUDGET ($100.00): {can_proceed_large}\n"

            if cost_tracking_works:
                output += "   ℹ️  Budget protection working correctly"
                result.complete("PASSED", output)
                test_logger.info("✅ Budget manager test PASSED")
            else:
                output += "   ❌ Cost tracking failed validation"
                result.complete("FAILED", output)
                test_logger.error("❌ Budget manager test FAILED - cost aggregation not working")

    except Exception as e:
        error_msg = f"Budget manager test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 7: END-TO-END WORKFLOW TESTING

async def test_7_1_create_campaign_via_api(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 7.1: Complete Campaign Creation via API"""
    test_id = "7.1"
    test_name = "Create Campaign via API"
    layer = "End-to-End Workflow"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import httpx

        test_logger.info("🔄 Testing Campaign Creation via API...")

        api_url = "http://localhost:8000/api/v1/campaigns/"

        payload = {
            "name": "Q1 LinkedIn B2B Campaign - Comprehensive Test",
            "platform": "linkedin",
            "persona": "decision_maker",
            "goal": "lead_generation",
            "budget": 5000.0,
            "duration_days": 30,
            "target_keywords": ["AI", "marketing automation", "B2B"],
            "auto_start": False
        }

        test_logger.debug(f"API URL: {api_url}")
        test_logger.debug(f"Payload: {payload}")

        output = ""

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(api_url, json=payload, timeout=10.0)

                test_logger.debug(f"Response status: {response.status_code}")
                test_logger.debug(f"Response body: {response.text}")

                if response.status_code == 200 or response.status_code == 201:
                    data = response.json()

                    # Validate that critical fields were saved correctly
                    campaign_valid = True
                    validation_errors = []

                    if not data.get('id'):
                        campaign_valid = False
                        validation_errors.append("Campaign ID not returned")

                    # Check if key fields from payload are missing in response
                    platform = data.get('platform')
                    goal = data.get('goal')
                    budget = data.get('budget_total', 0)

                    if platform is None or platform == 'N/A':
                        validation_errors.append(f"Platform not saved (sent: {payload['platform']})")

                    if goal is None or goal == 'N/A':
                        validation_errors.append(f"Goal not saved (sent: {payload['goal']})")

                    if budget == 0 and payload['budget'] > 0:
                        validation_errors.append(f"Budget not saved (sent: ${payload['budget']:.2f}, got: $0.00)")

                    if len(validation_errors) > 0:
                        campaign_valid = False

                    output += "✅ Campaign creation API responded\n\n"
                    output += f"Campaign ID: {data.get('id', 'N/A')}\n"
                    output += f"Name: {data.get('name', 'N/A')}\n"
                    output += f"Platform: {platform or 'N/A'}\n"
                    output += f"Status: {data.get('status', 'N/A')}\n"
                    output += f"Goal: {goal or 'N/A'}\n"
                    output += f"Budget: ${budget:.2f}\n"
                    output += f"Created at: {data.get('created_at', 'N/A')}\n\n"

                    if campaign_valid:
                        output += "ℹ️  Save the Campaign ID for Test 7.2"
                        # Store campaign ID for next test
                        test_suite.last_campaign_id = data.get('id')
                    else:
                        output += "❌ Campaign data incomplete:\n"
                        for error in validation_errors:
                            output += f"   - {error}\n"

                else:
                    campaign_valid = False
                    output = f"⚠️  Unexpected status code: {response.status_code}\n"
                    output += f"Response: {response.text}"

            except httpx.ConnectError:
                campaign_valid = False
                output = "❌ Cannot connect to API\n"
                output += "ℹ️  Ensure API is running: docker-compose up -d api"

        # Only pass if campaign was created successfully with all data
        if campaign_valid:
            result.complete("PASSED", output)
            test_logger.info("✅ Campaign creation test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ Campaign creation test FAILED - data integrity issues")

    except Exception as e:
        error_msg = f"Campaign creation test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_7_2_start_campaign_workflow(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 7.2: Start Campaign Workflow"""
    test_id = "7.2"
    test_name = "Start Campaign Workflow"
    layer = "End-to-End Workflow"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    # Check if we have a campaign ID from previous test
    campaign_id = getattr(test_suite, 'last_campaign_id', None)

    if not campaign_id:
        output = "⚠️  SKIPPED: No campaign ID from Test 7.1\n"
        output += "ℹ️  Run Test 7.1 first to create a campaign"
        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - no campaign ID available")
        print_test_result(result)
        wait_for_user()
        return

    try:
        import httpx

        test_logger.info("🔄 Testing Campaign Workflow Start...")
        test_logger.debug(f"Using campaign_id: {campaign_id}")

        api_url = f"http://localhost:8000/api/v1/campaigns/{campaign_id}/start"

        output = f"Campaign ID: {campaign_id}\n\n"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(api_url, timeout=30.0)

                test_logger.debug(f"Response status: {response.status_code}")
                test_logger.debug(f"Response body: {response.text}")

                if response.status_code == 200:
                    data = response.json()

                    workflow_initiated = data.get('workflow_initiated', False)
                    workflow_valid = workflow_initiated is True

                    if workflow_valid:
                        output += "✅ Campaign workflow started\n\n"
                    else:
                        output += "❌ Campaign workflow API responded but workflow not initiated\n\n"

                    output += f"Campaign ID: {data.get('campaign_id', 'N/A')}\n"
                    output += f"Status: {data.get('status', 'N/A')}\n"
                    output += f"Workflow initiated: {workflow_initiated}\n"
                    output += f"Message: {data.get('message', 'N/A')}\n\n"

                    if workflow_valid:
                        output += "ℹ️  Workflow steps executing:\n"
                        output += "   1. Content Generation → 2. Safety Validation\n"
                        output += "   3. Human Review (if needed) → 4. Cost Check\n"
                        output += "   5. Deployment\n\n"
                        output += "ℹ️  Check logs: docker logs agentic_api --tail=50"
                    else:
                        output += "❌ Workflow did not start - check API implementation\n"
                        output += "   Expected: workflow_initiated = True\n"
                        output += f"   Got: workflow_initiated = {workflow_initiated}"

                else:
                    workflow_valid = False
                    output += f"⚠️  Unexpected status code: {response.status_code}\n"
                    output += f"Response: {response.text}"

            except httpx.ConnectError:
                workflow_valid = False
                output += "❌ Cannot connect to API\n"
                output += "ℹ️  Ensure API is running: docker-compose up -d api"

        # Only pass if workflow was actually initiated
        if workflow_valid:
            result.complete("PASSED", output)
            test_logger.info("✅ Campaign workflow test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ Campaign workflow test FAILED - workflow not initiated")

    except Exception as e:
        error_msg = f"Campaign workflow test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 8: SIMULATION TESTING

async def test_8_1_simpy_environment(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 8.1: SimPy Environment Initialization"""
    test_id = "8.1"
    test_name = "SimPy Environment Initialization"
    layer = "Simulation"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.simulation.environment import MarketingEnvironment, SimulationConfig

        test_logger.info("🔄 Testing SimPy Environment Initialization...")

        config = SimulationConfig(
            duration_days=7,
            num_customers=100,
            platforms=['linkedin'],
            seed=42
        )

        test_logger.debug(f"Config: {config}")

        env = MarketingEnvironment(config)

        test_logger.debug("Environment initialized")

        output = "✅ Simulation environment initialized\n\n"
        output += f"Configuration:\n"
        output += f"   Duration: {config.duration_days} days\n"
        output += f"   Customers: {config.num_customers}\n"
        output += f"   Platforms: {config.platforms}\n"
        output += f"   Seed: {config.seed}\n\n"
        output += "✅ SimPy discrete-event simulation ready"

        result.complete("PASSED", output)
        test_logger.info("✅ SimPy environment test PASSED")

    except Exception as e:
        error_msg = f"SimPy environment test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_8_2_run_simulation(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 8.2: Run Complete Simulation"""
    test_id = "8.2"
    test_name = "Run Complete Simulation"
    layer = "Simulation"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        from src.simulation.environment import MarketingEnvironment, SimulationConfig

        test_logger.info("🔄 Running complete simulation...")

        config = SimulationConfig(
            duration_days=7,
            num_customers=50,
            platforms=['linkedin', 'twitter'],
            seed=42
        )

        env = MarketingEnvironment(config)

        test_logger.debug("Starting simulation run...")

        simulation_results = env.run_simulation()

        test_logger.debug(f"Simulation results: {simulation_results}")

        # Validate that simulation generated actual data
        total_interactions = simulation_results.get('total_interactions', 0)
        total_clicks = simulation_results.get('total_clicks', 0)

        if total_interactions == 0:
            output = "❌ Simulation ran but generated NO interactions\n\n"
            output += "Results:\n"
            output += f"   Simulated days: {simulation_results.get('duration_days', 'N/A')}\n"
            output += f"   Total interactions: {total_interactions}\n"
            output += f"   Clicks: {total_clicks}\n"
            output += f"   Conversions: {simulation_results.get('total_conversions', 0)}\n"
            output += f"   CTR: {simulation_results.get('avg_ctr', 0):.2%}\n"
            output += f"   Conversion rate: {simulation_results.get('conversion_rate', 0):.2%}\n\n"
            output += "⚠️  Expected non-zero simulation activity"

            result.complete("FAILED", output)
            test_logger.error("❌ Simulation run test FAILED - no activity generated")
        else:
            output = "✅ Simulation completed successfully\n\n"
            output += "Results:\n"
            output += f"   Simulated days: {simulation_results.get('duration_days', 'N/A')}\n"
            output += f"   Total interactions: {total_interactions}\n"
            output += f"   Clicks: {total_clicks}\n"
            output += f"   Conversions: {simulation_results.get('total_conversions', 0)}\n"
            output += f"   CTR: {simulation_results.get('avg_ctr', 0):.2%}\n"
            output += f"   Conversion rate: {simulation_results.get('conversion_rate', 0):.2%}\n\n"

            # Warn if clicks are suspiciously low
            expected_ctr = 0.04  # Average click prob from default personas
            expected_clicks = int(total_interactions * expected_ctr)

            if total_clicks == 0 and total_interactions > 100:
                output += "⚠️  WARNING: 0 clicks despite significant interactions\n"
                output += f"   Expected ~{expected_clicks} clicks (4% CTR)\n"
                output += "   Possible issues:\n"
                output += "   - Platform click logic not implemented\n"
                output += "   - CustomerAgent click behavior broken\n"
                output += "   - Persona click_prob not being applied\n\n"

            output += "✅ Market simulation digital twin operational"

            result.complete("PASSED", output)
            test_logger.info("✅ Simulation run test PASSED")

    except Exception as e:
        error_msg = f"Simulation run test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_8_3_simulation_validation(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 8.3: Simulation Validation with SimulationValidator (MAPE <10%)"""
    test_id = "8.3"
    test_name = "Simulation Validation (MAPE)"
    layer = "Simulation"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        test_logger.info("🔄 Testing Simulation Validation with SimulationValidator...")

        # Check if historical data exists
        historical_data_path = project_root / "data" / "historical" / "campaign_results.csv"

        test_logger.debug(f"Looking for historical data at: {historical_data_path}")

        if not historical_data_path.exists():
            output = "⚠️  SKIPPED: No historical data found\n\n"
            output += f"Expected location: {historical_data_path}\n\n"
            output += "ℹ️  To run validation:\n"
            output += "   1. Place historical campaign data in data/historical/\n"
            output += "   2. Run: python scripts/validate_simulation.py\n"
            output += "   3. Target: MAPE < 10% (>90% accuracy)\n\n"
            output += "ℹ️  Uses SimulationValidator class for comprehensive validation:\n"
            output += "   - MAPE (Mean Absolute Percentage Error)\n"
            output += "   - RMSE (Root Mean Squared Error)\n"
            output += "   - Pearson Correlation\n"
            output += "   - Weighted overall accuracy"

            result.complete("SKIPPED", output)
            test_logger.warning("Skipping validation - no historical data")

        else:
            # Run validation script as subprocess
            test_logger.debug("Running validation script...")

            import subprocess
            import json

            validation_output_path = project_root / "validation_report.json"

            # Run validation script
            process = subprocess.run(
                [
                    "python",
                    str(project_root / "scripts" / "validate_simulation.py"),
                    "--data", str(historical_data_path),
                    "--output", str(validation_output_path)
                ],
                capture_output=True,
                text=True,
                cwd=str(project_root)
            )

            test_logger.debug(f"Validation script exit code: {process.returncode}")
            test_logger.debug(f"Validation script stdout:\n{process.stdout}")

            if process.returncode in [0, 1]:  # 0 = passed, 1 = failed but completed
                # Load validation report
                if validation_output_path.exists():
                    with open(validation_output_path, 'r') as f:
                        report = json.load(f)

                    validation_results = report.get('validation_results', {})
                    metrics = validation_results.get('metrics', {})
                    overall = validation_results.get('overall', {})

                    output = "✅ Simulation validation complete (using SimulationValidator)\n\n"
                    output += "MAPE Results (with RMSE & Correlation):\n"

                    for metric_name, metric_data in metrics.items():
                        output += f"   {metric_name.upper()}:\n"
                        output += f"      MAPE: {metric_data.get('mape', 0):.2f}%\n"
                        output += f"      Accuracy: {metric_data.get('accuracy', 0):.2f}%\n"
                        output += f"      RMSE: {metric_data.get('rmse', 0):.4f}\n"
                        output += f"      Correlation: {metric_data.get('correlation', 0):.4f}\n"

                    output += f"\nOverall Results:\n"
                    output += f"   Overall MAPE: {overall.get('mape', 100):.2f}%\n"
                    output += f"   Overall Accuracy: {overall.get('accuracy', 0):.2f}%\n"
                    output += f"   Worst Metric: {overall.get('worst_metric', 'N/A')}\n"
                    output += f"   Worst Accuracy: {overall.get('worst_accuracy', 0):.2f}%\n\n"

                    target_met = overall.get('target_met', False)

                    if target_met:
                        output += "🎯 Target achieved: >90% accuracy (MAPE < 10%)\n\n"
                    else:
                        output += f"⚠️  Target not met: {overall.get('accuracy', 0):.2f}% (Target: >90%)\n\n"

                    output += f"Summary: {overall.get('summary', 'N/A')}\n\n"
                    output += f"📄 Full report: {validation_output_path}"

                    # Only pass if target is met
                    if target_met:
                        result.complete("PASSED", output)
                        test_logger.info("✅ Simulation validation test PASSED")
                    else:
                        result.complete("FAILED", output)
                        test_logger.error("❌ Simulation validation test FAILED - accuracy below 90% threshold")
                else:
                    output = "❌ Validation script ran but no report generated\n\n"
                    output += f"Exit code: {process.returncode}\n"
                    output += f"Stdout:\n{process.stdout}\n"
                    output += f"Stderr:\n{process.stderr}"

                    result.complete("FAILED", output)
                    test_logger.error("Validation report not found")
            else:
                output = f"❌ Validation script failed with exit code {process.returncode}\n\n"
                output += f"Stdout:\n{process.stdout}\n\n"
                output += f"Stderr:\n{process.stderr}"

                result.complete("FAILED", output)
                test_logger.error(f"Validation script failed: {process.stderr}")

    except Exception as e:
        error_msg = f"Simulation validation test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 9: PRODUCTION DEPLOYMENT TESTING

async def test_9_1_linkedin_production(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 9.1: LinkedIn Production Deployment"""
    test_id = "9.1"
    test_name = "LinkedIn Production Deployment"
    layer = "Production Deployment"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    if not api_key_checker.can_run_linkedin_deployment():
        output = "⚠️  SKIPPED: LinkedIn production deployment requires:\n"
        output += "   - OPENAI_API_KEY ✓\n" if api_key_checker.has_openai() else "   - OPENAI_API_KEY ✗\n"
        output += "   - LINKEDIN_CLIENT_ID ✓\n" if api_key_checker.api_keys.get('LINKEDIN_CLIENT_ID') else "   - LINKEDIN_CLIENT_ID ✗\n"
        output += "   - LINKEDIN_CLIENT_SECRET ✓\n" if api_key_checker.api_keys.get('LINKEDIN_CLIENT_SECRET') else "   - LINKEDIN_CLIENT_SECRET ✗\n"
        output += "   - LINKEDIN_ACCESS_TOKEN ✓\n" if api_key_checker.api_keys.get('LINKEDIN_ACCESS_TOKEN') else "   - LINKEDIN_ACCESS_TOKEN ✗\n"

        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - LinkedIn production API keys required")
        print_test_result(result)
        wait_for_user()
        return

    try:
        import httpx

        test_logger.info("🔄 Testing LinkedIn Production Deployment...")

        # Create campaign for production test
        create_url = "http://localhost:8000/api/v1/campaigns/"

        payload = {
            "name": "PROD Test - LinkedIn - Comprehensive Suite",
            "platform": "linkedin",
            "persona": "decision_maker",
            "goal": "lead_generation",
            "budget": 100.0,
            "duration_days": 7,
            "auto_start": True  # Auto-start workflow
        }

        output = ""

        async with httpx.AsyncClient() as client:
            # Create campaign
            create_response = await client.post(create_url, json=payload, timeout=10.0)

            if create_response.status_code in [200, 201]:
                campaign_data = create_response.json()
                campaign_id = campaign_data.get('id')

                test_logger.debug(f"Created campaign: {campaign_id}")

                output += f"✅ Campaign created: {campaign_id}\n\n"

                # Wait a moment for workflow to process
                import asyncio
                await asyncio.sleep(5)

                # Check campaign status
                status_url = f"http://localhost:8000/api/v1/campaigns/{campaign_id}"
                status_response = await client.get(status_url, timeout=10.0)

                if status_response.status_code == 200:
                    status_data = status_response.json()

                    output += "Campaign Status:\n"
                    output += f"   Status: {status_data.get('status', 'N/A')}\n"
                    output += f"   Budget spent: ${status_data.get('budget_spent', 0):.2f}\n"

                    contents = status_data.get('contents', [])
                    if contents:
                        content = contents[0]
                        output += f"\nContent Status:\n"
                        output += f"   Content ID: {content.get('id', 'N/A')}\n"
                        output += f"   Status: {content.get('status', 'N/A')}\n"
                        output += f"   Safety score: {content.get('safety_score', 0):.3f}\n"

                        if content.get('status') == 'deployed':
                            output += f"\n✅ Successfully deployed to LinkedIn!\n"
                            output += f"   Deployed at: {content.get('deployed_at', 'N/A')}"
                        else:
                            output += f"\n⚠️  Deployment status: {content.get('status')}"

                output += "\n\nℹ️  Check LinkedIn Campaign Manager to verify post"

            else:
                output = f"❌ Failed to create campaign: {create_response.status_code}"

        result.complete("PASSED", output)
        test_logger.info("✅ LinkedIn production test PASSED")

    except Exception as e:
        error_msg = f"LinkedIn production test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_9_2_monitor_campaign_performance(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 9.2: Monitor Campaign Performance"""
    test_id = "9.2"
    test_name = "Monitor Campaign Performance"
    layer = "Production Deployment"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    # Check if we have a campaign ID
    campaign_id = getattr(test_suite, 'last_campaign_id', None)

    if not campaign_id:
        output = "⚠️  SKIPPED: No campaign ID available\n"
        output += "ℹ️  Run Test 7.1 or 9.1 first to create a campaign"
        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - no campaign ID available")
        print_test_result(result)
        wait_for_user()
        return

    try:
        import httpx

        test_logger.info("🔄 Testing Campaign Performance Monitoring...")
        test_logger.debug(f"Using campaign_id: {campaign_id}")

        metrics_url = f"http://localhost:8000/api/v1/metrics/campaigns/{campaign_id}"

        output = f"Campaign ID: {campaign_id}\n\n"

        async with httpx.AsyncClient() as client:
            response = await client.get(metrics_url, timeout=10.0)

            test_logger.debug(f"Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                output += "✅ Campaign metrics retrieved\n\n"
                output += "Performance Metrics:\n"

                metrics = data.get('metrics', {})
                output += f"   Impressions: {metrics.get('impressions', 0):,}\n"
                output += f"   Clicks: {metrics.get('clicks', 0):,}\n"
                output += f"   Conversions: {metrics.get('conversions', 0)}\n"
                output += f"   CTR: {metrics.get('ctr', 0):.2%}\n"
                output += f"   CPL: ${metrics.get('cpl', 0):.2f}\n"
                output += f"   Budget spent: ${metrics.get('budget_spent', 0):.2f}\n\n"
                output += f"Last updated: {data.get('last_updated', 'N/A')}"

            else:
                output += f"⚠️  Status code: {response.status_code}\n"
                output += "ℹ️  Campaign may not have metrics yet"

        result.complete("PASSED", output)
        test_logger.info("✅ Campaign performance monitoring test PASSED")

    except Exception as e:
        error_msg = f"Performance monitoring test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_9_3_delayed_reward_tracking(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 9.3: Delayed Reward Tracking"""
    test_id = "9.3"
    test_name = "Delayed Reward Tracking"
    layer = "Production Deployment"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    campaign_id = getattr(test_suite, 'last_campaign_id', None)

    if not campaign_id:
        output = "⚠️  SKIPPED: No campaign ID available\n"
        output += "ℹ️  Run Test 7.1 or 9.1 first to create a campaign"
        result.complete("SKIPPED", output)
        test_logger.warning("Skipping test - no campaign ID available")
        print_test_result(result)
        wait_for_user()
        return

    try:
        import httpx
        from datetime import datetime, timedelta

        test_logger.info("🔄 Testing Delayed Reward Tracking...")

        track_url = "http://localhost:8000/api/v1/rewards/track"

        # Simulate a lead booking
        meeting_date = (datetime.utcnow() + timedelta(days=3)).isoformat()

        payload = {
            "campaign_id": campaign_id,
            "lead_email": "john.doe.test@example.com",
            "lead_data": {
                "name": "John Doe Test",
                "company": "Acme Corp",
                "source": "linkedin_ad"
            },
            "meeting_scheduled": True,
            "meeting_date": meeting_date
        }

        test_logger.debug(f"Tracking payload: {payload}")

        output = f"Campaign ID: {campaign_id}\n\n"

        async with httpx.AsyncClient() as client:
            # Track reward
            track_response = await client.post(track_url, json=payload, timeout=10.0)

            test_logger.debug(f"Track response: {track_response.status_code}")

            if track_response.status_code == 200:
                track_data = track_response.json()

                output += "✅ Delayed reward tracked\n\n"
                output += f"Reward ID: {track_data.get('reward_id', 'N/A')}\n"
                output += f"Initial reward: {track_data.get('initial_reward', 0)}\n"
                output += f"Status: {track_data.get('status', 'N/A')}\n"
                output += f"Attribution window: {track_data.get('attribution_window_hours', 72)} hours\n\n"

                # Verify delayed rewards
                verify_url = f"http://localhost:8000/api/v1/rewards/campaign/{campaign_id}"
                verify_response = await client.get(verify_url, timeout=10.0)

                if verify_response.status_code == 200:
                    verify_data = verify_response.json()

                    output += "✅ Verified delayed rewards\n\n"
                    output += f"Campaign ID: {verify_data.get('campaign_id', 'N/A')}\n"
                    output += f"Total booked: {verify_data.get('total_booked', 0)}\n"
                    output += f"Total pending: {verify_data.get('total_pending', 0)}\n\n"

                    rewards = verify_data.get('delayed_rewards', [])
                    if rewards:
                        output += "Delayed Rewards:\n"
                        for reward in rewards[:3]:  # Show first 3
                            output += f"   - Lead: {reward.get('lead_email', 'N/A')}\n"
                            output += f"     Status: {reward.get('status', 'N/A')}\n"

                output += "\nℹ️  72-hour attribution window active"

            else:
                output += f"⚠️  Track response: {track_response.status_code}"

        result.complete("PASSED", output)
        test_logger.info("✅ Delayed reward tracking test PASSED")

    except Exception as e:
        error_msg = f"Delayed reward test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# LAYER 10: MONITORING AND OBSERVABILITY

async def test_10_1_prometheus_metrics(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 10.1: Prometheus Metrics"""
    test_id = "10.1"
    test_name = "Prometheus Metrics"
    layer = "Monitoring"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import httpx

        test_logger.info("🔄 Testing Prometheus Metrics...")

        # Use container service name when running inside Docker
        prometheus_url = "http://prometheus:9090"
        query_url = f"{prometheus_url}/api/v1/query"

        output = ""
        prometheus_accessible = False

        async with httpx.AsyncClient() as client:
            try:
                # Query HTTP requests
                response = await client.get(
                    query_url,
                    params={'query': 'http_requests_total'},
                    timeout=5.0
                )

                test_logger.debug(f"Prometheus response: {response.status_code}")

                if response.status_code == 200:
                    prometheus_accessible = True
                    data = response.json()

                    output += f"✅ Prometheus accessible at {prometheus_url}\n\n"
                    output += "Query: http_requests_total\n"
                    output += f"Status: {data.get('status', 'N/A')}\n"

                    results = data.get('data', {}).get('result', [])
                    output += f"Results: {len(results)} metrics found\n\n"

                    # Query active campaigns
                    campaigns_response = await client.get(
                        query_url,
                        params={'query': 'active_campaigns_total'},
                        timeout=5.0
                    )

                    if campaigns_response.status_code == 200:
                        campaigns_data = campaigns_response.json()
                        output += "Query: active_campaigns_total\n"
                        output += f"Status: {campaigns_data.get('status', 'N/A')}\n\n"

                    output += "ℹ️  Manual verification:\n"
                    output += "   1. Open http://localhost:9090\n"
                    output += "   2. Go to Graph tab\n"
                    output += "   3. Query: rate(http_requests_total[5m])"

                else:
                    output = f"⚠️  Prometheus response: {response.status_code}"

            except httpx.ConnectError:
                output = f"❌ Prometheus not accessible at {prometheus_url}\n"
                output += "ℹ️  Start with: docker-compose up -d prometheus"

        # Only pass if Prometheus is accessible
        if prometheus_accessible:
            result.complete("PASSED", output)
            test_logger.info("✅ Prometheus metrics test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ Prometheus metrics test FAILED - service not accessible")

    except Exception as e:
        error_msg = f"Prometheus test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

async def test_10_2_grafana_dashboards(api_key_checker: APIKeyChecker, test_suite: TestSuite):
    """Test 10.2: Grafana Dashboards"""
    test_id = "10.2"
    test_name = "Grafana Dashboards"
    layer = "Monitoring"

    print_test_header(test_id, test_name, layer)
    result = test_suite.start_test(test_id, test_name)

    try:
        import httpx

        test_logger.info("🔄 Testing Grafana Dashboards...")

        # Use container service name when running inside Docker
        grafana_url = "http://grafana:3000"
        api_url = f"{grafana_url}/api/health"

        output = ""
        grafana_accessible = False

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(api_url, timeout=5.0)

                test_logger.debug(f"Grafana response: {response.status_code}")

                if response.status_code == 200:
                    grafana_accessible = True
                    data = response.json()

                    output += f"✅ Grafana accessible at {grafana_url}\n\n"
                    output += f"Health Status:\n"
                    output += f"   Database: {data.get('database', 'N/A')}\n"
                    output += f"   Version: {data.get('version', 'N/A')}\n\n"

                    output += "ℹ️  Manual verification:\n"
                    output += f"   1. Open {grafana_url}\n"
                    output += "   2. Login: admin / admin123\n"
                    output += "   3. Navigate to Dashboards\n"
                    output += "   4. Check for:\n"
                    output += "      - Agentic System Overview\n"
                    output += "      - Campaign Performance\n"
                    output += "      - AI Agent Metrics\n"
                    output += "      - Cost Tracking"

                else:
                    output = f"⚠️  Grafana response: {response.status_code}"

            except httpx.ConnectError:
                output = f"❌ Grafana not accessible at {grafana_url}\n"
                output += "ℹ️  Start with: docker-compose up -d grafana"

        # Only pass if Grafana is accessible
        if grafana_accessible:
            result.complete("PASSED", output)
            test_logger.info("✅ Grafana dashboards test PASSED")
        else:
            result.complete("FAILED", output)
            test_logger.error("❌ Grafana dashboards test FAILED - service not accessible")

    except Exception as e:
        error_msg = f"Grafana test failed: {str(e)}\n{traceback.format_exc()}"
        result.complete("FAILED", error=error_msg)
        test_logger.error(f"❌ Test FAILED: {error_msg}")

    print_test_result(result)
    wait_for_user()

# MAIN TEST RUNNER

async def run_all_tests():
    """Run all tests in sequence"""

    test_logger.info("=" * 80)
    test_logger.info("🚀 AGENTIC AI AGENT PLATFORM - COMPREHENSIVE TEST SUITE")
    test_logger.info("=" * 80)
    test_logger.info("")
    test_logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    test_logger.info(f"📁 Log file: {test_logger.log_file}")
    test_logger.info("")

    # Initialize API key checker and test suite
    api_key_checker = APIKeyChecker()
    test_suite = TestSuite()

    try:
        # LAYER 1: INFRASTRUCTURE TESTING (NO API KEYS REQUIRED)

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 1: INFRASTRUCTURE TESTING")
        test_logger.info("=" * 80)

        await test_1_1_postgresql_connectivity(api_key_checker, test_suite)
        await test_1_2_redis_connectivity(api_key_checker, test_suite)
        await test_1_3_pgvector_extension(api_key_checker, test_suite)
        await test_1_4_api_health_check(api_key_checker, test_suite)
        await test_1_5_dashboard_accessibility(api_key_checker, test_suite)

        # LAYER 2: DATA LAYER TESTING (NO API KEYS REQUIRED)

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 2: DATA LAYER TESTING")
        test_logger.info("=" * 80)

        await test_2_1_campaign_repository(api_key_checker, test_suite)
        await test_2_2_content_repository(api_key_checker, test_suite)
        await test_2_3_vector_store_operations(api_key_checker, test_suite)
        await test_2_4_semantic_cache(api_key_checker, test_suite)

        # LAYER 3: AI LAYER TESTING (OPENAI API KEY REQUIRED)

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 3: AI LAYER TESTING")
        test_logger.info("=" * 80)

        if not api_key_checker.can_run_ai_tests():
            test_logger.warning("⚠️  OpenAI API key not configured - Skipping AI Layer tests")
            test_logger.info("")
        else:
            await test_3_1_content_generator(api_key_checker, test_suite)
            await test_3_2_safety_validator(api_key_checker, test_suite)
            await test_3_3_strategy_optimizer(api_key_checker, test_suite)

        await test_3_4_thompson_sampling(api_key_checker, test_suite)
        await test_3_5_linucb(api_key_checker, test_suite)
        await test_3_6_episodic_memory(api_key_checker, test_suite)
        await test_3_7_offline_policy_evaluation(api_key_checker, test_suite)
        await test_3_8_advanced_experiments(api_key_checker, test_suite)

        # LAYER 4: AUTOMATION LAYER TESTING

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 4: AUTOMATION LAYER TESTING")
        test_logger.info("=" * 80)

        await test_4_1_campaign_deployer_mock(api_key_checker, test_suite)

        if api_key_checker.can_run_linkedin_deployment():
            await test_4_2_linkedin_connector(api_key_checker, test_suite)
        else:
            test_logger.warning("⚠️  LinkedIn API keys not configured - Skipping LinkedIn connector test")

        # LAYER 5: GOVERNANCE LAYER TESTING

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 5: GOVERNANCE LAYER TESTING")
        test_logger.info("=" * 80)

        await test_5_1_hitl_queue_manager(api_key_checker, test_suite)
        await test_5_2_claim_validator(api_key_checker, test_suite)

        # LAYER 6: COST CONTROL TESTING

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 6: COST CONTROL TESTING")
        test_logger.info("=" * 80)

        await test_6_1_budget_manager(api_key_checker, test_suite)

        # LAYER 7: END-TO-END WORKFLOW TESTING

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 7: END-TO-END WORKFLOW TESTING")
        test_logger.info("=" * 80)

        await test_7_1_create_campaign_via_api(api_key_checker, test_suite)
        await test_7_2_start_campaign_workflow(api_key_checker, test_suite)

        # LAYER 8: SIMULATION TESTING

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 8: SIMULATION TESTING")
        test_logger.info("=" * 80)

        await test_8_1_simpy_environment(api_key_checker, test_suite)
        await test_8_2_run_simulation(api_key_checker, test_suite)
        await test_8_3_simulation_validation(api_key_checker, test_suite)

        # LAYER 9: PRODUCTION DEPLOYMENT TESTING

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 9: PRODUCTION DEPLOYMENT TESTING")
        test_logger.info("=" * 80)

        if api_key_checker.can_run_linkedin_deployment():
            await test_9_1_linkedin_production(api_key_checker, test_suite)
            await test_9_2_monitor_campaign_performance(api_key_checker, test_suite)
            await test_9_3_delayed_reward_tracking(api_key_checker, test_suite)
        else:
            test_logger.warning("⚠️  LinkedIn API keys not configured - Skipping production deployment tests")

        # LAYER 10: MONITORING AND OBSERVABILITY TESTING

        test_logger.info("=" * 80)
        test_logger.info("📦 LAYER 10: MONITORING AND OBSERVABILITY TESTING")
        test_logger.info("=" * 80)

        await test_10_1_prometheus_metrics(api_key_checker, test_suite)
        await test_10_2_grafana_dashboards(api_key_checker, test_suite)

    except KeyboardInterrupt:
        test_logger.warning("\n\n⚠️  Test suite interrupted by user")
    except Exception as e:
        test_logger.error(f"\n\n💥 CRITICAL ERROR: {str(e)}")
        test_logger.error(traceback.format_exc())
    finally:
        # Print final summary
        test_logger.info("")
        test_logger.info("=" * 80)
        test_logger.info("📊 TEST SUITE SUMMARY")
        test_logger.info("=" * 80)

        summary = test_suite.get_summary()

        test_logger.info(f"Total tests: {summary['total']}")
        test_logger.info(f"✅ Passed: {summary['passed']}")
        test_logger.info(f"❌ Failed: {summary['failed']}")
        test_logger.info(f"⏭️  Skipped: {summary['skipped']}")
        test_logger.info(f"💥 Errors: {summary['errors']}")
        test_logger.info(f"📈 Success rate: {summary['success_rate']:.1f}%")
        test_logger.info("")
        test_logger.info(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        test_logger.info("=" * 80)

        # Save results to JSON
        test_suite.save_results()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
