
import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, "/app")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.ai_layer.learning.weekly_learning_report import generate_weekly_report
from src.data_layer.database.models import Platform

async def test_generate_report():
    print("🚀 Starting Weekly Report Generation Test...")
    try:
        report = await generate_weekly_report()
        
        if "error" in report:
            print(f"❌ Error in report generation: {report['error']}")
            sys.exit(1)
            
        print("✅ Report generation successful!")
        print(f"Report ID: {report.get('id', 'N/A')}")
        print(f"Week: {report.get('week_number')}/{report.get('year')}")
        
        # Verify defensive coding for Platform
        platform_perf = report.get('platform_performance', {})
        print(f"Platform Performance Keys: {list(platform_perf.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_generate_report())
    if not success:
        sys.exit(1)
