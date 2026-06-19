from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

from ...ai_layer.agents.market_scraper import MarketScraperAgent

logger = logging.getLogger(__name__)

router = APIRouter()

scraper = MarketScraperAgent()


class ScrapeRequest(BaseModel):
    keywords: List[str]
    limit: int = 20
    platform: str = "linkedin"


class AnalyzeRequest(BaseModel):
    posts: List[Dict[str, Any]]


class CompetitorWebsiteRequest(BaseModel):
    """Request model for competitor website scraping (uses BeautifulSoup, no APIFY required)"""
    url: str
    competitor_name: str = "Competitor"



@router.post("/scrape")
async def scrape_content(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks = None
):
    """Scrape market content for inspiration based on keywords"""
    try:
        result = await scraper.get_inspiration_for_campaign(
            keywords=request.keywords, 
            limit=request.limit,
            platform=request.platform
        )
        
        if not result or not result.get('success'):
            return {
                "status": "success",
                "total_posts": 0,
                "posts": [],
                "insights": None,
                "data_source": result.get('data_source', 'none') if result else 'none'
            }
        
        posts = result.get('sample_posts', [])
        
        return {
            "status": "success",
            "total_posts": len(posts),
            "posts": posts,
            "insights": result.get('insights'),
            "data_source": result.get('data_source', 'unknown')
        }
    except Exception as e:
        logger.error(f"Failed to scrape content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_posts(
    request: AnalyzeRequest
):
    """Analyze scraped posts for patterns and insights"""
    try:
        insights = await scraper.analyze_content_patterns(request.posts)
        return insights
    except Exception as e:
        logger.error(f"Failed to analyze posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/competitor-website")
async def scrape_competitor_website(
    request: CompetitorWebsiteRequest
):
    """
    Scrape competitor website for messaging themes and keywords.
    Uses BeautifulSoup - does NOT require APIFY token.
    """
    try:
        result = await scraper.scrape_competitor_website(
            competitor_url=request.url,
            competitor_name=request.competitor_name
        )
        
        if result.get('success'):
            return {
                "status": "success",
                "competitor": result.get('competitor'),
                "url": result.get('url'),
                "messaging_themes": result.get('messaging_themes', []),
                "meta_description": result.get('meta_description', ''),
                "keywords": result.get('keywords', []),
                "scraped_at": result.get('scraped_at')
            }
        else:
            return {
                "status": "failed",
                "error": "Failed to scrape website",
                "messaging_themes": []
            }
    except Exception as e:
        logger.error(f"Failed to scrape competitor website: {e}")
        raise HTTPException(status_code=500, detail=str(e))