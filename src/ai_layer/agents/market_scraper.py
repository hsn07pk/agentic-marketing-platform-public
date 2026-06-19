import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
import re
from collections import Counter

from ...config.settings import settings
from ...data_layer.database.connection import get_async_session, get_sync_session
from ...data_layer.database.models import ScrapedContent

logger = logging.getLogger(__name__)

def _get_config_value(key: str, default: Any = None) -> Any:
    """
    Get configuration value from database via configuration service.
    Falls back to settings object if service unavailable.
    """
    try:
        from ...config.configuration_service import ConfigurationService, DEFAULT_CONFIGURATIONS
        
        db_session = get_sync_session()
        try:
            config_service = ConfigurationService(db_session)
            value = config_service.get_value(key, default=None)
            if value is not None:
                return value
        finally:
            db_session.close()
        
        if key in DEFAULT_CONFIGURATIONS:
            return DEFAULT_CONFIGURATIONS[key]["default"]
            
    except Exception as e:
        logger.debug(f"Could not read config from service for {key}: {e}")
    
    return getattr(settings, key, default)


# Default scraping sources — loaded from config service at runtime.
# Structured as JSON so users can add/remove sources from the dashboard.
DEFAULT_SCRAPE_SOURCES = [
    {
        "url": "https://www.personneltoday.com/?s={query}",
        "fallback": "https://www.personneltoday.com/",
        "name": "Personnel Today",
        "extractor": "generic",
        "category": "industry",
    },
    {
        "url": "https://www.hcamag.com/au/search?q={query}",
        "fallback": "https://www.hcamag.com/au/news",
        "name": "HRD (HC Online)",
        "extractor": "generic",
        "category": "industry",
    },
    {
        "url": "https://lattice.com/blog?q={query}",
        "fallback": "https://lattice.com/blog",
        "name": "Lattice Blog",
        "extractor": "generic",
        "category": "competitor",
    },
    {
        "url": "https://www.cultureamp.com/blog?q={query}",
        "fallback": "https://www.cultureamp.com/blog",
        "name": "Culture Amp Blog",
        "extractor": "generic",
        "category": "competitor",
    },
    {
        "url": "https://www.visier.com/blog/?s={query}",
        "fallback": "https://www.visier.com/blog/",
        "name": "Visier Blog",
        "extractor": "generic",
        "category": "competitor",
    },
    {
        "url": "https://www.15five.com/blog/?s={query}",
        "fallback": "https://www.15five.com/blog/",
        "name": "15Five Blog",
        "extractor": "generic",
        "category": "competitor",
    },
]

# Domain-relevant keywords for the HR tech / Employee Experience industry
DOMAIN_KEYWORDS = [
    'employee', 'engagement', 'wellbeing', 'well-being', 'retention',
    'people analytics', 'hr', 'human resources', 'workplace', 'culture',
    'performance', 'productivity', 'leadership', 'team', 'talent',
    'attrition', 'turnover', 'onboarding', 'development', 'coaching',
    'burnout', 'resilience', 'inclusion', 'diversity', 'feedback',
    'survey', 'pulse', 'ebitda', 'roi', 'qwl', 'quality of working life',
    'digital twin', 'simulation', 'reinforcement learning', 'ai',
    'automation', 'analytics', 'insights', 'data-driven', 'evidence-based',
]

# Hook patterns relevant to B2B HR tech / thought leadership content
DOMAIN_HOOK_PATTERNS = [
    'research shows', 'data from', 'study finds', 'new research',
    'evidence suggests', 'according to', 'analysis of', 'report:',
    'case study:', 'lessons from', 'what we learned', 'the science of',
    'why most', 'the real cost', 'the hidden', 'rethinking',
    'beyond engagement', 'beyond surveys', 'the future of',
    'how to measure', 'how to improve', 'how to reduce',
    '📊', '📈', '🔬', '💡', '🧠', '⚡', '🎯',
]

# CTA patterns for B2B HR tech content
DOMAIN_CTA_PATTERNS = [
    r'(?:book a demo|schedule a demo|request a demo|get a demo)',
    r'(?:learn more|read more|discover how|find out how|explore)',
    r'(?:download|get the report|get the guide|get the whitepaper)',
    r'(?:start free|try free|free trial|free assessment)',
    r'(?:contact us|talk to|speak with|reach out|connect with)',
    r'(?:sign up|register|join|subscribe|get started)',
    r'(?:see how|watch|view|check out)',
    r'(?:comment below|share your|what do you think|agree\?|thoughts\?)',
    r'(?:👇|🔗|👆|➡️|link in)',
]

class MarketScraperAgent:

    def __init__(self, competitor_profiles_path: Optional[Path] = None):
        self.apify_token = settings.APIFY_API_TOKEN
        self.base_url = "https://api.apify.com/v2"
        self.enabled = settings.ENABLE_SCRAPING
        
        # Cache for competitor scrape results to avoid repeated failures
        self._scrape_cache: Dict[str, Dict[str, Any]] = {}

        if competitor_profiles_path is None:
            csv_path = Path("data/competitors/competitors.csv")
            json_path = Path("data/competitors/competitor_profiles.json")
            
            if csv_path.exists():
                self.competitor_profiles = self._load_competitor_profiles_csv(csv_path)
            elif json_path.exists():
                self.competitor_profiles = self._load_competitor_profiles_json(json_path)
            else:
                logger.warning("No competitor profiles found. Using empty list.")
                self.competitor_profiles = {"competitors": [], "market_landscape": {}}
        else:
            if str(competitor_profiles_path).endswith('.csv'):
                self.competitor_profiles = self._load_competitor_profiles_csv(competitor_profiles_path)
            else:
                self.competitor_profiles = self._load_competitor_profiles_json(competitor_profiles_path)

    def _load_competitor_profiles_csv(self, path: Path) -> Dict[str, Any]:
        """Load competitor profiles from CSV"""
        try:
            import csv
            competitors = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    competitor = {
                        "id": row.get('name', '').lower().replace(' ', '_'),
                        "name": row.get('name', ''),
                        "category": row.get('category', 'competitor'),
                        "website": row.get('url', row.get('website', '')),
                        "description": row.get('description', ''),
                        "key_features": [f.strip() for f in row.get('key_features', '').split(';') if f.strip()],
                        "pricing_model": row.get('pricing_model', ''),
                        "target_market": row.get('target_market', ''),
                        "strengths": [s.strip() for s in row.get('key_features', row.get('strengths', '')).split(';') if s.strip()],
                        "weaknesses": [w.strip() for w in row.get('risky_topics', row.get('weaknesses', '')).split(';') if w.strip()],
                        "typical_claims": [c.strip() for c in row.get('typical_claims', '').split(';') if c.strip()],
                        "differentiation_opportunities": [d.strip() for d in row.get('differentiators_vs_us', row.get('differentiation_opportunities', '')).split(';') if d.strip()],
                    }
                    competitors.append(competitor)
            
            logger.info(f"✅ Loaded {len(competitors)} competitor profiles from CSV")
            return {"competitors": competitors, "market_landscape": {}}
        except Exception as e:
            logger.error(f"Failed to load competitor profiles from CSV: {e}")
            return {"competitors": [], "market_landscape": {}}

    def _load_competitor_profiles_json(self, path: Path) -> Dict[str, Any]:
        """Load competitor profiles from JSON (legacy format)"""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"Loaded {len(profiles.get('competitors', []))} competitor profiles")
                return profiles
            else:
                logger.warning(f"Competitor profiles not found: {path}. Using defaults.")
                return {"competitors": [], "market_landscape": {}}
        except Exception as e:
            logger.error(f"Failed to load competitor profiles: {e}")
            return {"competitors": [], "market_landscape": {}}
    
    async def scrape_linkedin_posts(
        self,
        keywords: List[str],
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Scrape LinkedIn posts via APIFY API, falling back to web scraping.
        Returns empty list if no data source is available.
        """
        if not self.enabled:
            logger.warning("Market scraping disabled — returning empty results")
            return []
        
        if not self.apify_token:
            logger.info("No APIFY token — attempting web scraping fallback")
            return await self._scrape_web_content(keywords, limit)
        
        try:
            actor_id = "apify/linkedin-posts-scraper"
            
            run_input = {
                "searchQuery": " OR ".join(keywords),
                "maxResults": limit,
                "sortBy": "RELEVANCE"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/acts/{actor_id}/runs",
                    headers={"Authorization": f"Bearer {self.apify_token}"},
                    json=run_input,
                    timeout=30.0
                )
                
                if response.status_code != 201:
                    logger.error(f"Failed to start LinkedIn scraper: {response.status_code}")
                    return await self._scrape_web_content(keywords, limit)
                
                run_data = response.json()["data"]
                run_id = run_data["id"]
                
                max_wait = 300
                waited = 0
                
                while waited < max_wait:
                    await asyncio.sleep(10)
                    waited += 10
                    
                    status_response = await client.get(
                        f"{self.base_url}/actor-runs/{run_id}",
                        headers={"Authorization": f"Bearer {self.apify_token}"},
                        timeout=10.0
                    )
                    
                    status_data = status_response.json()["data"]
                    
                    if status_data["status"] == "SUCCEEDED":
                        dataset_id = status_data["defaultDatasetId"]
                        
                        results_response = await client.get(
                            f"{self.base_url}/datasets/{dataset_id}/items",
                            headers={"Authorization": f"Bearer {self.apify_token}"},
                            timeout=30.0
                        )
                        
                        posts = results_response.json()
                        logger.info(f"Scraped {len(posts)} LinkedIn posts via APIFY")
                        return posts
                    
                    elif status_data["status"] in ["FAILED", "ABORTED"]:
                        logger.error(f"APIFY run failed: {status_data['status']}")
                        return await self._scrape_web_content(keywords, limit)
                
                logger.warning("APIFY run timed out — falling back to web scraping")
                return await self._scrape_web_content(keywords, limit)
        
        except Exception as e:
            logger.error(f"Failed to scrape LinkedIn via APIFY: {e}")
            return await self._scrape_web_content(keywords, limit)

    async def get_inspiration_for_campaign(
        self,
        keywords: List[str],
        limit: int = 20,
        platform: str = "linkedin"
    ) -> Dict[str, Any]:
        """
        Gather market intelligence for campaign content generation.
        Scrapes real web sources and analyzes content patterns.
        Returns empty results if no data is available (no demo/mock fallback).
        """
        data_source = "unknown"
        
        if platform.lower() == "linkedin":
            posts = await self.scrape_linkedin_posts(keywords, limit)
            
            if posts:
                if posts[0].get('source') == 'web_scrape':
                    data_source = "web_scrape"
                else:
                    data_source = "apify"
            else:
                data_source = "none"
        else:
            posts = await self._scrape_web_content(keywords, limit)
            data_source = "web_scrape" if posts else "none"
        
        if not posts:
            return {
                "success": False,
                "insights": None,
                "data_source": data_source
            }
        
        insights = await self.analyze_content_patterns(posts)
        await self.store_insights(insights, keywords)
        
        return {
            "success": True,
            "insights": insights,
            "sample_posts": posts[:limit],
            "data_source": data_source
        }

    def _get_scrape_sources(self) -> List[Dict[str, str]]:
        """Load scraping sources from configuration service, falling back to defaults."""
        try:
            raw = _get_config_value('MARKET_SCRAPE_SOURCES', None)
            if raw:
                if isinstance(raw, str):
                    sources = json.loads(raw)
                else:
                    sources = raw
                if isinstance(sources, list) and sources:
                    return sources
        except Exception as e:
            logger.debug(f"Could not load scrape sources from config: {e}")
        return DEFAULT_SCRAPE_SOURCES

    def _extract_generic(self, soup: BeautifulSoup, base_url: str, per_source: int) -> List[Dict[str, str]]:
        """Generic extractor for blog pages — works with most WordPress/CMS sites."""
        from urllib.parse import urljoin
        results = []
        
        # Try structured article elements first
        articles = soup.find_all(['article', 'div'], class_=lambda x: x and any(
            t in x.lower() for t in ('post', 'article', 'card', 'query-loop-item', 'blog', 'entry')
        ))
        if not articles:
            articles = soup.find_all('article')

        for article in articles[:per_source * 2]:
            title_tag = article.find(['h2', 'h3', 'h1'])
            if not title_tag:
                link_tag = article.find('a', href=True)
                if link_tag and len(link_tag.get_text().strip()) > 10:
                    title_tag = link_tag
                else:
                    continue
            title = title_tag.get_text().strip()
            link = article.find('a', href=True)
            url = link['href'] if link else ''
            if url and not url.startswith('http'):
                url = urljoin(base_url, url)
            snippet = ''
            # Look for description/excerpt classes first
            desc_tag = article.find(['p', 'span', 'div'], class_=lambda x: x and any(
                t in str(x).lower() for t in ('excerpt', 'description', 'summary', 'teaser', 'intro')
            ))
            if desc_tag:
                snippet = desc_tag.get_text().strip()
            else:
                p = article.find('p')
                if p:
                    snippet = p.get_text().strip()
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= per_source:
                break
        
        # If structured extraction found nothing, fall back to blog-style links
        if not results:
            seen = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                if not href.startswith('http'):
                    href = urljoin(base_url, href)
                # Only follow links that look like blog posts (contain /blog/, /article/, etc.)
                if not any(seg in href for seg in ('/blog/', '/articles/', '/news/', '/insights/', '/resources/')):
                    continue
                if href in seen or href.rstrip('/') == base_url.rstrip('/'):
                    continue
                text = link.get_text().strip()
                if not text or len(text) < 10 or len(text) > 200:
                    continue
                seen.add(href)
                parent = link.find_parent(['div', 'article', 'li'])
                snippet = ''
                if parent:
                    p = parent.find('p')
                    if p and p.get_text().strip() != text:
                        snippet = p.get_text().strip()
                results.append({"title": text, "url": href, "snippet": snippet})
                if len(results) >= per_source:
                    break
        
        return results

    async def _scrape_web_content(
        self,
        keywords: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Scrape content from configured web sources using keyword-based search.
        Sources are loaded from the configuration service (MARKET_SCRAPE_SOURCES).
        Falls back to homepage scraping if search endpoints fail.
        """
        from urllib.parse import urljoin

        posts = []
        seen_titles = set()
        query = "+".join(kw.replace(" ", "+") for kw in keywords)
        
        sources = self._get_scrape_sources()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        per_source = max(8, limit // max(len(sources), 1) + 2)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                for source in sources:
                    if len(posts) >= limit:
                        break
                    
                    try:
                        url = source["url"].format(query=query)
                        fallback = source.get("fallback", url)
                        
                        response = await client.get(url, headers=headers, follow_redirects=True)
                        if response.status_code != 200:
                            response = await client.get(fallback, headers=headers, follow_redirects=True)
                        if response.status_code != 200:
                            continue
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        raw_articles = self._extract_generic(soup, fallback, per_source)
                        
                        source_name = source.get("name", "Unknown Source")
                        source_category = source.get("category", "industry")
                        
                        for item in raw_articles:
                            if len(posts) >= limit:
                                break
                            
                            title = item["title"]
                            if not title or len(title) < 10:
                                continue
                            
                            title_key = title.strip().lower()[:60]
                            if title_key in seen_titles:
                                continue
                            seen_titles.add(title_key)
                            
                            item_url = item.get("url", "") or fallback
                            if item_url and not item_url.startswith('http'):
                                item_url = urljoin(fallback, item_url)
                            
                            snippet = item.get("snippet", "")
                            full_text = f"{title} {snippet}".lower()
                            matched_keywords = [kw for kw in keywords if kw.lower() in full_text]
                            
                            posts.append({
                                "platform": "blog",
                                "author": source_name,
                                "text": f"{title}. {snippet}" if snippet else title,
                                "url": item_url,
                                "likes": 0,
                                "shares": 0,
                                "comments": 0,
                                "timestamp": datetime.utcnow().isoformat(),
                                "source": "web_scrape",
                                "source_category": source_category,
                                "keywords_matched": matched_keywords if matched_keywords else ["general"],
                            })
                        
                        logger.info(f"Scraped {len([p for p in posts if p['author'] == source_name])} posts from {source_name}")
                    except Exception as e:
                        logger.warning(f"Error scraping {source.get('name', 'unknown')}: {e}")
                        continue
                
                if posts:
                    logger.info(f"Web scraping collected {len(posts)} posts for keywords: {keywords}")
                    return posts
                else:
                    logger.warning("Web scraping found no relevant content from any configured source")
                    return []
                    
        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            return []
    
    def get_competitor_profiles(self) -> List[Dict[str, Any]]:
        """Get all competitor profiles"""
        return self.competitor_profiles.get('competitors', [])

    def get_differentiation_opportunities(self, competitor_id: Optional[str] = None) -> List[str]:
        """
        Get differentiation opportunities against competitors

        Args:
            competitor_id: Optional specific competitor to compare against

        Returns:
            List of differentiation opportunities
        """
        competitors = self.competitor_profiles.get('competitors', [])

        if not competitors:
            return []

        if competitor_id:
            competitor = next(
                (c for c in competitors if c.get('id') == competitor_id),
                None
            )
            if competitor:
                return competitor.get('differentiation_opportunities', [])
            return []

        all_opportunities = []
        for competitor in competitors:
            opportunities = competitor.get('differentiation_opportunities', [])
            all_opportunities.extend(opportunities)

        seen = set()
        unique_opportunities = []
        for opp in all_opportunities:
            if opp not in seen:
                seen.add(opp)
                unique_opportunities.append(opp)

        return unique_opportunities

    async def scrape_competitor_website(
        self,
        competitor_url: str,
        competitor_name: str
    ) -> Dict[str, Any]:
        """
        Scrape competitor website and blog for latest messaging themes.
        Scrapes both the homepage and /blog path for comprehensive intelligence.
        Uses in-memory cache to avoid repeated network failures.
        """
        if not self.enabled:
            logger.warning("Market scraping disabled")
            return {"success": False, "messaging_themes": []}

        # Return cached result if available and fresh (within 1 hour)
        cache_key = competitor_url
        if cache_key in self._scrape_cache:
            cached = self._scrape_cache[cache_key]
            from datetime import datetime as dt
            cached_at = dt.fromisoformat(cached.get('scraped_at', '2000-01-01'))
            if (datetime.utcnow() - cached_at).total_seconds() < 3600:
                logger.debug(f"Using cached scrape for {competitor_name}")
                return cached

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    competitor_url,
                    follow_redirects=True,
                    headers=headers
                )

                headlines = []
                meta_content = ''
                blog_titles = []
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for tag in ['h1', 'h2', 'h3']:
                        headlines.extend([h.get_text().strip() for h in soup.find_all(tag) if h.get_text().strip()])
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    meta_content = meta_desc.get('content', '') if meta_desc else ''
                
                blog_url = competitor_url.rstrip('/') + '/blog'
                try:
                    blog_response = await client.get(blog_url, follow_redirects=True, headers=headers)
                    if blog_response.status_code == 200:
                        blog_soup = BeautifulSoup(blog_response.text, 'html.parser')
                        blog_articles = self._extract_generic(blog_soup, blog_url, 10)
                        blog_titles = [a['title'] for a in blog_articles if a.get('title')]
                except Exception as blog_err:
                    logger.debug(f"Could not scrape blog for {competitor_name}: {blog_err}")

                keywords = self._extract_keywords(headlines + [meta_content] + blog_titles)

                result = {
                    "success": True,
                    "competitor": competitor_name,
                    "url": competitor_url,
                    "messaging_themes": headlines[:10],
                    "blog_topics": blog_titles[:10],
                    "meta_description": meta_content,
                    "keywords": keywords,
                    "scraped_at": datetime.utcnow().isoformat()
                }
                
                # Cache successful results
                self._scrape_cache[cache_key] = result
                return result

        except Exception as e:
            logger.warning(f"Competitor scrape failed for {competitor_name} ({type(e).__name__}: {e})")
            # Return cached data if available (even if stale)
            if cache_key in self._scrape_cache:
                logger.info(f"Using stale cached data for {competitor_name}")
                return self._scrape_cache[cache_key]
            return {"success": False, "messaging_themes": []}

    def _extract_keywords(self, text_list: List[str]) -> List[str]:
        """Extract domain-relevant keywords from text using DOMAIN_KEYWORDS."""
        found_keywords = []
        combined_text = ' '.join(text_list).lower()

        for keyword in DOMAIN_KEYWORDS:
            if keyword in combined_text:
                found_keywords.append(keyword)

        return found_keywords

    async def analyze_competitive_landscape(self) -> Dict[str, Any]:
        """
        Analyze full competitive landscape with scraping

        Returns:
            Comprehensive competitive analysis
        """
        competitors = self.get_competitor_profiles()

        if not competitors:
            return {
                "success": False,
                "message": "No competitor profiles loaded"
            }

        analysis = {
            "total_competitors": len(competitors),
            "market_positions": {},
            "aggregated_strengths": [],
            "aggregated_weaknesses": [],
            "differentiation_opportunities": self.get_differentiation_opportunities(),
            "competitor_details": []
        }

        for comp in competitors:
            position = comp.get('market_position', comp.get('category', 'Unknown'))
            if position not in analysis["market_positions"]:
                analysis["market_positions"][position] = 0
            analysis["market_positions"][position] += 1

            analysis["aggregated_strengths"].extend(comp.get('strengths', []))
            analysis["aggregated_weaknesses"].extend(comp.get('weaknesses', []))

            if self.enabled and comp.get('website'):
                scraped_data = await self.scrape_competitor_website(
                    comp['website'],
                    comp['name']
                )
                comp['latest_messaging'] = scraped_data

            comp_messaging_themes = comp.get('messaging_themes', [])
            if not comp_messaging_themes and comp.get('latest_messaging', {}).get('success'):
                comp_messaging_themes = comp['latest_messaging'].get('messaging_themes', [])

            analysis["competitor_details"].append({
                "name": comp['name'],
                "position": position,
                "strengths_count": len(comp.get('strengths', [])),
                "weaknesses_count": len(comp.get('weaknesses', [])),
                "messaging_themes": comp_messaging_themes,
                "latest_messaging": comp.get('latest_messaging', {})
            })

        analysis["top_common_strengths"] = sorted([
            {"strength": s, "count": analysis["aggregated_strengths"].count(s)}
            for s in set(analysis["aggregated_strengths"])
        ], key=lambda x: x["count"], reverse=True)[:5]

        analysis["top_common_weaknesses"] = sorted([
            {"weakness": w, "count": analysis["aggregated_weaknesses"].count(w)}
            for w in set(analysis["aggregated_weaknesses"])
        ], key=lambda x: x["count"], reverse=True)[:5]

        analysis["success"] = True

        logger.info(f"Analyzed {len(competitors)} competitors")

        return analysis

    def format_competitive_insights_for_content(
        self,
        persona: Optional[str] = None,
        content_patterns: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format competitive insights and market patterns for inclusion in content generation prompts.
        Combines competitor differentiation with scraped content patterns when available.
        """
        sections = []
        
        # Differentiation opportunities from competitor CSV
        differentiation_opps = self.get_differentiation_opportunities()
        if differentiation_opps:
            sections.append("### Competitive Differentiation Opportunities:\n")
            for i, opp in enumerate(differentiation_opps[:5], 1):
                sections.append(f"{i}. {opp}")
            sections.append("\nUse these differentiation points to highlight our unique value proposition.")
        
        # Scraped content patterns from market observation
        if content_patterns:
            top_hooks = content_patterns.get('top_hooks', [])
            if top_hooks:
                sections.append("\n### Market Content Patterns (from live scraping):\n")
                sections.append("Top-performing hooks in the industry:")
                for hook in top_hooks[:5]:
                    text = hook.get('text', '') if isinstance(hook, dict) else str(hook)
                    if text:
                        sections.append(f"- \"{text[:120]}\"")
            
            common_themes = content_patterns.get('common_themes', [])
            if common_themes:
                themes_text = ", ".join(
                    t.get('theme', '') if isinstance(t, dict) else str(t) 
                    for t in common_themes[:8]
                )
                if themes_text:
                    sections.append(f"\nTrending themes: {themes_text}")
            
            top_ctas = content_patterns.get('top_ctas', [])
            if top_ctas:
                ctas_text = ", ".join(
                    t.get('cta', '') if isinstance(t, dict) else str(t) 
                    for t in top_ctas[:5]
                )
                if ctas_text:
                    sections.append(f"Common CTAs: {ctas_text}")
        
        if not sections:
            return "No competitive insights available."
        
        return "\n".join(sections)

    async def analyze_content_patterns(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze scraped posts for content patterns, hooks, and CTAs.
        Uses domain-relevant patterns for the HR tech / Employee Experience industry.
        
        Per Research Plan Section 3 (Simulation Layer): Analyze market content
        to identify high-performing patterns for content generation.
        """
        if not posts:
            return {
                "total_analyzed": 0,
                "top_hooks": [],
                "top_ctas": [],
                "common_themes": [],
                "engagement_patterns": {}
            }
        
        hooks = []
        ctas = []
        themes = []
        
        total_likes = 0
        total_shares = 0
        total_comments = 0
        
        for post in posts:
            text = post.get('text', '')
            likes = post.get('likes', 0)
            shares = post.get('shares', 0)
            comments = post.get('comments', 0)
            engagement = likes + shares + comments
            
            total_likes += likes
            total_shares += shares
            total_comments += comments
            
            sentences = re.split(r'[.!?]\s', text)
            if sentences:
                first_sentence = sentences[0].strip()
                
                is_hook = any(
                    indicator.lower() in first_sentence.lower() 
                    for indicator in DOMAIN_HOOK_PATTERNS
                )
                
                hooks.append({
                    "text": first_sentence[:150],
                    "engagement": engagement,
                    "is_engaging": is_hook or engagement > 100,
                    "source": post.get('author', 'unknown'),
                })
            
            for pattern in DOMAIN_CTA_PATTERNS:
                matches = re.findall(pattern, text.lower())
                for match in matches:
                    ctas.append({
                        "text": match,
                        "engagement": engagement,
                        "platform": post.get('platform', 'unknown')
                    })
            
            # Extract meaningful domain terms (4+ chars), not just any word
            words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
            themes.extend(words)
        
        top_hooks = sorted(hooks, key=lambda x: x['engagement'], reverse=True)[:10]
        
        cta_counts = Counter(c['text'] for c in ctas)
        top_ctas = [{"cta": cta, "count": count} for cta, count in cta_counts.most_common(10)]
        
        theme_counts = Counter(themes)
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'been', 'were', 'what',
            'more', 'your', 'about', 'they', 'will', 'into', 'also', 'than',
            'just', 'like', 'most', 'only', 'other', 'some', 'when', 'very',
            'here', 'each', 'much', 'make', 'does', 'made', 'even', 'many',
            'read', 'need', 'want', 'help', 'best', 'good', 'well', 'know',
        }
        common_themes = [
            {"theme": theme, "count": count} 
            for theme, count in theme_counts.most_common(30)
            if theme not in stop_words
        ][:10]
        
        avg_engagement = (total_likes + total_shares + total_comments) / len(posts) if posts else 0
        
        return {
            "total_analyzed": len(posts),
            "top_hooks": top_hooks,
            "top_ctas": top_ctas,
            "common_themes": common_themes,
            "engagement_patterns": {
                "avg_likes": total_likes / len(posts) if posts else 0,
                "avg_shares": total_shares / len(posts) if posts else 0,
                "avg_comments": total_comments / len(posts) if posts else 0,
                "avg_total_engagement": avg_engagement
            },
            "platforms_analyzed": list(set(p.get('platform', 'unknown') for p in posts))
        }

    async def store_insights(self, insights: Dict[str, Any], keywords: List[str]) -> bool:
        """
        Store scraped insights to database for future reference.
        
        Args:
            insights: Analysis results from analyze_content_patterns
            keywords: Keywords used for scraping
            
        Returns:
            True if stored successfully
        """
        try:
            async with get_async_session() as session:
                scraped_record = ScrapedContent(
                    source="market_intelligence",
                    keywords=", ".join(keywords),
                    insights=insights,
                    scraped_at=datetime.utcnow()
                )
                session.add(scraped_record)
                await session.commit()
                logger.info(f"Stored market insights for keywords: {keywords}")
                return True
        except Exception as e:
            logger.warning(f"Failed to store insights (non-critical): {e}")
            # Non-critical - insights are still returned to user
            return False