"""Blog CMS API connector — supports WordPress REST API for blog content deployment."""
import aiohttp
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from .base_connector import BaseConnector, PlatformResponse
from ...config.settings import settings

logger = logging.getLogger(__name__)


class BlogConnector(BaseConnector):
    """WordPress REST API connector for blog content publishing."""

    def __init__(self):
        blog_url = getattr(settings, 'BLOG_CMS_URL', 'https://example.com')
        super().__init__(
            name="blog",
            base_url=f"{blog_url}/wp-json/wp/v2",
            rate_limit=100
        )
        self.config = {
            'cms_url': blog_url,
            'api_key': getattr(settings, 'BLOG_API_KEY', ''),
            'username': getattr(settings, 'BLOG_USERNAME', ''),
            'password': getattr(settings, 'BLOG_APP_PASSWORD', ''),
            'default_author': getattr(settings, 'BLOG_DEFAULT_AUTHOR', 'Agentic AI'),
            'default_category': getattr(settings, 'BLOG_DEFAULT_CATEGORY', 'Marketing'),
        }
        self.session = None

    async def validate_credentials(self) -> bool:
        """Validate WordPress API credentials."""
        if not self.config.get('username') or not self.config.get('password'):
            logger.error(
                "Blog CMS credentials not configured. Set BLOG_USERNAME and "
                "BLOG_APP_PASSWORD in System Settings → Blog Configuration"
            )
            return False
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/posts?per_page=1") as response:
                if response.status == 200:
                    logger.info("Blog CMS credentials validated")
                    return True
                elif response.status == 401:
                    logger.error(
                        "Blog CMS authentication failed (401). Verify BLOG_USERNAME "
                        "and BLOG_APP_PASSWORD are correct WordPress application passwords"
                    )
                    return False
                else:
                    logger.warning(f"Blog CMS returned status {response.status}")
                    return response.status < 400
        except Exception as e:
            logger.error(f"Blog CMS credential validation failed: {e}")
            return False

    async def create_post(
        self,
        title: str,
        content: str,
        meta_description: str = "",
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        status: str = "draft",
        featured_image_id: Optional[int] = None,
    ) -> PlatformResponse:
        """Create a new blog post via WordPress REST API."""
        try:
            session = await self._get_session()
            await self.check_rate_limit()

            payload = {
                'title': title,
                'content': content,
                'status': status,  # draft, publish, pending
                'excerpt': meta_description,
            }

            if categories:
                cat_ids = await self._resolve_category_ids(session, categories)
                if cat_ids:
                    payload['categories'] = cat_ids

            if tags:
                tag_ids = await self._resolve_tag_ids(session, tags)
                if tag_ids:
                    payload['tags'] = tag_ids

            if featured_image_id:
                payload['featured_media'] = featured_image_id

            async with session.post(f"{self.base_url}/posts", json=payload) as response:
                response_data = await response.json()

                if response.status in (200, 201):
                    return PlatformResponse(
                        success=True,
                        platform="blog",
                        action="create_post",
                        response_data={
                            'post_id': response_data.get('id'),
                            'title': response_data.get('title', {}).get('rendered', title),
                            'url': response_data.get('link', ''),
                            'status': response_data.get('status', status),
                            'slug': response_data.get('slug', ''),
                        }
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="blog",
                        action="create_post",
                        response_data=response_data,
                        error=f"WordPress API returned status {response.status}"
                    )

        except Exception as e:
            logger.error(f"Blog post creation failed: {e}")
            return PlatformResponse(
                success=False,
                platform="blog",
                action="create_post",
                response_data={},
                error=str(e)
            )

    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        """Create a blog post from campaign data."""
        content = campaign_data.get('content', {})
        return await self.create_post(
            title=content.get('title', campaign_data.get('name', 'Untitled')),
            content=content.get('body', ''),
            meta_description=content.get('meta_description', ''),
            categories=[self.config.get('default_category', 'Marketing')],
            tags=content.get('seo_keywords', '').split(', ') if content.get('seo_keywords') else None,
            status='draft',
        )

    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        """Get blog post metrics (alias for get_post_metrics)."""
        return await self.get_post_metrics(campaign_id)

    async def get_post_metrics(self, post_id: str) -> PlatformResponse:
        """Get blog post metrics (views, comments, etc.)."""
        try:
            session = await self._get_session()
            await self.check_rate_limit()

            async with session.get(f"{self.base_url}/posts/{post_id}") as response:
                if response.status == 200:
                    post_data = await response.json()
                    return PlatformResponse(
                        success=True,
                        platform="blog",
                        action="get_metrics",
                        response_data={
                            'post_id': post_id,
                            'title': post_data.get('title', {}).get('rendered', ''),
                            'status': post_data.get('status', ''),
                            'comment_count': post_data.get('comment_count', 0),
                            'url': post_data.get('link', ''),
                            'modified': post_data.get('modified', ''),
                        }
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="blog",
                        action="get_metrics",
                        response_data={},
                        error=f"Post not found (HTTP {response.status})"
                    )

        except Exception as e:
            logger.error(f"Failed to get blog post metrics: {e}")
            return PlatformResponse(
                success=False,
                platform="blog",
                action="get_metrics",
                response_data={},
                error=str(e)
            )

    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        """Update a blog post."""
        try:
            session = await self._get_session()
            await self.check_rate_limit()

            async with session.put(
                f"{self.base_url}/posts/{campaign_id}",
                json=updates
            ) as response:
                response_data = await response.json()
                return PlatformResponse(
                    success=response.status in (200, 201),
                    platform="blog",
                    action="update_post",
                    response_data=response_data,
                    error=None if response.status in (200, 201) else f"HTTP {response.status}"
                )
        except Exception as e:
            logger.error(f"Blog post update failed: {e}")
            return PlatformResponse(
                success=False,
                platform="blog",
                action="update_post",
                response_data={},
                error=str(e)
            )

    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        """Set blog post to draft (unpublish)."""
        return await self.update_campaign(campaign_id, {'status': 'draft'})

    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        """Publish a draft blog post."""
        return await self.update_campaign(campaign_id, {'status': 'publish'})

    async def _resolve_category_ids(
        self, session: aiohttp.ClientSession, category_names: List[str]
    ) -> List[int]:
        """Resolve category names to WordPress category IDs."""
        ids = []
        try:
            for name in category_names:
                async with session.get(
                    f"{self.base_url}/categories",
                    params={'search': name, 'per_page': 1}
                ) as response:
                    if response.status == 200:
                        cats = await response.json()
                        if cats:
                            ids.append(cats[0]['id'])
        except Exception as e:
            logger.warning(f"Failed to resolve categories: {e}")
        return ids

    async def _resolve_tag_ids(
        self, session: aiohttp.ClientSession, tag_names: List[str]
    ) -> List[int]:
        """Resolve tag names to WordPress tag IDs, creating if needed."""
        ids = []
        try:
            for name in tag_names[:10]:  # Limit to 10 tags
                async with session.get(
                    f"{self.base_url}/tags",
                    params={'search': name.strip(), 'per_page': 1}
                ) as response:
                    if response.status == 200:
                        tags = await response.json()
                        if tags:
                            ids.append(tags[0]['id'])
                        else:
                            # Create tag
                            async with session.post(
                                f"{self.base_url}/tags",
                                json={'name': name.strip()}
                            ) as create_resp:
                                if create_resp.status in (200, 201):
                                    new_tag = await create_resp.json()
                                    ids.append(new_tag['id'])
        except Exception as e:
            logger.warning(f"Failed to resolve tags: {e}")
        return ids

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an authenticated session."""
        if not self.session:
            headers = {'Content-Type': 'application/json'}

            auth = None
            if self.config.get('username') and self.config.get('password'):
                auth = aiohttp.BasicAuth(
                    self.config['username'],
                    self.config['password']
                )

            self.session = aiohttp.ClientSession(headers=headers, auth=auth)
        return self.session

    async def close(self):
        if self.session:
            await self.session.close()
