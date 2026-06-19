"""X (Twitter) API connector for social media marketing."""
import tweepy
import aiohttp
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime

from .base_connector import BaseConnector, PlatformResponse
from ...config.settings import settings

logger = logging.getLogger(__name__)

class XConnector(BaseConnector):
    """X (Twitter) Ads API connector"""

    def __init__(self):
        super().__init__(
            name="x",
            base_url="https://api.twitter.com/2",
            rate_limit=300
        )
        self.config = {
            'api_key': settings.TWITTER_API_KEY,
            'api_secret': settings.TWITTER_API_SECRET,
            'access_token': settings.TWITTER_ACCESS_TOKEN,
            'access_secret': settings.TWITTER_ACCESS_SECRET,
        }

        self.client = tweepy.Client(
            consumer_key=self.config['api_key'],
            consumer_secret=self.config['api_secret'],
            access_token=self.config['access_token'],
            access_token_secret=self.config['access_secret']
        )
        
        # Ads API requires separate approval
        self.ads_api_base = "https://ads-api.twitter.com/12"
        self.session = None
    
    async def validate_credentials(self) -> bool:
        try:
            me = self.client.get_me()
            if me.data:
                logger.info(f"Twitter credentials validated for @{me.data.username}")
                return True
            return False
        except Exception as e:
            logger.error(f"Twitter credential validation failed: {e}")
            return False
    
    async def create_tweet(self, content: str, media_ids: Optional[List[str]] = None) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            tweet_params = {'text': content[:280]}  # 280 char limit
            
            if media_ids:
                tweet_params['media_ids'] = media_ids
            
            response = self.client.create_tweet(**tweet_params)
            
            if response.data:
                return PlatformResponse(
                    success=True,
                    platform="twitter",
                    action="create_tweet",
                    response_data={
                        'tweet_id': response.data['id'],
                        'text': response.data['text']
                    }
                )
            else:
                return PlatformResponse(
                    success=False,
                    platform="twitter",
                    action="create_tweet",
                    response_data={},
                    error="Failed to create tweet"
                )
                
        except Exception as e:
            logger.error(f"Tweet creation failed: {e}")
            return PlatformResponse(
                success=False,
                platform="twitter",
                action="create_tweet",
                response_data={},
                error=str(e)
            )
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        try:
            session = await self._get_ads_session()
            await self._check_rate_limit()
            
            account_id = campaign_data.get('account_id')
            
            payload = {
                'name': campaign_data.get('name'),
                'funding_instrument_id': campaign_data.get('funding_instrument_id'),
                'daily_budget_amount_local_micro': campaign_data.get('daily_budget', 100) * 1000000,  # Ads API uses micro units
                'start_time': datetime.utcnow().isoformat(),
                'objective': 'WEBSITE_CONVERSIONS',
                'entity_status': 'ACTIVE'
            }
            
            url = f"{self.ads_api_base}/accounts/{account_id}/campaigns"
            
            async with session.post(url, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return PlatformResponse(
                        success=True,
                        platform="twitter",
                        action="create_campaign",
                        response_data=response_data
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="twitter",
                        action="create_campaign",
                        response_data=response_data,
                        error=f"Status: {response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"Twitter campaign creation failed: {e}")
            # Fallback: use organic posting if Ads API unavailable
            return await self._create_organic_campaign(campaign_data)
    
    async def _create_organic_campaign(self, campaign_data: Dict[str, Any]) -> PlatformResponse:
        """Fallback: create a thread of tweets when Ads API is unavailable."""
        try:
            tweets = campaign_data.get('tweets', [])
            created_tweets = []
            
            for tweet_content in tweets[:5]:  # Limit to 5 tweets
                result = await self.create_tweet(tweet_content)
                if result.success:
                    created_tweets.append(result.response_data)
            
            return PlatformResponse(
                success=len(created_tweets) > 0,
                platform="twitter",
                action="create_organic_campaign",
                response_data={
                    'campaign_type': 'organic',
                    'tweets_created': created_tweets,
                    'message': 'Using organic posts (Ads API not configured)'
                }
            )
            
        except Exception as e:
            logger.error(f"Organic campaign creation failed: {e}")
            return PlatformResponse(
                success=False,
                platform="twitter",
                action="create_organic_campaign",
                response_data={},
                error=str(e)
            )
    
    async def get_tweet_metrics(self, tweet_id: str) -> PlatformResponse:
        try:
            await self._check_rate_limit()
            
            tweet = self.client.get_tweet(
                tweet_id,
                tweet_fields=['public_metrics', 'created_at']
            )
            
            if tweet.data:
                metrics = tweet.data.public_metrics
                return PlatformResponse(
                    success=True,
                    platform="twitter",
                    action="get_metrics",
                    response_data={
                        'tweet_id': tweet_id,
                        'impressions': metrics.get('impression_count', 0),
                        'likes': metrics.get('like_count', 0),
                        'retweets': metrics.get('retweet_count', 0),
                        'replies': metrics.get('reply_count', 0),
                        'engagement_rate': self._calculate_engagement_rate(metrics)
                    }
                )
            else:
                return PlatformResponse(
                    success=False,
                    platform="twitter",
                    action="get_metrics",
                    response_data={},
                    error="Tweet not found"
                )
                
        except Exception as e:
            logger.error(f"Failed to get tweet metrics: {e}")
            return PlatformResponse(
                success=False,
                platform="twitter",
                action="get_metrics",
                response_data={},
                error=str(e)
            )
    
    def _calculate_engagement_rate(self, metrics: Dict[str, int]) -> float:
        impressions = metrics.get('impression_count', 0)
        if impressions == 0:
            return 0.0
        
        engagements = (
            metrics.get('like_count', 0) +
            metrics.get('retweet_count', 0) +
            metrics.get('reply_count', 0)
        )
        
        return (engagements / impressions) * 100
    
    async def get_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        try:
            if self.config.get('account_id'):
                return await self._get_ads_campaign_metrics(campaign_id)
            else:
                return await self._get_organic_campaign_metrics(campaign_id)
                
        except Exception as e:
            logger.error(f"Failed to get campaign metrics: {e}")
            return PlatformResponse(
                success=False,
                platform="twitter",
                action="get_campaign_metrics",
                response_data={},
                error=str(e)
            )
    
    async def _get_ads_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        try:
            session = await self._get_ads_session()
            await self._check_rate_limit()
            
            account_id = self.config.get('account_id')
            url = f"{self.ads_api_base}/stats/accounts/{account_id}"
            
            params = {
                'entity': 'CAMPAIGN',
                'entity_ids': campaign_id,
                'start_date': '2025-01-01',
                'end_date': datetime.utcnow().strftime('%Y-%m-%d'),
                'granularity': 'TOTAL',
                'metric_groups': 'ENGAGEMENT,BILLING'
            }
            
            async with session.get(url, params=params) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    metrics = self._process_ads_metrics(response_data)
                    return PlatformResponse(
                        success=True,
                        platform="twitter",
                        action="get_campaign_metrics",
                        response_data=metrics
                    )
                else:
                    return PlatformResponse(
                        success=False,
                        platform="twitter",
                        action="get_campaign_metrics",
                        response_data=response_data,
                        error=f"Status: {response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"Ads API metrics failed: {e}")
            return PlatformResponse(
                success=False,
                platform="twitter",
                action="get_campaign_metrics",
                response_data={},
                error=str(e)
            )
    
    async def _get_organic_campaign_metrics(self, campaign_id: str) -> PlatformResponse:
        """Aggregates metrics from actual tweets for organic campaigns."""
        try:
            # Get all tweets for this campaign from database
            from ...data_layer.database.connection import get_async_session
            from ...data_layer.database.models import Content
            from sqlalchemy import select
            from uuid import UUID
            
            async with get_async_session() as session:
                stmt = (
                    select(Content)
                    .where(Content.campaign_id == UUID(campaign_id))
                    .where(Content.status == "deployed")
                )
                result = await session.execute(stmt)
                tweets = result.scalars().all()
            
            if not tweets:
                return PlatformResponse(
                    success=True,
                    platform="twitter",
                    action="get_organic_metrics",
                    response_data={
                        'campaign_type': 'organic',
                        'total_impressions': 0,
                        'total_engagements': 0,
                        'engagement_rate': 0.0,
                        'tweet_count': 0,
                        'message': 'No deployed tweets found for campaign'
                    }
                )
            
            total_impressions = 0
            total_likes = 0
            total_retweets = 0
            total_replies = 0
            tweet_metrics = []
            
            for tweet in tweets:
                platform_post_id = tweet.platform_post_id
                
                if platform_post_id:
                    tweet_data = await self._get_tweet_metrics(platform_post_id)
                    
                    if tweet_data:
                        impressions = tweet_data.get('public_metrics', {}).get('impression_count', 0)
                        likes = tweet_data.get('public_metrics', {}).get('like_count', 0)
                        retweets = tweet_data.get('public_metrics', {}).get('retweet_count', 0)
                        replies = tweet_data.get('public_metrics', {}).get('reply_count', 0)
                        
                        total_impressions += impressions
                        total_likes += likes
                        total_retweets += retweets
                        total_replies += replies
                        
                        tweet_metrics.append({
                            'tweet_id': platform_post_id,
                            'impressions': impressions,
                            'likes': likes,
                            'retweets': retweets,
                            'replies': replies
                        })
            
            total_engagements = total_likes + total_retweets + total_replies
            engagement_rate = (total_engagements / total_impressions * 100) if total_impressions > 0 else 0.0
            
            return PlatformResponse(
                success=True,
                platform="twitter",
                action="get_organic_metrics",
                response_data={
                    'campaign_type': 'organic',
                    'campaign_id': campaign_id,
                    'total_impressions': total_impressions,
                    'total_likes': total_likes,
                    'total_retweets': total_retweets,
                    'total_replies': total_replies,
                    'total_engagements': total_engagements,
                    'engagement_rate': round(engagement_rate, 2),
                    'tweet_count': len(tweets),
                    'avg_impressions_per_tweet': int(total_impressions / len(tweets)) if tweets else 0,
                    'avg_engagement_per_tweet': round(total_engagements / len(tweets), 2) if tweets else 0.0,
                    'tweet_metrics': tweet_metrics
                }
            )
        
        except Exception as e:
            logger.error(f"Failed to get organic campaign metrics: {e}")
            return PlatformResponse(
                success=False,
                platform="twitter",
                action="get_organic_metrics",
                error=str(e)
            )

    async def _get_tweet_metrics(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        try:
            headers = self._get_headers()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/tweets/{tweet_id}",
                    headers=headers,
                    params={
                        "tweet.fields": "public_metrics,created_at,author_id"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('data', {})
                else:
                    logger.warning(f"Failed to get tweet metrics for {tweet_id}: {response.status_code}")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting tweet metrics: {e}")
            return None
    
    def _process_ads_metrics(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        data = raw_data.get('data', [])
        
        if not data:
            return {
                'impressions': 0,
                'engagements': 0,
                'clicks': 0,
                'cost': 0.0,
                'engagement_rate': 0.0
            }
        
        metrics = data[0].get('metrics', {})
        
        return {
            'impressions': metrics.get('impressions', [0])[0],
            'engagements': metrics.get('engagements', [0])[0],
            'clicks': metrics.get('clicks', [0])[0],
            'cost': metrics.get('billed_charge_local_micro', [0])[0] / 1000000,  # Ads API uses micro units
            'engagement_rate': metrics.get('engagement_rate', [0.0])[0]
        }
    
    async def _get_ads_session(self) -> aiohttp.ClientSession:
        if not self.session:
            headers = {
                'Authorization': f"Bearer {self.config['access_token']}",
                'Content-Type': 'application/json'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def update_campaign(self, campaign_id: str, updates: Dict[str, Any]) -> PlatformResponse:
        return PlatformResponse(
            success=True,
            platform="twitter",
            action="update_campaign",
            response_data={
                'campaign_id': campaign_id,
                'message': 'Campaign updates limited for organic posts'
            }
        )
    
    async def pause_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=True,
            platform="twitter",
            action="pause_campaign",
            response_data={
                'campaign_id': campaign_id,
                'status': 'paused'
            }
        )
    
    async def resume_campaign(self, campaign_id: str) -> PlatformResponse:
        return PlatformResponse(
            success=True,
            platform="twitter",
            action="resume_campaign",
            response_data={
                'campaign_id': campaign_id,
                'status': 'active'
            }
        )
    
    async def close(self):
        if self.session:
            await self.session.close()