"""Blog platform simulation — models organic SEO traffic, content engagement, and lead capture."""
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_platform import BasePlatform

logger = logging.getLogger(__name__)


class BlogPlatform(BasePlatform):
    """Simulates blog/content marketing platform (SEO-driven, long-tail traffic, high intent)."""

    def __init__(self, env, market_env):
        super().__init__(
            name="blog",
            base_cpm=0.0,   # Organic — no direct ad spend
            base_cpc=0.0,
            auction_mechanism="organic"
        )
        self.env = env
        self.market_env = market_env

    def _get_base_cpc(self) -> float:
        return 0.0

    def _get_base_cpm(self) -> float:
        return 0.0

    def _get_platform_fee(self) -> float:
        return 0.0

    def _get_peak_multiplier(self, hour: int) -> float:
        """Blog traffic peaks during business hours (9am-5pm) with a smaller evening bump."""
        if 9 <= hour <= 12:
            return 1.4
        elif 13 <= hour <= 17:
            return 1.2
        elif 19 <= hour <= 22:
            return 1.1
        elif 0 <= hour <= 5:
            return 0.4
        else:
            return 0.8

    def _calculate_seo_score(self, campaign: Dict) -> float:
        """Calculate SEO effectiveness [0.5-2.0] based on content quality and keywords."""
        score = 1.0
        config = campaign.get('config', {})

        keywords = config.get('keywords', [])
        if keywords:
            score += min(len(keywords) * 0.1, 0.3)

        content_quality = config.get('content_quality', 0.7)
        score += (content_quality - 0.5) * 0.8

        word_count = config.get('word_count', 800)
        if 1000 <= word_count <= 2000:
            score += 0.2
        elif word_count < 500:
            score -= 0.2

        return max(0.5, min(score, 2.0))

    def _calculate_our_bid(self, campaign: Dict, customer: Any) -> float:
        """Blog doesn't bid — returns SEO relevance score instead."""
        seo_score = self._calculate_seo_score(campaign)
        content_quality = campaign.get('config', {}).get('content_quality', 0.7)
        return seo_score * content_quality

    def run_campaign(self, campaign_config: Dict):
        """Override base: blog uses organic traffic, no CPM-based impressions."""
        campaign_id = campaign_config.get('campaign_id', f"blog_{len(self.active_campaigns)}")
        duration_hours = campaign_config.get('duration', 7) * 24
        budget = campaign_config.get('budget', 500)

        campaign_data = {
            'id': campaign_id,
            'type': 'blog_post',
            'config': campaign_config,
            'start_time': self.env.now,
            'duration': duration_hours,
            'budget': budget,
            'spent': 0.0,
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'page_views': 0,
            'unique_visitors': 0,
            'avg_time_on_page': 0.0,
            'bounce_rate': 0.0,
            'scroll_depth': 0.0,
            'cta_clicks': 0,
            'leads': 0,
            'shares': 0,
            'comments': 0,
            'backlinks': 0,
            'status': 'active'
        }

        self.active_campaigns[campaign_id] = campaign_data
        logger.info(f"Started blog campaign {campaign_id}")

        seo_score = self._calculate_seo_score(campaign_data)
        content_quality = campaign_config.get('content_quality', 0.7)
        bounce_rate = max(0.2, 0.6 - content_quality * 0.4)
        cta_rate = 0.02 + content_quality * 0.03
        lead_conversion = 0.15 + content_quality * 0.15

        for hour in range(duration_hours):
            peak_mult = self._get_peak_multiplier(hour % 24)
            day = hour // 24
            ramp_factor = min(1.0, 0.1 + (day / 14) * 0.9)
            evergreen_factor = 1.0 + (day / 60) * 0.1 if day > 14 else 1.0

            base_views = 15
            views = int(base_views * peak_mult * seo_score * ramp_factor * evergreen_factor)

            promotion_cost = campaign_config.get('promotion_budget_hourly', 0.5)
            if campaign_data['spent'] + promotion_cost <= budget:
                views = int(views * 1.3)
                campaign_data['spent'] += promotion_cost

            campaign_data['impressions'] += views
            campaign_data['page_views'] += views

            engaged_views = int(views * (1 - bounce_rate))
            campaign_data['unique_visitors'] += int(views * 0.85)

            # Probabilistic clicks: each engaged view has cta_rate chance
            cta_clicks = sum(1 for _ in range(engaged_views) if np.random.random() < cta_rate)
            campaign_data['clicks'] += cta_clicks
            campaign_data['cta_clicks'] += cta_clicks

            # Probabilistic conversions: each click has lead_conversion chance
            leads = sum(1 for _ in range(cta_clicks) if np.random.random() < lead_conversion)
            campaign_data['conversions'] += leads
            campaign_data['leads'] += leads

            if np.random.random() < 0.03 * content_quality:
                campaign_data['shares'] += 1
            if np.random.random() < 0.005 * content_quality and day > 7:
                campaign_data['backlinks'] += 1
            if np.random.random() < 0.01 * content_quality:
                campaign_data['comments'] += 1

            yield self.env.timeout(1)

        campaign_data['status'] = 'completed'
        campaign_data['bounce_rate'] = bounce_rate
        campaign_data['scroll_depth'] = min(0.95, 0.4 + content_quality * 0.5)
        campaign_data['avg_time_on_page'] = 60 + content_quality * 180

        logger.info(
            f"Blog campaign {campaign_id} completed. "
            f"Page views: {campaign_data['page_views']}, "
            f"Clicks: {campaign_data['clicks']}, "
            f"Conversions: {campaign_data['conversions']}, "
            f"Spent: €{campaign_data['spent']:.2f}"
        )

    def get_blog_specific_metrics(self, campaign_id: str) -> Dict:
        """Return blog-specific performance metrics."""
        campaign = self.active_campaigns.get(campaign_id)
        if not campaign:
            return {}

        page_views = max(campaign.get('page_views', 1), 1)

        return {
            'page_views': page_views,
            'unique_visitors': campaign.get('unique_visitors', 0),
            'avg_time_on_page_seconds': campaign.get('avg_time_on_page', 0),
            'bounce_rate': round(campaign.get('bounce_rate', 0), 3),
            'scroll_depth': round(campaign.get('scroll_depth', 0), 3),
            'cta_clicks': campaign.get('cta_clicks', 0),
            'cta_rate': round(campaign.get('cta_clicks', 0) / page_views, 4),
            'leads': campaign.get('leads', 0),
            'lead_conversion_rate': round(campaign.get('leads', 0) / page_views, 4),
            'shares': campaign.get('shares', 0),
            'comments': campaign.get('comments', 0),
            'backlinks': campaign.get('backlinks', 0),
            'cost_per_lead': round(
                campaign.get('spent', 0) / max(campaign.get('leads', 1), 1), 2
            ),
            'seo_score': self._calculate_seo_score(campaign),
            'promotion_spend': round(campaign.get('spent', 0), 2),
        }

    def get_platform_specific_metrics(self) -> Dict[str, Any]:
        return {
            'platform': 'blog',
            'base_cpc': 0.0,
            'base_cpm': 0.0,
            'organic': True,
            'seo_driven': True,
            'avg_ctr': 0.035,
            'avg_time_on_page': 180,
            'avg_bounce_rate': 0.45,
            'avg_lead_conversion': 0.02,
            'evergreen_content': True,
        }

    def simulate_user_behavior(
        self,
        content: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        import random

        content_quality = content.get('quality', 0.7)
        base_engagement = 0.035  # Blog has higher engagement than ads

        # Content type modifiers
        content_type = content.get('type', 'blog_post')
        if content_type == 'long_form':
            base_engagement *= 1.3
        elif content_type == 'listicle':
            base_engagement *= 1.1

        viewed = True
        time_on_page = random.randint(15, int(60 + content_quality * 300)) if viewed else 0
        scrolled = time_on_page > 30
        scroll_depth = min(1.0, time_on_page / 240) if scrolled else 0.0
        clicked_cta = scrolled and random.random() < (0.02 + content_quality * 0.03)
        converted = clicked_cta and random.random() < 0.2
        shared = scrolled and random.random() < 0.01 * content_quality

        return {
            'viewed': viewed,
            'time_on_page': time_on_page,
            'scrolled': scrolled,
            'scroll_depth': round(scroll_depth, 2),
            'clicked_cta': clicked_cta,
            'converted': converted,
            'shared': shared,
            'interaction_type': 'conversion' if converted else (
                'cta_click' if clicked_cta else (
                    'read' if scrolled else 'bounce'
                )
            )
        }
