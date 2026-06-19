"""
Email Platform Simulation
Per Research Plan Section 5.2 - Platform Simulations

Simulates email marketing behavior including:
- Open rates based on subject line quality
- Click-through rates based on content relevance
- Unsubscribe modeling
- Spam filtering effects
- Time-of-day optimization
"""
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import random

from .base_platform import BasePlatform

logger = logging.getLogger(__name__)


class EmailPlatform(BasePlatform):
    """
    Simulates email marketing platform behavior.

    Characteristics:
    - Direct communication channel
    - High personalization potential
    - Deliverability concerns
    - Strong open/click rate patterns by time
    - Unsubscribe risk management
    """

    def __init__(self, env=None, market_env=None):
        """
        Initialize Email platform simulation.

        Args:
            env: SimPy environment
            market_env: MarketingEnvironment instance
        """
        super().__init__(
            name="email",
            base_cpm=0.0,  # Email is typically cost per send, not CPM
            base_cpc=0.0,
            auction_mechanism="none"  # No auction for email
        )
        self.env = env
        self.market_env = market_env

        # Load benchmark rates from config service, falling back to centralized defaults
        from src.config.simulation_defaults import (
            DEFAULT_EMAIL_BENCHMARKS,
            DEFAULT_SPAM_TRIGGERS,
            DEFAULT_URGENCY_WORDS,
            DEFAULT_PERSONALIZATION_TOKENS,
        )
        try:
            from src.config.configuration_service import _get_config_value
            import json as _json
            raw = _get_config_value('EMAIL_BENCHMARK_RATES', None)
            benchmarks = _json.loads(raw) if raw and isinstance(raw, str) else (raw or DEFAULT_EMAIL_BENCHMARKS)
        except Exception:
            benchmarks = DEFAULT_EMAIL_BENCHMARKS

        self.benchmark_open_rate = benchmarks.get("open_rate", DEFAULT_EMAIL_BENCHMARKS["open_rate"])
        self.benchmark_click_rate = benchmarks.get("click_rate", DEFAULT_EMAIL_BENCHMARKS["click_rate"])
        self.benchmark_unsubscribe_rate = benchmarks.get("unsubscribe_rate", DEFAULT_EMAIL_BENCHMARKS["unsubscribe_rate"])
        self.spam_triggers = DEFAULT_SPAM_TRIGGERS
        self.urgency_words = DEFAULT_URGENCY_WORDS
        self.personalization_tokens = DEFAULT_PERSONALIZATION_TOKENS

        self.cost_per_send = 0.01  # €0.01 per email (ESP cost)
        self.cost_per_1000 = 10.0  # €10 per 1000 emails

        self.sender_reputation = 85.0

        self.has_spf = True
        self.has_dkim = True
        self.has_dmarc = True

    def _get_base_cpc(self) -> float:
        return 0.0

    def _get_base_cpm(self) -> float:
        return 0.0

    def _get_deliverability_rate(self) -> float:
        """
        Calculate email deliverability rate based on sender reputation.

        Returns:
            Deliverability rate (0-1)
        """
        base_rate = 0.95

        reputation_factor = self.sender_reputation / 100

        auth_bonus = 0.0
        if self.has_spf:
            auth_bonus += 0.02
        if self.has_dkim:
            auth_bonus += 0.02
        if self.has_dmarc:
            auth_bonus += 0.01

        deliverability = base_rate * reputation_factor + auth_bonus
        return min(0.99, deliverability)

    def _get_optimal_send_hour(self, persona: str = None) -> int:
        """
        Get optimal send hour based on persona.

        B2B email optimal times:
        - Decision makers: Early morning (6-8 AM) or lunch (12-1 PM)
        - Practitioners: Mid-morning (9-11 AM)
        - Researchers: Late afternoon (3-5 PM)

        Returns:
            Optimal hour (0-23)
        """
        if persona == "decision_maker":
            return random.choice([7, 12])
        elif persona == "practitioner":
            return random.choice([9, 10])
        elif persona == "researcher":
            return random.choice([15, 16])
        else:
            return 10  # Default mid-morning

    def _calculate_time_multiplier(self, send_hour: int, send_day: int) -> float:
        """
        Calculate engagement multiplier based on send time.

        Args:
            send_hour: Hour of day (0-23)
            send_day: Day of week (0=Monday, 6=Sunday)

        Returns:
            Time multiplier (0.5-1.5)
        """
        if send_day >= 5:
            base = 0.5
        else:
            base = 1.0

        if 9 <= send_hour <= 11:  # Peak morning
            hour_factor = 1.3
        elif 12 <= send_hour <= 14:  # Lunch time
            hour_factor = 1.1
        elif 6 <= send_hour <= 8:  # Early morning
            hour_factor = 1.2
        elif 15 <= send_hour <= 17:  # Afternoon
            hour_factor = 1.0
        elif 18 <= send_hour <= 20:  # Evening
            hour_factor = 0.8
        else:  # Night
            hour_factor = 0.4

        return base * hour_factor

    def _calculate_subject_line_score(self, subject: str) -> float:
        """
        Calculate subject line effectiveness score.

        Factors:
        - Length (ideal: 30-50 characters)
        - Personalization tokens
        - Urgency words
        - Spam trigger words (negative)

        Args:
            subject: Email subject line

        Returns:
            Score (0-1)
        """
        score = 0.5

        length = len(subject)
        if 30 <= length <= 50:
            score += 0.15
        elif 20 <= length <= 60:
            score += 0.10
        elif length > 80:
            score -= 0.10

        if any(token in subject for token in self.personalization_tokens):
            score += 0.10

        # Urgency words (can help but not spam)
        if any(word in subject.lower() for word in self.urgency_words):
            score += 0.05

        if "?" in subject:
            score += 0.05

        for trigger in self.spam_triggers:
            if trigger.lower() in subject.lower():
                score -= 0.05

        if subject.isupper():
            score -= 0.15

        return max(0.1, min(1.0, score))

    def _calculate_content_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate email content effectiveness score.

        Args:
            content: Email content dictionary

        Returns:
            Score (0-1)
        """
        score = 0.5

        body = content.get("body", "")

        word_count = len(body.split())
        if 150 <= word_count <= 300:
            score += 0.15
        elif 100 <= word_count <= 400:
            score += 0.10
        elif word_count > 500:
            score -= 0.05

        cta = content.get("cta", "")
        if cta:
            score += 0.10

        # Link count (too many is bad)
        link_count = body.count("http") + body.count("www.")
        if 1 <= link_count <= 3:
            score += 0.05
        elif link_count > 5:
            score -= 0.10

        if "{first_name}" in body or "{{" in body:
            score += 0.10

        if any(char.isdigit() for char in body):
            score += 0.05

        return max(0.1, min(1.0, score))

    def simulate_user_behavior(
        self,
        content: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate how a user interacts with an email.

        Args:
            content: Email content (subject, body, cta)
            user_profile: User profile information

        Returns:
            Interaction results including open, click, unsubscribe
        """
        if self.env:
            current_hour = int(self.env.now) % 24
            current_day = (int(self.env.now) // 24) % 7
        else:
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()

        time_multiplier = self._calculate_time_multiplier(current_hour, current_day)

        subject = content.get("subject", content.get("headline", ""))
        subject_score = self._calculate_subject_line_score(subject)

        content_score = self._calculate_content_score(content)

        persona_match = user_profile.get("persona_match", 0.7)

        engagement_history = user_profile.get("email_engagement_rate", 0.5)

        deliverability = self._get_deliverability_rate()

        base_open_rate = self.benchmark_open_rate
        open_probability = (
            base_open_rate
            * subject_score
            * time_multiplier
            * (0.8 + 0.4 * engagement_history)
            * deliverability
        )
        open_probability = min(0.5, max(0.05, open_probability))

        opened = random.random() < open_probability

        clicked = False
        if opened:
            base_click_rate = self.benchmark_click_rate / self.benchmark_open_rate
            click_probability = (
                base_click_rate
                * content_score
                * persona_match
                * (0.7 + 0.6 * engagement_history)
            )
            click_probability = min(0.3, max(0.01, click_probability))
            clicked = random.random() < click_probability

        unsubscribe_probability = self.benchmark_unsubscribe_rate
        if persona_match < 0.5:
            unsubscribe_probability *= 2
        if engagement_history < 0.3:
            unsubscribe_probability *= 1.5

        unsubscribed = random.random() < unsubscribe_probability

        converted = False
        if clicked:
            conversion_probability = 0.05 * persona_match * content_score
            converted = random.random() < conversion_probability

        time_spent = 0
        if opened:
            time_spent = random.randint(10, 60) if clicked else random.randint(5, 20)

        return {
            "delivered": random.random() < deliverability,
            "opened": opened,
            "clicked": clicked,
            "converted": converted,
            "unsubscribed": unsubscribed,
            "time_spent": time_spent,
            "interaction_type": "click" if clicked else ("open" if opened else "impression"),
            "subject_score": subject_score,
            "content_score": content_score,
            "open_rate": open_probability,
            "click_rate": click_probability if opened else 0.0
        }

    def run_email_campaign(self, campaign_config: Dict[str, Any]):
        """
        Run email campaign simulation.

        Args:
            campaign_config: Campaign configuration

        Yields:
            SimPy timeouts for simulation progression
        """
        campaign_id = campaign_config.get("campaign_id", f"email_{len(self.active_campaigns)}")
        list_size = campaign_config.get("list_size", 1000)
        duration_days = campaign_config.get("duration", 1)

        send_waves = campaign_config.get("send_waves", 1)
        sends_per_wave = list_size // send_waves

        campaign = {
            "id": campaign_id,
            "type": "email",
            "config": campaign_config,
            "start_time": self.env.now if self.env else 0,
            "duration": duration_days,
            "list_size": list_size,
            "sent": 0,
            "delivered": 0,
            "bounced": 0,
            "opens": 0,
            "unique_opens": 0,
            "clicks": 0,
            "unique_clicks": 0,
            "conversions": 0,
            "unsubscribes": 0,
            "spam_reports": 0,
            "spent": 0.0,
            "status": "active"
        }

        self.active_campaigns[campaign_id] = campaign

        logger.info(f"Started email campaign {campaign_id} to {list_size} recipients")

        content = {
            "subject": campaign_config.get("subject", "Check out our latest offer"),
            "body": campaign_config.get("body", ""),
            "cta": campaign_config.get("cta", "Learn More")
        }

        for wave in range(send_waves):
            recipients_this_wave = sends_per_wave if wave < send_waves - 1 else list_size - campaign["sent"]

            for i in range(recipients_this_wave):
                user_profile = {
                    "persona_match": campaign_config.get("persona_match", 0.7),
                    "email_engagement_rate": random.uniform(0.2, 0.8)
                }

                interaction = self.simulate_user_behavior(content, user_profile)

                campaign["sent"] += 1
                campaign["spent"] += self.cost_per_send

                if interaction["delivered"]:
                    campaign["delivered"] += 1

                    if interaction["opened"]:
                        campaign["opens"] += 1
                        campaign["unique_opens"] += 1

                        if interaction["clicked"]:
                            campaign["clicks"] += 1
                            campaign["unique_clicks"] += 1

                            if interaction["converted"]:
                                campaign["conversions"] += 1

                    if interaction["unsubscribed"]:
                        campaign["unsubscribes"] += 1
                else:
                    campaign["bounced"] += 1

            if send_waves > 1 and self.env:
                yield self.env.timeout(24)  # Wait 1 day between waves

        campaign["status"] = "completed"

        open_rate = campaign["opens"] / campaign["sent"] * 100 if campaign["sent"] > 0 else 0
        click_rate = campaign["clicks"] / campaign["opens"] * 100 if campaign["opens"] > 0 else 0
        ctr = campaign["clicks"] / campaign["sent"] * 100 if campaign["sent"] > 0 else 0
        conversion_rate = campaign["conversions"] / campaign["clicks"] * 100 if campaign["clicks"] > 0 else 0
        unsubscribe_rate = campaign["unsubscribes"] / campaign["sent"] * 100 if campaign["sent"] > 0 else 0

        logger.info(
            f"Email campaign {campaign_id} completed. "
            f"Sent: {campaign['sent']}, "
            f"Opens: {campaign['opens']} ({open_rate:.1f}%), "
            f"Clicks: {campaign['clicks']} ({ctr:.2f}% CTR), "
            f"Conversions: {campaign['conversions']} ({conversion_rate:.1f}%), "
            f"Unsubscribes: {campaign['unsubscribes']} ({unsubscribe_rate:.2f}%)"
        )

    def get_platform_specific_metrics(self) -> Dict[str, Any]:
        return {
            "platform": "email",
            "cost_per_send": self.cost_per_send,
            "sender_reputation": self.sender_reputation,
            "deliverability_rate": self._get_deliverability_rate(),
            "has_spf": self.has_spf,
            "has_dkim": self.has_dkim,
            "has_dmarc": self.has_dmarc,
            "benchmark_open_rate": self.benchmark_open_rate,
            "benchmark_click_rate": self.benchmark_click_rate,
            "benchmark_unsubscribe_rate": self.benchmark_unsubscribe_rate
        }

    def get_email_campaign_metrics(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get detailed metrics for an email campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Detailed email metrics
        """
        campaign = self.active_campaigns.get(campaign_id)
        if not campaign:
            return {}

        sent = campaign.get("sent", 0)
        delivered = campaign.get("delivered", 0)
        opens = campaign.get("opens", 0)
        clicks = campaign.get("clicks", 0)
        conversions = campaign.get("conversions", 0)
        unsubscribes = campaign.get("unsubscribes", 0)

        return {
            "campaign_id": campaign_id,
            "sent": sent,
            "delivered": delivered,
            "delivery_rate": delivered / sent if sent > 0 else 0,
            "bounced": campaign.get("bounced", 0),
            "bounce_rate": campaign.get("bounced", 0) / sent if sent > 0 else 0,
            "opens": opens,
            "unique_opens": campaign.get("unique_opens", 0),
            "open_rate": opens / delivered if delivered > 0 else 0,
            "clicks": clicks,
            "unique_clicks": campaign.get("unique_clicks", 0),
            "click_rate": clicks / opens if opens > 0 else 0,
            "click_to_open_rate": clicks / opens if opens > 0 else 0,
            "ctr": clicks / sent if sent > 0 else 0,
            "conversions": conversions,
            "conversion_rate": conversions / clicks if clicks > 0 else 0,
            "unsubscribes": unsubscribes,
            "unsubscribe_rate": unsubscribes / sent if sent > 0 else 0,
            "spam_reports": campaign.get("spam_reports", 0),
            "cost": campaign.get("spent", 0),
            "cost_per_open": campaign.get("spent", 0) / opens if opens > 0 else 0,
            "cost_per_click": campaign.get("spent", 0) / clicks if clicks > 0 else 0,
            "cost_per_conversion": campaign.get("spent", 0) / conversions if conversions > 0 else 0
        }

    def update_sender_reputation(self, bounce_rate: float, spam_rate: float, engagement_rate: float):
        """
        Update sender reputation based on campaign performance.

        Args:
            bounce_rate: Bounce rate from recent campaigns
            spam_rate: Spam complaint rate
            engagement_rate: Overall engagement rate
        """
        if bounce_rate > 0.05:
            self.sender_reputation -= (bounce_rate - 0.05) * 100

        if spam_rate > 0.001:
            self.sender_reputation -= spam_rate * 1000

        if engagement_rate > 0.1:
            self.sender_reputation += (engagement_rate - 0.1) * 50

        self.sender_reputation = max(0, min(100, self.sender_reputation))

        logger.info(f"Email sender reputation updated to {self.sender_reputation:.1f}")
