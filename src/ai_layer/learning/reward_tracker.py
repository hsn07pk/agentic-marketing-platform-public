import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID
import asyncio
from sqlalchemy import select, update, func, case

from ...data_layer.database.connection import get_async_session
from ...data_layer.database.models import DelayedReward, Campaign, Metric
from ...automation_layer.connectors.calendar_api import CalendarAPIConnector
from ...automation_layer.connectors.hubspot_api import HubSpotAPIConnector
from ...config.settings import settings

logger = logging.getLogger(__name__)

class RewardTracker:
    """
    Tracks delayed rewards for campaign conversions.

    Research Plan Section 2.3: Handling Delayed Rewards
    - Uses a pending ledger that credits the original action when a booking arrives
    - Provides surrogate reward (CTR × estimated_conversion_rate) during delay window
    """

    DEFAULT_ESTIMATED_CONVERSION_RATE = 0.10

    def __init__(self):
        self.calendar_api = CalendarAPIConnector()
        self.hubspot_api = HubSpotAPIConnector()
        self.reward_window_hours = settings.REWARD_DELAY_WINDOW_HOURS
        self.estimated_conversion_rate = getattr(
            settings, 'ESTIMATED_CONVERSION_RATE', self.DEFAULT_ESTIMATED_CONVERSION_RATE
        )

    def calculate_surrogate_reward(
        self,
        ctr: float,
        estimated_conversion_rate: Optional[float] = None
    ) -> float:
        """
        Research Plan Section 2.3:
        "The system will handle this using a pending ledger that credits the
        original action when a booking arrives and a surrogate reward
        (e.g., CTR × estimated_conversion_rate) during the delay window."

        Formula: surrogate_reward = CTR × estimated_conversion_rate
        """
        conv_rate = estimated_conversion_rate or self.estimated_conversion_rate
        surrogate = ctr * conv_rate

        logger.debug(
            f"Calculated surrogate reward: CTR={ctr:.4f} × CVR={conv_rate:.4f} = {surrogate:.6f}"
        )

        return surrogate

    def calculate_reward_with_surrogate(
        self,
        click_occurred: bool,
        ctr: float,
        has_final_conversion: bool = False,
        conversion_value: float = 10.0
    ) -> Dict[str, Any]:
        """
        Research Plan Section 2.3: During delay window, use surrogate reward.
        When final conversion arrives, use actual reward.
        """
        immediate_reward = 1.0 if click_occurred else 0.0
        surrogate_reward = self.calculate_surrogate_reward(ctr)
        final_reward = conversion_value if has_final_conversion else 0.0

        # Surrogate is used during delay window, replaced by final when known
        if has_final_conversion:
            total_reward = immediate_reward + final_reward
            reward_type = "final"
        else:
            total_reward = immediate_reward + surrogate_reward
            reward_type = "surrogate"

        return {
            "immediate_reward": immediate_reward,
            "surrogate_reward": surrogate_reward,
            "final_reward": final_reward,
            "total_reward": total_reward,
            "reward_type": reward_type,
            "formula": f"CTR({ctr:.4f}) × CVR({self.estimated_conversion_rate:.4f}) = {surrogate_reward:.6f}"
        }

    async def register_conversion(
        self,
        campaign_id: str,
        lead_email: str,
        lead_data: Dict[str, Any],
        initial_reward: float = 1.0
    ) -> bool:
        
        try:
            async with get_async_session() as session:
                delayed_reward = DelayedReward(
                    campaign_id=UUID(campaign_id),
                    lead_email=lead_email,
                    lead_data=lead_data,
                    initial_reward=initial_reward,
                    current_reward=initial_reward,
                    status="pending",
                    registered_at=datetime.utcnow()
                )
                
                session.add(delayed_reward)
                await session.commit()
                
                logger.info(f"Registered conversion for campaign {campaign_id}: {lead_email}")
                
                await self.hubspot_api.track_campaign_conversion(
                    campaign_id,
                    lead_email,
                    lead_data
                )
                
                return True
        
        except Exception as e:
            logger.error(f"Failed to register conversion: {e}")
            return False
    
    async def check_for_booking(
        self,
        lead_email: str
    ) -> Optional[Dict[str, Any]]:
        
        try:
            response = await self.calendar_api.track_conversion_to_booking(
                campaign_id="",
                lead_email=lead_email
            )
            
            if response.success and response.data.get("matched"):
                return response.data
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to check booking: {e}")
            return None
    
    async def update_reward_with_booking(
        self,
        reward_id: str,
        booking_data: Dict[str, Any],
        bonus_reward: float = 10.0
    ) -> bool:
        
        try:
            async with get_async_session() as session:
                stmt = select(DelayedReward).where(DelayedReward.id == UUID(reward_id))
                result = await session.execute(stmt)
                reward = result.scalar_one_or_none()
                
                if not reward:
                    return False
                
                reward.current_reward = reward.initial_reward + bonus_reward
                reward.status = "booked"
                reward.booking_data = booking_data
                reward.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(f"Updated reward {reward_id} with booking bonus")
                
                await self._update_campaign_metrics(
                    str(reward.campaign_id),
                    demos_booked=1
                )
                
                return True
        
        except Exception as e:
            logger.error(f"Failed to update reward: {e}")
            return False
    
    async def process_pending_rewards(self) -> Dict[str, int]:
        
        try:
            stats = {
                "checked": 0,
                "booked": 0,
                "expired": 0
            }
            
            async with get_async_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(hours=self.reward_window_hours)
                
                stmt = select(DelayedReward).where(
                    DelayedReward.status == "pending",
                    DelayedReward.registered_at >= cutoff_date
                )
                result = await session.execute(stmt)
                pending_rewards = result.scalars().all()
                
                stats["checked"] = len(pending_rewards)
                
                for reward in pending_rewards:
                    booking = await self.check_for_booking(reward.lead_email)
                    
                    if booking:
                        await self.update_reward_with_booking(
                            str(reward.id),
                            booking
                        )
                        stats["booked"] += 1
                    
                    elif reward.registered_at < cutoff_date:
                        reward.status = "expired"
                        stats["expired"] += 1
                
                await session.commit()
                
                logger.info(f"Processed pending rewards: {stats}")
                return stats
        
        except Exception as e:
            logger.error(f"Failed to process pending rewards: {e}")
            return {"checked": 0, "booked": 0, "expired": 0}
    
    async def get_campaign_delayed_rewards(
        self,
        campaign_id: str
    ) -> Dict[str, Any]:
        
        try:
            async with get_async_session() as session:
                stmt = (
                    select(
                        func.count(DelayedReward.id).label("total"),
                        func.sum(
                            case((DelayedReward.status == "booked", 1), else_=0)
                        ).label("booked"),
                        func.avg(DelayedReward.current_reward).label("avg_reward")
                    )
                    .where(DelayedReward.campaign_id == UUID(campaign_id))
                )
                
                result = await session.execute(stmt)
                row = result.first()
                
                return {
                    "total_conversions": row.total or 0,
                    "demos_booked": row.booked or 0,
                    "avg_reward": float(row.avg_reward or 0.0),
                    "conversion_to_demo_rate": (
                        (row.booked / row.total * 100) if row.total else 0.0
                    )
                }
        
        except Exception as e:
            logger.error(f"Failed to get delayed rewards: {e}")
            return {
                "total_conversions": 0,
                "demos_booked": 0,
                "avg_reward": 0.0,
                "conversion_to_demo_rate": 0.0
            }
    
    async def _update_campaign_metrics(
        self,
        campaign_id: str,
        demos_booked: int = 0
    ) -> bool:
        
        try:
            async with get_async_session() as session:
                stmt = (
                    update(Campaign)
                    .where(Campaign.id == UUID(campaign_id))
                    .values(demos_booked=Campaign.demos_booked + demos_booked)
                )
                
                await session.execute(stmt)
                await session.commit()
                
                return True
        
        except Exception as e:
            logger.error(f"Failed to update campaign metrics: {e}")
            return False
    
    async def start_background_processor(self):

        while True:
            try:
                await self.process_pending_rewards()
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                await asyncio.sleep(300)

    async def process_booking_conversion(
        self,
        lead_email: str,
        booking_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a booking conversion from Cal.com webhook.
        Called by /funnel/webhooks/calendar on BOOKING_CREATED events.
        """
        try:
            result = {
                "success": False,
                "lead_email": lead_email,
                "campaign_matched": False,
                "campaign_id": None,
                "reward_updated": False
            }

            async with get_async_session() as session:
                stmt = select(DelayedReward).where(
                    DelayedReward.lead_email == lead_email,
                    DelayedReward.status == "pending"
                ).order_by(DelayedReward.registered_at.desc()).limit(1)

                query_result = await session.execute(stmt)
                reward = query_result.scalar_one_or_none()

                if reward:
                    result["campaign_matched"] = True
                    result["campaign_id"] = str(reward.campaign_id)

                    # 10x multiplier for demo booking
                    booking_bonus = 10.0
                    reward.status = "booked"
                    reward.current_reward = reward.initial_reward + booking_bonus
                    reward.booking_data = {
                        "booking_uid": booking_data.get("uid"),
                        "event_type": booking_data.get("eventType", {}).get("title"),
                        "start_time": booking_data.get("startTime"),
                        "end_time": booking_data.get("endTime"),
                        "attendees": [
                            a.get("email") for a in booking_data.get("attendees", [])
                        ],
                        "booked_at": datetime.utcnow().isoformat()
                    }
                    reward.updated_at = datetime.utcnow()

                    from sqlalchemy import update as sql_update
                    update_stmt = (
                        sql_update(Campaign)
                        .where(Campaign.id == reward.campaign_id)
                        .values(demos_booked=Campaign.demos_booked + 1)
                    )
                    await session.execute(update_stmt)

                    await session.commit()

                    result["success"] = True
                    result["reward_updated"] = True
                    result["new_reward_value"] = reward.current_reward

                    logger.info(
                        f"Processed booking conversion for {lead_email}",
                        extra={
                            "event": "booking_conversion_processed",
                            "lead_email": lead_email,
                            "campaign_id": str(reward.campaign_id),
                            "reward_value": reward.current_reward
                        }
                    )

                    await self._update_experiment_from_booking(
                        session,
                        reward.campaign_id,
                        booking_data
                    )

                else:
                    logger.warning(
                        f"Booking received but no pending reward found for {lead_email}",
                        extra={
                            "event": "booking_no_match",
                            "lead_email": lead_email,
                            "booking_uid": booking_data.get("uid")
                        }
                    )

                    result["success"] = True  # Webhook processed successfully
                    result["message"] = "No pending reward found for this email"

            return result

        except Exception as e:
            logger.error(f"Failed to process booking conversion: {e}")
            return {
                "success": False,
                "error": str(e),
                "lead_email": lead_email
            }

    async def _update_experiment_from_booking(
        self,
        session,
        campaign_id: UUID,
        booking_data: Dict[str, Any]
    ):
        """Connect booking conversions to the bandit learning system."""
        try:
            from ...data_layer.database.models import Experiment, BanditArm

            stmt = select(Experiment).where(Experiment.campaign_id == campaign_id)
            result = await session.execute(stmt)
            experiment = result.scalar_one_or_none()

            if not experiment:
                return

            arm_stmt = select(BanditArm).where(
                BanditArm.experiment_id == experiment.id
            ).order_by(BanditArm.last_pulled_at.desc()).limit(1)

            arm_result = await session.execute(arm_stmt)
            arm = arm_result.scalar_one_or_none()

            if arm:
                # Booking is a strong positive signal
                arm.successes = (arm.successes or 0) + 1
                arm.total_reward = (arm.total_reward or 0) + 10.0
                arm.alpha = (arm.alpha or 1.0) + 1  # Bayesian update

                experiment.total_conversions = (experiment.total_conversions or 0) + 1

                await session.commit()

                logger.info(
                    f"Updated experiment {experiment.id} arm {arm.arm_id} with booking conversion"
                )

        except Exception as e:
            logger.error(f"Failed to update experiment from booking: {e}")