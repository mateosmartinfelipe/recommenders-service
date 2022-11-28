from typing import Any, Optional

from aiokafka import AIOKafkaProducer
from aioredis.client import Redis
from fastapi import APIRouter, Depends, Path, status

from ..config import logger, settings
from ..models import Recommendation
from ..repository import services
from ..repository.nfc import (
    InferenceModel,
    get_from_cache,
    get_recommended_items,
    infer,
    send_message,
    set_to_cache,
)

nfc = APIRouter(prefix="/nfc", tags=["NFC base recommendations"])

current_model: Optional[InferenceModel] = None


def get_model():
    return current_model


@nfc.get(
    "/get-items/{user}/",
    response_model=Recommendation,
    status_code=status.HTTP_200_OK,
)
async def get_recommendations(
    *,
    user: int = Path(None, description="User id"),
    max_number_iteration: int = settings.max_num_recommendation,
) -> Recommendation:
    logger.info(f"user::{user}")
    redis_idx = f"model:{settings.models.model_name}|user:{user}|version:{current_model.version}|max_num:{max_number_iteration}"
    if current_model:
        is_cached = await get_from_cache(
            idx=redis_idx, redis=services.redis_client
        )
        if is_cached:
            logger.info(f"Reading form cache")
            logger.info(f"redis_id-> {redis_idx} ")
            recommendations = Recommendation(**is_cached)
        # first check if in cache
        # if not , then infer and cache and return
        else:
            logger.info(f"Running inference")
            prob = await infer(user_id=user, infer_engine=current_model)
            # store to is_cached

            items_prob = await get_recommended_items(
                prob=prob,
                max_number=max_number_iteration,
            )
            logger.debug(f"Output-> {items_prob}")

            recommendations = Recommendation(
                model_name=settings.models.model_name,
                model_version=current_model.version,
                user_id=user,
                items_prob=items_prob,
            )
            logger.info(f"redis_id-> {redis_idx} ")
            await set_to_cache(
                idx=redis_idx,
                redis=services.redis_client,
                recommendations=recommendations,
            )

        logger.info(
            f"kafka consumer-> {settings.services.kafka.kafka_ml_topic_name} "
        )
        await send_message(
            producer=services.kafka_producer,
            settings=settings.services.kafka,
            data=recommendations,
        )
        return recommendations
    else:
        raise ValueError("Not inference engine") from None
