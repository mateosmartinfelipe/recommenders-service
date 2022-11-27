from typing import Optional

from fastapi import APIRouter, Depends, Path, status

from ..config import logger, settings
from ..models import Recommendation
from ..repository.nfc import (
    InferenceModel,
    get_from_cache,
    get_kafka_producer_fn,
    get_recommended_items,
    get_redis_server_fn,
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
def get_recommendations(
    *,
    user: int = Path(None, description="User id"),
    max_number_iteration: int = settings.max_num_recommendation,
    infer_engine: Optional[InferenceModel] = Depends(get_model),
    redis_server=Depends(get_redis_server_fn),
    kafka_producer=Depends(get_kafka_producer_fn),
) -> Recommendation:
    logger.info(f"user::{user}")
    redis_idx = f"model:{settings.models.model_name}|user:{user}|version:{infer_engine.version}|max_num:{max_number_iteration}"
    if infer_engine:
        is_cached = get_from_cache(idx=redis_idx, redis=redis_server)
        if is_cached:
            logger.info(f"Reading form cache")
            logger.info(f"redis_id-> {redis_idx} ")
            recommendations = Recommendation(**is_cached)
        # first check if in cache
        # if not , then infer and cache and return
        else:
            logger.info(f"Running inference")
            prob = infer(user_id=user, infer_engine=infer_engine)
            # store to is_cached

            items_prob = get_recommended_items(
                prob=prob,
                max_number=max_number_iteration,
            )
            logger.debug(f"Output-> {items_prob}")

            recommendations = Recommendation(
                model_name=settings.models.model_name,
                model_version=infer_engine.version,
                user_id=user,
                items_prob=items_prob,
            )
            logger.info(f"redis_id-> {redis_idx} ")
            set_to_cache(
                idx=redis_idx,
                redis=redis_server,
                recommendations=recommendations,
            )

        logger.info(
            f"kafka consumer-> {settings.services.kafka.kafka_ml_topic_name} "
        )
        send_message(
            producer=kafka_producer,
            settings=settings.services.kafka,
            data=recommendations,
        )
        return recommendations
    else:
        raise ValueError("Not inference engine") from None
