from typing import Optional

from fastapi import APIRouter, Depends, Path, status

from ..config import settings
from ..models import Recommendation
from ..repository.nfc import (
    InferenceModel,
    get_from_cache,
    get_recommended_items,
    get_redis_server_fn,
    infer,
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
) -> Recommendation:
    redis_idx = f"model:{settings.model.name}_user:{user}_version:{infer_engine.version}_num:{max_number_iteration}"
    if infer_engine:
        is_cached = get_from_cache(idx=redis_idx, redis=redis_server)
        if is_cached:
            return Recommendation(**is_cached)
        # first check if in cache
        # if not , then infer and cache and return
        probs = infer(user_id=user, infer_engine=infer_engine)
        # store to is_cached

        items_prob = get_recommended_items(
            probs=probs,
            items=infer_engine.items,
            max_number=max_number_iteration,
        )
        recommendations = Recommendation(user_id=user, items_prob=items_prob)
        set_to_cache(
            idx=redis_idx, redis=redis_server, recommendations=recommendations
        )
        return recommendations
    else:
        raise ValueError("Not inference engine")
