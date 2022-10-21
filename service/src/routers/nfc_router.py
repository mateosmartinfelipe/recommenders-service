from fastapi import APIRouter, Depends, Path, Request

from ..models import Recommendation
from ..repository.nfc import infer

nfc = APIRouter(prefix="/nfc", tags=["NFC base recommendations"])


@nfc.get("/get-items/{user}")  # , response_model=Recommendation)
def get_recommendations(
    *, user: int = Path(None, description="User id"), request: Request
) -> Recommendation:
    global inference_engine
    global items
    d = inference_engine
    out = infer(
        model=inference_engine,
        user=user,
        items=items,
    )
    return Recommendation(user_id=user)
