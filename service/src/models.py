from typing import List, Tuple

from pydantic import BaseModel


class Recommendation(BaseModel):
    user_id: int
    items_prob: List[Tuple[int, float]] = []

    class Config:
        omr_mode = True
