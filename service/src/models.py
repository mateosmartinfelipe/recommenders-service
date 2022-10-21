from typing import List

from pydantic import BaseModel


class Recommendation(BaseModel):
    user_id: int
    items: List[int] = []

    class Config:
        omr_mode = True
