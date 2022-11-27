from datetime import datetime
from typing import List, Tuple

from pydantic import BaseModel


class Recommendation(BaseModel):

    model_name: str
    model_version: str
    user_id: int
    time: str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    items_prob: List[Tuple[int, float]] = []

    class Config:
        omr_mode = True


class KafkaMessage(BaseModel):

    data: Recommendation
    time: str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    class Config:
        omr_mode = True
