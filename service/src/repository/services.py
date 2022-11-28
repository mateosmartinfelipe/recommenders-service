from typing import Optional

from aiokafka import AIOKafkaProducer
from aioredis import Redis

redis_client: Optional[Redis] = None
kafka_producer: Optional[AIOKafkaProducer] = None
