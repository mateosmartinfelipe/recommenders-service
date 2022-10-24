import os
from typing import Optional

from pydantic import BaseSettings


class MlflowConfig(BaseSettings):
    uri: str = "http://127.0.0.1:5000"
    items_folder: str = "/Users/felipemateos/mlflow"


class RedisConfig(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    server: str = f"{host}:{port}"


class Kafka(BaseSettings):
    kafka_host: str = "http://127.0.0.1/"
    kafka_ml_topic_name: str = "ml.predictions"
    max_kafka_message_size: int = 1024 * 1024 * 1


class MLFlowModelConfig(BaseSettings):
    name: str
    stage: str
    artifacts_dir: Optional[str] = "/Users/felipemateos/mlflow"


class Settings(BaseSettings):
    mlflow: MlflowConfig = MlflowConfig()
    redis: RedisConfig = RedisConfig()
    model: MLFlowModelConfig = MLFlowModelConfig(
        name="nfc_recommender.onnx", stage="Production"
    )
    kafka: Kafka = None
    http_timeout: float = 30
    http_pool_size: int = 100
    http_retires: int = 1
    max_num_recommendation: int = 100


settings = Settings()
