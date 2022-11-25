from typing import Optional

from pydantic import BaseSettings


class MlflowConfig(BaseSettings):
    local_host: str = "127.0.0.1"
    # web server
    container_name_web = "mlflow"
    port_web: int = 5000
    local_server_web: str = f"http://{local_host}:{port_web}"
    network_server_web: str = f"http://{container_name_web}:{port_web}"
    # artifacts server
    container_name_artifacts = "minio"
    artifacts_port: int = 9000
    local_server_artifacts: str = f"http://{local_host}:{artifacts_port}"
    network_server_artifacts: str = (
        f"http://{container_name_artifacts}:{port_web}"
    )


class RedisConfig(BaseSettings):
    db: int = 0
    port: int = 6379

    def get_server(self, environment: str):
        local_host: str = "127.0.0.1"
        container_name: str = "redis"
        local_server: str = f"{local_host}"
        network_server: str = f"{container_name}"
        return local_server if environment == "local" else network_server


class KafkaConfig(BaseSettings):
    kafka_ml_topic_name: str = "ml.predictions"
    kafka_consumer_group: str = "group-id"
    limit: int = 10
    max_kafka_message_size: int = 1024 * 1024 * 1

    def get_server(self, environment: str):
        local_host: str = "localhost"
        port: int = 29092
        container_name: str = "kafka"
        local_server: str = f"{local_host}:{port}"
        network_server: str = f"{container_name}:{port}"
        return local_server if environment == "local" else network_server


class MLFlowModelConfig(BaseSettings):
    experiment_name: str
    model_name: str
    stage: str


class Settings(BaseSettings):
    mlflow: MlflowConfig = MlflowConfig()
    redis: RedisConfig = RedisConfig()
    model: MLFlowModelConfig = MLFlowModelConfig(
        experiment_name="nfc_recommender",
        model_name="nfc_recommender.onnx",
        stage="Production",
    )
    kafka: KafkaConfig = KafkaConfig()
    http_timeout: float = 30
    http_pool_size: int = 100
    http_retires: int = 1
    max_num_recommendation: int = 100


settings = Settings()
