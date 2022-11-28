import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra import compose
from pydantic import BaseSettings

# setup loggers
logging_file = (Path(__file__).parent / "logging.ini").resolve().as_posix()
logging.config.fileConfig(logging_file, disable_existing_loggers=False)


config_folder = (Path(__file__).parent.parent / "config").resolve().as_posix()
hydra.initialize_config_dir(config_dir=config_folder)
api_config = compose(config_name="application.yaml")


class MlflowConfig(BaseSettings):
    port_web: int
    local_host: str
    container_name_web: str

    def get_web_server(self, environment: str):
        # web server
        local_server: str = f"http://{self.local_host}:{self.port_web}"
        network_server: str = (
            f"http://{self.container_name_web}:{self.port_web}"
        )
        return local_server if environment == "local" else network_server


class RedisConfig(BaseSettings):
    db: int
    port: int
    local_host: str
    container_name: str

    def get_server(self, environment: str):
        local_server: str = f"{self.local_host}"
        network_server: str = f"{self.container_name}"
        return local_server if environment == "local" else network_server


class KafkaConfig(BaseSettings):
    kafka_ml_topic_name: str
    kafka_consumer_group: str
    limit: int
    max_kafka_message_size: int
    local_port: int
    container_port: int
    local_host: str
    container_name: str

    def get_server(self, environment: str):
        local_server: str = f"{self.local_host}:{self.local_port}"
        network_server: str = f"{self.container_name}:{self.container_port}"
        return local_server if environment == "local" else network_server


class MLFlowModelConfig(BaseSettings):
    experiment_name: str
    model_name: str
    stage: str
    items_file: str


class ServicesConfig(BaseSettings):
    mlflow: MlflowConfig
    redis: RedisConfig
    kafka: KafkaConfig


class Settings(BaseSettings):
    http_timeout: float
    http_pool_size: int
    http_retires: int
    environment: str
    max_num_recommendation: int
    models: MLFlowModelConfig
    services: ServicesConfig


settings = Settings(**api_config)
# logging

# get root logger
logger = logging.getLogger(
    __name__
)  # the __name__ resolve to "main" since we are at the root of the project.
# This will get the root logger since no logger in the configuration has this name.
