import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlflow
import onnxruntime as rt
import redis
from fastapi import HTTPException, status
from kafka import KafkaProducer
from mlflow import MlflowClient
from onnx.onnx_ml_pb2 import ModelProto

from ..config import (
    KafkaConfig,
    MlflowConfig,
    MLFlowModelConfig,
    RedisConfig,
    settings,
)
from ..models import KafkaMessage, Recommendation

ENVIRONMENT = os.getenv("environment")


@dataclass
class InferenceModel:
    engine: str = None
    items: Dict[int, int] = None
    items_list: List[int] = None
    version: str = None


def get_mlflow_server(mlfow_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(mlfow_uri)
    client = mlflow.tracking.MlflowClient()
    return client


def get_model_info(
    model_conf: MLFlowModelConfig, client: MlflowClient
) -> Optional[Tuple[str, int]]:
    filter_string = f"name='{model_conf.model_name}'"
    all_models = client.search_model_versions(filter_string)
    for model in all_models:
        if model.current_stage == model_conf.stage:
            return model.run_id, model.version
    raise RuntimeError(
        f"{model_conf.name} in stage {model_conf.stage} not found "
    )


def download_model(
    mlflow_config: MlflowConfig,
    model_config: MLFlowModelConfig,
    current_model_version: Optional[str],
) -> Optional[InferenceModel]:
    client = get_mlflow_server(mlflow_config.local_server_web)
    run_id, version = get_model_info(model_config, client)
    if current_model_version is None or current_model_version != version:
        experiment = mlflow.get_experiment_by_name(
            model_config.experiment_name
        )
        model_uri = f"{experiment.artifact_location}/{run_id}/artifacts/{model_config.model_name}"
        inference_engine = mlflow.onnx.load_model(model_uri=model_uri)
        items_uri = (
            f"{experiment.artifact_location}/{run_id}/artifacts/items.pkl"
        )
        items_file = mlflow.artifacts.download_artifacts(
            artifact_uri=items_uri
        )
        items = pickle.load(open(items_file, "rb"))
        items_list = items.keys()

        return InferenceModel(
            inference_engine.SerializeToString(), items, items_list, version
        )
    return None


def get_recommended_items(
    probs: List[float], items: Dict[int, int], max_number: int
) -> List[Tuple[int, float]]:
    item_code_prob = zip(probs, list(range(len(probs))))
    item_code_prob = sorted(item_code_prob, key=lambda x: x[0], reverse=True)
    item_prob = [(item, prob[0]) for prob, item in item_code_prob[:max_number]]
    return item_prob


def infer(user_id: str, infer_engine: InferenceModel):
    # user = {"user": [user_id] * len(sample[:2])}
    # items = {"item": sample[:2]}
    user = [user_id] * len(infer_engine.items_list)
    item = list(infer_engine.items_list)
    session = rt.InferenceSession(infer_engine.engine)
    prob = session.run(
        output_names=["recommended"],
        input_feed={"user": user, "item": item},
    )
    return prob[0]


def get_redis(settings: RedisConfig, environment: str):
    def get() -> Any:
        return redis.StrictRedis(
            host=settings.get_server(environment),
            port=settings.port,
            db=settings.db,
        )

    return get


def get_from_cache(
    idx: str,
    redis: Callable[..., Any],
) -> Optional[Dict[Any, Any]]:

    output_json = redis.get(idx)
    if output_json:
        data = json.loads(output_json)
        return data
    return output_json


def set_to_cache(
    idx: str,
    redis,
    recommendations: Recommendation,
) -> None:
    exist = redis.get(idx)
    if exist:
        return HTTPException(
            status_code=status.HTTP_302_FOUND,
            detail=f"Recommendations id {idx} already exist",
        )
    recommendations_to_json = recommendations.json()
    redis.set(idx, recommendations_to_json, 60 * 60 * 24)


def get_kafka_producer(settings: KafkaConfig, environment: str):
    def get():
        kafka_producer = KafkaProducer(
            bootstrap_servers=settings.get_server(environment)
        )
        return kafka_producer

    return get


def send_message(
    producer, settings: KafkaConfig, data: Recommendation
) -> None:
    message = KafkaMessage(
        time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"), data=data
    )
    producer.send(
        topic=settings.kafka_ml_topic_name,
        value=message.json().encode("utf-8"),
    )


get_redis_server_fn = get_redis(settings.redis, ENVIRONMENT)
get_kafka_producer_fn = get_kafka_producer(settings.kafka, ENVIRONMENT)
