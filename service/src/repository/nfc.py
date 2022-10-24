import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import onnxruntime as rt
import redis
from fastapi import HTTPException, status
from mlflow import MlflowClient
from onnx.onnx_ml_pb2 import ModelProto

from ..config import MlflowConfig, MLFlowModelConfig, RedisConfig, settings
from ..models import Recommendation


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
    filter_string = f"name='{model_conf.name}'"
    all_models = client.search_model_versions(filter_string)
    for model in all_models:
        if model.current_stage == model_conf.stage:
            return model.run_id, model.version
    raise RuntimeError(
        f"{model_conf.name} in stage {model_conf.stage} not found "
    )


def get_items(folder: str, run_id: str) -> Dict[int, int]:
    items = Path(folder) / run_id / "items.pkl"
    return pickle.load(open(items, "rb"))


def download_model(
    mlflow_config: MlflowConfig,
    model_config: MLFlowModelConfig,
    current_model_version: str,
) -> Optional[InferenceModel]:
    client = get_mlflow_server(mlflow_config.uri)
    run_id, version = get_model_info(model_config, client)
    if current_model_version is None or current_model_version != version:
        inference_engine = mlflow.onnx.load_model(
            model_uri=f"{model_config.artifacts_dir}/artifacts/{run_id}/{version}/{model_config.name}"
        )
        items = get_items(mlflow_config.items_folder, run_id)
        items_list = list(items.keys())

        return InferenceModel(
            inference_engine.SerializeToString(), items, items_list, version
        )

    return None


def get_recommended_items(
    probs: List[float], items: Dict[int, int], max_number: int
) -> List[Tuple[int, float]]:
    item_code_prob = zip(probs, list(range(len(probs))))
    item_code_prob = sorted(item_code_prob, key=lambda x: x[0], reverse=True)
    item_prob = [
        (items[code], prob[0]) for prob, code in item_code_prob[:max_number]
    ]
    return item_prob


def infer(user_id: str, infer_engine: InferenceModel):
    onnx_inputs = {
        "user": [user_id] * len(infer_engine.items_list),
        "item": infer_engine.items_list,
    }
    session = rt.InferenceSession(infer_engine.engine)
    prob = session.run(output_names=["recommended"], input_feed=onnx_inputs)
    return prob[0]


def get_redis(settings: RedisConfig):
    def get() -> Any:
        return redis.StrictRedis(
            host=settings.host, port=settings.port, db=settings.db
        )

    return get


def get_from_cache(
    idx: str,
    redis: Callable[..., Any],
) -> Optional[Recommendation]:

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


get_redis_server_fn = get_redis(settings.redis)
