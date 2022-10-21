import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import Model
from pydantic import BaseModel

from ..config import MlflowConfig, MLFlowModelConfig


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
    current_model_run_id: str,
) -> Optional[Tuple[Any, Any, Any]]:
    client = get_mlflow_server(mlflow_config.uri)
    run_id, version = get_model_info(model_config, client)
    if current_model_run_id is None or current_model_run_id != run_id:
        inference_engine = mlflow.onnx.load_model(
            # f"models:/{model_config.name}/{version}"
            # model_uri=f"/mlflow/data/artifacts/{run_id}/{model_config.name}"
            model_uri=f"{model_config.artifacts_dir}/artifacts/{run_id}/{version}/{model_config.name}"
        )
        items = get_items(mlflow_config.items_folder, run_id)

        return inference_engine, items, run_id

    return None


def infer(model: Model, user: int, items: Dict[int, int]):
    inference_data = np.array(list(zip([user] * len(items), items.keys())))
    predictions = model.predict(inference_data)
    return predictions


model_config = MLFlowModelConfig(
    name="nfc_recommender.onnx", stage="Production"
)
