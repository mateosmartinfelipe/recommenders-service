import logging

from fastapi import FastAPI
from fastapi.logger import logger as fastapi_logger
from fastapi_utils.tasks import repeat_every

from . import PROJECT_NAME
from . import __version__ as VERSION
from .config import settings
from .repository import nfc
from .routers import TAGS_METADATA, nfc_router

app = FastAPI(title=PROJECT_NAME, version=VERSION, openapi_tags=TAGS_METADATA)
app.include_router(nfc_router.nfc)


@app.on_event("startup")
@repeat_every(seconds=300)
def startup_event():
    if not nfc_router.current_model:
        nfc_router.current_model = nfc.download_model(
            mlflow_config=settings.mlflow,
            model_config=settings.model,
            current_model_version=None,
        )

    else:
        new_model = nfc.download_model(
            mlflow_config=settings.mlflow,
            model_config=settings.model,
            current_model_version=nfc_router.current_model.version,
        )
        if new_model:
            nfc_router.current_model = new_model


@app.get("/health")
def healthcheck():
    return "OK"
