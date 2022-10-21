from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every

from . import PROJECT_NAME
from . import __version__ as VERSION
from .config import settings
from .repository import nfc
from .routers import TAGS_METADATA, nfc_router

app = FastAPI(title=PROJECT_NAME, version=VERSION, openapi_tags=TAGS_METADATA)
app.include_router(nfc_router.nfc)

app.run_id = None


@app.on_event("startup")
@repeat_every(seconds=60)
def startup_event():
    artifacts = nfc.download_model(
        mlflow_config=settings.mlflow,
        model_config=nfc.model_config,
        current_model_run_id=app.run_id,
    )
    if artifacts is not None:
        inference_engine, items, app.run_id = artifacts


@app.get("/health")
def healthcheck():
    return "OK"
