import random
import string
import time

from fastapi import FastAPI, Request
from fastapi_utils.tasks import repeat_every
from hydra.core.config_store import ConfigStore

from . import PROJECT_NAME
from . import __version__ as VERSION

# config
from .config import logger, settings
from .repository import nfc
from .routers import TAGS_METADATA, nfc_router

# overwrite config arguments
# python src/main.py model_params.run_hp=False

app = FastAPI(title=PROJECT_NAME, version=VERSION, openapi_tags=TAGS_METADATA)
app.include_router(nfc_router.nfc)

logger.info(f"ENVIRONMENT --------- {settings.environment}")


@app.middleware("http")
def log_requests(request: Request, call_next):
    idem = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()

    response = call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    logger.info(
        f"rid={idem} completed_in={formatted_process_time}ms status_code={response}"
    )

    return response


@app.on_event("startup")
@repeat_every(seconds=300)
def startup_event():
    if not nfc_router.current_model:
        logger.info("loading model for the first time at start up....")
        nfc_router.current_model = nfc.download_model(
            settings=settings,
            current_model_version=None,
        )
        logger.info(f"Model loaded :{nfc_router.current_model.version}")
    else:
        new_model = nfc.download_model(
            settings=settings,
            current_model_version=nfc_router.current_model.version,
        )
        if new_model:
            logger.info(
                f"New model available: old {nfc_router.current_model.version} , {new_model.version}"
            )
            nfc_router.current_model = new_model


@app.get("/health")
def healthcheck():
    return "OK"
