import asyncio
import random
import string
import time

import aioredis
from aiokafka import AIOKafkaProducer
from fastapi import FastAPI, Request
from fastapi_utils.tasks import repeat_every

from . import PROJECT_NAME
from . import __version__ as VERSION

# config
from .config import logger, settings
from .repository import nfc, services
from .routers import TAGS_METADATA, nfc_router

app = FastAPI(title=PROJECT_NAME, version=VERSION, openapi_tags=TAGS_METADATA)
app.include_router(nfc_router.nfc)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()

    response = await call_next(request)

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


@app.on_event("startup")
async def init_clients():
    logger.info("Initializing Redis client")
    logger.info(f"{settings.services.redis}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    services.redis_client = aioredis.from_url(
        url=f"redis://{settings.services.redis.get_server(settings.environment)}:{settings.services.redis.port}/{settings.services.redis.db}"
    )
    logger.info("Initializing kafka client producer")
    logger.info(f"{settings.services.kafka}")
    services.kafka_producer = AIOKafkaProducer(
        bootstrap_servers=settings.services.kafka.get_server(
            settings.environment
        )
    )
    await services.kafka_producer.start()


@app.get("/health")
def healthcheck():
    return "OK"
