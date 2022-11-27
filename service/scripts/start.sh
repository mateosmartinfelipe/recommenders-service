#!/usr/bin/env bash

uvicorn recomender-service.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --reload \
    --no-access-log
