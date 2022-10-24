#!/usr/bin/env bash

uvicorn recomender-service.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --preload
    --log-level info \
    --no-access-log
    --reload
