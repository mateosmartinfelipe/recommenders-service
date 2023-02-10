FROM python:3.9-slim
LABEL maintainer="felipe mateos martin"

COPY requirements.txt requirements.txt

RUN pip install -U --no-cache-dir pip && pip install \
    --no-cache-dir \
    -r requirements.txt

WORKDIR /app/

EXPOSE 8000 5000

COPY src recomender-service
COPY config config
COPY scripts/start.sh start.sh

# Directory to be mapped to the location on the host
# where the mlflow artifacts are stored.
# All unnecessary if using S3 u other provider
#RUN   mkdir -p /mlflow/data

ENTRYPOINT ["./start.sh"]

CMD ["api"]
