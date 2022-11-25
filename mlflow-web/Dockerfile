FROM python:3.9-slim
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
COPY scripts/start.sh start.sh
ENTRYPOINT ["./start.sh"]