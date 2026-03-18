FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    APP_STORAGE_MODE=none \
    APP_STORAGE_ROOT=/workspace \
    WHISPER_MODEL=base \
    WHISPER_BACKEND=auto \
    WHISPER_DEVICE=auto \
    WHISPER_COMPUTE_TYPE=auto \
    WHISPER_PORT=8000 \
    TS_STATE_DIR=/var/lib/tailscale \
    TS_HOSTNAME=whisper-api \
    TS_SERVICE_NAME=svc:whisper \
    TS_SERVE_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/api.txt /app/requirements/api.txt
RUN pip install -r /app/requirements/api.txt

COPY src /app/src
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
COPY scripts/validate-storage.sh /app/scripts/validate-storage.sh
COPY scripts/derive-storage-env.sh /app/scripts/derive-storage-env.sh

RUN chmod +x /app/docker/entrypoint.sh /app/scripts/validate-storage.sh /app/scripts/derive-storage-env.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
