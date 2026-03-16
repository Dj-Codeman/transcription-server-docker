FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_MODEL=base \
    WHISPER_BACKEND=auto \
    WHISPER_DEVICE=auto \
    WHISPER_COMPUTE_TYPE=auto \
    WHISPER_PORT=8000 \
    TS_STATE_DIR=/var/lib/tailscale \
    TS_HOSTNAME=whisper-api \
    TS_SERVE_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt /app/requirements-api.txt
RUN pip install -r /app/requirements-api.txt

COPY transcribe.py /app/transcribe.py
COPY api.py /app/api.py
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
