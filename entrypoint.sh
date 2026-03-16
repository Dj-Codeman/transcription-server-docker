#!/bin/sh
set -eu

mkdir -p "${TS_STATE_DIR}"

if [ -z "${TS_AUTHKEY:-}" ]; then
  echo "TS_AUTHKEY is required"
  exit 1
fi

echo "[start] starting tailscaled"
tailscaled \
  --tun=userspace-networking \
  --state="${TS_STATE_DIR}/tailscaled.state" \
  --socket=/tmp/tailscaled.sock &
TS_PID=$!

sleep 3

echo "[start] bringing tailscale up"
tailscale --socket=/tmp/tailscaled.sock up \
  --auth-key="${TS_AUTHKEY}" \
  --hostname="${TS_HOSTNAME}" \
  --accept-dns=false

echo "[start] starting whisper api"
uvicorn api:app --host 0.0.0.0 --port "${WHISPER_PORT}" &
API_PID=$!

sleep 3

echo "[start] publishing whisper api to tailnet with tailscale serve"
tailscale --socket=/tmp/tailscaled.sock serve --bg "${TS_SERVE_PORT}"

echo "[start] tailscale status"
tailscale --socket=/tmp/tailscaled.sock status || true

wait "${API_PID}"

kill "${TS_PID}" >/dev/null 2>&1 || true
