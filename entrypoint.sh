#!/bin/sh
set -eu

if [ "${APP_STORAGE_MODE:-none}" = "s3" ]; then
  echo "[start] deriving storage env for s3 mode"
  eval "$(/usr/local/bin/derive-storage-env.sh --export)"
fi

echo "[start] validating storage contract"
/usr/local/bin/validate-storage.sh

CUDA_LIB_DIRS="$(python3 - <<'PY'
import os

paths = []
for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_runtime.lib"):
    try:
        module = __import__(module_name, fromlist=["__name__"])
        module_paths = list(getattr(module, "__path__", []))
        if module_paths:
            paths.extend(module_paths)
            continue

        module_file = getattr(module, "__file__", None)
        if module_file:
            paths.append(os.path.dirname(module_file))
    except Exception:
        pass

print(":".join(dict.fromkeys(paths)))
PY
)"

if [ -n "${CUDA_LIB_DIRS}" ]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
  echo "[start] configured LD_LIBRARY_PATH for CUDA libraries"
fi

TS_SERVICE_NAME="${TS_SERVICE_NAME:-svc:whisper}"

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
tailscale --socket=/tmp/tailscaled.sock serve --yes --bg \
  --service="${TS_SERVICE_NAME}" \
  --http="${TS_SERVE_PORT}" \
  "127.0.0.1:${WHISPER_PORT}"

echo "[start] tailscale status"
tailscale --socket=/tmp/tailscaled.sock status || true

wait "${API_PID}"

kill "${TS_PID}" >/dev/null 2>&1 || true
