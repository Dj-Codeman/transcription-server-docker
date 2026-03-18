# Whisper API (Docker + Tailscale)

This service wraps the existing Whisper transcription flow in a simple HTTP API.

It supports your storage contract with `APP_STORAGE_MODE=none|mounted|s3` and writes files under `APP_STORAGE_ROOT`.

## What it does

- Accepts an uploaded audio file (`wav`, `mp3`, `m4a`, etc.)
- Normalizes audio with `ffmpeg` (mono, 16kHz)
- Runs transcription through the existing `src/transcribe_service/transcribe_core.py` backend selection logic
- Returns transcription text as JSON

## Build

```bash
docker build -f docker/Dockerfile -t whisper-api .
```

## GPU runtime notes

- Run container with GPU access enabled: `--gpus all`
- Host must have NVIDIA Container Toolkit installed
- Image includes CUDA runtime Python wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) and entrypoint sets `LD_LIBRARY_PATH`
- If `WHISPER_DEVICE=auto` and CUDA libs are unavailable at runtime, transcription falls back to CPU

## Run (ephemeral Tailscale)

### Default storage mode (`none`)

```bash
docker run --rm \
  -e TS_AUTHKEY=tskey-xxxxx \
  -e TS_HOSTNAME=whisper-api \
  -e TS_SERVE_PORT=8000 \
  -e WHISPER_PORT=8000 \
  whisper-api
```

### Mounted persistent storage

```bash
docker run --rm \
  -e TS_AUTHKEY=tskey-xxxxx \
  -e TS_HOSTNAME=whisper-api \
  -e TS_SERVE_PORT=8000 \
  -e WHISPER_PORT=8000 \
  -e APP_STORAGE_MODE=mounted \
  -e APP_STORAGE_ROOT=/workspace \
  -v "$PWD/data:/workspace" \
  whisper-api
```

The container entrypoint will:

1. start `tailscaled`
2. run `tailscale up`
3. start the API server (`uvicorn`)
4. publish the API with `tailscale serve`

## API endpoints

- `GET /health`
- `POST /transcribe` (multipart form upload)

When a file is transcribed, the service stores:

- uploaded audio: `<APP_STORAGE_ROOT>/uploads/...`
- optimized audio: `<APP_STORAGE_ROOT>/work/...`
- transcript text: `<APP_STORAGE_ROOT>/transcripts/...`

In `s3` mode, those artifacts are also uploaded to `${APP_STORAGE_S3_PREFIX}/{uploads|work|transcripts}/...`.

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@/path/to/audio.wav" \
  -F "model=base" \
  -F "backend=auto" \
  -F "device=auto" \
  -F "compute_type=auto"
```

Example response:

```json
{
  "request_id": "b4a8...",
  "filename": "audio.wav",
  "text": "transcribed text here",
  "info": {
    "language": "en"
  },
  "model": "base",
  "backend": "auto",
  "device": "auto",
  "compute_type": "auto",
  "storage": {
    "mode": "mounted",
    "root": "/workspace",
    "uploaded_audio": "/workspace/uploads/...",
    "optimized_audio": "/workspace/work/...",
    "transcript_file": "/workspace/transcripts/...",
    "s3_uploaded_key": null,
    "s3_optimized_key": null,
    "s3_transcript_key": null
  }
}
```

## Environment variables

Required in all runs:

- `TS_AUTHKEY`: Tailscale auth key

Required by storage mode:

- `APP_STORAGE_MODE` (valid: `none|mounted|s3`)
- `APP_STORAGE_ROOT` (default: `/workspace`)

`APP_STORAGE_MODE=mounted` required:

- `APP_STORAGE_MODE=mounted`
- `APP_STORAGE_ROOT` (or set `APP_STORAGE_MOUNT_PATH` explicitly)

`APP_STORAGE_MODE=s3` required (minimal ergonomic input, recommended):

- `APP_STORAGE_MODE=s3`
- `APP_STORAGE_S3_URL` (example: `https://<account>.r2.cloudflarestorage.com/<bucket>`)
- `APP_STORAGE_S3_ACCESS_KEY_ID`
- `APP_STORAGE_S3_SECRET_ACCESS_KEY`

`APP_STORAGE_MODE=s3` required (full explicit input, alternative):

- `APP_STORAGE_MODE=s3`
- `APP_STORAGE_S3_ENDPOINT`
- `APP_STORAGE_S3_BUCKET`
- `APP_STORAGE_S3_REGION`
- `APP_STORAGE_S3_PREFIX`
- `APP_STORAGE_S3_ACCESS_KEY_ID`
- `APP_STORAGE_S3_SECRET_ACCESS_KEY`

Optional:

- `TS_HOSTNAME` (default: `whisper-api`)
- `TS_SERVE_PORT` (default: `8000`)
- `TS_STATE_DIR` (default: `/var/lib/tailscale`)
- `WHISPER_PORT` (default: `8000`)
- `WHISPER_MODEL` (default: `base`)
- `WHISPER_BACKEND` (default: `auto`)
- `WHISPER_DEVICE` (default: `auto`)
- `WHISPER_COMPUTE_TYPE` (default: `auto`)
- `APP_STORAGE_MOUNT_PATH` (mounted mode, default: `APP_STORAGE_ROOT`)
- `APP_STORAGE_READONLY` (mounted mode, `true|false`, default: `false`)
- `APP_STORAGE_S3_PATH_STYLE` (s3 mode, `true|false`)
- `APP_STORAGE_S3_SESSION_TOKEN` (s3 mode)
- `APP_SESSION_ID` (s3 mode, used to derive default prefix)

When `APP_STORAGE_MODE=s3`, the entrypoint runs:

```sh
eval "$(/app/scripts/derive-storage-env.sh --export)"
```

This allows only the minimal S3 vars to be provided; derived defaults are:

- `APP_STORAGE_S3_REGION=auto`
- `APP_STORAGE_S3_PATH_STYLE=true`
- `APP_STORAGE_S3_PREFIX=sessions/<APP_SESSION_ID-or-manual>`

## Quick reference

`none` mode:

```bash
TS_AUTHKEY=tskey-xxxxx
APP_STORAGE_MODE=none
APP_STORAGE_ROOT=/workspace
```

`mounted` mode:

```bash
TS_AUTHKEY=tskey-xxxxx
APP_STORAGE_MODE=mounted
APP_STORAGE_ROOT=/workspace
APP_STORAGE_MOUNT_PATH=/workspace
APP_STORAGE_READONLY=false
```

`s3` mode (minimal ergonomic input):

```bash
TS_AUTHKEY=tskey-xxxxx
APP_STORAGE_MODE=s3
APP_STORAGE_ROOT=/workspace
APP_STORAGE_S3_URL=https://<account>.r2.cloudflarestorage.com/<bucket>
APP_STORAGE_S3_ACCESS_KEY_ID=...
APP_STORAGE_S3_SECRET_ACCESS_KEY=...
APP_SESSION_ID=session-123
```
