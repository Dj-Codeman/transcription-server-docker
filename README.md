# Whisper API (Docker + Tailscale)

This service wraps the existing Whisper transcription flow in a simple HTTP API.

## What it does

- Accepts an uploaded audio file (`wav`, `mp3`, `m4a`, etc.)
- Normalizes audio with `ffmpeg` (mono, 16kHz)
- Runs transcription through the existing `transcribe.py` backend selection logic
- Returns transcription text as JSON

## Build

```bash
docker build -t whisper-api .
```

## Run (ephemeral Tailscale)

```bash
docker run --rm \
  -e TS_AUTHKEY=tskey-xxxxx \
  -e TS_HOSTNAME=whisper-api \
  -e TS_SERVE_PORT=8000 \
  -e WHISPER_PORT=8000 \
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
  "filename": "audio.wav",
  "text": "transcribed text here",
  "info": {
    "language": "en"
  },
  "model": "base",
  "backend": "auto",
  "device": "auto",
  "compute_type": "auto"
}
```

## Environment variables

Required:

- `TS_AUTHKEY`: Tailscale auth key

Optional:

- `TS_HOSTNAME` (default: `whisper-api`)
- `TS_SERVE_PORT` (default: `8000`)
- `TS_STATE_DIR` (default: `/var/lib/tailscale`)
- `WHISPER_PORT` (default: `8000`)
- `WHISPER_MODEL` (default: `base`)
- `WHISPER_BACKEND` (default: `auto`)
- `WHISPER_DEVICE` (default: `auto`)
- `WHISPER_COMPUTE_TYPE` (default: `auto`)
