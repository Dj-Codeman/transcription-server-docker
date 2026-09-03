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
  -e TS_TAGS=tag:whisper-api \
  -e TS_HOSTNAME=whisper-api \
  -e TS_SERVE_PORT=8000 \
  -e WHISPER_PORT=8000 \
  whisper-api
```

### Mounted persistent storage

```bash
docker run --rm \
  -e TS_AUTHKEY=tskey-xxxxx \
  -e TS_TAGS=tag:whisper-api \
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

## Client

A stdlib-only CLI client (`client/transcribe`) is included for use as a
regular system command — no Python dependencies to install, just needs
Python 3.

Install:

```bash
sudo make install-client            # installs to /usr/bin/transcribe
sudo make install-client BINDIR=/usr/local/bin
```

Point it at your running API (once, e.g. in your shell profile, or per-call):

```bash
export TRANSCRIBE_API_URL=http://whisper-api:8000
# or persist it:
mkdir -p ~/.config/transcribe && echo "url = http://whisper-api:8000" > ~/.config/transcribe/config
```

Usage:

```bash
transcribe recording.wav
# -> writes recording.txt next to recording.wav (default: same dir, same
#    basename, transcript extension swapped in)

transcribe recording.wav -o /tmp/out.txt   # custom output path
transcribe recording.wav --stdout          # print instead of writing
transcribe recording.wav --model small --device cuda --backend faster-whisper
transcribe recording.wav --json            # full API response, incl. storage info
transcribe recording.wav --url http://192.168.1.50:8000  # override configured URL
```

Uninstall with `sudo make uninstall-client`.

## API endpoints

- `GET /health` — also reports `cpu_pool_workers` / `gpu_pool_workers` (`null` until that pool has been used; see [`chunk_seconds` / `chunk_overlap_seconds`](#chunk_seconds--chunk_overlap_seconds))
- `POST /transcribe` (multipart form upload)

When a file is transcribed, the service stores:

- uploaded audio: `<APP_STORAGE_ROOT>/uploads/...`
- optimized audio: `<APP_STORAGE_ROOT>/work/...`
- transcript text: `<APP_STORAGE_ROOT>/transcripts/...`

In `s3` mode, those artifacts are also uploaded to `${APP_STORAGE_S3_PREFIX}/{uploads|work|transcripts}/...`.

## Transcription options

These map to the `model` / `backend` / `device` / `compute_type` /
`chunk_seconds` / `chunk_overlap_seconds` form fields on `POST /transcribe`
(and the `WHISPER_MODEL` / `WHISPER_BACKEND` / `WHISPER_DEVICE` /
`WHISPER_COMPUTE_TYPE` / `WHISPER_CHUNK_SECONDS` /
`WHISPER_CHUNK_OVERLAP_SECONDS` env vars that set their defaults).
Logic lives in `src/transcribe_service/transcribe_core.py`.

### `model`

Any Whisper model size/name supported by the selected backend, e.g.:
`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`,
`medium.en`, `large-v1`, `large-v2`, `large-v3`. Larger models are more
accurate but slower and heavier on VRAM/RAM. Not validated server-side —
an unsupported name fails at model-load time inside the backend.

### `backend`

- `auto` (default) — prefer `faster-whisper` if installed; fall back to
  `openai-whisper` (used for AMD ROCm)
- `faster-whisper` — best for CPU and NVIDIA CUDA (requires `pip install faster-whisper`)
- `openai-whisper` — best for AMD ROCm via PyTorch ROCm (requires ROCm PyTorch + `pip install openai-whisper`)

### `device`

- `auto` (default) — use CUDA if a working NVIDIA (`faster-whisper`/ctranslate2) or
  ROCm (`openai-whisper`/torch) GPU is detected, otherwise CPU
- `cpu`
- `cuda`

If `device=auto` selects CUDA for `faster-whisper` but the CUDA runtime
libraries aren't actually loadable at inference time, the request
transparently falls back to CPU (`faster-whisper`, `compute_type=auto`).

### `compute_type`

Passed through to `faster-whisper` (CTranslate2): `auto` (default), `int8`,
`float16`, `int8_float16`, `float32`. For `openai-whisper` this only
controls whether `fp16` is enabled (true when `device=cuda` and
`compute_type` is `auto`, `float16`, or `int8_float16`).

### `chunk_seconds` / `chunk_overlap_seconds`

Long audio is split into `chunk_seconds`-long pieces (default `1200` = 20
min, `WHISPER_CHUNK_SECONDS`) and transcribed in parallel across a pool of
worker processes — real multiprocessing, not threads, so it uses multiple
CPU cores or multiple GPU-resident model instances rather than serializing
on one. Files shorter than `chunk_seconds` are transcribed as a single
chunk.

Each chunk after the first also carries the previous chunk's last
`chunk_overlap_seconds` of audio (default `5`, `WHISPER_CHUNK_OVERLAP_SECONDS`),
so audio near a cut gets transcribed twice — once at a chunk's tail, where
truncation is likeliest, and once with full leading context at the start of
the next chunk. The service detects the duplicated text at each seam and
stitches it back into one continuous transcript.

Worker pool sizing is a server-level concern, not a per-request one — see
`WHISPER_CPU_MAX_WORKERS`, `WHISPER_GPU_MAX_WORKERS`, and
`WHISPER_GPU_VRAM_BUFFER_INSTANCES` below. `GET /health` reports the current
`cpu_pool_workers` / `gpu_pool_workers` (`null` until a pool has been used).

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
    "language": "en",
    "chunks": 3,
    "chunk_seconds": 1200,
    "chunk_overlap_seconds": 5,
    "languages": ["en", "en", "en"]
  },
  "model": "base",
  "backend": "auto",
  "device": "auto",
  "compute_type": "auto",
  "chunk_seconds": 1200,
  "chunk_overlap_seconds": 5,
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
- `TS_TAGS`: Tag(s) to advertise, e.g. `tag:whisper-api` (comma-separated for
  more than one). `tailscale serve --service` refuses to run on an untagged
  node ("service hosts must be tagged nodes"). The tag must already exist in
  your tailnet's ACL `tagOwners`, and the identity behind `TS_AUTHKEY` must be
  allowed to apply it — easiest path is generating the auth key in the admin
  console with the tag pre-selected.

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
- `WHISPER_MODEL` (default: `base`; see [Transcription options](#transcription-options) for valid model names)
- `WHISPER_BACKEND` (default: `auto`; valid: `auto|faster-whisper|openai-whisper`)
- `WHISPER_DEVICE` (default: `auto`; valid: `auto|cpu|cuda`)
- `WHISPER_COMPUTE_TYPE` (default: `auto`; valid: `auto|int8|float16|int8_float16|float32`)
- `WHISPER_CHUNK_SECONDS` (default: `1200` = 20 min)
- `WHISPER_CHUNK_OVERLAP_SECONDS` (default: `5`)
- `WHISPER_CPU_MAX_WORKERS` (default: `cpu_count() // 2`) — size of the CPU worker pool
- `WHISPER_GPU_MAX_WORKERS` (default: unset/uncapped) — optional hard ceiling on the auto-sized GPU worker pool
- `WHISPER_GPU_VRAM_BUFFER_INSTANCES` (default: `1`) — model instances of VRAM headroom to leave free when auto-sizing the GPU pool
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
TS_TAGS=tag:whisper-api
APP_STORAGE_MODE=none
APP_STORAGE_ROOT=/workspace
```

`mounted` mode:

```bash
TS_AUTHKEY=tskey-xxxxx
TS_TAGS=tag:whisper-api
APP_STORAGE_MODE=mounted
APP_STORAGE_ROOT=/workspace
APP_STORAGE_MOUNT_PATH=/workspace
APP_STORAGE_READONLY=false
```

`s3` mode (minimal ergonomic input):

```bash
TS_AUTHKEY=tskey-xxxxx
TS_TAGS=tag:whisper-api
APP_STORAGE_MODE=s3
APP_STORAGE_ROOT=/workspace
APP_STORAGE_S3_URL=https://<account>.r2.cloudflarestorage.com/<bucket>
APP_STORAGE_S3_ACCESS_KEY_ID=...
APP_STORAGE_S3_SECRET_ACCESS_KEY=...
APP_SESSION_ID=session-123
```
