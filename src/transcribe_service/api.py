#!/usr/bin/env python3
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .transcribe_core import (
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_SECONDS,
    pool_status,
    run_ffmpeg,
    shutdown_pools,
    transcribe,
)

app = FastAPI(title="Whisper Transcription API", version="1.1.0")


@app.on_event("shutdown")
def _shutdown_pools() -> None:
    shutdown_pools()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisper-api")

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")
DEFAULT_BACKEND = os.getenv("WHISPER_BACKEND", "auto")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

APP_STORAGE_MODE = os.getenv("APP_STORAGE_MODE", "none")
APP_STORAGE_ROOT = Path(os.getenv("APP_STORAGE_ROOT", "/workspace"))
EFFECTIVE_STORAGE_ROOT = Path(
    os.getenv("APP_STORAGE_MOUNT_PATH", str(APP_STORAGE_ROOT)) if APP_STORAGE_MODE == "mounted" else str(APP_STORAGE_ROOT)
)


def _require_env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(f"missing required env: {name}")
    return value


def _validate_storage_config() -> None:
    if APP_STORAGE_MODE not in {"none", "mounted", "s3"}:
        raise RuntimeError(f"invalid APP_STORAGE_MODE: {APP_STORAGE_MODE}")

    if APP_STORAGE_MODE == "mounted":
        mount_path = Path(os.getenv("APP_STORAGE_MOUNT_PATH", str(APP_STORAGE_ROOT)))
        if not mount_path.is_dir():
            raise RuntimeError(f"mount path not found: {mount_path}")

    if APP_STORAGE_MODE == "s3":
        _require_env("APP_STORAGE_S3_ENDPOINT")
        _require_env("APP_STORAGE_S3_BUCKET")
        _require_env("APP_STORAGE_S3_REGION")
        _require_env("APP_STORAGE_S3_PREFIX")
        _require_env("APP_STORAGE_S3_ACCESS_KEY_ID")
        _require_env("APP_STORAGE_S3_SECRET_ACCESS_KEY")


def _s3_client() -> Any:
    endpoint_url = _require_env("APP_STORAGE_S3_ENDPOINT")
    region = _require_env("APP_STORAGE_S3_REGION")
    access_key = _require_env("APP_STORAGE_S3_ACCESS_KEY_ID")
    secret_key = _require_env("APP_STORAGE_S3_SECRET_ACCESS_KEY")
    session_token = os.getenv("APP_STORAGE_S3_SESSION_TOKEN")
    path_style = os.getenv("APP_STORAGE_S3_PATH_STYLE", "false").lower() == "true"

    config = boto3.session.Config(s3={"addressing_style": "path" if path_style else "virtual"})
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        config=config,
    )


def _s3_key(*parts: str) -> str:
    prefix = _require_env("APP_STORAGE_S3_PREFIX").strip("/")
    normalized = "/".join(p.strip("/") for p in parts if p)
    if prefix:
        return f"{prefix}/{normalized}"
    return normalized


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return cleaned or "upload.audio"


_validate_storage_config()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception method=%s path=%s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check container logs for traceback."},
    )


def _ensure_dirs() -> tuple[Path, Path, Path]:
    uploads_dir = EFFECTIVE_STORAGE_ROOT / "uploads"
    work_dir = EFFECTIVE_STORAGE_ROOT / "work"
    transcripts_dir = EFFECTIVE_STORAGE_ROOT / "transcripts"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    return uploads_dir, work_dir, transcripts_dir


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "storage_mode": APP_STORAGE_MODE,
        "storage_root": str(APP_STORAGE_ROOT),
        "effective_storage_root": str(EFFECTIVE_STORAGE_ROOT),
        **pool_status(),
    }


@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    backend: str = Form(DEFAULT_BACKEND),
    device: str = Form(DEFAULT_DEVICE),
    compute_type: str = Form(DEFAULT_COMPUTE_TYPE),
    chunk_seconds: int = Form(DEFAULT_CHUNK_SECONDS),
    chunk_overlap_seconds: int = Form(DEFAULT_CHUNK_OVERLAP_SECONDS),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    request_id = uuid.uuid4().hex
    safe_name = _safe_filename(file.filename)
    uploads_dir, work_dir, transcripts_dir = _ensure_dirs()
    source_path = uploads_dir / f"{request_id}_{safe_name}"
    optimized_path = work_dir / f"{request_id}.16k_mono.wav"
    transcript_path = transcripts_dir / f"{request_id}.txt"

    s3_uploaded_key = None
    s3_optimized_key = None
    s3_transcript_key = None
    started_at = time.monotonic()

    logger.info(
        "transcribe start request_id=%s filename=%s model=%s backend=%s device=%s compute_type=%s "
        "chunk_seconds=%s chunk_overlap_seconds=%s storage_mode=%s",
        request_id,
        file.filename,
        model,
        backend,
        device,
        compute_type,
        chunk_seconds,
        chunk_overlap_seconds,
        APP_STORAGE_MODE,
    )

    try:
        with source_path.open("wb") as source_file:
            shutil.copyfileobj(file.file, source_file)

        logger.info("ffmpeg normalize request_id=%s input=%s output=%s", request_id, source_path, optimized_path)
        run_ffmpeg(source_path, optimized_path)

        logger.info("whisper transcribe request_id=%s optimized=%s", request_id, optimized_path)
        text, info = transcribe(
            optimized_path,
            model,
            backend,
            device,
            compute_type,
            chunk_seconds=chunk_seconds,
            overlap_seconds=chunk_overlap_seconds,
        )
        transcript_path.write_text(text, encoding="utf-8")

        if APP_STORAGE_MODE == "s3":
            s3 = _s3_client()
            bucket = _require_env("APP_STORAGE_S3_BUCKET")

            s3_uploaded_key = _s3_key("uploads", source_path.name)
            s3_optimized_key = _s3_key("work", optimized_path.name)
            s3_transcript_key = _s3_key("transcripts", transcript_path.name)

            s3.upload_file(str(source_path), bucket, s3_uploaded_key)
            s3.upload_file(str(optimized_path), bucket, s3_optimized_key)
            s3.put_object(Bucket=bucket, Key=s3_transcript_key, Body=text.encode("utf-8"), ContentType="text/plain")

        elapsed = time.monotonic() - started_at
        logger.info("transcribe success request_id=%s elapsed_seconds=%.2f", request_id, elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - started_at
        logger.exception(
            "transcribe failed request_id=%s elapsed_seconds=%.2f filename=%s source=%s optimized=%s transcript=%s",
            request_id,
            elapsed,
            file.filename,
            source_path,
            optimized_path,
            transcript_path,
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed (request_id={request_id}): {exc}") from exc
    finally:
        file.file.close()

    return {
        "request_id": request_id,
        "filename": file.filename,
        "text": text,
        "info": info,
        "model": model,
        "backend": backend,
        "device": device,
        "compute_type": compute_type,
        "chunk_seconds": chunk_seconds,
        "chunk_overlap_seconds": chunk_overlap_seconds,
        "storage": {
            "mode": APP_STORAGE_MODE,
            "root": str(APP_STORAGE_ROOT),
            "effective_root": str(EFFECTIVE_STORAGE_ROOT),
            "uploaded_audio": str(source_path),
            "optimized_audio": str(optimized_path),
            "transcript_file": str(transcript_path),
            "s3_uploaded_key": s3_uploaded_key,
            "s3_optimized_key": s3_optimized_key,
            "s3_transcript_key": s3_transcript_key,
        },
    }
