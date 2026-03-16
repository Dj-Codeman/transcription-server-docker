#!/usr/bin/env python3
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from transcribe import run_ffmpeg, transcribe

app = FastAPI(title="Whisper Transcription API", version="1.0.0")

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")
DEFAULT_BACKEND = os.getenv("WHISPER_BACKEND", "auto")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    backend: str = Form(DEFAULT_BACKEND),
    device: str = Form(DEFAULT_DEVICE),
    compute_type: str = Form(DEFAULT_COMPUTE_TYPE),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    suffix = Path(file.filename).suffix or ".audio"

    with tempfile.TemporaryDirectory(prefix="tx-api-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_path = tmp_path / f"input{suffix}"
        optimized_path = tmp_path / "optimized.16k_mono.wav"

        try:
            with source_path.open("wb") as source_file:
                shutil.copyfileobj(file.file, source_file)

            run_ffmpeg(source_path, optimized_path)
            text, info = transcribe(optimized_path, model, backend, device, compute_type)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
        finally:
            file.file.close()

    return {
        "filename": file.filename,
        "text": text,
        "info": info,
        "model": model,
        "backend": backend,
        "device": device,
        "compute_type": compute_type,
    }
