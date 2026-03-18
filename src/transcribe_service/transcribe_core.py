#!/usr/bin/env python3
import argparse
import logging
import subprocess
import os
import importlib
import importlib.util
from pathlib import Path

APP_DESCRIPTION = (
    "Transcribe audio with automatic backend selection for CPU, NVIDIA CUDA, and AMD ROCm."
)

APP_EPILOG = """Examples:
  PYTHONPATH=src python -m transcribe_service.transcribe_core input.wav
  PYTHONPATH=src python -m transcribe_service.transcribe_core input.wav --backend faster-whisper --device cuda --compute-type float16
  PYTHONPATH=src python -m transcribe_service.transcribe_core input.wav --backend openai-whisper --device cuda

Backend notes:
  auto            Prefer faster-whisper when available; use openai-whisper for ROCm when detected.
  faster-whisper  Best for CPU and NVIDIA CUDA.
  openai-whisper  Best for AMD ROCm via PyTorch ROCm.
"""

LOGGER = logging.getLogger(__name__)

def run_ffmpeg(in_path: Path, out_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        "-c:a", "pcm_s16le",   # 16-bit PCM
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

def transcribe_faster_whisper(
    wav_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(wav_path))
    lines = []
    for s in segments:
        lines.append(s.text.strip())
    info_dict = {
        "backend": "faster-whisper",
        "device": device,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "duration_after_vad": getattr(info, "duration_after_vad", None),
    }
    return "\n".join(lines), info_dict


def transcribe_openai_whisper(
    wav_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
):
    whisper = importlib.import_module("whisper")

    model = whisper.load_model(model_size, device=device)
    fp16 = device == "cuda" and compute_type in {"auto", "float16", "int8_float16"}
    result = model.transcribe(str(wav_path), fp16=fp16)
    text = result.get("text", "").strip()
    info = {
        "backend": "openai-whisper",
        "device": device,
        "language": result.get("language"),
    }
    return text, info


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_error() -> RuntimeError:
    return RuntimeError(
        "No transcription backend is installed. Install at least one:\n"
        "- NVIDIA/CPU path: pip install faster-whisper\n"
        "- AMD ROCm path: install ROCm PyTorch, then pip install openai-whisper"
    )


def is_cuda_library_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "libcublas",
            "libcudnn",
            "libcuda",
            "cannot be loaded",
            "cuda driver",
        )
    )


def detect_runtime(backend: str, device: str):
    has_faster = module_available("faster_whisper")
    has_whisper = module_available("whisper")

    if backend == "faster-whisper":
        if not has_faster:
            raise RuntimeError("Backend 'faster-whisper' requested but package is not installed (pip install faster-whisper).")
        return "faster-whisper", device

    if backend == "openai-whisper":
        if not has_whisper:
            raise RuntimeError("Backend 'openai-whisper' requested but package is not installed (pip install openai-whisper).")
        return "openai-whisper", device

    if device == "cpu":
        if has_faster:
            return "faster-whisper", "cpu"
        if has_whisper:
            return "openai-whisper", "cpu"
        raise dependency_error()

    if device in {"auto", "cuda"}:
        if has_faster:
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    return "faster-whisper", "cuda"
            except Exception:
                pass

        if has_whisper and module_available("torch"):
            try:
                torch = importlib.import_module("torch")

                has_rocm = bool(getattr(torch.version, "hip", None))
                if torch.cuda.is_available() and has_rocm:
                    return "openai-whisper", "cuda"
            except Exception:
                pass

    if has_faster:
        return "faster-whisper", "cpu"
    if has_whisper:
        return "openai-whisper", "cpu"

    raise dependency_error()


def transcribe(wav_path: Path, model_size: str, backend: str, device: str, compute_type: str):
    selected_backend, selected_device = detect_runtime(backend, device)

    if selected_backend == "openai-whisper":
        return transcribe_openai_whisper(wav_path, model_size, selected_device, compute_type)

    try:
        return transcribe_faster_whisper(wav_path, model_size, selected_device, compute_type)
    except RuntimeError as exc:
        should_fallback = (
            selected_backend == "faster-whisper"
            and selected_device == "cuda"
            and device == "auto"
            and is_cuda_library_error(exc)
        )
        if not should_fallback:
            raise

        LOGGER.warning("CUDA runtime unavailable, falling back to CPU for faster-whisper: %s", exc)
        return transcribe_faster_whisper(wav_path, model_size, "cpu", "auto")

def main():
    ap = argparse.ArgumentParser(
        description=APP_DESCRIPTION,
        epilog=APP_EPILOG,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="Input WAV file")
    ap.add_argument(
        "--model",
        default="large-v3",
        help="Model size/name (e.g., tiny|base|small|medium|large-v3)",
    )
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "faster-whisper", "openai-whisper"],
        help="Inference backend",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Target device",
    )
    ap.add_argument(
        "--compute-type",
        default="auto",
        help="Compute type for faster-whisper (auto|int8|float16|int8_float16|float32)",
    )
    args = ap.parse_args()

    in_path = args.input.resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    storage_root = Path(os.getenv("APP_STORAGE_ROOT", "/workspace"))
    work_dir = storage_root / "work"
    transcripts_dir = storage_root / "transcripts"
    work_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    out_file_prefix = in_path.absolute().as_posix().encode("utf-8").hex()[0:10]
    optimized = work_dir / f"{out_file_prefix}.16k_mono.wav"
    out_txt = transcripts_dir / f"{out_file_prefix}.txt"

    print(f"Optimizing audio -> {optimized}")
    run_ffmpeg(in_path, optimized)

    print(
        f"Transcribing ({args.model}, backend={args.backend}, device={args.device}, compute={args.compute_type})..."
    )
    text, info = transcribe(optimized, args.model, args.backend, args.device, args.compute_type)

    out_txt.write_text(text, encoding="utf-8")
    print(f"Wrote transcript -> {out_txt}")
    print(f"Transcription info -> {info}")

if __name__ == "__main__":
    main()
