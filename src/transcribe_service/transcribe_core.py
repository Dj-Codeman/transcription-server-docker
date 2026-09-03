#!/usr/bin/env python3
import argparse
import concurrent.futures
import difflib
import importlib
import importlib.util
import logging
import multiprocessing
import os
import shutil
import subprocess
import threading
import wave
from collections import OrderedDict
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

Long inputs are split into chunks and transcribed in parallel worker
processes (see --chunk-seconds / --chunk-overlap-seconds).
"""

LOGGER = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _cpu_default_workers() -> int:
    return max(1, (os.cpu_count() or 1) // 2)


DEFAULT_CHUNK_SECONDS = _int_env("WHISPER_CHUNK_SECONDS", 1200)
DEFAULT_CHUNK_OVERLAP_SECONDS = _int_env("WHISPER_CHUNK_OVERLAP_SECONDS", 5)
CPU_MAX_WORKERS = max(1, _int_env("WHISPER_CPU_MAX_WORKERS", _cpu_default_workers()))
GPU_MAX_WORKERS_CAP = os.getenv("WHISPER_GPU_MAX_WORKERS")
GPU_VRAM_BUFFER_INSTANCES = _int_env("WHISPER_GPU_VRAM_BUFFER_INSTANCES", 1)


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


# ---------------------------------------------------------------------------
# Per-worker model cache. Each ProcessPoolExecutor worker is a separate
# process with its own copy of this module-level state, so a small LRU here
# means a worker reuses a loaded model across chunks/requests instead of
# reloading it from disk every call.
# ---------------------------------------------------------------------------

_MODEL_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_MODEL_CACHE_MAXSIZE = 1
_WORKER_CPU_THREADS = 0


def _cache_get_or_load(key, loader):
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]

    model = loader()
    _MODEL_CACHE[key] = model
    while len(_MODEL_CACHE) > _MODEL_CACHE_MAXSIZE:
        _MODEL_CACHE.popitem(last=False)
    return model


def _get_faster_whisper_model(model_size: str, device: str, compute_type: str, cpu_threads: int):
    key = ("faster-whisper", model_size, device, compute_type, cpu_threads)

    def _load():
        from faster_whisper import WhisperModel

        kwargs = {}
        if device == "cpu" and cpu_threads:
            kwargs["cpu_threads"] = cpu_threads
        return WhisperModel(model_size, device=device, compute_type=compute_type, **kwargs)

    return _cache_get_or_load(key, _load)


def _get_openai_whisper_model(model_size: str, device: str):
    key = ("openai-whisper", model_size, device)

    def _load():
        whisper = importlib.import_module("whisper")
        return whisper.load_model(model_size, device=device)

    return _cache_get_or_load(key, _load)


def transcribe_faster_whisper(
    wav_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
    cpu_threads: int = 0,
):
    model = _get_faster_whisper_model(model_size, device, compute_type, cpu_threads)
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
    model = _get_openai_whisper_model(model_size, device)
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


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------

def probe_wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.getnframes()
        return frames / float(rate) if rate else 0.0


def split_wav(wav_path: Path, chunk_seconds: int, overlap_seconds: int, out_dir: Path) -> list[Path]:
    """Split a mono/16-bit PCM WAV into ~chunk_seconds pieces.

    Every chunk after the first also carries the trailing overlap_seconds of
    audio from the previous chunk, so that audio near a cut is transcribed
    twice (once at a chunk's tail, where truncation is likeliest, and once
    with full leading context in the next chunk). If the file is shorter
    than chunk_seconds, no split is needed and [wav_path] is returned as-is.
    """
    with wave.open(str(wav_path), "rb") as src:
        rate = src.getframerate()
        total_frames = src.getnframes()
        sampwidth = src.getsampwidth()
        nchannels = src.getnchannels()

        chunk_frames = int(chunk_seconds * rate)
        overlap_frames = int(overlap_seconds * rate) if overlap_seconds > 0 else 0

        if chunk_frames <= 0 or total_frames <= chunk_frames:
            return [wav_path]

        out_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        idx = 0
        start = 0
        while start < total_frames:
            read_start = max(0, start - overlap_frames) if idx > 0 else start
            end = min(total_frames, start + chunk_frames)
            frame_count = end - read_start

            src.setpos(read_start)
            frames = src.readframes(frame_count)

            chunk_path = out_dir / f"chunk_{idx:04d}.wav"
            with wave.open(str(chunk_path), "wb") as dst:
                dst.setnchannels(nchannels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(rate)
                dst.writeframes(frames)

            chunk_paths.append(chunk_path)
            idx += 1
            start = end

        return chunk_paths


# ---------------------------------------------------------------------------
# Chunk text stitching: adjacent chunk texts share overlap_seconds of audio,
# so their transcripts share a run of duplicated words at the seam. Find it
# with a word-level diff and keep the second chunk's version of it (it had
# full leading context, unlike the first chunk where those words sat right
# at the edge). Only the last/first couple of segments at each seam are
# touched; the rest of each chunk's per-segment line formatting is kept.
# ---------------------------------------------------------------------------

_STITCH_BOUNDARY_LINES = 2


def _stitch_pair(prev_text: str, next_text: str, overlap_seconds: int) -> str:
    if not prev_text:
        return next_text
    if not next_text:
        return prev_text

    prev_lines = prev_text.split("\n")
    next_lines = next_text.split("\n")

    prev_head_lines = prev_lines[:-_STITCH_BOUNDARY_LINES] if len(prev_lines) > _STITCH_BOUNDARY_LINES else []
    prev_tail_lines = prev_lines[-_STITCH_BOUNDARY_LINES:]
    next_tail_lines = next_lines[_STITCH_BOUNDARY_LINES:] if len(next_lines) > _STITCH_BOUNDARY_LINES else []
    next_head_lines = next_lines[:_STITCH_BOUNDARY_LINES]

    tail_words = " ".join(prev_tail_lines).split()
    head_words = " ".join(next_head_lines).split()

    if not tail_words or not head_words:
        merged_boundary = " ".join(tail_words + head_words)
    else:
        window = max(4, overlap_seconds * 4)
        tail_window = tail_words[-window:]
        head_window = head_words[:window]
        tail_window_offset = len(tail_words) - len(tail_window)

        matcher = difflib.SequenceMatcher(None, tail_window, head_window, autojunk=False)
        match = matcher.find_longest_match(0, len(tail_window), 0, len(head_window))

        if match.size < 2:
            LOGGER.warning("Chunk stitch fallback: no reliable overlap match found at boundary")
            merged_boundary = " ".join(tail_words) + "\n" + " ".join(head_words)
        else:
            cut = tail_window_offset + match.a
            merged_words = tail_words[:cut] + head_words[match.b:]
            merged_boundary = " ".join(merged_words)

    merged_lines = prev_head_lines + [merged_boundary] + next_tail_lines
    return "\n".join(line for line in merged_lines if line)


def stitch_chunk_texts(chunk_texts: list[str], overlap_seconds: int) -> str:
    if not chunk_texts:
        return ""

    result = chunk_texts[0]
    for text in chunk_texts[1:]:
        result = _stitch_pair(result, text, overlap_seconds)
    return result


# ---------------------------------------------------------------------------
# GPU VRAM probing / worker sizing
# ---------------------------------------------------------------------------

_GPU_VRAM_GB_BY_MODEL = {
    "tiny": 1.0, "tiny.en": 1.0,
    "base": 1.0, "base.en": 1.0,
    "small": 2.0, "small.en": 2.0,
    "medium": 5.0, "medium.en": 5.0,
    "large": 10.0, "large-v1": 10.0, "large-v2": 10.0, "large-v3": 10.0,
    "distil-large-v2": 6.0, "distil-large-v3": 6.0,
    "turbo": 6.0, "large-v3-turbo": 6.0,
}
_GPU_VRAM_FALLBACK_GB = 1.0


def _estimate_gpu_vram_bytes_per_instance(model_size: str, compute_type: str) -> int:
    base_gb = _GPU_VRAM_GB_BY_MODEL.get(model_size.lower(), _GPU_VRAM_FALLBACK_GB)
    if compute_type in {"int8", "int8_float16", "int8_float32"}:
        base_gb *= 0.6
    return max(1, int(base_gb * (1024 ** 3)))


def probe_gpu_free_vram_bytes(device_index: int = 0) -> int | None:
    try:
        import pynvml
    except ImportError:
        LOGGER.warning("pynvml not installed; cannot probe GPU VRAM, defaulting to 1 GPU worker")
        return None

    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.free)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        LOGGER.warning("GPU VRAM probe failed: %s", exc)
        return None


def estimate_gpu_worker_count(model_size: str, compute_type: str) -> int:
    free_bytes = probe_gpu_free_vram_bytes()
    if free_bytes is None:
        return 1

    per_instance = _estimate_gpu_vram_bytes_per_instance(model_size, compute_type)
    raw_capacity = free_bytes // per_instance
    workers = max(1, raw_capacity - GPU_VRAM_BUFFER_INSTANCES)

    if GPU_MAX_WORKERS_CAP:
        try:
            workers = min(workers, max(1, int(GPU_MAX_WORKERS_CAP)))
        except ValueError:
            pass

    LOGGER.info(
        "GPU pool sized to %d workers: %.1fGB free, ~%.1fGB/instance for model=%s compute_type=%s, buffer=%d instance(s)",
        workers,
        free_bytes / (1024 ** 3),
        per_instance / (1024 ** 3),
        model_size,
        compute_type,
        GPU_VRAM_BUFFER_INSTANCES,
    )
    return workers


# ---------------------------------------------------------------------------
# Worker pools
# ---------------------------------------------------------------------------

_POOL_LOCK = threading.Lock()
_MP_CONTEXT = multiprocessing.get_context("spawn")

_CPU_POOL: "concurrent.futures.ProcessPoolExecutor | None" = None
_CPU_POOL_SIZE: "int | None" = None
_GPU_POOL: "concurrent.futures.ProcessPoolExecutor | None" = None
_GPU_POOL_SIZE: "int | None" = None


def _init_cpu_worker(pool_size: int) -> None:
    global _WORKER_CPU_THREADS, _MODEL_CACHE_MAXSIZE
    total = os.cpu_count() or 1
    _WORKER_CPU_THREADS = max(1, total // max(1, pool_size))
    _MODEL_CACHE_MAXSIZE = 3
    try:
        torch = importlib.import_module("torch")
        torch.set_num_threads(_WORKER_CPU_THREADS)
    except Exception:
        pass


def _init_gpu_worker() -> None:
    global _MODEL_CACHE_MAXSIZE
    _MODEL_CACHE_MAXSIZE = 1


def _get_cpu_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _CPU_POOL, _CPU_POOL_SIZE
    if _CPU_POOL is None:
        with _POOL_LOCK:
            if _CPU_POOL is None:
                LOGGER.info("Starting CPU worker pool with %d workers", CPU_MAX_WORKERS)
                _CPU_POOL = concurrent.futures.ProcessPoolExecutor(
                    max_workers=CPU_MAX_WORKERS,
                    mp_context=_MP_CONTEXT,
                    initializer=_init_cpu_worker,
                    initargs=(CPU_MAX_WORKERS,),
                )
                _CPU_POOL_SIZE = CPU_MAX_WORKERS
    return _CPU_POOL


def _get_gpu_pool(model_size: str, compute_type: str) -> concurrent.futures.ProcessPoolExecutor:
    global _GPU_POOL, _GPU_POOL_SIZE
    if _GPU_POOL is None:
        with _POOL_LOCK:
            if _GPU_POOL is None:
                workers = estimate_gpu_worker_count(model_size, compute_type)
                LOGGER.info("Starting GPU worker pool with %d workers", workers)
                _GPU_POOL = concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=_MP_CONTEXT,
                    initializer=_init_gpu_worker,
                )
                _GPU_POOL_SIZE = workers
    return _GPU_POOL


def _select_pool(device: str, model_size: str, compute_type: str) -> concurrent.futures.ProcessPoolExecutor:
    if device == "cuda":
        return _get_gpu_pool(model_size, compute_type)
    return _get_cpu_pool()


def pool_status() -> dict:
    return {
        "cpu_pool_workers": _CPU_POOL_SIZE,
        "gpu_pool_workers": _GPU_POOL_SIZE,
    }


def shutdown_pools() -> None:
    global _CPU_POOL, _CPU_POOL_SIZE, _GPU_POOL, _GPU_POOL_SIZE
    with _POOL_LOCK:
        if _CPU_POOL is not None:
            # wait=True: block until worker processes actually exit. With
            # wait=False the parent process can exit before the pool's
            # queue-management thread finishes signaling/joining workers,
            # leaving orphaned processes behind (observed in testing).
            _CPU_POOL.shutdown(wait=True, cancel_futures=True)
            _CPU_POOL = None
            _CPU_POOL_SIZE = None
        if _GPU_POOL is not None:
            _GPU_POOL.shutdown(wait=True, cancel_futures=True)
            _GPU_POOL = None
            _GPU_POOL_SIZE = None


def _worker_transcribe_chunk(chunk_path: str, model_size: str, backend: str, device: str, compute_type: str):
    path = Path(chunk_path)
    if backend == "openai-whisper":
        return transcribe_openai_whisper(path, model_size, device, compute_type)
    cpu_threads = _WORKER_CPU_THREADS if device == "cpu" else 0
    return transcribe_faster_whisper(path, model_size, device, compute_type, cpu_threads=cpu_threads)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _transcribe_chunked(
    wav_path: Path,
    model_size: str,
    backend: str,
    device: str,
    compute_type: str,
    chunk_seconds: int,
    overlap_seconds: int,
):
    chunk_dir = wav_path.parent / f"{wav_path.stem}.chunks"
    chunk_paths = split_wav(wav_path, chunk_seconds, overlap_seconds, chunk_dir)
    is_split = len(chunk_paths) > 1

    try:
        pool = _select_pool(device, model_size, compute_type)
        futures = [
            pool.submit(_worker_transcribe_chunk, str(p), model_size, backend, device, compute_type)
            for p in chunk_paths
        ]
        results = []
        try:
            for f in futures:
                results.append(f.result())
        except Exception:
            for f in futures:
                f.cancel()
            raise
    finally:
        if is_split:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    texts = [r[0] for r in results]
    infos = [r[1] for r in results]

    text = texts[0] if len(texts) == 1 else stitch_chunk_texts(texts, overlap_seconds)

    merged_info = dict(infos[0]) if infos else {}
    merged_info["backend"] = backend
    merged_info["device"] = device
    merged_info["chunks"] = len(chunk_paths)
    merged_info["chunk_seconds"] = chunk_seconds
    merged_info["chunk_overlap_seconds"] = overlap_seconds if is_split else 0
    merged_info["languages"] = [i.get("language") for i in infos]

    for numeric_key in ("duration", "duration_after_vad"):
        values = [i.get(numeric_key) for i in infos if isinstance(i.get(numeric_key), (int, float))]
        if values:
            merged_info[numeric_key] = sum(values)

    return text, merged_info


def transcribe(
    wav_path: Path,
    model_size: str,
    backend: str,
    device: str,
    compute_type: str,
    chunk_seconds: "int | None" = None,
    overlap_seconds: "int | None" = None,
):
    selected_backend, selected_device = detect_runtime(backend, device)
    chunk_seconds = DEFAULT_CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
    overlap_seconds = DEFAULT_CHUNK_OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds

    try:
        return _transcribe_chunked(
            wav_path, model_size, selected_backend, selected_device, compute_type, chunk_seconds, overlap_seconds
        )
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
        return _transcribe_chunked(
            wav_path, model_size, "faster-whisper", "cpu", "auto", chunk_seconds, overlap_seconds
        )


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
    ap.add_argument(
        "--chunk-seconds",
        type=int,
        default=None,
        help=f"Chunk length in seconds for parallel transcription (default: {DEFAULT_CHUNK_SECONDS}, or $WHISPER_CHUNK_SECONDS)",
    )
    ap.add_argument(
        "--chunk-overlap-seconds",
        type=int,
        default=None,
        help=f"Overlap in seconds carried into the next chunk (default: {DEFAULT_CHUNK_OVERLAP_SECONDS}, or $WHISPER_CHUNK_OVERLAP_SECONDS)",
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
    try:
        text, info = transcribe(
            optimized,
            args.model,
            args.backend,
            args.device,
            args.compute_type,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.chunk_overlap_seconds,
        )
    finally:
        shutdown_pools()

    out_txt.write_text(text, encoding="utf-8")
    print(f"Wrote transcript -> {out_txt}")
    print(f"Transcription info -> {info}")

if __name__ == "__main__":
    main()
