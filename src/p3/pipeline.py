"""The P3 pipeline: verify, load, encode, decode, check, save.

All audio work is delegated to the APAC codec in p2: load_file and save_wav
(file IO), wav_issues (the shape contract), compress and decompress (the
codec). This module adds manifest verification, timing and memory
measurement, atomic output writes, and a run log.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
import tracemalloc
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audio_codec.codec.codec import compress, decompress
from audio_codec.preprocessing.pipeline import wav_issues
from audio_codec.shared.load_file import load_file, save_wav

from p3 import config as cfg
from p3 import integrity

log = logging.getLogger("p3")


class InvalidShape(Exception):
    """Loaded audio violates the int16 (frames, channels) contract."""


class RoundTripError(Exception):
    """Lossless settings did not reproduce the input exactly."""


EXIT_CODES = {
    FileNotFoundError: 2,
    integrity.DigestMismatch: 3,
    InvalidShape: 4,
    RoundTripError: 5,
    KeyboardInterrupt: 130,
}


def exit_code_for(exc: BaseException) -> int:
    for exc_type, code in EXIT_CODES.items():
        if isinstance(exc, exc_type):
            return code
    return 1


def attach_file_log(out_dir: Path) -> logging.Handler:
    """Append run lines to <out_dir>/run.log; the caller detaches the handler."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return handler


def preflight(root: Path, config: dict) -> dict:
    """Check that every input exists; raise FileNotFoundError naming it if not."""
    paths = {
        "dataset_dir": cfg.dataset_dir(root, config),
        "manifest": cfg.manifest_path(root),
        "input_clip": cfg.input_clip(root, config),
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    return paths


def check_shape(metadata: dict, samples: np.ndarray) -> None:
    """The audio contract: int16, 2D (frames, channels), consistent with the file header.

    The structural checks are p2's wav_issues; only the dtype rule is added
    here, since the loader already guarantees it for real files and the tests
    exercise the contract on raw arrays.
    """
    if samples.dtype != np.int16:
        raise InvalidShape(f"expected int16 samples, got {samples.dtype}")
    issues = wav_issues(metadata, samples)
    if issues:
        raise InvalidShape("; ".join(str(issue) for issue in issues))


def encode(samples: np.ndarray, rate: int, quant_step: int) -> tuple[bytes, float, int]:
    """Compress (samples, rate) with one quantization step; return blob, seconds, peak bytes."""
    tracemalloc.start()
    try:
        start = time.perf_counter()
        blob = compress((samples, rate), {"quant_step": quant_step})
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return blob, elapsed, peak


def decode(blob: bytes) -> tuple[np.ndarray, int, float, int]:
    """Decompress an APAC blob; return samples, rate, seconds, peak bytes."""
    tracemalloc.start()
    try:
        start = time.perf_counter()
        samples, rate = decompress(blob)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return samples, rate, elapsed, peak


def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def _atomic_write_all(out_dir: Path,
                      writers: list[tuple[str, Callable[[Path], None]]]) -> None:
    """Write every output to a temp file first, then rename all into place.

    An interrupted run leaves either every final file or none: each file is
    complete before it appears under its final name, and a failure during the
    rename phase removes whatever was already placed.
    """
    staged, placed = [], []
    try:
        for name, write in writers:
            # prefix the temp name so the real extension survives: soundfile
            # and other writers infer the format from the extension
            tmp = out_dir / (".tmp-" + name)
            write(tmp)
            staged.append((tmp, out_dir / name))
        for tmp, final in staged:
            os.replace(tmp, final)
            placed.append(final)
    except BaseException:
        for final in placed:
            final.unlink(missing_ok=True)
        raise
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)


def run_pipeline(root: Path, config: dict) -> dict:
    """Run the full pipeline on the configured clip; return the results dict.

    Steps: preflight, verify against the manifest, load and check the shape,
    encode, decode, check reconstruction, write outputs atomically. Every
    step logs one line to the output run log.
    """
    paths = preflight(root, config)
    out_dir = cfg.output_dir(root, config)
    handler = attach_file_log(out_dir)
    started = time.perf_counter()
    step = "preflight"
    try:
        log.info("run started: input %s, quant_step %s",
                 config["input_clip"], config["quant_step"])

        step = "verify"
        manifest = integrity.load_manifest(paths["manifest"])
        entries = {f["path"]: f for f in manifest["files"]}
        rel = config["input_clip"]
        if rel not in entries:
            raise FileNotFoundError(f"{rel}: not listed in the manifest")
        clip = paths["input_clip"]
        sha = integrity.verify_file(clip, entries[rel])
        log.info("verify: sha256 ok for %s", rel)

        step = "load"
        metadata, samples = load_file(clip)
        check_shape(metadata, samples)
        rate = metadata["samplerate"]
        log.info("load: %s frames=%d channels=%d rate=%d subtype=%s",
                 rel, samples.shape[0], samples.shape[1], rate, metadata["subtype"])

        step = "encode"
        blob, encode_s, encode_peak = encode(samples, rate, config["quant_step"])
        log.info("encode: %d bytes in %.3fs", len(blob), encode_s)

        step = "decode"
        decoded, decoded_rate, decode_s, decode_peak = decode(blob)
        log.info("decode: %d frames in %.3fs", decoded.shape[0], decode_s)

        step = "check"
        bit_exact = bool(decoded_rate == rate and np.array_equal(samples, decoded))
        error = rmse(samples, decoded)
        if config["quant_step"] == 1 and not bit_exact:
            raise RoundTripError("quant_step 1 must round trip bit-exactly")

        wav_bytes = clip.stat().st_size
        results = {
            "input_file": rel,
            "input_sha256": sha,
            "sample_rate": rate,
            "frames": int(samples.shape[0]),
            "channels": int(samples.shape[1]),
            "subtype": metadata["subtype"],
            "quant_step": config["quant_step"],
            "wav_bytes": wav_bytes,
            "apac_bytes": len(blob),
            "compression_ratio": round(wav_bytes / len(blob), 3),
            "space_saving_percent": round((1 - len(blob) / wav_bytes) * 100, 2),
            "bit_exact": bit_exact,
            "rmse": round(error, 6),
            "encode_seconds": round(encode_s, 4),
            "decode_seconds": round(decode_s, 4),
            "encode_peak_bytes": encode_peak,
            "decode_peak_bytes": decode_peak,
            "run_seconds": round(time.perf_counter() - started, 3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "soundfile": __import__("soundfile").__version__,
            },
        }

        step = "save"
        _atomic_write_all(out_dir, [
            ("compressed.apac", lambda p: p.write_bytes(blob)),
            ("decoded.wav", lambda p: save_wav(p, decoded, decoded_rate)),
            ("results.json", lambda p: p.write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8")),
        ])
        log.info("save: compressed.apac, decoded.wav, results.json -> %s",
                 config["output_dir"])
        log.info("run finished: bit_exact=%s ratio=%.3f in %.2fs",
                 bit_exact, results["compression_ratio"], results["run_seconds"])
        return results
    except KeyboardInterrupt:
        log.error("run interrupted during %s; partial outputs removed", step)
        raise
    except Exception as exc:
        log.error("run failed during %s: %s", step, exc)
        raise
    finally:
        log.removeHandler(handler)
        handler.close()


def benchmark(root: Path, config: dict) -> dict:
    """Timing and memory evidence for one storage choice on the configured clip.

    Compares the original PCM WAV file with APAC at quantization step 1
    (lossless) and step 4 (lossy; the first step past the measured FLAC
    crossover point in the p2 evaluation): bytes, encode and decode wall
    time, and tracemalloc peak memory.
    """
    clip = cfg.input_clip(root, config)
    metadata, samples = load_file(clip)
    check_shape(metadata, samples)
    rate = metadata["samplerate"]
    rows = [{
        "storage": "WAV PCM_16 (original file)",
        "bytes": clip.stat().st_size,
        "encode_s": None, "decode_s": None,
        "encode_peak_mb": None, "decode_peak_mb": None,
        "bit_exact": True, "rmse": 0.0,
    }]
    for step in (1, 4):
        blob, encode_s, encode_peak = encode(samples, rate, step)
        decoded, decoded_rate, decode_s, decode_peak = decode(blob)
        rows.append({
            "storage": f"APAC quant_step {step}",
            "bytes": len(blob),
            "encode_s": round(encode_s, 4),
            "decode_s": round(decode_s, 4),
            "encode_peak_mb": round(encode_peak / 2 ** 20, 2),
            "decode_peak_mb": round(decode_peak / 2 ** 20, 2),
            "bit_exact": bool(decoded_rate == rate and np.array_equal(samples, decoded)),
            "rmse": round(rmse(samples, decoded), 4),
        })
    return {"input_file": config["input_clip"], "rows": rows}