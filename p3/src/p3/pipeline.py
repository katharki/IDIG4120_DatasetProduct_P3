#The P3 pipeline: verify, load, encode, decode, check, save; all audio work is APAC's from p2.

'''
    Runs one clip end to end against the manifest. load_file and save_wav
    (file IO), wav_issues (the shape contract) and compress/decompress
    (the codec) come from p2; this module adds the manifest verification,
    the timing and memory measurement, the all-at-once output writes and
    the run log.
'''

from __future__ import annotations

import json
import logging
import os
import platform
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile

from audio_codec.codec.codec import compress, decompress
from audio_codec.preprocessing.pipeline import wav_issues
from audio_codec.shared.load_file import load_file, save_wav

from p3 import config as cfg
from p3 import integrity

log = logging.getLogger("p3")


class InvalidShape(ValueError):
    '''Loaded audio violates the int16 (frames, channels) contract.'''


class RoundTripError(ValueError):
    '''Lossless settings did not reproduce the loaded samples exactly.'''


# Save config
TMP_PREFIX = ".tmp-"    #temp files keep the real extension: writers infer the format from it
MIB = 2 ** 20           #tracemalloc peaks are reported in MiB

# benchmark
BENCH_STEPS = (1, 4)    #1 = lossless, 4 = lossy


def timed(func):
    '''
        Runs func between perf_counter and tracemalloc: the timing and memory
        evidence in results.json and benchmark.

        Returns:
            - tuple: (func's result, wall seconds, tracemalloc peak bytes).
    '''
    tracemalloc.start()
    try:
        start = time.perf_counter()
        result = func()
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, seconds, peak


def attach_file_log(out_dir: Path) -> logging.Handler:
    '''
        Appends run lines to <out_dir>/run.log; the caller detaches the handler.

        Parameters:
            - out_dir (Path): The output directory, created when missing.

        Returns:
            - logging.Handler: The attached file handler.
    '''
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return handler


def preflight(root: Path, config: dict) -> dict:
    '''
        Checks that every input exists, before the output directory is created.

        Parameters:
            - root (Path): The product root.
            - config (dict): The loaded config.

        Returns:
            - dict: The dataset dir, manifest and input clip paths.

        Raises:
            - FileNotFoundError: If an input is missing, naming it.
    '''
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
    '''
        The audio contract: int16, 2D (frames, channels), consistent with the file
        header. The structural rules are p2's wav_issues; only the dtype rule is
        added here.

        Parameters:
            - metadata (dict): The file header metadata (samplerate, channels, frames).
            - samples (np.ndarray): The loaded audio.

        Raises:
            - InvalidShape: If the samples violate the contract.
    '''
    if samples.dtype != np.int16:
        raise InvalidShape(f"expected int16 samples, got {samples.dtype}")
    issues = wav_issues(metadata, samples)
    if issues:
        raise InvalidShape("; ".join(str(issue) for issue in issues))


def encode(samples: np.ndarray, rate: int, quant_step: int) -> tuple[bytes, float, int]:
    '''Compresses the samples with APAC at one quantization step, timed.'''
    return timed(lambda: compress((samples, rate), {"quant_step": quant_step}))


def decode(blob: bytes) -> tuple[tuple[np.ndarray, int], float, int]:
    '''Decompresses an APAC blob, timed; result is ((samples, rate), seconds, peak bytes).'''
    return timed(lambda: decompress(blob))


def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    '''Root mean square error between two sample arrays, as float64.'''
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))



def run_pipeline(root: Path, config: dict) -> dict:
    '''
        Runs the full pipeline on the configured clip and returns the results dict.

        Steps: preflight, verify against the manifest, load and check the shape,
        encode, decode, check the reconstruction, save. Every step logs one line
        to the output run log.

        Parameters:
            - root (Path): The product root.
            - config (dict): The loaded config.

        Returns:
            - dict: The results recorded in results.json.

        Raises:
            - FileNotFoundError: A missing input.
            - integrity.DigestMismatch: A failed manifest verification.
            - InvalidShape: Audio that violates the contract.
            - RoundTripError: A lossless run that is not bit-exact.
    '''
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
        (decoded, decoded_rate), decode_s, decode_peak = decode(blob)
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
                "soundfile": soundfile.__version__,
            },
        }

        step = "save"
        outputs = ("compressed.apac", "decoded.wav", "results.json")
        try:
            (out_dir / (TMP_PREFIX + "compressed.apac")).write_bytes(blob)
            save_wav(out_dir / (TMP_PREFIX + "decoded.wav"), decoded, decoded_rate)
            (out_dir / (TMP_PREFIX + "results.json")).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

            # Finalize outputs: rename temporary files into place
            for name in outputs:
                os.replace(out_dir / (TMP_PREFIX + name), out_dir / name)

        except BaseException:
            for name in outputs:
                (out_dir / (TMP_PREFIX + name)).unlink(missing_ok=True)
                (out_dir / name).unlink(missing_ok=True)
            raise

        log.info("save: compressed.apac, decoded.wav, results.json -> %s", config["output_dir"])
        log.info("run finished: bit_exact=%s ratio=%.3f in %.2fs", bit_exact, results["compression_ratio"], results["run_seconds"])
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
    '''
        Timing and memory evidence for one storage choice on the configured clip:
        the original PCM WAV file against APAC at each BENCH_STEPS step, measured
        as bytes, encode and decode wall time, and tracemalloc peak memory.
        Step 4 is the measured FLAC-crossover step in the p2 evaluation.

        Parameters:
            - root (Path): The product root.
            - config (dict): The loaded config.

        Returns:
            - dict: input_file and one row per storage choice.
    '''
    clip = cfg.input_clip(root, config)
    metadata, samples = load_file(clip)
    check_shape(metadata, samples)
    rate = metadata["samplerate"]
    rows = [{
        "storage": f"WAV {metadata['subtype']} (original file)",
        "bytes": clip.stat().st_size,
        "encode_s": None, "decode_s": None,
        "encode_peak_mb": None, "decode_peak_mb": None,
        "bit_exact": True, "rmse": 0.0,
    }]
    for step in BENCH_STEPS:
        blob, encode_s, encode_peak = encode(samples, rate, step)
        (decoded, decoded_rate), decode_s, decode_peak = decode(blob)
        rows.append({
            "storage": f"APAC quant_step {step}",
            "bytes": len(blob),
            "encode_s": round(encode_s, 4),
            "decode_s": round(decode_s, 4),
            "encode_peak_mb": round(encode_peak / MIB, 2),
            "decode_peak_mb": round(decode_peak / MIB, 2),
            "bit_exact": bool(decoded_rate == rate and np.array_equal(samples, decoded)),
            "rmse": round(rmse(samples, decoded), 6),
        })
    return {"input_file": config["input_clip"], "rows": rows}
