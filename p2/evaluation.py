#Compression quality metrics for the APAC codec: ratio, space saving, bitrate, SNR and sample-accurate error.

'''
    Evaluation of compressed files against their originals:

        - sizes, compression ratio and space saving
        - average bitrate of the compressed stream (vs 16-bit PCM)
        - SNR and maximum sample error, measured in the int16 domain the codec operates on

    Usable as a module (evaluate/format_report) or from the command line:
        python evaluation.py ORIGINAL.wav COMPRESSED.apac [--decoded DECODED.wav]
'''

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

#The repo has no packaging metadata: make src/ importable when run from the repo root.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from audio_codec.codec.codec import decompress
from audio_codec.codec.format import read_apac
from audio_codec.shared.load_file import load_apac, load_wav, save_wav


def compression_ratio(original_size: int, compressed_size: int) -> float:
    '''
        Original bytes per compressed byte: 1.0 is break-even, higher is better.

        Parameters:
            - original_size (int): size of the original file in bytes
            - compressed_size (int): size of the compressed file in bytes

        Returns:
            - float: compression ratio (inf when the compressed file is empty)
    '''
    if compressed_size == 0:
        return float("inf")
    return original_size / compressed_size


def space_saving(original_size: int, compressed_size: int) -> float:
    '''
        Fraction of the original bytes saved by compression: 1 - compressed/original.

        Parameters:
            - original_size (int): size of the original file in bytes
            - compressed_size (int): size of the compressed file in bytes

        Returns:
            - float: space saving in [0, 1], negative when the file grew
    '''
    if original_size == 0:
        return 0.0
    return 1.0 - compressed_size / original_size


def bitrate_kbps(size_bytes: int, frames: int, sample_rate: int) -> float:
    '''
        Average bitrate of a stream: total bits over duration.

        Parameters:
            - size_bytes (int): size of the stream in bytes
            - frames (int): number of sample frames
            - sample_rate (int): samples per second

        Returns:
            - float: bitrate in kbit/s (0.0 for an empty stream)
    '''
    if frames <= 0 or sample_rate <= 0:
        return 0.0
    return size_bytes * 8.0 / frames * sample_rate / 1000.0


def snr_db(original: np.ndarray, decoded: np.ndarray) -> float:
    '''
        Signal-to-noise ratio of the decoded samples against the original, in dB.

        Measured sample-wise in the int16 domain the codec operates on: a
        bit-exact decode is inf dB, silent input with a non-silent decode is -inf dB.

        Parameters:
            - original (np.ndarray): original samples, (frames, channels)
            - decoded (np.ndarray): decoded samples, same shape

        Returns:
            - float: SNR in dB
    '''
    signal = np.asarray(original, dtype=np.float64)
    noise = np.asarray(decoded, dtype=np.float64) - signal
    noise_power = float(np.sum(noise * noise))
    if noise_power == 0.0:
        return float("inf")
    signal_power = float(np.sum(signal * signal))
    if signal_power == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / noise_power)


def max_error(original: np.ndarray, decoded: np.ndarray) -> int:
    '''
        Largest absolute sample error between original and decoded, in LSBs.

        Parameters:
            - original (np.ndarray): original samples, (frames, channels)
            - decoded (np.ndarray): decoded samples, same shape

        Returns:
            - int: maximum absolute sample error, 0 = lossless
    '''
    if original.size == 0:
        return 0
    return int(np.max(np.abs(decoded.astype(np.int64) - original.astype(np.int64))))


def evaluate(original_path: Path, compressed_path: Path, decoded_path: Path | None = None) -> dict:
    '''
        Measures one compressed file against its original.

        Reads the APAC header for stream info and settings, decodes the
        compressed file to a temporary WAV when no decoded file is given, and
        reports sizes, ratio, space saving, bitrate, SNR and sample-accurate error.

        Parameters:
            - original_path (Path): the original WAV file
            - compressed_path (Path): the compressed APAC file
            - decoded_path (Path | None): an already decoded WAV, None = decode to a temp file

        Returns:
            - dict: paths, sizes, header fields and every metric

        Raises:
            - ValueError: if the decoded stream does not match the original's shape
            - IOError: if the APAC file cannot be read
            - HeaderError: if the APAC header is invalid
    '''
    original_path, compressed_path = Path(original_path), Path(compressed_path)
    original_size = original_path.stat().st_size
    compressed_size = compressed_path.stat().st_size
    header, _ = read_apac(load_apac(compressed_path))

    temp_name: str | None = None
    if decoded_path is None:
        fd, temp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)                                       #sf.write needs the path free (Windows file locking)
        decoded_path = Path(temp_name)
        samples, sample_rate = decompress(load_apac(compressed_path))
        save_wav(decoded_path, samples, sample_rate)

    try:
        _, original = load_wav(original_path)
        _, decoded = load_wav(Path(decoded_path))
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)

    if original.shape != decoded.shape:
        raise ValueError(f"decoded shape {decoded.shape} does not match original {original.shape}")

    settings = {
        "predictor": header.predictor.name,
        "predictor_order": header.predictor_order,
        "block_size_log2": header.block_size_log2,
        "quant_step": header.quant_step,
        "dct_mode": header.dct_mode.name,
        "dct_window_size": header.dct_window_size,
        "dct_param": header.dct_param,
        "entropy_coder": header.entropy_coder.name,
    }
    return {
        "original": str(original_path),
        "compressed": str(compressed_path),
        "original_bytes": original_size,
        "compressed_bytes": compressed_size,
        "ratio": compression_ratio(original_size, compressed_size),
        "space_saving": space_saving(original_size, compressed_size),
        "sample_rate": header.sample_rate,
        "frames": header.frames,
        "channels": header.channels,
        "duration_s": header.frames / header.sample_rate,
        "bitrate_kbps": bitrate_kbps(compressed_size, header.frames, header.sample_rate),
        "pcm_bitrate_kbps": 16 * header.channels * header.sample_rate / 1000.0,
        "snr_db": snr_db(original, decoded),
        "max_error_lsb": max_error(original, decoded),
        "lossless": bool(np.array_equal(original, decoded)),
        "settings": settings,
    }


def format_report(metrics: dict) -> str:
    '''
        Formats an evaluate() dict as a human-readable multi-line report.

        Parameters:
            - metrics (dict): the dict returned by evaluate

        Returns:
            - str: the formatted report
    '''
    s = metrics["settings"]
    return "\n".join([
        f"original:    {metrics['original']} ({metrics['original_bytes']} bytes)",
        f"compressed:  {metrics['compressed']} ({metrics['compressed_bytes']} bytes)",
        f"stream:      {metrics['frames']} frames, {metrics['channels']} channel(s), "
        f"{metrics['sample_rate']} Hz, {metrics['duration_s']:.3f} s",
        f"settings:    predictor {s['predictor'].lower()}{s['predictor_order']}, quant_step {s['quant_step']}, "
        f"dct {s['dct_mode'].lower()}, block 2^{s['block_size_log2']}",
        f"ratio:       {metrics['ratio']:.3f} : 1 (space saving {metrics['space_saving']:.1%})",
        f"bitrate:     {metrics['bitrate_kbps']:.1f} kbit/s (16-bit PCM: {metrics['pcm_bitrate_kbps']:.1f} kbit/s)",
        f"quality:     SNR {metrics['snr_db']:.2f} dB, max error {metrics['max_error_lsb']} LSB"
        + (" (lossless)" if metrics["lossless"] else ""),
    ])


def main(argv: list[str] | None = None) -> int:
    '''
        CLI entry point: evaluates one original/compressed pair and prints the report.

        Parameters:
            - argv (list[str] | None): command line arguments (None = sys.argv)

        Returns:
            - int: process exit code, 0 on success
    '''
    parser = argparse.ArgumentParser(description="Evaluate APAC compression quality")
    parser.add_argument("original", help="original WAV file")
    parser.add_argument("compressed", help="compressed APAC file")
    parser.add_argument("--decoded", "-d", default=None, help="decoded WAV file (default: decode to a temp file)")
    args = parser.parse_args(argv)

    print(format_report(evaluate(args.original, args.compressed, args.decoded)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
