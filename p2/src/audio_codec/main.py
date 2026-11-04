#CLI entry point: wires the loader (all file I/O) to the codec's pure byte-level transforms.

'''
    Command line interface for the APAC codec - full usage lives in --help.
    All knobs are compression-only; decompression reads its settings from the APAC header.
'''

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from audio_codec.codec.codec import compress, decompress
from audio_codec.config import DCT_QUALITY_MAX, DCT_QUALITY_MIN, DCTMode, MAX_QUANT_STEP, MIN_QUANT_STEP
from audio_codec.preprocessing.pipeline import wav_issues
from audio_codec.shared import logger
from audio_codec.shared.load_file import load_apac, load_file, save_apac, save_wav


def quality_to_quant_step(quality: int) -> int:
    '''
        Maps the 0-100 quality setting onto the residual quantization step: 100 is
        the lossless anchor, every 20 points of quality lost doubles the step with;
        81-100 -> 1, 61-80 -> 2, 41-60 -> 4, 21-40 -> 8, 0-20 -> 16 so the
        default 50 lands on step 4, the measured FLAC-crossover point
        (corpus table in notebooks/codec_results.ipynb).

        Raises:
            - ValueError: if quality is outside [0-100]
    '''
    if not 0 <= quality <= 100:
        raise ValueError(f"expected quality in [0-100], got {quality}")
    return 1 << min(4, (100 - quality) // 20)


def ranged_int(name: str, low: int, high: int) -> Callable[[str], int]:
    '''Argparse type factory: an int in [low-high] with a clean error message.'''
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid {name}: '{text}' is not an integer")
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{name} must be in the range [{low}-{high}], got {value}")
        return value
    return parse


quality_arg = ranged_int("quality", 0, 100)
quant_step_arg = ranged_int("quantization step", MIN_QUANT_STEP, MAX_QUANT_STEP)
dct_quality_arg = ranged_int("DCT quality", DCT_QUALITY_MIN, DCT_QUALITY_MAX)


def build_settings(quality: int, quant_step: int | None, dct_quality: int | None,
                   dct_window_size: int | None) -> dict:
    '''
        Builds the codec settings dict for compress() from the CLI knobs: an
        explicit --quant-step wins over the quality mapping, --dct-quality N
        turns on the JPEG-style DCT pre-shaping (with optional --dct-window),
        and anything unset falls back to DEFAULT_SETTINGS inside encode_pipeline.

        Raises:
            - ValueError: if --dct-window is given without --dct-quality
    '''
    settings: dict = {"quant_step": quant_step if quant_step is not None else quality_to_quant_step(quality)}
    if dct_quality is not None:
        settings["dct_mode"] = DCTMode.QUANTIZE
        settings["dct_param"] = dct_quality
        if dct_window_size is not None:
            settings["dct_window_size"] = dct_window_size
    elif dct_window_size is not None:
        raise ValueError("--dct-window requires --dct-quality")
    return settings


def compress_file(infile: Path, outfile: Path, settings: dict) -> None:
    '''
        Compresses one WAV into an APAC container: load_file -> wav_issues
        (invalid/empty WAVs rejected) -> compress -> save_apac.

        Raises:
            - ValueError: if the WAV is invalid or empty, or the settings are bad
            - HeaderError: if the combined settings produce an invalid header
            - OSError: if reading or writing fails
    '''
    metadata, samples = load_file(infile)
    issues = wav_issues(metadata, samples)
    if issues:
        raise ValueError("invalid WAV file: " + "; ".join(str(issue) for issue in issues))
    original_size = infile.stat().st_size
    blob = compress((samples, metadata["samplerate"]), settings)
    save_apac(outfile, blob)
    print(f"compressed {infile} -> {outfile}: {original_size:,} -> {len(blob):,} bytes "
          f"(ratio {original_size / len(blob):.2f}: 1)")


def decompress_file(infile: Path, outfile: Path) -> None:
    '''
        Decompresses one APAC container into a WAV: load_apac -> decompress -> save_wav.

        Raises:
            - FormatError: if the container or header is invalid
            - BlockError: if the data section is truncated or malformed
            - OSError: if reading or writing fails
    '''
    samples, sample_rate = decompress(load_apac(infile))
    save_wav(outfile, samples, sample_rate)
    print(f"decompressed {infile} -> {outfile}: {samples.shape[0]} frames, "
          f"{samples.shape[1]} channel(s), {sample_rate} Hz")


def build_parser() -> argparse.ArgumentParser:
    '''Declares the APAC CLI: one of --compress/--decompress required, knobs are compression-only.'''
    parser = argparse.ArgumentParser(
        description="APAC audio codec: lossless and lossy WAV compression",
        epilog="examples:\n"
               "  apac --compress -i in.wav -o out.apac --quality 100   # lossless\n"
               "  apac --compress -i in.wav -o out.apac                 # lossy, quality 50\n"
               "  apac --decompress -i out.apac -o back.wav\n"
               "  (or: python -m audio_codec.main ... before pip install -e .)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--compress", action="store_true", help="compress a WAV file into an APAC container")
    direction.add_argument("--decompress", action="store_true", help="decompress an APAC container into a WAV file")
    parser.add_argument("--quality", "-q", type=quality_arg, default=50,
                        help="compression quality 0-100, 100 = lossless (default 50); maps onto quant_step, see quality_to_quant_step")
    parser.add_argument("--quant-step", type=quant_step_arg, default=None,
                        help=f"explicit quantization step, overrides --quality (range [{MIN_QUANT_STEP}-{MAX_QUANT_STEP}])")
    parser.add_argument("--dct-quality", type=dct_quality_arg, default=None,
                        help=f"DCT pre-shaping quality [{DCT_QUALITY_MIN}-{DCT_QUALITY_MAX}], JPEG-style spectral quantization (compression only); omit to leave the DCT off")
    parser.add_argument("--dct-window", type=int, default=None,
                        help="DCT window size in samples (only with --dct-quality)")
    parser.add_argument("--input", "-i", type=Path, required=True, help="input file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable verbose (debug) logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    '''
        CLI entry point: parses arguments and runs one compress or decompress.
        Usage errors (bad flags, wrong file suffixes) exit through argparse
        with code 2; runtime errors (missing files, invalid WAV, malformed
        APAC) print a clean message and return 1.
    '''
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:   #--verbose flips the shared logger to DEBUG so codec-layer debug calls show
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(handler)

    if args.compress:
        if args.input.suffix.lower() != ".wav":
            parser.error("--compress expects a .wav input file")
        if args.output.suffix.lower() != ".apac":
            parser.error("--compress expects an .apac output file")
    else:
        if args.input.suffix.lower() != ".apac":
            parser.error("--decompress expects an .apac input file")
        if args.output.suffix.lower() != ".wav":
            parser.error("--decompress expects a .wav output file")

    try:
        if args.compress:
            settings = build_settings(quality=args.quality, quant_step=args.quant_step, dct_quality=args.dct_quality,
                                      dct_window_size=args.dct_window)
            compress_file(args.input, args.output, settings)
        else:
            decompress_file(args.input, args.output)
    except (OSError, ValueError) as e:   #FormatError, HeaderError and BlockError are ValueErrors
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())