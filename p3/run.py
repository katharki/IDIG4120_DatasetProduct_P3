#P3 entry point: verify the dataset against the manifest, run the pipeline.

'''
    Runs one clip from the configured dataset through verify -> load -> APAC
    encode -> decode -> bit-exact check -> save. Full usage lives in --help;
    runtime errors print a clean message and return 1.
'''

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))     #src layout: p3 imports resolve without an install step

from p3 import config as cfg
from p3 import integrity, pipeline


def print_verify_summary(summary: dict) -> None:
    '''
        The --verify-only report: counts first, then every mismatch and unexpected file.
    '''
    print(f"manifest entries: {summary['manifest_files']}")
    print(f"verified (size and sha256 match): {summary['verified']}")
    print(f"digest mismatches: {len(summary['mismatched'])}")
    for rel in summary["mismatched"]:
        print(f"  mismatch: {rel}")
    print(f"absent (not under the dataset dir): {summary['absent']}")
    print(f"unexpected files: {len(summary['unexpected'])}")
    for rel in summary["unexpected"]:
        print(f"  unexpected: {rel}")


def main() -> int:
    '''
        Parses the arguments and runs one pipeline, or one verify with --verify-only.
    '''
    parser = argparse.ArgumentParser(description="P3 the dataset as a product")

    parser.add_argument("--verify-only", action="store_true", help="verify only, no encoding will be performed")

    parser.add_argument("--input", help="custom input clip")

    parser.add_argument("--quant-step", type=int, help="override the quantization step (1 = lossless)")
    parser.add_argument("--dataset-dir", help="override the dataset directory")
    args = parser.parse_args()

    if args.quant_step is not None and args.quant_step < cfg.MIN_QUANT_STEP:
        parser.error(f"--quant-step must be >= {cfg.MIN_QUANT_STEP}")

    root = cfg.find_root(ROOT)
    config = cfg.load_config(root)

    if args.input:
        config["input_clip"] = Path(args.input).as_posix()

    if args.quant_step is not None:
        config["quant_step"] = args.quant_step

    if args.dataset_dir:
        config["dataset_dir"] = Path(args.dataset_dir).as_posix()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    try:
        if args.verify_only:
            manifest = integrity.load_manifest(cfg.manifest_path(root))
            summary = integrity.verify_dataset(cfg.dataset_dir(root, config), manifest)
            print_verify_summary(summary)
            return 1 if summary["mismatched"] or summary["unexpected"] else 0

        results = pipeline.run_pipeline(root, config)
        for key, value in results.items():
            print(f"{key}: {value}")
        return 0

    except (OSError, ValueError, integrity.DigestMismatch) as e:    #InvalidShape and RoundTripError are ValueErrors
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
