"""P3 entry point: verify the dataset against the manifest, run the pipeline.

Usage:
    python run.py                          full pipeline on the configured clip
    python run.py --verify-only            verify every present file against the manifest
    python run.py --input fold1/x.wav     use another clip
    python run.py --quant-step 4           lossy run
    python run.py --dataset-dir PATH      point at another copy of the release

Exit codes: 0 ok, 2 missing input, 3 digest mismatch, 4 invalid shape,
5 failed round trip, 130 interrupted, 1 anything else.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from p3 import config as cfg        # noqa: E402
from p3 import integrity, pipeline  # noqa: E402


def print_verify_summary(summary: dict) -> None:
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
    parser = argparse.ArgumentParser(description="P3 dataset product entry point")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify every file present under the dataset dir "
                             "against the manifest; no encoding")
    parser.add_argument("--input",
                        help="override the input clip, relative to the dataset dir")
    parser.add_argument("--quant-step", type=int,
                        help="override the quantization step (1 = lossless)")
    parser.add_argument("--dataset-dir",
                        help="override the dataset directory, relative to the repository root")
    args = parser.parse_args()

    if args.quant_step is not None and args.quant_step < 1:
        parser.error("--quant-step must be >= 1")

    root = cfg.find_root(ROOT)
    config = cfg.load_config(root)
    if args.input:
        config["input_clip"] = Path(args.input).as_posix()
    if args.quant_step is not None:
        config["quant_step"] = args.quant_step
    if args.dataset_dir:
        config["dataset_dir"] = Path(args.dataset_dir).as_posix()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                         handlers=[logging.StreamHandler(sys.stdout)])
    handler = pipeline.attach_file_log(cfg.output_dir(root, config))
    try:
        if args.verify_only:
            manifest = integrity.load_manifest(cfg.manifest_path(root))
            summary = integrity.verify_dataset(cfg.dataset_dir(root, config), manifest)
            print_verify_summary(summary)
            return 0 if not summary["mismatched"] and not summary["unexpected"] else 3
        results = pipeline.run_pipeline(root, config)
        for key, value in results.items():
            print(f"{key}: {value}")
        return 0
    except KeyboardInterrupt:
        return pipeline.exit_code_for(KeyboardInterrupt())
    except BaseException as exc:
        pipeline.log.error("%s", exc)
        return pipeline.exit_code_for(exc)
    finally:
        pipeline.log.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    sys.exit(main())