#P3 entry point: verify the dataset against the manifest, run the pipeline.

'''
    Runs one clip from the configured dataset through verify -> load -> APAC
    encode -> decode -> bit-exact check -> save. Full usage lives in --help.

    Exit codes: 0 ok, 2 missing input, 3 digest mismatch, 4 invalid shape,
    5 failed round trip, 6 interrupted, 1 anything else.
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

        Parameters:
            - summary (dict): The verify_dataset return value.
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
        Parse the arguments, run one pipeline or one verify, and map the exit code.

        Returns:
            - int: The exit code; pipeline errors print one line naming the problem.
    '''
    parser = argparse.ArgumentParser(description="P3 dataset product entry point")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify every file present under the dataset dir "
                             "against the manifest; no encoding")
    parser.add_argument("--input",
                        help="override the input clip, relative to the dataset dir")
    parser.add_argument("--quant-step", type=int,
                        help="override the quantization step (1 = lossless)")
    parser.add_argument("--dataset-dir",
                        help="override the dataset directory, relative to the product root")
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
        return 6
    except BaseException as exc:
        pipeline.log.error("%s", exc)
        return pipeline.exit_code_for(exc)
    finally:
        pipeline.log.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    sys.exit(main())
