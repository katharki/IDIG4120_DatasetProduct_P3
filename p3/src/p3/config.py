#Product root discovery, config loading and the derived paths.

from __future__ import annotations

import json
from pathlib import Path

#The file that marks the product root.
MANIFEST = Path("manifest") / "urbansound8k_manifest.json"

REQUIRED_KEYS = {"dataset_dir", "input_clip", "quant_step", "output_dir"}

#quant_step floor, shared with run.py's argument check; 1 = lossless (same floor as p2's codec).
MIN_QUANT_STEP = 1


def find_root(start: Path | str) -> Path:
    '''
        Walk up from start until the manifest marker file is found.

        Parameters:
            - start (Path | str): The folder to start the walk from.

        Returns:
            - Path: The product root, the folder holding the manifest.

        Raises:
            - FileNotFoundError: If no manifest is found above start.
    '''
    candidate = Path(start).resolve()
    for root in (candidate, *candidate.parents):
        if (root / MANIFEST).is_file():
            return root
    raise FileNotFoundError(f"product root not found above {start}")


def load_config(root: Path, name: str = "config.json") -> dict:
    '''
        Load config.json and check that every expected key is present.

        Parameters:
            - root (Path): The product root.
            - name (str): The config file name.

        Returns:
            - dict: The loaded config.

        Raises:
            - FileNotFoundError: If the config file is missing.
            - ValueError: If a key is missing or quant_step is below MIN_QUANT_STEP.
    '''
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"missing config file: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")

    step = config["quant_step"]
    if not isinstance(step, int) or step < MIN_QUANT_STEP:
        raise ValueError(f"quant_step must be a positive integer, got {step!r}")
    return config


def manifest_path(root: Path) -> Path:
    return root / MANIFEST


def dataset_dir(root: Path, config: dict) -> Path:
    return (root / config["dataset_dir"]).resolve()


def input_clip(root: Path, config: dict) -> Path:
    return dataset_dir(root, config) / config["input_clip"]


def output_dir(root: Path, config: dict) -> Path:
    return root / config["output_dir"]
