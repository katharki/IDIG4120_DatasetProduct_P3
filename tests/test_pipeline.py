"""The pipeline end to end on the shipped clip."""

import json

from p3 import config as cfg
from p3 import pipeline


def test_full_run_on_the_shipped_clip(tmp_path):
    root = cfg.find_root(__file__)
    config = cfg.load_config(root)
    config["output_dir"] = str(tmp_path / "out")  # keep the repo output dir clean

    results = pipeline.run_pipeline(root, config)

    out = tmp_path / "out"
    assert (out / "results.json").is_file()
    assert (out / "compressed.apac").is_file()
    assert (out / "decoded.wav").is_file()
    assert not list(out.glob(".tmp-*"))

    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved["input_file"] == results["input_file"]
    assert saved["bit_exact"] is True
    assert saved["rmse"] == 0.0
    assert saved["compression_ratio"] > 1.0
    assert saved["apac_bytes"] < saved["wav_bytes"]