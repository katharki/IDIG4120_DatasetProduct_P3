"""The four required failure cases: missing input, digest mismatch, invalid
shape, interrupted run."""

import json
import os

import numpy as np
import pytest
import soundfile as sf

from p3 import config as cfg
from p3 import integrity, pipeline


@pytest.fixture
def repo(tmp_path):
    """A miniature repository with its own manifest, clip and config."""
    (tmp_path / "manifest").mkdir()
    data = tmp_path / "data"
    data.mkdir()
    sf.write(data / "clip.wav", np.zeros((100, 1), dtype=np.int16), 8000,
             subtype="PCM_16")
    manifest = integrity.build_manifest(data)
    integrity.write_manifest(tmp_path / "manifest" / "urbansound8k_manifest.json",
                             manifest)
    (tmp_path / "config.json").write_text(json.dumps({
        "dataset_dir": "data",
        "input_clip": "clip.wav",
        "quant_step": 1,
        "output_dir": "output",
    }), encoding="utf-8")
    return tmp_path


def outputs_other_than_the_log(out_dir):
    return sorted(p.name for p in out_dir.iterdir()
                  if p.is_file() and not p.name.endswith(".log"))


def test_missing_input_aborts_before_anything_runs(repo):
    config = cfg.load_config(repo)
    config["input_clip"] = "clip-that-does-not-exist.wav"
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(repo, config)
    assert not (repo / "output").exists() or \
        outputs_other_than_the_log(repo / "output") == []
    assert pipeline.exit_code_for(FileNotFoundError("x")) == 2


def test_digest_mismatch_aborts_before_encoding(repo):
    clip = repo / "data" / "clip.wav"
    raw = bytearray(clip.read_bytes())
    raw[-1] ^= 0xFF
    clip.write_bytes(bytes(raw))
    config = cfg.load_config(repo)
    with pytest.raises(integrity.DigestMismatch):
        pipeline.run_pipeline(repo, config)
    assert outputs_other_than_the_log(repo / "output") == []
    log_text = (repo / "output" / "run.log").read_text(encoding="utf-8")
    assert "run failed during verify" in log_text
    assert pipeline.exit_code_for(integrity.DigestMismatch("x")) == 3


def test_invalid_shape_is_rejected():
    metadata = {"samplerate": 8000, "channels": 1, "frames": 100}
    with pytest.raises(pipeline.InvalidShape):
        pipeline.check_shape(metadata, np.zeros(100, dtype=np.int16))        # 1D
    with pytest.raises(pipeline.InvalidShape):
        pipeline.check_shape(metadata, np.zeros((100, 1), dtype=np.float64))  # not int16
    with pytest.raises(pipeline.InvalidShape):
        pipeline.check_shape(metadata, np.zeros((50, 1), dtype=np.int16))   # frames mismatch
    assert pipeline.exit_code_for(pipeline.InvalidShape("x")) == 4


def test_interrupted_run_leaves_no_partial_outputs(repo, monkeypatch):
    config = cfg.load_config(repo)
    real_replace = os.replace
    seen = []

    def flaky_replace(src, dst):
        seen.append(dst.name)
        if len(seen) == 2:
            raise KeyboardInterrupt
        real_replace(src, dst)

    monkeypatch.setattr(pipeline.os, "replace", flaky_replace)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_pipeline(repo, config)

    out_dir = repo / "output"
    assert outputs_other_than_the_log(out_dir) == []
    assert not list(out_dir.glob(".tmp-*"))
    log_text = (out_dir / "run.log").read_text(encoding="utf-8")
    assert "interrupted during save" in log_text
    assert pipeline.exit_code_for(KeyboardInterrupt()) == 130