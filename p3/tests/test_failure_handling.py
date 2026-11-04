#The four required failure cases: missing input, digest mismatch, invalid shape, interrupted run.

import os

import numpy as np
import pytest

from p3 import config as cfg
from p3 import integrity, pipeline


def saved_outputs(out_dir):
    '''Files in out_dir other than the log: none should survive a failed run.'''
    return sorted(p.name for p in out_dir.iterdir()
                  if p.is_file() and not p.name.endswith(".log"))


def test_missing_input_aborts_before_anything_runs(repo):
    '''A configured clip that does not exist aborts in preflight, before the output dir is even created.'''
    config = cfg.load_config(repo)
    config["input_clip"] = "clip-that-does-not-exist.wav"
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(repo, config)
    assert not (repo / "output").exists()


def test_digest_mismatch_aborts_before_encoding(repo):
    '''A clip that no longer matches its manifest entry aborts during verify: no outputs, the log names the step.'''
    clip = repo / "data" / "clip.wav"
    raw = bytearray(clip.read_bytes())
    raw[-1] ^= 0xFF                                        #same size, new sha256
    clip.write_bytes(bytes(raw))
    with pytest.raises(integrity.DigestMismatch):
        pipeline.run_pipeline(repo, cfg.load_config(repo))
    assert saved_outputs(repo / "output") == []
    assert "run failed during verify" in (repo / "output" / "run.log").read_text(encoding="utf-8")


def test_invalid_shape_is_rejected():
    '''check_shape rejects 1D arrays, non-int16 dtypes and frames that disagree with the header.'''
    metadata = {"samplerate": 8000, "channels": 1, "frames": 100}
    with pytest.raises(pipeline.InvalidShape):
        pipeline.check_shape(metadata, np.zeros(100, dtype=np.int16))              #1D, not (frames, channels)
    with pytest.raises(pipeline.InvalidShape):
        pipeline.check_shape(metadata, np.zeros((100, 1), dtype=np.float64))       #not int16
    with pytest.raises(pipeline.InvalidShape):
        pipeline.check_shape(metadata, np.zeros((50, 1), dtype=np.int16))          #frames disagree with the header


def test_interrupted_run_leaves_no_partial_outputs(repo, monkeypatch):
    '''A Ctrl+C between two renames leaves no output and no temp file; the log names the step.'''
    real_replace = os.replace
    seen = []

    def flaky_replace(src, dst):
        seen.append(dst.name)
        if len(seen) == 2:
            raise KeyboardInterrupt                        #drop the interrupt between two renames
        real_replace(src, dst)

    monkeypatch.setattr(pipeline.os, "replace", flaky_replace)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_pipeline(repo, cfg.load_config(repo))

    out_dir = repo / "output"
    assert saved_outputs(out_dir) == []
    assert not list(out_dir.glob(pipeline.TMP_PREFIX + "*"))
    assert "interrupted during save" in (out_dir / "run.log").read_text(encoding="utf-8")
