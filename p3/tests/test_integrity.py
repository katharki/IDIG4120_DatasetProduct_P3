"""Integrity checks: manifest build, per-file verify, dataset-wide verify."""

import numpy as np
import pytest
import soundfile as sf

from p3 import config as cfg
from p3 import integrity


def write_clip(directory, name="clip.wav"):
    path = directory / name
    sf.write(path, np.zeros((100, 1), dtype=np.int16), 8000, subtype="PCM_16")
    return path


def test_build_and_verify_roundtrip(tmp_path):
    write_clip(tmp_path)
    manifest = integrity.build_manifest(tmp_path)
    summary = integrity.verify_dataset(tmp_path, manifest)
    assert summary["manifest_files"] == 1
    assert summary["verified"] == 1
    assert summary["mismatched"] == []
    assert summary["absent"] == 0
    assert summary["unexpected"] == []


def test_tampered_file_is_a_digest_mismatch(tmp_path):
    clip = write_clip(tmp_path)
    manifest = integrity.build_manifest(tmp_path)
    raw = bytearray(clip.read_bytes())
    raw[-1] ^= 0xFF
    clip.write_bytes(bytes(raw))
    with pytest.raises(integrity.DigestMismatch):
        integrity.verify_file(clip, manifest["files"][0])
    summary = integrity.verify_dataset(tmp_path, manifest)
    assert summary["mismatched"] == ["clip.wav"]


def test_absent_and_unexpected_files_are_counted(tmp_path):
    write_clip(tmp_path)
    manifest = integrity.build_manifest(tmp_path)
    (tmp_path / "clip.wav").unlink()
    (tmp_path / "extra.wav").write_bytes(b"not audio")
    summary = integrity.verify_dataset(tmp_path, manifest)
    assert summary["absent"] == 1
    assert summary["unexpected"] == ["extra.wav"]


def test_shipped_clip_matches_the_published_manifest():
    root = cfg.find_root(__file__)
    config = cfg.load_config(root)
    manifest = integrity.load_manifest(cfg.manifest_path(root))
    entry = {f["path"]: f for f in manifest["files"]}[config["input_clip"]]
    assert integrity.verify_file(cfg.input_clip(root, config), entry)