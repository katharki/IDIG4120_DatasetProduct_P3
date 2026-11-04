#Integrity-layer tests: manifest build, per-file verify, dataset-wide verify.

import pytest

from p3 import config as cfg
from p3 import integrity


def test_build_and_verify_roundtrip(tmp_path, clip):
    '''A manifest built from a directory verifies that directory cleanly.'''
    manifest = integrity.build_manifest(tmp_path)
    summary = integrity.verify_dataset(tmp_path, manifest)
    assert summary["manifest_files"] == 1
    assert summary["verified"] == 1
    assert summary["mismatched"] == [] and summary["unexpected"] == []
    assert summary["absent"] == 0


def test_tampered_file_is_a_digest_mismatch(tmp_path, clip):
    '''One flipped byte becomes a DigestMismatch naming the file.'''
    manifest = integrity.build_manifest(tmp_path)
    raw = bytearray(clip.read_bytes())
    raw[-1] ^= 0xFF                                       #same size, new sha256
    clip.write_bytes(bytes(raw))
    with pytest.raises(integrity.DigestMismatch):
        integrity.verify_file(clip, manifest["files"][0])
    assert integrity.verify_dataset(tmp_path, manifest)["mismatched"] == ["clip.wav"]


def test_absent_and_unexpected_files_are_counted(tmp_path, clip):
    '''A manifest entry gone from disk counts as absent; a file not in the manifest counts as unexpected.'''
    manifest = integrity.build_manifest(tmp_path)
    clip.unlink()
    (tmp_path / "extra.wav").write_bytes(b"not audio")
    summary = integrity.verify_dataset(tmp_path, manifest)
    assert summary["absent"] == 1
    assert summary["unexpected"] == ["extra.wav"]


def test_shipped_clip_matches_the_published_manifest():
    '''The configured clip is byte-identical to its entry in the shipped manifest.'''
    root = cfg.find_root(__file__)
    config = cfg.load_config(root)
    manifest = integrity.load_manifest(cfg.manifest_path(root))
    entry = {f["path"]: f for f in manifest["files"]}[config["input_clip"]]
    assert integrity.verify_file(cfg.input_clip(root, config), entry)
