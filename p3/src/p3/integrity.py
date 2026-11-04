"""SHA-256 manifest: build, load, per-file and per-dataset verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK = 1 << 20


class DigestMismatch(Exception):
    """A file does not match its manifest entry (size or SHA-256)."""


def digest_file(path: Path) -> str:
    """SHA-256 of one file, read in chunks so large files stay cheap."""
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_manifest(dataset_dir: Path, dataset: str = "UrbanSound8K",
                   version: str = "1.0") -> dict:
    """Walk a dataset directory and build a manifest in the shipped format."""
    files = []
    for path in sorted(Path(dataset_dir).rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(dataset_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest_file(path),
            })
    return {"dataset": dataset, "manifest_version": version, "files": files}


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(manifest_path: Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def verify_file(path: Path, entry: dict) -> str:
    """Verify one file against its manifest entry; return the digest.

    Raises DigestMismatch naming the file on any size or digest difference.
    """
    path = Path(path)
    size = path.stat().st_size
    if size != entry["size_bytes"]:
        raise DigestMismatch(
            f"{path.name}: {size} bytes on disk, manifest says {entry['size_bytes']}"
        )
    actual = digest_file(path)
    if actual != entry["sha256"]:
        raise DigestMismatch(
            f"{path.name}: sha256 {actual}, manifest says {entry['sha256']}"
        )
    return actual


def verify_dataset(dataset_dir: Path, manifest: dict) -> dict:
    """Verify every file present under dataset_dir against the manifest.

    Returns counts plus the digests seen, so callers can reuse them (duplicate
    detection) without hashing the files twice. Files listed in the manifest
    but not on disk are reported as absent, not as errors: a repository can
    legitimately ship a subset of the release.
    """
    dataset_dir = Path(dataset_dir)
    entries = {f["path"]: f for f in manifest["files"]}
    verified, mismatched, absent, digests = 0, [], 0, {}
    for rel, entry in entries.items():
        path = dataset_dir / rel
        if not path.is_file():
            absent += 1
            continue
        try:
            digests[rel] = verify_file(path, entry)
            verified += 1
        except DigestMismatch:
            mismatched.append(rel)
    on_disk = {p.relative_to(dataset_dir).as_posix()
               for p in dataset_dir.rglob("*") if p.is_file()}
    unexpected = sorted(on_disk - set(entries))
    return {
        "manifest_files": len(entries),
        "verified": verified,
        "mismatched": mismatched,
        "absent": absent,
        "unexpected": unexpected,
        "digests": digests,
    }