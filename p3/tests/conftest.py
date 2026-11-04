#Makes src importable for pytest and holds the shared miniature-repo fixtures.

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from p3 import config as cfg
from p3 import integrity

#synthetic clip parameters
CLIP_FRAMES = 100
CLIP_CHANNELS = 1
CLIP_RATE = 8000


@pytest.fixture
def clip(tmp_path):
    '''
        Creates a minimal PCM_16 clip in tmp_path and returns its path.
    '''
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros((CLIP_FRAMES, CLIP_CHANNELS), dtype=np.int16), CLIP_RATE, subtype="PCM_16")
    
    return path


@pytest.fixture
def repo(tmp_path, clip):
    '''
        A miniature repository with its own manifest, clip and config, so the
        failure tests can run against a dataset that is not the shipped one.
    '''
    data = tmp_path / "data"
    data.mkdir()
    clip.rename(data / clip.name)

    integrity.write_manifest(tmp_path / cfg.MANIFEST, integrity.build_manifest(data))
    (tmp_path / "config.json").write_text(json.dumps({
        "dataset_dir": "data",
        "input_clip": clip.name,
        "quant_step": 1,
        "output_dir": "output",
    }), encoding="utf-8")

    return tmp_path
