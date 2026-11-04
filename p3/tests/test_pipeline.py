#The pipeline end to end on the shipped clip.

import json

from p3 import config as cfg
from p3 import pipeline


def test_full_run_on_the_shipped_clip(tmp_path):
    '''One full run writes the three outputs, leaves no temp files and reports a bit-exact lossless round trip.'''
    root = cfg.find_root(__file__)
    config = cfg.load_config(root)
    config["output_dir"] = str(tmp_path / "out")

    pipeline.run_pipeline(root, config)

    out = tmp_path / "out"
    assert (out / "results.json").is_file()
    assert (out / "compressed.apac").is_file()
    assert (out / "decoded.wav").is_file()
    assert not list(out.glob(pipeline.TMP_PREFIX + "*"))    #nothing partial left behind

    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved["bit_exact"] is True
    assert saved["apac_bytes"] < saved["wav_bytes"]
