"""P3 dataset product: manifest verification and a reproducible APAC pipeline.

The audio work lives in p2 (the APAC codec); this package only wires it to the
UrbanSound8K manifest. Importing p3 puts p2/src on sys.path.
"""

import sys
from pathlib import Path

P2_SRC = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / "p2" / "src"
    if (candidate / "audio_codec").is_dir():
        P2_SRC = candidate
        break
if P2_SRC and str(P2_SRC) not in sys.path:
    sys.path.insert(0, str(P2_SRC))