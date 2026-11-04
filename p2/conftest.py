#Makes src/ importable for the test suite.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src")) #NOTE: Can fail depending on environment or renaming of src or similar, consider adding constant to config