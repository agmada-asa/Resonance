from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.actions import (  # noqa: E402,F401
    AudioActionProcessor,
    apply_action,
    change_pitch,
    encode_action,
    gain,
    high_pass,
    low_pass,
)

