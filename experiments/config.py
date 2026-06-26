from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.config import (  # noqa: E402,F401
    BINS_PER_OCTAVE,
    CQT_DB_FLOOR,
    DEFAULT_CONFIG,
    DEFAULT_PATHS,
    DURATION,
    FMIN,
    HOP_LENGTH,
    N_BINS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    AudioConfig,
    ProjectPaths,
)

