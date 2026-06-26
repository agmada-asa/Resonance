from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.features.spectrogram import (  # noqa: E402,F401
    SpectrogramTransformer,
    audio_to_cqt,
    audio_to_spectrogram,
)

