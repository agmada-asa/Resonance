from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.data.synthetic import (  # noqa: E402,F401
    SyntheticTrainingDataGenerator,
    WaveformSynthesizer,
    cqt_max_frequency,
    generate_sawtooth_wave,
    generate_sin_wave,
    generate_square_wave,
    generate_waveform,
    sample_frequency_for_action,
)
from resonance.actions import apply_action, encode_action  # noqa: E402,F401
from resonance.config import (  # noqa: E402,F401
    BINS_PER_OCTAVE,
    CQT_DB_FLOOR,
    DURATION,
    FMIN,
    HOP_LENGTH,
    N_BINS,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)
from resonance.features.spectrogram import audio_to_cqt  # noqa: E402,F401


def generate_training_data(seed=42):
    return SyntheticTrainingDataGenerator().generate_training_data(seed=seed, pitch_only=False)


if __name__ == "__main__":
    generate_training_data()
