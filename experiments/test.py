from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.evaluation.evaluator import (  # noqa: E402,F401
    EVAL_BATCH_SIZE,
    SpectrogramEvaluator,
    evaluate_losses_by_action,
    export_audio_comparison_samples,
    peak_normalize_audio,
    plot_sample,
)
from resonance.actions import apply_action  # noqa: E402,F401
from resonance.config import BINS_PER_OCTAVE, DURATION, FMIN, HOP_LENGTH, SAMPLE_RATE  # noqa: E402,F401
from resonance.data.dataset import SyntheticSpectrogramDataset  # noqa: E402,F401
from resonance.data.synthetic import generate_waveform  # noqa: E402,F401
from resonance.models.unet import SpectrogramUNetModel  # noqa: E402,F401
from resonance.training.trainer import load_normalization_stats  # noqa: E402,F401

DATA_DIR = ROOT / "data/synthetic/v001"


def main():
    SpectrogramEvaluator().run()


if __name__ == "__main__":
    main()
