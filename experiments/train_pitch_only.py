from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.config import DEFAULT_PATHS  # noqa: E402
from resonance.data.dataset import SyntheticSpectrogramDataset  # noqa: E402,F401
from resonance.models.basic import SpectrogramTransitionModel  # noqa: E402,F401
from resonance.models.unet import SpectrogramUNetModel  # noqa: E402,F401
from resonance.training.trainer import (  # noqa: E402,F401
    SpectrogramTransitionTrainer,
    TrainingConfig,
    evaluate,
    load_normalization_stats,
    move_batch_to_device,
    train_one_epoch,
)

DATA_DIR = DEFAULT_PATHS.data_dir
BUILD_DIR = DEFAULT_PATHS.build_dir

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "50"))
LEARNING_RATE = 1e-3

_TRAINER = SpectrogramTransitionTrainer(
    TrainingConfig(
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        train_filename="train_pitch_only.npz",
        val_filename="val_pitch_only.npz",
        test_filename="test_pitch_only.npz",
        best_checkpoint_name="best_spectrogram_transition_model_pitch_only.pth",
        final_checkpoint_name="spectrogram_transition_model_pitch_only_1.pth",
    )
)


def create_dataloaders(mean, std):
    return _TRAINER.create_dataloaders(mean, std)


def main():
    _TRAINER.run(plot_sample_indices=[0, 1, 2, 3, 7])


if __name__ == "__main__":
    main()
