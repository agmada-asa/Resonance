import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.training.cq_nsgt_trainer import (  # noqa: E402
    CQNSGTTrainingConfig,
    CQNSGTTransitionTrainer,
)


# Hyperparameters
BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 2


def parse_args():
    default_latest_checkpoint = (
        ROOT / 'build' / CQNSGTTrainingConfig().latest_checkpoint_name
    )
    parser = argparse.ArgumentParser(
        description='Train the CQ-NSGT transition model.',
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        const=default_latest_checkpoint,
        type=Path,
        metavar='CHECKPOINT',
        help=(
            'resume from CHECKPOINT; without a path, use the latest '
            f'checkpoint ({default_latest_checkpoint})'
        ),
    )
    parser.add_argument(
        '--start-epoch',
        type=int,
        help=(
            'number of already-completed epochs; only needed for legacy '
            'weight-only checkpoints'
        ),
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='training and evaluation batch size (default: %(default)s)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=NUM_WORKERS,
        help='data loader worker processes (default: %(default)s)',
    )
    args = parser.parse_args()
    if args.start_epoch is not None:
        if args.resume is None:
            parser.error('--start-epoch requires --resume')
        if args.start_epoch < 0:
            parser.error('--start-epoch must be non-negative')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if args.num_workers < 0:
        parser.error('--num-workers cannot be negative')
    return args


def main():
    args = parse_args()
    trainer = CQNSGTTransitionTrainer(
        CQNSGTTrainingConfig(
            batch_size=args.batch_size,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            num_workers=args.num_workers,
        )
    )
    trainer.run(
        resume_from=args.resume,
        start_epoch=args.start_epoch,
    )


if __name__ == '__main__':
    main()
