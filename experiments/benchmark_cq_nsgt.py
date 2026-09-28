import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from resonance.config import ACTION_VECTOR_SIZE  # noqa: E402
from resonance.models.cq_nsgt import CQNSGTTransitionModel  # noqa: E402
from resonance.training.cq_nsgt_trainer import (  # noqa: E402
    CQNSGTTrainingConfig,
    CQNSGTTransitionTrainer,
)


def synchronize(device):
    # Include GPU execution in timings instead of measuring only dispatch
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark_case(args, batch_size, cache_layouts):
    torch.manual_seed(42)
    trainer = CQNSGTTransitionTrainer(CQNSGTTrainingConfig(
        batch_size=batch_size, num_workers=args.num_workers,
    ))
    device = trainer.select_device()
    train_loader, val_loader, _ = trainer.create_dataloaders()
    model = CQNSGTTransitionModel(
        ACTION_VECTOR_SIZE, cache_layouts=cache_layouts,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = {
        'batch_size': batch_size,
        'cache_layouts': cache_layouts,
        'device': str(device),
    }
    print(f'Benchmarking {result}', flush=True)

    for training, loader in ((True, train_loader), (False, val_loader)):
        model.train(training)
        iterator = iter(loader)
        batch_count = min(args.batches, len(loader) - args.warmup_batches)
        if batch_count <= 0:
            raise ValueError('The loader must have more batches than warmup.')
        losses = []
        with torch.set_grad_enabled(training):
            for index in range(args.warmup_batches + batch_count):
                if index == args.warmup_batches:
                    # Exclude worker startup and first-use GPU compilation
                    synchronize(device)
                    started = perf_counter()
                batch = trainer.move_batch_to_device(next(iterator), device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                prediction = model(batch['input'], batch['action_vector'])
                loss = trainer.block_mse(prediction, batch['target_flat'])
                if training:
                    loss.backward()
                    optimizer.step()
                else:
                    # Match the trainer's identity-baseline evaluation work
                    trainer.block_mse(batch['input_flat'], batch['target_flat'])
                losses.append(loss.detach())
        synchronize(device)
        seconds_per_batch = (perf_counter() - started) / batch_count
        phase = 'train' if training else 'validation'
        result[f'{phase}_seconds_per_batch'] = seconds_per_batch
        result[f'{phase}_epoch_minutes'] = seconds_per_batch * len(loader) / 60
        result[f'{phase}_finite_losses'] = bool(
            torch.isfinite(torch.stack(losses)).all().item()
        )
        print(f'{phase}: {seconds_per_batch:.4f} seconds/batch', flush=True)

    result['projected_30_epoch_hours'] = (
        30 * (result['train_epoch_minutes'] + result['validation_epoch_minutes'])
        + result['validation_epoch_minutes']
    ) / 60
    if device.type == 'mps':
        result['driver_memory_gib'] = torch.mps.driver_allocated_memory() / 2**30
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Time disposable CQ-NSGT models without writing checkpoints.',
    )
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--warmup-batches', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--compare-uncached-layouts', action='store_true')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'build' / 'cq_nsgt_benchmark.json',
    )
    args = parser.parse_args()
    if min(args.batch_sizes) <= 0 or args.batches <= 0:
        parser.error('Batch sizes and measured batch count must be positive.')
    if args.warmup_batches < 1 or args.num_workers < 0:
        parser.error('Use at least one warmup batch and non-negative workers.')

    cases = [(batch_size, True) for batch_size in args.batch_sizes]
    if args.compare_uncached_layouts:
        cases.insert(0, (args.batch_sizes[0], False))
    report = {
        'torch_version': str(torch.__version__),
        'warmup_batches': args.warmup_batches,
        'requested_measured_batches': args.batches,
        'num_workers': args.num_workers,
        'note': 'Short-run projections exclude startup, checkpoint saves, and thermal changes.',
        'cases': [],
    }
    for batch_size, cache_layouts in cases:
        report['cases'].append(benchmark_case(args, batch_size, cache_layouts))
        # Release each model and its loader workers before measuring the next
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved benchmark results to {args.output}', flush=True)


if __name__ == '__main__':
    main()
