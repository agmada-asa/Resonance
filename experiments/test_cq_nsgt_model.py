import argparse
from collections import defaultdict
from pathlib import Path
from time import time
import json
import sys

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.config import (  # noqa: E402
    ACTION_VECTOR_SIZE,
    DEFAULT_PATHS,
)
from resonance.data.dataset import SyntheticCQNSGTDataset  # noqa: E402
from resonance.data.cq_nsgt_cache import (  # noqa: E402
    DEFAULT_CQ_NSGT_CACHE_DIRNAME,
    CachedCQNSGTDataset,
)
from resonance.models.cq_nsgt import CQNSGTTransitionModel  # noqa: E402
from resonance.training.cq_nsgt_trainer import (  # noqa: E402
    CQNSGTTransitionTrainer,
)


BATCH_SIZE = 2
ACTION_TYPES = ('no_action', 'gain', 'pitch_change', 'low_pass', 'high_pass')
WAVEFORM_TYPES = ('sine', 'square', 'sawtooth')
SIGNAL_TYPES = (('waveform', False), ('chord', True))


def sample_block_mse(predicted_blocks, target_blocks):
    batch_size = predicted_blocks[0].shape[0]
    squared_error = torch.zeros(
        batch_size,
        device=predicted_blocks[0].device,
    )
    value_count = 0

    # Sum each sample's error across every differently sized CQ-NSGT block
    for predicted, target in zip(predicted_blocks, target_blocks):
        squared_error += torch.sum(
            (predicted - target) ** 2,
            dim=(1, 2, 3),
        )
        value_count += target[0].numel()

    return squared_error / value_count


def add_metric(totals, name, target_loss, identity_loss, zero_loss):
    totals[name]['count'] += 1
    totals[name]['target_loss'] += target_loss
    totals[name]['identity_loss'] += identity_loss
    totals[name]['zero_loss'] += zero_loss


def metadata_categories(metadata):
    actions = [action['action'] for action in metadata.get('actions', [])]
    if not actions:
        return ['no_action']
    return sorted(set(actions))


def evaluate_test_split(model, dataset, device):
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    totals = defaultdict(
        lambda: {
            'count': 0,
            'target_loss': 0.0,
            'identity_loss': 0.0,
            'zero_loss': 0.0,
        }
    )
    trainer = CQNSGTTransitionTrainer()
    sample_offset = 0
    started = time()

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            batch = trainer.move_batch_to_device(batch, device)
            predicted = model(batch['input'], batch['action_vector'])
            target_losses = sample_block_mse(predicted, batch['target'])
            identity_losses = sample_block_mse(
                batch['input'],
                batch['target'],
            )
            zero_losses = sample_block_mse(
                [torch.zeros_like(target) for target in batch['target']],
                batch['target'],
            )

            for index in range(batch['action_vector'].shape[0]):
                metadata = dataset.metadata[sample_offset + index]
                target_loss = target_losses[index].item()
                identity_loss = identity_losses[index].item()
                zero_loss = zero_losses[index].item()
                signal_type = (
                    'chord'
                    if metadata.get('is_chord', False)
                    else 'waveform'
                )

                add_metric(
                    totals,
                    'overall',
                    target_loss,
                    identity_loss,
                    zero_loss,
                )
                add_metric(
                    totals,
                    signal_type,
                    target_loss,
                    identity_loss,
                    zero_loss,
                )
                for action in metadata_categories(metadata):
                    add_metric(
                        totals,
                        action,
                        target_loss,
                        identity_loss,
                        zero_loss,
                    )
                    add_metric(
                        totals,
                        f'{signal_type}_{action}',
                        target_loss,
                        identity_loss,
                        zero_loss,
                    )

            sample_offset += batch['action_vector'].shape[0]

            if batch_index % 50 == 0 or batch_index == len(data_loader):
                elapsed = time() - started
                remaining = (
                    elapsed / batch_index
                    * (len(data_loader) - batch_index)
                )
                print(
                    f'Test batch [{batch_index}/{len(data_loader)}] '
                    f'Elapsed: {elapsed / 60:.1f}m '
                    f'ETA: {remaining / 60:.1f}m',
                    flush=True,
                )

    # Replace the accumulated sums with averages for the report
    results = {}
    for name, values in totals.items():
        count = values['count']
        target_loss = values['target_loss'] / count
        identity_loss = values['identity_loss'] / count
        zero_loss = values['zero_loss'] / count
        results[name] = {
            'count': count,
            'target_loss': target_loss,
            'identity_loss': identity_loss,
            'zero_loss': zero_loss,
            'improvement': (
                1 - target_loss / identity_loss
                if identity_loss > 0
                else None
            ),
            'improvement_over_zero': 1 - target_loss / zero_loss,
        }

    return results


def print_metrics(results):
    print('\nHeld-out CQ-NSGT test results:')
    print(
        'category                 count  target_loss  identity_loss  '
        'zero_loss  vs_identity  vs_zero'
    )

    result_order = [
        'overall',
        'waveform',
        'chord',
        *ACTION_TYPES,
    ]
    for name in result_order:
        values = results[name]
        improvement = (
            f"{values['improvement'] * 100:.2f}%"
            if values['improvement'] is not None
            else 'n/a'
        )
        print(
            f'{name:<25}'
            f"{values['count']:>5}  "
            f"{values['target_loss']:>11.6f}  "
            f"{values['identity_loss']:>13.6f}  "
            f"{values['zero_loss']:>9.6f}  "
            f'{improvement:>11}  '
            f"{values['improvement_over_zero'] * 100:>7.2f}%"
        )

    print('\nAction and signal breakdown:')
    for action in ACTION_TYPES:
        for signal_type, _ in SIGNAL_TYPES:
            name = f'{signal_type}_{action}'
            values = results[name]
            improvement = (
                f"{values['improvement'] * 100:.2f}%"
                if values['improvement'] is not None
                else 'n/a'
            )
            print(
                f'{name:<25}'
                f"{values['count']:>5}  "
                f"{values['target_loss']:>11.6f}  "
                f"{values['identity_loss']:>13.6f}  "
                f"{values['zero_loss']:>9.6f}  "
                f'{improvement:>11}  '
                f"{values['improvement_over_zero'] * 100:>7.2f}%"
            )


def select_audio_samples(dataset):
    selected = []

    # Pick isolated actions so each exported result has one clear meaning
    for action in ACTION_TYPES:
        for signal_type, is_chord in SIGNAL_TYPES:
            for waveform_type in WAVEFORM_TYPES:
                for index, metadata in enumerate(dataset.metadata):
                    actions = metadata.get('actions', [])
                    action_matches = (
                        not actions
                        if action == 'no_action'
                        else len(actions) == 1
                        and actions[0]['action'] == action
                    )
                    if (
                        action_matches
                        and metadata['waveform_type'] == waveform_type
                        and bool(metadata.get('is_chord', False)) == is_chord
                    ):
                        selected.append(
                            (index, action, signal_type, waveform_type)
                        )
                        break

    return selected


def audio_metrics(target_audio, predicted_audio):
    error = target_audio - predicted_audio
    target_power = np.sum(target_audio ** 2)
    error_power = np.sum(error ** 2)
    snr = 10 * np.log10(target_power / max(error_power, 1e-12))

    return {
        'audio_mse': float(np.mean(error ** 2)),
        'snr_db': float(snr),
        'target_peak': float(np.max(np.abs(target_audio))),
        'predicted_peak': float(np.max(np.abs(predicted_audio))),
    }


def export_audio_samples(model, dataset, device, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, action, signal_type, waveform_type in select_audio_samples(dataset):
        sample = dataset[index]
        input_blocks = [block.unsqueeze(0) for block in sample['input']]
        target_blocks = [block.unsqueeze(0) for block in sample['target']]
        model_input = [block.to(device) for block in input_blocks]
        action_vector = sample['action_vector'].unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_blocks = model(model_input, action_vector)
        predicted_blocks = [block.cpu() for block in predicted_blocks]

        input_audio = dataset.codec.vectors_to_audio(input_blocks)
        target_audio = dataset.codec.vectors_to_audio(target_blocks)
        predicted_audio = dataset.codec.vectors_to_audio(predicted_blocks)
        input_audio = input_audio.squeeze().numpy()
        target_audio = target_audio.squeeze().numpy()
        predicted_audio = predicted_audio.squeeze().numpy()

        sample_name = (
            f'{action}_{signal_type}_{waveform_type}_sample_{index}'
        )
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Use one shared listening scale to preserve predicted gain differences
        shared_peak = max(
            np.max(np.abs(input_audio)),
            np.max(np.abs(target_audio)),
            np.max(np.abs(predicted_audio)),
        )
        listening_scale = min(1.0, 0.95 / max(shared_peak, 1e-12))
        for filename, audio in (
            ('input.wav', input_audio),
            ('target.wav', target_audio),
            ('predicted.wav', predicted_audio),
        ):
            sf.write(
                sample_dir / filename,
                (audio * listening_scale).astype(np.float32),
                dataset.config.sample_rate,
                subtype='FLOAT',
            )

        metrics = audio_metrics(target_audio, predicted_audio)
        metrics.update(
            {
                'sample_index': index,
                'action': action,
                'signal_type': signal_type,
                'waveform_type': waveform_type,
                'listening_scale': float(listening_scale),
                'directory': str(sample_dir),
            }
        )
        results.append(metrics)
        print(
            f"Exported {sample_name}: SNR={metrics['snr_db']:.2f}dB "
            f"MSE={metrics['audio_mse']:.6f}"
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a CQ-NSGT checkpoint and export listening samples.',
    )
    parser.add_argument(
        '--checkpoint', type=Path,
        default=DEFAULT_PATHS.build_dir / 'best_cq_nsgt_transition_model.pth',
    )
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    trainer = CQNSGTTransitionTrainer()
    device = trainer.select_device()

    print(f'Using checkpoint: {checkpoint_path}')
    print(f'Using device: {device}')

    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )
    completed_epochs = checkpoint.get('completed_epochs', 'unknown')
    output_dir = args.output_dir or (
        DEFAULT_PATHS.build_dir
        / f'cq_nsgt_model_evaluation_epoch_{completed_epochs}'
    )
    model = CQNSGTTransitionModel(ACTION_VECTOR_SIZE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device).eval()

    audio_dataset = SyntheticCQNSGTDataset(
        DEFAULT_PATHS.cq_nsgt_data_dir / 'metadata_test.jsonl'
    )
    vector_dataset = CachedCQNSGTDataset(
        DEFAULT_PATHS.cq_nsgt_data_dir / DEFAULT_CQ_NSGT_CACHE_DIRNAME,
        'test',
    )
    # Metrics need the action labels that accompany the cached coefficient data
    vector_dataset.metadata = audio_dataset.metadata
    vector_metrics = evaluate_test_split(model, vector_dataset, device)
    print_metrics(vector_metrics)

    audio_results = export_audio_samples(
        model,
        audio_dataset,
        device,
        output_dir / 'audio_samples',
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'results.json').open('w') as f:
        json.dump(
            {
                'checkpoint': str(checkpoint_path),
                'completed_epochs': completed_epochs,
                'vector_metrics': vector_metrics,
                'audio_samples': audio_results,
            },
            f,
            indent=2,
        )

    print(f'\nSaved evaluation results to {output_dir}.')


if __name__ == '__main__':
    main()
