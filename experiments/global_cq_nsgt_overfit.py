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

from resonance.config import DEFAULT_PATHS  # noqa: E402
from resonance.data.dataset import SyntheticCQNSGTDataset  # noqa: E402
from resonance.features.cq_nsgt import CQNSGTVectorCodec  # noqa: E402
from resonance.models.cq_nsgt_global import (  # noqa: E402
    CQNSGTGlobalTransitionModel,
)


ACTION_TYPES = ('no_action', 'gain', 'pitch_change', 'low_pass', 'high_pass')
WAVEFORM_TYPES = ('sine', 'square', 'sawtooth')
SIGNAL_TYPES = (('waveform', False), ('chord', True))
BATCH_SIZE = 5
PITCH_EPOCHS = 100
BALANCED_EPOCHS = 25
PITCH_LEARNING_RATE = 1e-3
BALANCED_LEARNING_RATE = 3e-4


def select_overfit_indices(dataset):
    selected = []

    # Select one isolated example for every action, signal, and waveform type
    for action_name in ACTION_TYPES:
        for _, is_chord in SIGNAL_TYPES:
            for waveform_type in WAVEFORM_TYPES:
                for index, metadata in enumerate(dataset.metadata):
                    actions = metadata.get('actions', [])
                    action_matches = (
                        not actions
                        if action_name == 'no_action'
                        else len(actions) == 1
                        and actions[0]['action'] == action_name
                    )
                    has_real_pitch_change = (
                        action_name != 'pitch_change'
                        or actions
                        and actions[0]['parameter'] != 0
                    )
                    if (
                        action_matches
                        and has_real_pitch_change
                        and metadata['waveform_type'] == waveform_type
                        and bool(metadata.get('is_chord', False)) == is_chord
                    ):
                        selected.append(index)
                        break

    if len(selected) != 30:
        raise RuntimeError(
            f'Expected 30 balanced overfit examples, found {len(selected)}.'
        )
    return selected


def preload_samples(dataset, indices, model):
    samples = []

    # Encode once so every epoch measures model learning instead of data setup
    for sample_number, index in enumerate(indices, start=1):
        sample = dataset[index]
        input_blocks = [block.unsqueeze(0) for block in sample['input']]
        target_blocks = [block.unsqueeze(0) for block in sample['target']]
        input_audio = dataset.codec.vectors_to_audio(input_blocks).squeeze(0)
        target_audio = dataset.codec.vectors_to_audio(target_blocks).squeeze(0)

        # The native controllers contain no learned parameters. Cache their
        # result once instead of repeating the same work on every epoch.
        with torch.no_grad():
            native_input = model._apply_native_actions(
                input_blocks,
                sample['action_vector'].unsqueeze(0),
            )

        samples.append(
            {
                'input': sample['input'],
                'native_input': [block.squeeze(0) for block in native_input],
                'target': sample['target'],
                'action_vector': sample['action_vector'],
                'input_audio': input_audio,
                'target_audio': target_audio,
                'sample_index': index,
                'action_name': (
                    dataset.metadata[index]['actions'][0]['action']
                    if dataset.metadata[index].get('actions')
                    else 'no_action'
                ),
            }
        )
        print(f'Preloaded overfit sample [{sample_number}/{len(indices)}]')

    return samples


def relative_complex_loss(predicted_blocks, target_blocks):
    batch_size = predicted_blocks[0].shape[0]
    target_energy = torch.zeros(
        batch_size,
        device=predicted_blocks[0].device,
    )
    value_count = 0

    for target in target_blocks:
        target_energy += torch.sum(target ** 2, dim=(1, 2, 3))
        value_count += target[0].numel()

    target_rms = torch.sqrt(target_energy / value_count + 1e-8)
    loss = torch.zeros_like(target_energy)
    for predicted, target in zip(predicted_blocks, target_blocks):
        normalized_error = (
            predicted - target
        ) / target_rms[:, None, None, None]
        loss += torch.sum(
            torch.nn.functional.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                reduction='none',
            ),
            dim=(1, 2, 3),
        )

    return torch.mean(loss / value_count)


def relative_magnitude_loss(predicted_blocks, target_blocks):
    batch_size = predicted_blocks[0].shape[0]
    target_energy = torch.zeros(batch_size, device=predicted_blocks[0].device)
    value_count = 0

    for target in target_blocks:
        target_block_magnitude = torch.sqrt(
            torch.sum(target ** 2, dim=1) + 1e-8
        )
        target_energy += torch.sum(target ** 2, dim=(1, 2, 3))
        value_count += target_block_magnitude[0].numel()

    target_rms = torch.sqrt(target_energy / (value_count * 2) + 1e-8)
    loss = torch.zeros_like(target_energy)
    for predicted, target in zip(predicted_blocks, target_blocks):
        predicted_magnitude = torch.sqrt(
            torch.sum(predicted ** 2, dim=1) + 1e-8
        )
        target_block_magnitude = torch.sqrt(
            torch.sum(target ** 2, dim=1) + 1e-8
        )
        normalized_error = (
            predicted_magnitude - target_block_magnitude
        ) / target_rms[:, None, None]
        loss += torch.sum(
            torch.nn.functional.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                reduction='none',
            ),
            dim=(1, 2),
        )

    return torch.mean(loss / value_count)


def relative_waveform_loss(predicted_audio, target_audio):
    # Absolute audio error preserves phase and sample accuracy. The separate
    # energy term handles relative loudness, including very quiet filters.
    return torch.nn.functional.l1_loss(predicted_audio, target_audio)


def waveform_energy_loss(predicted_audio, target_audio):
    predicted_rms = torch.sqrt(
        torch.mean(predicted_audio ** 2, dim=(1, 2)) + 1e-8
    )
    target_rms = torch.sqrt(
        torch.mean(target_audio ** 2, dim=(1, 2)) + 1e-8
    )
    return torch.mean(
        torch.abs(torch.log(predicted_rms) - torch.log(target_rms))
    )


def calculate_loss(predicted_blocks, target_blocks, predicted_audio, target_audio):
    complex_loss = relative_complex_loss(predicted_blocks, target_blocks)
    magnitude_loss = relative_magnitude_loss(predicted_blocks, target_blocks)
    waveform_loss = relative_waveform_loss(predicted_audio, target_audio)
    energy_loss = waveform_energy_loss(predicted_audio, target_audio)
    total_loss = (
        waveform_loss
        + 0.1 * complex_loss
        + 0.05 * magnitude_loss
        + 0.1 * energy_loss
    )
    return {
        'total': total_loss,
        'complex': complex_loss,
        'magnitude': magnitude_loss,
        'waveform': waveform_loss,
        'energy': energy_loss,
    }


def move_batch_to_device(batch, device):
    return {
        'input': [block.to(device) for block in batch['input']],
        'native_input': [
            block.to(device)
            for block in batch['native_input']
        ],
        'target': [block.to(device) for block in batch['target']],
        'action_vector': batch['action_vector'].to(device),
        'target_audio': batch['target_audio'].to(device),
        'sample_index': batch['sample_index'],
    }


def train(
    model,
    samples,
    codec,
    device,
    output_dir,
    stage_name,
    max_epochs,
    learning_rate,
    checkpoint_name,
    batch_size=BATCH_SIZE,
):
    data_loader = DataLoader(samples, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    started = time()
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        totals = defaultdict(float)

        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            predicted_blocks = model.forward_native(
                batch['native_input'],
                batch['action_vector'],
            )
            predicted_audio = codec.vectors_to_audio(predicted_blocks)
            losses = calculate_loss(
                predicted_blocks,
                batch['target'],
                predicted_audio,
                batch['target_audio'],
            )
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for name, loss in losses.items():
                totals[name] += loss.item()

        averages = {
            name: value / len(data_loader)
            for name, value in totals.items()
        }
        history.append({'epoch': epoch, **averages})

        if averages['total'] < best_loss:
            best_loss = averages['total']
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                },
                output_dir / checkpoint_name,
            )

        if epoch == 1 or epoch % 5 == 0:
            elapsed = time() - started
            remaining = elapsed / epoch * (max_epochs - epoch)
            print(
                f'{stage_name} epoch [{epoch}/{max_epochs}] '
                f"Total: {averages['total']:.4f} "
                f"Complex: {averages['complex']:.4f} "
                f"Magnitude: {averages['magnitude']:.4f} "
                f"Waveform: {averages['waveform']:.4f} "
                f"Energy: {averages['energy']:.4f} "
                f'Elapsed: {elapsed / 60:.1f}m '
                f'ETA: {remaining / 60:.1f}m',
                flush=True,
            )

        if averages['total'] < 0.02:
            print(f'Overfit target reached at epoch {epoch}.')
            break

    return history


def block_mse(predicted_blocks, target_blocks):
    squared_error = sum(
        torch.sum((predicted - target) ** 2).item()
        for predicted, target in zip(predicted_blocks, target_blocks)
    )
    value_count = sum(target.numel() for target in target_blocks)
    return squared_error / value_count


def export_and_evaluate(model, dataset, samples, device, output_dir):
    totals = defaultdict(lambda: {'count': 0, 'snr': 0.0, 'mse': 0.0})
    results = []
    model.eval()

    for sample in samples:
        index = sample['sample_index']
        metadata = dataset.metadata[index]
        input_blocks = [block.unsqueeze(0).to(device) for block in sample['input']]
        target_blocks = [block.unsqueeze(0).to(device) for block in sample['target']]
        action_vector = sample['action_vector'].unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_blocks = model(input_blocks, action_vector)
        predicted_blocks_cpu = [block.cpu() for block in predicted_blocks]

        evaluation_codec = dataset.codec
        predicted_audio = evaluation_codec.vectors_to_audio(
            predicted_blocks_cpu
        ).squeeze().numpy()
        input_audio = sample['input_audio'].squeeze().numpy()
        target_audio = sample['target_audio'].squeeze().numpy()
        error = target_audio - predicted_audio
        snr = 10 * np.log10(
            np.sum(target_audio ** 2) / max(np.sum(error ** 2), 1e-12)
        )

        actions = metadata.get('actions', [])
        action_name = actions[0]['action'] if actions else 'no_action'
        signal_type = 'chord' if metadata.get('is_chord', False) else 'waveform'
        waveform_type = metadata['waveform_type']
        name = f'{action_name}_{signal_type}_{waveform_type}_{index}'
        sample_dir = output_dir / 'audio' / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        shared_peak = max(
            np.max(np.abs(input_audio)),
            np.max(np.abs(target_audio)),
            np.max(np.abs(predicted_audio)),
        )
        scale = min(1.0, 0.95 / max(shared_peak, 1e-12))
        for filename, audio in (
            ('input.wav', input_audio),
            ('target.wav', target_audio),
            ('predicted.wav', predicted_audio),
        ):
            sf.write(
                sample_dir / filename,
                (audio * scale).astype(np.float32),
                dataset.config.sample_rate,
                subtype='FLOAT',
            )

        mse = block_mse(predicted_blocks_cpu, [
            block.unsqueeze(0)
            for block in sample['target']
        ])
        totals[action_name]['count'] += 1
        totals[action_name]['snr'] += float(snr)
        totals[action_name]['mse'] += mse
        results.append(
            {
                'sample_index': index,
                'action': action_name,
                'signal_type': signal_type,
                'waveform_type': waveform_type,
                'snr_db': float(snr),
                'vector_mse': mse,
                'directory': str(sample_dir),
            }
        )

    action_results = {}
    print('\nOverfit audio results:')
    for action_name in ACTION_TYPES:
        values = totals[action_name]
        count = values['count']
        if count == 0:
            continue
        action_results[action_name] = {
            'count': count,
            'mean_snr_db': values['snr'] / count,
            'mean_vector_mse': values['mse'] / count,
        }
        print(
            f'{action_name:<14} '
            f"SNR: {action_results[action_name]['mean_snr_db']:.2f}dB "
            f"MSE: {action_results[action_name]['mean_vector_mse']:.6f}"
        )

    return action_results, results


def main():
    # Keep this run separate from the earlier band-shifting experiment
    output_dir = DEFAULT_PATHS.build_dir / 'global_cq_nsgt_native_overfit'
    output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device('mps')
        if torch.backends.mps.is_available()
        else torch.device('cpu')
    )

    print(f'Using device: {device}')
    dataset = SyntheticCQNSGTDataset(
        DEFAULT_PATHS.cq_nsgt_data_dir / 'metadata_train.jsonl'
    )
    indices = select_overfit_indices(dataset)
    model = CQNSGTGlobalTransitionModel().to(device)
    samples = preload_samples(dataset, indices, model.cpu())
    model = model.to(device)
    training_codec = CQNSGTVectorCodec(device=device)

    # Pitch is the remaining difficult action, so first give its six examples
    # an undiluted overfit stage before checking that the balanced set remains
    # stable. This also makes failure much easier to diagnose.
    pitch_samples = [
        sample
        for sample in samples
        if sample['action_name'] == 'pitch_change'
    ]
    pitch_history = train(
        model,
        pitch_samples,
        training_codec,
        device,
        output_dir,
        stage_name='Pitch',
        max_epochs=PITCH_EPOCHS,
        learning_rate=PITCH_LEARNING_RATE,
        checkpoint_name='best_pitch_overfit_model.pth',
        batch_size=len(pitch_samples),
    )

    pitch_checkpoint = torch.load(
        output_dir / 'best_pitch_overfit_model.pth',
        map_location='cpu',
        weights_only=True,
    )
    model.load_state_dict(pitch_checkpoint['model_state_dict'])
    model = model.to(device)

    balanced_history = train(
        model,
        samples,
        training_codec,
        device,
        output_dir,
        stage_name='Balanced',
        max_epochs=BALANCED_EPOCHS,
        learning_rate=BALANCED_LEARNING_RATE,
        checkpoint_name='best_native_overfit_model.pth',
    )

    checkpoint = torch.load(
        output_dir / 'best_native_overfit_model.pth',
        map_location='cpu',
        weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    action_results, sample_results = export_and_evaluate(
        model,
        dataset,
        samples,
        device,
        output_dir,
    )

    with (output_dir / 'results.json').open('w') as f:
        json.dump(
            {
                'selected_indices': indices,
                'best_epoch': checkpoint['epoch'],
                'best_loss': checkpoint['loss'],
                'pitch_history': pitch_history,
                'balanced_history': balanced_history,
                'actions': action_results,
                'samples': sample_results,
            },
            f,
            indent=2,
        )

    print(f'\nSaved overfit results to {output_dir}.')


if __name__ == '__main__':
    main()
