import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random
import sys
from time import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.actions import AudioActionProcessor  # noqa: E402
from resonance.config import (  # noqa: E402
    ACTION_DIMENSIONS,
    ACTION_VECTOR_SIZE,
    DEFAULT_CONFIG,
    MAX_ACTIONS,
)
from resonance.data.synthetic import WaveformSynthesizer  # noqa: E402
from resonance.models.latent_audio import (  # noqa: E402
    LatentActionTransitionModel,
    WaveformAutoencoder,
)


DEFAULT_OUTPUT_DIR = ROOT / 'build' / 'latent_delta_prototype'
DEFAULT_METADATA_DIR = ROOT / 'data' / 'synthetic' / 'cq_nsgt_v001'
_WORKER_SYNTHESIZER = None
_WORKER_ACTIONS = None


def initialize_audio_worker():
    global _WORKER_SYNTHESIZER, _WORKER_ACTIONS
    _WORKER_SYNTHESIZER = WaveformSynthesizer(DEFAULT_CONFIG)
    _WORKER_ACTIONS = AudioActionProcessor(DEFAULT_CONFIG)


def generate_audio_pair(metadata):
    if _WORKER_SYNTHESIZER is None:
        initialize_audio_worker()
    generate = (
        _WORKER_SYNTHESIZER.generate_chords
        if metadata.get('is_chord', False)
        else _WORKER_SYNTHESIZER.generate_waveform
    )
    source = generate(
        metadata['waveform_type'],
        metadata['frequency'],
        metadata['amplitude'],
    )
    target = source
    action_vector = np.zeros(
        (MAX_ACTIONS, ACTION_DIMENSIONS), dtype=np.float32,
    )
    for slot, action in enumerate(metadata.get('actions', [])):
        target, encoded = _WORKER_ACTIONS.apply_action(
            target, action['action'], action['parameter'],
        )
        action_vector[slot] = encoded
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError(f"Generated non-finite audio for sample {metadata['id']}.")
    return source, target, action_vector.reshape(ACTION_VECTOR_SIZE)


def choose_metadata(path, count, seed):
    with Path(path).open() as source:
        rows = [json.loads(line) for line in source]
    if count > len(rows):
        raise ValueError(f'Requested {count} examples from a {len(rows)} row split.')
    generator = random.Random(seed)
    generator.shuffle(rows)

    # Keep the uncommon identity examples represented in every small prototype
    identities = [row for row in rows if not row.get('actions')]
    others = [row for row in rows if row.get('actions')]
    identity_count = min(len(identities), max(8, count // 32))
    selected = identities[:identity_count] + others[:count - identity_count]
    generator.shuffle(selected)
    return selected


def cache_paths(cache_dir, split):
    return {
        'audio': cache_dir / f'{split}_audio.npy',
        'actions': cache_dir / f'{split}_actions.npy',
        'metadata': cache_dir / f'{split}_metadata.json',
    }


def build_audio_cache(metadata_dir, cache_dir, split, count, seed, workers):
    paths = cache_paths(cache_dir, split)
    expected_shape = (count, 2, DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.duration)
    if all(path.is_file() for path in paths.values()):
        audio = np.load(paths['audio'], mmap_mode='r')
        actions = np.load(paths['actions'], mmap_mode='r')
        if audio.shape == expected_shape and actions.shape == (count, ACTION_VECTOR_SIZE):
            print(f'Reusing {split} waveform cache with {count} examples.')
            return paths

    metadata = choose_metadata(
        metadata_dir / f'metadata_{split}.jsonl', count, seed,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio = np.lib.format.open_memmap(
        paths['audio'], mode='w+', dtype=np.float16, shape=expected_shape,
    )
    actions = np.lib.format.open_memmap(
        paths['actions'], mode='w+', dtype=np.float32,
        shape=(count, ACTION_VECTOR_SIZE),
    )
    print(f'Generating {count} {split} waveform pairs with {workers} workers...')
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_audio_worker,
    ) as executor:
        for index, (source, target, action_vector) in enumerate(
            executor.map(generate_audio_pair, metadata, chunksize=1)
        ):
            audio[index, 0] = source
            audio[index, 1] = target
            actions[index] = action_vector
            if (index + 1) % 50 == 0 or index + 1 == count:
                print(f'  {split} [{index + 1}/{count}]', flush=True)
    audio.flush()
    actions.flush()
    paths['metadata'].write_text(json.dumps(metadata, indent=2) + '\n')
    return paths


class WaveformPairDataset(Dataset):
    def __init__(self, paths):
        self.audio = np.load(paths['audio'], mmap_mode='r')
        self.actions = np.load(paths['actions'], mmap_mode='r')
        self.metadata = json.loads(paths['metadata'].read_text())

    def __len__(self):
        return self.audio.shape[0]

    def __getitem__(self, index):
        # Copy from the read-only mapping before DataLoader creates tensors
        audio = torch.from_numpy(np.array(self.audio[index], dtype=np.float32))
        action = torch.from_numpy(np.array(self.actions[index], dtype=np.float32))
        return audio, action


class LatentPairDataset(Dataset):
    def __init__(self, path, actions_path):
        self.latents = np.load(path, mmap_mode='r')
        self.actions = np.load(actions_path, mmap_mode='r')

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):
        latents = torch.from_numpy(
            np.array(self.latents[index], dtype=np.float32)
        )
        action = torch.from_numpy(
            np.array(self.actions[index], dtype=np.float32)
        )
        return latents, action


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def stft_loss(predicted, target, window):
    predicted_stft = torch.stft(
        predicted[:, 0], n_fft=1024, hop_length=256,
        window=window, return_complex=True,
    )
    target_stft = torch.stft(
        target[:, 0], n_fft=1024, hop_length=256,
        window=window, return_complex=True,
    )
    return F.l1_loss(
        torch.log1p(torch.abs(predicted_stft)),
        torch.log1p(torch.abs(target_stft)),
    )


def relative_squared_error(predicted, target, dimensions):
    error = torch.sum((predicted - target) ** 2, dim=dimensions)
    signal = torch.sum(target ** 2, dim=dimensions)
    value_count = math.prod(target.shape[dimension] for dimension in dimensions)
    return torch.mean(error / signal.clamp_min(value_count * 1e-6))


def autoencoder_loss(
    predicted, target, predicted_coefficients, target_coefficients,
):
    waveform = relative_squared_error(predicted, target, (-2, -1))
    coefficients = relative_squared_error(
        predicted_coefficients, target_coefficients, (-3, -2, -1),
    )
    predicted_magnitude = torch.sqrt(
        torch.sum(predicted_coefficients ** 2, dim=1) + 1e-8
    )
    target_magnitude = torch.sqrt(
        torch.sum(target_coefficients ** 2, dim=1) + 1e-8
    )
    magnitude = relative_squared_error(
        predicted_magnitude, target_magnitude, (-2, -1),
    )
    total = coefficients + 0.5 * waveform + 0.1 * magnitude
    return total, waveform, coefficients


def signal_to_noise_ratio(predicted, target):
    signal = torch.sum(target ** 2, dim=(-2, -1))
    error = torch.sum((target - predicted) ** 2, dim=(-2, -1))
    return 10 * torch.log10(signal.clamp_min(1e-12) / error.clamp_min(1e-12))


def run_autoencoder_epoch(
    model, loader, device, window, optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = 0
    with torch.set_grad_enabled(training):
        for audio, _ in loader:
            # Both the before and after states must survive the same bottleneck
            states = audio.reshape(-1, 1, audio.shape[-1]).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            target_coefficients = model.analysis(states)
            latent = model.encode(states)
            predicted_coefficients = model.decode_coefficients(latent)
            reconstructed = model.synthesis(
                predicted_coefficients, states.shape[-1],
            )
            total, waveform, spectral = autoencoder_loss(
                reconstructed, states,
                predicted_coefficients, target_coefficients,
            )
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch_size = states.shape[0]
            totals['loss'] += total.detach().item() * batch_size
            totals['waveform_l1'] += waveform.detach().item() * batch_size
            totals['spectral_l1'] += spectral.detach().item() * batch_size
            totals['snr'] += signal_to_noise_ratio(
                reconstructed.detach(), states,
            ).sum().item()
            examples += batch_size
    return {name: value / examples for name, value in totals.items()}


def train_autoencoder(args, train_dataset, val_dataset, device, output_dir):
    model = WaveformAutoencoder(args.latent_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.autoencoder_lr, weight_decay=1e-5,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.autoencoder_batch_size,
        shuffle=True, num_workers=args.loader_workers,
        persistent_workers=args.loader_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.autoencoder_batch_size,
        num_workers=args.loader_workers,
        persistent_workers=args.loader_workers > 0,
    )
    window = torch.hann_window(1024, device=device)
    checkpoint_path = output_dir / 'best_autoencoder.pth'
    best_loss = float('inf')
    stale_epochs = 0
    history = []
    for epoch in range(1, args.autoencoder_epochs + 1):
        started = time()
        train_metrics = run_autoencoder_epoch(
            model, train_loader, device, window, optimizer,
        )
        with torch.no_grad():
            val_metrics = run_autoencoder_epoch(
                model, val_loader, device, window,
            )
        record = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': val_metrics,
            'seconds': time() - started,
        }
        history.append(record)
        print(
            f"Autoencoder [{epoch}/{args.autoencoder_epochs}] "
            f"train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} "
            f"val_snr={val_metrics['snr']:.2f}dB "
            f"time={record['seconds']:.1f}s",
            flush=True,
        )
        if val_metrics['loss'] < best_loss - 1e-5:
            best_loss = val_metrics['loss']
            stale_epochs = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_channels': args.latent_channels,
                'validation': val_metrics,
                'epoch': epoch,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print('Stopping autoencoder after validation stopped improving.')
                break
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval(), history, checkpoint


def encode_latent_cache(model, dataset, path, device, batch_size):
    latent_channels, latent_length = model.latent_shape(dataset.audio.shape[-1])
    expected_shape = (
        len(dataset), 2, latent_channels, latent_length,
    )
    # The encoding changes whenever the autoencoder checkpoint changes, so
    # rebuild this small cache rather than silently reusing stale latent values.
    latents = np.lib.format.open_memmap(
        path, mode='w+', dtype=np.float16, shape=expected_shape,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    offset = 0
    with torch.no_grad():
        for audio, _ in loader:
            batch = audio.shape[0]
            states = audio.reshape(-1, 1, audio.shape[-1]).to(device)
            encoded = model.encode(states).reshape(
                batch, 2, latent_channels, -1,
            )
            latents[offset:offset + batch] = encoded.cpu().numpy()
            offset += batch
    latents.flush()
    print(f'Saved latent cache: {path}')
    return path


def latent_statistics(path):
    values = np.load(path, mmap_mode='r')
    mean = np.mean(values, axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
    variance = np.mean(
        (values.astype(np.float32) - mean[None, None, :, None]) ** 2,
        axis=(0, 1, 3),
        dtype=np.float64,
    ).astype(np.float32)
    return mean, np.sqrt(variance + 1e-6)


def run_transition_epoch(
    model, loader, mean, std, device, optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_identity = 0.0
    examples = 0
    with torch.set_grad_enabled(training):
        for latents, action in loader:
            source = latents[:, 0].to(device)
            target = latents[:, 1].to(device)
            action = action.to(device)
            source = (source - mean) / std
            target = (target - mean) / std
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted = model(source, action)
            loss = F.mse_loss(predicted, target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch_size = source.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_identity += F.mse_loss(source, target).item() * batch_size
            examples += batch_size
    return {
        'target_loss': total_loss / examples,
        'identity_loss': total_identity / examples,
        'improvement': 1 - total_loss / max(total_identity, 1e-12),
    }


def train_transition(
    args, train_latent_path, val_latent_path,
    train_actions_path, val_actions_path, device, output_dir,
):
    mean_values, std_values = latent_statistics(train_latent_path)
    mean = torch.from_numpy(mean_values)[None, :, None].to(device)
    std = torch.from_numpy(std_values)[None, :, None].to(device)
    train_dataset = LatentPairDataset(train_latent_path, train_actions_path)
    val_dataset = LatentPairDataset(val_latent_path, val_actions_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.transition_batch_size,
        shuffle=True, num_workers=args.loader_workers,
        persistent_workers=args.loader_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.transition_batch_size,
        num_workers=args.loader_workers,
        persistent_workers=args.loader_workers > 0,
    )
    model = LatentActionTransitionModel(
        ACTION_VECTOR_SIZE,
        latent_channels=np.load(train_latent_path, mmap_mode='r').shape[2],
        width=args.transition_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.transition_lr, weight_decay=1e-5,
    )
    checkpoint_path = output_dir / 'best_latent_transition.pth'
    best_loss = float('inf')
    stale_epochs = 0
    history = []
    for epoch in range(1, args.transition_epochs + 1):
        started = time()
        train_metrics = run_transition_epoch(
            model, train_loader, mean, std, device, optimizer,
        )
        with torch.no_grad():
            val_metrics = run_transition_epoch(
                model, val_loader, mean, std, device,
            )
        record = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': val_metrics,
            'seconds': time() - started,
        }
        history.append(record)
        print(
            f"Transition [{epoch}/{args.transition_epochs}] "
            f"train={train_metrics['target_loss']:.5f} "
            f"val={val_metrics['target_loss']:.5f} "
            f"vs_identity={val_metrics['improvement'] * 100:.1f}% "
            f"time={record['seconds']:.1f}s",
            flush=True,
        )
        if val_metrics['target_loss'] < best_loss - 1e-5:
            best_loss = val_metrics['target_loss']
            stale_epochs = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': torch.from_numpy(mean_values),
                'std': torch.from_numpy(std_values),
                'validation': val_metrics,
                'epoch': epoch,
                'latent_channels': model.latent_channels,
                'width': args.transition_width,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print('Stopping transition model after validation stopped improving.')
                break
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return (
        model.to(device).eval(), history, checkpoint,
        mean, std,
    )


def action_categories(metadata):
    names = sorted({action['action'] for action in metadata.get('actions', [])})
    return names or ['no_action']


def add_audio_metrics(totals, name, predicted, target, identity, ceiling):
    snr = signal_to_noise_ratio(predicted, target)
    identity_snr = signal_to_noise_ratio(identity, target)
    ceiling_snr = signal_to_noise_ratio(ceiling, target)
    totals[name]['count'] += predicted.shape[0]
    totals[name]['snr'] += snr.sum().item()
    totals[name]['identity_snr'] += identity_snr.sum().item()
    totals[name]['reconstruction_ceiling_snr'] += ceiling_snr.sum().item()
    totals[name]['mse'] += torch.sum((predicted - target) ** 2).item()
    totals[name]['target_values'] += target.numel()


def evaluate_audio(
    autoencoder, transition, dataset, mean, std, device,
    output_dir, sample_count=12,
):
    loader = DataLoader(dataset, batch_size=8)
    totals = defaultdict(lambda: defaultdict(float))
    metadata_offset = 0
    audio_samples = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for audio, action in loader:
            source = audio[:, 0:1].to(device)
            target = audio[:, 1:2].to(device)
            source_latent = autoencoder.encode(source)
            target_latent = autoencoder.encode(target)
            predicted_normalized = transition(
                (source_latent - mean) / std,
                action.to(device),
            )
            predicted_latent = predicted_normalized * std + mean
            predicted = autoencoder.decode(predicted_latent, source.shape[-1])
            ceiling = autoencoder.decode(target_latent, target.shape[-1])
            for index in range(source.shape[0]):
                metadata = dataset.metadata[metadata_offset + index]
                names = ['overall', *action_categories(metadata)]
                for name in names:
                    add_audio_metrics(
                        totals, name,
                        predicted[index:index + 1], target[index:index + 1],
                        source[index:index + 1], ceiling[index:index + 1],
                    )
                if len(audio_samples) < sample_count:
                    sample_dir = output_dir / f"sample_{metadata['id']}"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    arrays = {
                        'input': source[index, 0].cpu().numpy(),
                        'target': target[index, 0].cpu().numpy(),
                        'predicted': predicted[index, 0].cpu().numpy(),
                        'target_reconstruction': ceiling[index, 0].cpu().numpy(),
                    }
                    peak = max(np.max(np.abs(value)) for value in arrays.values())
                    scale = min(1.0, 0.95 / max(peak, 1e-12))
                    for filename, values in arrays.items():
                        sf.write(
                            sample_dir / f'{filename}.wav', values * scale,
                            DEFAULT_CONFIG.sample_rate, subtype='FLOAT',
                        )
                    audio_samples.append({
                        'sample_id': metadata['id'],
                        'actions': metadata.get('actions', []),
                        'directory': str(sample_dir),
                        'snr_db': signal_to_noise_ratio(
                            predicted[index:index + 1], target[index:index + 1],
                        ).item(),
                    })
            metadata_offset += source.shape[0]
    metrics = {}
    for name, values in totals.items():
        count = int(values['count'])
        metrics[name] = {
            'count': count,
            'mean_snr_db': values['snr'] / count,
            'identity_mean_snr_db': values['identity_snr'] / count,
            'target_reconstruction_mean_snr_db': (
                values['reconstruction_ceiling_snr'] / count
            ),
            'waveform_mse': values['mse'] / values['target_values'],
        }
    return metrics, audio_samples


def encode_single_action(action):
    processor = AudioActionProcessor(DEFAULT_CONFIG)
    vector = np.zeros((MAX_ACTIONS, ACTION_DIMENSIONS), dtype=np.float32)
    vector[0] = processor.encode_action(action['action'], action['parameter'])
    return torch.from_numpy(vector.reshape(ACTION_VECTOR_SIZE))


def evaluate_rollouts(
    autoencoder, transition, dataset, mean, std, device, limit=24,
):
    results = []
    with torch.no_grad():
        for index, metadata in enumerate(dataset.metadata):
            actions = metadata.get('actions', [])
            if len(actions) < 2:
                continue
            audio, full_action = dataset[index]
            source = audio[0:1].unsqueeze(0).to(device)
            target = audio[1:2].unsqueeze(0).to(device)
            latent = (autoencoder.encode(source) - mean) / std
            direct = transition(latent, full_action[None].to(device))
            rollout = latent
            for action in actions:
                rollout = transition(
                    rollout, encode_single_action(action)[None].to(device),
                )
            direct_audio = autoencoder.decode(
                direct * std + mean, source.shape[-1],
            )
            rollout_audio = autoencoder.decode(
                rollout * std + mean, source.shape[-1],
            )
            results.append({
                'sample_id': metadata['id'],
                'action_count': len(actions),
                'actions': [action['action'] for action in actions],
                'direct_snr_db': signal_to_noise_ratio(
                    direct_audio, target,
                ).item(),
                'rollout_snr_db': signal_to_noise_ratio(
                    rollout_audio, target,
                ).item(),
                'identity_snr_db': signal_to_noise_ratio(source, target).item(),
            })
            if len(results) >= limit:
                break
    return {
        'count': len(results),
        'mean_direct_snr_db': float(np.mean([
            result['direct_snr_db'] for result in results
        ])),
        'mean_rollout_snr_db': float(np.mean([
            result['rollout_snr_db'] for result in results
        ])),
        'mean_identity_snr_db': float(np.mean([
            result['identity_snr_db'] for result in results
        ])),
        'samples': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test a compact learned waveform state and latent deltas.',
    )
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--metadata-dir', type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument('--train-samples', type=int, default=768)
    parser.add_argument('--val-samples', type=int, default=192)
    parser.add_argument('--cache-workers', type=int, default=6)
    parser.add_argument('--loader-workers', type=int, default=2)
    parser.add_argument('--latent-channels', type=int, default=24)
    parser.add_argument('--autoencoder-batch-size', type=int, default=4)
    parser.add_argument('--autoencoder-epochs', type=int, default=20)
    parser.add_argument('--autoencoder-lr', type=float, default=3e-4)
    parser.add_argument('--transition-batch-size', type=int, default=32)
    parser.add_argument('--transition-epochs', type=int, default=40)
    parser.add_argument('--transition-width', type=int, default=64)
    parser.add_argument('--transition-lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=6)
    args = parser.parse_args()
    if min(args.train_samples, args.val_samples, args.latent_channels) <= 0:
        parser.error('Sample counts and latent channels must be positive.')

    torch.manual_seed(42)
    np.random.seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'waveform_cache'
    train_paths = build_audio_cache(
        args.metadata_dir, cache_dir, 'train', args.train_samples,
        seed=2026, workers=args.cache_workers,
    )
    val_paths = build_audio_cache(
        args.metadata_dir, cache_dir, 'val', args.val_samples,
        seed=2027, workers=args.cache_workers,
    )
    train_dataset = WaveformPairDataset(train_paths)
    val_dataset = WaveformPairDataset(val_paths)
    device = select_device()
    print(f'Using device: {device}')

    autoencoder, autoencoder_history, autoencoder_checkpoint = train_autoencoder(
        args, train_dataset, val_dataset, device, args.output_dir,
    )
    train_latent_path = encode_latent_cache(
        autoencoder, train_dataset,
        args.output_dir / 'train_latents.npy', device, batch_size=16,
    )
    val_latent_path = encode_latent_cache(
        autoencoder, val_dataset,
        args.output_dir / 'val_latents.npy', device, batch_size=16,
    )
    transition, transition_history, transition_checkpoint, mean, std = (
        train_transition(
            args, train_latent_path, val_latent_path,
            train_paths['actions'], val_paths['actions'],
            device, args.output_dir,
        )
    )
    audio_metrics, audio_samples = evaluate_audio(
        autoencoder, transition, val_dataset, mean, std, device,
        args.output_dir / 'audio_samples',
    )
    rollout_metrics = evaluate_rollouts(
        autoencoder, transition, val_dataset, mean, std, device,
    )
    compression_ratio = autoencoder.compression_ratio(
        DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.duration,
    )
    results = {
        'configuration': dict(vars(args)),
        'device': str(device),
        'waveform_values': DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.duration,
        'latent_shape': list(autoencoder.latent_shape(
            DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.duration,
        )),
        'compression_ratio': compression_ratio,
        'autoencoder_best': {
            'epoch': autoencoder_checkpoint['epoch'],
            'validation': autoencoder_checkpoint['validation'],
        },
        'transition_best': {
            'epoch': transition_checkpoint['epoch'],
            'validation': transition_checkpoint['validation'],
        },
        'audio_metrics': audio_metrics,
        'rollout_metrics': rollout_metrics,
        'audio_samples': audio_samples,
        'autoencoder_history': autoencoder_history,
        'transition_history': transition_history,
    }
    results['configuration']['output_dir'] = str(args.output_dir)
    results['configuration']['metadata_dir'] = str(args.metadata_dir)
    result_path = args.output_dir / 'results.json'
    result_path.write_text(json.dumps(results, indent=2) + '\n')
    print('\nPrototype summary:')
    print(f'  Compression: {compression_ratio:.1f}x')
    print(
        '  Reconstruction ceiling: '
        f"{audio_metrics['overall']['target_reconstruction_mean_snr_db']:.2f}dB"
    )
    print(
        '  Direct action prediction: '
        f"{audio_metrics['overall']['mean_snr_db']:.2f}dB "
        f"(identity {audio_metrics['overall']['identity_mean_snr_db']:.2f}dB)"
    )
    print(
        '  Multi-action rollout: '
        f"{rollout_metrics['mean_rollout_snr_db']:.2f}dB"
    )
    print(f'  Results: {result_path}')


if __name__ == '__main__':
    main()
