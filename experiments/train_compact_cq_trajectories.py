"""Train and audit the compact CQ world model on complete trajectories."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.actions import AudioActionProcessor  # noqa: E402
from resonance.config import ACTION_VECTOR_SIZE, DEFAULT_CONFIG  # noqa: E402
from resonance.data.synthetic import (  # noqa: E402
    DEFAULT_CHORD_INTERVALS,
    WaveformSynthesizer,
)
from resonance.features.cq_nsgt import CQNSGTVectorCodec  # noqa: E402
from resonance.models.compact_cq import (  # noqa: E402
    CompactCQStateCodec,
    CompactCQWorldModel,
    constrain_normalized_compact_state,
)


ACTION_NAMES = ("gain", "pitch_change", "low_pass", "high_pass")
DEFAULT_DATA_DIR = ROOT / "data/synthetic/cq_nsgt_v001"
DEFAULT_CACHE_DIR = ROOT / "build/compact_cq_trajectories"


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_metadata(data_dir, split):
    with (data_dir / f"metadata_{split}.jsonl").open() as source:
        return [json.loads(line) for line in source]


class TransitionDataset(Dataset):
    def __init__(self, cache_dir):
        self.states = np.load(cache_dir / "states.npy", mmap_mode="r")
        self.actions = np.load(
            cache_dir / "transition_actions.npy", mmap_mode="r",
        )
        self.source_indices = np.load(
            cache_dir / "transition_source_indices.npy", mmap_mode="r",
        )
        self.target_indices = np.load(
            cache_dir / "transition_target_indices.npy", mmap_mode="r",
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return (
            torch.from_numpy(np.array(
                self.states[self.source_indices[index]], dtype=np.float32,
            )),
            torch.from_numpy(np.array(
                self.states[self.target_indices[index]], dtype=np.float32,
            )),
            torch.from_numpy(np.array(self.actions[index], dtype=np.float32)),
        )


class TrajectoryDataset(Dataset):
    def __init__(self, cache_dir):
        self.states = np.load(cache_dir / "states.npy", mmap_mode="r")
        self.actions = np.load(
            cache_dir / "transition_actions.npy", mmap_mode="r",
        )
        self.source_indices = np.load(
            cache_dir / "transition_source_indices.npy", mmap_mode="r",
        )
        self.target_indices = np.load(
            cache_dir / "transition_target_indices.npy", mmap_mode="r",
        )
        self.offsets = np.load(
            cache_dir / "trajectory_transition_offsets.npy", mmap_mode="r",
        )

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return (
            torch.from_numpy(np.array(
                self.states[self.source_indices[start]], dtype=np.float32,
            )),
            torch.from_numpy(np.array(
                self.states[self.target_indices[start:stop]], dtype=np.float32,
            )),
            torch.from_numpy(np.array(
                self.actions[start:stop], dtype=np.float32,
            )),
        )


def collate_trajectories(samples):
    maximum_steps = max(len(sample[1]) for sample in samples)
    state_shape = samples[0][0].shape
    initial = torch.stack([sample[0] for sample in samples])
    targets = torch.zeros(len(samples), maximum_steps, *state_shape)
    actions = torch.zeros(len(samples), maximum_steps, ACTION_VECTOR_SIZE)
    mask = torch.zeros(len(samples), maximum_steps, dtype=torch.bool)
    for index, (_, sample_targets, sample_actions) in enumerate(samples):
        steps = len(sample_targets)
        targets[index, :steps] = sample_targets
        actions[index, :steps] = sample_actions
        mask[index, :steps] = True
    return initial, targets, actions, mask


def state_statistics(state_path):
    states = np.load(state_path, mmap_mode="r")
    sums = np.zeros(2, dtype=np.float64)
    squared = np.zeros(2, dtype=np.float64)
    values = 0
    for start in range(0, len(states), 64):
        batch = np.asarray(states[start:start + 64], dtype=np.float32)
        sums += np.sum(batch, axis=(0, 2, 3), dtype=np.float64)
        squared += np.sum(batch ** 2, axis=(0, 2, 3), dtype=np.float64)
        values += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = sums / values
    variance = squared / values - mean ** 2
    return mean.astype(np.float32), np.sqrt(
        np.maximum(variance, 1e-6)
    ).astype(np.float32)


def estimate_frequency_shift(dataset, mean_values, std_values, limit=512):
    pitch_indices = np.flatnonzero(np.asarray(dataset.actions)[:, 1] != 0)[:limit]
    if not len(pitch_indices):
        return 0.0, []
    source = torch.from_numpy(np.array(
        dataset.states[dataset.source_indices[pitch_indices]], dtype=np.float32,
    ))
    target = torch.from_numpy(np.array(
        dataset.states[dataset.target_indices[pitch_indices]], dtype=np.float32,
    ))
    actions = torch.from_numpy(np.array(
        dataset.actions[pitch_indices], dtype=np.float32,
    ))
    mean = torch.from_numpy(mean_values)[None, :, None, None]
    std = torch.from_numpy(std_values)[None, :, None, None]
    source = (source - mean) / std
    target = (target - mean) / std
    parameter = actions[:, 5]
    candidates = np.linspace(-0.3, 0.3, 49)
    losses = []
    with torch.inference_mode():
        for candidate in candidates:
            warped = CompactCQWorldModel._warp_frequency(
                source, float(candidate) * parameter[:, None],
            )
            losses.append(float(F.smooth_l1_loss(warped, target)))
    best = int(np.argmin(losses))
    return float(candidates[best]), [
        {"shift": float(candidate), "loss": loss}
        for candidate, loss in zip(candidates, losses)
    ]


def complex_relative_loss(predicted, target, reduction="mean"):
    predicted_phase = CompactCQStateCodec.phase_cycles(predicted)
    target_phase = CompactCQStateCodec.phase_cycles(target)
    predicted_complex = torch.stack([
        predicted[:, 0] * torch.cos(2 * torch.pi * predicted_phase),
        predicted[:, 0] * torch.sin(2 * torch.pi * predicted_phase),
    ], dim=1)
    target_complex = torch.stack([
        target[:, 0] * torch.cos(2 * torch.pi * target_phase),
        target[:, 0] * torch.sin(2 * torch.pi * target_phase),
    ], dim=1)
    error = torch.mean((predicted_complex - target_complex) ** 2, dim=(1, 2, 3))
    signal = torch.mean(target_complex ** 2, dim=(1, 2, 3)).clamp_min(1e-5)
    # Quiet filtered targets can make a raw relative error arbitrarily large.
    # Log compression keeps those examples useful without allowing them to
    # overwhelm state learning or destabilize recurrent rollouts.
    result = torch.log1p(error / signal)
    return result.mean() if reduction == "mean" else result


def spectral_state_losses(predicted_normalized, target_normalized, mean, std):
    predicted = predicted_normalized * std + mean
    target = target_normalized * std + mean
    state = F.smooth_l1_loss(
        predicted_normalized, target_normalized, reduction="none",
    ).mean(dim=(1, 2, 3))
    predicted_magnitude = predicted[:, 0].flatten(1).clamp_min(1e-6)
    target_magnitude = target[:, 0].flatten(1).clamp_min(1e-6)
    magnitude_cosine = 1 - F.cosine_similarity(
        predicted_magnitude, target_magnitude, dim=1,
    )
    predicted_level = predicted_magnitude.mean(dim=1)
    target_level = target_magnitude.mean(dim=1)
    magnitude_level = torch.abs(torch.log(
        (predicted_level + 1e-4) / (target_level + 1e-4)
    )).clamp_max(10)
    complex_loss = complex_relative_loss(predicted, target, reduction="none")
    total = (
        state
        + 0.5 * magnitude_cosine
        + 0.2 * magnitude_level
        + 0.1 * complex_loss
    )
    return total, {
        "state_loss": state,
        "magnitude_cosine_loss": magnitude_cosine,
        "magnitude_level_loss": magnitude_level,
        "complex_loss": complex_loss,
    }


def transition_loss(model, source, target, actions, mean, std):
    source_normalized = (source - mean) / std
    target_normalized = (target - mean) / std
    predicted_normalized = constrain_normalized_compact_state(
        model(source_normalized, actions), mean, std,
    )
    total_per_example, loss_terms = spectral_state_losses(
        predicted_normalized, target_normalized, mean, std,
    )
    identity = F.smooth_l1_loss(source_normalized, target_normalized)
    return total_per_example.mean(), {
        **{
            name: value.mean().detach()
            for name, value in loss_terms.items()
        },
        "identity_state_loss": identity.detach(),
    }


def rollout_loss(model, initial, targets, actions, mask, mean, std):
    predicted = (initial - mean) / std
    total = predicted.new_zeros(())
    terms = predicted.new_zeros(())
    for step in range(targets.shape[1]):
        active = mask[:, step]
        candidate = constrain_normalized_compact_state(
            model(predicted, actions[:, step]), mean, std,
        )
        target = targets[:, step]
        target_normalized = (target - mean) / std
        loss_per_example, _ = spectral_state_losses(
            candidate, target_normalized, mean, std,
        )
        total = total + torch.sum(
            loss_per_example * active
        )
        terms = terms + active.sum()
        predicted = torch.where(
            active[:, None, None, None], candidate, predicted,
        )
    return total / terms.clamp_min(1)


def run_training_epoch(
    model, transition_loader, trajectory_loader, mean, std, optimizer,
    device, rollout_every, rollout_weight,
):
    model.train()
    totals = defaultdict(float)
    examples = 0
    trajectory_iterator = iter(trajectory_loader)
    for batch_index, (source, target, actions) in enumerate(transition_loader):
        source = source.to(device)
        target = target.to(device)
        actions = actions.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = transition_loss(
            model, source, target, actions, mean, std,
        )
        rollout_value = loss.new_zeros(())
        if rollout_every and batch_index % rollout_every == 0:
            try:
                rollout_batch = next(trajectory_iterator)
            except StopIteration:
                trajectory_iterator = iter(trajectory_loader)
                rollout_batch = next(trajectory_iterator)
            initial, targets, rollout_actions, mask = (
                value.to(device) for value in rollout_batch
            )
            rollout_value = rollout_loss(
                model, initial, targets, rollout_actions, mask, mean, std,
            )
            loss = loss + rollout_weight * rollout_value
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        count = len(source)
        totals["loss"] += float(loss.detach()) * count
        totals["rollout_loss"] += float(rollout_value.detach()) * count
        for name, value in metrics.items():
            totals[name] += float(value) * count
        examples += count
    result = {name: value / examples for name, value in totals.items()}
    result["state_improvement"] = 1 - result["state_loss"] / max(
        result["identity_state_loss"], 1e-12,
    )
    return result


def evaluate_state(model, transition_loader, trajectory_loader, mean, std, device):
    model.eval()
    totals = defaultdict(float)
    examples = 0
    with torch.inference_mode():
        for source, target, actions in transition_loader:
            source = source.to(device)
            target = target.to(device)
            actions = actions.to(device)
            loss, metrics = transition_loss(
                model, source, target, actions, mean, std,
            )
            count = len(source)
            totals["loss"] += float(loss) * count
            for name, value in metrics.items():
                totals[name] += float(value) * count
            examples += count
        rollout_total = 0.0
        trajectory_count = 0
        for initial, targets, actions, mask in trajectory_loader:
            initial, targets, actions, mask = (
                value.to(device) for value in (initial, targets, actions, mask)
            )
            value = rollout_loss(
                model, initial, targets, actions, mask, mean, std,
            )
            rollout_total += float(value) * len(initial)
            trajectory_count += len(initial)
    result = {name: value / examples for name, value in totals.items()}
    result["state_improvement"] = 1 - result["state_loss"] / max(
        result["identity_state_loss"], 1e-12,
    )
    result["rollout_loss"] = rollout_total / trajectory_count
    return result


def load_matching_weights(model, checkpoint_path):
    if checkpoint_path is None or not checkpoint_path.exists():
        return []
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    current = model.state_dict()
    matched = {
        name: value for name, value in checkpoint["model"].items()
        if name in current and current[name].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    if model.action_specific_heads:
        shared_weight = checkpoint["model"].get("output.weight")
        shared_bias = checkpoint["model"].get("output.bias")
        if shared_weight is not None and shared_bias is not None:
            with torch.no_grad():
                for output_head in model.output_heads:
                    output_head.weight.copy_(shared_weight)
                    output_head.bias.copy_(shared_bias)
            matched["output.* -> output_heads.*"] = shared_weight
    return sorted(matched)


def train(args, train_transitions, train_trajectories, val_transitions, val_trajectories):
    device = select_device()
    mean_values, std_values = state_statistics(
        args.cache_dir / "train/states.npy"
    )
    mean = torch.from_numpy(mean_values)[None, :, None, None].to(device)
    std = torch.from_numpy(std_values)[None, :, None, None].to(device)
    initial_shift, shift_search = estimate_frequency_shift(
        train_transitions, mean_values, std_values,
    )
    model = CompactCQWorldModel(
        ACTION_VECTOR_SIZE,
        width=args.width,
        learned_frequency_warp=True,
        initial_frequency_shift=initial_shift,
        action_specific_heads=True,
    )
    warm_started = load_matching_weights(model, args.warm_start)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5,
    )
    train_transition_loader = DataLoader(
        train_transitions, batch_size=args.batch_size, shuffle=True,
    )
    train_trajectory_loader = DataLoader(
        train_trajectories, batch_size=args.rollout_batch_size, shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_transition_loader = DataLoader(
        val_transitions, batch_size=args.batch_size,
    )
    val_trajectory_loader = DataLoader(
        val_trajectories, batch_size=args.rollout_batch_size,
        collate_fn=collate_trajectories,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_compact_world_model.pth"
    history = []
    baseline = evaluate_state(
        model, val_transition_loader, val_trajectory_loader,
        mean, std, device,
    )
    best = baseline["loss"] + 0.25 * baseline["rollout_loss"]
    stale = 0
    torch.save({
        "model": model.state_dict(),
        "mean": torch.from_numpy(mean_values),
        "std": torch.from_numpy(std_values),
        "width": args.width,
        "action_specific_heads": True,
        "learned_frequency_warp": True,
        "initial_frequency_shift": initial_shift,
        "frequency_shift_search": shift_search,
        "warm_started_keys": warm_started,
        "epoch": 0,
        "validation": baseline,
        "selection_loss": best,
    }, checkpoint_path)
    print(
        f"Warm start: val={baseline['state_loss']:.4f}, "
        f"one_step={baseline['state_improvement']:.1%}, "
        f"rollout={baseline['rollout_loss']:.4f}.",
        flush=True,
    )
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head_only = epoch <= args.head_only_epochs
        for parameter in model.parameters():
            parameter.requires_grad_(not head_only)
        for output_head in model.output_heads:
            for parameter in output_head.parameters():
                parameter.requires_grad_(True)
        if epoch == args.head_only_epochs + 1:
            stale = 0
        epoch_started = time.perf_counter()
        training = run_training_epoch(
            model, train_transition_loader, train_trajectory_loader,
            mean, std, optimizer, device,
            0 if head_only else args.rollout_every,
            args.rollout_weight,
        )
        validation = evaluate_state(
            model, val_transition_loader, val_trajectory_loader,
            mean, std, device,
        )
        record = {
            "epoch": epoch,
            "head_only": head_only,
            "train": training,
            "validation": validation,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        projected = (time.perf_counter() - started) / epoch * args.epochs
        print(
            f"Epoch {epoch:3d} ({'heads' if head_only else 'full'}): "
            f"train={training['state_loss']:.4f}, "
            f"val={validation['state_loss']:.4f}, "
            f"one_step={validation['state_improvement']:.1%}, "
            f"rollout={validation['rollout_loss']:.4f}, "
            f"time={record['seconds']:.1f}s, projected={projected / 60:.1f}m",
            flush=True,
        )
        selection_loss = validation["loss"] + 0.25 * validation["rollout_loss"]
        if selection_loss < best - args.min_delta:
            best = selection_loss
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "mean": torch.from_numpy(mean_values),
                "std": torch.from_numpy(std_values),
                "width": args.width,
                "action_specific_heads": True,
                "learned_frequency_warp": True,
                "initial_frequency_shift": initial_shift,
                "frequency_shift_search": shift_search,
                "warm_started_keys": warm_started,
                "epoch": epoch,
                "validation": validation,
                "selection_loss": selection_loss,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Stopped at epoch {epoch} after validation plateaued.")
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint, history, mean, std, device


def waveform_snr(predicted, target):
    signal = torch.sum(target ** 2, dim=(-2, -1))
    error = torch.sum((predicted - target) ** 2, dim=(-2, -1))
    return 10 * torch.log10(signal.clamp_min(1e-12) / error.clamp_min(1e-12))


def magnitude_cosine(predicted, target):
    window = torch.hann_window(1024, device=target.device)
    predicted_stft = torch.stft(
        predicted[:, 0], 1024, hop_length=256, window=window,
        return_complex=True,
    ).abs().flatten(1)
    target_stft = torch.stft(
        target[:, 0], 1024, hop_length=256, window=window,
        return_complex=True,
    ).abs().flatten(1)
    return F.cosine_similarity(predicted_stft, target_stft, dim=1)


def generate_audio(row, synthesizer, processor):
    if row.get("is_chord", False):
        source = synthesizer.generate_chords(
            row["waveform_type"], row["frequency"], row["amplitude"],
            intervals=DEFAULT_CHORD_INTERVALS,
        )
    else:
        source = synthesizer.generate_waveform(
            row["waveform_type"], row["frequency"], row["amplitude"],
        )
    target = source
    for action in row.get("actions", []):
        target, _ = processor.apply_action(
            target, action["action"], action["parameter"],
        )
    return np.asarray(source, dtype=np.float32), np.asarray(target, dtype=np.float32)


def predict_final_states(model, dataset, mean, std, device, batch_size):
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_trajectories,
    )
    predictions = []
    targets_final = []
    with torch.inference_mode():
        for initial, targets, actions, mask in loader:
            initial, targets, actions, mask = (
                value.to(device) for value in (initial, targets, actions, mask)
            )
            predicted = (initial - mean) / std
            for step in range(targets.shape[1]):
                candidate = constrain_normalized_compact_state(
                    model(predicted, actions[:, step]), mean, std,
                )
                predicted = torch.where(
                    mask[:, step, None, None, None], candidate, predicted,
                )
            final_indices = mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(initial), device=device)
            predictions.append((predicted * std + mean).cpu())
            targets_final.append(targets[batch_indices, final_indices].cpu())
    return torch.cat(predictions), torch.cat(targets_final)


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def add_grouped_metrics(result, group, indices, metrics):
    if not indices:
        return
    result[group] = {
        "count": len(indices),
        **{
            name: metric_summary(np.asarray(values)[indices])
            for name, values in metrics.items()
        },
    }


def audit_audio(
    args, split, model, dataset, mean, std, device, layout,
):
    rows = load_metadata(args.data_dir, split)[:args.audio_eval_count]
    predictions, compact_targets = predict_final_states(
        model, dataset, mean, std, device, args.rollout_batch_size,
    )
    predictions = predictions[:len(rows)]
    compact_targets = compact_targets[:len(rows)]
    synthesizer = WaveformSynthesizer(DEFAULT_CONFIG)
    processor = AudioActionProcessor(DEFAULT_CONFIG)
    sources = []
    targets = []
    for row in rows:
        source, target = generate_audio(row, synthesizer, processor)
        sources.append(source)
        targets.append(target)
    sources = torch.from_numpy(np.stack(sources))[:, None]
    targets = torch.from_numpy(np.stack(targets))[:, None]

    compact_codec = CompactCQStateCodec(predictions.shape[-1])
    audio_codec = CQNSGTVectorCodec(config=DEFAULT_CONFIG, device="cpu")
    templates = [
        torch.empty(1, 2, item["frequencies"], item["time"])
        for item in layout
    ]
    predicted_audio = []
    ceiling_audio = []
    for start in range(0, len(predictions), args.decode_batch_size):
        predicted_batch = predictions[start:start + args.decode_batch_size].clone()
        predicted_batch[:, 0].clamp_(min=0)
        target_batch = compact_targets[start:start + args.decode_batch_size]
        with torch.inference_mode():
            predicted_audio.append(audio_codec.vectors_to_audio(
                compact_codec.decode(predicted_batch, templates)
            ).cpu())
            ceiling_audio.append(audio_codec.vectors_to_audio(
                compact_codec.decode(target_batch, templates)
            ).cpu())
    predicted_audio = torch.cat(predicted_audio)
    ceiling_audio = torch.cat(ceiling_audio)
    metrics = {}
    for name, audio in (
        ("predicted", predicted_audio),
        ("identity", sources),
        ("ceiling", ceiling_audio),
    ):
        metrics[f"{name}_snr_db"] = waveform_snr(audio, targets).tolist()
        metrics[f"{name}_magnitude_cosine"] = magnitude_cosine(
            audio, targets,
        ).tolist()
    metrics["predicted_rms_ratio"] = (
        predicted_audio.square().mean(dim=(-2, -1)).sqrt()
        / targets.square().mean(dim=(-2, -1)).sqrt().clamp_min(1e-8)
    ).tolist()

    grouped = {}
    add_grouped_metrics(grouped, "all", list(range(len(rows))), metrics)
    for length in range(5):
        indices = [
            index for index, row in enumerate(rows)
            if len(row.get("actions", [])) == length
        ]
        add_grouped_metrics(grouped, f"steps_{length}", indices, metrics)
    for action_name in ACTION_NAMES:
        indices = [
            index for index, row in enumerate(rows)
            if any(
                action["action"] == action_name
                for action in row.get("actions", [])
            )
        ]
        add_grouped_metrics(grouped, f"contains_{action_name}", indices, metrics)

    sample_root = args.output_dir / f"{split}_audio_samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    cosine = np.asarray(metrics["predicted_magnitude_cosine"])
    snr = np.asarray(metrics["predicted_snr_db"])
    rms = np.log10(np.asarray(metrics["predicted_rms_ratio"]) + 1e-8)
    chosen = []
    for length in range(1, 5):
        candidates = np.asarray([
            index for index, row in enumerate(rows)
            if len(row.get("actions", [])) == length
        ])
        if not len(candidates):
            continue
        score = np.zeros(len(candidates))
        for values in (cosine, snr, rms):
            selected = values[candidates]
            scale = np.percentile(selected, 75) - np.percentile(selected, 25)
            score += np.abs(selected - np.median(selected)) / max(scale, 1e-6)
        chosen.append(int(candidates[np.argmin(score)]))
    for index in chosen:
        destination = sample_root / f"trajectory_{index:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        for name, audio in (
            ("source", sources[index, 0]),
            ("target", targets[index, 0]),
            ("predicted", predicted_audio[index, 0]),
            ("compact_target_ceiling", ceiling_audio[index, 0]),
        ):
            wavfile.write(
                destination / f"{name}.wav",
                DEFAULT_CONFIG.sample_rate,
                audio.numpy().astype(np.float32),
            )
        (destination / "metadata.json").write_text(json.dumps({
            "dataset_index": index,
            "actions": rows[index].get("actions", []),
            "predicted_magnitude_cosine": float(cosine[index]),
            "predicted_snr_db": float(metrics["predicted_snr_db"][index]),
        }, indent=2) + "\n")
    return grouped


def per_action_state_audit(model, dataset, mean, std, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size)
    values = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for source, target, actions in loader:
            source = source.to(device)
            target = target.to(device)
            actions = actions.to(device)
            source_normalized = (source - mean) / std
            target_normalized = (target - mean) / std
            predicted = constrain_normalized_compact_state(
                model(source_normalized, actions), mean, std,
            )
            predicted_error = F.smooth_l1_loss(
                predicted, target_normalized, reduction="none",
            ).mean(dim=(1, 2, 3))
            identity_error = F.smooth_l1_loss(
                source_normalized, target_normalized, reduction="none",
            ).mean(dim=(1, 2, 3))
            action_type = actions[:, :4].argmax(dim=1)
            active = actions[:, :4].sum(dim=1) > 0
            for index in range(len(source)):
                name = ACTION_NAMES[int(action_type[index])] if active[index] else "no_action"
                values[name]["predicted_state_loss"].append(float(predicted_error[index]))
                values[name]["identity_state_loss"].append(float(identity_error[index]))
    result = {}
    for name, metrics in values.items():
        predicted = np.asarray(metrics["predicted_state_loss"])
        identity = np.asarray(metrics["identity_state_loss"])
        result[name] = {
            "count": len(predicted),
            "predicted_state_loss": metric_summary(predicted),
            "identity_state_loss": metric_summary(identity),
            "aggregate_improvement": float(
                1 - predicted.mean() / max(identity.mean(), 1e-12)
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "build/compact_cq_world_complete",
    )
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--rollout-every", type=int, default=12)
    parser.add_argument("--rollout-weight", type=float, default=0.35)
    parser.add_argument("--head-only-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--audio-eval-count", type=int, default=750)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warm-start", type=Path,
        default=ROOT / "build/compact_cq_world_single_step_warp_estimated/best_compact_world_model.pth",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for split in ("train", "val", "test"):
        manifest = json.loads(
            (args.cache_dir / split / "manifest.json").read_text()
        )
        if not manifest.get("complete"):
            raise RuntimeError(f"Incomplete trajectory cache for {split}.")
    train_transitions = TransitionDataset(args.cache_dir / "train")
    train_trajectories = TrajectoryDataset(args.cache_dir / "train")
    val_transitions = TransitionDataset(args.cache_dir / "val")
    val_trajectories = TrajectoryDataset(args.cache_dir / "val")
    test_transitions = TransitionDataset(args.cache_dir / "test")
    test_trajectories = TrajectoryDataset(args.cache_dir / "test")
    print(
        f"Training on {len(train_transitions)} transitions from "
        f"{len(train_trajectories)} trajectories.",
        flush=True,
    )
    model, checkpoint, history, mean, std, device = train(
        args, train_transitions, train_trajectories,
        val_transitions, val_trajectories,
    )
    test_state = evaluate_state(
        model,
        DataLoader(test_transitions, batch_size=args.batch_size),
        DataLoader(
            test_trajectories, batch_size=args.rollout_batch_size,
            collate_fn=collate_trajectories,
        ),
        mean, std, device,
    )
    per_action = per_action_state_audit(
        model, test_transitions, mean, std, device, args.batch_size,
    )
    layout = json.loads(
        (ROOT / "build/compact_cq_world_single_step/cq_layout.json").read_text()
    )
    validation_audio = audit_audio(
        args, "val", model, val_trajectories, mean, std, device, layout,
    )
    test_audio = audit_audio(
        args, "test", model, test_trajectories, mean, std, device, layout,
    )
    results = {
        "model": "learned_compact_cq_action_world_model",
        "inference_uses_deterministic_effects": False,
        "device": str(device),
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_shape": [2, 122, 96],
        "state_compression": 181398 / (2 * 122 * 96),
        "train_trajectories": len(train_trajectories),
        "train_transitions": len(train_transitions),
        "validation_trajectories": len(val_trajectories),
        "validation_transitions": len(val_transitions),
        "test_trajectories": len(test_trajectories),
        "test_transitions": len(test_transitions),
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "test_state": test_state,
        "test_state_by_action": per_action,
        "validation_audio": validation_audio,
        "test_audio": test_audio,
        "learned_frequency_shift": float(torch.tanh(
            model.frequency_shift_scale.detach().cpu()
        )),
        "history": history,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({
        "best_epoch": results["best_epoch"],
        "test_one_step_improvement": test_state["state_improvement"],
        "test_rollout_loss": test_state["rollout_loss"],
        "test_rollout_snr_median": test_audio["all"]["predicted_snr_db"]["median"],
        "identity_snr_median": test_audio["all"]["identity_snr_db"]["median"],
        "test_rollout_cosine_median": test_audio["all"]["predicted_magnitude_cosine"]["median"],
        "identity_cosine_median": test_audio["all"]["identity_magnitude_cosine"]["median"],
    }, indent=2), flush=True)
    print(f"Saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
