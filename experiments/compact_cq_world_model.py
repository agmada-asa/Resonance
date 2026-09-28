"""Train a learned delta world model on a phase-aware compact CQ state."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.config import ACTION_VECTOR_SIZE, DEFAULT_CONFIG  # noqa: E402
from resonance.data.cq_nsgt_cache import CachedCQNSGTDataset  # noqa: E402
from resonance.features.cq_nsgt import CQNSGTVectorCodec  # noqa: E402
from resonance.models.compact_cq import (  # noqa: E402
    CompactCQStateCodec,
    CompactCQWorldModel,
)


DEFAULT_DATA_DIR = ROOT / "data/synthetic/cq_nsgt_v001"


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(data_dir):
    return json.loads(
        (data_dir / "vector_cache_v1/manifest.json").read_text()
    )


def build_state_cache(data_dir, output_dir, split, count, time_size):
    state_path = output_dir / f"{split}_states_t{time_size}.npy"
    action_path = output_dir / f"{split}_actions.npy"
    layout_path = output_dir / "cq_layout.json"
    expected_state_shape = (count, 2, 2, 122, time_size)
    if state_path.exists() and action_path.exists() and layout_path.exists():
        states = np.load(state_path, mmap_mode="r")
        actions = np.load(action_path, mmap_mode="r")
        if states.shape == expected_state_shape and actions.shape == (
            count, ACTION_VECTOR_SIZE,
        ):
            print(f"Reusing {split} compact state cache with {count} examples.")
            return state_path, action_path, json.loads(layout_path.read_text())

    manifest = load_manifest(data_dir)
    split_manifest = manifest["splits"][split]
    if count > split_manifest["sample_count"]:
        raise ValueError(f"Requested {count} examples from {split}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(
        state_path, mode="w+", dtype=np.float32, shape=expected_state_shape,
    )
    actions = np.lib.format.open_memmap(
        action_path, mode="w+", dtype=np.float32,
        shape=(count, ACTION_VECTOR_SIZE),
    )
    codec = CompactCQStateCodec(time_size)
    offset = 0
    layout = None
    started = time.perf_counter()
    for descriptor in split_manifest["shards"]:
        if offset >= count:
            break
        payload = torch.load(
            data_dir / "vector_cache_v1" / descriptor["path"],
            map_location="cpu", weights_only=True, mmap=True,
        )
        take = min(descriptor["length"], count - offset)
        input_blocks = [block[:take] for block in payload["input_blocks"]]
        target_blocks = [block[:take] for block in payload["target_blocks"]]
        if layout is None:
            layout = [
                {"frequencies": block.shape[2], "time": block.shape[3]}
                for block in input_blocks
            ]
            frequency_count = sum(item["frequencies"] for item in layout)
            if frequency_count != expected_state_shape[3]:
                raise ValueError(
                    f"Expected 122 CQ bands, found {frequency_count}."
                )
        with torch.inference_mode():
            states[offset:offset + take, 0] = codec.encode(
                input_blocks
            ).numpy()
            states[offset:offset + take, 1] = codec.encode(
                target_blocks
            ).numpy()
        actions[offset:offset + take] = payload["action_vectors"][:take].numpy()
        offset += take
        print(f"  {split} [{offset}/{count}]", flush=True)
    states.flush()
    actions.flush()
    layout_path.write_text(json.dumps(layout, indent=2) + "\n")
    print(
        f"Saved {split} compact states in {time.perf_counter() - started:.1f}s.",
        flush=True,
    )
    return state_path, action_path, layout


class StatePairDataset(Dataset):
    def __init__(self, state_path, action_path, indices=None):
        self.states = np.load(state_path, mmap_mode="r")
        self.actions = np.load(action_path, mmap_mode="r")
        self.indices = (
            np.arange(len(self.states), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        return (
            torch.from_numpy(np.array(self.states[source_index], dtype=np.float32)),
            torch.from_numpy(np.array(self.actions[source_index], dtype=np.float32)),
        )


def select_action_indices(data_dir, split, limit, maximum_action_count=None):
    if maximum_action_count is None:
        return np.arange(limit, dtype=np.int64)
    metadata_path = data_dir / f"metadata_{split}.jsonl"
    indices = []
    with metadata_path.open() as source:
        for index, line in enumerate(source):
            if index >= limit:
                break
            row = json.loads(line)
            if len(row.get("actions", [])) <= maximum_action_count:
                indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def state_statistics(path):
    states = np.load(path, mmap_mode="r")
    sums = np.zeros(2, dtype=np.float64)
    squared = np.zeros(2, dtype=np.float64)
    values_per_channel = 0
    for start in range(0, len(states), 64):
        batch = np.asarray(states[start:start + 64], dtype=np.float32)
        sums += np.sum(batch, axis=(0, 1, 3, 4), dtype=np.float64)
        squared += np.sum(batch ** 2, axis=(0, 1, 3, 4), dtype=np.float64)
        values_per_channel += batch.shape[0] * batch.shape[1] * batch.shape[3] * batch.shape[4]
    mean = sums / values_per_channel
    variance = squared / values_per_channel - mean ** 2
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)


def estimate_frequency_shift(dataset, mean_values, std_values, limit=256):
    pitch_indices = []
    for dataset_index, source_index in enumerate(dataset.indices):
        action = dataset.actions[source_index]
        if action[1] != 0 and action[5] != 0:
            pitch_indices.append(dataset_index)
            if len(pitch_indices) >= limit:
                break
    if not pitch_indices:
        return 0.0, []

    states = torch.stack([dataset[index][0] for index in pitch_indices])
    actions = torch.stack([dataset[index][1] for index in pitch_indices])
    mean = torch.from_numpy(mean_values)[None, :, None, None]
    std = torch.from_numpy(std_values)[None, :, None, None]
    source = (states[:, 0] - mean) / std
    target = (states[:, 1] - mean) / std
    batch, _, frequency_count, time_size = source.shape
    frequency_grid = torch.linspace(-1, 1, frequency_count)[
        None, :, None
    ].expand(batch, -1, time_size)
    time_grid = torch.linspace(-1, 1, time_size)[
        None, None, :
    ].expand(batch, frequency_count, -1)
    parameter = actions[:, 5]
    candidates = np.linspace(-0.3, 0.3, 49)
    losses = []
    with torch.inference_mode():
        for candidate in candidates:
            grid = torch.stack([
                time_grid,
                frequency_grid + float(candidate) * parameter[:, None, None],
            ], dim=-1)
            warped = F.grid_sample(source, grid, align_corners=True)
            losses.append(float(F.smooth_l1_loss(warped, target)))
    best_index = int(np.argmin(losses))
    history = [
        {"shift": float(candidate), "loss": loss}
        for candidate, loss in zip(candidates, losses)
    ]
    return float(candidates[best_index]), history


def run_epoch(model, loader, mean, std, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    examples = 0
    for states, actions in loader:
        source = states[:, 0].to(device)
        target = states[:, 1].to(device)
        actions = actions.to(device)
        source_normalized = (source - mean) / std
        target_normalized = (target - mean) / std
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predicted_normalized = model(source_normalized, actions)
            state_loss = F.smooth_l1_loss(predicted_normalized, target_normalized)
            predicted = predicted_normalized * std + mean
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
            complex_signal = torch.mean(target_complex ** 2).clamp_min(1e-5)
            complex_loss = torch.mean((predicted_complex - target_complex) ** 2)
            complex_loss = complex_loss / complex_signal
            loss = state_loss + 0.25 * complex_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        count = len(states)
        identity_loss = F.smooth_l1_loss(source_normalized, target_normalized)
        totals["loss"] += float(loss.detach()) * count
        totals["state_loss"] += float(state_loss.detach()) * count
        totals["complex_loss"] += float(complex_loss.detach()) * count
        totals["identity_state_loss"] += float(identity_loss) * count
        examples += count
    result = {name: value / examples for name, value in totals.items()}
    result["state_improvement"] = (
        1 - result["state_loss"] / max(result["identity_state_loss"], 1e-12)
    )
    return result


def train(args, train_dataset, val_dataset, mean_values, std_values, device):
    mean = torch.from_numpy(mean_values)[None, :, None, None].to(device)
    std = torch.from_numpy(std_values)[None, :, None, None].to(device)
    initial_frequency_shift = 0.0
    frequency_shift_search = []
    if args.learned_frequency_warp:
        initial_frequency_shift, frequency_shift_search = estimate_frequency_shift(
            train_dataset, mean_values, std_values,
        )
        print(
            f"Estimated initial learned frequency shift from training data: "
            f"{initial_frequency_shift:.4f}.",
            flush=True,
        )
    model = CompactCQWorldModel(
        ACTION_VECTOR_SIZE, width=args.width,
        learned_frequency_warp=args.learned_frequency_warp,
        initial_frequency_shift=initial_frequency_shift,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    checkpoint_path = args.output_dir / "best_compact_world_model.pth"
    history = []
    best = float("inf")
    stale = 0
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, mean, std, device, optimizer,
        )
        with torch.inference_mode():
            validation = run_epoch(model, val_loader, mean, std, device)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        if epoch == 1 or epoch % 5 == 0:
            elapsed = time.perf_counter() - training_started
            projected = elapsed / epoch * args.epochs
            print(
                f"Epoch {epoch:3d}: train={train_metrics['state_loss']:.4f} "
                f"val={validation['state_loss']:.4f} "
                f"vs_identity={validation['state_improvement']:.1%} "
                f"time={record['seconds']:.1f}s "
                f"projected_total={projected / 60:.1f}m",
                flush=True,
            )
        if validation["state_loss"] < best - args.min_delta:
            best = validation["state_loss"]
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "mean": torch.from_numpy(mean_values),
                "std": torch.from_numpy(std_values),
                "width": args.width,
                "learned_frequency_warp": args.learned_frequency_warp,
                "frequency_shift_search": frequency_shift_search,
                "epoch": epoch,
                "validation": validation,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Stopped at epoch {epoch} after validation plateaued.")
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint, history, mean, std


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


def templates_from_layout(layout, device):
    return [
        torch.empty(
            1, 2, item["frequencies"], item["time"], device=device,
        )
        for item in layout
    ]


def original_validation_batches(data_dir, count, indices=None):
    if indices is not None:
        dataset = CachedCQNSGTDataset(
            data_dir / "vector_cache_v1", "val",
        )
        selected = list(np.asarray(indices, dtype=np.int64)[:count])
        for start in range(0, len(selected), 32):
            samples = [dataset[int(index)] for index in selected[start:start + 32]]
            yield (
                [
                    torch.stack([sample["input"][block_index] for sample in samples])
                    for block_index in range(len(samples[0]["input"]))
                ],
                [
                    torch.stack([sample["target"][block_index] for sample in samples])
                    for block_index in range(len(samples[0]["target"]))
                ],
                torch.stack([sample["action_vector"] for sample in samples]),
            )
        return
    manifest = load_manifest(data_dir)
    offset = 0
    for descriptor in manifest["splits"]["val"]["shards"]:
        if offset >= count:
            break
        payload = torch.load(
            data_dir / "vector_cache_v1" / descriptor["path"],
            map_location="cpu", weights_only=True, mmap=True,
        )
        take = min(descriptor["length"], count - offset)
        yield (
            [block[:take] for block in payload["input_blocks"]],
            [block[:take] for block in payload["target_blocks"]],
            payload["action_vectors"][:take],
        )
        offset += take


def evaluate(
    args, model, compact_codec, audio_codec, layout, mean, std, device,
    validation_indices,
):
    templates = templates_from_layout(layout, device)
    values = defaultdict(list)
    sample_offset = 0
    with torch.inference_mode():
        for input_blocks_cpu, target_blocks_cpu, actions_cpu in original_validation_batches(
            args.data_dir, args.eval_count,
            validation_indices if args.maximum_action_count is not None else None,
        ):
            for batch_start in range(0, actions_cpu.shape[0], args.eval_batch_size):
                batch_stop = min(batch_start + args.eval_batch_size, actions_cpu.shape[0])
                input_blocks = [
                    block[batch_start:batch_stop].to(device)
                    for block in input_blocks_cpu
                ]
                target_blocks = [
                    block[batch_start:batch_stop].to(device)
                    for block in target_blocks_cpu
                ]
                actions = actions_cpu[batch_start:batch_stop].to(device)
                source_state = compact_codec.encode(input_blocks)
                target_state = compact_codec.encode(target_blocks)
                predicted_normalized = model((source_state - mean) / std, actions)
                predicted_state = predicted_normalized * std + mean
                predicted_state[:, 0].clamp_(min=0)
                predicted_blocks = compact_codec.decode(predicted_state, templates)
                ceiling_blocks = compact_codec.decode(target_state, templates)
                source_audio = audio_codec.vectors_to_audio(input_blocks)
                target_audio = audio_codec.vectors_to_audio(target_blocks)
                predicted_audio = audio_codec.vectors_to_audio(predicted_blocks)
                ceiling_audio = audio_codec.vectors_to_audio(ceiling_blocks)
                for name, audio in (
                    ("predicted", predicted_audio),
                    ("identity", source_audio),
                    ("ceiling", ceiling_audio),
                ):
                    values[f"{name}_snr_db"].extend(
                        waveform_snr(audio, target_audio).cpu().tolist()
                    )
                    values[f"{name}_magnitude_cosine"].extend(
                        magnitude_cosine(audio, target_audio).cpu().tolist()
                    )
                if sample_offset == 0:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    np.save(
                        args.output_dir / "evaluation_audio.npy",
                        torch.stack([
                            source_audio[:2, 0], target_audio[:2, 0],
                            predicted_audio[:2, 0], ceiling_audio[:2, 0],
                        ], dim=1).cpu().numpy(),
                    )
                sample_offset += batch_stop - batch_start
    return {
        name: {
            "mean": float(np.mean(metric)),
            "median": float(np.median(metric)),
            "p10": float(np.percentile(metric, 10)),
            "p90": float(np.percentile(metric, 90)),
        }
        for name, metric in values.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "build/compact_cq_world_model",
    )
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--val-count", type=int, default=256)
    parser.add_argument("--time-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--maximum-action-count", type=int)
    parser.add_argument("--learned-frequency-warp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_state, train_actions, layout = build_state_cache(
        args.data_dir, args.output_dir, "train", args.train_count, args.time_size,
    )
    val_state, val_actions, _ = build_state_cache(
        args.data_dir, args.output_dir, "val", args.val_count, args.time_size,
    )
    train_indices = select_action_indices(
        args.data_dir, "train", args.train_count, args.maximum_action_count,
    )
    validation_indices = select_action_indices(
        args.data_dir, "val", args.val_count, args.maximum_action_count,
    )
    train_dataset = StatePairDataset(train_state, train_actions, train_indices)
    val_dataset = StatePairDataset(val_state, val_actions, validation_indices)
    mean_values, std_values = state_statistics(train_state)
    device = select_device()
    print(
        f"Training {len(train_dataset)} compact transitions on {device}; "
        f"state compression={181398 / (2 * 122 * args.time_size):.1f}x; "
        f"mean={mean_values.tolist()}; std={std_values.tolist()}.",
        flush=True,
    )
    model, checkpoint, history, mean, std = train(
        args, train_dataset, val_dataset, mean_values, std_values, device,
    )
    compact_codec = CompactCQStateCodec(args.time_size)
    audio_codec = CQNSGTVectorCodec(config=DEFAULT_CONFIG, device=device)
    evaluation = evaluate(
        args, model, compact_codec, audio_codec, layout, mean, std, device,
        validation_indices,
    )
    epoch_seconds = [item["seconds"] for item in history]
    result = {
        "device": str(device),
        "train_count": len(train_dataset),
        "validation_count": len(val_dataset),
        "maximum_action_count": args.maximum_action_count,
        "learned_frequency_warp": args.learned_frequency_warp,
        "time_size": args.time_size,
        "state_shape": [2, 122, args.time_size],
        "state_compression": 181398 / (2 * 122 * args.time_size),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "mean_epoch_seconds": float(np.mean(epoch_seconds)),
        "projected_6000_sample_epoch_seconds": (
            float(np.mean(epoch_seconds)) * 6000 / len(train_dataset)
        ),
        "history": history,
        "evaluation": evaluation,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "state_improvement": result["best_validation"]["state_improvement"],
        "predicted_snr": evaluation["predicted_snr_db"]["median"],
        "identity_snr": evaluation["identity_snr_db"]["median"],
        "ceiling_snr": evaluation["ceiling_snr_db"]["median"],
        "projected_full_epoch_seconds": result[
            "projected_6000_sample_epoch_seconds"
        ],
    }, indent=2))
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
