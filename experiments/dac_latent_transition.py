"""Train an action-conditioned transition over frozen DAC projected latents."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / ".prototype-packages"
SRC = ROOT / "src"
for path in (PACKAGE_DIR, SRC):
    if path.is_dir():
        sys.path.insert(0, str(path))

import dac  # noqa: E402
from resonance.config import ACTION_VECTOR_SIZE  # noqa: E402
from resonance.models.latent_audio import LatentActionTransitionModel  # noqa: E402

from dac_latent_probe import select_device, waveform_metrics  # noqa: E402


class DacLatentDataset(Dataset):
    def __init__(self, states_path, actions_path):
        states = np.load(states_path, mmap_mode="r")
        self.source = states["source_latents"]
        self.target = states["target_latents"]
        self.target_codes = states["target_codes"]
        self.actions = np.load(actions_path, mmap_mode="r")
        if len(self.source) != len(self.actions):
            raise ValueError("Latent states and actions have different lengths.")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return (
            torch.from_numpy(np.array(self.source[index], dtype=np.float32)),
            torch.from_numpy(np.array(self.target[index], dtype=np.float32)),
            torch.from_numpy(np.array(self.target_codes[index], dtype=np.int64)),
            torch.from_numpy(np.array(self.actions[index], dtype=np.float32)),
        )


def encode_states(model, audio_path, output_path, batch_size, device):
    audio = np.load(audio_path, mmap_mode="r")
    source_latents = []
    target_latents = []
    source_codes = []
    target_codes = []
    print(f"Encoding {len(audio)} waveform pairs into frozen DAC states...", flush=True)
    started = time.perf_counter()
    for start in range(0, len(audio), batch_size):
        batch = torch.from_numpy(
            np.array(audio[start:start + batch_size], dtype=np.float32)
        ).to(device)
        pair = torch.cat([batch[:, 0:1], batch[:, 1:2]], dim=0)
        with torch.inference_mode():
            pair = model.preprocess(pair, 44100)
            _, codes, latents, *_ = model.encode(pair)
        count = batch.shape[0]
        source_latents.append(latents[:count].cpu().numpy().astype(np.float16))
        target_latents.append(latents[count:].cpu().numpy().astype(np.float16))
        source_codes.append(codes[:count].cpu().numpy().astype(np.int16))
        target_codes.append(codes[count:].cpu().numpy().astype(np.int16))
        if start == 0 or start + count == len(audio) or (start + count) % 32 == 0:
            print(f"  [{start + count}/{len(audio)}]", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_latents=np.concatenate(source_latents),
        target_latents=np.concatenate(target_latents),
        source_codes=np.concatenate(source_codes),
        target_codes=np.concatenate(target_codes),
    )
    elapsed = time.perf_counter() - started
    print(f"Saved {output_path} in {elapsed:.1f}s", flush=True)
    return elapsed


def latent_statistics(path):
    with np.load(path) as states:
        both = np.concatenate([
            states["source_latents"].astype(np.float32),
            states["target_latents"].astype(np.float32),
        ])
    mean = np.mean(both, axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.std(both, axis=(0, 2), dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-4)


def run_epoch(model, loader, mean, std, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    target_loss = identity_loss = 0.0
    examples = 0
    for source, target, _, actions in loader:
        source = (source.to(device) - mean) / std
        target = (target.to(device) - mean) / std
        actions = actions.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predicted = model(source, actions)
            loss = F.mse_loss(predicted, target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        count = len(source)
        target_loss += float(loss.detach()) * count
        identity_loss += float(F.mse_loss(source, target)) * count
        examples += count
    target_loss /= examples
    identity_loss /= examples
    return {
        "target_loss": target_loss,
        "identity_loss": identity_loss,
        "improvement": 1 - target_loss / max(identity_loss, 1e-12),
    }


def train(args, train_dataset, val_dataset, mean, std, device, output_dir):
    latent_channels = train_dataset.source.shape[1]
    model = LatentActionTransitionModel(
        ACTION_VECTOR_SIZE, latent_channels=latent_channels, width=args.width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.transition_batch_size, shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.transition_batch_size)
    mean_tensor = torch.from_numpy(mean)[None, :, None].to(device)
    std_tensor = torch.from_numpy(std)[None, :, None].to(device)
    checkpoint_path = output_dir / "best_transition.pth"
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, mean_tensor, std_tensor, device, optimizer,
        )
        with torch.inference_mode():
            validation = run_epoch(
                model, val_loader, mean_tensor, std_tensor, device,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d}: train {train_metrics['target_loss']:.4f}; "
                f"val {validation['target_loss']:.4f}; "
                f"vs identity {validation['improvement']:.1%}; "
                f"{record['seconds']:.1f}s",
                flush=True,
            )
        if validation["target_loss"] < best - args.min_delta:
            best = validation["target_loss"]
            stale = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "validation": validation,
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
                "width": args.width,
                "latent_channels": latent_channels,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Stopped at epoch {epoch} after validation plateaued.")
                break
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint, history, mean_tensor, std_tensor


def mean_metric(storage):
    return {
        name: float(np.mean(values))
        for name, values in storage.items()
    }


def evaluate(transition, codec, dataset, mean, std, device, output_dir):
    loader = DataLoader(dataset, batch_size=2)
    metric_values = defaultdict(list)
    code_matches = []
    latent_losses = []
    identity_latent_losses = []
    audio = np.load(
        ROOT / "build/latent_delta_dense_pilot/waveform_cache/val_audio.npy",
        mmap_mode="r",
    )
    offset = 0
    with torch.inference_mode():
        for source, target, target_codes, actions in loader:
            source = source.to(device)
            target = target.to(device)
            target_codes = target_codes.to(device)
            actions = actions.to(device)
            predicted_normalized = transition((source - mean) / std, actions)
            predicted = predicted_normalized * std + mean
            quantized, _, predicted_codes = codec.quantizer.from_latents(predicted)
            predicted_audio = codec.decode(quantized)[..., :audio.shape[-1]]
            target_quantized, _, _ = codec.quantizer.from_latents(target)
            ceiling_audio = codec.decode(target_quantized)[..., :audio.shape[-1]]
            source_quantized, _, _ = codec.quantizer.from_latents(source)
            identity_audio = codec.decode(source_quantized)[..., :audio.shape[-1]]
            count = len(source)
            original_target = torch.from_numpy(
                np.array(audio[offset:offset + count, 1:2], dtype=np.float32)
            ).to(device)
            for prefix, candidate, reference in (
                ("predicted_vs_original", predicted_audio, original_target),
                ("identity_vs_original", identity_audio, original_target),
                ("ceiling_vs_original", ceiling_audio, original_target),
                ("predicted_vs_ceiling", predicted_audio, ceiling_audio),
            ):
                for name, values in waveform_metrics(candidate, reference).items():
                    metric_values[f"{prefix}_{name}"].extend(
                        values.cpu().tolist()
                    )
            code_matches.extend(
                torch.mean((predicted_codes == target_codes).float(), dim=(1, 2))
                .cpu().tolist()
            )
            latent_losses.extend(
                torch.mean((predicted - target) ** 2, dim=(1, 2)).cpu().tolist()
            )
            identity_latent_losses.extend(
                torch.mean((source - target) ** 2, dim=(1, 2)).cpu().tolist()
            )
            if offset == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / "example_predicted.npy", predicted_audio[:2].cpu().numpy())
                np.save(output_dir / "example_target.npy", original_target[:2].cpu().numpy())
                np.save(output_dir / "example_ceiling.npy", ceiling_audio[:2].cpu().numpy())
            offset += count
    latent_loss = float(np.mean(latent_losses))
    identity_loss = float(np.mean(identity_latent_losses))
    return {
        "metrics": mean_metric(metric_values),
        "code_accuracy": float(np.mean(code_matches)),
        "latent_mse": latent_loss,
        "identity_latent_mse": identity_loss,
        "latent_improvement": 1 - latent_loss / max(identity_loss, 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitrate", default="8kbps", choices=("8kbps", "16kbps"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/dac_latent_transition")
    parser.add_argument("--audio-cache-dir", type=Path, default=ROOT / "build/latent_delta_dense_pilot/waveform_cache")
    parser.add_argument("--codec-batch-size", type=int, default=2)
    parser.add_argument("--transition-batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    codec = dac.utils.load_model(
        model_type="44khz", model_bitrate=args.bitrate,
    ).eval().to(device)
    encode_seconds = {}
    state_paths = {}
    for split in ("train", "val"):
        state_path = args.output_dir / f"{split}_states.npz"
        state_paths[split] = state_path
        if not state_path.exists():
            encode_seconds[split] = encode_states(
                codec, args.audio_cache_dir / f"{split}_audio.npy",
                state_path, args.codec_batch_size, device,
            )

    train_dataset = DacLatentDataset(
        state_paths["train"], args.audio_cache_dir / "train_actions.npy",
    )
    val_dataset = DacLatentDataset(
        state_paths["val"], args.audio_cache_dir / "val_actions.npy",
    )
    mean, std = latent_statistics(state_paths["train"])
    transition, checkpoint, history, mean_tensor, std_tensor = train(
        args, train_dataset, val_dataset, mean, std, device, args.output_dir,
    )
    evaluation = evaluate(
        transition, codec, val_dataset, mean_tensor, std_tensor,
        device, args.output_dir,
    )
    result = {
        "bitrate": args.bitrate,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "state_shape": list(train_dataset.source.shape[1:]),
        "continuous_float16_compression": (
            88200 * 2 / np.prod(train_dataset.source.shape[1:])
        ),
        "encoding_seconds": encode_seconds,
        "best_epoch": checkpoint["epoch"],
        "best_validation": checkpoint["validation"],
        "evaluation": evaluation,
        "history": history,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "latent_improvement": evaluation["latent_improvement"],
        "code_accuracy": evaluation["code_accuracy"],
        "predicted_magnitude_cosine": evaluation["metrics"][
            "predicted_vs_original_magnitude_cosine"
        ],
        "identity_magnitude_cosine": evaluation["metrics"][
            "identity_vs_original_magnitude_cosine"
        ],
        "ceiling_magnitude_cosine": evaluation["metrics"][
            "ceiling_vs_original_magnitude_cosine"
        ],
    }, indent=2))
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
