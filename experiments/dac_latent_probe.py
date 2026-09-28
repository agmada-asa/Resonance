"""Measure whether pretrained DAC states are useful for action deltas.

This is a gate for the learned-representation prototype.  It measures codec
reconstruction quality, the size of its continuous and discrete states, and
how sparse the discrete source-to-target changes are.
"""

import argparse
from collections import defaultdict
import json
import lzma
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / ".prototype-packages"
if PACKAGE_DIR.is_dir():
    sys.path.insert(0, str(PACKAGE_DIR))

import dac  # noqa: E402


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def spectral_metrics(predicted, target):
    window = torch.hann_window(1024, device=target.device)
    predicted_stft = torch.stft(
        predicted[:, 0], 1024, hop_length=256, window=window,
        return_complex=True,
    )
    target_stft = torch.stft(
        target[:, 0], 1024, hop_length=256, window=window,
        return_complex=True,
    )
    predicted_magnitude = torch.abs(predicted_stft)
    target_magnitude = torch.abs(target_stft)
    difference = predicted_magnitude - target_magnitude
    dimensions = (-2, -1)
    spectral_convergence = torch.sqrt(torch.sum(difference ** 2, dimensions))
    spectral_convergence /= torch.sqrt(
        torch.sum(target_magnitude ** 2, dimensions)
    ).clamp_min(1e-8)
    magnitude_cosine = torch.nn.functional.cosine_similarity(
        predicted_magnitude.flatten(1), target_magnitude.flatten(1), dim=1,
    )
    log_magnitude_mae = torch.mean(
        torch.abs(
            torch.log1p(predicted_magnitude)
            - torch.log1p(target_magnitude)
        ),
        dim=dimensions,
    )
    return {
        "spectral_convergence": spectral_convergence,
        "magnitude_cosine": magnitude_cosine,
        "log_magnitude_mae": log_magnitude_mae,
    }


def waveform_metrics(predicted, target):
    error = torch.mean((predicted - target) ** 2, dim=(-2, -1))
    power = torch.mean(target ** 2, dim=(-2, -1))
    snr = 10 * torch.log10(power / error.clamp_min(1e-12))

    target_flat = target.flatten(1)
    predicted_flat = predicted.flatten(1)
    scale = torch.sum(predicted_flat * target_flat, dim=1)
    scale /= torch.sum(target_flat ** 2, dim=1).clamp_min(1e-12)
    scaled_target = scale[:, None] * target_flat
    residual = predicted_flat - scaled_target
    si_sdr = 10 * torch.log10(
        torch.sum(scaled_target ** 2, dim=1)
        / torch.sum(residual ** 2, dim=1).clamp_min(1e-12)
    )
    return {"snr_db": snr, "si_sdr_db": si_sdr, **spectral_metrics(predicted, target)}


def append_metrics(storage, prefix, values):
    for name, value in values.items():
        storage[f"{prefix}_{name}"].extend(value.detach().cpu().tolist())


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def probe_model(bitrate, audio, batch_size, device, output_dir):
    model = dac.utils.load_model(
        model_type="44khz", model_bitrate=bitrate,
    ).eval().to(device)
    metrics = defaultdict(list)
    all_codes = [[], []]
    all_latents = [[], []]
    elapsed = 0.0

    for start in range(0, len(audio), batch_size):
        batch = torch.from_numpy(
            np.array(audio[start:start + batch_size], dtype=np.float32)
        ).to(device)
        source = batch[:, 0:1]
        target = batch[:, 1:2]
        pair = torch.cat([source, target], dim=0)
        with torch.inference_mode():
            padded = model.preprocess(pair, 44100)
            begun = time.perf_counter()
            z, codes, latents, *_ = model.encode(padded)
            reconstructed = model.decode(z)[..., :pair.shape[-1]]
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed += time.perf_counter() - begun

        count = source.shape[0]
        reconstructed_source = reconstructed[:count]
        reconstructed_target = reconstructed[count:]
        append_metrics(metrics, "source_codec", waveform_metrics(
            reconstructed_source, source,
        ))
        append_metrics(metrics, "target_codec", waveform_metrics(
            reconstructed_target, target,
        ))
        append_metrics(metrics, "action_identity", waveform_metrics(source, target))
        all_codes[0].append(codes[:count].cpu().numpy().astype(np.int16))
        all_codes[1].append(codes[count:].cpu().numpy().astype(np.int16))
        all_latents[0].append(latents[:count].cpu().numpy().astype(np.float16))
        all_latents[1].append(latents[count:].cpu().numpy().astype(np.float16))

        if start == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / f"{bitrate}_example_original.npy", pair[:2].cpu().numpy())
            np.save(
                output_dir / f"{bitrate}_example_reconstructed.npy",
                reconstructed[:2].cpu().numpy(),
            )

    source_codes = np.concatenate(all_codes[0])
    target_codes = np.concatenate(all_codes[1])
    source_latents = np.concatenate(all_latents[0])
    target_latents = np.concatenate(all_latents[1])
    code_delta = target_codes - source_codes
    changed = code_delta != 0
    packed_target = lzma.compress(target_codes.astype("<u2").tobytes())
    packed_delta = lzma.compress(code_delta.astype("<i2").tobytes())
    latent_delta = target_latents.astype(np.float32) - source_latents.astype(np.float32)

    sample_count, codebook_count, frame_count = source_codes.shape
    waveform_values = audio.shape[-1]
    projected_channels = source_latents.shape[1]
    report = {
        "bitrate": bitrate,
        "sample_count": sample_count,
        "encode_decode_seconds": elapsed,
        "state": {
            "frames": frame_count,
            "codebooks": codebook_count,
            "projected_latent_channels": projected_channels,
            "continuous_float32_compression": waveform_values / (
                projected_channels * frame_count
            ),
            "continuous_float16_compression": waveform_values * 2 / (
                projected_channels * frame_count
            ),
            "discrete_code_count_compression": waveform_values / (
                codebook_count * frame_count
            ),
            "estimated_code_bit_compression_vs_float32": (
                waveform_values * 32 / (codebook_count * frame_count * 10)
            ),
        },
        "delta": {
            "changed_code_fraction": float(np.mean(changed)),
            "unchanged_code_fraction": float(1 - np.mean(changed)),
            "changed_frames_any_codebook_fraction": float(
                np.mean(np.any(changed, axis=1))
            ),
            "lzma_full_target_bytes": len(packed_target),
            "lzma_delta_bytes": len(packed_delta),
            "lzma_delta_saving_fraction": float(
                1 - len(packed_delta) / len(packed_target)
            ),
            "continuous_delta_relative_rms": float(
                np.sqrt(np.mean(latent_delta ** 2))
                / max(np.sqrt(np.mean(target_latents.astype(np.float32) ** 2)), 1e-12)
            ),
        },
        "metrics": {name: summarize(values) for name, values in metrics.items()},
    }
    np.savez_compressed(
        output_dir / f"{bitrate}_states.npz",
        source_codes=source_codes,
        target_codes=target_codes,
        source_latents=source_latents,
        target_latents=target_latents,
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-cache", type=Path,
        default=ROOT / "build/latent_delta_dense_pilot/waveform_cache/val_audio.npy",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/dac_latent_probe")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bitrates", nargs="+", default=["8kbps", "16kbps"])
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
    audio = np.load(args.audio_cache, mmap_mode="r")[:args.count]
    device = select_device()
    print(f"Probing {len(audio)} pairs on {device}.")
    results = {
        "device": str(device),
        "audio_cache": str(args.audio_cache),
        "models": {},
    }
    for bitrate in args.bitrates:
        print(f"Probing DAC 44.1 kHz {bitrate}...", flush=True)
        results["models"][bitrate] = probe_model(
            bitrate, audio, args.batch_size, device, args.output_dir,
        )
        model_report = results["models"][bitrate]
        print(
            f"  target spectral convergence: "
            f"{model_report['metrics']['target_codec_spectral_convergence']['mean']:.3f}; "
            f"changed codes: {model_report['delta']['changed_code_fraction']:.1%}; "
            f"latent compression: "
            f"{model_report['state']['continuous_float32_compression']:.1f}x",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
