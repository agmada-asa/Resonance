"""Test dense sample deltas against compact action-program deltas."""

import argparse
from collections import Counter, defaultdict
import json
import lzma
from pathlib import Path
import struct
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.actions import AudioActionProcessor  # noqa: E402
from resonance.config import DEFAULT_CONFIG  # noqa: E402
from resonance.data.synthetic import WaveformSynthesizer  # noqa: E402


ACTION_IDS = {
    "gain": 0,
    "pitch_change": 1,
    "low_pass": 2,
    "high_pass": 3,
}
ID_ACTIONS = {value: key for key, value in ACTION_IDS.items()}


def encode_program(actions):
    return b"".join(
        struct.pack("<Bd", ACTION_IDS[action["action"]], action["parameter"])
        for action in actions
    )


def decode_program(payload):
    record_size = struct.calcsize("<Bd")
    if len(payload) % record_size:
        raise ValueError("Action program has a partial record.")
    return [
        {
            "action": ID_ACTIONS[action_id],
            "parameter": parameter,
        }
        for action_id, parameter in struct.iter_unpack("<Bd", payload)
    ]


def generate_source(synthesizer, metadata):
    generate = (
        synthesizer.generate_chords
        if metadata.get("is_chord", False)
        else synthesizer.generate_waveform
    )
    return generate(
        metadata["waveform_type"], metadata["frequency"], metadata["amplitude"],
    )


def apply_program(processor, source, program):
    states = [source]
    current = source
    for action in decode_program(program):
        current, _ = processor.apply_action(
            current, action["action"], action["parameter"],
        )
        states.append(current)
    return states


def snr_db(predicted, target):
    target = target.astype(np.float64)
    predicted = predicted.astype(np.float64)
    signal = np.sum(target ** 2)
    error = np.sum((predicted - target) ** 2)
    if error == 0:
        return float("inf")
    return float(10 * np.log10(max(signal, 1e-24) / error))


def probe_split(cache_dir, split, limit=None):
    audio = np.load(cache_dir / f"{split}_audio.npy", mmap_mode="r")
    metadata = json.loads((cache_dir / f"{split}_metadata.json").read_text())
    if limit is not None:
        audio = audio[:limit]
        metadata = metadata[:limit]

    synthesizer = WaveformSynthesizer(DEFAULT_CONFIG)
    processor = AudioActionProcessor(DEFAULT_CONFIG)
    action_counts = Counter()
    sequence_lengths = Counter()
    errors = defaultdict(list)
    action_program_bytes = 0
    full_sequence_bytes = 0
    initial_plus_program_bytes = 0
    metadata_plus_program_bytes = 0
    all_targets = []
    all_numeric_deltas = []
    elapsed_by_action = defaultdict(float)

    started = time.perf_counter()
    for index, row in enumerate(metadata):
        source = generate_source(synthesizer, row)
        program = encode_program(row.get("actions", []))
        action_program_bytes += len(program)
        sequence_lengths[len(row.get("actions", []))] += 1
        for action in row.get("actions", []):
            action_counts[action["action"]] += 1

        action_started = time.perf_counter()
        states = apply_program(processor, source, program)
        action_elapsed = time.perf_counter() - action_started
        action_key = "+".join(action["action"] for action in row.get("actions", []))
        elapsed_by_action[action_key or "no_action"] += action_elapsed

        cached_source = np.array(audio[index, 0], dtype=np.float32)
        cached_target = np.array(audio[index, 1], dtype=np.float32)
        target = states[-1]
        errors["source_max_abs"].append(float(np.max(np.abs(source - cached_source))))
        errors["target_max_abs"].append(float(np.max(np.abs(target - cached_target))))
        errors["source_snr_db"].append(snr_db(source, cached_source))
        errors["target_snr_db"].append(snr_db(target, cached_target))

        sample_bytes = source.size * np.dtype(np.float32).itemsize
        full_sequence_bytes += sample_bytes * len(states)
        initial_plus_program_bytes += sample_bytes + len(program)
        # Synthetic source metadata: waveform id, chord flag, frequency, amplitude.
        metadata_plus_program_bytes += struct.calcsize("<BBdd") + len(program)
        all_targets.append(cached_target.astype(np.float16))
        all_numeric_deltas.append((cached_target - cached_source).astype(np.float16))

    target_bytes = np.stack(all_targets).astype("<f2").tobytes()
    numeric_delta_bytes = np.stack(all_numeric_deltas).astype("<f2").tobytes()
    compressed_target_size = len(lzma.compress(target_bytes, preset=3))
    compressed_numeric_delta_size = len(lzma.compress(numeric_delta_bytes, preset=3))
    elapsed = time.perf_counter() - started
    action_count = sum(action_counts.values())
    sample_count = len(metadata)

    finite_target_snr = [value for value in errors["target_snr_db"] if np.isfinite(value)]
    report = {
        "sample_count": sample_count,
        "action_count": action_count,
        "mean_actions_per_sample": action_count / sample_count,
        "action_counts": dict(action_counts),
        "sequence_lengths": {str(key): value for key, value in sequence_lengths.items()},
        "roundtrip": {
            "maximum_source_abs_error_vs_float16_cache": max(errors["source_max_abs"]),
            "maximum_target_abs_error_vs_float16_cache": max(errors["target_max_abs"]),
            "median_target_snr_db_vs_float16_cache": float(np.median(finite_target_snr)),
            "minimum_target_snr_db_vs_float16_cache": float(np.min(finite_target_snr)),
            "program_decode_is_exact": all(
                decode_program(encode_program(row.get("actions", [])))
                == row.get("actions", [])
                for row in metadata
            ),
        },
        "storage": {
            "dense_full_state_sequences_bytes": full_sequence_bytes,
            "initial_float32_plus_program_bytes": initial_plus_program_bytes,
            "synthetic_metadata_plus_program_bytes": metadata_plus_program_bytes,
            "action_program_bytes": action_program_bytes,
            "initial_plus_program_saving_fraction": (
                1 - initial_plus_program_bytes / full_sequence_bytes
            ),
            "synthetic_metadata_plus_program_saving_fraction": (
                1 - metadata_plus_program_bytes / full_sequence_bytes
            ),
            "bytes_per_action_record": struct.calcsize("<Bd"),
            "float32_waveform_bytes_per_state": (
                DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.duration * 4
            ),
            "per_followup_state_compression": (
                DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.duration * 4
                / struct.calcsize("<Bd")
            ),
            "lzma_target_float16_bytes": compressed_target_size,
            "lzma_dense_numeric_delta_float16_bytes": compressed_numeric_delta_size,
            "lzma_numeric_delta_saving_fraction": (
                1 - compressed_numeric_delta_size / compressed_target_size
            ),
        },
        "runtime": {
            "total_seconds_including_compression": elapsed,
            "action_application_seconds": float(sum(elapsed_by_action.values())),
            "milliseconds_per_action": (
                1000 * sum(elapsed_by_action.values()) / max(action_count, 1)
            ),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "build/latent_delta_dense_pilot/waveform_cache",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "build/action_program_delta_probe/results.json",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    args = parser.parse_args()

    results = {
        "train": probe_split(args.cache_dir, "train", args.train_limit),
        "validation": probe_split(args.cache_dir, "val", args.val_limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for split, report in results.items():
        print(
            f"{split}: {report['sample_count']} samples, "
            f"{report['roundtrip']['median_target_snr_db_vs_float16_cache']:.1f}dB "
            f"round-trip SNR, "
            f"{report['storage']['initial_plus_program_saving_fraction']:.1%} "
            f"sequence storage saved, "
            f"{report['runtime']['milliseconds_per_action']:.1f}ms/action",
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
