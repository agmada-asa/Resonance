"""Build compact CQ states for every intermediate synthetic trajectory step."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compact_cq_world_model import build_state_cache  # noqa: E402
from resonance.actions import AudioActionProcessor  # noqa: E402
from resonance.config import ACTION_VECTOR_SIZE, DEFAULT_CONFIG  # noqa: E402
from resonance.data.synthetic import (  # noqa: E402
    DEFAULT_CHORD_INTERVALS,
    WaveformSynthesizer,
)
from resonance.features.cq_nsgt import CQNSGTVectorCodec  # noqa: E402
from resonance.models.compact_cq import CompactCQStateCodec  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "data/synthetic/cq_nsgt_v001"


def load_metadata(data_dir, split):
    with (data_dir / f"metadata_{split}.jsonl").open() as source:
        return [json.loads(line) for line in source]


def trajectory_plan(rows):
    action_counts = np.asarray([
        len(row.get("actions", [])) for row in rows
    ], dtype=np.int64)
    state_lengths = action_counts + 1
    transition_lengths = np.maximum(action_counts, 1)
    state_offsets = np.concatenate([
        np.zeros(1, dtype=np.int64), np.cumsum(state_lengths),
    ])
    transition_offsets = np.concatenate([
        np.zeros(1, dtype=np.int64), np.cumsum(transition_lengths),
    ])
    return state_offsets, transition_offsets


def create_arrays(output_dir, rows, time_size):
    state_offsets, transition_offsets = trajectory_plan(rows)
    state_count = int(state_offsets[-1])
    transition_count = int(transition_offsets[-1])
    state_shape = (state_count, 2, 122, time_size)
    paths = {
        "states": output_dir / "states.npy",
        "written": output_dir / "states_written.npy",
        "actions": output_dir / "transition_actions.npy",
        "source_indices": output_dir / "transition_source_indices.npy",
        "target_indices": output_dir / "transition_target_indices.npy",
        "trajectory_indices": output_dir / "transition_trajectory_indices.npy",
        "step_indices": output_dir / "transition_step_indices.npy",
        "state_offsets": output_dir / "trajectory_state_offsets.npy",
        "transition_offsets": output_dir / "trajectory_transition_offsets.npy",
    }
    if all(path.exists() for path in paths.values()):
        states = np.load(paths["states"], mmap_mode="r+")
        written = np.load(paths["written"], mmap_mode="r+")
        if states.shape == state_shape and written.shape == (state_count,):
            return paths, states, written, state_offsets, transition_offsets

    output_dir.mkdir(parents=True, exist_ok=True)
    states = np.lib.format.open_memmap(
        paths["states"], mode="w+", dtype=np.float32, shape=state_shape,
    )
    written = np.lib.format.open_memmap(
        paths["written"], mode="w+", dtype=np.bool_, shape=(state_count,),
    )
    written[:] = False
    actions = np.lib.format.open_memmap(
        paths["actions"], mode="w+", dtype=np.float32,
        shape=(transition_count, ACTION_VECTOR_SIZE),
    )
    source_indices = np.lib.format.open_memmap(
        paths["source_indices"], mode="w+", dtype=np.int64,
        shape=(transition_count,),
    )
    target_indices = np.lib.format.open_memmap(
        paths["target_indices"], mode="w+", dtype=np.int64,
        shape=(transition_count,),
    )
    trajectory_indices = np.lib.format.open_memmap(
        paths["trajectory_indices"], mode="w+", dtype=np.int64,
        shape=(transition_count,),
    )
    step_indices = np.lib.format.open_memmap(
        paths["step_indices"], mode="w+", dtype=np.int16,
        shape=(transition_count,),
    )
    processor = AudioActionProcessor(DEFAULT_CONFIG)
    for trajectory_index, row in enumerate(rows):
        state_start = int(state_offsets[trajectory_index])
        transition_start = int(transition_offsets[trajectory_index])
        row_actions = row.get("actions", [])
        if not row_actions:
            actions[transition_start] = 0
            source_indices[transition_start] = state_start
            target_indices[transition_start] = state_start
            trajectory_indices[transition_start] = trajectory_index
            step_indices[transition_start] = 0
            continue
        for step, action in enumerate(row_actions):
            transition_index = transition_start + step
            encoded = np.zeros(ACTION_VECTOR_SIZE, dtype=np.float32)
            encoded[:7] = processor.encode_action(
                action["action"], action["parameter"],
            )
            actions[transition_index] = encoded
            source_indices[transition_index] = state_start + step
            target_indices[transition_index] = state_start + step + 1
            trajectory_indices[transition_index] = trajectory_index
            step_indices[transition_index] = step
    np.save(paths["state_offsets"], state_offsets)
    np.save(paths["transition_offsets"], transition_offsets)
    for array in (
        states, written, actions, source_indices, target_indices,
        trajectory_indices, step_indices,
    ):
        array.flush()
    return paths, states, written, state_offsets, transition_offsets


def source_audio(row, synthesizer):
    if row.get("is_chord", False):
        return synthesizer.generate_chords(
            row["waveform_type"], row["frequency"], row["amplitude"],
            intervals=DEFAULT_CHORD_INTERVALS,
        )
    return synthesizer.generate_waveform(
        row["waveform_type"], row["frequency"], row["amplitude"],
    )


def build_split(
    data_dir, endpoint_dir, output_root, split, time_size, trajectory_batch,
    cq_batch,
):
    rows = load_metadata(data_dir, split)
    endpoint_path, _, layout = build_state_cache(
        data_dir, endpoint_dir, split, len(rows), time_size,
    )
    endpoints = np.load(endpoint_path, mmap_mode="r")
    output_dir = output_root / split
    paths, states, written, state_offsets, transition_offsets = create_arrays(
        output_dir, rows, time_size,
    )
    progress_path = output_dir / "progress.json"
    next_trajectory = 0
    if progress_path.exists():
        next_trajectory = int(json.loads(progress_path.read_text()).get(
            "next_trajectory", 0,
        ))

    synthesizer = WaveformSynthesizer(DEFAULT_CONFIG)
    processor = AudioActionProcessor(DEFAULT_CONFIG)
    audio_codec = CQNSGTVectorCodec(config=DEFAULT_CONFIG, device="cpu")
    compact_codec = CompactCQStateCodec(time_size)
    started = time.perf_counter()
    for chunk_start in range(next_trajectory, len(rows), trajectory_batch):
        chunk_stop = min(chunk_start + trajectory_batch, len(rows))
        intermediate_indices = []
        intermediate_audio = []
        for trajectory_index in range(chunk_start, chunk_stop):
            row = rows[trajectory_index]
            state_start = int(state_offsets[trajectory_index])
            row_actions = row.get("actions", [])
            states[state_start] = endpoints[trajectory_index, 0]
            written[state_start] = True
            if not row_actions:
                continue
            audio = source_audio(row, synthesizer)
            for step, action in enumerate(row_actions):
                audio, _ = processor.apply_action(
                    audio, action["action"], action["parameter"],
                )
                target_index = state_start + step + 1
                if step == len(row_actions) - 1:
                    states[target_index] = endpoints[trajectory_index, 1]
                    written[target_index] = True
                else:
                    intermediate_indices.append(target_index)
                    intermediate_audio.append(np.asarray(audio, dtype=np.float32))
        for batch_start in range(0, len(intermediate_audio), cq_batch):
            batch_stop = min(batch_start + cq_batch, len(intermediate_audio))
            audio_batch = np.stack(intermediate_audio[batch_start:batch_stop])
            with torch.inference_mode():
                compact = compact_codec.encode(
                    audio_codec.audio_to_vectors(audio_batch)
                ).cpu().numpy()
            indices = intermediate_indices[batch_start:batch_stop]
            states[indices] = compact
            written[indices] = True
        states.flush()
        written.flush()
        progress = {
            "next_trajectory": chunk_stop,
            "trajectory_count": len(rows),
            "state_count": int(state_offsets[-1]),
            "transition_count": int(transition_offsets[-1]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(progress_path)
        print(
            f"{split} trajectories [{chunk_stop}/{len(rows)}], "
            f"states={int(written.sum())}/{len(written)}, "
            f"elapsed={progress['elapsed_seconds']:.1f}s",
            flush=True,
        )

    missing = np.flatnonzero(~np.asarray(written))
    if len(missing):
        raise RuntimeError(f"{split} cache is missing {len(missing)} states.")
    for start in range(0, len(states), 64):
        if not np.isfinite(np.asarray(states[start:start + 64])).all():
            raise RuntimeError(f"{split} cache contains non-finite states.")
    manifest = {
        "split": split,
        "trajectory_count": len(rows),
        "state_count": int(state_offsets[-1]),
        "transition_count": int(transition_offsets[-1]),
        "time_size": time_size,
        "state_shape": [2, 122, time_size],
        "state_dtype": "float32",
        "layout": layout,
        "complete": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Completed {split}: {manifest}", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--endpoint-dir", type=Path,
        default=ROOT / "build/compact_cq_world_single_step",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "build/compact_cq_trajectories",
    )
    parser.add_argument("--time-size", type=int, default=96)
    parser.add_argument("--trajectory-batch", type=int, default=128)
    parser.add_argument("--cq-batch", type=int, default=32)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
    )
    args = parser.parse_args()
    for split in args.splits:
        build_split(
            args.data_dir, args.endpoint_dir, args.output_dir, split,
            args.time_size, args.trajectory_batch, args.cq_batch,
        )


if __name__ == "__main__":
    main()
