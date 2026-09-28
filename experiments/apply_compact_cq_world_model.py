"""Apply a trained compact CQ world model to a two-second audio clip."""

import argparse
import json
from pathlib import Path
import sys
import time

import librosa
import numpy as np
from scipy.io import wavfile
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.actions import AudioActionProcessor  # noqa: E402
from resonance.config import (  # noqa: E402
    ACTION_DIMENSIONS,
    ACTION_VECTOR_SIZE,
    DEFAULT_CONFIG,
    MAX_ACTIONS,
)
from resonance.features.cq_nsgt import CQNSGTVectorCodec  # noqa: E402
from resonance.models.compact_cq import (  # noqa: E402
    CompactCQStateCodec,
    CompactCQWorldModel,
    constrain_normalized_compact_state,
)


DEFAULT_CHECKPOINT = (
    ROOT / "build/compact_cq_world_complete_energy/best_compact_world_model.pth"
)


def parse_action(value):
    try:
        name, raw_parameter = value.split("=", 1)
        parameter = float(raw_parameter)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Actions must use NAME=VALUE, for example pitch_change=4."
        ) from error
    if name not in {"gain", "pitch_change", "low_pass", "high_pass"}:
        raise argparse.ArgumentTypeError(f"Unsupported action: {name}.")
    return name, parameter


def load_audio(path):
    audio, _ = librosa.load(
        path, sr=DEFAULT_CONFIG.sample_rate, mono=True,
    )
    expected = int(DEFAULT_CONFIG.sample_rate * DEFAULT_CONFIG.duration)
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    return np.asarray(audio[:expected], dtype=np.float32)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True,
    )
    model = CompactCQWorldModel(
        ACTION_VECTOR_SIZE,
        width=int(checkpoint["width"]),
        learned_frequency_warp=bool(checkpoint["learned_frequency_warp"]),
        action_specific_heads=bool(checkpoint["action_specific_heads"]),
    )
    model.load_state_dict(checkpoint["model"])
    return (
        model.to(device).eval(),
        checkpoint["mean"][None, :, None, None].to(device),
        checkpoint["std"][None, :, None, None].to(device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--action", type=parse_action, action="append", required=True,
        help="Repeat NAME=VALUE up to four times in rollout order.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if len(args.action) > MAX_ACTIONS:
        parser.error(f"At most {MAX_ACTIONS} actions are supported.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    # CPU is faster for this transform at the current two-second batch size.
    device = torch.device("cpu")
    model, mean, std = load_model(args.checkpoint, device)
    action_processor = AudioActionProcessor(DEFAULT_CONFIG)
    audio_codec = CQNSGTVectorCodec(config=DEFAULT_CONFIG, device=device)
    compact_codec = CompactCQStateCodec(time_size=96)
    audio = load_audio(args.input)
    started = time.perf_counter()
    with torch.inference_mode():
        input_blocks = audio_codec.audio_to_vectors(audio)
        state = compact_codec.encode(input_blocks)
        normalized = (state - mean) / std
        for action_name, parameter in args.action:
            action_vector = torch.zeros(1, ACTION_VECTOR_SIZE, device=device)
            action_vector[0, :ACTION_DIMENSIONS] = torch.tensor(
                action_processor.encode_action(action_name, parameter),
                dtype=torch.float32,
                device=device,
            )
            normalized = constrain_normalized_compact_state(
                model(normalized, action_vector), mean, std,
            )
        predicted_state = normalized * std + mean
        predicted_blocks = compact_codec.decode(predicted_state, input_blocks)
        predicted_audio = audio_codec.vectors_to_audio(
            predicted_blocks
        )[0, 0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        args.output, DEFAULT_CONFIG.sample_rate,
        predicted_audio.astype(np.float32),
    )
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "actions": [
            {"action": name, "parameter": parameter}
            for name, parameter in args.action
        ],
        "inference_uses_deterministic_effects": False,
        "seconds": time.perf_counter() - started,
    }, indent=2) + "\n")
    print(f"Saved {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
