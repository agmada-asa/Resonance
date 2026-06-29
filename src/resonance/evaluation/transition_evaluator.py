import re

import numpy as np
import torch

from resonance.actions import apply_action
from resonance.config import (
    BINS_PER_OCTAVE,
    DEFAULT_PATHS,
    FMIN,
    HOP_LENGTH,
    N_BINS,
    SAMPLE_RATE,
)
from resonance.data.synthetic import generate_chords, generate_waveform
from resonance.features.spectrogram import audio_to_cqt


TRANSITION_CASES = [
    {
        "name": "sine_440_pitch_up_then_gain_down",
        "waveform_type": "sine",
        "frequency": 440.0,
        "amplitude": 0.5,
        "is_chord": False,
        "actions": [("pitch_change", 2), ("gain", -2)],
    },
    {
        "name": "sine_880_gain_up_then_pitch_down",
        "waveform_type": "sine",
        "frequency": 880.0,
        "amplitude": 0.4,
        "is_chord": False,
        "actions": [("gain", 6), ("pitch_change", -5)],
    },
    {
        "name": "square_220_gain_up_then_low_pass",
        "waveform_type": "square",
        "frequency": 220.0,
        "amplitude": 0.45,
        "is_chord": False,
        "actions": [("gain", 6), ("low_pass", 2500)],
    },
    {
        "name": "square_330_low_pass_then_pitch_up",
        "waveform_type": "square",
        "frequency": 330.0,
        "amplitude": 0.4,
        "is_chord": False,
        "actions": [("low_pass", 3000), ("pitch_change", 5)],
    },
    {
        "name": "sawtooth_330_high_pass_then_gain_up",
        "waveform_type": "sawtooth",
        "frequency": 330.0,
        "amplitude": 0.35,
        "is_chord": False,
        "actions": [("high_pass", 800), ("gain", 4)],
    },
    {
        "name": "sawtooth_180_pitch_down_then_low_pass",
        "waveform_type": "sawtooth",
        "frequency": 180.0,
        "amplitude": 0.5,
        "is_chord": False,
        "actions": [("pitch_change", -7), ("low_pass", 1800)],
    },
    {
        "name": "sine_chord_gain_up_then_pitch_up",
        "waveform_type": "sine",
        "frequency": 261.63,
        "amplitude": 0.6,
        "is_chord": True,
        "actions": [("gain", 5), ("pitch_change", 7)],
    },
    {
        "name": "square_chord_low_pass_then_gain_down",
        "waveform_type": "square",
        "frequency": 196.0,
        "amplitude": 0.45,
        "is_chord": True,
        "actions": [("low_pass", 2500), ("gain", -4)],
    },
    {
        "name": "sawtooth_chord_pitch_up_then_high_pass",
        "waveform_type": "sawtooth",
        "frequency": 220.0,
        "amplitude": 0.4,
        "is_chord": True,
        "actions": [("pitch_change", 5), ("high_pass", 700)],
    },
    {
        "name": "sine_660_pitch_round_trip_with_gain",
        "waveform_type": "sine",
        "frequency": 660.0,
        "amplitude": 0.5,
        "is_chord": False,
        "actions": [("pitch_change", 4), ("gain", -3), ("pitch_change", -4)],
    },
]


def normalize_spectrogram(spectrogram, mean, std, device):
    return (
        torch.tensor(spectrogram, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .sub(mean)
        .div(std)
        .to(device)
    )


def denormalize_spectrogram(normalized_spectrogram, mean, std):
    return normalized_spectrogram.squeeze().detach().cpu().numpy() * std + mean


def make_input_audio(case):
    if case["is_chord"]:
        return generate_chords(
            case["waveform_type"],
            case["frequency"],
            case["amplitude"],
        )

    return generate_waveform(
        case["waveform_type"],
        case["frequency"],
        case["amplitude"],
    )


def audio_to_model_cqt(audio):
    return audio_to_cqt(
        audio,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )


def apply_action_sequence(audio, actions):
    action_vectors = []

    for action, parameter in actions:
        audio, action_vector = apply_action(audio, action=action, parameter=parameter)
        action_vectors.append(action_vector)

    return audio, action_vectors


def action_label(action, parameter):
    if action == "pitch_change":
        return f"pitch {parameter:+g} st"
    if action == "gain":
        return f"gain {parameter:+g} dB"
    if action in {"low_pass", "high_pass"}:
        return f"{action.replace('_', '-')} {parameter:g} Hz"
    return action


def action_sequence_label(actions):
    return " -> ".join(action_label(*action) for action in actions)


def slugify(value):
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def predict_action_sequence(model, input_spectrogram_normalized, action_vectors, device):
    predicted_spectrogram = input_spectrogram_normalized
    predicted_steps = []

    with torch.no_grad():
        for action_vector in action_vectors:
            action_tensor = (
                torch.tensor(action_vector, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            predicted_delta = model(predicted_spectrogram, action_tensor)
            predicted_spectrogram = predicted_spectrogram + predicted_delta
            predicted_steps.append(predicted_spectrogram)

    return predicted_steps


def compose_action_vectors(action_vectors):
    return np.asarray(action_vectors, dtype=np.float32).sum(axis=0)


def predict_composed_action(model, input_spectrogram_normalized, action_vector, device):
    with torch.no_grad():
        action_tensor = (
            torch.tensor(action_vector, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        predicted_delta = model(input_spectrogram_normalized, action_tensor)
        return input_spectrogram_normalized + predicted_delta


class LatentTransitionEvaluator:
    def __init__(
        self,
        model,
        device,
        mean,
        std,
        criterion=None,
        cases=TRANSITION_CASES,
    ):
        self.model = model
        self.device = device
        self.mean = mean
        self.std = std
        self.criterion = criterion or torch.nn.MSELoss()
        self.cases = cases

    def evaluate_chained_transitions(self, output_dir=None):
        output_dir = output_dir or DEFAULT_PATHS.build_dir / "latent_space_transition_samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.evaluate_chained_transition_case(case, output_dir)
            for case in self.cases
        ]

    def evaluate_vector_chaining(self, output_dir=None):
        output_dir = output_dir or DEFAULT_PATHS.build_dir / "latent_space_vector_chaining_samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.evaluate_vector_chaining_case(case, output_dir)
            for case in self.cases
        ]

    def prepare_case(self, case):
        input_audio = make_input_audio(case)
        target_audio, action_vectors = apply_action_sequence(input_audio, case["actions"])

        input_spectrogram = audio_to_model_cqt(input_audio)
        target_final_spectrogram = audio_to_model_cqt(target_audio)
        input_spectrogram_normalized = normalize_spectrogram(
            input_spectrogram,
            self.mean,
            self.std,
            self.device,
        )
        target_final_spectrogram_normalized = normalize_spectrogram(
            target_final_spectrogram,
            self.mean,
            self.std,
            self.device,
        )

        return {
            "input_spectrogram": input_spectrogram,
            "target_final_spectrogram": target_final_spectrogram,
            "input_spectrogram_normalized": input_spectrogram_normalized,
            "target_final_spectrogram_normalized": target_final_spectrogram_normalized,
            "action_vectors": action_vectors,
        }

    def evaluate_chained_transition_case(self, case, output_dir):
        prepared = self.prepare_case(case)
        predicted_steps = predict_action_sequence(
            self.model,
            prepared["input_spectrogram_normalized"],
            prepared["action_vectors"],
            self.device,
        )
        predicted_final_spectrogram = predicted_steps[-1]
        loss = self.criterion(
            predicted_final_spectrogram,
            prepared["target_final_spectrogram_normalized"],
        ).item()

        output_path = output_dir / f"{slugify(case['name'])}.png"
        plot_transition_sample(
            case,
            prepared["input_spectrogram"],
            [denormalize_spectrogram(step, self.mean, self.std) for step in predicted_steps],
            prepared["target_final_spectrogram"],
            loss,
            output_path,
        )

        return {
            "name": case["name"],
            "waveform_type": case["waveform_type"],
            "signal_type": "chord" if case["is_chord"] else "waveform",
            "actions": action_sequence_label(case["actions"]),
            "loss": loss,
            "plot_path": output_path,
        }

    def evaluate_vector_chaining_case(self, case, output_dir):
        prepared = self.prepare_case(case)
        chained_steps = predict_action_sequence(
            self.model,
            prepared["input_spectrogram_normalized"],
            prepared["action_vectors"],
            self.device,
        )
        chained_final_spectrogram = chained_steps[-1]

        composed_action_vector = compose_action_vectors(prepared["action_vectors"])
        composed_final_spectrogram = predict_composed_action(
            self.model,
            prepared["input_spectrogram_normalized"],
            composed_action_vector,
            self.device,
        )

        chained_mse = self.criterion(
            chained_final_spectrogram,
            prepared["target_final_spectrogram_normalized"],
        ).item()
        composed_mse = self.criterion(
            composed_final_spectrogram,
            prepared["target_final_spectrogram_normalized"],
        ).item()
        composed_vs_chained_mse = self.criterion(
            composed_final_spectrogram,
            chained_final_spectrogram,
        ).item()
        losses = {
            "chained_mse": chained_mse,
            "composed_mse": composed_mse,
            "composed_vs_chained_mse": composed_vs_chained_mse,
        }

        output_path = output_dir / f"{slugify(case['name'])}.png"
        plot_vector_chaining_sample(
            case,
            prepared["input_spectrogram"],
            denormalize_spectrogram(chained_final_spectrogram, self.mean, self.std),
            denormalize_spectrogram(composed_final_spectrogram, self.mean, self.std),
            prepared["target_final_spectrogram"],
            losses,
            output_path,
        )

        return {
            "name": case["name"],
            "actions": action_sequence_label(case["actions"]),
            "composed_action_vector": composed_action_vector,
            "plot_path": output_path,
            **losses,
        }


def plot_transition_sample(
    case,
    input_spectrogram,
    predicted_step_spectrograms,
    target_final_spectrogram,
    loss,
    output_path,
):
    import matplotlib.pyplot as plt

    predicted_first_spectrogram = predicted_step_spectrograms[0]
    predicted_final_spectrogram = predicted_step_spectrograms[-1]
    action_sequence = action_sequence_label(case["actions"])

    spectrogram_vmin = min(
        input_spectrogram.min(),
        predicted_first_spectrogram.min(),
        predicted_final_spectrogram.min(),
        target_final_spectrogram.min(),
    )
    spectrogram_vmax = max(
        input_spectrogram.max(),
        predicted_first_spectrogram.max(),
        predicted_final_spectrogram.max(),
        target_final_spectrogram.max(),
    )
    final_error = predicted_final_spectrogram - target_final_spectrogram
    error_abs_max = max(abs(final_error).max(), 1e-6)

    figure = plt.figure(figsize=(13, 8))
    figure.suptitle(
        f"{case['name']} | {action_sequence} | MSE={loss:.4f}",
        fontsize=12,
    )

    plt.subplot(2, 3, 1)
    plt.title("Input CQT")
    plt.imshow(
        input_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 2)
    plt.title(f"Predicted After {action_label(*case['actions'][0])}")
    plt.imshow(
        predicted_first_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 3)
    plt.title("Final Error")
    plt.imshow(
        final_error,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-error_abs_max,
        vmax=error_abs_max,
    )

    plt.subplot(2, 3, 4)
    plt.title("Predicted Final")
    plt.imshow(
        predicted_final_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 5)
    plt.title("True Final")
    plt.imshow(
        target_final_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 6)
    plt.title("|Final Error|")
    plt.imshow(abs(final_error), aspect="auto", origin="lower", cmap="magma")

    plt.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_vector_chaining_sample(
    case,
    input_spectrogram,
    chained_final_spectrogram,
    composed_final_spectrogram,
    target_final_spectrogram,
    losses,
    output_path,
):
    import matplotlib.pyplot as plt

    action_sequence = action_sequence_label(case["actions"])
    spectrogram_vmin = min(
        input_spectrogram.min(),
        chained_final_spectrogram.min(),
        composed_final_spectrogram.min(),
        target_final_spectrogram.min(),
    )
    spectrogram_vmax = max(
        input_spectrogram.max(),
        chained_final_spectrogram.max(),
        composed_final_spectrogram.max(),
        target_final_spectrogram.max(),
    )
    chained_error = chained_final_spectrogram - target_final_spectrogram
    composed_error = composed_final_spectrogram - target_final_spectrogram
    error_abs_max = max(
        abs(chained_error).max(),
        abs(composed_error).max(),
        1e-6,
    )

    figure = plt.figure(figsize=(13, 8))
    figure.suptitle(
        f"{case['name']} | {action_sequence} | "
        f"chained MSE={losses['chained_mse']:.4f} | "
        f"vector-once MSE={losses['composed_mse']:.4f}",
        fontsize=12,
    )

    plt.subplot(2, 3, 1)
    plt.title("Input CQT")
    plt.imshow(
        input_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 2)
    plt.title("Chained Model Inputs")
    plt.imshow(
        chained_final_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 3)
    plt.title("Summed Vector, One Run")
    plt.imshow(
        composed_final_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 4)
    plt.title("True Final")
    plt.imshow(
        target_final_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 5)
    plt.title("Chained Error")
    plt.imshow(
        chained_error,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-error_abs_max,
        vmax=error_abs_max,
    )

    plt.subplot(2, 3, 6)
    plt.title("Vector-Once Error")
    plt.imshow(
        composed_error,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-error_abs_max,
        vmax=error_abs_max,
    )

    plt.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def print_chained_transition_results(results):
    print("\nLatent transition composition results:")
    print("case                                      signal    waveform   loss")
    for result in sorted(results, key=lambda item: item["loss"]):
        print(
            f"{result['name']:<42}"
            f"{result['signal_type']:<10}"
            f"{result['waveform_type']:<10}"
            f"{result['loss']:.6f}"
        )


def print_vector_chaining_results(results):
    print("\nVector chaining comparison:")
    print(
        "case                                      "
        "chained_mse  vector_once_mse  vector/chained  vector_vs_chained"
    )
    for result in sorted(results, key=lambda item: item["composed_mse"]):
        ratio = result["composed_mse"] / max(result["chained_mse"], 1e-12)
        print(
            f"{result['name']:<42}"
            f"{result['chained_mse']:>11.6f}  "
            f"{result['composed_mse']:>15.6f}  "
            f"{ratio:>14.2f}  "
            f"{result['composed_vs_chained_mse']:>17.6f}"
        )

    chained_average = sum(result["chained_mse"] for result in results) / len(results)
    composed_average = sum(result["composed_mse"] for result in results) / len(results)
    print(
        f"\nAverage chained-input MSE: {chained_average:.6f}\n"
        f"Average vector-once MSE:   {composed_average:.6f}\n"
        f"Average ratio:             {composed_average / max(chained_average, 1e-12):.2f}x"
    )
