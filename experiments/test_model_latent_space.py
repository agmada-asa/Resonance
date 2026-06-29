import re

import matplotlib.pyplot as plt
import torch

from resonance.actions import apply_action
from resonance.config import BINS_PER_OCTAVE, DEFAULT_PATHS, FMIN, HOP_LENGTH, N_BINS, SAMPLE_RATE
from resonance.data.synthetic import generate_chords, generate_waveform
from resonance.evaluation.evaluator import SpectrogramEvaluator
from resonance.features.spectrogram import audio_to_cqt

# Test cases for latent space transition evaluation
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

# Generate input audio based on the case definition, either a single waveform or a chord
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


# Apply a sequence of actions to the input audio and return the final audio and the corresponding action vectors
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


def slugify(value):
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def predict_action_sequence(model, input_spectrogram_normalized, action_vectors, device):
    predicted_spectrogram = input_spectrogram_normalized
    predicted_steps = []

    with torch.no_grad():
        for action_vector in action_vectors:
            action_tensor = torch.tensor(action_vector, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_delta = model(predicted_spectrogram, action_tensor)
            predicted_spectrogram = predicted_spectrogram + predicted_delta
            predicted_steps.append(predicted_spectrogram)

    return predicted_steps


# Plot the input, predicted, and target spectrograms for a given transition case
def plot_transition_sample(
    case,
    input_spectrogram,
    predicted_step_spectrograms,
    target_final_spectrogram,
    loss,
    output_path,
):
    predicted_first_spectrogram = predicted_step_spectrograms[0]
    predicted_final_spectrogram = predicted_step_spectrograms[-1]
    action_sequence = " -> ".join(action_label(*action) for action in case["actions"])

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
    plt.imshow(input_spectrogram, aspect="auto", origin="lower", vmin=spectrogram_vmin, vmax=spectrogram_vmax)

    plt.subplot(2, 3, 2)
    plt.title(f"Predicted After {action_label(*case['actions'][0])}")
    plt.imshow(predicted_first_spectrogram, aspect="auto", origin="lower", vmin=spectrogram_vmin, vmax=spectrogram_vmax)

    plt.subplot(2, 3, 3)
    plt.title("Final Error")
    plt.imshow(final_error, aspect="auto", origin="lower", cmap="coolwarm", vmin=-error_abs_max, vmax=error_abs_max)

    plt.subplot(2, 3, 4)
    plt.title("Predicted Final")
    plt.imshow(predicted_final_spectrogram, aspect="auto", origin="lower", vmin=spectrogram_vmin, vmax=spectrogram_vmax)

    plt.subplot(2, 3, 5)
    plt.title("True Final")
    plt.imshow(target_final_spectrogram, aspect="auto", origin="lower", vmin=spectrogram_vmin, vmax=spectrogram_vmax)

    plt.subplot(2, 3, 6)
    plt.title("|Final Error|")
    plt.imshow(abs(final_error), aspect="auto", origin="lower", cmap="magma")

    plt.tight_layout()

    # Save the figure to the specified output path and display it without blocking execution
    plt.show(block=False)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate_transition_case(model, device, mean, std, criterion, case, output_dir):
    # Generate input audio based on the case definition, either a single waveform or a chord
    input_audio = make_input_audio(case)
    target_audio, action_vectors = apply_action_sequence(input_audio, case["actions"])

    # Create the normalized spectrograms for the input and target audio
    input_spectrogram = audio_to_model_cqt(input_audio)
    target_final_spectrogram = audio_to_model_cqt(target_audio)
    input_spectrogram_normalized = normalize_spectrogram(input_spectrogram, mean, std, device)
    target_final_spectrogram_normalized = normalize_spectrogram(
        target_final_spectrogram,
        mean,
        std,
        device,
    )

    # Predict the sequence of spectrograms after applying the actions to the input spectrogram
    predicted_steps = predict_action_sequence(
        model,
        input_spectrogram_normalized,
        action_vectors,
        device,
    )
    predicted_final_spectrogram = predicted_steps[-1]
    loss = criterion(predicted_final_spectrogram, target_final_spectrogram_normalized).item()

    # Save the plot of the transition case to the output directory
    output_path = output_dir / f"{slugify(case['name'])}.png"
    plot_transition_sample(
        case,
        input_spectrogram,
        [denormalize_spectrogram(step, mean, std) for step in predicted_steps],
        target_final_spectrogram,
        loss,
        output_path,
    )

    return {
        "name": case["name"],
        "waveform_type": case["waveform_type"],
        "signal_type": "chord" if case["is_chord"] else "waveform",
        "actions": " -> ".join(action_label(*action) for action in case["actions"]),
        "loss": loss,
        "plot_path": output_path,
    }


def main():
    model, _, device, mean, std = SpectrogramEvaluator().load_model_and_dataset(
        checkpoint_name="spectrogram_transition_model_1.pth",
        train_filename="train.npz",
        test_filename="test.npz",
    )

    model.eval()
    criterion = torch.nn.MSELoss()
    output_dir = DEFAULT_PATHS.build_dir / "latent_space_transition_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_transition_case(model, device, mean, std, criterion, case, output_dir)
        for case in TRANSITION_CASES
    ]

    print("\nLatent transition composition results:")
    print("case                                      signal    waveform   loss")
    for result in sorted(results, key=lambda item: item["loss"]):
        print(
            f"{result['name']:<42}"
            f"{result['signal_type']:<10}"
            f"{result['waveform_type']:<10}"
            f"{result['loss']:.6f}"
        )

    print(f"\nSaved {len(results)} plots to: {output_dir}")


if __name__ == "__main__":
    main()
