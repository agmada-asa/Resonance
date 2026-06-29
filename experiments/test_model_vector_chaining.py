import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from resonance.config import DEFAULT_PATHS
from resonance.evaluation.evaluator import SpectrogramEvaluator
from test_model_latent_space import (
    TRANSITION_CASES,
    action_label,
    apply_action_sequence,
    audio_to_model_cqt,
    denormalize_spectrogram,
    make_input_audio,
    normalize_spectrogram,
    predict_action_sequence,
    slugify,
)


def compose_action_vectors(action_vectors):
    return np.asarray(action_vectors, dtype=np.float32).sum(axis=0)


def predict_composed_action(model, input_spectrogram_normalized, composed_action_vector, device):
    with torch.no_grad():
        action_tensor = (
            torch.tensor(composed_action_vector, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        predicted_delta = model(input_spectrogram_normalized, action_tensor)
        return input_spectrogram_normalized + predicted_delta


def plot_vector_chaining_sample(
    case,
    input_spectrogram,
    chained_final_spectrogram,
    composed_final_spectrogram,
    target_final_spectrogram,
    losses,
    output_path,
):
    action_sequence = " -> ".join(action_label(*action) for action in case["actions"])
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


def evaluate_vector_chaining_case(model, device, mean, std, criterion, case, output_dir):
    # Generate input audio based on the case definition, either a single waveform or a chord
    input_audio = make_input_audio(case)
    target_audio, action_vectors = apply_action_sequence(input_audio, case["actions"])

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
    chained_steps = predict_action_sequence(
        model,
        input_spectrogram_normalized,
        action_vectors,
        device,
    )
    chained_final_spectrogram_normalized = chained_steps[-1]
    composed_action_vector = compose_action_vectors(action_vectors)
    composed_final_spectrogram_normalized = predict_composed_action(
        model,
        input_spectrogram_normalized,
        composed_action_vector,
        device,
    )

    chained_mse = criterion(
        chained_final_spectrogram_normalized,
        target_final_spectrogram_normalized,
    ).item()

    composed_mse = criterion(
        composed_final_spectrogram_normalized,
        target_final_spectrogram_normalized,
    ).item()

    composed_vs_chained_mse = criterion(
        composed_final_spectrogram_normalized,
        chained_final_spectrogram_normalized,
    ).item()


    losses = {
        "chained_mse": chained_mse,
        "composed_mse": composed_mse,
        "composed_vs_chained_mse": composed_vs_chained_mse,
    }

    # Save the plot of the vector chaining case to the output directory
    output_path = output_dir / f"{slugify(case['name'])}.png"
    plot_vector_chaining_sample(
        case,
        input_spectrogram,
        denormalize_spectrogram(chained_final_spectrogram_normalized, mean, std),
        denormalize_spectrogram(composed_final_spectrogram_normalized, mean, std),
        target_final_spectrogram,
        losses,
        output_path,
    )

    return {
        "name": case["name"],
        "actions": " -> ".join(action_label(*action) for action in case["actions"]),
        "composed_action_vector": composed_action_vector,
        "plot_path": output_path,
        **losses,
    }


def print_results(results):
    # Print the results of the vector chaining evaluation in a formatted table
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


def main():
    # Load the trained model and dataset normalization statistics
    model, _, device, mean, std = SpectrogramEvaluator().load_model_and_dataset(
        checkpoint_name="spectrogram_transition_model_1.pth",
        train_filename="train.npz",
        test_filename="test.npz",
    )

    # Set the model to evaluation mode and define the loss criterion
    model.eval()
    criterion = torch.nn.MSELoss()
    output_dir = DEFAULT_PATHS.build_dir / "latent_space_vector_chaining_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each transition case using both the chained model and the composed action vector approach, saving plots and collecting results
    results = [
        evaluate_vector_chaining_case(model, device, mean, std, criterion, case, output_dir)
        for case in TRANSITION_CASES
    ]

    print_results(results)
    print(f"\nSaved {len(results)} plots to: {output_dir}")


if __name__ == "__main__":
    main()
