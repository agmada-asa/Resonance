from pathlib import Path
import sys

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataset import SyntheticSpectrogramDataset

import torch

from experiments.train import load_normalization_stats
from experiments.unet_model import SpectrogramUNetModel

DATA_DIR = ROOT / "data/synthetic/v001"
EVAL_BATCH_SIZE = 32

# Plot a sample from the dataset, showing the input spectrogram, predicted target spectrogram, and true target spectrogram
def plot_sample(model, dataset, device, sample_index=0):
    import matplotlib.pyplot as plt

    sample = dataset[sample_index]
    metadata = dataset.metadata[sample_index] if dataset.metadata else None
    sample_input = sample["input"].unsqueeze(0).to(device)
    sample_action_vector = sample["action_vector"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predicted_delta = model(sample_input, sample_action_vector)
        predicted_target = sample_input + predicted_delta

    input_spectrogram = sample["input"].squeeze().cpu().numpy()
    predicted_spectrogram = predicted_target.squeeze().cpu().numpy()
    target_spectrogram = sample["target"].squeeze().cpu().numpy()
    predicted_delta_spectrogram = predicted_delta.squeeze().cpu().numpy()
    target_delta_spectrogram = sample["target_delta"].squeeze().cpu().numpy()
    delta_error = predicted_delta_spectrogram - target_delta_spectrogram

    spectrogram_vmin = min(
        input_spectrogram.min(),
        predicted_spectrogram.min(),
        target_spectrogram.min(),
    )
    spectrogram_vmax = max(
        input_spectrogram.max(),
        predicted_spectrogram.max(),
        target_spectrogram.max(),
    )
    delta_abs_max = max(
        abs(predicted_delta_spectrogram).max(),
        abs(target_delta_spectrogram).max(),
        abs(delta_error).max(),
    )

    if metadata:
        print(
            "Sample metadata: "
            f"action={metadata['action']}, "
            f"parameter={metadata['parameter']}, "
            f"waveform={metadata['waveform_type']}, "
            f"frequency={metadata['frequency']:.2f}Hz"
        )

    # Print the loss metrics for this sample
    mse_loss = torch.nn.functional.mse_loss(
        torch.tensor(predicted_delta_spectrogram),
        torch.tensor(target_delta_spectrogram),
    ).item()
    l1_loss = torch.nn.functional.l1_loss(
        torch.tensor(predicted_delta_spectrogram),
        torch.tensor(target_delta_spectrogram),
    ).item()
    print(
        f"Loss for sample {sample_index} with action {metadata['action']}: "
        f"MSE={mse_loss:.6f} L1={l1_loss:.6f}"
    )

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Input Spectrogram")
    plt.imshow(
        input_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 2)
    plt.title("Predicted Target")
    plt.imshow(
        predicted_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 3)
    plt.title("True Target")
    plt.imshow(
        target_spectrogram,
        aspect="auto",
        origin="lower",
        vmin=spectrogram_vmin,
        vmax=spectrogram_vmax,
    )

    plt.subplot(2, 3, 4)
    plt.title("Predicted Delta")
    plt.imshow(
        predicted_delta_spectrogram,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    plt.subplot(2, 3, 5)
    plt.title("True Delta")
    plt.imshow(
        target_delta_spectrogram,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    plt.subplot(2, 3, 6)
    plt.title("Delta Error")
    plt.imshow(
        delta_error,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )

    plt.tight_layout()
    plt.show()


def evaluate_losses_by_action(model, dataset, device, batch_size=EVAL_BATCH_SIZE):
    if dataset.metadata is None:
        raise ValueError("Test metadata file is required to evaluate losses by action.")

    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size)
    totals = {}
    example_offset = 0

    with torch.no_grad():
        for batch in data_loader:
            batch_size = batch["input"].shape[0]
            sample_input = batch["input"].to(device)
            target_delta = batch["target_delta"].to(device)
            target = batch["target"].to(device)
            action_vector = batch["action_vector"].to(device)

            predicted_delta = model(sample_input, action_vector)
            predicted_target = sample_input + predicted_delta

            delta_mse = torch.nn.functional.mse_loss(
                predicted_delta,
                target_delta,
                reduction="none",
            ).mean(dim=(1, 2, 3))
            delta_l1 = torch.nn.functional.l1_loss(
                predicted_delta,
                target_delta,
                reduction="none",
            ).mean(dim=(1, 2, 3))
            identity_mse = torch.nn.functional.mse_loss(
                sample_input,
                target,
                reduction="none",
            ).mean(dim=(1, 2, 3))
            identity_l1 = torch.nn.functional.l1_loss(
                sample_input,
                target,
                reduction="none",
            ).mean(dim=(1, 2, 3))

            for batch_index in range(batch_size):
                action = dataset.metadata[example_offset + batch_index]["action"]
                if action not in totals:
                    totals[action] = {
                        "count": 0,
                        "delta_mse": 0.0,
                        "delta_l1": 0.0,
                        "identity_mse": 0.0,
                        "identity_l1": 0.0,
                    }

                totals[action]["count"] += 1
                totals[action]["delta_mse"] += delta_mse[batch_index].item()
                totals[action]["delta_l1"] += delta_l1[batch_index].item()
                totals[action]["identity_mse"] += identity_mse[batch_index].item()
                totals[action]["identity_l1"] += identity_l1[batch_index].item()

            example_offset += batch_size

    print("\nPer-action test losses:")
    print("action          count  delta_mse  delta_l1   identity_mse  identity_l1")
    for action in sorted(totals):
        action_totals = totals[action]
        count = action_totals["count"]
        print(
            f"{action:<14}"
            f"{count:>5}  "
            f"{action_totals['delta_mse'] / count:>9.6f}  "
            f"{action_totals['delta_l1'] / count:>8.6f}  "
            f"{action_totals['identity_mse'] / count:>12.6f}  "
            f"{action_totals['identity_l1'] / count:>11.6f}"
        )


def main():
    # Determine the device to use (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load the saved model checkpoint
    model_path = ROOT / "build" / "spectrogram_transition_model_1.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model = SpectrogramUNetModel(checkpoint["action_vector_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load the dataset
    mean, std = load_normalization_stats(DATA_DIR / "train.npz")

    dataset = SyntheticSpectrogramDataset(DATA_DIR / "test.npz", mean, std)

    # Find a sample index for each action type in the dataset
    if dataset.metadata is None:
        raise ValueError("Test metadata file is required to pick samples by action.")

    action_types = sorted({metadata["action"] for metadata in dataset.metadata})
    for action in action_types:
        # Find the first sample with the current action type
        sample_index = next(
            (
                i
                for i, metadata in enumerate(dataset.metadata)
                if metadata["action"] == action
            ),
            None,
        )
        if sample_index is not None:
            # Plot the sample for the current action type
            print(f"Plotting sample for action: {action}")
            plot_sample(model, dataset, device, sample_index)

        else:
            print(f"No samples found for action: {action}")

    evaluate_losses_by_action(model, dataset, device)



if __name__ == "__main__":
    main()
