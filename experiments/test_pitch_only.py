from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.actions import apply_action
from experiments.config import BINS_PER_OCTAVE, DURATION, FMIN, HOP_LENGTH, SAMPLE_RATE
from experiments.dataset import SyntheticSpectrogramDataset
from experiments.generate_synthetic_audio_pitch_only import generate_waveform
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


def peak_normalize_audio(audio, target_peak=0.95):
    audio = np.nan_to_num(audio).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak == 0 or peak <= target_peak:
        return audio

    return audio * (target_peak / peak)


def estimate_dominant_frequency_from_cqt(cqt_db):
    bin_energy = cqt_db.mean(axis=1)
    dominant_bin = int(np.argmax(bin_energy))
    return FMIN * 2 ** (dominant_bin / BINS_PER_OCTAVE)


def export_audio_comparison_samples(
    model,
    dataset,
    device,
    mean,
    std,
    samples_per_waveform=3,
    output_dir=ROOT / "build" / "pitch_only_audio_samples",
    reconstruction_iterations=32,
):
    if dataset.metadata is None:
        raise ValueError("Test metadata file is required to export audio samples.")

    output_dir.mkdir(parents=True, exist_ok=True)
    waveform_types = ["sine", "square", "sawtooth"]
    selected_indices_by_waveform = {}

    for waveform_type in waveform_types:
        selected_indices = [
            index
            for index, metadata in enumerate(dataset.metadata)
            if metadata["waveform_type"] == waveform_type
        ][:samples_per_waveform]

        if len(selected_indices) < samples_per_waveform:
            raise ValueError(
                f"Only found {len(selected_indices)} {waveform_type} samples; "
                f"need {samples_per_waveform}."
            )

        selected_indices_by_waveform[waveform_type] = selected_indices

    model.eval()
    exported_files = []

    print(f"\nExporting pitch comparison audio samples to {output_dir}:")
    with torch.no_grad():
        for waveform_type, selected_indices in selected_indices_by_waveform.items():
            waveform_dir = output_dir / waveform_type
            waveform_dir.mkdir(parents=True, exist_ok=True)

            for sample_number, sample_index in enumerate(selected_indices, start=1):
                metadata = dataset.metadata[sample_index]
                sample = dataset[sample_index]
                sample_input = sample["input"].unsqueeze(0).to(device)
                sample_action_vector = sample["action_vector"].unsqueeze(0).to(device)

                predicted_delta = model(sample_input, sample_action_vector)
                predicted_target = sample_input + predicted_delta
                predicted_cqt_db = (
                    predicted_target.squeeze().detach().cpu().numpy() * std + mean
                )
                predicted_frequency = estimate_dominant_frequency_from_cqt(
                    predicted_cqt_db
                )
                predicted_cqt_amplitude = librosa.db_to_amplitude(
                    np.nan_to_num(predicted_cqt_db),
                    ref=1.0,
                )
                predicted_audio = librosa.griffinlim_cqt(
                    predicted_cqt_amplitude,
                    n_iter=reconstruction_iterations,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_LENGTH,
                    fmin=FMIN,
                    bins_per_octave=BINS_PER_OCTAVE,
                    length=int(SAMPLE_RATE * DURATION),
                    random_state=0,
                )

                before_audio = generate_waveform(
                    metadata["waveform_type"],
                    metadata["frequency"],
                    metadata["amplitude"],
                )
                target_after_audio, _ = apply_action(
                    before_audio,
                    metadata["action"],
                    metadata["parameter"],
                )
                predicted_pitch_resynth_audio = generate_waveform(
                    metadata["waveform_type"],
                    predicted_frequency,
                    metadata["amplitude"],
                )

                file_stem = (
                    f"{sample_number:02d}_idx{sample_index}_"
                    f"{metadata['parameter']:+d}st_"
                    f"{metadata['frequency']:.2f}hz"
                ).replace("+", "plus").replace("-", "minus")
                sample_dir = waveform_dir / file_stem
                sample_dir.mkdir(parents=True, exist_ok=True)

                before_path = sample_dir / "before.wav"
                target_after_path = sample_dir / "target_after.wav"
                predicted_after_path = sample_dir / "predicted_after.wav"
                predicted_pitch_resynth_path = sample_dir / "predicted_pitch_resynth.wav"

                sf.write(
                    before_path,
                    peak_normalize_audio(before_audio),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
                sf.write(
                    target_after_path,
                    peak_normalize_audio(target_after_audio),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
                sf.write(
                    predicted_after_path,
                    peak_normalize_audio(predicted_audio),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
                sf.write(
                    predicted_pitch_resynth_path,
                    peak_normalize_audio(predicted_pitch_resynth_audio),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )

                exported_files.extend(
                    [
                        before_path,
                        target_after_path,
                        predicted_after_path,
                        predicted_pitch_resynth_path,
                    ]
                )
                print(
                    f"  {waveform_type} sample {sample_number}: "
                    f"{metadata['frequency']:.2f}Hz -> "
                    f"{metadata['pitch_shifted_frequency']:.2f}Hz "
                    f"({metadata['parameter']:+d} semitones), "
                    f"predicted peak {predicted_frequency:.2f}Hz"
                )

    print(f"Exported {len(exported_files)} audio files.")
    return exported_files


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
    model_path = ROOT / "build" / "spectrogram_transition_model_pitch_only_1.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model = SpectrogramUNetModel(checkpoint["action_vector_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load the dataset
    mean, std = load_normalization_stats(DATA_DIR / "train_pitch_only.npz")

    dataset = SyntheticSpectrogramDataset(DATA_DIR / "test_pitch_only.npz", mean, std)

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

    export_audio_comparison_samples(model, dataset, device, mean, std)
    evaluate_losses_by_action(model, dataset, device)



if __name__ == "__main__":
    main()
