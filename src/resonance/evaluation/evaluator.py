import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from resonance.actions import AudioActionProcessor
from resonance.config import DEFAULT_CONFIG, DEFAULT_PATHS, AudioConfig, ProjectPaths
from resonance.data.dataset import SyntheticSpectrogramDataset
from resonance.data.synthetic import WaveformSynthesizer
from resonance.models.unet import SpectrogramUNetModel
from resonance.training.trainer import SpectrogramTransitionTrainer

EVAL_BATCH_SIZE = 32
WAVEFORM_TYPES = ("sine", "square", "sawtooth")
SIGNAL_TYPES = (("waveforms", False), ("chords", True))


class SpectrogramEvaluator:
    audio_export_directory = "audio_samples"
    audio_export_description = "comparison"

    def __init__(
        self,
        config: AudioConfig = DEFAULT_CONFIG,
        paths: ProjectPaths = DEFAULT_PATHS,
        action_processor: AudioActionProcessor | None = None,
        waveform_synthesizer: WaveformSynthesizer | None = None,
        batch_size=EVAL_BATCH_SIZE,
    ):
        self.config = config
        self.paths = paths
        self.action_processor = action_processor or AudioActionProcessor(config)
        self.waveform_synthesizer = waveform_synthesizer or WaveformSynthesizer(config)
        self.batch_size = batch_size

    # Plot a sample from the dataset, showing the input spectrogram, predicted target spectrogram, and true target spectrogram
    def plot_sample(self, model, dataset, device, sample_index=0):
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
            signal_type = "chord" if metadata.get("is_chord", False) else "waveform"
            print(
                "Sample metadata: "
                f"action={metadata['action']}, "
                f"parameter={metadata['parameter']}, "
                f"signal_type={signal_type}, "
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

    def evaluate_losses_by_action(self, model, dataset, device, batch_size=None):
        if dataset.metadata is None:
            raise ValueError("Test metadata file is required to evaluate losses by action.")

        model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size or self.batch_size)
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
                    metadata = dataset.metadata[example_offset + batch_index]
                    action = metadata["action"]
                    if action not in totals:
                        totals[action] = {
                            "count": 0,
                            "delta_mse": 0.0,
                            "delta_l1": 0.0,
                            "identity_mse": 0.0,
                            "identity_l1": 0.0,
                            "breakdown": {
                                signal_type: {} for signal_type, _ in SIGNAL_TYPES
                            },
                        }

                    metric_values = {
                        "delta_mse": delta_mse[batch_index].item(),
                        "delta_l1": delta_l1[batch_index].item(),
                        "identity_mse": identity_mse[batch_index].item(),
                        "identity_l1": identity_l1[batch_index].item(),
                    }
                    self._add_loss_values(totals[action], metric_values)

                    signal_type = (
                        "chords" if metadata.get("is_chord", False) else "waveforms"
                    )
                    waveform_type = metadata["waveform_type"]
                    signal_breakdown = totals[action]["breakdown"][signal_type]
                    if waveform_type not in signal_breakdown:
                        signal_breakdown[waveform_type] = self._empty_loss_totals()
                    self._add_loss_values(
                        signal_breakdown[waveform_type],
                        metric_values,
                    )

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

        for signal_type, _ in SIGNAL_TYPES:
            title = "Chord" if signal_type == "chords" else "Simple waveform"
            print(f"\n{title} loss breakdown:")
            print(
                "waveform  action          count  delta_mse  delta_l1   "
                "identity_mse  identity_l1"
            )
            for waveform_type in WAVEFORM_TYPES:
                for action in sorted(totals):
                    action_totals = totals[action]
                    breakdown = action_totals["breakdown"][signal_type]
                    if waveform_type not in breakdown:
                        continue
                    self._print_loss_row(
                        waveform_type,
                        action,
                        breakdown[waveform_type],
                    )

        return totals

    @staticmethod
    def _empty_loss_totals():
        return {
            "count": 0,
            "delta_mse": 0.0,
            "delta_l1": 0.0,
            "identity_mse": 0.0,
            "identity_l1": 0.0,
        }

    @staticmethod
    def _add_loss_values(totals, metric_values):
        totals["count"] += 1
        for metric, value in metric_values.items():
            totals[metric] += value

    @staticmethod
    def _print_loss_row(waveform_type, action, totals):
        count = totals["count"]
        print(
            f"{waveform_type:<10}"
            f"{action:<14}"
            f"{count:>5}  "
            f"{totals['delta_mse'] / count:>9.6f}  "
            f"{totals['delta_l1'] / count:>8.6f}  "
            f"{totals['identity_mse'] / count:>12.6f}  "
            f"{totals['identity_l1'] / count:>11.6f}"
        )

    def peak_normalize_audio(self, audio, target_peak=0.5):
        audio = np.nan_to_num(audio).astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak == 0 or peak <= target_peak:
            return audio

        return audio * (target_peak / peak)

    def _select_audio_export_indices(self, dataset, samples_per_waveform):
        selected_indices_by_signal_type = {}
        for signal_type, is_chord in SIGNAL_TYPES:
            selected_indices_by_signal_type[signal_type] = {}
            for waveform_type in WAVEFORM_TYPES:
                selected_indices = [
                    index
                    for index, metadata in enumerate(dataset.metadata)
                    if metadata["waveform_type"] == waveform_type
                    and bool(metadata.get("is_chord", False)) == is_chord
                ][:samples_per_waveform]

                if len(selected_indices) < samples_per_waveform:
                    raise ValueError(
                        f"Only found {len(selected_indices)} {waveform_type} "
                        f"{signal_type}; need {samples_per_waveform}."
                    )

                selected_indices_by_signal_type[signal_type][waveform_type] = (
                    selected_indices
                )

        return selected_indices_by_signal_type

    def _prepare_audio_comparison(
        self,
        model,
        dataset,
        device,
        mean,
        std,
        sample_index,
        reconstruction_iterations,
    ):
        metadata = dataset.metadata[sample_index]
        sample = dataset[sample_index]
        sample_input = sample["input"].unsqueeze(0).to(device)
        sample_action_vector = sample["action_vector"].unsqueeze(0).to(device)

        predicted_delta = model(sample_input, sample_action_vector)
        predicted_target = sample_input + predicted_delta
        predicted_cqt_db = predicted_target.squeeze().detach().cpu().numpy() * std + mean
        predicted_cqt_amplitude = librosa.db_to_amplitude(
            np.nan_to_num(predicted_cqt_db),
            ref=1.0,
        )
        predicted_audio = librosa.griffinlim_cqt(
            predicted_cqt_amplitude,
            n_iter=reconstruction_iterations,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            fmin=self.config.fmin,
            bins_per_octave=self.config.bins_per_octave,
            length=int(self.config.sample_rate * self.config.duration),
            random_state=0,
        )

        generate_audio = (
            self.waveform_synthesizer.generate_chords
            if metadata.get("is_chord", False)
            else self.waveform_synthesizer.generate_waveform
        )
        before_audio = generate_audio(
            metadata["waveform_type"], metadata["frequency"], metadata["amplitude"]
        )
        target_after_audio, _ = self.action_processor.apply_action(
            before_audio,
            metadata["action"],
            metadata["parameter"],
        )
        return {
            "metadata": metadata,
            "predicted_cqt_db": predicted_cqt_db,
            "audio": {
                "before.wav": before_audio,
                "target_after.wav": target_after_audio,
                "predicted_after.wav": predicted_audio,
            },
        }

    def _format_audio_export(self, comparison, sample_number, sample_index):
        metadata = comparison["metadata"]
        parameter = metadata["parameter"]
        parameter_text = "None" if parameter is None else f"{parameter:.2f}"
        file_stem = (
            f"{sample_number:02d}_idx{sample_index}_"
            f"{metadata['action']}_{parameter_text}"
        ).replace("+", "plus").replace("-", "minus")
        message = (
            f"  {'chord' if metadata.get('is_chord', False) else 'waveform'} "
            f"{metadata['waveform_type']} sample {sample_number}: "
            f"action={metadata['action']}, parameter={parameter}"
        )
        return file_stem, message

    def export_audio_comparison_samples(
        self,
        model,
        dataset,
        device,
        mean,
        std,
        samples_per_waveform=3,
        output_dir=None,
        reconstruction_iterations=32,
    ):
        if dataset.metadata is None:
            raise ValueError("Test metadata file is required to export audio samples.")

        output_dir = (
            self.paths.build_dir / self.audio_export_directory
            if output_dir is None
            else output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_indices_by_signal_type = self._select_audio_export_indices(
            dataset,
            samples_per_waveform,
        )

        model.eval()
        exported_files = []

        print(f"\nExporting {self.audio_export_description} audio samples to {output_dir}:")
        with torch.no_grad():
            for signal_type, indices_by_waveform in (
                selected_indices_by_signal_type.items()
            ):
                for waveform_type, selected_indices in indices_by_waveform.items():
                    waveform_dir = output_dir / signal_type / waveform_type
                    waveform_dir.mkdir(parents=True, exist_ok=True)

                    for sample_number, sample_index in enumerate(
                        selected_indices,
                        start=1,
                    ):
                        comparison = self._prepare_audio_comparison(
                            model,
                            dataset,
                            device,
                            mean,
                            std,
                            sample_index,
                            reconstruction_iterations,
                        )
                        file_stem, message = self._format_audio_export(
                            comparison,
                            sample_number,
                            sample_index,
                        )
                        sample_dir = waveform_dir / file_stem
                        sample_dir.mkdir(parents=True, exist_ok=True)

                        for filename, audio in comparison["audio"].items():
                            path = sample_dir / filename
                            sf.write(
                                path,
                                self.peak_normalize_audio(audio),
                                self.config.sample_rate,
                                subtype="FLOAT",
                            )
                            exported_files.append(path)
                        print(message)

        print(f"Exported {len(exported_files)} audio files.")
        return exported_files

    def load_model_and_dataset(
        self,
        checkpoint_name="spectrogram_transition_model_1.pth",
        train_filename="train.npz",
        test_filename="test.npz",
    ):
        trainer = SpectrogramTransitionTrainer(paths=self.paths)
        device = trainer.select_device()

        print(f"Using device: {device}")

        # Load the saved model checkpoint
        model_path = self.paths.build_dir / checkpoint_name
        checkpoint = torch.load(model_path, map_location=device)
        model = SpectrogramUNetModel(checkpoint["action_vector_size"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load the dataset
        mean, std = trainer.load_normalization_stats(self.paths.data_dir / train_filename)

        dataset = SyntheticSpectrogramDataset(self.paths.data_dir / test_filename, mean, std)
        return model, dataset, device, mean, std

    def run(
        self,
        checkpoint_name="spectrogram_transition_model_1.pth",
        train_filename="train.npz",
        test_filename="test.npz",
    ):
        model, dataset, device, mean, std = self.load_model_and_dataset(
            checkpoint_name=checkpoint_name,
            train_filename=train_filename,
            test_filename=test_filename,
        )

        # Find a sample index for each action type in the dataset
        if dataset.metadata is None:
            raise ValueError("Test metadata file is required to pick samples by action.")

        action_types = sorted({metadata["action"] for metadata in dataset.metadata})
        for signal_type, is_chord in SIGNAL_TYPES:
            for waveform_type in WAVEFORM_TYPES:
                for action in action_types:
                    sample_index = next(
                        (
                            i
                            for i, metadata in enumerate(dataset.metadata)
                            if metadata["action"] == action
                            and metadata["waveform_type"] == waveform_type
                            and bool(metadata.get("is_chord", False)) == is_chord
                        ),
                        None,
                    )
                    if sample_index is not None:
                        print(
                            f"Plotting {signal_type[:-1]} sample for action "
                            f"{action}, waveform: {waveform_type}"
                        )
                        self.plot_sample(model, dataset, device, sample_index)
                    else:
                        print(
                            f"No {signal_type} samples found for action {action}, "
                            f"waveform: {waveform_type}"
                        )

        self.export_audio_comparison_samples(model, dataset, device, mean, std)
        self.evaluate_losses_by_action(model, dataset, device)


class PitchOnlySpectrogramEvaluator(SpectrogramEvaluator):
    audio_export_directory = "pitch_only_audio_samples"
    audio_export_description = "pitch comparison"

    def peak_normalize_audio(self, audio, target_peak=0.95):
        return super().peak_normalize_audio(audio, target_peak)

    def estimate_dominant_frequency_from_cqt(self, cqt_db):
        bin_energy = cqt_db.mean(axis=1)
        dominant_bin = int(np.argmax(bin_energy))
        return self.config.fmin * 2 ** (dominant_bin / self.config.bins_per_octave)

    def _format_audio_export(self, comparison, sample_number, sample_index):
        metadata = comparison["metadata"]
        predicted_frequency = self.estimate_dominant_frequency_from_cqt(
            comparison["predicted_cqt_db"]
        )
        comparison["audio"]["predicted_pitch_resynth.wav"] = (
            (
                self.waveform_synthesizer.generate_chords
                if metadata.get("is_chord", False)
                else self.waveform_synthesizer.generate_waveform
            )(
                metadata["waveform_type"],
                predicted_frequency,
                metadata["amplitude"],
            )
        )
        file_stem = (
            f"{sample_number:02d}_idx{sample_index}_"
            f"{metadata['parameter']:+d}st_"
            f"{metadata['frequency']:.2f}hz"
        ).replace("+", "plus").replace("-", "minus")
        message = (
            f"  {'chord' if metadata.get('is_chord', False) else 'waveform'} "
            f"{metadata['waveform_type']} sample {sample_number}: "
            f"{metadata['frequency']:.2f}Hz -> "
            f"{metadata['pitch_shifted_frequency']:.2f}Hz "
            f"({metadata['parameter']:+d} semitones), "
            f"predicted peak {predicted_frequency:.2f}Hz"
        )
        return file_stem, message

    def run(self):
        return super().run(
            checkpoint_name="spectrogram_transition_model_pitch_only_1.pth",
            train_filename="train_pitch_only.npz",
            test_filename="test_pitch_only.npz",
        )


DEFAULT_EVALUATOR = SpectrogramEvaluator()


def plot_sample(model, dataset, device, sample_index=0):
    return DEFAULT_EVALUATOR.plot_sample(model, dataset, device, sample_index)


def evaluate_losses_by_action(model, dataset, device, batch_size=EVAL_BATCH_SIZE):
    return DEFAULT_EVALUATOR.evaluate_losses_by_action(model, dataset, device, batch_size)


def peak_normalize_audio(audio, target_peak=0.5):
    return DEFAULT_EVALUATOR.peak_normalize_audio(audio, target_peak)


def export_audio_comparison_samples(
    model,
    dataset,
    device,
    mean,
    std,
    samples_per_waveform=3,
    output_dir=None,
    reconstruction_iterations=32,
):
    return DEFAULT_EVALUATOR.export_audio_comparison_samples(
        model,
        dataset,
        device,
        mean,
        std,
        samples_per_waveform,
        output_dir,
        reconstruction_iterations,
    )
