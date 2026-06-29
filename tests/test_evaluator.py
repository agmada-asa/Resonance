import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from resonance.evaluation.evaluator import (
    SIGNAL_TYPES,
    WAVEFORM_TYPES,
    SpectrogramEvaluator,
)


class EvaluationDataset(Dataset):
    def __init__(self):
        self.metadata = []
        self.samples = []
        for action_index, action in enumerate(("gain", "no_action"), start=1):
            for waveform_type in WAVEFORM_TYPES:
                for _, is_chord in SIGNAL_TYPES:
                    sample_input = torch.zeros(1, 2, 2)
                    target_delta = torch.full((1, 2, 2), float(action_index))
                    self.samples.append(
                        {
                            "input": sample_input,
                            "target_delta": target_delta,
                            "target": sample_input + target_delta,
                            "action_vector": torch.zeros(2),
                        }
                    )
                    self.metadata.append(
                        {
                            "action": action,
                            "waveform_type": waveform_type,
                            "is_chord": is_chord,
                            "frequency": 220.0,
                            "amplitude": 0.5,
                            "parameter": None,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ZeroModel(torch.nn.Module):
    def forward(self, sample_input, action_vector):
        return torch.zeros_like(sample_input)


class RecordingEvaluator(SpectrogramEvaluator):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.plotted_indices = []

    def load_model_and_dataset(self, **kwargs):
        return ZeroModel(), self.dataset, torch.device("cpu"), 0.0, 1.0

    def plot_sample(self, model, dataset, device, sample_index=0):
        self.plotted_indices.append(sample_index)

    def export_audio_comparison_samples(self, *args, **kwargs):
        return []

    def evaluate_losses_by_action(self, *args, **kwargs):
        return {}


class SpectrogramEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.dataset = EvaluationDataset()
        self.evaluator = SpectrogramEvaluator(batch_size=4)

    def test_loss_totals_keep_overall_and_add_waveform_and_chord_breakdowns(self):
        output = io.StringIO()
        with redirect_stdout(output):
            totals = self.evaluator.evaluate_losses_by_action(
                ZeroModel(), self.dataset, torch.device("cpu")
            )

        self.assertEqual(totals["gain"]["count"], 6)
        self.assertEqual(totals["no_action"]["count"], 6)
        for action in ("gain", "no_action"):
            for signal_type, _ in SIGNAL_TYPES:
                breakdown = totals[action]["breakdown"][signal_type]
                self.assertEqual(set(breakdown), set(WAVEFORM_TYPES))
                self.assertEqual(sum(row["count"] for row in breakdown.values()), 3)

        report = output.getvalue()
        self.assertIn("Per-action test losses:", report)
        self.assertIn("Simple waveform loss breakdown:", report)
        self.assertIn("Chord loss breakdown:", report)

    def test_audio_selection_returns_each_waveform_for_both_signal_types(self):
        selected = self.evaluator._select_audio_export_indices(
            self.dataset, samples_per_waveform=1
        )

        self.assertEqual(set(selected), {"waveforms", "chords"})
        for signal_type, is_chord in SIGNAL_TYPES:
            self.assertEqual(set(selected[signal_type]), set(WAVEFORM_TYPES))
            for waveform_type, indices in selected[signal_type].items():
                self.assertEqual(len(indices), 1)
                metadata = self.dataset.metadata[indices[0]]
                self.assertEqual(metadata["waveform_type"], waveform_type)
                self.assertIs(metadata["is_chord"], is_chord)

    def test_run_plots_every_action_waveform_and_signal_combination(self):
        evaluator = RecordingEvaluator(self.dataset)
        with redirect_stdout(io.StringIO()):
            evaluator.run()

        self.assertEqual(len(evaluator.plotted_indices), 12)
        plotted_metadata = [self.dataset.metadata[i] for i in evaluator.plotted_indices]
        combinations = {
            (row["action"], row["waveform_type"], row["is_chord"])
            for row in plotted_metadata
        }
        self.assertEqual(len(combinations), 12)

    def test_chord_audio_comparison_regenerates_a_chord(self):
        class Synthesizer:
            def __init__(self):
                self.calls = []

            def generate_waveform(self, *args):
                self.calls.append(("waveform", args))
                return np.ones(8, dtype=np.float32)

            def generate_chords(self, *args):
                self.calls.append(("chord", args))
                return np.full(8, 2.0, dtype=np.float32)

        class ActionProcessor:
            def apply_action(self, audio, action, parameter):
                return audio, np.zeros(2)

        chord_index = next(
            index
            for index, metadata in enumerate(self.dataset.metadata)
            if metadata["is_chord"]
        )
        synthesizer = Synthesizer()
        evaluator = SpectrogramEvaluator(
            waveform_synthesizer=synthesizer,
            action_processor=ActionProcessor(),
        )
        with (
            patch(
                "resonance.evaluation.evaluator.librosa.db_to_amplitude",
                side_effect=lambda value, ref: value,
            ),
            patch(
                "resonance.evaluation.evaluator.librosa.griffinlim_cqt",
                return_value=np.zeros(8, dtype=np.float32),
            ),
        ):
            comparison = evaluator._prepare_audio_comparison(
                ZeroModel(),
                self.dataset,
                torch.device("cpu"),
                0.0,
                1.0,
                chord_index,
                1,
            )

        self.assertEqual(synthesizer.calls[0][0], "chord")
        np.testing.assert_array_equal(
            comparison["audio"]["before.wav"],
            np.full(8, 2.0, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
