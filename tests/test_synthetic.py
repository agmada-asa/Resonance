import unittest

import numpy as np

from resonance.config import ACTION_DIMENSIONS, ACTION_VECTOR_SIZE, MAX_ACTIONS
from resonance.data.synthetic import (
    DEFAULT_CHORD_INTERVALS,
    SyntheticTrainingDataGenerator,
    WaveformSynthesizer,
)


class WaveformSynthesizerTests(unittest.TestCase):
    def setUp(self):
        self.synthesizer = WaveformSynthesizer()

    def test_band_limited_square_and_sawtooth_use_requested_peak_amplitude(self):
        for generator in (
            self.synthesizer.generate_square_wave,
            self.synthesizer.generate_sawtooth_wave,
        ):
            for frequency in (100.0, 1000.0, 10000.0, 20000.0):
                audio = generator(frequency, amplitude=0.4)
                self.assertAlmostEqual(float(np.max(np.abs(audio))), 0.4, places=5)

    def test_chord_frequency_sampling_keeps_pitch_shifted_notes_under_ceiling(self):
        rng = np.random.default_rng(123)
        ceiling = self.synthesizer._frequency_ceiling()
        highest_interval_factor = max(2 ** (interval / 12) for interval in DEFAULT_CHORD_INTERVALS)

        for semitones in (1, 7, 12, -12):
            pitch_factor = 2 ** (semitones / 12)

            for _ in range(200):
                root_frequency = self.synthesizer.sample_frequency_for_action(
                    rng,
                    'pitch_change',
                    semitones,
                    intervals=DEFAULT_CHORD_INTERVALS,
                )

                self.assertLess(root_frequency * highest_interval_factor, ceiling)
                self.assertLess(root_frequency * highest_interval_factor * pitch_factor, ceiling)
                self.assertGreaterEqual(root_frequency * min(1.0, pitch_factor), self.synthesizer.config.fmin)

    def test_chord_sampling_derives_bounds_from_custom_intervals(self):
        rng = np.random.default_rng(321)
        octave_chord_intervals = (0, 12)

        root_frequency = self.synthesizer.sample_frequency_for_action(
            rng,
            'pitch_change',
            12,
            intervals=octave_chord_intervals,
        )

        self.assertLess(root_frequency * 2 * 2, self.synthesizer._frequency_ceiling())

    def test_generate_chords_rejects_notes_above_ceiling(self):
        unsafe_root = self.synthesizer._frequency_ceiling() / (2 ** (7 / 12))

        with self.assertRaises(ValueError):
            self.synthesizer.generate_chords(
                'sine',
                unsafe_root,
                intervals=DEFAULT_CHORD_INTERVALS,
            )


class SyntheticTrainingDataGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticTrainingDataGenerator()

    def test_action_sequence_is_flattened_and_zero_padded(self):
        audio = np.ones(32, dtype=np.float32)
        actions = [('gain', 6), ('gain', -3)]

        modified_audio, action_vector = self.generator._apply_action_sequence(
            audio,
            actions,
        )

        self.assertEqual(action_vector.shape, (ACTION_VECTOR_SIZE,))
        np.testing.assert_allclose(
            action_vector[:ACTION_DIMENSIONS],
            self.generator.action_processor.encode_action(*actions[0]),
        )
        np.testing.assert_allclose(
            action_vector[ACTION_DIMENSIONS:2 * ACTION_DIMENSIONS],
            self.generator.action_processor.encode_action(*actions[1]),
        )
        np.testing.assert_array_equal(action_vector[2 * ACTION_DIMENSIONS:], 0)
        np.testing.assert_allclose(modified_audio, audio * 10 ** (3 / 20))

    def test_empty_action_sequence_is_an_identity_sample(self):
        audio = np.linspace(-1, 1, 32, dtype=np.float32)

        modified_audio, action_vector = self.generator._apply_action_sequence(audio, [])

        np.testing.assert_array_equal(modified_audio, audio)
        np.testing.assert_array_equal(action_vector, 0)

    def test_sampled_sequence_has_requested_length_and_required_action(self):
        rng = np.random.default_rng(123)

        actions = self.generator._sample_action_sequence(
            rng,
            num_actions=4,
            required_action='high_pass',
        )

        self.assertEqual(len(actions), 4)
        self.assertIn('high_pass', [action for action, _ in actions])


if __name__ == '__main__':
    unittest.main()
