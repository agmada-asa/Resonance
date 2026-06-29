import unittest

import numpy as np

from resonance.data.synthetic import DEFAULT_CHORD_INTERVALS, WaveformSynthesizer


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


if __name__ == '__main__':
    unittest.main()
