import json

import numpy as np
from sklearn.model_selection import train_test_split

from resonance.actions import AudioActionProcessor
from resonance.config import (
    ACTION_DIMENSIONS,
    ACTION_VECTOR_SIZE,
    MAX_ACTIONS,
    AudioConfig,
    DEFAULT_CONFIG,
    DEFAULT_PATHS,
    ProjectPaths,
)
from resonance.features.spectrogram import SpectrogramTransformer


DEFAULT_CHORD_INTERVALS = (0, 4, 7)
ACTION_TYPES = ('gain', 'pitch_change', 'low_pass', 'high_pass')


class WaveformSynthesizer:
    def __init__(self, config: AudioConfig = DEFAULT_CONFIG):
        self.config = config

    def _sample_count(self):
        return int(self.config.sample_rate * self.config.duration)

    def _time_axis(self):
        return np.linspace(0, self.config.duration, self._sample_count(), endpoint=False)

    def _frequency_ceiling(self):
        return min(self.cqt_max_frequency(), self.config.sample_rate / 2)

    def _validate_frequency(self, frequency, maximum_frequency=None):
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"Frequency must be positive and finite, got {frequency}.")

        maximum_frequency = self.config.sample_rate / 2 if maximum_frequency is None else maximum_frequency
        maximum_frequency = min(maximum_frequency, self.config.sample_rate / 2)
        if frequency >= maximum_frequency:
            raise ValueError(
                f"Frequency {frequency:.2f} Hz must be below {maximum_frequency:.2f} Hz."
            )

    def _maximum_harmonic(self, frequency):
        self._validate_frequency(frequency)
        nyquist_ratio = (self.config.sample_rate / 2) / frequency
        return max(1, int(np.ceil(nyquist_ratio) - 1))

    @staticmethod
    def _scale_to_peak(waveform, amplitude):
        peak = np.max(np.abs(waveform))
        if peak == 0:
            raise ValueError("Cannot normalize a silent waveform.")
        return waveform * (amplitude / peak)

    # SINE WAVES
    def generate_sin_wave(self, frequency, amplitude=0.5):
        """
        Generate a sine wave of a given frequency and amplitude.
        :param frequency: Frequency of the sine wave in Hz
        :param amplitude: Amplitude of the sine wave (0.0 to 1.0)
        :return: Numpy array containing the sine wave samples
        """
        self._validate_frequency(frequency)
        t = self._time_axis()
        return amplitude * np.sin(2 * np.pi * frequency * t)

    # SQUARE WAVES
    def generate_square_wave(self, frequency, amplitude=0.5):
        """
        Generate a square wave of a given frequency and amplitude.
        :param frequency: Frequency of the square wave in Hz
        :param amplitude: Amplitude of the square wave (0.0 to 1.0)
        :return: Numpy array containing the square wave samples
        """
        max_harmonic = self._maximum_harmonic(frequency)

        t = self._time_axis()
        waveform = np.zeros_like(t)

        for n in range(1, max_harmonic + 1, 2):  # Only odd harmonics
            waveform += (1 / n) * np.sin(2 * np.pi * n * frequency * t)

        waveform *= 4 / np.pi
        return self._scale_to_peak(waveform, amplitude)

    # SAWTOOTH WAVES
    def generate_sawtooth_wave(self, frequency, amplitude=0.5):
        """
        Generate a sawtooth wave of a given frequency and amplitude.
        :param frequency: Frequency of the sawtooth wave in Hz
        :param amplitude: Amplitude of the sawtooth wave (0.0 to 1.0)
        :return: Numpy array containing the sawtooth wave samples"""
        max_harmonic = self._maximum_harmonic(frequency)

        t = self._time_axis()
        waveform = np.zeros_like(t)

        for n in range(1, max_harmonic + 1):
            waveform += ((-1) ** (n + 1) / n) * np.sin(2 * np.pi * n * frequency * t)

        waveform *= 2 / np.pi
        return self._scale_to_peak(waveform, amplitude)

    def generate_chords(
        self,
        waveform_type,
        root_frequency,
        amplitude=0.5,
        intervals=DEFAULT_CHORD_INTERVALS,
    ):
        """
        Generate a chord by summing multiple waveforms of the same type at different frequencies.
        :param waveform_type: Type of waveform ('sine', 'square', 'sawtooth')
        :param root_frequency: Root frequency of the chord in Hz
        :param amplitude: Amplitude of the chord (0.0 to 1.0)
        :param intervals: List of intervals in semitones to generate the chord
        :return: Numpy array containing the chord samples
        """
        intervals = tuple(intervals)
        if not intervals:
            raise ValueError("Chord intervals must contain at least one interval.")

        ceiling = self._frequency_ceiling()
        frequencies = [root_frequency * (2 ** (interval / 12)) for interval in intervals]
        for frequency in frequencies:
            self._validate_frequency(frequency, maximum_frequency=ceiling)

        chord = np.zeros(self._sample_count())

        for interval in intervals:
            frequency = root_frequency * (2 ** (interval / 12))
            chord += self.generate_waveform(waveform_type, frequency, amplitude / len(intervals))

        return chord

    def generate_waveform(self, waveform_type, frequency, amplitude):
        if waveform_type == 'sine':
            return self.generate_sin_wave(frequency, amplitude)
        elif waveform_type == 'square':
            return self.generate_square_wave(frequency, amplitude)
        elif waveform_type == 'sawtooth':
            return self.generate_sawtooth_wave(frequency, amplitude)
        else:
            raise ValueError("Unsupported waveform type. Use 'sine', 'square', or 'sawtooth'.")

    def cqt_max_frequency(self):
        return self.config.fmin * 2 ** ((self.config.n_bins - 1) / self.config.bins_per_octave)

    def sample_frequency_for_action(self, rng, action, parameter, intervals=None):
        return self.sample_frequency_for_actions(
            rng,
            [(action, parameter)],
            intervals=intervals,
        )

    def sample_frequency_for_actions(self, rng, actions, intervals=None):
        intervals = (0,) if intervals is None else tuple(intervals)

        interval_factors = [2 ** (interval / 12) for interval in intervals]
        transformed_factors = list(interval_factors)
        cumulative_pitch_factor = 1.0

        # Apply pitch change actions to the cumulative pitch factor and transform the interval factors accordingly
        for action, parameter in actions:
            if action == 'pitch_change':
                cumulative_pitch_factor *= 2 ** (parameter / 12)
                transformed_factors.extend(
                    interval_factor * cumulative_pitch_factor for interval_factor in interval_factors
                )

        min_frequency = self.config.fmin / min(transformed_factors)
        max_frequency = self._frequency_ceiling() / max(transformed_factors)

        if min_frequency >= max_frequency:
            raise ValueError(
                f"No valid frequency range for action sequence {actions}."
            )

        return rng.uniform(min_frequency, max_frequency)


class SyntheticTrainingDataGenerator:
    def __init__(
        self,
        config: AudioConfig = DEFAULT_CONFIG,
        paths: ProjectPaths = DEFAULT_PATHS,
        action_processor: AudioActionProcessor | None = None,
        spectrogram_transformer: SpectrogramTransformer | None = None,
        waveform_synthesizer: WaveformSynthesizer | None = None,
    ):
        self.config = config
        self.paths = paths
        self.action_processor = action_processor or AudioActionProcessor(config)
        self.spectrogram_transformer = spectrogram_transformer or SpectrogramTransformer(config)
        self.waveform_synthesizer = waveform_synthesizer or WaveformSynthesizer(config)

    def _sample_parameter(self, rng, action):
        # Choose a random parameter for the audio transformation
        if action == 'gain':
            return rng.uniform(-12, 12)  # Gain in dB
        elif action == 'pitch_change':
            return int(rng.integers(-12, 13))  # Pitch change in semitones
        elif action in ['low_pass', 'high_pass']:
            return rng.uniform(20, 20000)  # Cutoff frequency in Hz
        else:
            return None  # No parameter needed for 'no_action'

    def _sample_action_sequence(self, rng, num_actions, required_action=None):
        if not 0 <= num_actions <= MAX_ACTIONS:
            raise ValueError(f"num_actions must be between 0 and {MAX_ACTIONS}.")
        if num_actions == 0:
            return []

        action_names = []
        if required_action is not None:
            if required_action not in ACTION_TYPES:
                raise ValueError(f"Unsupported required action: {required_action}.")
            action_names.append(required_action)

        remaining = num_actions - len(action_names)
        action_names.extend(rng.choice(ACTION_TYPES, size=remaining).tolist())
        rng.shuffle(action_names)
        return [
            (action, self._sample_parameter(rng, action))
            for action in action_names
        ]

    def _apply_action_sequence(self, audio, actions):
        # Create an empty array to hold the encoded action vectors
        encoded = np.zeros((MAX_ACTIONS, ACTION_DIMENSIONS), dtype=np.float32)
        modified_audio = audio

        # Apply each action in the sequence to the audio and encode the action vector
        for slot, (action, parameter) in enumerate(actions):
            modified_audio, action_vector = self.action_processor.apply_action(
                modified_audio,
                action,
                parameter,
            )
            encoded[slot] = action_vector

        # Flatten the encoded action vectors into a single 1D array with the correct dimensions for the model input
        return modified_audio, encoded.reshape(-1)

    def _metadata_for_sample(
        self,
        index,
        waveform_type,
        frequency,
        pitch_shifted_frequency,
        amplitude,
        actions,
        is_chord=False,
        seed=None,
    ):
        return {
            'id': index,
            'waveform_type': waveform_type,
            'frequency': frequency,
            'pitch_shifted_frequency': pitch_shifted_frequency,
            'amplitude': amplitude,
            'action': ' -> '.join(action for action, _ in actions) or 'no_action',
            'parameter': [parameter for _, parameter in actions],
            'actions': [
                {'action': action, 'parameter': parameter} for action, parameter in actions
            ],
            'num_actions': len(actions),
            'is_chord': is_chord,
            'sample_rate': self.config.sample_rate,
            'duration': self.config.duration,
            'hop_length': self.config.hop_length,
            'bins_per_octave': self.config.bins_per_octave,
            'n_bins': self.config.n_bins,
            'fmin': self.config.fmin,
            'cqt_db_floor': self.config.cqt_db_floor,
            'seed': seed
        }

    def _write_split(self, output_dir, name, inputs, outputs, actions, metadata):
        np.savez_compressed(
            output_dir / f"{name}.npz",
            input_spectrograms=inputs.astype(np.float32),
            target_spectrograms=outputs.astype(np.float32),
            action_vectors=actions.astype(np.float32),
        )

        # Write metadata to a JSONL file
        with open(output_dir / f"metadata_{name}.jsonl", "w") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

    def generate_training_data(self, seed=42):
        input_spectrograms = []
        output_spectrograms = []
        action_vectors = []
        metadata = []
        index = 0

        rng = np.random.default_rng(seed)

        scenarios = [(None, 0, 50)] # Generate 50 samples with no actions (identity)

        # Generate scenarios for each required action type with varying numbers of actions
        for required_action in ACTION_TYPES:
            sample_count = 100 if required_action in ('pitch_change', 'high_pass') else 50
            scenarios.extend(
                (required_action, num_actions, sample_count) for num_actions in range(1, MAX_ACTIONS + 1)
            )

        total_expected = 3 * sum(sample_count for _, _, sample_count in scenarios)

        for waveform_type in ['sine', 'square', 'sawtooth']:
            for required_action, num_actions, sample_count in scenarios:
                for _ in range(sample_count):
                    # Sample a sequence of actions and their parameters
                    actions = self._sample_action_sequence(
                        rng,
                        num_actions,
                        required_action=required_action,
                    )
                    amplitude = rng.uniform(0.1, 1.0)

                    frequency = self.waveform_synthesizer.sample_frequency_for_actions(
                        rng,
                        actions,
                    )

                    # Create the original audio waveform and a chord based on the sampled frequency and amplitude
                    audio = self.waveform_synthesizer.generate_waveform(
                        waveform_type,
                        frequency,
                        amplitude,
                    )

                    chord_frequency = self.waveform_synthesizer.sample_frequency_for_actions(
                        rng,
                        actions,
                        intervals=DEFAULT_CHORD_INTERVALS,
                    )
                    chord = self.waveform_synthesizer.generate_chords(
                        waveform_type,
                        chord_frequency,
                        amplitude,
                        intervals=DEFAULT_CHORD_INTERVALS,
                    )

                    # Convert the original audio and chord to their corresponding spectrograms
                    spectrogram = self.spectrogram_transformer.audio_to_cqt(audio)
                    chord_spectrogram = self.spectrogram_transformer.audio_to_cqt(chord)

                    pitch_factor = np.prod([
                        2 ** (parameter / 12)
                        for action, parameter in actions
                        if action == 'pitch_change'
                    ])
                    pitch_shifted_frequency = frequency * pitch_factor
                    pitch_shifted_chord_frequency = chord_frequency * pitch_factor

                    # Apply the action sequence to both the single waveform and the chord, obtaining the modified audio and the corresponding action vector
                    modified_audio, action_vector = self._apply_action_sequence(audio, actions)
                    modified_chord_audio, _ = self._apply_action_sequence(chord, actions)

                    modified_spectrogram = self.spectrogram_transformer.audio_to_cqt(modified_audio)
                    modified_chord_spectrogram = self.spectrogram_transformer.audio_to_cqt(
                        modified_chord_audio
                    )

                    # Append the training samples and metadata for both the single waveform and the chord
                    input_spectrograms.append(spectrogram)
                    output_spectrograms.append(modified_spectrogram)
                    action_vectors.append(action_vector)

                    input_spectrograms.append(chord_spectrogram)
                    output_spectrograms.append(modified_chord_spectrogram)
                    action_vectors.append(action_vector)

                    metadata.append(
                        self._metadata_for_sample(
                            index * 2,
                            waveform_type,
                            frequency,
                            pitch_shifted_frequency,
                            amplitude,
                            actions,
                            is_chord=False,
                            seed=seed,
                        )
                    )

                    metadata.append(
                        self._metadata_for_sample(
                            index * 2 + 1,
                            waveform_type,
                            chord_frequency,
                            pitch_shifted_chord_frequency,
                            amplitude,
                            actions,
                            is_chord=True,
                            seed=seed,
                        )
                    )

                    index += 1

                    print(f"Data generation progress: {((index / total_expected) * 100):.2f}%")

        # Convert lists to numpy arrays
        input_spectrograms = np.array(input_spectrograms)
        output_spectrograms = np.array(output_spectrograms)
        action_vectors = np.asarray(action_vectors, dtype=np.float32)
        if action_vectors.shape[1:] != (ACTION_VECTOR_SIZE,):
            raise RuntimeError(
                f"Expected action vectors with shape (N, {ACTION_VECTOR_SIZE}), "
                f"got {action_vectors.shape}."
            )

        print(f"\nGenerated {len(input_spectrograms)} samples. Splitting into train/val/test sets...")

        # Split the data into train / val / test sets (e.g., 80% train, 10% val, 10% test)
        train, other = train_test_split(list(zip(input_spectrograms, output_spectrograms, action_vectors, metadata)), test_size=0.2, random_state=42)
        val, test = train_test_split(other, test_size=0.5, random_state=42)

        train_inputs, train_outputs, train_actions, train_metadata = map(np.array, zip(*train))
        val_inputs, val_outputs, val_actions, val_metadata = map(np.array, zip(*val))
        test_inputs, test_outputs, test_actions, test_metadata = map(np.array, zip(*test))

        output_dir = self.paths.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the datasets and metadata to disk using np.savez
        print(f"Saving datasets to {output_dir}...")

        self._write_split(output_dir, "train", train_inputs, train_outputs, train_actions, train_metadata)
        self._write_split(output_dir, "val", val_inputs, val_outputs, val_actions, val_metadata)
        self._write_split(output_dir, "test", test_inputs, test_outputs, test_actions, test_metadata)

        print(f"Datasets saved successfully in {output_dir}.")
        print(f"Metadata saved successfully in {output_dir}.")
        print(f"Data generation completed. Total samples generated: {len(input_spectrograms)}")


class SyntheticCQNSGTDataGenerator(SyntheticTrainingDataGenerator):
    def _write_metadata_split(self, output_dir, name, metadata):
        # CQ-NSGT vectors are generated when the dataset is read, so only the
        # information required to recreate each audio example is stored here
        metadata_path = output_dir / f'metadata_{name}.jsonl'
        with metadata_path.open('w') as f:
            for entry in metadata:
                f.write(json.dumps(entry) + '\n')

    def generate_training_data(self, seed=42):
        metadata = []
        index = 0
        rng = np.random.default_rng(seed)

        # Keep the same scenario balance as the current spectrogram dataset
        scenarios = [(None, 0, 50)]
        for required_action in ACTION_TYPES:
            sample_count = 100 if required_action in ('pitch_change', 'high_pass') else 50
            scenarios.extend(
                (required_action, num_actions, sample_count)
                for num_actions in range(1, MAX_ACTIONS + 1)
            )

        total_expected = 3 * sum(
            sample_count for _, _, sample_count in scenarios
        )

        for waveform_type in ['sine', 'square', 'sawtooth']:
            for required_action, num_actions, sample_count in scenarios:
                for _ in range(sample_count):
                    # Sample the action sequence and source amplitude
                    actions = self._sample_action_sequence(
                        rng,
                        num_actions,
                        required_action=required_action,
                    )
                    amplitude = rng.uniform(0.1, 1.0)

                    # Sample valid source frequencies for a waveform and chord
                    frequency = self.waveform_synthesizer.sample_frequency_for_actions(
                        rng,
                        actions,
                    )
                    chord_frequency = self.waveform_synthesizer.sample_frequency_for_actions(
                        rng,
                        actions,
                        intervals=DEFAULT_CHORD_INTERVALS,
                    )

                    # Record the final pitch for evaluation and inspection
                    pitch_factor = np.prod([
                        2 ** (parameter / 12)
                        for action, parameter in actions
                        if action == 'pitch_change'
                    ])

                    # Store one waveform and one chord example for this scenario
                    metadata.append(
                        self._metadata_for_sample(
                            index * 2,
                            waveform_type,
                            frequency,
                            frequency * pitch_factor,
                            amplitude,
                            actions,
                            is_chord=False,
                            seed=seed,
                        )
                    )
                    metadata.append(
                        self._metadata_for_sample(
                            index * 2 + 1,
                            waveform_type,
                            chord_frequency,
                            chord_frequency * pitch_factor,
                            amplitude,
                            actions,
                            is_chord=True,
                            seed=seed,
                        )
                    )

                    index += 1
                    if index % 100 == 0 or index == total_expected:
                        progress = (index / total_expected) * 100
                        print(f'CQ-NSGT data generation progress: {progress:.2f}%')

        print(
            f'\nGenerated metadata for {len(metadata)} samples. '
            'Splitting into train/val/test sets...'
        )

        # Split metadata into the same 80/10/10 proportions as the old dataset
        train, other = train_test_split(
            metadata,
            test_size=0.2,
            random_state=seed,
        )
        val, test = train_test_split(
            other,
            test_size=0.5,
            random_state=seed,
        )

        output_dir = self.paths.cq_nsgt_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_metadata_split(output_dir, 'train', train)
        self._write_metadata_split(output_dir, 'val', val)
        self._write_metadata_split(output_dir, 'test', test)

        # Store the representation settings next to the generated splits
        with (output_dir / 'dataset.json').open('w') as f:
            json.dump(
                {
                    'representation': 'cq_nsgt_complex_vectors',
                    'sample_rate': self.config.sample_rate,
                    'duration': self.config.duration,
                    'bins_per_octave': self.config.bins_per_octave,
                    'fmin': self.config.fmin,
                    'train_samples': len(train),
                    'val_samples': len(val),
                    'test_samples': len(test),
                    'seed': seed,
                },
                f,
                indent=2,
            )

        print(f'CQ-NSGT metadata saved successfully in {output_dir}.')
        return {
            'output_dir': output_dir,
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
        }


DEFAULT_WAVEFORM_SYNTHESIZER = WaveformSynthesizer()
DEFAULT_SYNTHETIC_DATA_GENERATOR = SyntheticTrainingDataGenerator()
DEFAULT_CQ_NSGT_DATA_GENERATOR = SyntheticCQNSGTDataGenerator()


def generate_sin_wave(frequency, amplitude=0.5):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_sin_wave(frequency, amplitude)


def generate_square_wave(frequency, amplitude=0.5):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_square_wave(frequency, amplitude)


def generate_sawtooth_wave(frequency, amplitude=0.5):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_sawtooth_wave(frequency, amplitude)


def generate_waveform(waveform_type, frequency, amplitude):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_waveform(waveform_type, frequency, amplitude)


def generate_chords(waveform_type, root_frequency, amplitude=0.5, intervals=DEFAULT_CHORD_INTERVALS):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_chords(waveform_type, root_frequency, amplitude, intervals)


def cqt_max_frequency():
    return DEFAULT_WAVEFORM_SYNTHESIZER.cqt_max_frequency()


def sample_frequency_for_action(rng, action, parameter, intervals=None):
    return DEFAULT_WAVEFORM_SYNTHESIZER.sample_frequency_for_action(rng, action, parameter, intervals)


def generate_training_data(seed=42):
    return DEFAULT_SYNTHETIC_DATA_GENERATOR.generate_training_data(seed=seed)


def generate_cq_nsgt_training_data(seed=42):
    return DEFAULT_CQ_NSGT_DATA_GENERATOR.generate_training_data(seed=seed)
