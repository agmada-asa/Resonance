import json
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

from resonance.actions import AudioActionProcessor
from resonance.config import AudioConfig, DEFAULT_CONFIG, DEFAULT_PATHS, ProjectPaths
from resonance.features.spectrogram import SpectrogramTransformer


class WaveformSynthesizer:
    def __init__(self, config: AudioConfig = DEFAULT_CONFIG):
        self.config = config

    # SINE WAVES
    def generate_sin_wave(self, frequency, amplitude=0.5):
        """
        Generate a sine wave of a given frequency and amplitude.
        :param frequency: Frequency of the sine wave in Hz
        :param amplitude: Amplitude of the sine wave (0.0 to 1.0)
        :return: Numpy array containing the sine wave samples
        """
        t = np.linspace(0, self.config.duration, int(self.config.sample_rate * self.config.duration), endpoint=False)
        return amplitude * np.sin(2 * np.pi * frequency * t)

    # SQUARE WAVES
    def generate_square_wave(self, frequency, amplitude=0.5):
        """
        Generate a square wave of a given frequency and amplitude.
        :param frequency: Frequency of the square wave in Hz
        :param amplitude: Amplitude of the square wave (0.0 to 1.0)
        :return: Numpy array containing the square wave samples
        """
        t = np.linspace(0, self.config.duration, int(self.config.sample_rate * self.config.duration), endpoint=False)
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

    # SAWTOOTH WAVES
    def generate_sawtooth_wave(self, frequency, amplitude=0.5):
        """
        Generate a sawtooth wave of a given frequency and amplitude.
        :param frequency: Frequency of the sawtooth wave in Hz
        :param amplitude: Amplitude of the sawtooth wave (0.0 to 1.0)
        :return: Numpy array containing the sawtooth wave samples"""
        t = np.linspace(0, self.config.duration, int(self.config.sample_rate * self.config.duration), endpoint=False)
        return amplitude * (2 * (t * frequency - np.floor(t * frequency + 0.5)))

    def create_audio_file(self, filepath, waveform_type, frequency, amplitude=0.5):
        """
        Create an audio file with the specified waveform type, frequency, and amplitude.
        :param filepath: Path to save the audio file
        :param waveform_type: Type of waveform ('sine', 'square', 'sawtooth')
        :param frequency: Frequency of the waveform in Hz
        :param amplitude: Amplitude of the waveform (0.0 to 1.0)
        :return: None"""
        if waveform_type == 'sine':
            sf.write(filepath, self.generate_sin_wave(frequency, amplitude), self.config.sample_rate)
        elif waveform_type == 'square':
            sf.write(filepath, self.generate_square_wave(frequency, amplitude), self.config.sample_rate)
        elif waveform_type == 'sawtooth':
            sf.write(filepath, self.generate_sawtooth_wave(frequency, amplitude), self.config.sample_rate)
        else:
            raise ValueError("Unsupported waveform type. Use 'sine', 'square', or 'sawtooth'.")

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

    def sample_frequency_for_action(self, rng, action, parameter):
        min_frequency = self.config.fmin
        max_frequency = min(self.cqt_max_frequency(), self.config.sample_rate / 2)

        if action == 'pitch_change':
            pitch_factor = 2 ** (parameter / 12)
            min_frequency = max(min_frequency, self.config.fmin / pitch_factor)
            max_frequency = min(max_frequency, self.cqt_max_frequency() / pitch_factor, (self.config.sample_rate / 2) / pitch_factor)

        if min_frequency >= max_frequency:
            raise ValueError(
                f"No valid frequency range for action={action}, parameter={parameter}."
            )

        return rng.uniform(min_frequency, max_frequency)

    def generate_data(self):
        """
        Generate synthetic audio data for training a model
        Types: sine wave, square wave, sawtooth wave
        # Frequencies: Frequencies from 20Hz to 20kHz
        Duration: 2 seconds
        Amplitude: 0.1 - 1.0
        Save the generated audio files in a directory structure based on type, frequency and amplitude
        :return: None
        """

        # Generate samples
        waveform_types = ['sine', 'square', 'sawtooth']
        frequencies = [262, 277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494]  # Example frequencies (C4 to B4)
        amplitudes = [0.1, 0.3, 0.5, 0.7, 1.0]  # Example amplitudes

        for waveform_type in waveform_types:
            for frequency in frequencies:
                for amplitude in amplitudes:
                    filename = Path(f"../data/{waveform_type}/{frequency}Hz/{amplitude}amp.wav")
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    self.create_audio_file(filename, waveform_type, frequency, amplitude)
                    print(f"Generated {filename}")


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

    def _metadata_for_sample(
        self,
        index,
        waveform_type,
        frequency,
        pitch_shifted_frequency,
        amplitude,
        action,
        parameter,
        seed,
    ):
        return {
            'id': index,
            'waveform_type': waveform_type,
            'frequency': frequency,
            'pitch_shifted_frequency': pitch_shifted_frequency,
            'amplitude': amplitude,
            'action': action,
            'parameter': parameter,
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

    def generate_training_data(self, seed=42, pitch_only=False):
        input_spectrograms = []
        output_spectrograms = []
        action_vectors = []
        metadata = []  # To store metadata about the generated samples (e.g., waveform type, frequency, amplitude, action applied)
        index = 0

        rng = np.random.default_rng(seed)

        for waveform_type in ['sine', 'square', 'sawtooth']:
            actions = ['pitch_change'] if pitch_only else ['no_action', 'gain', 'pitch_change', 'low_pass', 'high_pass']
            for action in actions:
                if pitch_only:
                    sample_count = 1000  # Generate 1000 samples for each waveform type
                else:
                    # Increase proportion of high pass and pitch change samples as these are harder to learn and have worst performance
                    sample_count = 1000 if action == 'pitch_change' or action == 'high_pass' else 500

                for _ in range(sample_count):  # Generate 1000 samples for each waveform type
                    amplitude = rng.uniform(0.1, 1.0)  # Random amplitude between 0.1 and 1.0

                    if pitch_only:
                        # Choose a random action and parameter for the audio transformation
                        action = 'pitch_change'  # For this specific script, we are only applying pitch change
                        parameter = int(rng.integers(-12, 13))  # Pitch change in semitone
                    else:
                        parameter = self._sample_parameter(rng, action)

                    frequency = self.waveform_synthesizer.sample_frequency_for_action(rng, action, parameter)

                    audio = self.waveform_synthesizer.generate_waveform(waveform_type, frequency, amplitude)
                    spectogram = self.spectrogram_transformer.audio_to_cqt(audio)

                    # Apply the action to the audio and get the action vector and the modified audio spectrogram
                    pitch_shifted_frequency = None
                    if action == 'pitch_change':
                        pitch_factor = 2 ** (parameter / 12)
                        pitch_shifted_frequency = frequency * pitch_factor

                    modified_audio, action_vector = self.action_processor.apply_action(audio, action, parameter)
                    modified_spectrogram = self.spectrogram_transformer.audio_to_cqt(modified_audio)

                    # Append the input spectrogram, output spectrogram, action vector, and metadata to the respective lists
                    input_spectrograms.append(spectogram)
                    output_spectrograms.append(modified_spectrogram)
                    action_vectors.append(action_vector)
                    metadata.append(
                        self._metadata_for_sample(
                            index,
                            waveform_type,
                            frequency,
                            pitch_shifted_frequency,
                            amplitude,
                            action,
                            parameter,
                            seed,
                        )
                    )

                    index += 1

                    total_expected = 3 * 1000 if pitch_only else 3 * 3500
                    print(f"Data generation progress: {((index / total_expected) * 100):.2f}%")

        # Convert lists to numpy arrays
        input_spectrograms = np.array(input_spectrograms)
        output_spectrograms = np.array(output_spectrograms)
        action_vectors = np.array(action_vectors)

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

        suffix = "_pitch_only" if pitch_only else ""
        self._write_split(output_dir, f"train{suffix}", train_inputs, train_outputs, train_actions, train_metadata)
        self._write_split(output_dir, f"val{suffix}", val_inputs, val_outputs, val_actions, val_metadata)
        self._write_split(output_dir, f"test{suffix}", test_inputs, test_outputs, test_actions, test_metadata)

        print(f"Datasets saved successfully in {output_dir}.")
        print(f"Metadata saved successfully in {output_dir}.")
        print(f"Data generation completed. Total samples generated: {len(input_spectrograms)}")


DEFAULT_WAVEFORM_SYNTHESIZER = WaveformSynthesizer()
DEFAULT_SYNTHETIC_DATA_GENERATOR = SyntheticTrainingDataGenerator()


def generate_sin_wave(frequency, amplitude=0.5):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_sin_wave(frequency, amplitude)


def generate_square_wave(frequency, amplitude=0.5):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_square_wave(frequency, amplitude)


def generate_sawtooth_wave(frequency, amplitude=0.5):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_sawtooth_wave(frequency, amplitude)


def create_audio_file(filepath, waveform_type, frequency, amplitude=0.5):
    return DEFAULT_WAVEFORM_SYNTHESIZER.create_audio_file(filepath, waveform_type, frequency, amplitude)


def generate_waveform(waveform_type, frequency, amplitude):
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_waveform(waveform_type, frequency, amplitude)


def cqt_max_frequency():
    return DEFAULT_WAVEFORM_SYNTHESIZER.cqt_max_frequency()


def sample_frequency_for_action(rng, action, parameter):
    return DEFAULT_WAVEFORM_SYNTHESIZER.sample_frequency_for_action(rng, action, parameter)


def generate_data():
    return DEFAULT_WAVEFORM_SYNTHESIZER.generate_data()


def generate_training_data(seed=42, pitch_only=False):
    return DEFAULT_SYNTHETIC_DATA_GENERATOR.generate_training_data(seed=seed, pitch_only=pitch_only)

