import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.actions import apply_action, encode_action
from experiments.audio_to_spectogram import audio_to_cqt
from experiments.config import BINS_PER_OCTAVE, CQT_DB_FLOOR, FMIN, HOP_LENGTH, N_BINS, N_FFT, N_MELS, SAMPLE_RATE, DURATION

# SINE WAVES
def generate_sin_wave(frequency, amplitude=0.5):
    """
    Generate a sine wave of a given frequency and amplitude.
    :param frequency: Frequency of the sine wave in Hz
    :param amplitude: Amplitude of the sine wave (0.0 to 1.0)
    :return: Numpy array containing the sine wave samples
    """
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

# SQUARE WAVES
def generate_square_wave(frequency, amplitude=0.5):
    """
    Generate a square wave of a given frequency and amplitude.
    :param frequency: Frequency of the square wave in Hz
    :param amplitude: Amplitude of the square wave (0.0 to 1.0)
    :return: Numpy array containing the square wave samples
    """
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

# SAWTOOTH WAVES
def generate_sawtooth_wave(frequency, amplitude=0.5):
    """
    Generate a sawtooth wave of a given frequency and amplitude.
    :param frequency: Frequency of the sawtooth wave in Hz
    :param amplitude: Amplitude of the sawtooth wave (0.0 to 1.0)
    :return: Numpy array containing the sawtooth wave samples"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    return amplitude * (2 * (t * frequency - np.floor(t * frequency + 0.5)))

def create_audio_file(filepath, waveform_type, frequency, amplitude=0.5):
    """
    Create an audio file with the specified waveform type, frequency, and amplitude.
    :param filepath: Path to save the audio file
    :param waveform_type: Type of waveform ('sine', 'square', 'sawtooth')
    :param frequency: Frequency of the waveform in Hz
    :param amplitude: Amplitude of the waveform (0.0 to 1.0)
    :return: None"""
    if waveform_type == 'sine':
        sf.write(filepath, generate_sin_wave(frequency, amplitude), SAMPLE_RATE)
    elif waveform_type == 'square':
        sf.write(filepath, generate_square_wave(frequency, amplitude), SAMPLE_RATE)
    elif waveform_type == 'sawtooth':
        sf.write(filepath, generate_sawtooth_wave(frequency, amplitude), SAMPLE_RATE)
    else:
        raise ValueError("Unsupported waveform type. Use 'sine', 'square', or 'sawtooth'.")


def generate_waveform(waveform_type, frequency, amplitude):
    if waveform_type == 'sine':
        return generate_sin_wave(frequency, amplitude)
    elif waveform_type == 'square':
        return generate_square_wave(frequency, amplitude)
    elif waveform_type == 'sawtooth':
        return generate_sawtooth_wave(frequency, amplitude)
    else:
        raise ValueError("Unsupported waveform type. Use 'sine', 'square', or 'sawtooth'.")


def cqt_max_frequency():
    return FMIN * 2 ** ((N_BINS - 1) / BINS_PER_OCTAVE)


def sample_frequency_for_action(rng, action, parameter):
    min_frequency = FMIN
    max_frequency = min(cqt_max_frequency(), SAMPLE_RATE / 2)

    if action == 'pitch_change':
        pitch_factor = 2 ** (parameter / 12)
        min_frequency = max(min_frequency, FMIN / pitch_factor)
        max_frequency = min(max_frequency, cqt_max_frequency() / pitch_factor, (SAMPLE_RATE / 2) / pitch_factor)

    if min_frequency >= max_frequency:
        raise ValueError(
            f"No valid frequency range for action={action}, parameter={parameter}."
        )

    return rng.uniform(min_frequency, max_frequency)


def generate_data():
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
                create_audio_file(filename, waveform_type, frequency, amplitude)
                print(f"Generated {filename}")

def generate_training_data(seed=42):
    input_spectrograms = []
    output_spectrograms = []
    action_vectors = []
    metadata = []  # To store metadata about the generated samples (e.g., waveform type, frequency, amplitude, action applied)
    index = 0

    rng = np.random.default_rng(seed)

    for waveform_type in ['sine', 'square', 'sawtooth']:
        for _ in range(1000):  # Generate 1000 samples for each waveform type
            amplitude = rng.uniform(0.1, 1.0)  # Random amplitude between 0.1 and 1.0

            # Choose a random action and parameter for the audio transformation
            action = rng.choice(['no_action', 'gain', 'pitch_change', 'low_pass', 'high_pass'])
            if action == 'gain':
                parameter = rng.uniform(-12, 12)  # Gain in dB
            elif action == 'pitch_change':
                parameter = int(rng.integers(-12, 13))  # Pitch change in semitones
            elif action in ['low_pass', 'high_pass']:
                parameter = rng.uniform(20, 20000)  # Cutoff frequency in Hz
            else:
                parameter = None  # No parameter needed for 'no_action'

            frequency = sample_frequency_for_action(rng, action, parameter)

            audio = generate_waveform(waveform_type, frequency, amplitude)
            spectogram = audio_to_cqt(audio)

            # Apply the action to the audio and get the action vector and the modified audio spectrogram
            pitch_shifted_frequency = None
            if action == 'pitch_change':
                pitch_factor = 2 ** (parameter / 12)
                pitch_shifted_frequency = frequency * pitch_factor
                
            modified_audio, action_vector = apply_action(audio, action, parameter)
            modified_spectrogram = audio_to_cqt(modified_audio)

            # Append the input spectrogram, output spectrogram, action vector, and metadata to the respective lists
            input_spectrograms.append(spectogram)
            output_spectrograms.append(modified_spectrogram)
            action_vectors.append(action_vector)
            metadata.append({
                'id': index,
                'waveform_type': waveform_type,
                'frequency': frequency,
                'pitch_shifted_frequency': pitch_shifted_frequency,
                'amplitude': amplitude,
                'action': action,
                'parameter': parameter,
                'sample_rate': SAMPLE_RATE,
                'duration': DURATION,
                'hop_length': HOP_LENGTH,
                'bins_per_octave': BINS_PER_OCTAVE,
                'n_bins': N_BINS,
                'fmin': FMIN,
                'cqt_db_floor': CQT_DB_FLOOR,
                'seed': seed
            })

            index += 1

            print(f"Data generation progress: {((index / (3 * 1000)) * 100):.2f}%")

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

    output_dir = ROOT / "data/synthetic/v001"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the datasets and metadata to disk using np.savez
    print(f"Saving datasets to {output_dir}...")

    np.savez_compressed(
        output_dir / "train.npz",
        input_spectrograms=train_inputs.astype(np.float32),
        target_spectrograms=train_outputs.astype(np.float32),
        action_vectors=train_actions.astype(np.float32),
    )

    np.savez_compressed(
        output_dir / "val.npz",
        input_spectrograms=val_inputs.astype(np.float32),
        target_spectrograms=val_outputs.astype(np.float32),
        action_vectors=val_actions.astype(np.float32),
    )

    np.savez_compressed(
        output_dir / "test.npz",
        input_spectrograms=test_inputs.astype(np.float32),
        target_spectrograms=test_outputs.astype(np.float32),
        action_vectors=test_actions.astype(np.float32),
    )

    print(f"Datasets saved successfully in {output_dir}.")

    print(f"Saving metadata to JSONL files in {output_dir}...")

    # Write metadata to a JSONL file
    with open(output_dir / "metadata_train.jsonl", "w") as f:
        for entry in train_metadata: 
            f.write(json.dumps(entry) + "\n")

    with open(output_dir / "metadata_val.jsonl", "w") as f:
        for entry in val_metadata: 
            f.write(json.dumps(entry) + "\n")

    with open(output_dir / "metadata_test.jsonl", "w") as f:
        for entry in test_metadata: 
            f.write(json.dumps(entry) + "\n")

    print(f"Metadata saved successfully in {output_dir}.")
    print(f"Data generation completed. Total samples generated: {len(input_spectrograms)}")

if __name__ == "__main__":
    generate_training_data()