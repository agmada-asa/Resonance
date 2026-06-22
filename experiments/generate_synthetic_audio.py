import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 44100 # Hz
DURATION = 2  # seconds

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

if __name__ == "__main__":
    generate_data()