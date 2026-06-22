import librosa
import numpy as np
from experiments.actions import change_pitch, gain, low_pass, high_pass
from experiments.generate_synthetic_audio import SAMPLE_RATE, generate_sin_wave
import matplotlib.pyplot as plt


N_FFT = 2048 # FFT window size
HOP_LENGTH = 512 # Hop length for STFT
N_MELS = 128 # Number of Mel bands

def audio_to_spectrogram(audio, sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Convert a numpy array of audio samples to a Mel spectrogram.
    :param audio: Numpy array containing audio samples
    :param sample_rate: Sample rate of the audio
    :param n_fft: FFT window size
    :param hop_length: Hop length for STFT
    :param n_mels: Number of Mel bands
    :return: Numpy array containing the Mel spectrogram
    """
    
    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
    return mel_spectrogram_db

if __name__ == "__main__":
    # Generate a synthetic sine wave audio signal
    frequency = 440  # A4 note
    audio = generate_sin_wave(frequency)

    # Convert the audio to a Mel spectrogram
    spectrogram = audio_to_spectrogram(audio)
    print("Mel Spectrogram shape:", spectrogram.shape) 

    # Apply gain to the audio signal
    audio_gained = gain(audio, gain_db=6)  # Increase volume by 6 dB

    # Convert the gained audio to a Mel spectrogram
    spectrogram_gained = audio_to_spectrogram(audio_gained)

    plt.figure(figsize=(10, 8))

    # Set the same vmin and vmax for both spectrograms to ensure they are on the same scale and the colorbars are comparable
    vmin = min(spectrogram.min(), spectrogram_gained.min())
    vmax = max(spectrogram.max(), spectrogram_gained.max())

    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mel Spectrogram')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(spectrogram_gained, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mel Spectrogram after Gain')

    plt.tight_layout()
    plt.show()

    # Apply pitch change to the audio signal
    audio_pitch_changed = change_pitch(audio, semitones=12)  # Shift pitch up by 12 semitones
    # Convert the pitch-changed audio to a Mel spectrogram
    spectrogram_pitch_changed = audio_to_spectrogram(audio_pitch_changed)

    plt.figure(figsize=(10, 8))

    # Set the same vmin and vmax for both spectrograms to ensure they are on the same scale and the colorbars are comparable
    vmin = min(spectrogram.min(), spectrogram_pitch_changed.min())
    vmax = max(spectrogram.max(), spectrogram_pitch_changed.max())

    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mel Spectrogram')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(spectrogram_pitch_changed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mel Spectrogram after Pitch Change')
    
    plt.tight_layout()
    plt.show()

    # Use a mixed signal so the filters have both low and high frequencies to act on.
    low_frequency = 100
    high_frequency = 2000
    filter_cutoff = 1000
    mixed_audio = (
        generate_sin_wave(low_frequency, amplitude=0.35)
        + generate_sin_wave(high_frequency, amplitude=0.35)
    )
    spectrogram_mixed = audio_to_spectrogram(mixed_audio)

    # Apply low pass filter. This should keep the 100 Hz tone and attenuate the 2000 Hz tone.
    audio_low_passed = low_pass(mixed_audio, cutoff=filter_cutoff)
    # Convert the low-passed audio to a Mel spectrogram
    spectrogram_low_passed = audio_to_spectrogram(audio_low_passed)

    # Set the same vmin and vmax for both spectrograms to ensure they are on the same scale and the colorbars are comparable
    vmin = min(spectrogram_mixed.min(), spectrogram_low_passed.min())
    vmax = max(spectrogram_mixed.max(), spectrogram_low_passed.max())
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram_mixed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mixed Signal Mel Spectrogram')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(spectrogram_low_passed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mel Spectrogram after Low Pass Filter')

    plt.tight_layout()
    plt.show()

    # Apply high pass filter. This should attenuate the 100 Hz tone and keep the 2000 Hz tone.
    audio_high_passed = high_pass(mixed_audio, cutoff=filter_cutoff)
    # Convert the high-passed audio to a Mel spectrogram
    spectrogram_high_passed = audio_to_spectrogram(audio_high_passed)

    # Set the same vmin and vmax for both spectrograms to ensure they are on the same scale and the colorbars are comparable
    vmin = min(spectrogram_mixed.min(), spectrogram_high_passed.min())
    vmax = max(spectrogram_mixed.max(), spectrogram_high_passed.max())

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram_mixed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mixed Signal Mel Spectrogram')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(spectrogram_high_passed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', vmin=vmin, vmax=vmax)
    plt.colorbar(format='%+.2f dB')
    plt.title('Mel Spectrogram after High Pass Filter')

    plt.tight_layout()
    plt.show()
