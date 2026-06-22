import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from experiments.actions import change_pitch, gain, high_pass, low_pass
from experiments.audio_to_spectogram import audio_to_spectrogram
from experiments.config import DURATION, HOP_LENGTH, SAMPLE_RATE


def generate_sin_wave(frequency, amplitude=0.5):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def plot_spectrogram_pair(first, second, first_title, second_title):
    vmin = min(first.min(), second.min())
    vmax = max(first.max(), second.max())

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        first,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(format="%+.2f dB")
    plt.title(first_title)

    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        second,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(format="%+.2f dB")
    plt.title(second_title)

    plt.tight_layout()
    plt.show()


def main():
    audio = generate_sin_wave(440)
    spectrogram = audio_to_spectrogram(audio)
    print("Mel Spectrogram shape:", spectrogram.shape)

    audio_gained = gain(audio, gain_db=6)
    spectrogram_gained = audio_to_spectrogram(audio_gained)
    plot_spectrogram_pair(
        spectrogram,
        spectrogram_gained,
        "Mel Spectrogram",
        "Mel Spectrogram after Gain",
    )

    audio_pitch_changed = change_pitch(audio, semitones=12)
    spectrogram_pitch_changed = audio_to_spectrogram(audio_pitch_changed)
    plot_spectrogram_pair(
        spectrogram,
        spectrogram_pitch_changed,
        "Mel Spectrogram",
        "Mel Spectrogram after Pitch Change",
    )

    low_frequency = 100
    high_frequency = 2000
    filter_cutoff = 1000
    mixed_audio = (
        generate_sin_wave(low_frequency, amplitude=0.35)
        + generate_sin_wave(high_frequency, amplitude=0.35)
    )
    spectrogram_mixed = audio_to_spectrogram(mixed_audio)

    audio_low_passed = low_pass(mixed_audio, cutoff=filter_cutoff)
    spectrogram_low_passed = audio_to_spectrogram(audio_low_passed)
    plot_spectrogram_pair(
        spectrogram_mixed,
        spectrogram_low_passed,
        "Mixed Signal Mel Spectrogram",
        "Mel Spectrogram after Low Pass Filter",
    )

    audio_high_passed = high_pass(mixed_audio, cutoff=filter_cutoff)
    spectrogram_high_passed = audio_to_spectrogram(audio_high_passed)
    plot_spectrogram_pair(
        spectrogram_mixed,
        spectrogram_high_passed,
        "Mixed Signal Mel Spectrogram",
        "Mel Spectrogram after High Pass Filter",
    )


if __name__ == "__main__":
    main()
