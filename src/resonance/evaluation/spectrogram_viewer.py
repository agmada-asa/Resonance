import librosa.display
import matplotlib.pyplot as plt

from resonance.actions import AudioActionProcessor
from resonance.config import AudioConfig, DEFAULT_CONFIG
from resonance.data.synthetic import WaveformSynthesizer
from resonance.features.spectrogram import SpectrogramTransformer


class SpectrogramViewer:
    def __init__(
        self,
        config: AudioConfig = DEFAULT_CONFIG,
        action_processor: AudioActionProcessor | None = None,
        spectrogram_transformer: SpectrogramTransformer | None = None,
        waveform_synthesizer: WaveformSynthesizer | None = None,
    ):
        self.config = config
        self.action_processor = action_processor or AudioActionProcessor(config)
        self.spectrogram_transformer = spectrogram_transformer or SpectrogramTransformer(config)
        self.waveform_synthesizer = waveform_synthesizer or WaveformSynthesizer(config)

    def generate_sin_wave(self, frequency, amplitude=0.5):
        return self.waveform_synthesizer.generate_sin_wave(frequency, amplitude)

    def plot_spectrogram_pair(self, first, second, first_title, second_title):
        vmin = min(first.min(), second.min())
        vmax = max(first.max(), second.max())

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(
            first,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
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
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            x_axis="time",
            y_axis="mel",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(format="%+.2f dB")
        plt.title(second_title)

        plt.tight_layout()
        plt.show()

    def run(self):
        audio = self.generate_sin_wave(440)
        spectrogram = self.spectrogram_transformer.audio_to_spectrogram(audio)
        print("Mel Spectrogram shape:", spectrogram.shape)

        audio_gained = self.action_processor.gain(audio, gain_db=6)
        spectrogram_gained = self.spectrogram_transformer.audio_to_spectrogram(audio_gained)
        self.plot_spectrogram_pair(
            spectrogram,
            spectrogram_gained,
            "Mel Spectrogram",
            "Mel Spectrogram after Gain",
        )

        audio_pitch_changed = self.action_processor.change_pitch(audio, semitones=12)
        spectrogram_pitch_changed = self.spectrogram_transformer.audio_to_spectrogram(audio_pitch_changed)
        self.plot_spectrogram_pair(
            spectrogram,
            spectrogram_pitch_changed,
            "Mel Spectrogram",
            "Mel Spectrogram after Pitch Change",
        )

        low_frequency = 100
        high_frequency = 2000
        filter_cutoff = 1000
        mixed_audio = (
            self.generate_sin_wave(low_frequency, amplitude=0.35)
            + self.generate_sin_wave(high_frequency, amplitude=0.35)
        )
        spectrogram_mixed = self.spectrogram_transformer.audio_to_spectrogram(mixed_audio)

        audio_low_passed = self.action_processor.low_pass(mixed_audio, cutoff=filter_cutoff)
        spectrogram_low_passed = self.spectrogram_transformer.audio_to_spectrogram(audio_low_passed)
        self.plot_spectrogram_pair(
            spectrogram_mixed,
            spectrogram_low_passed,
            "Mixed Signal Mel Spectrogram",
            "Mel Spectrogram after Low Pass Filter",
        )

        audio_high_passed = self.action_processor.high_pass(mixed_audio, cutoff=filter_cutoff)
        spectrogram_high_passed = self.spectrogram_transformer.audio_to_spectrogram(audio_high_passed)
        self.plot_spectrogram_pair(
            spectrogram_mixed,
            spectrogram_high_passed,
            "Mixed Signal Mel Spectrogram",
            "Mel Spectrogram after High Pass Filter",
        )


DEFAULT_SPECTROGRAM_VIEWER = SpectrogramViewer()


def generate_sin_wave(frequency, amplitude=0.5):
    return DEFAULT_SPECTROGRAM_VIEWER.generate_sin_wave(frequency, amplitude)


def plot_spectrogram_pair(first, second, first_title, second_title):
    return DEFAULT_SPECTROGRAM_VIEWER.plot_spectrogram_pair(first, second, first_title, second_title)
