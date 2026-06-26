import librosa
import numpy as np

from resonance.config import AudioConfig, DEFAULT_CONFIG


class SpectrogramTransformer:
    def __init__(self, config: AudioConfig = DEFAULT_CONFIG):
        self.config = config

    def audio_to_spectrogram(self, audio, sample_rate=None, n_fft=None, hop_length=None, n_mels=None):
        """
        Convert a numpy array of audio samples to a Mel spectrogram.
        :param audio: Numpy array containing audio samples
        :param sample_rate: Sample rate of the audio
        :param n_fft: FFT window size
        :param hop_length: Hop length for STFT
        :param n_mels: Number of Mel bands
        :return: Numpy array containing the Mel spectrogram
        """
        sample_rate = self.config.sample_rate if sample_rate is None else sample_rate
        n_fft = self.config.n_fft if n_fft is None else n_fft
        hop_length = self.config.hop_length if hop_length is None else hop_length
        n_mels = self.config.n_mels if n_mels is None else n_mels

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
        return mel_spectrogram_db

    def audio_to_cqt(self, audio, sample_rate=None, hop_length=None, bins_per_octave=None, n_bins=None, fmin=None):
        """
        Convert a numpy array of audio samples to a Constant-Q Transform (CQT) spectrogram.
        :param audio: Numpy array containing audio samples
        :param sample_rate: Sample rate of the audio
        :param hop_length: Hop length for CQT
        :param bins_per_octave: Number of bins per octave
        :param n_bins: Total number of frequency bins
        :param fmin: Minimum frequency for the CQT
        :return: Numpy array containing the CQT spectrogram
        """
        sample_rate = self.config.sample_rate if sample_rate is None else sample_rate
        hop_length = self.config.hop_length if hop_length is None else hop_length
        bins_per_octave = self.config.bins_per_octave if bins_per_octave is None else bins_per_octave
        n_bins = self.config.n_bins if n_bins is None else n_bins
        fmin = self.config.fmin if fmin is None else fmin

        # Compute the CQT spectrogram
        cqt_spectrogram = librosa.cqt(y=audio, sr=sample_rate, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins, fmin=fmin)
        # Convert to decibels
        cqt_spectrogram_db = librosa.amplitude_to_db(
            np.abs(cqt_spectrogram),
            ref=1.0,
            top_db=None,
        )
        return np.maximum(cqt_spectrogram_db, self.config.cqt_db_floor)


DEFAULT_SPECTROGRAM_TRANSFORMER = SpectrogramTransformer()


def audio_to_spectrogram(audio, sample_rate=None, n_fft=None, hop_length=None, n_mels=None):
    return DEFAULT_SPECTROGRAM_TRANSFORMER.audio_to_spectrogram(audio, sample_rate, n_fft, hop_length, n_mels)


def audio_to_cqt(audio, sample_rate=None, hop_length=None, bins_per_octave=None, n_bins=None, fmin=None):
    return DEFAULT_SPECTROGRAM_TRANSFORMER.audio_to_cqt(audio, sample_rate, hop_length, bins_per_octave, n_bins, fmin)

