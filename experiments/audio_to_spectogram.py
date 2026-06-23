import numpy as np
import librosa

from experiments.config import BINS_PER_OCTAVE, CQT_DB_FLOOR, FMIN, HOP_LENGTH, N_BINS, N_FFT, N_MELS, SAMPLE_RATE


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

def audio_to_cqt(audio, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_BINS, fmin=FMIN):
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
    
    # Compute the CQT spectrogram
    cqt_spectrogram = librosa.cqt(y=audio, sr=sample_rate, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins, fmin=fmin)
    # Convert to decibels
    cqt_spectrogram_db = librosa.amplitude_to_db(
        np.abs(cqt_spectrogram),
        ref=1.0,
        top_db=None,
    )
    return np.maximum(cqt_spectrogram_db, CQT_DB_FLOOR)
