import librosa
from experiments.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


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
