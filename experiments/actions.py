from scipy.signal import butter, lfilter
from experiments.generate_synthetic_audio import SAMPLE_RATE, generate_sin_wave
import librosa

def gain(audio, gain_db):
    """
    Apply gain to the audio data.
    :param audio: Numpy array containing audio samples
    :param gain_db: Gain in decibels (positive for amplification, negative for attenuation)
    :return: Numpy array containing the modified audio samples
    """
    gain_factor = 10 ** (gain_db / 20)
    return audio * gain_factor

def change_pitch(audio, semitones):
    """
    Change the pitch of the audio data by a specified number of semitones.
    :param audio: Numpy array containing audio samples
    :param semitones: Number of semitones to shift the pitch (positive for higher pitch, negative for lower pitch)
    :return: Numpy array containing the modified audio samples
    """
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=semitones)

def low_pass(audio, cutoff):
    x = butter(N=5, Wn=cutoff, fs=SAMPLE_RATE, btype="low", analog=False)
    y = lfilter(x[0], x[1], audio)
    return y

def high_pass(audio, cutoff):
    x = butter(N=5, Wn=cutoff, fs=SAMPLE_RATE, btype="high", analog=False)
    y = lfilter(x[0], x[1], audio)
    return y