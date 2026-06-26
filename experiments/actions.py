from scipy.signal import butter, sosfilt
from experiments.config import SAMPLE_RATE
import librosa
import numpy as np

"""
Representation vector shape for actions:
Use one hot encoding for the type of action (no action, gain, pitch change, low pass filter, high pass filter) and then additional parameters for each action. [gain_normalized, semitones_normalized, cutoff_frequency_normalized]:
- No Action: [1, 0, 0, 0, 0, 0, 0, 0]
- Gain: [0, 1, 0, 0, 0, gain_db_normalized, 0, 0]
- Pitch Change: [0, 0, 1, 0, 0, 0, semitones_normalized, 0]
- Low Pass Filter: [0, 0, 0, 1, 0, 0, 0, cutoff_frequency_normalized]
- High Pass Filter: [0, 0, 0, 0, 1, 0, 0, cutoff_frequency_normalized]
"""

def encode_action(action, parameter):
    if action == 'no_action':
        return [1, 0, 0, 0, 0, 0, 0, 0]
    elif action == 'gain':
        db = min(12, max(-12, parameter))
        normalized_parameter = (db + 12) / 24 * 2 - 1
        return [0, 1, 0, 0, 0, normalized_parameter, 0, 0]
    elif action == 'pitch_change':
        semitones = min(12, max(-12, parameter))
        normalized_parameter = (semitones + 12) / 24 * 2 - 1
        return [0, 0, 1, 0, 0, 0, normalized_parameter, 0]
    elif action in ['low_pass', 'high_pass']:
        log_cutoff = np.log10(parameter)
        normalized_parameter = (log_cutoff - np.log10(20)) / (np.log10(20000) - np.log10(20))
        if action == 'low_pass':
            return [0, 0, 0, 1, 0, 0, 0, normalized_parameter]
        return [0, 0, 0, 0, 1, 0, 0, normalized_parameter]
    else:
        raise ValueError("Unsupported action type. Use 'no_action', 'gain', 'pitch_change', 'low_pass', or 'high_pass'.")


def apply_action(audio, action, parameter):
    """
    Apply the specified action to the audio data.
    :param audio: Numpy array containing audio samples
    :param action: String specifying the type of action ('no_action', 'gain', 'pitch_change', 'low_pass', 'high_pass')
    :param parameter: Parameter for the action (e.g., gain in dB, semitones for pitch change, cutoff frequency for filters)
    :return: Tuple containing the modified audio samples and the action vector
    """
    if action == 'no_action':
        return audio, encode_action(action, parameter)
    elif action == 'gain':
        # Normalize gain parameter to a range of -1 to 1 for representation vector
        db = min(12, max(-12, parameter))  # Clamp gain to [-12 dB, 12 dB]

        return gain(audio, db), encode_action(action, db)
    elif action == 'pitch_change':
        # Normalize semitones parameter to a range of -1 to 1 for representation vector
        semitones = min(12, max(-12, parameter))  # Clamp semitones to [-12, 12]
        
        return change_pitch(audio, semitones), encode_action(action, semitones)
    elif action == 'low_pass':
        return low_pass(audio, parameter), encode_action(action, parameter)
    elif action == 'high_pass':
        return high_pass(audio, parameter), encode_action(action, parameter)
    else:
        raise ValueError("Unsupported action type. Use 'no_action', 'gain', 'pitch_change', 'low_pass', or 'high_pass'.")

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
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=semitones, scale=True) # scale=True ensures that the duration and volume of the audio remains the same after pitch shifting

def low_pass(audio, cutoff):
    nyquist = 0.5 * SAMPLE_RATE

    if not 20 <= cutoff <= nyquist:
        raise ValueError("Cutoff frequency is out of bounds.")

    sos = butter(N=5, Wn=cutoff, fs=SAMPLE_RATE, btype='low', analog=False, output='sos')
    return sosfilt(sos, audio)

def high_pass(audio, cutoff):
    nyquist = 0.5 * SAMPLE_RATE

    if not 20 <= cutoff <= nyquist:
        raise ValueError("Cutoff frequency is out of bounds.")

    sos = butter(N=5, Wn=cutoff, fs=SAMPLE_RATE, btype='high', analog=False, output='sos')
    return sosfilt(sos, audio)
