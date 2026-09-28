from pathlib import Path
import json
import math
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from cqt_nsgt_pytorch import CQT_nsgt
except ImportError as error:
    raise ImportError(
        'This experiment requires cqt-nsgt-pytorch==0.0.10. Install it with: '
        '.venv/bin/pip install cqt-nsgt-pytorch==0.0.10'
    ) from error

from resonance.config import AudioConfig  # noqa: E402
from resonance.data.synthetic import WaveformSynthesizer  # noqa: E402


SAMPLE_RATE = 44100
DURATIONS = [0.5, 2.0]
NUM_OCTAVES = 10
BINS_PER_OCTAVE = 12
MODES = ['critical', 'matrix_pow2']
SEED = 7

OUTPUT_DIR = ROOT / 'build/cq_nsgt_reconstruction'
MINIMUM_PASSING_SNR = 60.0


def create_transform(duration, mode):
    audio_length = int(SAMPLE_RATE * duration)

    # cqt-nsgt-pytorch 0.0.10 uses an integer NumPy output with an infinite
    # clip bound. NumPy 2 rejects that cast, so replace infinity with the
    # largest value of the integer output only while the filter bank is built.
    original_clip = np.clip

    def numpy_2_compatible_clip(values, minimum=None, maximum=None, out=None, **kwargs):
        if (
            out is not None
            and np.issubdtype(out.dtype, np.integer)
            and maximum is not None
            and np.isinf(maximum)
        ):
            maximum = np.iinfo(out.dtype).max
        return original_clip(
            values,
            minimum,
            maximum,
            out=out,
            **kwargs,
        )

    np.clip = numpy_2_compatible_clip
    try:
        transform = CQT_nsgt(
            NUM_OCTAVES,
            BINS_PER_OCTAVE,
            mode=mode,
            fs=SAMPLE_RATE,
            audio_len=audio_length,
            device='cpu',
            dtype=torch.float32,
        )
    finally:
        np.clip = original_clip

    return transform


def create_test_signals(duration):
    config = AudioConfig(
        sample_rate=SAMPLE_RATE,
        duration=duration,
    )
    synthesizer = WaveformSynthesizer(config)
    rng = np.random.default_rng(SEED)

    return {
        'sine_110': synthesizer.generate_sin_wave(110.0, 0.25),
        'sine_440': synthesizer.generate_sin_wave(440.0, 0.25),
        'square_220': synthesizer.generate_square_wave(220.0, 0.25),
        'sawtooth_220': synthesizer.generate_sawtooth_wave(220.0, 0.25),
        'chord_220': synthesizer.generate_chords('sine', 220.0, 0.25),
        'noise': rng.normal(0, 0.1, int(SAMPLE_RATE * duration)),
    }


def forward_and_inverse(transform, audio):
    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        coefficients = transform.fwd(audio_tensor)
        reconstructed_audio = transform.bwd(coefficients)

    return coefficients, reconstructed_audio.squeeze().cpu().numpy()


def calculate_snr(target, prediction):
    signal = np.sum(np.square(target.astype(np.float64)))
    error = np.sum(
        np.square(
            target.astype(np.float64)
            - prediction.astype(np.float64)
        )
    )
    return float(10 * math.log10((signal + 1e-20) / (error + 1e-20)))


def calculate_rms_ratio(target, prediction):
    target_rms = np.sqrt(np.mean(np.square(target.astype(np.float64))))
    prediction_rms = np.sqrt(
        np.mean(np.square(prediction.astype(np.float64)))
    )
    return float(prediction_rms / max(target_rms, 1e-12))


def calculate_envelope_modulation(audio):
    envelope = np.abs(hilbert(audio.astype(np.float64)))
    return float(np.std(envelope) / max(np.mean(envelope), 1e-12))


def describe_coefficients(coefficients):
    if isinstance(coefficients, list):
        shapes = [list(values.shape) for values in coefficients]
        total_values = sum(values.numel() for values in coefficients)
        return {
            'format': 'multi_resolution_list',
            'shapes': shapes,
            'total_complex_values': total_values,
        }

    return {
        'format': 'matrix',
        'shape': list(coefficients.shape),
        'total_complex_values': coefficients.numel(),
    }


def save_listening_sample(directory, target, prediction):
    directory.mkdir(parents=True, exist_ok=True)

    # Use one safety scale so the target and reconstruction remain comparable
    peak = max(np.max(np.abs(target)), np.max(np.abs(prediction)))
    shared_scale = min(1.0, 0.8 / peak) if peak > 0 else 1.0

    sf.write(
        directory / 'target.wav',
        (target * shared_scale).astype(np.float32),
        SAMPLE_RATE,
        subtype='FLOAT',
    )
    sf.write(
        directory / 'reconstructed.wav',
        (prediction * shared_scale).astype(np.float32),
        SAMPLE_RATE,
        subtype='FLOAT',
    )

    return float(shared_scale)


def run_reconstruction_test():
    results = []

    print('CQ-NSGT oracle reconstruction test:')
    print('duration  mode          signal          snr_db  rms_ratio  envelope_mod')

    for duration in DURATIONS:
        signals = create_test_signals(duration)

        for mode in MODES:
            transform = create_transform(duration, mode)

            for signal_name, target_audio in signals.items():
                target_audio = target_audio.astype(np.float32)
                coefficients, reconstructed_audio = forward_and_inverse(
                    transform,
                    target_audio,
                )

                result = {
                    'duration': duration,
                    'mode': mode,
                    'signal': signal_name,
                    'snr': calculate_snr(target_audio, reconstructed_audio),
                    'maximum_error': float(
                        np.max(np.abs(target_audio - reconstructed_audio))
                    ),
                    'rms_ratio': calculate_rms_ratio(
                        target_audio,
                        reconstructed_audio,
                    ),
                    'envelope_modulation': calculate_envelope_modulation(
                        reconstructed_audio
                    ),
                    'coefficients': describe_coefficients(coefficients),
                }
                result['passed'] = result['snr'] >= MINIMUM_PASSING_SNR
                results.append(result)

                print(
                    f'{duration:>8.1f}  '
                    f'{mode:<12}  '
                    f'{signal_name:<14}  '
                    f'{result["snr"]:>6.2f}  '
                    f'{result["rms_ratio"]:>9.6f}  '
                    f'{result["envelope_modulation"]:>12.6f}'
                )

                if duration == 2.0 and signal_name in ['sine_110', 'sawtooth_220']:
                    listening_dir = OUTPUT_DIR / 'listening_samples' / mode / signal_name
                    result['listening_shared_scale'] = save_listening_sample(
                        listening_dir,
                        target_audio,
                        reconstructed_audio,
                    )
                    result['listening_directory'] = str(listening_dir)

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_reconstruction_test()

    critical_results = [
        result for result in results if result['mode'] == 'critical'
    ]
    matrix_results = [
        result for result in results if result['mode'] == 'matrix_pow2'
    ]
    summary = {
        'minimum_passing_snr': MINIMUM_PASSING_SNR,
        'all_critical_passed': all(result['passed'] for result in critical_results),
        'all_matrix_passed': all(result['passed'] for result in matrix_results),
        'minimum_critical_snr': min(result['snr'] for result in critical_results),
        'minimum_matrix_snr': min(result['snr'] for result in matrix_results),
        'results': results,
    }

    with (OUTPUT_DIR / 'results.json').open('w') as f:
        json.dump(summary, f, indent=2)

    print('\nSummary:')
    print(
        f'Critical representation passed: '
        f'{summary["all_critical_passed"]} '
        f'(minimum SNR {summary["minimum_critical_snr"]:.2f} dB)'
    )
    print(
        f'Matrix representation passed: '
        f'{summary["all_matrix_passed"]} '
        f'(minimum SNR {summary["minimum_matrix_snr"]:.2f} dB)'
    )
    print(f'Saved experiment to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
