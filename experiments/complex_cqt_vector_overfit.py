from pathlib import Path
import json
import math
import random
import sys
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from resonance.actions import AudioActionProcessor  # noqa: E402
from resonance.config import AudioConfig  # noqa: E402
from resonance.data.synthetic import WaveformSynthesizer  # noqa: E402


# Audio configuration
SAMPLE_RATE = 44100
DURATION = 0.5
HOP_LENGTH = 512
BINS_PER_OCTAVE = 12
N_BINS = 113
FMIN = 32.70

# Vector and model configuration
COMPRESSION_ALPHA = 10.0
HIDDEN_CHANNELS = 48
LEARNING_RATE = 3e-3
NUM_STEPS = 1000
SEED = 7

# Oracle inverse CQT sweep configuration
SWEEP_DURATIONS = [0.5, 2.0]
SWEEP_HOP_LENGTHS = [128, 256, 512]
SWEEP_BINS_PER_OCTAVE = [12, 24, 36]
SWEEP_FREQUENCY = 110.0

OUTPUT_DIR = ROOT / 'build/complex_cqt_vector_overfit'


class ComplexCQTVectorCodec:
    def __init__(self, compression_alpha=COMPRESSION_ALPHA, epsilon=1e-12):
        self.compression_alpha = compression_alpha
        self.epsilon = epsilon

    def encode(self, coefficients):
        # Get the magnitude and compress it so loud and quiet CQT values are closer together
        magnitude = np.abs(coefficients)
        compressed_magnitude = np.log1p(self.compression_alpha * magnitude)

        # Turn each complex coefficient into a unit direction that holds its phase
        direction = np.zeros_like(coefficients, dtype=np.complex64)
        nonzero = magnitude > self.epsilon
        direction[nonzero] = coefficients[nonzero] / magnitude[nonzero]

        # Store the phase direction and compressed magnitude as two real vector channels
        compressed_coefficients = compressed_magnitude * direction
        return np.stack(
            [compressed_coefficients.real, compressed_coefficients.imag],
            axis=0,
        ).astype(np.float32)

    def decode(self, vectors):
        if vectors.shape[0] != 2:
            raise ValueError(
                f'Expected 2 vector channels, got shape {vectors.shape}.'
            )

        # Rebuild the compressed complex coefficient from the two vector channels
        compressed_coefficients = (
            vectors[0].astype(np.float64)
            + 1j * vectors[1].astype(np.float64)
        )
        compressed_magnitude = np.abs(compressed_coefficients)

        # Reverse the log compression to recover the original CQT magnitude
        magnitude = np.expm1(compressed_magnitude) / self.compression_alpha

        # Recover the phase direction and combine it with the original magnitude
        direction = np.zeros_like(compressed_coefficients, dtype=np.complex128)
        nonzero = compressed_magnitude > self.epsilon
        direction[nonzero] = (
            compressed_coefficients[nonzero]
            / compressed_magnitude[nonzero]
        )
        return (magnitude * direction).astype(np.complex64)


class TinyComplexCQTModel(nn.Module):
    def __init__(self, action_vector_size=7):
        super().__init__()

        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )

        self.network = nn.Sequential(
            nn.Conv2d(2 + 16, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(HIDDEN_CHANNELS, 2, kernel_size=1),
        )

    def forward(self, vectors, action_vector):
        # Encode the action and repeat it across every CQT frequency and time position
        action_embedding = self.action_encoder(action_vector)
        action_embedding = action_embedding[:, :, None, None]
        action_embedding = action_embedding.expand(
            -1,
            -1,
            vectors.shape[-2],
            vectors.shape[-1],
        )

        # Predict a vector change and add it to the input complex CQT vectors
        model_input = torch.cat([vectors, action_embedding], dim=1)
        predicted_delta = self.network(model_input)
        return vectors + predicted_delta


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def create_audio_config(duration=DURATION):
    return AudioConfig(
        sample_rate=SAMPLE_RATE,
        duration=duration,
        hop_length=HOP_LENGTH,
        bins_per_octave=BINS_PER_OCTAVE,
        n_bins=N_BINS,
        fmin=FMIN,
    )


def get_n_bins(bins_per_octave):
    # Keep the same CQT frequency range when increasing the bins per octave
    n_bins = round(N_BINS * bins_per_octave / BINS_PER_OCTAVE)

    # Remove top bins until the full filter fits under Nyquist
    while n_bins > 0:
        frequencies = librosa.cqt_frequencies(
            n_bins,
            fmin=FMIN,
            bins_per_octave=bins_per_octave,
        )
        _, cutoff = librosa.filters.wavelet_lengths(
            freqs=frequencies,
            sr=SAMPLE_RATE,
        )
        if cutoff < SAMPLE_RATE / 2:
            return n_bins
        n_bins -= 1

    raise ValueError('Could not create a CQT filter bank below Nyquist.')


def audio_to_complex_cqt(
    audio,
    hop_length=HOP_LENGTH,
    bins_per_octave=BINS_PER_OCTAVE,
    n_bins=N_BINS,
):
    return librosa.cqt(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        n_bins=n_bins,
        fmin=FMIN,
    ).astype(np.complex64)


def complex_cqt_to_audio(
    coefficients,
    duration=DURATION,
    hop_length=HOP_LENGTH,
    bins_per_octave=BINS_PER_OCTAVE,
):
    return librosa.icqt(
        coefficients,
        sr=SAMPLE_RATE,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        fmin=FMIN,
        length=int(SAMPLE_RATE * duration),
    ).astype(np.float32)


def build_dataset(codec, audio_config):
    action_processor = AudioActionProcessor(audio_config)
    waveform_synthesizer = WaveformSynthesizer(audio_config)

    input_vectors = []
    target_vectors = []
    action_vectors = []
    input_audio = []
    target_audio = []
    target_coefficients = []
    metadata = []

    # Begin with phase-preserving actions before testing pitch changes
    actions = [
        ('no_action', None),
        ('gain', -6.0),
        ('gain', 6.0),
    ]

    for frequency in [110.0, 220.0, 440.0, 880.0]:
        for action, parameter in actions:
            before_audio = waveform_synthesizer.generate_sin_wave(
                frequency,
                amplitude=0.25,
            ).astype(np.float32)
            after_audio, action_vector = action_processor.apply_action(
                before_audio,
                action,
                parameter,
            )
            after_audio = after_audio.astype(np.float32)

            before_cqt = audio_to_complex_cqt(before_audio)
            after_cqt = audio_to_complex_cqt(after_audio)

            input_vectors.append(codec.encode(before_cqt))
            target_vectors.append(codec.encode(after_cqt))
            action_vectors.append(np.array(action_vector, dtype=np.float32))
            input_audio.append(before_audio)
            target_audio.append(after_audio)
            target_coefficients.append(after_cqt)
            metadata.append(
                {
                    'frequency': frequency,
                    'action': action,
                    'parameter': parameter,
                }
            )

    return {
        'input_vectors': np.stack(input_vectors),
        'target_vectors': np.stack(target_vectors),
        'action_vectors': np.stack(action_vectors),
        'input_audio': np.stack(input_audio),
        'target_audio': np.stack(target_audio),
        'target_coefficients': np.stack(target_coefficients),
        'metadata': metadata,
    }


def train_model(dataset, device):
    input_vectors = torch.from_numpy(dataset['input_vectors']).to(device)
    target_vectors = torch.from_numpy(dataset['target_vectors']).to(device)
    action_vectors = torch.from_numpy(dataset['action_vectors']).to(device)

    model = TinyComplexCQTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    initial_loss = None
    final_loss = None
    print_every = NUM_STEPS // 10

    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        predicted_vectors = model(input_vectors, action_vectors)
        loss = nn.functional.mse_loss(predicted_vectors, target_vectors)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if initial_loss is None:
            initial_loss = final_loss

        if step == 0 or (step + 1) % print_every == 0:
            print(
                f'Step [{step + 1}/{NUM_STEPS}] '
                f'Vector loss: {final_loss:.10f}'
            )

    model.eval()
    with torch.no_grad():
        predicted_vectors = model(input_vectors, action_vectors)

    return model, predicted_vectors.cpu().numpy(), initial_loss, final_loss


def calculate_relative_error(target, prediction):
    target_size = max(np.linalg.norm(target), 1e-12)
    return float(np.linalg.norm(target - prediction) / target_size)


def calculate_snr(target, prediction):
    signal = np.sum(np.square(target.astype(np.float64)))
    error = np.sum(
        np.square(
            target.astype(np.float64)
            - prediction.astype(np.float64)
        )
    )
    return float(10 * math.log10((signal + 1e-20) / (error + 1e-20)))


def calculate_phase_similarity(target, prediction):
    magnitude = np.abs(target).astype(np.float64)
    active = magnitude > 1e-7
    phase_difference = (
        np.angle(prediction[active])
        - np.angle(target[active])
    )
    return float(
        np.sum(magnitude[active] * np.cos(phase_difference))
        / np.sum(magnitude[active])
    )


def calculate_envelope_modulation(audio):
    # A steady sine should have a nearly flat envelope, so variation measures rattle
    envelope = np.abs(hilbert(audio.astype(np.float64)))
    return float(np.std(envelope) / max(np.mean(envelope), 1e-12))


def calculate_rms_ratio(target, prediction):
    target_rms = np.sqrt(np.mean(np.square(target.astype(np.float64))))
    prediction_rms = np.sqrt(
        np.mean(np.square(prediction.astype(np.float64)))
    )
    return float(prediction_rms / max(target_rms, 1e-12))


def run_oracle_reconstruction_sweep():
    results = []

    print('\nOracle inverse CQT sweep:')
    print('duration  hop  bins/oct  n_bins  snr_db  rms_ratio  envelope_mod')

    for duration in SWEEP_DURATIONS:
        sample_count = int(SAMPLE_RATE * duration)
        time = np.linspace(0, duration, sample_count, endpoint=False)
        target_audio = 0.25 * np.sin(2 * np.pi * SWEEP_FREQUENCY * time)
        target_audio = target_audio.astype(np.float32)

        for hop_length in SWEEP_HOP_LENGTHS:
            for bins_per_octave in SWEEP_BINS_PER_OCTAVE:
                n_bins = get_n_bins(bins_per_octave)

                try:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter('always')
                        coefficients = audio_to_complex_cqt(
                            target_audio,
                            hop_length=hop_length,
                            bins_per_octave=bins_per_octave,
                            n_bins=n_bins,
                        )
                        reconstructed_audio = complex_cqt_to_audio(
                            coefficients,
                            duration=duration,
                            hop_length=hop_length,
                            bins_per_octave=bins_per_octave,
                        )

                    result = {
                        'duration': duration,
                        'hop_length': hop_length,
                        'bins_per_octave': bins_per_octave,
                        'n_bins': n_bins,
                        'cqt_shape': list(coefficients.shape),
                        'snr': calculate_snr(target_audio, reconstructed_audio),
                        'rms_ratio': calculate_rms_ratio(
                            target_audio,
                            reconstructed_audio,
                        ),
                        'envelope_modulation': calculate_envelope_modulation(
                            reconstructed_audio
                        ),
                        'warning_count': len(caught_warnings),
                    }
                    print(
                        f'{duration:>8.1f}  '
                        f'{hop_length:>3}  '
                        f'{bins_per_octave:>8}  '
                        f'{n_bins:>6}  '
                        f'{result["snr"]:>6.2f}  '
                        f'{result["rms_ratio"]:>9.3f}  '
                        f'{result["envelope_modulation"]:>12.4f}'
                    )
                except Exception as error:
                    result = {
                        'duration': duration,
                        'hop_length': hop_length,
                        'bins_per_octave': bins_per_octave,
                        'n_bins': n_bins,
                        'error': str(error),
                    }
                    print(
                        f'{duration:>8.1f}  '
                        f'{hop_length:>3}  '
                        f'{bins_per_octave:>8}  '
                        f'{n_bins:>6}  failed'
                    )

                results.append(result)

    successful_results = [result for result in results if 'error' not in result]
    best_result = max(successful_results, key=lambda result: result['snr'])
    return results, best_result


def save_best_oracle_reconstruction(best_result):
    duration = best_result['duration']
    hop_length = best_result['hop_length']
    bins_per_octave = best_result['bins_per_octave']
    n_bins = best_result['n_bins']

    sample_count = int(SAMPLE_RATE * duration)
    time = np.linspace(0, duration, sample_count, endpoint=False)
    target_audio = 0.25 * np.sin(2 * np.pi * SWEEP_FREQUENCY * time)
    target_audio = target_audio.astype(np.float32)

    coefficients = audio_to_complex_cqt(
        target_audio,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        n_bins=n_bins,
    )
    reconstructed_audio = complex_cqt_to_audio(
        coefficients,
        duration=duration,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
    )

    # Apply one scale to both files so the reconstruction level remains comparable
    peak = max(
        np.max(np.abs(target_audio)),
        np.max(np.abs(reconstructed_audio)),
    )
    shared_scale = min(1.0, 0.8 / peak) if peak > 0 else 1.0
    output_dir = OUTPUT_DIR / 'best_oracle_reconstruction'
    output_dir.mkdir(parents=True, exist_ok=True)

    sf.write(
        output_dir / 'target.wav',
        target_audio * shared_scale,
        SAMPLE_RATE,
        subtype='FLOAT',
    )
    sf.write(
        output_dir / 'oracle_complex_icqt.wav',
        reconstructed_audio * shared_scale,
        SAMPLE_RATE,
        subtype='FLOAT',
    )

    return output_dir, float(shared_scale)


def save_audio_comparison(dataset, oracle_audio, predicted_audio):
    sample_index = next(
        index
        for index, metadata in enumerate(dataset['metadata'])
        if metadata['frequency'] == 110.0
        and metadata['action'] == 'gain'
        and metadata['parameter'] == 6.0
    )

    audio = {
        'input.wav': dataset['input_audio'][sample_index],
        'target.wav': dataset['target_audio'][sample_index],
        'oracle_complex_icqt.wav': oracle_audio,
        'predicted_complex_icqt.wav': predicted_audio,
    }

    # Apply one shared safety scale so the files preserve their relative levels
    peak = max(np.max(np.abs(signal)) for signal in audio.values())
    shared_scale = min(1.0, 0.8 / peak) if peak > 0 else 1.0
    output_dir = OUTPUT_DIR / 'listening_sample'
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, signal in audio.items():
        sf.write(
            output_dir / filename,
            np.nan_to_num(signal * shared_scale).astype(np.float32),
            SAMPLE_RATE,
            subtype='FLOAT',
        )

    return sample_index, output_dir, float(shared_scale)


def main():
    set_seed()
    device = select_device()
    audio_config = create_audio_config()
    codec = ComplexCQTVectorCodec()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(codec, audio_config)

    # Check that the two vector channels can recover the original complex CQT
    first_cqt = dataset['target_coefficients'][0]
    decoded_cqt = codec.decode(codec.encode(first_cqt))
    codec_error = calculate_relative_error(first_cqt, decoded_cqt)

    print(f'Using device: {device}')
    print(f'Dataset shape: {dataset["input_vectors"].shape}')
    print(f'Codec relative error: {codec_error:.10e}')

    np.savez_compressed(
        OUTPUT_DIR / 'vectors.npz',
        input_vectors=dataset['input_vectors'],
        target_vectors=dataset['target_vectors'],
        action_vectors=dataset['action_vectors'],
    )

    model, predicted_vectors, initial_loss, final_loss = train_model(
        dataset,
        device,
    )
    torch.save(
        {'model_state_dict': model.state_dict()},
        OUTPUT_DIR / 'model.pth',
    )

    # Decode the model outputs back into normal complex CQT coefficients
    predicted_coefficients = np.stack(
        [codec.decode(vectors) for vectors in predicted_vectors]
    )
    complex_error = calculate_relative_error(
        dataset['target_coefficients'],
        predicted_coefficients,
    )
    phase_similarity = calculate_phase_similarity(
        dataset['target_coefficients'],
        predicted_coefficients,
    )

    sample_index = next(
        index
        for index, metadata in enumerate(dataset['metadata'])
        if metadata['frequency'] == 110.0
        and metadata['action'] == 'gain'
        and metadata['parameter'] == 6.0
    )
    oracle_audio = complex_cqt_to_audio(
        dataset['target_coefficients'][sample_index]
    )
    predicted_audio = complex_cqt_to_audio(
        predicted_coefficients[sample_index]
    )
    _, audio_dir, shared_scale = save_audio_comparison(
        dataset,
        oracle_audio,
        predicted_audio,
    )

    oracle_sweep, best_oracle_result = run_oracle_reconstruction_sweep()
    best_oracle_dir, best_oracle_scale = save_best_oracle_reconstruction(
        best_oracle_result
    )

    target_audio = dataset['target_audio'][sample_index]
    results = {
        'dataset_examples': len(dataset['metadata']),
        'vector_shape': list(dataset['input_vectors'].shape),
        'codec_complex_relative_error': codec_error,
        'initial_vector_loss': initial_loss,
        'final_vector_loss': final_loss,
        'loss_reduction': initial_loss / max(final_loss, 1e-20),
        'predicted_complex_relative_error': complex_error,
        'phase_similarity': phase_similarity,
        'oracle_icqt_vs_target_snr': calculate_snr(
            target_audio,
            oracle_audio,
        ),
        'predicted_icqt_vs_target_snr': calculate_snr(
            target_audio,
            predicted_audio,
        ),
        'predicted_icqt_vs_oracle_icqt_snr': calculate_snr(
            oracle_audio,
            predicted_audio,
        ),
        'audio_shared_scale': shared_scale,
        'audio_directory': str(audio_dir),
        'best_oracle_reconstruction': best_oracle_result,
        'best_oracle_audio_directory': str(best_oracle_dir),
        'best_oracle_audio_shared_scale': best_oracle_scale,
        'oracle_reconstruction_sweep': oracle_sweep,
    }

    with (OUTPUT_DIR / 'results.json').open('w') as f:
        json.dump(results, f, indent=2)

    print('\nResults:')
    print(json.dumps(results, indent=2))
    print(f'\nSaved experiment to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
