import math
import importlib

import numpy as np
import torch

from resonance.config import DEFAULT_CONFIG


class CQNSGTVectorCodec:
    def __init__(
        self,
        config=DEFAULT_CONFIG,
        compression_alpha=10.0,
        device='cpu',
    ):
        self.config = config
        self.sample_rate = config.sample_rate
        self.duration = config.duration
        self.audio_length = int(config.sample_rate * config.duration)
        self.bins_per_octave = config.bins_per_octave
        self.compression_alpha = compression_alpha
        self.device = torch.device(device)
        self.epsilon = 1e-12

        # CQ-NSGT works with complete octaves, so cover the configured range
        # from the minimum CQT frequency through to the Nyquist frequency
        frequency_range = (config.sample_rate / 2) / config.fmin
        self.num_octaves = math.ceil(math.log2(frequency_range))

        # Build one matched CQ-NSGT analysis and synthesis filter bank
        self.transform = self._create_transform()

    def _create_transform(self):
        try:
            from cqt_nsgt_pytorch import CQT_nsgt
        except ImportError as error:
            raise ImportError(
                'CQNSGTVectorCodec requires cqt-nsgt-pytorch==0.0.10.'
            ) from error

        # The package uses an infinite clip bound with an integer NumPy output.
        # NumPy 2 rejects that cast, so use the largest integer during setup.
        original_clip = np.clip

        # Its Hann window also requests float64 directly on the target device.
        # MPS has no float64 tensors, so construct the same window as float32.
        nsgfwin_module = importlib.import_module(
            'cqt_nsgt_pytorch.nsgfwin'
        )
        original_hannwin = nsgfwin_module.hannwin

        def mps_compatible_hannwin(length, device='cpu'):
            values = torch.arange(
                length,
                dtype=torch.float32,
                device=torch.device(device),
            )
            values *= torch.pi * 2 / length
            values = torch.cos(values)
            values += 1
            values *= 0.5
            return values

        def numpy_2_compatible_clip(
            values,
            minimum=None,
            maximum=None,
            out=None,
            **kwargs,
        ):
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
        nsgfwin_module.hannwin = mps_compatible_hannwin
        try:
            transform = CQT_nsgt(
                self.num_octaves,
                self.bins_per_octave,
                mode='critical',
                fs=self.sample_rate,
                audio_len=self.audio_length,
                device=str(self.device),
                dtype=torch.float32,
            )
        finally:
            np.clip = original_clip
            nsgfwin_module.hannwin = original_hannwin

        return transform

    def _prepare_audio(self, audio):
        # Convert numpy audio into the float tensor used by CQ-NSGT
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        else:
            audio = audio.to(dtype=torch.float32)

        # Add missing batch and mono channel dimensions
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(1)

        # The first version of the codec only handles mono fixed-length audio
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError(
                'Expected mono audio with shape [time], [batch, time], '
                'or [batch, 1, time].'
            )
        if audio.shape[-1] != self.audio_length:
            raise ValueError(
                f'Expected {self.audio_length} audio samples, '
                f'got {audio.shape[-1]}.'
            )

        return audio.to(self.device)

    def _encode_coefficients(self, coefficients):
        # Compress the coefficient magnitude into a range that is easier to learn
        magnitude = torch.abs(coefficients)
        compressed_magnitude = torch.log1p(
            self.compression_alpha * magnitude
        )

        # Keep phase as the direction of the complex coefficient
        direction = torch.zeros_like(coefficients)
        nonzero = magnitude > self.epsilon
        direction[nonzero] = coefficients[nonzero] / magnitude[nonzero]
        compressed_coefficients = compressed_magnitude * direction

        # The current audio pipeline is mono, so remove its single channel
        vectors = torch.stack(
            [compressed_coefficients.real, compressed_coefficients.imag],
            dim=2,
        )
        return vectors.squeeze(1)

    def _decode_vectors(self, vectors):
        if vectors.ndim != 4 or vectors.shape[1] != 2:
            raise ValueError(
                'Expected vectors with shape [batch, 2, frequency, time].'
            )

        # Join the two real vector channels back into one complex value
        compressed_coefficients = torch.complex(
            vectors[:, 0],
            vectors[:, 1],
        )
        # Reverse the log compression to recover the original magnitude
        compressed_magnitude = torch.abs(compressed_coefficients)
        magnitude = torch.expm1(compressed_magnitude) / self.compression_alpha

        # Recover the phase from the direction of the compressed vector
        direction = torch.zeros_like(compressed_coefficients)
        nonzero = compressed_magnitude > self.epsilon
        direction[nonzero] = (
            compressed_coefficients[nonzero]
            / compressed_magnitude[nonzero]
        )

        # Add the mono channel back for the CQ-NSGT inverse
        return (magnitude * direction).unsqueeze(1)

    def encode(self, coefficient_blocks):
        # Each block contains frequency bands that share one time resolution
        return [
            self._encode_coefficients(coefficients)
            for coefficients in coefficient_blocks
        ]

    def decode(self, vector_blocks):
        # Rebuild the matching list of complex CQ-NSGT coefficient blocks
        return [
            self._decode_vectors(vectors)
            for vectors in vector_blocks
        ]

    def audio_to_vectors(self, audio):
        # Analyse the audio into perfectly invertible multi-resolution coefficients
        audio = self._prepare_audio(audio)
        coefficient_blocks = self.transform.fwd(audio)

        # Convert the complex coefficients into two real model input channels
        return self.encode(coefficient_blocks)

    def vectors_to_audio(self, vector_blocks):
        # Convert the two vector channels back into complex coefficients
        coefficient_blocks = self.decode(vector_blocks)

        # The package changes the coefficient list during inversion
        coefficient_blocks = [
            coefficients.clone()
            for coefficients in coefficient_blocks
        ]

        # Use the matched CQ-NSGT synthesis filters to rebuild the audio
        return self.transform.bwd(coefficient_blocks)
