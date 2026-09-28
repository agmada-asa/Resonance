import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from resonance.config import (
    ACTION_DIMENSIONS,
    ACTION_VECTOR_SIZE,
    DEFAULT_CONFIG,
    MAX_ACTIONS,
)


# Convolution block for the shared frequency and time representation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


# Upsampling block for restoring the global frequency and time resolution
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CQNSGTGlobalTransitionModel(nn.Module):
    def __init__(
        self,
        action_vector_size=ACTION_VECTOR_SIZE,
        shared_time_size=256,
        action_embedding_size=16,
        compression_alpha=10.0,
        config=DEFAULT_CONFIG,
    ):
        super().__init__()
        if action_vector_size != ACTION_VECTOR_SIZE:
            raise ValueError(
                f'Expected action_vector_size={ACTION_VECTOR_SIZE}, '
                f'got {action_vector_size}.'
            )

        self.config = config
        self.shared_time_size = shared_time_size
        self.compression_alpha = compression_alpha
        self.filter_order = 5
        self.epsilon = 1e-8

        # The CQ-NSGT package covers complete octaves up to Nyquist. It also
        # returns separate DC and Nyquist bands around the logarithmic bands.
        frequency_range = (config.sample_rate / 2) / config.fmin
        num_octaves = math.ceil(math.log2(frequency_range))
        frequency_count = num_octaves * config.bins_per_octave
        minimum_frequency = (
            (config.sample_rate / 2 - 1e-6) / (2 ** num_octaves)
        )
        logarithmic_frequencies = torch.logspace(
            math.log10(minimum_frequency),
            math.log10(config.sample_rate / 2 - 1e-6),
            frequency_count,
        )
        band_frequencies = torch.cat(
            [
                torch.tensor([0.0]),
                logarithmic_frequencies,
                torch.tensor([config.sample_rate / 2]),
            ]
        )
        self.register_buffer('band_frequencies', band_frequencies)

        # These are the five unit poles of the analogue Butterworth filter.
        # They are moved to the requested cutoff before the bilinear transform.
        pole_indices = torch.arange(self.filter_order, dtype=torch.float32)
        pole_angles = (
            torch.pi
            * (2 * pole_indices + self.filter_order + 1)
            / (2 * self.filter_order)
        )
        butterworth_poles = torch.polar(
            torch.ones_like(pole_angles),
            pole_angles,
        )
        self.register_buffer('butterworth_poles', butterworth_poles)

        # Encode the complete action sequence for every frequency and time point
        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, 32),
            nn.GELU(),
            nn.Linear(32, action_embedding_size),
            nn.GELU(),
        )

        # Input channels contain real, imaginary, frequency, time resolution,
        # and the broadcast action embedding.
        input_channels = 2 + 2 + action_embedding_size
        self.e1 = ConvBlock(input_channels, 24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e2 = ConvBlock(24, 48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(48, 96)
        self.d2 = UpBlock(96, 48, 48)
        self.d1 = UpBlock(48, 24, 24)

        # Predict native magnitude, phase rotation, and complex correction
        self.output = nn.Conv2d(24, 4, kernel_size=1)

        # Start with only the deterministic native action controllers. The
        # U-Net then learns the smaller errors left by their approximations.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _action_slots(action_vector):
        # Separate the four ordered actions so they can be applied in sequence
        return action_vector.reshape(
            action_vector.shape[0],
            MAX_ACTIONS,
            ACTION_DIMENSIONS,
        )

    @staticmethod
    def _unwrap_phase(phase):
        # Remove jumps at -pi and pi before changing the phase rate
        difference = phase[..., 1:] - phase[..., :-1]
        wrapped = torch.remainder(
            difference + torch.pi,
            2 * torch.pi,
        ) - torch.pi
        wrapped = torch.where(
            (wrapped == -torch.pi) & (difference > 0),
            torch.pi,
            wrapped,
        )
        return torch.cat(
            [
                phase[..., :1],
                phase[..., :1] + torch.cumsum(wrapped, dim=-1),
            ],
            dim=-1,
        )

    @staticmethod
    def _native_frequency_bands(vector_blocks):
        # Flatten the ragged blocks into their true low-to-high frequency order
        return [
            block[:, :, frequency_index, :]
            for block in vector_blocks
            for frequency_index in range(block.shape[-2])
        ]

    @staticmethod
    def _rebuild_blocks(frequency_bands, template_blocks):
        rebuilt_blocks = []
        frequency_offset = 0

        # Restore the exact block layout expected by CQ-NSGT synthesis
        for block in template_blocks:
            frequency_count = block.shape[-2]
            rebuilt_blocks.append(
                torch.stack(
                    frequency_bands[
                        frequency_offset:frequency_offset + frequency_count
                    ],
                    dim=2,
                )
            )
            frequency_offset += frequency_count

        return rebuilt_blocks

    def _pitch_shift_sample(self, vector_blocks, semitones, batch_index):
        frequency_bands = self._native_frequency_bands(vector_blocks)

        # A zero-semitone action should be a completely exact identity path
        if abs(semitones) < self.epsilon:
            return [
                band[batch_index:batch_index + 1]
                for band in frequency_bands
            ]

        pitch_ratio = 2 ** (semitones / 12)
        frequencies = self.band_frequencies
        positive_frequencies = frequencies.clamp_min(self.epsilon)
        shifted_bands = []

        for target_index, target_band in enumerate(frequency_bands):
            target_frequency = float(frequencies[target_index])
            target_length = target_band.shape[-1]

            # DC and Nyquist cannot be moved beyond the represented range
            if target_frequency <= 0 or target_frequency >= self.config.sample_rate / 2:
                shifted_bands.append(torch.zeros_like(
                    target_band[batch_index:batch_index + 1]
                ))
                continue

            wanted_source_frequency = target_frequency / pitch_ratio
            frequency_distance = torch.abs(
                torch.log(positive_frequencies)
                - math.log(wanted_source_frequency)
            )
            source_index = int(torch.argmin(frequency_distance))
            source_band = frequency_bands[source_index][
                batch_index:batch_index + 1
            ]
            source_frequency = float(frequencies[source_index])

            # Interpolate magnitude and continuous phase separately. Mixing
            # real and imaginary values directly causes the rattly cancellation.
            real = source_band[:, 0]
            imaginary = source_band[:, 1]
            magnitude = torch.sqrt(real ** 2 + imaginary ** 2 + self.epsilon)
            phase = self._unwrap_phase(torch.atan2(imaginary, real))
            initial_phase = phase[:, :1]

            magnitude = F.interpolate(
                magnitude[:, None],
                size=target_length,
                mode='linear',
                align_corners=False,
            )[:, 0]
            phase = F.interpolate(
                phase[:, None],
                size=target_length,
                mode='linear',
                align_corners=False,
            )[:, 0]

            # CQ-NSGT phase measures the tone's offset from the band centre.
            # Scale that offset and account for the exact new centre frequency.
            time = torch.arange(
                target_length,
                device=phase.device,
                dtype=phase.dtype,
            )[None] * (self.config.duration / target_length)
            centre_frequency_correction = (
                source_frequency * pitch_ratio - target_frequency
            )
            phase = (
                initial_phase
                + (phase - initial_phase) * pitch_ratio
                + 2 * torch.pi * centre_frequency_correction * time
            )

            shifted_bands.append(
                torch.stack(
                    [
                        magnitude * torch.cos(phase),
                        magnitude * torch.sin(phase),
                    ],
                    dim=1,
                )
            )

        return shifted_bands

    def _apply_pitch_action(self, vector_blocks, pitch_normalization):
        batch_size = vector_blocks[0].shape[0]
        shifted_samples = []

        # Every sample can request a different pitch change, so route each one
        # independently before joining the training batch back together.
        for batch_index in range(batch_size):
            semitones = float(
                pitch_normalization[batch_index].detach().cpu()
            ) * 12
            shifted_samples.append(
                self._pitch_shift_sample(
                    vector_blocks,
                    semitones,
                    batch_index,
                )
            )

        shifted_bands = [
            torch.cat(
                [sample[band_index] for sample in shifted_samples],
                dim=0,
            )
            for band_index in range(len(shifted_samples[0]))
        ]
        return self._rebuild_blocks(shifted_bands, vector_blocks)

    def _butterworth_response(self, cutoff_normalization, high_pass):
        # Reverse the logarithmic cutoff encoding used by AudioActionProcessor
        minimum_cutoff = math.log10(20)
        maximum_cutoff = math.log10(20000)
        log_cutoff = (
            minimum_cutoff
            + cutoff_normalization * (maximum_cutoff - minimum_cutoff)
        )
        cutoff = torch.pow(10, log_cutoff).clamp(
            20,
            self.config.sample_rate / 2 - 1,
        )

        sample_rate = self.config.sample_rate
        warped_cutoff = (
            2 * sample_rate * torch.tan(torch.pi * cutoff / sample_rate)
        )
        prototype_poles = self.butterworth_poles[None]

        if high_pass:
            analogue_poles = warped_cutoff[:, None] / prototype_poles
            zero = 1.0
            normalization_point = -1.0
        else:
            analogue_poles = warped_cutoff[:, None] * prototype_poles
            zero = -1.0
            normalization_point = 1.0

        digital_poles = (
            2 * sample_rate + analogue_poles
        ) / (
            2 * sample_rate - analogue_poles
        )
        digital_zeros = torch.full_like(digital_poles, zero)

        # Normalize low-pass at DC and high-pass at Nyquist, matching scipy
        normalization = torch.tensor(
            normalization_point,
            device=digital_poles.device,
            dtype=digital_poles.dtype,
        )
        gain = torch.prod(
            normalization - digital_poles,
            dim=1,
        ) / torch.prod(
            normalization - digital_zeros,
            dim=1,
        )

        frequencies = self.band_frequencies.to(cutoff.dtype)
        angle = 2 * torch.pi * frequencies / sample_rate
        points = torch.polar(torch.ones_like(angle), angle)
        numerator = torch.prod(
            points[None, :, None] - digital_zeros[:, None, :],
            dim=2,
        )
        denominator = torch.prod(
            points[None, :, None] - digital_poles[:, None, :],
            dim=2,
        )
        return gain[:, None] * numerator / denominator

    def _apply_complex_response(self, vector_blocks, response):
        transformed_blocks = []
        frequency_offset = 0
        identity_samples = torch.all(
            torch.abs(response - 1) < self.epsilon,
            dim=1,
        )

        for block in vector_blocks:
            frequency_count = block.shape[-2]
            block_response = response[
                :,
                frequency_offset:frequency_offset + frequency_count,
            ]
            block_response = block_response[:, None, :, None]

            # The vectors contain log-compressed magnitude. Decode only the
            # magnitude, apply the response, then compress it again.
            compressed_magnitude = torch.sqrt(
                torch.sum(block ** 2, dim=1, keepdim=True) + self.epsilon
            )
            magnitude = torch.expm1(compressed_magnitude) / self.compression_alpha
            response_magnitude = torch.abs(block_response)
            transformed_magnitude = torch.log1p(
                self.compression_alpha * magnitude * response_magnitude
            )

            direction = torch.complex(
                block[:, 0:1],
                block[:, 1:2],
            ) / compressed_magnitude
            response_direction = block_response / response_magnitude.clamp_min(
                self.epsilon
            )
            transformed = transformed_magnitude * direction * response_direction
            transformed_block = torch.cat(
                [transformed.real, transformed.imag],
                dim=1,
            )

            # Do not introduce compression rounding into untouched samples
            transformed_blocks.append(
                torch.where(
                    identity_samples[:, None, None, None],
                    block,
                    transformed_block,
                )
            )
            frequency_offset += frequency_count

        return transformed_blocks

    def _apply_native_actions(self, vector_blocks, action_vector):
        transformed_blocks = vector_blocks

        # Apply controllers in slot order so multi-action examples retain the
        # same meaning as the action processing used during data generation.
        for action in self._action_slots(action_vector).unbind(dim=1):
            is_gain = action[:, 0]
            is_pitch = action[:, 1]
            is_low_pass = action[:, 2]
            is_high_pass = action[:, 3]

            # Skip empty action slots instead of running all native controllers
            if not bool(torch.any(action[:, :4]).detach().cpu()):
                continue

            if bool(torch.any(is_pitch).detach().cpu()):
                transformed_blocks = self._apply_pitch_action(
                    transformed_blocks,
                    action[:, 5] * is_pitch,
                )

            has_frequency_response = torch.any(
                torch.stack([is_gain, is_low_pass, is_high_pass])
            )
            if not bool(has_frequency_response.detach().cpu()):
                continue

            # Gain and filter are mutually exclusive within one action slot,
            # so their responses can be applied in one pass over the blocks.
            gain_db = action[:, 4] * 12 * is_gain
            gain = torch.pow(10, gain_db / 20)
            response = gain[:, None].expand(
                -1,
                len(self.band_frequencies),
            ).to(torch.complex64)

            if bool(torch.any(is_low_pass).detach().cpu()):
                low_pass_response = self._butterworth_response(
                    action[:, 6],
                    high_pass=False,
                )
                response = torch.where(
                    is_low_pass[:, None].bool(),
                    low_pass_response,
                    response,
                )

            if bool(torch.any(is_high_pass).detach().cpu()):
                high_pass_response = self._butterworth_response(
                    action[:, 6],
                    high_pass=True,
                )
                response = torch.where(
                    is_high_pass[:, None].bool(),
                    high_pass_response,
                    response,
                )

            transformed_blocks = self._apply_complex_response(
                transformed_blocks,
                response,
            )

        return transformed_blocks

    def _pack_blocks(self, vector_blocks):
        packed_blocks = []
        time_resolutions = []
        maximum_length = max(block.shape[-1] for block in vector_blocks)

        # Each frequency row becomes one ordered band on a shared time grid
        for block in vector_blocks:
            batch_size, channels, frequencies, time_length = block.shape
            frequency_rows = block.permute(0, 2, 1, 3).reshape(
                batch_size * frequencies,
                channels,
                time_length,
            )
            frequency_rows = F.interpolate(
                frequency_rows,
                size=self.shared_time_size,
                mode='linear',
                align_corners=False,
            )
            frequency_rows = frequency_rows.reshape(
                batch_size,
                frequencies,
                channels,
                self.shared_time_size,
            ).permute(0, 2, 1, 3)
            packed_blocks.append(frequency_rows)

            resolution = time_length / maximum_length
            time_resolutions.extend([resolution] * frequencies)

        packed_vectors = torch.cat(packed_blocks, dim=2)
        time_resolutions = torch.tensor(
            time_resolutions,
            device=packed_vectors.device,
            dtype=packed_vectors.dtype,
        )
        return packed_vectors, time_resolutions

    def _apply_global_transformation(self, global_output, vector_blocks):
        predicted_blocks = []
        frequency_offset = 0

        # Return every frequency row to its original CQ-NSGT time resolution
        for block in vector_blocks:
            batch_size, channels, frequencies, time_length = block.shape
            block_output = global_output[
                :,
                :,
                frequency_offset:frequency_offset + frequencies,
                :,
            ]
            block_output = block_output.permute(0, 2, 1, 3).reshape(
                batch_size * frequencies,
                4,
                self.shared_time_size,
            )
            block_output = F.interpolate(
                block_output,
                size=time_length,
                mode='linear',
                align_corners=False,
            )
            block_output = block_output.reshape(
                batch_size,
                frequencies,
                4,
                time_length,
            ).permute(0, 2, 1, 3)

            # Rotate and scale the complete native complex sequence. This keeps
            # its phase detail while still allowing learned local correction.
            magnitude_scale = torch.exp(block_output[:, :1].clamp(-6, 6))
            phase_rotation = block_output[:, 1:2]
            cosine = torch.cos(phase_rotation)
            sine = torch.sin(phase_rotation)
            real = block[:, 0:1]
            imaginary = block[:, 1:2]
            rotated = torch.cat(
                [
                    real * cosine - imaginary * sine,
                    real * sine + imaginary * cosine,
                ],
                dim=1,
            )
            complex_correction = block_output[:, 2:]
            predicted_blocks.append(
                rotated * magnitude_scale + complex_correction
            )
            frequency_offset += frequencies

        return predicted_blocks

    def forward_native(self, native_blocks, action_vector):
        # This entry point lets fixed experiment data cache the deterministic
        # controllers while training the learned refinement normally.
        global_vectors, time_resolutions = self._pack_blocks(native_blocks)
        batch_size, _, frequency_count, time_size = global_vectors.shape

        if frequency_count != len(self.band_frequencies):
            raise ValueError(
                f'Expected {len(self.band_frequencies)} frequency bands, '
                f'got {frequency_count}.'
            )

        # Use logarithmic centre frequency rather than an arbitrary row number
        frequency_positions = torch.log1p(self.band_frequencies)
        frequency_positions = frequency_positions / frequency_positions[-1]
        frequency_positions = frequency_positions * 2 - 1
        frequency_positions = frequency_positions[
            None, None, :, None
        ].expand(batch_size, 1, -1, time_size)
        resolution_channel = time_resolutions[None, None, :, None].expand(
            batch_size,
            1,
            -1,
            time_size,
        )

        action_embedding = self.action_encoder(action_vector)
        action_embedding = action_embedding[:, :, None, None].expand(
            -1,
            -1,
            frequency_count,
            time_size,
        )

        h = torch.cat(
            [
                global_vectors,
                frequency_positions,
                resolution_channel,
                action_embedding,
            ],
            dim=1,
        )

        # Mix information across every frequency band and shared time position
        skip1 = self.e1(h)
        h = self.pool1(skip1)
        skip2 = self.e2(h)
        h = self.pool2(skip2)
        h = self.bottleneck(h)
        h = self.d2(h, skip2)
        h = self.d1(h, skip1)
        transformation = self.output(h)

        # Apply learned scaling and refinement to the native controller output
        return self._apply_global_transformation(
            transformation,
            native_blocks,
        )

    def forward(self, vector_blocks, action_vector):
        if not isinstance(vector_blocks, list):
            raise ValueError('Expected a list of CQ-NSGT vector blocks.')

        # Perform known audio operations on the full native CQ-NSGT vectors
        native_blocks = self._apply_native_actions(
            vector_blocks,
            action_vector,
        )
        return self.forward_native(native_blocks, action_vector)
