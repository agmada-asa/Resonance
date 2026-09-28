import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def constrain_normalized_compact_state(state, mean, std):
    """Keep recurrent compact-state predictions inside training support."""
    physical = state * std + mean
    magnitude = physical[:, 0:1].clamp(1e-6, 10.0)
    phase = physical[:, 1:2].clamp(-26.0, 26.0)
    return (torch.cat([magnitude, phase], dim=1) - mean) / std


class CompactCQStateCodec:
    """Pack ragged complex CQ bands into a small phase-aware grid."""

    def __init__(self, time_size=96):
        if time_size < 2:
            raise ValueError("time_size must be at least two.")
        self.time_size = time_size

    @staticmethod
    def _unwrap(phase):
        difference = phase[..., 1:] - phase[..., :-1]
        wrapped = torch.remainder(
            difference + torch.pi, 2 * torch.pi,
        ) - torch.pi
        wrapped = torch.where(
            (wrapped == -torch.pi) & (difference > 0),
            torch.pi,
            wrapped,
        )
        return torch.cat([
            phase[..., :1],
            phase[..., :1] + torch.cumsum(wrapped, dim=-1),
        ], dim=-1)

    def encode(self, vector_blocks):
        if not vector_blocks:
            raise ValueError("Expected at least one CQ block.")
        packed = []
        batch_size = vector_blocks[0].shape[0]
        for block in vector_blocks:
            if block.ndim != 4 or block.shape[1] != 2:
                raise ValueError("Expected CQ blocks shaped [batch, 2, frequency, time].")
            if block.shape[0] != batch_size:
                raise ValueError("CQ blocks have inconsistent batch sizes.")
            magnitude = torch.sqrt(torch.sum(block ** 2, dim=1) + 1e-12)
            phase_cycles = self._unwrap(
                torch.atan2(block[:, 1], block[:, 0])
            ) / (2 * torch.pi)
            values = torch.stack([magnitude, phase_cycles], dim=2)
            frequencies = block.shape[2]
            values = values.reshape(
                batch_size * frequencies, 2, block.shape[-1],
            )
            values = F.interpolate(
                values, size=self.time_size, mode="linear",
                align_corners=True,
            )
            # Store one initial phase followed by local phase increments. This
            # reconstructs the same unwrapped phase while avoiding a running
            # value that can span hundreds of cycles.
            phase_cycles = values[:, 1].clone()
            values[:, 1, 0] = phase_cycles[:, 0]
            values[:, 1, 1:] = phase_cycles[:, 1:] - phase_cycles[:, :-1]
            packed.append(values.reshape(
                batch_size, frequencies, 2, self.time_size,
            ).permute(0, 2, 1, 3))
        return torch.cat(packed, dim=2)

    def decode(self, state, template_blocks):
        if state.ndim != 4 or state.shape[1] != 2:
            raise ValueError("Expected state shaped [batch, 2, frequency, time].")
        expected_frequencies = sum(block.shape[2] for block in template_blocks)
        if state.shape[2] != expected_frequencies:
            raise ValueError("State frequency count does not match the CQ layout.")
        if state.shape[-1] != self.time_size:
            raise ValueError("State time size does not match the codec.")

        decoded = []
        frequency_offset = 0
        for template in template_blocks:
            frequency_count = template.shape[2]
            time_length = template.shape[-1]
            values = state[
                :, :, frequency_offset:frequency_offset + frequency_count,
            ]
            values = torch.stack([
                values[:, 0],
                self.phase_cycles(values),
            ], dim=1)
            values = values.permute(0, 2, 1, 3).reshape(
                state.shape[0] * frequency_count, 2, self.time_size,
            )
            values = F.interpolate(
                values, size=time_length, mode="linear",
                align_corners=True,
            )
            values = values.reshape(
                state.shape[0], frequency_count, 2, time_length,
            ).permute(0, 2, 1, 3)
            magnitude = values[:, 0]
            phase = values[:, 1] * (2 * torch.pi)
            decoded.append(torch.stack([
                magnitude * torch.cos(phase),
                magnitude * torch.sin(phase),
            ], dim=1))
            frequency_offset += frequency_count
        return decoded

    @staticmethod
    def phase_cycles(state):
        """Expand initial phase plus increments into unwrapped phase cycles."""
        initial = state[:, 1:2, ..., :1]
        increments = state[:, 1:2, ..., 1:]
        return torch.cat([
            initial,
            initial + torch.cumsum(increments, dim=-1),
        ], dim=-1).squeeze(1)

    def compression_ratio(self, template_blocks):
        original_values = sum(block[0].numel() for block in template_blocks)
        frequency_count = sum(block.shape[2] for block in template_blocks)
        state_values = 2 * frequency_count * self.time_size
        return original_values / state_values


class CompactCQWorldModel(nn.Module):
    """Learn an action-conditioned residual transition on a compact CQ grid."""

    def __init__(
        self, action_vector_size, width=32, learned_frequency_warp=False,
        initial_frequency_shift=0.0, action_specific_heads=False,
    ):
        super().__init__()
        if width % 8:
            raise ValueError("width must be divisible by eight.")
        self.action_vector_size = action_vector_size
        self.learned_frequency_warp = learned_frequency_warp
        self.action_specific_heads = action_specific_heads
        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.input = nn.Conv2d(2 + width + 2, width, kernel_size=3, padding=1)
        self.encoder_1 = self._block(width, width)
        self.down_1 = nn.Conv2d(width, width * 2, kernel_size=4, stride=2, padding=1)
        self.encoder_2 = self._block(width * 2, width * 2)
        self.down_2 = nn.Conv2d(width * 2, width * 3, kernel_size=4, stride=2, padding=1)
        self.middle = self._block(width * 3, width * 3)
        self.decoder_2 = self._block(width * 3 + width * 2, width * 2)
        self.decoder_1 = self._block(width * 2 + width, width)
        if action_specific_heads:
            self.output_heads = nn.ModuleList([
                nn.Conv2d(width, 2, kernel_size=1)
                for _ in range(4)
            ])
            for output_head in self.output_heads:
                nn.init.zeros_(output_head.weight)
                nn.init.zeros_(output_head.bias)
        else:
            self.output = nn.Conv2d(width, 2, kernel_size=1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        if learned_frequency_warp:
            # Learn the geometry of the pitch action from data while preserving
            # the known neutral point: a zero action parameter cannot move the
            # frequency grid.
            initial_frequency_shift = min(
                0.999, max(-0.999, initial_frequency_shift),
            )
            self.frequency_shift_scale = nn.Parameter(torch.tensor(
                math.atanh(initial_frequency_shift), dtype=torch.float32,
            ))

    @staticmethod
    def _block(input_channels, output_channels):
        groups = math.gcd(output_channels, 8)
        return nn.Sequential(
            nn.GroupNorm(groups, input_channels),
            nn.GELU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _coordinates(state):
        frequency = torch.linspace(
            -1, 1, state.shape[-2], device=state.device, dtype=state.dtype,
        )[None, None, :, None].expand(state.shape[0], 1, -1, state.shape[-1])
        time = torch.linspace(
            -1, 1, state.shape[-1], device=state.device, dtype=state.dtype,
        )[None, None, None, :].expand(state.shape[0], 1, state.shape[-2], -1)
        return frequency, time

    @staticmethod
    def _warp_frequency(state, normalized_shift):
        """Linearly resample frequency without relying on grid_sample backward."""
        frequency_count = state.shape[-2]
        positions = torch.arange(
            frequency_count, device=state.device, dtype=state.dtype,
        )[None, :] + normalized_shift * ((frequency_count - 1) / 2)
        lower = torch.floor(positions)
        upper = lower + 1
        fraction = positions - lower

        def gather(indices):
            valid = (indices >= 0) & (indices < frequency_count)
            clamped = indices.clamp(0, frequency_count - 1).to(torch.long)
            gather_index = clamped[:, None, :, None].expand(
                -1, state.shape[1], -1, state.shape[-1],
            )
            values = torch.gather(state, dim=2, index=gather_index)
            return values * valid[:, None, :, None]

        return (
            gather(lower) * (1 - fraction[:, None, :, None])
            + gather(upper) * fraction[:, None, :, None]
        )

    def forward(self, state, action_vector):
        if state.ndim != 4 or state.shape[1] != 2:
            raise ValueError("Expected state shaped [batch, 2, frequency, time].")
        if action_vector.shape != (state.shape[0], self.action_vector_size):
            raise ValueError("Action vector has an incompatible shape.")
        action_embedding = self.action_encoder(action_vector)
        transformed_state = state
        if self.learned_frequency_warp:
            action_slots = action_vector.reshape(state.shape[0], -1, 7)
            pitch_parameter = torch.sum(
                action_slots[:, :, 5] * action_slots[:, :, 1], dim=1,
            )
            shift = (
                torch.tanh(self.frequency_shift_scale)
                * pitch_parameter
            )[:, None]
            transformed_state = self._warp_frequency(state, shift)

        action = action_embedding[:, :, None, None].expand(
            -1, -1, state.shape[-2], state.shape[-1],
        )
        frequency, time = self._coordinates(transformed_state)
        h1 = self.encoder_1(self.input(torch.cat([
            transformed_state, action, frequency, time,
        ], dim=1)))
        h2 = self.encoder_2(self.down_1(h1))
        middle = self.middle(self.down_2(h2))
        up_2 = F.interpolate(
            middle, size=h2.shape[-2:], mode="bilinear", align_corners=False,
        )
        up_2 = self.decoder_2(torch.cat([up_2, h2], dim=1))
        up_1 = F.interpolate(
            up_2, size=h1.shape[-2:], mode="bilinear", align_corners=False,
        )
        action_slots = action_vector.reshape(state.shape[0], -1, 7)
        decoded = self.decoder_1(torch.cat([up_1, h1], dim=1))
        if self.action_specific_heads:
            action_types = torch.sum(action_slots[:, :, :4], dim=1)
            action_types = action_types / action_types.sum(
                dim=1, keepdim=True,
            ).clamp_min(1)
            head_deltas = torch.stack([
                output_head(decoded) for output_head in self.output_heads
            ], dim=1)
            delta = torch.sum(
                head_deltas * action_types[:, :, None, None, None], dim=1,
            )
        else:
            delta = self.output(decoded)

        active = torch.any(action_slots[:, :, :4] != 0, dim=(1, 2))
        transformed_state = transformed_state + delta
        return state + (
            transformed_state - state
        ) * active[:, None, None, None]
