import torch
import torch.nn as nn
import torch.nn.functional as F

from resonance.config import (
    ACTION_DIMENSIONS,
    ACTION_VECTOR_SIZE,
    MAX_ACTIONS,
    DEFAULT_CONFIG,
)


# Convolution Block for encoding
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


# Upsampling Block for decoding and passing skips
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SpectrogramUNetModel(nn.Module):
    def __init__(self, action_vector_size, action_embedding_size=32, config=DEFAULT_CONFIG):
        super().__init__()
        if action_vector_size != ACTION_VECTOR_SIZE:
            raise ValueError(
                f"Expected action_vector_size={ACTION_VECTOR_SIZE}, "
                f"got {action_vector_size}."
            )
        self.config = config

        # Encoder block 1: 1 -> 32
        self.e1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Encoder block 2: 32 -> 64
        self.e2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # downsample both time and frequency axes to allow pitch change to fit within UNet receptive field

        # Encoder for action vectors
        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, 64),
            nn.GELU(),
            nn.Linear(64, action_embedding_size),
            nn.GELU()
        )

        # Bottleneck to combine input and action embedding: 64 + action embedding -> 128
        self.bottleneck = ConvBlock(64 + action_embedding_size, 128)

        # Decoder block 1: 128 -> 64
        self.d2 = UpBlock(128, 64, 64)

        # Decoder block 2: 64 -> 32
        self.d1 = UpBlock(64, 32, 32)

        # Output decoder block: 32 -> 1
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _net_pitch_normalization(action_vector):
        # Reshape the action vector to separate the individual actions
        action_slots = action_vector.reshape(
            action_vector.shape[0],
            MAX_ACTIONS,
            ACTION_DIMENSIONS,
        )
        # Extract the pitch change normalization values from the action slots
        is_pitch_change = action_slots[:, :, 1]
        pitch_normalization = action_slots[:, :, 5]

        # Compute the net pitch normalization by summing the pitch changes across all action slots
        return (is_pitch_change * pitch_normalization).sum(dim=1, keepdim=True)

    def forward(self, x, action_vector):
        # Get shape of input
        input_shape = x.shape[-2:]
        
        # Calculate the net pitch normalization from the action vector
        net_pitch_normalization = self._net_pitch_normalization(action_vector)

        # Calculate Y-axis (frequency) translation
        # pitch_norm * BINS_PER_OCTAVE gives the number of bins to shift
        # Grid range is [-1, 1], so distance between bins is 2 / (N_BINS - 1)
        # A negative ty shifts the image UP (higher frequencies) in grid_sample
        ty = -net_pitch_normalization * (
            2.0 * self.config.bins_per_octave / (self.config.n_bins - 1)
        )

        theta = torch.zeros(x.size(0), 2, 3, device=x.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 1, 2] = ty.squeeze(1)

        grid = F.affine_grid(theta, x.size(), align_corners=True)

        # Use border padding to extend the CQT floor to shifted regions
        x_aligned = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Pass aligned input through encoders 1 and 2
        skip1 = self.e1(x_aligned)
        h = self.pool1(skip1)

        skip2 = self.e2(h)
        h = self.pool2(skip2)

        # Turn the action vector into an embedding in the same dimension as h
        action_embd = self.action_encoder(action_vector)
        action_embd = action_embd[:, :, None, None]
        action_embd = action_embd.expand(-1, -1, h.shape[-2], h.shape[-1])

        # Merge h and the action embedding together in the bottleneck
        h = torch.cat([h, action_embd], dim=1)
        h = self.bottleneck(h)

        # Pass h through the decoders 2 and 1
        h = self.d2(h, skip2)
        h = self.d1(h, skip1)

        # Get the predicted output refinement
        unet_refinement = self.output(h)

        unet_refinement = F.interpolate(
            unet_refinement,
            size=input_shape,
            mode="bilinear",
            align_corners=False
        )

        # The true delta relative to the original input x is the alignment shift + the UNet's refinement
        predicted_delta = (x_aligned - x) + unet_refinement

        return predicted_delta
