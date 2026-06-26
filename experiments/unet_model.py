import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, action_vector_size, action_embedding_size=32):
        super().__init__()

        # Encoder block 1: 1 -> 32
        self.e1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Encoder block 2: 32 -> 64
        self.e2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) # (1, 2) pooling to downsample time dimension only

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

    def forward(self, x, action_vector):
        # Get shape of input
        input_shape = x.shape[-2:]

        # Pass x through encoders 1 and 2
        skip1 = self.e1(x)
        x = self.pool1(skip1)

        skip2 = self.e2(x)
        x = self.pool2(skip2)

        # Turn the action vector into an embedding in the same dimension as x
        action_embd = self.action_encoder(action_vector)
        action_embd = action_embd[:, :, None, None]
        action_embd = action_embd.expand(-1, -1, x.shape[-2], x.shape[-1])

        # Merge x and the action embedding together in the bottleneck
        x = torch.cat([x, action_embd], dim=1)
        x = self.bottleneck(x)

        # Pass x through the decoders 2 and 1
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        # Get the predicted output
        predicted_delta = self.output(x)

        return F.interpolate(
            predicted_delta,
            size=input_shape,
            mode="bilinear",
            align_corners=False
        )
