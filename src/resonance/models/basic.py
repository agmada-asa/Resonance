import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramTransitionModel(nn.Module):
    def __init__(self, action_vector_size, action_embedding_size=32):
        super().__init__()
        # Define the encoder, action encoder, and decoder networks

        # Encoder is a simple CNN that processes the input spectrogram
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        # Action encoder is a simple feedforward network that processes the action vector
        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, 64),
            nn.GELU(),
            nn.Linear(64, action_embedding_size),
            nn.GELU(),
        )

        # Decoder is a simple CNN that reconstructs the output spectrogram from the encoded features and action embedding
        self.decoder = nn.Sequential(
            nn.Conv2d(64 + action_embedding_size, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, action_vector):
        input_shape = x.shape[-2:]
        features = self.encoder(x)

        # Create an action embedding and expand it to match the spatial dimensions of the features
        action_embedding = self.action_encoder(action_vector)
        action_embedding = action_embedding[:, :, None, None]
        action_embedding = action_embedding.expand(
            -1,
            -1,
            features.shape[-2],
            features.shape[-1],
        )

        # Concatenate the features and action embedding along the channel dimension
        conditioned_features = torch.cat((features, action_embedding), dim=1)

        # Pass the conditioned features through the decoder to predict the delta spectrogram
        predicted_delta = self.decoder(conditioned_features)
        return F.interpolate(
            predicted_delta,
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )

