import torch
import torch.nn as nn


class WaveformAutoencoder(nn.Module):
    def __init__(self, latent_channels=24):
        super().__init__()
        if latent_channels <= 0:
            raise ValueError('latent_channels must be positive.')
        self.latent_channels = latent_channels
        self.n_fft = 510
        self.hop_length = 255
        self.frequency_bins = self.n_fft // 2 + 1
        self.encoded_channels = latent_channels
        self.register_buffer('window', torch.hann_window(self.n_fft))

        # The deterministic transform retains invertible complex phase. A
        # nonlinear frame encoder learns a compact code for the sparse spectra.
        self.encoder = nn.Sequential(
            nn.Linear(self.frequency_bins * 2, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, latent_channels),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, self.frequency_bins * 2),
        )

    def latent_shape(self, waveform_length):
        frame_count = waveform_length // self.hop_length + 1
        return self.encoded_channels, frame_count

    def analysis(self, waveform):
        coefficients = torch.stft(
            waveform[:, 0], n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, normalized=True, return_complex=True,
        )
        return torch.stack([coefficients.real, coefficients.imag], dim=1)

    def synthesis(self, coefficients, output_length):
        complex_coefficients = torch.complex(
            coefficients[:, 0], coefficients[:, 1],
        )
        waveform = torch.istft(
            complex_coefficients, n_fft=self.n_fft,
            hop_length=self.hop_length, window=self.window,
            normalized=True, length=output_length,
        )
        return waveform[:, None]

    def encode(self, waveform):
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError('Expected waveform shape [batch, 1, time].')
        coefficients = self.analysis(waveform)
        frames = coefficients.permute(0, 3, 1, 2).reshape(
            coefficients.shape[0], coefficients.shape[-1], -1,
        )
        return self.encoder(frames).transpose(1, 2)

    def decode_coefficients(self, latent):
        if latent.ndim != 3 or latent.shape[1] != self.encoded_channels:
            raise ValueError('Expected a flattened complex-STFT latent state.')
        frames = self.decoder(latent.transpose(1, 2))
        return frames.reshape(
            latent.shape[0], latent.shape[-1], 2, self.frequency_bins,
        ).permute(0, 2, 3, 1)

    def decode(self, latent, output_length):
        if latent.ndim != 3 or latent.shape[1] != self.encoded_channels:
            raise ValueError(
                'Expected a flattened complex-STFT latent state.'
            )
        return self.synthesis(self.decode_coefficients(latent), output_length)

    def forward(self, waveform):
        return self.decode(self.encode(waveform), waveform.shape[-1])

    def compression_ratio(self, waveform_length):
        channels, frames = self.latent_shape(waveform_length)
        return waveform_length / (channels * frames)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, values):
        return values + self.layers(values)


class LatentActionTransitionModel(nn.Module):
    def __init__(self, action_vector_size, latent_channels=16, width=64):
        super().__init__()
        self.latent_channels = latent_channels
        self.action_vector_size = action_vector_size
        self.action_encoder = nn.Sequential(
            nn.Linear(action_vector_size, width),
            nn.GELU(),
            nn.Linear(width, 32),
            nn.GELU(),
        )
        self.input = nn.Conv1d(latent_channels + 32, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(width, dilation)
            for dilation in (1, 2, 4, 8, 16, 32, 64, 128)
        ])
        self.output = nn.Conv1d(width, latent_channels, kernel_size=1)

        # Begin as an identity transition. This makes empty action slots exact
        # and gives training a stable residual target from the first update.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, action_vector):
        if latent.ndim != 3 or latent.shape[1] != self.latent_channels:
            raise ValueError('Latent input has an incompatible shape.')
        if action_vector.shape != (latent.shape[0], self.action_vector_size):
            raise ValueError('Action vector has an incompatible shape.')

        action_embedding = self.action_encoder(action_vector)
        action_embedding = action_embedding[:, :, None].expand(
            -1, -1, latent.shape[-1],
        )
        values = self.input(torch.cat([latent, action_embedding], dim=1))
        for block in self.blocks:
            values = block(values)
        delta = self.output(values)

        # Padding/no-action vectors remain an exact identity even after training
        action_slots = action_vector.reshape(latent.shape[0], -1, 7)
        active = torch.any(action_slots[:, :, :4] != 0, dim=(1, 2))
        return latent + delta * active[:, None, None]
