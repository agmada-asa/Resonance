import unittest

import torch

from resonance.config import ACTION_VECTOR_SIZE
from resonance.models.latent_audio import (
    LatentActionTransitionModel,
    WaveformAutoencoder,
)


class WaveformAutoencoderTests(unittest.TestCase):
    def test_shapes_and_compression_ratio(self):
        model = WaveformAutoencoder(latent_channels=24)
        waveform = torch.randn(3, 1, 88200)
        latent = model.encode(waveform)
        reconstructed = model.decode(latent, waveform.shape[-1])

        self.assertEqual(latent.shape, (3, 24, 346))
        self.assertEqual(reconstructed.shape, waveform.shape)
        self.assertGreater(model.compression_ratio(88200), 10.5)

    def test_autoencoder_backpropagates_through_the_bottleneck(self):
        model = WaveformAutoencoder(latent_channels=12)
        waveform = torch.randn(2, 1, 1025)
        model(waveform).square().mean().backward()
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in model.parameters()
        ))


class LatentActionTransitionModelTests(unittest.TestCase):
    def test_no_action_is_an_exact_identity(self):
        model = LatentActionTransitionModel(ACTION_VECTOR_SIZE)
        latent = torch.randn(2, 16, 23)
        action = torch.zeros(2, ACTION_VECTOR_SIZE)
        torch.testing.assert_close(
            model(latent, action), latent, rtol=0, atol=0,
        )

    def test_action_transition_can_learn_a_delta(self):
        torch.manual_seed(42)
        model = LatentActionTransitionModel(ACTION_VECTOR_SIZE, width=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        latent = torch.randn(4, 16, 12)
        action = torch.zeros(4, ACTION_VECTOR_SIZE)
        action[:, 0] = 1
        target = latent + 0.25
        initial = torch.mean((model(latent, action) - target) ** 2).item()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(latent, action) - target) ** 2)
            loss.backward()
            optimizer.step()
        final = torch.mean((model(latent, action) - target) ** 2).item()
        self.assertLess(final, initial)


if __name__ == '__main__':
    unittest.main()
