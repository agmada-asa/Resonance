import unittest

import torch

from resonance.config import ACTION_DIMENSIONS, ACTION_VECTOR_SIZE
from resonance.models.unet import SpectrogramUNetModel


class SpectrogramUNetModelTests(unittest.TestCase):
    def test_net_pitch_normalization_sums_pitch_slots_only(self):
        action_vector = torch.zeros(2, ACTION_VECTOR_SIZE)

        first_pitch_slot = 0
        third_pitch_slot = 2 * ACTION_DIMENSIONS
        gain_slot = ACTION_DIMENSIONS

        action_vector[0, first_pitch_slot + 1] = 1
        action_vector[0, first_pitch_slot + 5] = 7 / 12
        action_vector[0, gain_slot] = 1
        action_vector[0, gain_slot + 4] = 1
        action_vector[0, third_pitch_slot + 1] = 1
        action_vector[0, third_pitch_slot + 5] = -3 / 12

        net_pitch = SpectrogramUNetModel._net_pitch_normalization(action_vector)

        torch.testing.assert_close(net_pitch, torch.tensor([[4 / 12], [0.0]]))

    def test_model_rejects_action_vectors_with_the_wrong_size(self):
        with self.assertRaisesRegex(ValueError, "Expected action_vector_size=28"):
            SpectrogramUNetModel(action_vector_size=7)


if __name__ == '__main__':
    unittest.main()
