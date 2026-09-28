import unittest

import torch

from resonance.models.compact_cq import CompactCQStateCodec, CompactCQWorldModel


class CompactCQStateCodecTests(unittest.TestCase):
    def test_shape_and_round_trip_for_constant_complex_values(self):
        codec = CompactCQStateCodec(time_size=8)
        blocks = [
            torch.ones(2, 2, 3, 5),
            torch.ones(2, 2, 2, 17),
        ]
        state = codec.encode(blocks)
        rebuilt = codec.decode(state, blocks)
        self.assertEqual(state.shape, (2, 2, 5, 8))
        for actual, expected in zip(rebuilt, blocks):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_rejects_invalid_time_size(self):
        with self.assertRaises(ValueError):
            CompactCQStateCodec(time_size=1)


class CompactCQWorldModelTests(unittest.TestCase):
    def test_empty_action_is_exact_identity(self):
        model = CompactCQWorldModel(action_vector_size=28, width=8)
        state = torch.randn(2, 2, 12, 16)
        action = torch.zeros(2, 28)
        torch.testing.assert_close(model(state, action), state, rtol=0, atol=0)

    def test_action_transition_has_gradients(self):
        model = CompactCQWorldModel(action_vector_size=28, width=8)
        state = torch.randn(2, 2, 12, 16)
        action = torch.zeros(2, 28)
        action[:, 0] = 1
        output = model(state, action)
        output.square().mean().backward()
        self.assertIsNotNone(model.output.weight.grad)

    def test_learned_frequency_warp_starts_as_identity(self):
        model = CompactCQWorldModel(
            action_vector_size=28, width=8, learned_frequency_warp=True,
        )
        state = torch.randn(2, 2, 12, 16)
        action = torch.zeros(2, 28)
        action[:, 1] = 1
        torch.testing.assert_close(model(state, action), state, atol=1e-5, rtol=1e-5)

    def test_action_specific_head_receives_its_own_gradient(self):
        model = CompactCQWorldModel(
            action_vector_size=28, width=8, action_specific_heads=True,
        )
        state = torch.randn(2, 2, 12, 16)
        action = torch.zeros(2, 28)
        action[:, 2] = 1
        model(state, action).square().mean().backward()
        self.assertIsNotNone(model.output_heads[2].weight.grad)
        self.assertEqual(
            float(model.output_heads[0].weight.grad.abs().sum()), 0.0,
        )

    def test_frequency_warp_matches_bilinear_grid_sampling(self):
        state = torch.randn(3, 2, 12, 16)
        shift = torch.tensor([[0.0], [0.11], [-0.17]])
        frequency = torch.linspace(-1, 1, state.shape[-2])[
            None, :, None
        ].expand(state.shape[0], -1, state.shape[-1])
        time = torch.linspace(-1, 1, state.shape[-1])[
            None, None, :
        ].expand(state.shape[0], state.shape[-2], -1)
        grid = torch.stack([time, frequency + shift[:, None]], dim=-1)
        expected = torch.nn.functional.grid_sample(
            state, grid, align_corners=True,
        )
        actual = CompactCQWorldModel._warp_frequency(state, shift)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
