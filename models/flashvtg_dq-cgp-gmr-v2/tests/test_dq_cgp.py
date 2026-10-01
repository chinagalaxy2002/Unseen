import importlib
import unittest

import torch


DQ_MODULE = importlib.import_module("models.flashvtg_dq-cgp-gmr-v2.dq_cgp")
FlashPointDQCGP = DQ_MODULE.FlashPointDQCGP


class FlashPointDQCGPTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 2
        self.num_points = 6
        self.video_length = 5
        self.hidden_dim = 16
        self.candidate = torch.randn(
            self.batch_size, self.num_points, self.hidden_dim, requires_grad=True
        )
        self.memory = torch.randn(
            self.batch_size, self.video_length, self.hidden_dim, requires_grad=True
        )
        self.video_mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
        )
        self.candidate_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
        )
        self.semantic = torch.randn(
            self.batch_size, self.hidden_dim, requires_grad=True
        )
        self.point = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 4.0, 2.0],
                [2.0, 1.0, 4.0, 2.0],
            ]
        )
        self.level_ids = torch.tensor([0, 0, 0, 0, 1, 1])

    def _forward(self, module):
        return module(
            candidate_state=self.candidate,
            video_memory=self.memory,
            video_valid_mask=self.video_mask,
            candidate_valid_mask=self.candidate_mask,
            query_semantic=self.semantic,
            point_metadata=self.point,
            level_ids=self.level_ids,
        )

    def test_zero_beta_is_exact_identity(self):
        module = FlashPointDQCGP(
            hidden_dim=self.hidden_dim,
            num_basis=4,
            prompt_length=3,
            router_hidden_dim=12,
            frf_hidden_dim=24,
            beta=0.0,
            num_levels=2,
        )
        output = self._forward(module)
        self.assertIs(output, self.candidate)
        self.assertIsNone(module.last_output)

    def test_active_shapes_masks_and_gradients(self):
        module = FlashPointDQCGP(
            hidden_dim=self.hidden_dim,
            num_basis=4,
            prompt_length=3,
            router_hidden_dim=12,
            frf_hidden_dim=24,
            beta=0.05,
            num_levels=2,
            locality_strength=0.1,
        )
        output = self._forward(module)
        diagnostics = module.last_output
        self.assertEqual(output.shape, self.candidate.shape)
        self.assertIsNotNone(diagnostics)
        self.assertEqual(
            diagnostics.temporal_attention.shape,
            (self.batch_size, self.num_points, self.video_length),
        )
        self.assertEqual(
            diagnostics.basis_weights.shape,
            (self.batch_size, self.num_points, 4),
        )
        self.assertEqual(
            diagnostics.prompt_attention.shape,
            (self.batch_size, self.num_points, 3),
        )
        self.assertEqual(
            diagnostics.update_gate.shape,
            (self.batch_size, self.num_points),
        )
        torch.testing.assert_close(
            diagnostics.temporal_attention[1, :, 3:],
            torch.zeros_like(diagnostics.temporal_attention[1, :, 3:]),
        )
        torch.testing.assert_close(
            output[1, 4:], self.candidate[1, 4:], rtol=0, atol=0
        )

        output.square().mean().backward()
        self.assertIsNotNone(module.basis_prompts.grad)
        self.assertGreater(float(module.basis_prompts.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.memory.grad)
        self.assertGreater(float(self.memory.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
