import importlib
import unittest
from types import SimpleNamespace

import torch


DQ_MODULE = importlib.import_module("models.flashvtg_dq_cgp_v3_gmr.dq_cgp")
FlashPointHSDQCGP = DQ_MODULE.FlashPointHSDQCGP
MODEL_MODULE = importlib.import_module("models.flashvtg_dq_cgp_v3_gmr.model")


class RoutingLossHarness:
    args = SimpleNamespace()
    _dq_cgp_js_divergence = staticmethod(
        MODEL_MODULE.SetCriterion._dq_cgp_js_divergence
    )
    _dq_cgp_routing_terms = MODEL_MODULE.SetCriterion._dq_cgp_routing_terms
    _dq_cgp_relation_terms = MODEL_MODULE.SetCriterion._dq_cgp_relation_terms


class FlashPointHSDQCGPTest(unittest.TestCase):
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
        # Four stride-1 points and two stride-2 points, ordered as FlashVTG's
        # PointGenerator concatenates the pyramid levels.
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

    def _module(self, **overrides):
        options = dict(
            hidden_dim=self.hidden_dim,
            num_basis=4,
            prompt_length=3,
            router_hidden_dim=12,
            point_router_hidden_dim=8,
            frf_hidden_dim=24,
            beta=0.05,
            num_levels=2,
            locality_strength=0.1,
            point_mixture_ratio=0.1,
            routing_topk=4,
            local_prototype_radius=1,
        )
        options.update(overrides)
        return FlashPointHSDQCGP(**options)

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
        module = self._module(beta=0.0)
        output = self._forward(module)
        self.assertIs(output, self.candidate)
        self.assertIsNone(module.last_output)

    def test_hierarchy_shapes_masks_bound_and_gradients(self):
        module = self._module()
        output = self._forward(module)
        diagnostics = module.last_output
        self.assertEqual(output.shape, self.candidate.shape)
        self.assertIsNotNone(diagnostics)
        self.assertEqual(
            diagnostics.temporal_attention.shape,
            (self.batch_size, self.num_points, self.video_length),
        )
        self.assertEqual(diagnostics.level_prototypes.shape, (self.batch_size, 2, self.hidden_dim))
        self.assertEqual(diagnostics.local_prototypes.shape, (self.batch_size, self.num_points, self.hidden_dim))
        self.assertEqual(diagnostics.level_basis_weights.shape, (self.batch_size, 2, 4))
        self.assertEqual(diagnostics.basis_weights.shape, (self.batch_size, self.num_points, 4))
        self.assertEqual(diagnostics.prompt_attention.shape, (self.batch_size, self.num_points, 3))
        self.assertFalse(hasattr(module, "update_gate"))
        self.assertLessEqual(
            float(diagnostics.level_router_logits.detach().abs().max()),
            module.router_logit_scale + 1e-7,
        )
        uniform = torch.full_like(diagnostics.level_basis_weights, 0.25)
        self.assertLess(
            float((diagnostics.level_basis_weights.detach() - uniform).abs().max()),
            0.02,
        )
        torch.testing.assert_close(
            diagnostics.temporal_attention[1, :, 3:],
            torch.zeros_like(diagnostics.temporal_attention[1, :, 3:]),
        )
        torch.testing.assert_close(output[1, 4:], self.candidate[1, 4:], rtol=0, atol=0)
        expected_weights = (
            (1.0 - module.point_mixture_ratio)
            * diagnostics.level_basis_weights[:, self.level_ids]
            + module.point_mixture_ratio * diagnostics.point_basis_weights
        )
        torch.testing.assert_close(diagnostics.basis_weights, expected_weights)
        self.assertLessEqual(
            float(
                0.5 * diagnostics.point_correction_weights.detach().abs().sum(dim=-1).max()
            ),
            module.point_mixture_ratio + 1e-7,
        )

        # Pooling is normalized over valid points within every present level.
        for batch_index in range(self.batch_size):
            for level_index in range(2):
                valid = self.candidate_mask[batch_index] & (self.level_ids == level_index)
                pooled = diagnostics.level_pool_attention[batch_index, level_index]
                if bool(valid.any()):
                    torch.testing.assert_close(pooled[valid].sum(), torch.ones(()))
                    torch.testing.assert_close(pooled[~valid], torch.zeros_like(pooled[~valid]))

        output.square().mean().backward()
        self.assertIsNotNone(module.basis_prompts.grad)
        self.assertGreater(float(module.basis_prompts.grad.abs().sum()), 0.0)
        self.assertIsNotNone(module.level_router[0].weight.grad)
        self.assertGreater(float(module.level_router[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(module.point_router[0].weight.grad)
        self.assertGreater(float(module.point_router[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.memory.grad)
        self.assertGreater(float(self.memory.grad.abs().sum()), 0.0)

    def test_zero_point_mixture_recovers_exact_level_routing(self):
        module = self._module(point_mixture_ratio=0.0)
        self._forward(module)
        diagnostics = module.last_output
        torch.testing.assert_close(
            diagnostics.basis_weights,
            diagnostics.level_basis_weights[:, self.level_ids],
        )
        torch.testing.assert_close(
            diagnostics.point_correction_weights,
            torch.zeros_like(diagnostics.point_correction_weights),
        )

    def test_sparse_topk_keeps_exactly_k_active_bases(self):
        module = self._module(num_basis=6, routing_topk=4)
        self._forward(module)
        diagnostics = module.last_output
        self.assertTrue(
            bool(((diagnostics.level_basis_weights > 0).sum(dim=-1) == 4).all())
        )
        self.assertTrue(
            bool(((diagnostics.point_basis_weights > 0).sum(dim=-1) == 4).all())
        )

    def test_relation_loss_is_finite_and_reaches_point_routes(self):
        harness = RoutingLossHarness()
        route_logits = torch.randn(
            self.batch_size, self.num_points, 4, requires_grad=True
        )
        basis_weights = torch.softmax(route_logits, dim=-1)
        attention = torch.softmax(
            torch.randn(self.batch_size, self.num_points, self.video_length), dim=-1
        )
        loss, diagnostics = harness._dq_cgp_relation_terms(
            temporal_attention=attention,
            basis_weights=basis_weights,
            candidate_mask=self.candidate_mask,
            level_ids=self.level_ids,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(
            float(diagnostics["loss_dq_cgp_relation_route_js"].detach()), 0.0
        )
        loss.backward()
        self.assertIsNotNone(route_logits.grad)
        self.assertGreater(float(route_logits.grad.abs().sum()), 0.0)

    def test_anti_collapse_route_loss_prefers_balanced_composition(self):
        harness = RoutingLossHarness()
        candidate_mask = torch.ones(2, self.num_points, dtype=torch.bool)

        # Ideal for N=4 and entropy target 0.5: every route mixes two bases,
        # while each level and the global bank use all bases across the batch.
        ideal_level = torch.tensor(
            [
                [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]],
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.0, 0.0]],
            ]
        )
        collapsed_level = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            ]
        )

        def route_terms(level_weights):
            point_weights = level_weights[:, self.level_ids]
            return harness._dq_cgp_routing_terms(
                basis_weights=point_weights,
                level_basis_weights=level_weights,
                point_basis_weights=point_weights,
                candidate_mask=candidate_mask,
                level_ids=self.level_ids,
            )

        ideal_loss, ideal_diagnostics = route_terms(ideal_level)
        collapsed_loss, collapsed_diagnostics = route_terms(collapsed_level)
        self.assertLess(float(ideal_loss), float(collapsed_loss))
        torch.testing.assert_close(
            ideal_diagnostics["loss_dq_cgp_effective_basis_count"],
            torch.tensor(4.0),
        )
        torch.testing.assert_close(
            collapsed_diagnostics["loss_dq_cgp_effective_basis_count"],
            torch.tensor(2.0),
        )
        self.assertGreater(
            float(ideal_diagnostics["loss_dq_cgp_level_query_js"]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
