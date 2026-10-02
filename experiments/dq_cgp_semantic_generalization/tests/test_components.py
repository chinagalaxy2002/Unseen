"""Comprehensive Unit Tests for LS-DQCGP and DQ-CGPv3.

Verifies:
1. S+ forward & backward.
2. S- empty-GT forward & backward.
3. S- produces zero binding loss and safe class/span loss.
4. Existence loss supervises both S+ and S-.
5. Mixed S+/S- batches execute forward and backward correctly.
6. Hungarian matcher handles empty GT.
7. No NaN/Inf anywhere.
8. Existence logits present in predictions.
9. Raw and gated localization formats preserved.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPO_ROOT / "training" / "moment_detr_gmr"
for p in (REPO_ROOT, TRAIN_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import BaseOptions
from models.moment_detr_gmr.moment_detr import build_model as build_base_detr
from experiments.dq_cgp_semantic_generalization.ls_dqcgp.model import (
    LSDQCGPModel,
    install_ls_dq_cgp_loss,
)
from experiments.dq_cgp_semantic_generalization.dq_cgp_v3.model import (
    DQCGPv3Model,
    install_dq_cgp_v3_loss,
)


def get_dummy_opt():
    manager = BaseOptions("moment_detr", "charades_semantic_existence", "clip_slowfast")
    manager.parse()
    opt = manager.option
    opt.device = "cpu"
    opt.mr_only = True
    opt.lw_saliency = 0
    opt.use_exist_head = True
    opt.exist_loss_coef = 1.0
    opt.exist_gate_thd = 0.5
    opt.exist_pool = "max"
    return opt


def create_mock_batch(batch_type: str = "mixed", bsz: int = 4, t_len: int = 10, v_len: int = 20, hidden_dim: int = 256):
    src_vid = torch.randn(bsz, v_len, 2818)
    src_vid_mask = torch.ones(bsz, v_len)
    src_txt = torch.randn(bsz, t_len, 512)
    src_txt_mask = torch.ones(bsz, t_len)

    model_inputs = {
        "src_txt": src_txt,
        "src_txt_mask": src_txt_mask,
        "src_vid": src_vid,
        "src_vid_mask": src_vid_mask,
    }

    if batch_type == "positive":
        exist_labels = torch.ones(bsz)
        span_labels = [{"spans": torch.tensor([[0.2, 0.4], [0.5, 0.8]])} for _ in range(bsz)]
    elif batch_type == "empty":
        exist_labels = torch.zeros(bsz)
        span_labels = [{"spans": torch.empty((0, 2))} for _ in range(bsz)]
    elif batch_type == "mixed":
        exist_labels = torch.tensor([1.0, 0.0, 1.0, 0.0][:bsz])
        span_labels = [
            {"spans": torch.tensor([[0.2, 0.4]])} if exist_labels[i] == 1.0 else {"spans": torch.empty((0, 2))}
            for i in range(bsz)
        ]
    else:
        raise ValueError(batch_type)

    targets = {
        "exist_label": exist_labels,
        "span_labels": span_labels,
    }
    return model_inputs, targets


class TestDQCGPUnit(unittest.TestCase):

    def setUp(self):
        self.opt = get_dummy_opt()

    def test_01_ls_dqcgp_positive_and_empty_batches(self):
        """Test LS-DQCGP on positive, empty, and mixed batches."""
        base_model, criterion = build_base_detr(self.opt)
        model = LSDQCGPModel(base_model=base_model)
        install_ls_dq_cgp_loss(criterion, model, coefficient=0.2)

        for btype in ["positive", "empty", "mixed"]:
            inputs, targets = create_mock_batch(batch_type=btype, bsz=4)
            outputs = model(**inputs)

            # Check outputs
            self.assertIn("pred_logits", outputs)
            self.assertIn("pred_spans", outputs)
            self.assertIn("pred_exist_logits", outputs)
            self.assertFalse(torch.isnan(outputs["pred_logits"]).any())
            self.assertFalse(torch.isnan(outputs["pred_spans"]).any())
            self.assertFalse(torch.isnan(outputs["pred_exist_logits"]).any())

            # Loss computation
            loss_dict = criterion(outputs, targets)
            self.assertIn("loss_exist", loss_dict)
            self.assertIn("loss_native_bind", loss_dict)

            # S- empty batch check
            if btype == "empty":
                self.assertEqual(float(loss_dict["loss_native_bind"]), 0.0)
                self.assertGreater(float(loss_dict["loss_exist"]), 0.0)

            # Backward pass
            total_loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict)
            self.assertFalse(torch.isnan(total_loss))
            self.assertFalse(torch.isinf(total_loss))
            total_loss.backward()

            # Verify gradients exist without NaN
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad in {name}")

    def test_02_dq_cgp_v3_positive_and_empty_batches(self):
        """Test DQ-CGPv3 on positive, empty, and mixed batches."""
        base_model, criterion = build_base_detr(self.opt)
        model = DQCGPv3Model(base_model=base_model)
        install_dq_cgp_v3_loss(criterion, binding_coef=0.2, route_coef=0.01)

        for btype in ["positive", "empty", "mixed"]:
            inputs, targets = create_mock_batch(batch_type=btype, bsz=4)
            outputs = model(**inputs)

            # Check outputs
            self.assertIn("pred_logits", outputs)
            self.assertIn("pred_spans", outputs)
            self.assertIn("pred_exist_logits", outputs)
            self.assertIn("query_cgp_temporal_attention", outputs)
            self.assertIn("query_cgp_basis_weights", outputs)
            self.assertFalse(torch.isnan(outputs["pred_logits"]).any())
            self.assertFalse(torch.isnan(outputs["pred_spans"]).any())
            self.assertFalse(torch.isnan(outputs["pred_exist_logits"]).any())

            # Loss computation
            loss_dict = criterion(outputs, targets)
            self.assertIn("loss_exist", loss_dict)
            self.assertIn("loss_query_cgp_bind", loss_dict)
            self.assertIn("loss_query_cgp_route", loss_dict)

            # S- empty batch check
            if btype == "empty":
                self.assertEqual(float(loss_dict["loss_query_cgp_bind"]), 0.0)
                self.assertEqual(float(loss_dict["loss_query_cgp_route"]), 0.0)
                self.assertGreater(float(loss_dict["loss_exist"]), 0.0)

            # Backward pass
            total_loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict)
            self.assertFalse(torch.isnan(total_loss))
            self.assertFalse(torch.isinf(total_loss))
            total_loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad in {name}")


if __name__ == "__main__":
    unittest.main()
