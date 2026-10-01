import importlib
import unittest
import warnings

import torch


RUN_MODULE = importlib.import_module("models.flashvtg_dq_cgp_v3_gmr.run")


class VariantSchedulerTest(unittest.TestCase):
    def test_metric_scheduler_detaches_loss_and_reduces_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=3e-5)
        scheduler = RUN_MODULE.DetachedReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            scheduler.step(torch.tensor(1.0, requires_grad=True))
            scheduler.step(torch.tensor(1.1, requires_grad=True))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.5e-5)


if __name__ == "__main__":
    unittest.main()
