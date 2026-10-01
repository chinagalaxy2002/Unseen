"""Training/inference entry point without modifying FlashVTG baseline files.

Examples::

    python -m models.flashvtg_dq-cgp-gmr-v2.run train \
        models/flashvtg_dq-cgp-gmr-v2/model_config.py [FlashVTG arguments]

    python -m models.flashvtg_dq-cgp-gmr-v2.run infer \
        models/flashvtg_dq-cgp-gmr-v2/model_config.py [FlashVTG arguments]

The data loops and evaluation implementation are reused read-only from the
released FlashVTG-GMR training package.  Only their model factory is replaced
at runtime by the independent variant defined in this directory.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime

import nncore
# nncore.Config.__getattr__ raises KeyError instead of AttributeError on missing keys,
# breaking hasattr() and pandas inspection. Safely convert KeyError to AttributeError.
_orig_config_getattr = nncore.Config.__getattr__
def _safe_config_getattr(self, key):
    try:
        return _orig_config_getattr(self, key)
    except KeyError:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
nncore.Config.__getattr__ = _safe_config_getattr
import torch

from .model import build_model1, load_flashvtg_baseline_state


LOGGER = logging.getLogger(__name__)


def _config_value(opt, name, default):
    model_cfg = getattr(getattr(opt, "cfg", None), "model", None)
    if model_cfg is None:
        return default
    if isinstance(model_cfg, dict):
        return model_cfg.get(name, default)
    return getattr(model_cfg, name, default)


def _configure_variant_training(model, opt):
    """Freeze the warm-started baseline and expose only DQ-CGP parameters."""

    freeze_baseline = bool(
        _config_value(opt, "freeze_flashvtg_baseline", True)
    )
    if freeze_baseline:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("dq_cgp."))

    threshold = float(_config_value(opt, "gmr_decision_threshold", 0.5))
    opt.exist_gate_thd = threshold
    opt.pred_score_thd_for_cls = threshold

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("variant training has no trainable parameters")
    LOGGER.info(
        "Trainable tensors: %d; freeze_flashvtg_baseline=%s; GMR threshold=%.3f",
        len(trainable),
        freeze_baseline,
        threshold,
    )
    return trainable


def _patch_baseline_inference_runtime():
    """Supply a symbol missing from the released GMR inference module.

    ``compute_mr_results`` constructs ``defaultdict(AverageMeter)``, but the
    baseline module does not import ``AverageMeter``.  Keep the baseline tree
    read-only and inject the dependency from this variant's launcher instead.
    """

    variant_dir = os.path.dirname(os.path.abspath(__file__))
    if variant_dir not in sys.path:
        sys.path.insert(0, variant_dir)

    from models.flash_vtg_gmr.utils.basic_utils import AverageMeter
    from training.flash_vtg_gmr import inference as baseline_inference

    baseline_inference.AverageMeter = AverageMeter
    return baseline_inference


def _strip_distributed_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def setup_model(opt):
    """Build the variant and load either variant or baseline checkpoints."""

    model, criterion = build_model1(opt)
    if opt.device.type == "cuda":
        model.to(opt.device)
        criterion.to(opt.device)

    trainable_parameters = _configure_variant_training(model, opt)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=opt.lr,
        weight_decay=opt.wd,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, opt.lr_drop, gamma=0.5
    )

    if opt.resume is not None:
        LOGGER.info("Loading checkpoint from %s", opt.resume)
        checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict"))
        if state is None:
            raise KeyError("checkpoint must contain 'model' or 'state_dict'")
        state = _strip_distributed_prefix(state)
        is_variant = any(key.startswith("dq_cgp.") for key in state)
        if is_variant:
            model.load_state_dict(state, strict=True)
        else:
            missing = load_flashvtg_baseline_state(model, state)
            LOGGER.info(
                "Warm-started from FlashVTG baseline; initialized %d DQ-CGP tensors",
                len(missing),
            )

        if opt.resume_all:
            if not is_variant:
                raise ValueError(
                    "--resume_all is invalid for a baseline warm-start because "
                    "the optimizer has no DQ-CGP parameter state"
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            opt.start_epoch = checkpoint["epoch"] + 1

    return model, criterion, optimizer, lr_scheduler


def train_main():
    """Run the existing FlashVTG training loop with the variant model factory."""

    import torch.backends.cudnn as cudnn
    from training.flash_vtg_gmr import train as baseline_train
    from training.flash_vtg_gmr.config import BaseOptions

    _patch_baseline_inference_runtime()

    opt = BaseOptions().parse()
    baseline_train.set_seed(opt.seed)
    if opt.debug:
        cudnn.benchmark = False
        cudnn.deterministic = True
    opt.cfg = nncore.Config.from_file(opt.config)

    logger = logging.getLogger("flashvtg_dq_cgp_gmr.train")
    logger.setLevel(logging.INFO)
    log_path = os.path.join(
        opt.results_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # The reused functions resolve these names from their defining module.
    baseline_train.opt = opt
    baseline_train.logger = logger
    baseline_train.setup_model = setup_model
    return baseline_train.start_training()


def inference_main():
    """Run released FlashVTG evaluation with the variant model factory."""

    baseline_inference = _patch_baseline_inference_runtime()

    baseline_inference.setup_model = setup_model
    return baseline_inference.start_inference()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "infer"}:
        raise SystemExit(
            "usage: python -m models.flashvtg_dq-cgp-gmr-v2.run "
            "{train|infer} MODEL_CONFIG [FlashVTG arguments]"
        )
    mode = sys.argv.pop(1)
    if mode == "train":
        train_main()
    else:
        inference_main()


if __name__ == "__main__":
    main()
