"""FlashVTG-DQ-CGP-GMR v2 model package.

The directory name follows the requested experiment name and contains a
hyphen.  Import it with :mod:`importlib`, for example::

    import importlib
    model_module = importlib.import_module(
        "models.flashvtg_dq-cgp-gmr-v2.model"
    )
"""

from .dq_cgp import FlashPointDQCGP, FlashPointDQCGPOutput

__all__ = ["FlashPointDQCGP", "FlashPointDQCGPOutput"]

from .model import build_model1

__all__ = ["build_model1"]
