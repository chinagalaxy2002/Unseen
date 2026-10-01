"""Copied FlashVTG blocks used internally by the DQ-CGP variant.

The variant reuses the released FlashVTG nncore registry entries rather than
registering a second class with the same names.  This keeps the independent
model loadable alongside the baseline training package in one Python process.
"""

from .generator import PointGenerator

__all__ = ["PointGenerator"]
