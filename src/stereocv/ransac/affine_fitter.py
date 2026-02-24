# Andy Zhao
"""
Adapter: makes affine functions conform to the ModelFitter Protocol.

This keeps ransac/core.py generic and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import Points2D, Mat3x3, FloatArray
from .types import ModelFitter
from .affine import fit_affine_minimal, fit_affine_least_squares, residuals_L2


@dataclass(frozen=True)
class AffineFitter(ModelFitter[Mat3x3]):
    eps_area: float = 1e-6

    def fit_minimal(self, pts0: Points2D, pts1: Points2D) -> Optional[Mat3x3]:
        return fit_affine_minimal(pts0, pts1, eps_area=self.eps_area)

    def fit_least_squares(self, pts0: Points2D, pts1: Points2D) -> Optional[Mat3x3]:
        return fit_affine_least_squares(pts0, pts1)

    def residuals(self, model: Mat3x3, pts0: Points2D, pts1: Points2D) -> FloatArray:
        return residuals_L2(model, pts0, pts1)