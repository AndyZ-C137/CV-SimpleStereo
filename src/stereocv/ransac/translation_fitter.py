# Andy Zhao
"""
Adapter class for translation-only model to match ModelFitter protocol.

This allows using your existing generic RANSAC implementation
without modification.
"""

from dataclasses import dataclass
from typing import Optional

from .types import Points2D, Mat3x3, FloatArray, ModelFitter

from .translation import (
    fit_translation_minimal,
    fit_translation_least_squares,
    residuals_L2_translation,
)


@dataclass(frozen=True)
class TranslationFitter(ModelFitter[Mat3x3]):
    """
    Translation-only motion model for RANSAC.

    This class plugs into your existing RANSAC engine.
    """

    def fit_minimal(
        self,
        pts0: Points2D,
        pts1: Points2D
    ) -> Optional[Mat3x3]:
        """
        Called by RANSAC during hypothesis generation.

        Uses minimal 1-point estimator.
        """
        return fit_translation_minimal(pts0, pts1)

    def fit_least_squares(
        self,
        pts0: Points2D,
        pts1: Points2D
    ) -> Optional[Mat3x3]:
        """
        Called by RANSAC after inliers are selected.

        Refits model using all inliers.
        """
        return fit_translation_least_squares(pts0, pts1)

    def residuals(
        self,
        model: Mat3x3,
        pts0: Points2D,
        pts1: Points2D
    ) -> FloatArray:
        """
        Computes error per point for inlier test.

        Used inside RANSAC threshold comparison:
            residual < tau
        """
        return residuals_L2_translation(model, pts0, pts1)