# Andy Zhao
"""
RANSAC package

This module provides:
- A reusable generic RANSAC implementation
- Typed geometry primitives
- Model interface definitions
- Support for affine and homography models
"""

from .types import (
    FloatArray, BoolArray, Points2D, PointsHomog, Mask2D, Mat3x3,
    ModelFitter, RansacResult, as_homogeneous, is_valid_mat3x3,
)

from .affine import (
    fit_affine_minimal, fit_affine_least_squares, apply_T, residuals_L2,
)

from .affine_fitter import AffineFitter

from .core import ransac

from .translation import (
    fit_translation_minimal, fit_translation_least_squares, residuals_L2_translation,
)

from .translation_fitter import TranslationFitter

# from .homography import HomographyModel

__all__ = [
    "FloatArray", "BoolArray", "Points2D", "PointsHomog", "Mask2D", "Mat3x3",
    "ModelFitter", "RansacResult", "as_homogeneous", "is_valid_mat3x3",
    "fit_affine_minimal", "fit_affine_least_squares", "apply_T", "residuals_L2",
    "AffineFitter",
    "ransac",
    "fit_translation_minimal", "fit_translation_least_squares", "residuals_L2_translation",
    "TranslationFitter",
]
