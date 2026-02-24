# Andy Zhao

"""
Shared typed primitives for the stereo/RANSAC pipeline.

Defines:
- Typed NumPy aliases for geometry
    - Points are (N,2) float arrays
    - Transforms are 3x3 homogeneous matrices
- Generic model protocol for RANSAC
- Structured RANSAC result container(model + inliers + stats)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Optional, TypeAlias

import numpy as np
import numpy.typing as npt

# ---------- Numpy typing aliases ----------
# standardize numeric dtypes so bugs are easier to spot and code is consistent.
# - float64 for geometry / matrices (more stable for linear algebra)
# - bool_ for masks

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# Points in 2D image coordinates. Stored as float64 for consistency in math.
Points2D: TypeAlias = FloatArray      # shape: (N, 2)

# Homogeneous points [x, y, 1] for 3x3 transforms (affine/homography).
PointsHomog: TypeAlias = FloatArray   # shape: (N, 3)

# Boolean inlier mask: True as inlier, False as outlier
Mask2D: TypeAlias = BoolArray         # shape: (N,)

# 3x3 homogeneous transform matrix.
# Affine is represented as 3x3 with last row [0,0,1].
Mat3x3: TypeAlias = FloatArray        # shape: (3, 3)

# ---------- Generic model typing ----------
# For affine/homography, will use Mat3x3
# Keeping it generic to make RANSAC reusable.
M = TypeVar("M")


class ModelFitter(Protocol[M]):
    """
    Interface that a model must implement to be usable by the generic RANSAC implementation.

    RANSAC steps:
    1) Fit a model from a minimal sample
    2) Refit a better model from all inliers (least squares)
    3) Score all correspondences with a per-point residual error
    """

    def fit_minimal(self, pts0: Points2D, pts1: Points2D) -> Optional[M]:
        """
        Fit from the minimal number of correspondences required.
        Return None if the sample is degenerate (e.g., points collinear for affine).
        """
        ...

    def fit_least_squares(self, pts0: Points2D, pts1: Points2D) -> Optional[M]:
        """
        Refit the model using all inliers.
        Return None if the set is degenerate or the solve fails.
        """
        ...

    def residuals(self, model: M, pts0: Points2D, pts1: Points2D) -> FloatArray:
        """
        Return a vector of residual errors, one per correspondence.
        Shape: (N,). Smaller = better.
        """
        ...


# ---------- RANSAC output container ----------
# A typed result struct to store RANSAC output
# frozen=True means "immutable" after construction
@dataclass(frozen=True)
class RansacResult(Generic[M]):
    model: M            # best model found (e.g., 3x3 affine matrix)
    inliers: Mask2D     # boolean mask of inliers under the best model
    num_inliers: int    # count of True values in inliers
    rms_error: float    # RMS error of inliers under the refit model
    iterations: int     # how many RANSAC iterations were actually run
    threshold: float    # the inlier threshold tau used


# ---------- Helper Function ----------
def as_homogeneous(pts: Points2D) ->PointsHomog:
    """
    Convert (N,2) points -> (N,3) homogeneous points: [x, y, 1].
    - 3x3 transforms (affine/homography) are easiest to apply to homogeneous points.
    """
    # Validate shape
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points shape (N, 2) but got {pts.shape}")

    # Create a column filled with 1 for the homogeneous coordinate.
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)

    # Horizontally stack [x, y] with [1].
    # Ensure float64
    return np.hstack([pts.astype(np.float64), ones])


def is_valid_mat3x3(T: Mat3x3) -> bool:
    """
    Verify a 3x3 transform matrix.
    Used for rejecting failed fits.
    """
    return isinstance(T, np.ndarray) and T.shape == (3, 3) and np.isfinite(T).all()
