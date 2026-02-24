# Andy Zhao
"""
Translation-only motion model for RANSAC stabilization.

Assume: Every tracked feature moves by the same displacement.
    pts1 ≈ pts0 + t
    t = (tx, ty)

This is ideal for:
    - Vertical jitter correction
    - Small handheld shake
    - Fast stabilization

This model has only 2 degrees of freedom:
    tx (horizontal shift)
    ty (vertical shift)
"""

import numpy as np
from .types import Points2D, Mat3x3, FloatArray

def _make_T(tx: float, ty: float) -> Mat3x3:
    """
    Construct a 3x3 homogeneous transformation matrix
    representing pure translation.

    Homogeneous matrix form:

        [ 1   0   tx ]
        [ 0   1   ty ]
        [ 0   0    1 ]

    This allows compatibility with other affine models.
    """
    # Start with identity matrix (no transformation)
    T = np.eye(3, dtype=np.float64)

    # Insert translation into last column
    T[0, 2] = tx    # horizontal shift
    T[1, 2] = ty    # vertical shift
    return T


# Minimal fit (used inside RANSAC hypothesis step)
def fit_translation_minimal(pts0: Points2D, pts1: Points2D) -> Mat3x3 | None:
    """
    Minimal sample estimator for translation.

    Translation model only needs 1 correspondence: t = pts1 - pts0
        p1 = p0 + t → t = p1 - p0

    Parameters:
    - pts0 : (1,2)
    - pts1 : (1,2)

    Returns:
    - 3x3 translation matrix
    """
    # RANSAC might pass insufficient samples
    if pts0.shape[0] < 1:
        return None

    # Compute displacement vector for this single point
    displacement = pts1[0] - pts0[0]

    tx = float(displacement[0])     # x displacement
    ty = float(displacement[1])     # y   displacement

    # Convert to homogeneous transform
    return _make_T(tx=tx, ty=ty)


# Least squares refit (used after inliers found)
def fit_translation_least_squares(pts0: Points2D, pts1: Points2D) -> Mat3x3 | None:
    """
    Least-squares estimator for translation.

    Once RANSAC finds inliers, recompute the best translation using ALL inliers.

    pts1_i = pts0_i + t
    → t = average(pts1_i - pts0_i)

    So compute the mean displacement.
    """
    # Must have at least one point
    if pts0.shape[0] < 1:
        return None

    # Compute displacement vectors for ALL correspondences
    # Shape: (N, 2)
    displacements = pts1 - pts0

    # Average displacement gives least-squares solution
    tx = float(np.mean(displacements[:, 0]))
    ty = float(np.mean(displacements[:, 1]))

    return _make_T(tx=tx, ty=ty)


# Residual computation (used by RANSAC inlier test)
def residuals_L2_translation(
    T: Mat3x3,
    pts0: Points2D,
    pts1: Points2D
) -> FloatArray:
    """
    Compute L2 residual per correspondence.

    For each point:
        predicted_p1 = pts0 + t
        error = || predicted_p1 - pts1 ||

    Returns:
    - residuals : (N,)
        Euclidean distance per correspondence
    """
    # Extract translation components from matrix
    tx = float(T[0, 2])
    ty = float(T[1, 2])

    # Apply translation to pts0
    predicted_pts1 = pts0 + np.array([tx, ty], dtype=np.float64)

    # Compute difference between prediction and actual pts1
    diff = predicted_pts1 - pts1

    # L2 norm per row
    residuals = np.linalg.norm(diff, axis=1)
    return residuals










