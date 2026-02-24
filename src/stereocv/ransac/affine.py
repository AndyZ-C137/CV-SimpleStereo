# Andy Zhao
"""
Affine model utilities (3x3 homogeneous form).

We estimate an affine transform T such that:

    [x', y', 1]^T  ≈  T @ [x, y, 1]^T

where:

    T = [[a, b, tx],
         [c, d, ty],
         [0, 0,  1]]

Unknowns are 6 parameters: a, b, tx, c, d, ty.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .types import (
    Points2D, PointsHomog, Mat3x3, FloatArray,
    as_homogeneous, is_valid_mat3x3)


# ---------- Degeneracy Check Helpers ----------
def _triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Return 2x the triangle area formed by (p1, p2, p3).
    Compute the magnitude of the 2D cross product:

        area2 = |(p2 - p1) x (p3 - p1)|

    If area2 is near 0, the three points are collinear (degenerate for affine minimal fit).
    """
    u = p2 - p1
    v = p3 - p1

    # In 2D, "cross product magnitude" is a scalar
    return float(abs(u[0] * v[1] - u[1] * v[0]))


def _is_degenerate_triplet(pts: Points2D, eps_area: float = 1e-6) -> bool:
    """
    Check whether 3 points (shape (3,2)) are nearly collinear.

    eps_area2 is threshold on 2x area. You may tune later.
    """
    if pts.shape != (3, 2):
        raise ValueError(f"Expected (3,2) triplet, got {pts.shape}")

    area = _triangle_area(pts[0], pts[1], pts[2])
    return area < eps_area


# ---------- Affine Fitting ----------
def _theta_to_mat3x3(theta: np.ndarray) -> Mat3x3:
    """
    Convert parameter vector theta = [a, b, tx, c, d, ty] into a 3x3 affine matrix.
    """
    a, b, tx, c, d, ty = map(float, theta.tolist())
    T = np.array(
        [
            [a, b, tx],
            [c, d, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T


def fit_affine_minimal(pts0: Points2D, pts1: Points2D, eps_area: float = 1e-6) -> Optional[Mat3x3]:
    """
    Fit affine transform from exactly 3 point correspondences.

    pts0: (3,2) source points
    pts1: (3,2) target points

    Returns:
      3x3 affine matrix, or None if degenerate / solve fails.
    """

    # Validate shapes
    if pts0.shape != (3, 2) or (pts1.shape != (3, 2)):
        raise ValueError(f"fit_affine_minimal expects (3,2) inputs, got {pts0.shape} and {pts1.shape}")

    # Degeneracy check:
    # If any triplet is collinear, the affine solve is not uniquely determined.
    if _is_degenerate_triplet(pts0, eps_area) or _is_degenerate_triplet(pts1, eps_area=eps_area):
        return None

    # Build a linear system A theta = b with 6 unknowns :
    # For each point:
    #   A = [[x, y, 1, 0, 0, 0],
    #        [0, 0, 0, x, y, 1]]

    # For each correspondence (x, y) -> (x', y'):
    #   x' = a*x + b*y + tx
    #   y' = c*x + d*y + ty
    #
    # Unknown vector:
    #   theta = [a, b, tx, c, d, ty]^T
    #
    # Each point gives 2 equations, so 3 points gives 6 equations -> can solve affine fitting if non-degenerate.
    A = np.zeros((6, 6), dtype=np.float64)
    b_vec = np.zeros((6,), dtype=np.float64)

    for i in range (3):
        x, y = float(pts0[i, 0]), float(pts0[i, 1])
        x_prime, y_prime = float(pts1[i, 0]), float(pts1[i, 1])

        # Row for x' equation
        # x' = a*x + b*y + tx
        # Each point gives 2 rows, one for x, one for y
        A[2 * i + 0, :] = [x, y, 1.0, 0, 0, 0]
        b_vec[2 * i + 0] = x_prime

        A[2 * i + 1, :] = [0, 0, 0, x, y, 1.0]
        b_vec[2 * i + 1] = y_prime

    # Use np.linalg.solve since A is square (6x6).
    # If A is singular (numerically), this will throw.
    try:
        theta = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        return None

    T = _theta_to_mat3x3(theta)
    if not is_valid_mat3x3(T):
        return None
    return T

def fit_affine_least_squares(pts0: Points2D, pts1: Points2D) -> Optional[Mat3x3]:
    """
    Fit affine transform from N >= 3 correspondences using least squares.

    This is used after RANSAC picks inliers: refit with all inliers for best estimate.

    Uses np.linalg.lstsq(A, b): finds theta that minimizes ||A theta - b||^2.
    """
    if pts0.shape != pts1.shape:
        raise ValueError(f"pts0 and pts1 must have same shape, got {pts0.shape} vs {pts1.shape}")
    if pts0.ndim != 2 or pts0.shape[1] != 2:
        raise ValueError(f"Expected pts shape (N,2), got {pts0.shape}")
    if pts0.shape[0] < 3:
        return None

    n = pts0.shape[0]

    # A is (2N x 6), b is (2N,)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    bvec = np.zeros((2 * n,), dtype=np.float64)

    # Fill the linear system row-by-row
    for i in range(n):
        x, y = float(pts0[i, 0]), float(pts0[i, 1])
        x_prime, y_prime = float(pts1[i, 0]), float(pts1[i, 1])

        A[2 * i + 0, :] = [x, y, 1.0, 0.0, 0.0, 0.0]
        bvec[2 * i + 0] = x_prime

        A[2 * i + 1, :] = [0.0, 0.0, 0.0, x, y, 1.0]
        bvec[2 * i + 1] = y_prime

    # Least squares solve:
    # theta minimizes squared residual error across all correspondences.
    try:
        # rank: how many independent constraints we actually have
        # small singular values -> near-degenerate
        theta, residuals, rank, singular_vals = np.linalg.lstsq(A, bvec, rcond=None)
    except np.linalg.LinAlgError:
        return None

    # Affine has 6 unknowns.
    # To determine them uniquely/stably, the linear system must provide 6
    # If rank is too low, the data doesn't constrain the model well.
    #   - Points are collinear: the data can’t see 2D shear/rotation properly.
    #   - Not enough variation in points:
    #     - all points nearly on a line
    #     - repeated points
    #     - points clustered too tightly.
    #   - Bad numeric conditioning: A becomes ill-conditioned (rank might still be 6 but small singular values)
    #     - points are extremely large or extremely small
    #     - points are nearly degenerate
    if rank < 6:
        return None

    T = _theta_to_mat3x3(theta)
    if not is_valid_mat3x3(T):
        return None
    return T


# ---------- Apply transform + residuals ----------
def apply_T(T: Mat3x3, pts: Points2D) -> Points2D:
    """
    Apply a 3x3 transform to (N,2) points, returning (N,2) points.

    For affine:
        [x', y', 1]^T = T @ [x, y, 1]^T

    For homography:
        divide by w (for affine w is always 1.)
    """
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected pts shape (N,2), got {pts.shape}")
    if T.shape != (3, 3):
        raise ValueError(f"Expected T shape (3,3), got {T.shape}")

    # Convert points to homogeneous coordinate (N,3)
    ph: PointsHomog = as_homogeneous(pts)

    # Apply transform: (N,3) = (N,3) @ (3,3)^T
    # Each point is a row, so multiply by T^T to get transformed homogeneous point
    ph_t = ph @ T.T  # shape (N,3)

    # For affine, ph_t[:, 2] should all be 1
    # {
    #   [x1', y1', 1],
    #   [x2', y2', 1],
    #   ...
    #  }
    # Return x,y as float64 array
    out = ph_t[:, :2]
    return out.astype(np.float64)


def residuals_L2(T: Mat3x3, pts0: Points2D, pts1: Points2D) -> Points2D:
    """
    Compute per-point L2 residuals in pixels:

        e_i = || apply_T(T, pts0[i]) - pts1[i] ||_2

    Returns shape (N,)
    """
    if pts0.shape != pts1.shape:
        raise ValueError(f"pts0 and pts1 must have same shape, got {pts0.shape} vs {pts1.shape}")
    predicted = apply_T(T, pts0)
    diff = predicted - pts1.astype(np.float64)
    return np.linalg.norm(diff, axis=1).astype(np.float64)
    # return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float64)


