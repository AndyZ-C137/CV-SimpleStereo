# Andy Zhao
"""
Utilities for cleaning correspondence sets before robust estimation.

Remove:
- LK failures (st==0)
- NaNs/Infs
- extreme motion outliers (helps stability and speed)
"""

from __future__ import annotations
import numpy as np
from ..ransac.types import Points2D,BoolArray


def clean_points(
    pts0: Points2D,
    pts1: Points2D,
    *,
    status: np.ndarray | None = None,
    max_motion_px: float | None = None,
) -> tuple[Points2D, Points2D, BoolArray]:
    pts0 = np.asarray(pts0, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64)

    if pts0.ndim != 2 or pts1.ndim != 2 or pts0.shape != pts1.shape or pts0.shape[1] != 2:
        raise ValueError(f"Expected pts0/pts1 shape (N,2) matching; got {pts0.shape} vs {pts1.shape}")

    mask = np.ones((pts0.shape[0],), dtype=bool)

    # LK status filter
    if status is not None:
        status = np.asarray(status).reshape(-1)
        if status.shape[0] != pts0.shape[0]:
            raise ValueError(f"status must have length N; got {status.shape[0]} vs {pts0.shape[0]}")
        mask &= (status.astype(np.uint8) == 1)

    # Check if points are finite
    mask &= np.isfinite(pts0).all(axis=1)
    mask &= np.isfinite(pts1).all(axis=1)

    # big-jump pruning
    if max_motion_px is not None:
        motion = np.linalg.norm(pts1 - pts0, axis=1)
        # diff = pts1 - pts0
        # motion = np.sqrt(np.sum(diff * diff, axis=1))
        mask &= motion <= float(max_motion_px)

    return pts0[mask], pts1[mask], mask

