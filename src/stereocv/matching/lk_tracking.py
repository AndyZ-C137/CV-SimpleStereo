# Andy Zhao
"""
Lucas–Kanade (LK) optical flow tracking wrapper

- RANSAC expects pts0, pts1 as (N,2) float arrays, and a status mask
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from ..ransac.types import Points2D

@dataclass(frozen=True)
class LKParams:
    """
    Parameters for pyramidal Lucas–Kanade tracking.

    winSize:
      - Size of the search window at each pyramid level.
      - Larger handles bigger motions but can be less precise.

    maxLevel:
      - Number of pyramid levels (0 means single-scale).
      - More levels helps with larger motions.

    criteria:
      - (type, max_iter, epsilon)
      - stops when either max_iter reached or the update is small enough.
    """
    winSize: Tuple[int, int] = (21, 21)
    maxLevel: int = 3
    criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def lk_track(
    gray0: np.ndarray,
    gray1: np.ndarray,
    pts0: Points2D,
    *,
    params: LKParams = LKParams(),
) -> tuple[Points2D, Points2D, np.ndarray, np.ndarray]:
    """
    Track points from gray0 -> gray1.

    Inputs:
        gray0, gray1:
          - Grayscale images (H,W), uint8 preferred.
          - If original image is color, need to convert to gray before tracking

        pts0:
          - (N,2) float array of keypoints in frame 0.

    Returns:
        pts0_out:
          - (N,2) float64 copy of input points

        pts1:
          - (N,2) float64 tracked points in frame 1
          - For failed tracks, pts1 returned as 0

        status:
          - (N, ) uint8 array: 1 = success, 0 = fail

        err:
          - (N, ) float32/float64: per-point tracking error from OpenCV
    """
    # ---------- Input Check ----------
    if gray0.ndim != 2 or gray1.ndim != 2:
        raise ValueError("lk_track expects grayscale frames (H,W). Convert BGR->gray before calling.")

    pts0 = np.asarray(pts0, dtype=np.float32)
    if pts0.ndim != 2 or pts0.shape[1] != 2:
        raise ValueError(f"pts0 must be (N,2), got {pts0.shape}")

    # Match OpenCV expected point shape (N,1,2)
    p0 = pts0.reshape(-1, 1, 2)

    # ---------- Run LK ----------
    p1, status, err = cv2.calcOpticalFlowPyrLK(
        gray0,
        gray1,
        p0,
        None,
        winSize=params.winSize,
        maxLevel=params.maxLevel,
        criteria=params.criteria,
    )

    if p1 is None or status is None:
        # If LK fail, return arrays with all status=0
        n = pts0.shape[0]
        pts1 = pts0.astype(np.float64, copy=True)
        status_out = np.zeros((n,), dtype=np.uint8)
        err_out = np.full((n,), np.inf, dtype=np.float64)
        return pts0.astype(np.float64), pts1, status_out, err_out

    # Convert back to (N,2)
    pts1 = p1.reshape(-1, 2).astype(np.float64)
    st_out = status.reshape(-1).astype(np.uint8)
    err_out = err.reshape(-1).astype(np.float64)

    return pts0.astype(np.float64), pts1, st_out, err_out



