# Andy Zhao
"""
Corners / keypoint detection

Shi–Tomasi via cv2.goodFeaturesToTrack.
FAST / ORB / Harris can be added
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

from ..ransac.types import Points2D


@dataclass(frozen=True)
class ShiTomasiParams:
    """
    Parameters for Shi–Tomasi corner detection (goodFeaturesToTrack).

    maxCorners:
      - Upper bound on number of corner_params returned.
    qualityLevel:
      - Rejects corner_params with response < qualityLevel * best_response.
    minDistance:
      - Minimum allowed distance between detected corner_params.
    blockSize:
      - Size of neighborhood used for corner score.
    """
    maxCorners: int = 400
    qualityLevel: float = 0.01
    minDistance: float = 7.0
    blockSize: int = 7


def shitomasi_detect(
        gray: np.ndarray,
        *,
        params: ShiTomasiParams = ShiTomasiParams(),
) -> Points2D:
    """
    Detect Shi–Tomasi corner_params on a grayscale image.

    Input:
      gray: (H,W) grayscale image (uint8 preferred)
    Output:
      pts: (N,2) float64 points
    """
    if gray.ndim != 2:
        raise ValueError("detect_shitomasi expects grayscale (H,W).")

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=params.maxCorners,
        qualityLevel=params.qualityLevel,
        minDistance=params.minDistance,
        blockSize=params.blockSize,
    )

    if pts is None:
        return np.zeros((0, 2), dtype=np.float64)

    return pts.reshape(-1, 2).astype(np.float64)