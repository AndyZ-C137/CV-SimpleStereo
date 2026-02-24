# Andy Zhao

"""
Provide a clean wrapper for warping frames using OpenCV.

RANSAC modules represent motion as 3x3 homogeneous matrices (Mat3x3).
Great for geometry and consistent across:
  - translation model
  - affine model

OpenCV has two different warping APIs:
1) cv2.warpAffine:
    - expects a 2x3 matrix (affine transform)
    - used for translation / rotation / scale / shear (no projective)
2) cv2.warpPerspective:
    - expects a 3x3 matrix (homography)
    - used for projective transforms
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2

from ..ransac.types import Mat3x3

# ---------- Warp parameters ----------
@dataclass(frozen=True)
class WarpParams:
    """
    These settings control how OpenCV fills pixels that get "exposed"
    when the frame shifts.

    Parameters:
    - border_mode:
      OpenCV border mode constant.
        - cv2.BORDER_CONSTANT: fill with a constant color (border_value)
        - cv2.BORDER_REFLECT: mirror reflect at edge
        - cv2.BORDER_REPLICATE: repeat edge pixels
        - cv2.BORDER_WRAP: wrap around

    - border_value:
      Used only when border_mode == cv2.BORDER_CONSTANT.
      For BGR frames, this should be a 3-tuple like (0,0,0).

    - interpolation:
      Controls resampling when warping.
      - cv2.INTER_LINEAR: good default for video.
      - cv2.INTER_NEAREST: faster but blocky.
      - cv2.INTER_CUBIC: smoother but slower.
    """
    border_mode: int = cv2.BORDER_REFLECT
    border_value: Tuple[int, int, int] = (0, 0, 0)
    interpolation: int = cv2.INTER_LINEAR


# ---------- Affine warp helper ----------
def warp_frame_affine(
        frame: np.ndarray,
        T: Mat3x3,
        *,
        params: WarpParams,
) -> np.ndarray:
    """
    Warp an image using the affine part of a 3x3 homogeneous transform.

    - frame:
      - Image array (H x W x 3) for BGR, or (H x W) for grayscale.
      - We treat it generically; OpenCV warpAffine supports both.
    - T:
      - 3x3 homogeneous matrix.
      - For affine transforms, last row should be [0, 0, 1].
      - The first two rows are passed to OpenCV:
            A = T[:2, :]  (shape 2x3)

    params:
    - WarpParams controlling border behavior and interpolation

    Output: warped frame with same shape as input.

    Notes:
    ------
    - Stabilization is typically a small transform each frame (a few pixels).
    - warpAffine is fast and appropriate for this.
    """
    # ---------- Basic validation ----------
    if frame is None or frame.size == 0:
        # Nothing to do
        return frame

    if T.shape != (3, 3):
        raise ValueError(f"warp_frame_affine expected T shape (3,3), got {T.shape}")

    # OpenCV wants output size (width, height)
    H, W = frame.shape[:2]

    # OpenCV warpAffine expects 2x3 matrix
    A = T[:2, :].astype(np.float64, copy=False)

    # ---------- Warp ----------
    out = cv2.warpAffine(
        frame,
        A,
        (W, H),
        flags=params.interpolation,
        borderMode=params.border_mode,
        borderValue=params.border_value,
    )
    return out


# Perspective warp helper
def warp_frame_perspective(
    frame: np.ndarray,
    H_3x3: Mat3x3,
    *,
    params: WarpParams = WarpParams(),
) -> np.ndarray:
    """
    Warp an image using a full 3x3 homography (projective transform).

    NOT using this for Peleg stabilization v1

    Inputs:
    - frame: image
    - H_3x3: 3x3 homography matrix

    Output: warped frame

    - warpPerspective is more expensive than warpAffine.
    - For stabilization, homography can introduce more distortion.
    """
    if frame is None or frame.size == 0:
        return frame
    if H_3x3.shape != (3, 3):
        raise ValueError(f"warp_frame_perspective expected (3,3), got {H_3x3.shape}")

    Ht, Wt = frame.shape[:2]

    out = cv2.warpPerspective(
        frame,
        H_3x3.astype(np.float64, copy=False),
        (Wt, Ht),
        flags=params.interpolation,
        borderMode=params.border_mode,
        borderValue=params.border_value,
    )
    return out