"""
vergence.py

Stereo panorama "vergence" alignment utilities.

Context:
--------
In Peleg-style cylindrical stereo, we construct two panoramas:
  - left panorama
  - right panorama

Even if the panoramas are built correctly, the two images will often be
horizontally misaligned by a constant amount.

This is analogous to "vergence": rotating the eyes so points at infinity
have zero disparity. In practice, we usually want to find a single constant
horizontal shift such that far-away content aligns between left and right.

This module provides:
---------------------
1) estimate_vergence_shift_px(...)
   Estimate a constant horizontal shift between the two panoramas.

2) apply_horizontal_shift(...)  (optional helper)
   Apply that shift to the right panorama.

3) crop_valid_overlap(...)
   After shifting, crop both images to a shared valid overlap region.

4) align_and_crop_pair(...)
   Convenience wrapper: estimate shift -> shift right -> crop overlap.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from stereocv.peleg import PelegStereo


# Core shift estimation
def estimate_vergence_shift_px(
    left_pan_bgr: np.ndarray,
    right_pan_bgr: np.ndarray,
    *,
    max_shift: int = 200,
    far_region: str = "top_half",
) -> int:
    """
    Estimate a constant horizontal shift (in pixels) to align "infinity".

    Far-away content should have ~zero disparity after vergence alignment.
    In a typical indoor/outdoor panorama, the top portion often contains
    more distant structures (walls/ceiling/sky/buildings), so we use that
    region as a proxy for "far".

    1) Convert left/right panorama regions to grayscale.
    2) Compute horizontal gradient magnitude (Scharr dx) to emphasize vertical edges.
       - Vertical edges are stable and informative for alignment.
    3) Reduce each image region to a 1D "edge energy per column" signal.
    4) Normalize signals.
    5) Search shift in [-max_shift, +max_shift] maximizing correlation (dot product).

    Returned shift is defined as:
        "shift > 0 means shift the RIGHT panorama to the RIGHT by shift px"

    Parameters:
    - left_pan_bgr, right_pan_bgr:
        Cylindrical panoramas (BGR uint8).
    - max_shift:
        Search range in pixels.
    - far_region:
        Which vertical region to use:
          - "top_half": y in [0, H/2)
          - "top_third": y in [0, H/3)
          - "full": use entire height

    Returns:
    - best_shift: int
        The estimated constant horizontal shift.
    """
    if left_pan_bgr is None or right_pan_bgr is None:
        raise ValueError("estimate_vergence_shift_px received None image(s).")
    if left_pan_bgr.size == 0 or right_pan_bgr.size == 0:
        raise ValueError("estimate_vergence_shift_px received empty image(s).")

    H = min(left_pan_bgr.shape[0], right_pan_bgr.shape[0])

    if far_region == "top_half":
        y0, y1 = 0, H // 2
    elif far_region == "top_third":
        y0, y1 = 0, H // 3
    elif far_region == "full":
        y0, y1 = 0, H
    else:
        raise ValueError(f"Unknown far_region: {far_region}")

    L = cv2.cvtColor(left_pan_bgr[y0:y1], cv2.COLOR_BGR2GRAY)
    R = cv2.cvtColor(right_pan_bgr[y0:y1], cv2.COLOR_BGR2GRAY)

    # Compute horizontal gradients (dx). Strong response for vertical edges.
    gxL = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    gxR = cv2.Scharr(R, cv2.CV_32F, 1, 0)

    # Build 1D signals by summing absolute edge strength per column.
    sL = np.sum(np.abs(gxL), axis=0)
    sR = np.sum(np.abs(gxR), axis=0)

    # Normalize (helps correlation be less sensitive to exposure/contrast).
    sL = (sL - sL.mean()) / (sL.std() + 1e-6)
    sR = (sR - sR.mean()) / (sR.std() + 1e-6)

    best_shift = 0
    best_score = -1e18

    # Search all shifts in range.
    # shift > 0 means right shifted right.
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = sL[shift:]
            b = sR[: len(a)]
        else:
            a = sL[: shift]     # shift is negative → drop right end
            b = sR[-shift : -shift + len(a)]

        if len(a) < 50:     # too small, ignore
            continue

        score = float(np.dot(a, b))
        if score > best_score:
            best_score = score
            best_shift = int(shift)

    return best_shift


# Overlap cropping utilities
def apply_horizontal_shift(
    pano_bgr: np.ndarray,
    *,
    shift_px: int,
) -> np.ndarray:
    """
    Apply a constant horizontal shift to a panorama.

    shift_px:
      Positive -> shift right
      Negative -> shift left

    Returns:
      shifted panorama (same size)
    """
    if pano_bgr is None or pano_bgr.size == 0:
        return pano_bgr

    h, w = pano_bgr.shape[:2]
    if shift_px == 0:
        return pano_bgr.copy()

    if abs(shift_px) >= w:
        return pano_bgr.copy()

    shifted = np.zeros_like(pano_bgr)

    if shift_px > 0:
        shifted[:, shift_px:] = pano_bgr[:, : w - shift_px]
    else:
        s = -shift_px
        shifted[:, : w - s] = pano_bgr[:, s:]

    return shifted


def crop_valid_overlap(
    left_pan_bgr: np.ndarray,
    right_pan_shifted_bgr: np.ndarray,
    *,
    shift_px: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Crop left and (shifted) right panoramas to their valid overlap region.

    Parameters:
    - left_pan_bgr:
        Left panorama (unshifted).
    - right_pan_shifted_bgr:
        Right panorama AFTER shifting (so it contains black padding).
    - shift_px:
        The same shift used to create right_pan_shifted_bgr.
        shift_px > 0: right shifted right -> leftmost columns invalid
        shift_px < 0: right shifted left  -> rightmost columns invalid

    Returns:
    (left_crop, right_crop)
        Cropped images with identical shape.
        If images are empty, returns (None, None).
    """
    if left_pan_bgr is None or right_pan_shifted_bgr is None:
        return None, None
    if left_pan_bgr.size == 0 or right_pan_shifted_bgr.size == 0:
        return None, None

    h = min(left_pan_bgr.shape[0], right_pan_shifted_bgr.shape[0])
    w = min(left_pan_bgr.shape[1], right_pan_shifted_bgr.shape[1])

    L = left_pan_bgr[:h, :w]
    R = right_pan_shifted_bgr[:h, :w]

    if shift_px > 0:
        # right shifted right -> leftmost shift_px columns invalid
        return L[:, shift_px:], R[:, shift_px:]
    elif shift_px < 0:
        s = -shift_px
        return L[:, : w - s], R[:, : w - s]
    else:
        return L, R


def align_and_crop_pair(
    left_raw_bgr: np.ndarray,
    right_raw_bgr: np.ndarray,
    *,
    max_shift: int = 200,
    far_region: str = "top_half",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Wrapper:
      1) Estimate vergence shift
      2) Shift RIGHT panorama
      3) Crop to valid overlap

    Returns:
      left_crop, right_crop, shift_px
    """
    shift_px = estimate_vergence_shift_px(
        left_raw_bgr,
        right_raw_bgr,
        max_shift=max_shift,
        far_region=far_region,
    )

    right_shifted = apply_horizontal_shift(right_raw_bgr, shift_px=shift_px)
    left_crop, right_crop = crop_valid_overlap(left_raw_bgr, right_shifted, shift_px=shift_px)

    if left_crop is None or right_crop is None:
        raise RuntimeError("align_and_crop_pair failed: empty overlap after shifting/cropping.")

    return left_crop, right_crop, int(shift_px)

