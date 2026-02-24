# Andy ZHao

"""
Quantitative and visual validation utilities for stereo panoramas.

In Peleg cylindrical stereo, epipolar lines are horizontal (rectified by construction),
so corresponding points should satisfy:
    y_left ≈ y_right

With real handheld motion (torso rotation), this will not be perfect, but it should
usually be close.

This file provides two complementary validations:

(1) Row-band correlation validation
    - For selected rows y, search a small vertical band dy ∈ [-band, +band]
    - For each dy, search horizontal shift dx ∈ [-max_shift, +max_shift]
    - Report best dy, dx
    - If epipolar constraint holds, best dy should be near 0

(2) Sparse feature match validation (ORB)
    - Detect ORB features in left/right panoramas
    - Match descriptors
    - Compute dy distribution for matches
    - Report median/mean/max |dy|
    - Optionally draw top matches and annotate dy on the visualization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


# (1) Row-band correlation validation
def estimate_row_disparity_shift(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        y: int,
        band: int = 3,
        max_shift: int = 120,
        patch_half_height: int = 2,
        min_overlap: int = 100,
) -> tuple[int, int]:
    """
    Verify epipolar behavior around a chosen row.
        i.e., Estimate (best_dy, best_dx)

    In a perfectly rectified stereo pair, the best match for row y in left
    should occur at the same row y in right (dy=0), with some horizontal shift dx.

    We search for the best match of left row y against right rows (y+dy),
    we allow small vertical offsets dy in [-band, +band]

    1) Convert left/right to grayscale.
    2) Extract a small vertical patch around y (a few rows) to make the signal stable.
    3) Build a 1D signal over columns using horizontal gradient energy:
         s(x) = sum_y |dI/dx|
       This makes it less sensitive to brightness changes.
    4) For each candidate dy and dx, score correlation using dot product.

    Returns:
      best_dy: vertical offset of best match (should be ~0 if epipolar lines are horizontal)
      best_shift: horizontal shift at that dy

    Interpretation:
        best_dy ≈ 0  -> good epipolar behavior (horizontal epipolar lines)
        best_dy large -> vertical mismatch (non-ideal motion or alignment)
    """
    if left_bgr is None or right_bgr is None:
        raise ValueError("estimate_row_disparity_shift received None image(s).")
    if left_bgr.size == 0 or right_bgr.size == 0:
        raise ValueError("estimate_row_disparity_shift received empty image(s).")

    Lg = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    Rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    h = min(Lg.shape[0], Rg.shape[0])
    w = min(Lg.shape[1], Rg.shape[1])
    Lg = Lg[:h, :w]
    Rg = Rg[:h, :w]

    y = int(max(0, min(h - 1, y)))

    # Use a small vertical patch (a few rows) to make signal more stable
    y0 = max(0, y - patch_half_height)
    y1 = min(h, y + patch_half_height + 1)
    L_patch = Lg[y0:y1, :].astype(np.float32)

    best_score = -1e18
    best_dy = 0
    best_shift = 0

    for dy in range(-band, band + 1):
        yr = y + dy
        if yr < 0 or yr >= h:
            continue

        r0 = max(0, yr - patch_half_height)
        r1 = min(h, yr + patch_half_height + 1)
        R_patch = Rg[r0:r1, :].astype(np.float32)

        # Make same height (in case at borders)
        hh = min(L_patch.shape[0], R_patch.shape[0])
        A = L_patch[:hh]
        B = R_patch[:hh]

        # Build 1D signals per column using horizontal edge energy
        gxA = cv2.Scharr(A, cv2.CV_32F, 1, 0)
        gxB = cv2.Scharr(B, cv2.CV_32F, 1, 0)
        sA = np.sum(np.abs(gxA), axis=0)
        sB = np.sum(np.abs(gxB), axis=0)

        # Normalize
        sA = (sA - sA.mean()) / (sA.std() + 1e-6)
        sB = (sB - sB.mean()) / (sB.std() + 1e-6)

        # Search horizontal shift
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                a = sA[shift:]
                b = sB[: len(a)]
            else:
                a = sA[: shift]
                b = sB[-shift : -shift + len(a)]

            if len(a) < min_overlap:
                continue

            score = float(np.dot(a, b))
            if score > best_score:
                best_score = score
                best_dy = dy
                best_shift = shift

    return best_dy, best_shift


def  validate_epipolar_rows(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        rows: Optional[list[int]] = None,
        band: int = 4,
        max_shift: int = 150,
) -> list[dict]:
    """
    Run row-band validation for a set of rows and return results.

    Returns a list of dicts like:
      {"y": y, "best_dy": dy, "best_dx": dx}
    """
    if rows is None:
        h = min(left_bgr.shape[0], right_bgr.shape[0])
        rows = [h // 4, h // 2, (3 * h) // 4]

    out = []
    for y in rows:
        dy, dx = estimate_row_disparity_shift(
            left_bgr,
            right_bgr,
            y=y,
            band=band,
            max_shift=max_shift,
        )
        out.append({"y": int(y), "best_dy": int(dy), "best_dx": int(dx)})
    return out


# (2) ORB match-based epipolar validation
def validate_epipolar_with_orb_matches(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        max_matches: int = 200,
        nfeatures: int = 1500,
) -> Dict[str, float]:
    """
    Quantitatively validate epipolar constraint using sparse ORB feature matches.

    Output stats focus on dy = y_right - y_left.

    For good rectified/cylindrical stereo:
      dy should be near 0 for correct matches.

    Returns:
      {
        "num_matches": ...,
        "median_abs_dy": ...,
        "mean_abs_dy": ...,
        "max_abs_dy": ...,
      }
    """
    if left_bgr is None or right_bgr is None:
        raise ValueError("validate_epipolar_with_orb_matches received None image(s).")
    if left_bgr.size == 0 or right_bgr.size == 0:
        raise ValueError("validate_epipolar_with_orb_matches received empty image(s).")

    Lg = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    Rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)

    kL, dL = orb.detectAndCompute(Lg, None)
    kR, dR = orb.detectAndCompute(Rg, None)

    if dL is None or dR is None or len(kL) < 10 or len(kR) < 10:
        return {
            "num_matches": 0.0,
            "median_abs_dy": float("nan"),
            "mean_abs_dy": float("nan"),
            "max_abs_dy": float("nan"),
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dL, dR)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]

    dys = []
    for m in matches:
        (_, yL) = kL[m.queryIdx].pt
        (_, yR) = kR[m.trainIdx].pt
        dys.append(float(yR - yL))

    dys = np.array(dys, dtype=np.float64)
    abs_dy = np.abs(dys)

    return {
        "num_matches": float(len(matches)),
        "median_abs_dy": float(np.median(abs_dy)),
        "mean_abs_dy": float(np.mean(abs_dy)),
        "max_abs_dy": float(np.max(abs_dy)),
    }


def draw_top_matches_with_row_info(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        max_draw: int = 12,
        dy_thresh: float = 4.0,
        nfeatures: int = 1500,
) -> np.ndarray:
    """
    Draw top ORB matches between left/right panoramas and annotate dy values.

    Returns:
      match_vis: image showing matches and dy labels.
    """
    if left_bgr is None or right_bgr is None:
        raise ValueError("draw_top_matches_with_row_info received None image(s).")
    if left_bgr.size == 0 or right_bgr.size == 0:
        raise ValueError("draw_top_matches_with_row_info received empty image(s).")

    Lg = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    Rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kL, dL = orb.detectAndCompute(Lg, None)
    kR, dR = orb.detectAndCompute(Rg, None)

    if dL is None or dR is None:
        # Fall back to side-by-side image if we can't compute matches
        h = min(left_bgr.shape[0], right_bgr.shape[0])
        w = min(left_bgr.shape[1], right_bgr.shape[1])
        return np.concatenate([left_bgr[:h, :w], right_bgr[:h, :w]], axis=1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = sorted(bf.match(dL, dR), key=lambda m: m.distance)

    good = []
    for m in matches:
        _, yL = kL[m.queryIdx].pt
        _, yR = kR[m.trainIdx].pt
        if abs(yR - yL) <= dy_thresh:
            good.append(m)
        if len(good) >= max_draw:
            break

    matches = good

    vis = cv2.drawMatches(
        left_bgr, kL,
        right_bgr, kR,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # drawMatches places right image to the right; offset is left image width
    wL = left_bgr.shape[1]

    for m in matches:
        xL, yL = kL[m.queryIdx].pt
        xR, yR = kR[m.trainIdx].pt
        dy = yR - yL

        # Put label near the left keypoint
        px = int(xL + 5)
        py = int(yL - 5)

        cv2.putText(
            vis,
            f"dy={dy:+.1f}",
            (px, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


# Convenience "report" helper
def run_epipolar_validation_report(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        band_rows: Optional[list[int]] = None,
        band: int = 4,
        max_shift: int = 150,
        max_matches: int = 200,
) -> dict:
    """
    Run both row-band and ORB validations and return a combined report dict.

    Returns:
      {
        "row_band": [ ... ],
        "orb": { ... }
      }
    """
    row_band = validate_epipolar_rows(
        left_bgr,
        right_bgr,
        rows=band_rows,
        band=band,
        max_shift=max_shift,
    )

    orb_stats = validate_epipolar_with_orb_matches(
        left_bgr,
        right_bgr,
        max_matches=max_matches,
    )

    return {"row_band": row_band, "orb": orb_stats}




