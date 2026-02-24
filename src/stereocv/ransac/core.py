# Andy Zhao
"""
Generic RANSAC loop (model-agnostic).

RANSAC overview:
- Randomly sample a *minimal* subset of correspondences
- Fit a candidate model from that subset
- Score all correspondences by computing residual errors
- Mark inliers where error < tau
- Keep the model with the most inliers (and optionally best error)
- Refit using all inliers (least squares) to get the final model

Uses the ModelFitter Protocol from types.py:
    RANSAC works with both affine and homography
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar
import os

import numpy as np

from .types import Points2D, Mask2D, FloatArray, ModelFitter, RansacResult

M = TypeVar("M")
_RANSAC_DEBUG = os.environ.get("STEREOCV_RANSAC_DEBUG", "0") == "1"


def _required_iter_for_confidence(
        *,
        p_all_inliers: float,
        inlier_ratio: float,
        sample_size: int,
) -> int:
    """
    Compute the number of RANSAC iterations needed so that the probability
    of having drawn at least ONE all-inlier minimal sample is >= p_all_inliers.

    inlier ratio w = (# inliers) / N, Minimal sample s = min_samples,
    - P(all-inliers) = w^s
    - P(not-all-inliers) = 1 - w^s
    - P(not-all-inlier-for-k-times) = (1 - w^s)^k
    - P(at-least-once-all-inliers) = 1 - (1 - w^s)^k >= p
    - k ≥ log(1−p)/log(1−w^s)

    Formula:
       k >= log(1 - p) / log(1 - w^s)

    where:
     - p is desired success probability (e.g., 0.99)
     - w is inlier ratio (inliers / total)
     - s is sample_size (min_samples)

    Edge cases:
     - w == 0  -> impossible, return "infinite-ish" (we'll cap elsewhere)
     - w == 1  -> 1 iteration is enough
    """
    # Clamp inputs to avoid log(0)
    p = float(np.clip(p_all_inliers, 1e-12, 1.0 - 1e-12))
    w = float(np.clip(inlier_ratio, 0.0, 1.0))
    s = int(sample_size)

    if s <= 0:
        raise ValueError("sample_size must be >= 1")

    # If w == 1, every point is an inlier
    if w >= 1.0:
        return 1

    # If w == 0, unable to draw an all-inlier sample
    if w <= 0.0:
        return int(1e9)

    # Probability a minimal sample is all inliers:
    w_to_s = w ** s

    # If w^s is extremely tiny, log(1 - w^s) close to 0
    w_to_s = float(np.clip(w_to_s, 1e-12, 1.0 - 1e-12))

    denominator = np.log(1 - w_to_s)
    numerator = np.log(1 - p)
    k = int(np.ceil(numerator / denominator))
    return max(1, k)


def ransac(
        model_fitter: ModelFitter[M],
        pts0: Points2D,
        pts1: Points2D,
        *,
        min_samples: int,
        tau: float = 3.0,
        max_iters: int = 2000,
        seed: int = 0,
)-> Optional[RansacResult[M]]:
    """
    Run RANSAC to fit a model between pts0 -> pts1.

    Inputs:
    - model_fitter: provides fit_minimal, fit_least_squares, residuals
    - pts0, pts1: (N,2) corresponding points (same N)
    - min_samples: minimal number of correspondences needed (affine=3, homography=4)
    - tau: inlier threshold in pixels (default 3.0)
    - max_iters: upper bound of number of RANSAC iterations
    - seed: RNG seed for reproducibility

    Returns:
    - RansacResult with best model + inlier mask, or None if it fails.
    """
    # ---------- Input validation ----------
    if pts0.shape != pts1.shape:
        raise ValueError(f"pts0 and pts1 must have same shape, got {pts0.shape} vs {pts1.shape}")
    if pts0.ndim != 2 or pts0.shape[1] != 2:
        raise ValueError(f"Expected pts shape (N,2), got {pts0.shape}")

    n = pts0.shape[0]
    if n < min_samples:
        # Not enough matches to fit the model
        return None

    # RNG: reproducible sampling
    rng = np.random.default_rng(seed)

    # Track the best hypothesis
    best_model: Optional[M] = None
    best_inliers: Optional[Mask2D] = None
    best_num_inliers = -1
    best_rms = float("inf")

    # Pre-allocate an array of indices for fast sampling
    all_idx = np.arange(n)

    # ---------- Adaptive Stopping ----------
    # Probability(at-least-one-good-sample) = 0.99
    P = 0.99

    target_iters = max_iters
    iters_run = 0

    # ---------- Main RANSAC Loop ----------
    # Keep looping until min(target_iters, max_iters)
    i = 0
    while i < max_iters and i < target_iters:
        iters_run = i + 1

        # Sample a minimal subset of correspondences (unique indices, no replacement)
        sample_idx = rng.choice(all_idx, size=min_samples, replace=False)

        # Extract the sampled point pairs
        s0 = pts0[sample_idx]
        s1 = pts1[sample_idx]

        # Fit model from minimal set, return None if degenerate
        model = model_fitter.fit_minimal(s0, s1)
        if model is None:
            i += 1
            continue

        # Compute residuals for all correspondences (shape: (N,))
        err = model_fitter.residuals(model, pts0, pts1)

        # Inliers are those with error < tau
        inliers: Mask2D = (err < tau)

        num_inliers = int(np.count_nonzero(inliers))
        if num_inliers < min_samples:
            # Not enough inliers to be meaningful
            i += 1
            continue

        # compute RMS error on inliers
        inlier_err = err[inliers]
        rms = float(np.sqrt(np.mean(inlier_err * inlier_err)))

        # Decide if this model is better than current best
        # Primary criterion: more inliers
        # If tie: lower RMS error
        is_better = (num_inliers > best_num_inliers) or (
                num_inliers == best_num_inliers and rms < best_rms
        )

        if is_better:
            best_model = model
            best_inliers = inliers
            best_num_inliers = num_inliers
            best_rms = rms

            # Inlier ratio from current best
            w = best_num_inliers / float(n)

            # Compute iterations needed to reach confidence P
            iter_needed = _required_iter_for_confidence(
                p_all_inliers=P,
                inlier_ratio=w,
                sample_size=min_samples,
            )
            target_iters = min(target_iters, max(iter_needed, iters_run))
            if _RANSAC_DEBUG:
                print(f"[RANSAC] better model: inliers={best_num_inliers}/{n}, w={w:.3f}, target_iters={target_iters}")

        i += 1

    # If valid model not found, return None
    if best_model is None or best_inliers is None:
        return None

    # Refit on all inliers
    inliers0 = pts0[best_inliers]
    inliers1 = pts1[best_inliers]

    refit = model_fitter.fit_least_squares(inliers0, inliers1)

    # If least squares refit fails, fall back to the best minimal model
    final_model = refit if refit is not None else best_model

    # Recompute RMS on inliers for the final model
    final_err = model_fitter.residuals(final_model, pts0, pts1)
    final_inlier_err = final_err[best_inliers]
    final_rms = float(np.sqrt(np.mean(final_inlier_err * final_inlier_err)))

    return RansacResult(
        model=final_model,
        inliers=best_inliers,
        num_inliers=int(np.count_nonzero(best_inliers)),
        rms_error=final_rms,
        iterations=iters_run,
        threshold=float(tau),
    )








