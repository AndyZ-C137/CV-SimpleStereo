# Andy Zhao
"""
Tracking pipeline: corner detection + LK tracking + re-detection

This class is STATEFUL.

It manages:
  - previous grayscale frame
  - current active points
  - when to re-detect corner_params
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..ransac.types import Points2D
from .corners import shitomasi_detect, ShiTomasiParams
from .lk_tracking import lk_track, LKParams
from .clean_points import clean_points


@dataclass
class LKTrackPipeline:
    """
    state machine that turns consecutive frames into correspondences.

    - initialize(gray0)
    - step(gray1)

    Return cleaned correspondences (pts0, pts1) ready for RANSAC.

    Adaptive corner refresh:
      - Demo (or caller) can call notify_quality(inlier_ratio)
      - If inlier_ratio is too low, pipeline re-detects corners next step
    """
    corner_params: ShiTomasiParams = field(default_factory=ShiTomasiParams)
    lk_params: LKParams = field(default_factory=LKParams)

    # If tracked points drop below threshold, re-initialize/re-detect
    min_active_points: int = 120

    # Ignore tracks that jump too far in one frame.
    max_motion_px: float | None = 80.0

    # ---------- Adaptive refresh corners -----------
    adaptive_refresh: bool = True
    # If RANSAC inlier ratio < min_inlier_ratio, request a re-detect next step.
    min_inlier_ratio: float = 0.70

    # After a refresh, wait a few frames before allowing another refresh.
    refresh_cooldown_frames: int = 8

    # ---------- Internal state tracking -----------
    _prev_gray: Optional[np.ndarray] = None
    _prev_pts: Optional[Points2D] = None

    # Adaptive refresh internal flags
    _force_redetect: bool = False
    _cooldown_left: int = 0

    def initialize(self, gray0: np.ndarray) -> None:
        """
        Initialize pipeline with first frame.
        Detect initial corner_params.
        """
        if gray0.ndim != 2:
            raise ValueError("initialize expects grayscale frame.")

        self._prev_gray = gray0
        self._prev_pts = shitomasi_detect(gray0, params=self.corner_params)

        # Reset adaptive refresh state
        self._force_redetect = False
        self._cooldown_left = 0

        if self._prev_pts.shape[0] == 0:
            raise RuntimeError("No corners detected during initialization.")


    def update_quality(self, inlier_ratio: float | None) -> None:
        """
        Update the pipeline about tracking / geometry quality from outside.

        Demo after running RANSAC:
            ratio = result.num_inliers / len(pts0_clean)
            pipeline.update_quality(ratio)

        Process:
          - If adaptive_refresh is enabled
          - And not in cooldown
          - And ratio is below threshold
          -> request a corner re-detect on the NEXT step() call.
        """
        if not self.adaptive_refresh:
            return
        if inlier_ratio is None:
            self._force_redetect = True
            return
        if self._cooldown_left > 0:
            return

        ratio = float(inlier_ratio)
        if ratio <= self.min_inlier_ratio:
            self._force_redetect = True


    def step(self, gray1: np.ndarray) -> tuple[Points2D, Points2D, dict]:
        """
        Process the next frame and produce correspondences.

        Returns:
          pts0_clean: (M,2) points in previous frame
          pts1_clean: (M,2) corresponding points in current frame
          info: small debug dict with counts / raw status
        """
        if gray1.ndim != 2:
            raise ValueError("step expects grayscale (H,W).")

        if self._prev_gray is None or self._prev_pts is None:
            raise RuntimeError("Call initialize() first.")

        # Decrement cooldown each frame
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # Decide whether to re-detect corners on the PREVIOUS frame
        refreshed = False

        # Check if too few points left to track
        too_few_pts = (self._prev_pts.shape[0] < self.min_active_points)

        # Check if inlier ratios too low
        need_redetect_quality = self._force_redetect and (self._cooldown_left == 0)

        # If too few points left, re-initialize on previous frame
        if too_few_pts or need_redetect_quality:
            self._prev_pts = shitomasi_detect(self._prev_gray, params=self.corner_params)
            refreshed = True

            # Clear the force flag and start cooldown
            self._force_redetect = False
            self._cooldown_left = self.refresh_cooldown_frames

        pts0 = self._prev_pts

        # If still empty, unable to track
        # Advance state and return empty.
        if pts0.shape[0] == 0:
            # no points available
            self._prev_gray = gray1
            self._prev_pts = pts0
            empty = np.zeros((0, 2), dtype=np.float64)
            info = {
                "num_raw": 0,
                "num_tracked": 0,
                "num_clean": 0,
                "refreshed": refreshed,
                "cooldown_left": self._cooldown_left,
            }
            return empty, empty, info

        # Track using LK, gives raw correspondences and status
        pts0_raw, pts1_raw, status, err = lk_track(
            self._prev_gray,
            gray1,
            pts0,
            params=self.lk_params,
        )

        # Clean correspondences for downstream geometry (RANSAC)
        pts0_clean, pts1_clean, clean_mask = clean_points(
            pts0_raw,
            pts1_raw,
            status=status,
            max_motion_px=self.max_motion_px,
        )

        # Update state for next step:
        # *current frame coords* -> the next "prev_pts".
        # Use only successfully tracked points (LK status == 1).
        self._prev_gray = gray1
        self._prev_pts = pts1_raw[status == 1]

        info = {
            "num_raw": int(pts0_raw.shape[0]),
            "num_tracked": int(np.count_nonzero(status == 1)),
            "num_clean": int(pts0_clean.shape[0]),
            "refreshed": refreshed,
            "cooldown_left": self._cooldown_left,
            "clean_mask": clean_mask,
            "lk_status": status,
        }

        return pts0_clean, pts1_clean, info







