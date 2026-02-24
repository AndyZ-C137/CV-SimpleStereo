# Andy Zhao
"""
Provide a single integrated pipeline for:
    (1) tracking correspondences (LK)
    (2) motion estimation (RANSAC)
    (3) stabilization (warp current frame)
    (4) Peleg stereo panorama construction (strip stacking)

Given a new frame_k:
  a) gray_{k-1}, gray_k -> LKTrackPipeline.step() -> pts0, pts1
  b) RANSAC(pts0, pts1) -> motion model T_motion
  c) stabilizer.update(T_motion) -> correction transform T_corr
  d) warp frame_k by T_corr -> frame_stable_k
  e) peleg.process_frame(frame_stable_k)

Default stabilization mode is "vertical":
  - remove vertical jitter (ty)
  - preserve horizontal rotation flow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2

from ..matching.pipeline import LKTrackPipeline
from ..ransac.core import ransac
from ..ransac.types import Mat3x3
from ..ransac.translation_fitter import TranslationFitter
from ..peleg import PelegStereo, PelegStereoConfig
from .stabilizer import Stabilizer, StabilizeMode, SmoothMode, TrajectoryMode


# Configuration
@dataclass(frozen=True)
class StablePelegConfig:
    """
    Config for Peleg stereo pipeline that's stabilized

    RANSAC:
    - tau: Inlier threshold in pixels. (default: 3px)
    - max_iters:
      Upper bound on RANSAC iterations.
    - seed:
      Keep deterministic for debugging / reproducibility.

    Stabilization:
    - stabilize_mode:
      "vertical" = recommended for Peleg (default)
      "xy" = general stabilization
    - smoothing:
      "moving_average" or "exp"
    """
    tau: float = 3.0
    max_iters: int = 1500
    seed: int = 0

    stabilize_mode: StabilizeMode = "vertical"  # "vertical" or "xy"
    smoothing: SmoothMode = "exp"  # "moving_average" or "exp"
    trajectory_mode: TrajectoryMode = "incremental"  # "incremental" or "cumulative"
    ma_window: int = 15
    exp_alpha: float = 0.2

    # If RANSAC fails, either
    # - do nothing (no stabilization this frame)
    # - or reuse last correction
    reuse_last_if_fail: bool = False
    return_debug_points: bool = False
    debug_sample_n: int = 80


class StablePelegPipeline:
    """
    StablePelegSPipeline

    - initialize(first_frame_bgr)
    - step(next_frame_bgr) -> (stable_frame, (left_pano, right_pano), info)

    'info' contains:
        - tracking stats from LKTrackPipeline
        - ransac stats if available (num_inliers, rms_error, iterations)
        - stabilization correction (tx, ty) applied
        - flags for "ransac_failed" or "tracking_empty"
    """
    def __init__(
            self,
            *,
            peleg_cfg: PelegStereoConfig,
            pipe_cfg: StablePelegConfig = StablePelegConfig(),
            lk_pipeline: Optional[LKTrackPipeline] = None,
    ) -> None:
        self.pipe_cfg = pipe_cfg

        # Core components (compose existing modules)
        self.lk = lk_pipeline if lk_pipeline is not None else LKTrackPipeline()
        self.fitter = TranslationFitter()  # recommended for Peleg stabilization

        self.stabilizer = Stabilizer(
            mode=self.pipe_cfg.stabilize_mode,
            trajectory_mode=self.pipe_cfg.trajectory_mode,
            smooth=self.pipe_cfg.smoothing,
            ma_window=self.pipe_cfg.ma_window,
            exp_alpha=self.pipe_cfg.exp_alpha,
        )

        self.peleg = PelegStereo(peleg_cfg)

        # Internal state
        self._initialized: bool = False
        self._last_Tcorr: Mat3x3 = np.eye(3, dtype=np.float64)

    def initialize(self, frame_bgr: np.ndarray) -> None:
        """
        Initialize pipeline with the first frame.

        1) convert to grayscale
        2) initialize LK tracking pipeline
        3) reset stabilizer + peleg buffers
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("initialize received empty frame.")

        gray0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # sets internal prev frame + corners
        self.lk.initialize(gray0)
        self.stabilizer.reset()

        # PelegStereo stores strips internally; easiest reset is to re-create
        self.peleg = PelegStereo(self.peleg.cfg)
        self._last_Tcorr = np.eye(3, dtype=np.float64)
        self._initialized = True


    def step(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, Any]]:
        """
        Process the next frame.

        Returns:
        - stable_frame:
          The stabilized version of the input frame for THIS time step.
        - (left_pano, right_pano):
          Current panoramas from PelegStereo
        - info:
          Debug dictionary with tracking/ransac/stabilization stats.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize(first_frame) before step().")

        if frame_bgr is None or frame_bgr.size == 0:
            # Return as-is with debug info
            return frame_bgr, (None, None), {"error": "empty_frame"}

        gray1 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Step 1. Tracking: get correspondences pts0 -> pts1
        pts0, pts1, track_info = self.lk.step(gray1)
        info: Dict[str, Any] = {"tracking": track_info}

        # ------------------------------------------------------------
        # Optional debug: sample a small subset of point correspondences
        # so the demo can visualize motion + inliers without huge memory.
        # ------------------------------------------------------------
        dbg_idx = None
        if self.pipe_cfg.return_debug_points:
            n = int(pts0.shape[0])
            k = int(min(self.pipe_cfg.debug_sample_n, n))
            # sample without replacement for a representative set
            rng = np.random.default_rng(self.pipe_cfg.seed)
            dbg_idx = rng.choice(n, size=k, replace=False) if k > 0 else None

            if dbg_idx is not None:
                info["debug_points"] = {
                    "pts0": pts0[dbg_idx].copy(),   # (k,2)
                    "pts1": pts1[dbg_idx].copy(),   # (k,2)
                    # inliers filled later after RANSAC runs
                }

        if pts0.shape[0] == 0:
            # No correspondences => cannot estimate motion
            info["tracking_empty"] = True

            # Peleg still can proceed, feed raw frame.
            stable_frame = frame_bgr
            self.peleg.process_frame(stable_frame)
            return stable_frame, self.peleg.get_panoramas(), info

        # Step 2. RANSAC: motion estimate
        result = ransac(
            model_fitter=self.fitter,
            pts0=pts0,
            pts1=pts1,
            min_samples=1,
            tau=self.pipe_cfg.tau,
            max_iters=self.pipe_cfg.max_iters,
            seed=self.pipe_cfg.seed,
        )

        if result is None:
            # RANSAC failed: let LK pipeline know quality is bad
            self.lk.update_quality(None)
            info["ransac_failed"] = True

            if self.pipe_cfg.return_debug_points and dbg_idx is not None and "debug_points" in info:
                info["debug_points"]["inliers"] = np.zeros((len(dbg_idx),), dtype=bool)

            if self.pipe_cfg.reuse_last_if_fail:
                T_corr = self._last_Tcorr
            else:
                T_corr = np.eye(3, dtype=np.float64)

        else:
            # Feed back inlier ratio to tracking pipeline quality logic
            inlier_ratio = result.num_inliers / max(1, pts0.shape[0])
            self.lk.update_quality(inlier_ratio)

            info["ransac"] = {
                "num_inliers": result.num_inliers,
                "rms_error": result.rms_error,
                "iterations": result.iterations,
                "threshold": result.threshold,
                "inlier_ratio": float(inlier_ratio),
            }

            # If debug sampling is enabled, attach inlier labels for sampled points
            if self.pipe_cfg.return_debug_points and dbg_idx is not None and "debug_points" in info:
                # result.inliers is a boolean mask for all correspondences
                info["debug_points"]["inliers"] = result.inliers[dbg_idx].copy()

            # Step 3. Stabilization: compute correction transform for current frame
            T_corr = self.stabilizer.update(result.model)

        # Keep last correction if needed
        self._last_Tcorr = T_corr

        # extract correction translation
        corr_tx = float(T_corr[0, 2])
        corr_ty = float(T_corr[1, 2])
        info["stabilization"] = {"corr_tx": corr_tx, "corr_ty": corr_ty}

        # Step 4. Apply warp to current frame
        stable_frame = self.stabilizer.apply(frame_bgr, T_corr)

        # Step 5. Feed stabilized frame into Peleg strip stacking
        self.peleg.process_frame(stable_frame)

        return stable_frame, self.peleg.get_panoramas(), info



















