"""
Video stabilization helper used by the Peleg pipeline.

This stabilizer turns per-frame motion estimates (tx, ty) into a warp correction
matrix T_corr (3x3 translation) that we apply to the current frame.

Trajectory Mode:

1) trajectory_mode = "cumulative"
   - Classic stabilization: accumulate camera path over time and smooth the path.
   - Good for general handheld video stabilization.
   - Can "fight" slow drift (sometimes undesirable for rotate-in-place videos).

    Suppose each frame-to-frame motion gives translation (tx, ty).
    If we accumulate them:

      cum_tx[k] = cum_tx[k-1] + tx[k]
      cum_ty[k] = cum_ty[k-1] + ty[k]

    This cum_* is the camera trajectory in image coordinates.

2) trajectory_mode = "incremental"
   - Smooth only the per-frame motion estimate (tx, ty) and cancel it each frame.
   - Great for Peleg rotate-in-place videos:
       * preserve intended horizontal motion from rotation
       * remove high-frequency vertical jitter
   - Less likely to produce large warps over time.

Smoothing Mode:
- smooth="moving_average"
- smooth="exp" (EMA)

Stabilization Mode:
- mode="vertical": correct ty only
- mode="xy": correct both tx and ty (general stabilization)

# Stabilization:
#   - Smooth the trajectory: smooth_cum_ty[k] = Smooth(cum_ty history)
#   - Compute a correction so the actual frame follows the smoothed path:
#       correction_ty[k] = smooth_cum_ty[k] - cum_ty[k]
#
# Then we warp frame k by translation (0, correction_ty[k]).
#
# That removes the high-frequency "shake" in ty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..ransac.types import Mat3x3
from .smoothing import MovingAverage1D, ExpSmoother1D
from .warp import warp_frame_affine, WarpParams


# ---------- Small matrix helpers ----------
def _make_translation(tx: float, ty: float) -> Mat3x3:
    """
    Construct a 3x3 homogeneous translation matrix:

        [ 1  0  tx ]
        [ 0  1  ty ]
        [ 0  0   1 ]
    """
    T = np.eye(3, dtype=np.float64)
    T[0, 2] = float(tx)
    T[1, 2] = float(ty)
    return T

def _extract_translation(T: Mat3x3) -> tuple[float, float]:
    """
    Extract translation components (tx, ty) from any 3x3 affine-like matrix.

    For translation model, it's exactly the model.
    For affine model, ignore rotation/scale/shear and just use the last column.
    """
    if T.shape != (3, 3):
        raise ValueError(f"Expected T shape (3,3), got {T.shape}")
    return float(T[0, 2]), float(T[1, 2])


# ---------- Stabilizer ----------
SmoothMode = Literal["moving_average", "exp"]
StabilizeMode = Literal["vertical", "xy"]
TrajectoryMode = Literal["cumulative", "incremental"]

@dataclass
class Stabilizer:
    """
    Maintain a smoothed camera trajectory and produce a per-frame correction warp.

    For each frame k (k>=1):
        1) estimate motion from frame k-1 -> k (RANSAC output model)
            T_motion = result.model
        2) update stabilizer, get correction transform for current frame
            T_corr = stab.update(T_motion)
        3) warp the current frame with T_corr
            frame_stable = stab.apply(frame_bgr, T_corr)

    mode:
        "vertical" -> correct only ty, preserves horizontal motion from rotation
        "xy" -> correct tx and ty, can be useful for general stabilization, but can fight Peleg rotation

    trajectory_mode:
        "cumulative"  -> smooth the accumulated path (classic)
        "incremental" -> smooth per-frame motion (Peleg-friendly)

    Smoothing:
        smooth="moving_average": uses a fixed window average
        smooth="exp": exponential smoother with alpha

    ma_window / exp_alpha:
      smoothing parameters.

    warp_params:
      OpenCV warp settings (border mode, interpolation, etc.)
    """

    mode: StabilizeMode = "vertical"
    trajectory_mode: TrajectoryMode = "incremental"
    smooth: SmoothMode = "exp"

    # Moving average tuning (if smooth="moving_average")
    ma_window: int = 15

    # Exponential tuning (if smooth="exp")
    exp_alpha: float = 0.2

    # OpenCV warp params
    warp_params: WarpParams = field(default_factory=WarpParams)

    # ---------- Internal state: cumulative camera path (trajectory) ----------
    _cum_tx: float = 0.0
    _cum_ty: float = 0.0

    # ---------- Internal state: incremental filters (trajectory) ----------
    _filt_tx: float = 0.0
    _filt_ty: float = 0.0

    # ---------- Internal smoothers -----------
    _ma_tx: MovingAverage1D = field(init=False)
    _ma_ty: MovingAverage1D = field(init=False)
    _exp_tx: ExpSmoother1D = field(init=False)
    _exp_ty: ExpSmoother1D = field(init=False)

    def __post_init__(self) -> None:
        self._ma_tx = MovingAverage1D(window=self.ma_window)
        self._ma_ty = MovingAverage1D(window=self.ma_window)

        self._exp_tx = ExpSmoother1D(alpha=self.exp_alpha)
        self._exp_ty = ExpSmoother1D(alpha=self.exp_alpha)

    def reset(self) -> None:
        """
        Reset trajectory + smoothing history.
        Called at the start of a new run/video.
        """
        self._cum_tx = 0.0
        self._cum_ty = 0.0

        self._filt_tx = 0.0
        self._filt_ty = 0.0

        self._ma_tx.reset()
        self._ma_ty.reset()
        self._exp_tx.reset()
        self._exp_ty.reset()

    # Internal helper: apply smoother to a scalar value
    def _smooth_scalar(self, which: str, x: float) -> float:
        """
        Smooth a scalar using the chosen smoothing mode.
        `which` is either "tx" or "ty" so we use the matching smoother state.
        """
        if self.smooth == "moving_average":
            if which == "tx":
                return self._ma_tx.update(x)
            else:
                return self._ma_ty.update(x)
        elif self.smooth == "exp":
            if which == "tx":
                return self._exp_tx.update(x)
            else:
                return self._exp_ty.update(x)
        else:
            raise ValueError(f"Unknown smoothing mode: {self.smooth}")

    # convert motion estimate -> correction transform
    def update(self, T_motion: Mat3x3) -> Mat3x3:
        """
        Update stabilizer using the latest estimated motion and return a correction transform.

        T_motion:
          Estimated motion from previous frame -> current frame.
          This comes from RANSAC (translation fitter or affine fitter).

        1) Extract (tx, ty) from T_motion
        2) Update cumulative trajectory: cum += (tx, ty)
        3) Smooth the cumulative trajectory to get desired path
        4) Correction = smoothed - actual
        5) Return a translation matrix T_corr

        Return: T_corr
          A 3x3 translation matrix that should be applied to the *current* frame.

        Behavior depends on trajectory_mode:
        A) trajectory_mode="cumulative"
           - accumulate path: cum += (tx, ty)
           - smooth the path: smooth(cum)
           - correction = smooth(cum) - cum
           This makes the actual path follow the smoothed path.

        B) trajectory_mode="incremental"
           - smooth the per-frame tx,ty directly
           - correction = -(smoothed tx, smoothed ty)
           This cancels high-frequency motion without trying to reshape long-term drift.

        In mode="vertical", correction_tx = 0.
        """
        tx, ty = _extract_translation(T_motion)

        # (A) Cumulative-path mode
        if self.trajectory_mode == "cumulative":
            # Update cumulative path
            self._cum_tx += tx
            self._cum_ty += ty

            # Smooth cumulative path
            sm_tx = self._smooth_scalar("tx", self._cum_tx)
            sm_ty = self._smooth_scalar("ty", self._cum_ty)

            # Correction to move actual path -> smoothed path
            corr_tx = sm_tx - self._cum_tx
            corr_ty = sm_ty - self._cum_ty

        # (B) Incremental mode (per-frame jitter cancel)
        elif self.trajectory_mode == "incremental":
            # Smooth instantaneous (per-frame) motion
            self._filt_tx = self._smooth_scalar("tx", tx)
            self._filt_ty = self._smooth_scalar("ty", ty)

            # Cancel the filtered motion each frame
            corr_tx = -self._filt_tx
            corr_ty = -self._filt_ty

        else:
            raise ValueError(f"Unknown trajectory_mode: {self.trajectory_mode}")

        # Apply stabilization mode constraints
        if self.mode == "vertical":
            corr_tx = 0.0

        # Build correction transform
        return _make_translation(corr_tx, corr_ty)


    def apply(self, frame: np.ndarray, T_corr: Mat3x3) -> np.ndarray:
        """
        Apply the correction transform to a frame using OpenCV warpAffine.

        Treat correction as affine (translation), so warpAffine is correct and fast.
        """
        return warp_frame_affine(frame, T_corr, params=self.warp_params)




