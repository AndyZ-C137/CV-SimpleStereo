# Andy Zhao
"""
Pure cylindrical stereo panorama builder following Peleg & Ben-Ezra (CVPR 1999).

A single rotating camera is used to generate two cylindrical panoramas
by extracting vertical strips from each frame.

For each incoming frame:
    - Extract a strip on the LEFT side  → contributes to RIGHT-eye panorama
    - Extract a strip on the RIGHT side → contributes to LEFT-eye panorama
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from typing import Optional, Tuple, List


@dataclass(frozen=True)
class PelegStereoConfig:
    """
    Configuration for Peleg cylindrical stereo construction.

    Parameters:
    - strip_offset_px : int
        Horizontal distance from image center to extract strip.
        Controls effective stereo baseline.
        Larger offset → larger disparity.

    - strip_width_px : int
        Width (in pixels) of each vertical strip.
        1 px = sharp but thin panorama.
        2–4 px = visually smoother.

    - max_columns : Optional[int]
        Maximum panorama width.
        If set, older strips are discarded to keep memory bounded.

    - display_scale:
        Optional uniform scaling factor for display windows only (not saved output).
        Example: 0.5 for large 4K frames.

    - save_outputs:
        Whether to save left/right panoramas as PNG at the end of the run.
    """
    strip_offset_px: int = 120
    strip_width_px: int = 2
    max_columns: Optional[int] = None
    display_scale: float = 1
    save_outputs: bool = True


class PelegStereo:
    """
    Build cylindrical stereo panoramas using Peleg-style strip stacking.

    Geometry:
    - When the camera rotates about a vertical axis behind the camera,
    - each extracted vertical strip corresponds to rays tangent to a viewing circle.

    By stacking strips over time:
    - Approximate a cylindrical projection.
    - Use opposite-side strips yields left/right stereo pair.

    Process:
    For each frame
      1) compute image center column cx
      2) choose a LEFT strip at (cx - offset) and a RIGHT strip at (cx + offset)
      3) append RIGHT strip to left_panorama buffer (left-eye projection)
      4) append LEFT  strip to right_panorama buffer (right-eye projection)
    """

    def __init__(self, cfg: PelegStereoConfig) -> None:
        self.cfg = cfg

        # Internal buffers storing vertical strip images
        self._left_columns: List[np.ndarray] = []
        self._right_columns: List[np.ndarray] = []

        # Frame geometry
        self._frame_height: Optional[int] = None
        self._frame_width: Optional[int] = None


    def process_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Process a single video frame and append stereo strips.

        Parameters
        - frame_bgr : np.ndarray
            Input frame in BGR format (H x W x 3).

        1. Determine image center.
        2. Compute left and right strip x-coordinates.
        3. Extract vertical strips.
        4. Append strips to panorama buffers.

        LEFT panorama receives strip from RIGHT side.
        RIGHT panorama receives strip from LEFT side.

        This inversion creates the circular projection geometry
        described by Peleg (1999).
        """
        strip_offset_px = self.cfg.strip_offset_px
        strip_width_px = self.cfg.strip_width_px
        max_columns = self.cfg.max_columns

        if frame_bgr is None or frame_bgr.size == 0:
            return

        height, width = frame_bgr.shape[:2]

        # Initialize geometry on first frame
        if self._frame_height is None:
            self._frame_height = height
            self._frame_width = width
        else:
            # Enforce consistent frame size
            if (height, width) != (self._frame_height, self._frame_width):
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (self._frame_width, self._frame_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                height, width = self._frame_height, self._frame_width

        # Compute strip positions relative to image center
        center_x: int = width // 2

        x_left_strip: int = center_x - strip_offset_px
        x_right_strip: int = center_x + strip_offset_px

        # Clamp to valid region
        x_left_strip = max(0, min(width - strip_width_px, x_left_strip))
        x_right_strip = max(0, min(width - strip_width_px, x_right_strip))

        # Extract vertical strips
        right_side_strip = frame_bgr[:, x_right_strip : x_right_strip + strip_width_px]
        left_side_strip = frame_bgr[:, x_left_strip : x_left_strip + strip_width_px]

        # Append to panorama buffers (inverse: right to left, left to right)
        self._left_columns.append(right_side_strip)
        self._right_columns.append(left_side_strip)

        # Enforce bounded panorama width
        if max_columns is not None:
            self._left_columns = self._left_columns[-max_columns :]
            self._right_columns = self._right_columns[-max_columns :]


    def get_panoramas(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Construct and return current stereo panoramas.

        Returns:
        - left_panorama  : np.ndarray or None
        - right_panorama : np.ndarray or None

        Each panorama is constructed by horizontally concatenating
        stored vertical strips.

        If no strips are available, returns (None, None).
        """

        if not self._left_columns or not self._right_columns:
            return None, None

        left_panorama = np.concatenate(self._left_columns, axis=1)
        right_panorama = np.concatenate(self._right_columns, axis=1)

        return left_panorama, right_panorama

    @staticmethod
    def apply_horizontal_shift(
            right_panorama: np.ndarray,
            shift_px: int,
    ) -> np.ndarray:
        """
        Apply a constant horizontal shift to the right panorama.

        Purpose:
         - Align left and right panoramas such that points at infinity
           have zero disparity (vergence alignment).

        Parameters:
        - right_panorama : np.ndarray
            The right-eye cylindrical panorama.

        - shift_px : int
            Number of pixels to shift horizontally.
            Positive → shift right.
            Negative → shift left.

        Returns:
        - shifted_panorama : np.ndarray
            Horizontally shifted version (same size).
        """

        if right_panorama is None or right_panorama.size == 0:
            return right_panorama

        h, w = right_panorama.shape[:2]
        if shift_px == 0:
            return right_panorama.copy()

        # If panorama is too narrow to shift, just return it
        if abs(shift_px) >= w:
            return right_panorama.copy()
        shifted = np.zeros_like(right_panorama)

        if shift_px > 0:
            # shift right: source loses rightmost shift_px columns
            shifted[:, shift_px:] = right_panorama[:, : w - shift_px]
        elif shift_px < 0:
            # shift left: source loses leftmost shift_px columns
            shifted[:, : w + shift_px] = right_panorama[:, -shift_px:]
        else:
            shifted = right_panorama.copy()

        return shifted



