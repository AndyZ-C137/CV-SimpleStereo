"""
Visualization utilities for stereo panoramas / stereo pairs.
  - building visualization images
  - optionally showing or saving them
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np


# Display helper
def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Utility: resize only for display so big panoramas don't overflow screen.

    Parameters
    - img:
        Image to resize.
    - scale:
        Uniform scaling factor.

    Returns:
        Resized image (BGR).
    """
    if img is None or img.size == 0:
        return img
    if scale == 1.0:
        return img

    h, w = img.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Stereo composition display
def make_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side stereo image: [ Left | Right ].

    The two images are cropped to a shared min height and min width.
    """
    if left_bgr is None or right_bgr is None:
        raise ValueError("make_side_by_side received None image(s).")
    if left_bgr.size == 0 or right_bgr.size == 0:
        raise ValueError("make_side_by_side received empty image(s).")

    h = min(left_bgr.shape[0], right_bgr.shape[0])
    w = min(left_bgr.shape[1], right_bgr.shape[1])

    L = left_bgr[:h, :w]
    R = right_bgr[:h, :w]

    return np.concatenate([L, R], axis=1)

def make_anaglyph(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """
    Create a red-cyan anaglyph from a stereo pair.

    left image → red channel
    right image → green + blue channels
    """
    if left_bgr is None or right_bgr is None:
        raise ValueError("make_anaglyph received None image(s).")
    if left_bgr.size == 0 or right_bgr.size == 0:
        raise ValueError("make_anaglyph received empty image(s).")

    # Ensure same size
    h = min(left_bgr.shape[0], right_bgr.shape[0])
    w = min(left_bgr.shape[1], right_bgr.shape[1])

    L = left_bgr[:h, :w]
    R = right_bgr[:h, :w]

    anaglyph = np.zeros_like(L)
    anaglyph[:, :, 2] = L[:, :, 2]  # Red from left
    anaglyph[:, :, 1] = R[:, :, 1]  # Green from right
    anaglyph[:, :, 0] = R[:, :, 0]  # Blue from right

    return anaglyph

def show_anaglyph(left_bgr: np.ndarray, right_bgr: np.ndarray, *, title: str = "Anaglyph") -> None:
    """
    Display an anaglyph preview and wait for a keypress.

    Parameters:
    - left_bgr, right_bgr: Stereo pair (BGR).
    - title: Window title.
    - display_scale: Resize factor for display only.
    """
    anaglyph = make_anaglyph(left_bgr, right_bgr)
    cv2.imshow(title, anaglyph)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def show_and_save_anaglyph(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        output_path: Optional[Path] = None,
        title: str = "Anaglyph",
        display_scale: float = 1.0,
        show: bool = True,
):
    """
    Display an anaglyph preview and wait for a keypress.
    Generate and save a red-cyan anaglyph image.

    Parameters:
    - left_bgr, right_bgr: stereo image pair (BGR).
    - output_path: If provided, save the anaglyph PNG to this path.
    - title: Window title if show=True.
    - display_scale: Resize factor for display only.
    - show: If True, display and wait for a keypress.

    Returns:
    - anaglyph_bgr: The generated anaglyph image.
    """
    anaglyph = make_anaglyph(left_bgr, right_bgr)

    if show:
        cv2.imshow(title, resize_for_display(anaglyph, display_scale))
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), anaglyph)
        print(f"\n[saved] {output_path}")

    return anaglyph


# Epipolar overlays (for rectified/cylindrical stereo)
def overlay_epipolar_lines(
        img_bgr: np.ndarray,
        *,
        num_lines: int = 6,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
) -> np.ndarray:
    """
    Overlay horizontal epipolar lines on cylindrical stereo panorama.

    In Peleg cylindrical stereo:
        Corresponding points lie on the same row.
        Therefore epipolar lines are horizontal.

    Parameters:
    - img_bgr:
        Input cylindrical panorama.
    - num_lines:
        Number of evenly spaced horizontal lines.
    - color:
        Line color (BGR).
    - thickness:
        Line thickness.

    Returns:
        Image with overlay.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Choose evenly spaced row positions
    ys = np.linspace(0, h - 1, num_lines + 2, dtype=int)[1:-1]

    for y in ys:
        cv2.line(vis, (0, y), (w - 1, y), color, thickness)

    return vis

def build_epipolar_overlays(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        num_lines: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Build overlay visualizations for a stereo pair.

    Returns a dict with:
      - left_epi
      - right_epi
      - sbs_epi
      - anaglyph_epi
    """
    left_epi = overlay_epipolar_lines(left_bgr, num_lines=num_lines)
    right_epi = overlay_epipolar_lines(right_bgr, num_lines=num_lines)

    sbs_epi = make_side_by_side(left_epi, right_epi)
    ana_epi = make_anaglyph(left_epi, right_epi)

    return {
        "left_epi": left_epi,
        "right_epi": right_epi,
        "sbs_epi": sbs_epi,
        "anaglyph_epi": ana_epi,
    }


def show_and_save_epipolar_overlay(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        out_dir: Optional[Path] = None,
        prefix: str = "stereo",
        num_lines: int = 6,
        display_scale: float = 1.0,
        show: bool = True,
) -> dict[str, np.ndarray]:
    """
    Create epipolar-line overlays for cylindrical stereo, optionally show and save.

    Saved outputs:
      {prefix}_left_epipolar.png
      {prefix}_right_epipolar.png
      {prefix}_sbs_epipolar.png
      {prefix}_anaglyph_epipolar.png
    """
    overlays = build_epipolar_overlays(left_bgr, right_bgr, num_lines=num_lines)

    if show:
        cv2.imshow(
            "Epipolar Lines (Left | Right)",
            resize_for_display(overlays["sbs_epi"], display_scale),
        )
        cv2.imshow(
            "Epipolar Lines Anaglyph",
            resize_for_display(overlays["anaglyph_epi"], display_scale),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{prefix}_left_epipolar.png"), overlays["left_epi"])
        cv2.imwrite(str(out_dir / f"{prefix}_right_epipolar.png"), overlays["right_epi"])
        cv2.imwrite(str(out_dir / f"{prefix}_sbs_epipolar.png"), overlays["sbs_epi"])
        cv2.imwrite(str(out_dir / f"{prefix}_anaglyph_epipolar.png"), overlays["anaglyph_epi"])
        print(f"\n[saved] {out_dir / f'{prefix}_sbs_epipolar.png'}")

    return overlays


def draw_status_text(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    """
    Draw a small multi-line debug HUD at top-left of an image.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    out = img_bgr.copy()
    x0, y0 = 10, 25
    dy = 22

    for i, text in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(out, text, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)   # shadow
        cv2.putText(out, text, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out