"""
Visualization utilities for stereo panoramas / stereo pairs.
  - building visualization images
  - optionally showing or saving them
"""

from __future__ import annotations

import math
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
        # print(f"[saved] {output_path}")

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
    dy = 40

    for i, text in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(out, text, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 3, cv2.LINE_AA)   # shadow
        cv2.putText(out, text, (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 215, 255), 2, cv2.LINE_AA)
    return out


def render_planar_view_from_cylinder(
        pano_bgr: np.ndarray,
        *,
        yaw_deg: float,
        fov_deg: float = 60.0,
        panorama_fov_deg: float = 180.0,
        out_w: int = 960,
        out_h: int | None = None,
) -> np.ndarray:
    """
    Render a rectilinear (pinhole) view from a cylindrical panorama.

    My panoramas do NOT cover full 360 degrees.
    So we cannot assume u in [0,W) maps to theta in [0, 2*pi).

    Instead, assume the panorama covers `panorama_fov_deg` degrees total.

    Coordinate conventions:
    - Panorama horizontal coordinate u ∈ [0, W)
      maps to angle theta ∈ [-panorama_fov/2, +panorama_fov/2].
    - yaw_deg is the virtual camera look direction relative to panorama center.
      yaw_deg = 0 means looking at the panorama center.
      yaw_deg = -panorama_fov/2 means looking at left edge.
      yaw_deg = +panorama_fov/2 means looking at right edge.
    - fov_deg is the virtual camera horizontal field of view.
    """
    if pano_bgr is None or pano_bgr.size == 0:
        return pano_bgr

    H, W = pano_bgr.shape[:2]
    if out_h is None:
        out_h = H

    # Convert degrees -> radians
    yaw = math.radians(float(yaw_deg))
    fov = math.radians(float(fov_deg))
    pano_fov = math.radians(float(panorama_fov_deg))

    # Virtual camera focal length (pixels)
    cx = (out_w - 1) * 0.5
    f = cx / math.tan(fov * 0.5)

    # Precompute angle offsets alpha for each output column x
    xs = (np.arange(out_w, dtype=np.float32) - cx)
    alpha = np.arctan(xs / float(f)).astype(np.float32)  # per-column angle offset

    # Total ray angle for each column, relative to panorama center
    theta = (yaw + alpha).astype(np.float32)

    # Clamp theta to the panorama angular support:
    # only have pano pixels for angles within [-pano_fov/2, +pano_fov/2].
    half = np.float32(0.5 * pano_fov)
    theta = np.clip(theta, -half, +half)

    # Map theta -> panorama u in [0, W)
    # theta = -half => u=0
    # theta = +half => u=W-1 (approximately)
    u = ((theta + half) / np.float32(pano_fov)) * np.float32(W - 1)

    # Build remap grids
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    map_x[:, :] = u[None, :]

    # Vertical mapping: direct y, or scale if out_h != H
    if out_h == H:
        map_y[:, :] = np.arange(out_h, dtype=np.float32)[:, None]
    else:
        ys = (np.arange(out_h, dtype=np.float32) * (H / float(out_h))).astype(np.float32)
        map_y[:, :] = ys[:, None]

    view = cv2.remap(
        pano_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return view


def draw_correspondence_arrows(
        frame_bgr: np.ndarray,
        pts0: np.ndarray,
        pts1: np.ndarray,
        inliers: np.ndarray | None,
        *,
        max_draw: int = 80,
) -> np.ndarray:
    """
    Draw motion arrows pts0 -> pts1 on a frame.
    - inliers==True : green arrows
    - inliers==False: red arrows
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    vis = frame_bgr.copy()
    n = min(int(pts0.shape[0]), int(pts1.shape[0]), int(max_draw))

    for i in range(n):
        x0, y0 = float(pts0[i, 0]), float(pts0[i, 1])
        x1, y1 = float(pts1[i, 0]), float(pts1[i, 1])

        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))

        ok = True
        if inliers is not None and i < len(inliers):
            ok = bool(inliers[i])

        color = (0, 255, 0) if ok else (0, 0, 255)  # green=inlier, red=outlier

        cv2.arrowedLine(vis, p0, p1, color, 1, tipLength=0.25)
        cv2.circle(vis, p1, 2, color, -1)

    return vis
