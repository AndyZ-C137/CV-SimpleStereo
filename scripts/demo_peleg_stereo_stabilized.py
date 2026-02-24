# Andy Zhao

"""
Peleg-style cylindrical stereo panorama from a prerecorded video.

This script implements the "pure cylindrical" version:
- no feature matching
- no homography / warping
- no stabilization
- no depth computation

It builds two panoramas by stacking vertical strips from each frame:
    left_panorama  <- strips from RIGHT side of each frame
    right_panorama <- strips from LEFT  side of each frame
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import math

import cv2
import numpy as np

from stereocv.peleg import PelegStereo, PelegStereoConfig

from stereocv.stabilize.peleg_pipeline import (
    StablePelegPipeline,
    StablePelegConfig,
)

@dataclass(frozen=True)
class PelegRunResult:
    left_raw: np.ndarray
    right_raw: np.ndarray
    left: np.ndarray          # cropped/aligned
    right: np.ndarray         # cropped/aligned
    shift_px: int

#
# def _resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
#     """
#     Utility: resize only for display so big panoramas don't overflow screen.
#
#     Parameters
#     - img:
#         Image to resize.
#     - scale:
#         Uniform scaling factor.
#
#     Returns:
#     - Resized image (BGR).
#     """
#     if scale == 1.0:
#         return img
#     h, w = img.shape[:2]
#     new_h = max(1, int(h * scale))
#     new_w = max(1, int(w * scale))
#     return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#

def render_planar_view_from_cylinder(
    pano_bgr: np.ndarray,
    *,
    yaw_deg: float,
    fov_deg: float = 60.0,
    out_w: int = 960,
    out_h: int | None = None,
) -> np.ndarray:
    """
    Render a standard planar (rectilinear) view from a cylindrical panorama.

    Why this exists:
    ---------------
    Your PelegStereo builder creates a *cylindrical* panorama (x-axis is angle).
    But most people want to VIEW stereo as normal left/right perspective images.

    This function creates a "virtual camera":
      - located at the cylinder center
      - looking at yaw angle yaw_deg
      - with horizontal field of view fov_deg
      - output image size out_w x out_h

    Geometry model:
    ---------------
    Cylinder panorama parameterization:
      u in [0, W) corresponds to angle theta in [-pi, +pi) (or 0..2pi)

      theta = 2*pi * (u / W)

    Planar pinhole camera parameterization:
      For each output pixel x, compute the ray direction angle offset:

        alpha = atan( (x - cx) / f )

      where f is focal length in pixels:
        f = (out_w/2) / tan(fov/2)

      Then theta = yaw + alpha

    Sampling:
    ---------
    For each output pixel (x, y):
      - theta gives panorama u
      - y maps directly (same vertical coordinate)

    Notes:
    ------
    - This is a purely geometric viewing transform; it does NOT change the Peleg stereo construction.
    - It makes the stereo easier to judge and share.
    """
    if pano_bgr is None or pano_bgr.size == 0:
        return pano_bgr

    H, W = pano_bgr.shape[:2]
    if out_h is None:
        # Preserve vertical size by default
        out_h = H

    # Convert degrees to radians
    yaw = math.radians(float(yaw_deg))
    fov = math.radians(float(fov_deg))

    # Focal length in pixels for the virtual camera
    cx = (out_w - 1) * 0.5
    f = cx / math.tan(fov * 0.5)

    # Prepare remap grids
    # map_x, map_y are float32 images where each pixel stores the source coordinate to sample
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    # For each output x, compute angle theta
    # We vectorize over x for speed.
    xs = (np.arange(out_w, dtype=np.float32) - cx)  # shape (out_w,)
    alpha = np.arctan(xs / float(f)).astype(np.float32)  # angle offset per column
    theta = (yaw + alpha).astype(np.float32)             # total yaw per column

    # Wrap theta into [0, 2*pi)
    two_pi = np.float32(2.0 * math.pi)
    theta = np.mod(theta, two_pi)

    # Convert theta to panorama u coordinate
    # u = (theta / 2pi) * W
    u = (theta / two_pi) * np.float32(W)  # shape (out_w,)

    # Fill map_x for all rows with the same u values
    map_x[:, :] = u[None, :]

    # Vertical mapping: direct (clamped if sizes differ)
    # If out_h != H, we scale y accordingly.
    if out_h == H:
        map_y[:, :] = np.arange(out_h, dtype=np.float32)[:, None]
    else:
        ys = (np.arange(out_h, dtype=np.float32) * (H / float(out_h))).astype(np.float32)
        map_y[:, :] = ys[:, None]

    # Sample using OpenCV remap
    # BORDER_WRAP is nice for x wrapping, but remap borderMode wrap is limited;
    # since we already wrap theta->u, BORDER_REFLECT is fine.
    view = cv2.remap(
        pano_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return view


# def make_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
#     """
#     Make a side-by-side stereo image: [ left | right ].
#     """
#     h = min(left_bgr.shape[0], right_bgr.shape[0])
#     w = min(left_bgr.shape[1], right_bgr.shape[1])
#     L = left_bgr[:h, :w]
#     R = right_bgr[:h, :w]
#     return np.concatenate([L, R], axis=1)
#
# # def estimate_vergence_shift_px(left_pan: np.ndarray, right_pan: np.ndarray, max_shift: int = 200) -> int:
# #     """
# #     Estimate constant horizontal shift to align infinity using image data.
# #
# #     Strategy:
# #     - Use top part of panorama as "far region"
# #     - Build 1D signal per column using vertical edge strength
# #     - Cross-correlate signals to find best shift within [-max_shift, +max_shift]
# #     """
# #     # Use top half (tune if needed)
# #     H = min(left_pan.shape[0], right_pan.shape[0])
# #     y0, y1 = 0, H // 2
# #
# #     L = cv2.cvtColor(left_pan[y0:y1], cv2.COLOR_BGR2GRAY)
# #     R = cv2.cvtColor(right_pan[y0:y1], cv2.COLOR_BGR2GRAY)
# #
# #     # Vertical edges (changes across x)
# #     # Scharr is strong
# #     gxL = cv2.Scharr(L, cv2.CV_32F, 1, 0)
# #     gxR = cv2.Scharr(R, cv2.CV_32F, 1, 0)
# #
# #     # 1D column signals: sum of abs edge strength per column
# #     sL = np.sum(np.abs(gxL), axis=0)
# #     sR = np.sum(np.abs(gxR), axis=0)
# #
# #     # Normalize (optional but helps)
# #     sL = (sL - sL.mean()) / (sL.std() + 1e-6)
# #     sR = (sR - sR.mean()) / (sR.std() + 1e-6)
# #
# #     # Search best shift in range
# #     best_shift = 0
# #     best_score = -1e18
# #
# #     # shift > 0 means: right_pan shifted right
# #     for shift in range(-max_shift, max_shift + 1):
# #         if shift >= 0:
# #             a = sL[shift:]
# #             b = sR[: len(a)]
# #         else:
# #             a = sL[: shift]          # shift is negative → drop right end
# #             b = sR[-shift : -shift + len(a)]
# #
# #         if len(a) < 50:  # too small, ignore
# #             continue
# #
# #         score = float(np.dot(a, b))
# #         if score > best_score:
# #             best_score = score
# #             best_shift = shift
# #
# #     return best_shift
#
# # def crop_valid_overlap(
# #         left_pan: np.ndarray,
# #         right_pan_shifted: np.ndarray,
# #         shift: int
# # ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
# #
# #     h = min(left_pan.shape[0], right_pan_shifted.shape[0])
# #     w = min(left_pan.shape[1], right_pan_shifted.shape[1])
# #
# #     L = left_pan[:h, :w]
# #     R = right_pan_shifted[:h, :w]
# #
# #     if shift > 0:
# #         # right shifted right → leftmost shift columns invalid
# #         return L[:, shift:], R[:, shift:]
# #     elif shift < 0:
# #         s = -shift
# #         return L[:, : w - s], R[:, : w - s]
# #     else:
# #         return L, R
#
# # def align_and_crop_pair(
# #     left_raw: np.ndarray,
# #     right_raw: np.ndarray,
# #     *,
# #     max_shift: int = 200,
# # ) -> Tuple[np.ndarray, np.ndarray, int]:
# #     """
# #     Estimate vergence shift (infinity alignment), apply shift to right, and crop overlap.
# #     """
# #     shift = estimate_vergence_shift_px(left_raw, right_raw, max_shift=max_shift)
# #
# #     right_aligned = PelegStereo.apply_horizontal_shift(right_raw, shift)
# #     left_crop, right_crop = crop_valid_overlap(left_raw, right_aligned, shift)
# #
# #     assert left_crop is not None and right_crop is not None
# #     return left_crop, right_crop, shift
#
# def overlay_epipolar_lines(
#         img_bgr: np.ndarray,
#         *,
#         num_lines: int = 5,
#         color=(0, 255, 0),
#         thickness: int = 1,
# ) -> np.ndarray:
#     """
#     Overlay horizontal epipolar lines on cylindrical stereo panorama.
#
#     In Peleg cylindrical stereo:
#         Corresponding points lie on the same row.
#         Therefore epipolar lines are horizontal.
#
#     Parameters:
#     - img_bgr:
#         Input cylindrical panorama.
#     - num_lines:
#         Number of evenly spaced horizontal lines.
#     - color:
#         Line color (BGR).
#     - thickness:
#         Line thickness.
#
#     Returns:
#         Image with overlay.
#     """
#     if img_bgr is None or img_bgr.size == 0:
#         return img_bgr
#
#     vis = img_bgr.copy()
#     h, w = vis.shape[:2]
#
#     # Choose evenly spaced row positions
#     ys = np.linspace(0, h - 1, num_lines + 2, dtype=int)[1:-1]
#
#     for y in ys:
#         cv2.line(vis, (0, y), (w - 1, y), color, thickness)
#
#     return vis
#
# def show_and_save_epipolar_overlay(
#         left_bgr: np.ndarray,
#         right_bgr: np.ndarray,
#         *,
#         out_dir: Path | None = None,
#         prefix: str = "peleg",
#         num_lines: int = 6,
#         display_scale: float = 1.0,
#         show: bool = True,
# ) -> dict[str, np.ndarray]:
#     """
#     Create epipolar-line overlays for cylindrical stereo, optionally show and save.
#
#     Returns a dict with:
#       - left_epi
#       - right_epi
#       - sbs_epi
#       - anaglyph_epi
#     """
#     left_epi = overlay_epipolar_lines(left_bgr, num_lines=num_lines)
#     right_epi = overlay_epipolar_lines(right_bgr, num_lines=num_lines)
#
#     sbs_epi = make_side_by_side(left_epi, right_epi)
#     anaglyph_epi = make_anaglyph(left_epi, right_epi)
#
#     if show:
#         cv2.imshow("Epipolar Lines (Left | Right)", _resize_for_display(sbs_epi, display_scale))
#         cv2.imshow("Epipolar Lines Anaglyph", _resize_for_display(anaglyph_epi, display_scale))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     if out_dir is not None:
#         out_dir.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(out_dir / f"{prefix}_left_epipolar.png"), left_epi)
#         cv2.imwrite(str(out_dir / f"{prefix}_right_epipolar.png"), right_epi)
#         cv2.imwrite(str(out_dir / f"{prefix}_sbs_epipolar.png"), sbs_epi)
#         cv2.imwrite(str(out_dir / f"{prefix}_anaglyph_epipolar.png"), anaglyph_epi)
#         print(f"[saved] {out_dir / f'{prefix}_sbs_epipolar.png'}")
#
#     return {
#         "left_epi": left_epi,
#         "right_epi": right_epi,
#         "sbs_epi": sbs_epi,
#         "anaglyph_epi": anaglyph_epi,
#     }

# def estimate_row_disparity_shift(
#         left_bgr: np.ndarray,
#         right_bgr: np.ndarray,
#         *,
#         y: int,
#         band: int = 3,
#         max_shift: int = 120,
# ) -> tuple[int, int]:
#     """
#     Verify epipolar behavior around a chosen row.
#
#     We search for the best match of left row y against right rows (y+dy),
#     where dy is in [-band, +band], and for each dy we also search an x-shift.
#
#     Returns:
#       best_dy: vertical offset of best match (should be ~0 if epipolar lines are horizontal)
#       best_shift: horizontal shift at that dy
#     """
#     Lg = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
#     Rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
#
#     h, w = Lg.shape[:2]
#     y = int(max(0, min(h - 1, y)))
#
#     # Use a small vertical strip (a few rows) to make signal more stable
#     y0 = max(0, y - 2)
#     y1 = min(h, y + 3)
#     L_patch = Lg[y0:y1, :].astype(np.float32)
#
#     best_score = -1e18
#     best_dy = 0
#     best_shift = 0
#
#     for dy in range(-band, band + 1):
#         yr = y + dy
#         if yr < 0 or yr >= h:
#             continue
#
#         r0 = max(0, yr - 2)
#         r1 = min(h, yr + 3)
#         R_patch = Rg[r0:r1, :].astype(np.float32)
#
#         # Make same height (in case at borders)
#         hh = min(L_patch.shape[0], R_patch.shape[0])
#         A = L_patch[:hh]
#         B = R_patch[:hh]
#
#         # Turn into 1D signals by summing abs horizontal gradients (robust)
#         gxA = cv2.Scharr(A, cv2.CV_32F, 1, 0)
#         gxB = cv2.Scharr(B, cv2.CV_32F, 1, 0)
#         sA = np.sum(np.abs(gxA), axis=0)
#         sB = np.sum(np.abs(gxB), axis=0)
#
#         # Normalize
#         sA = (sA - sA.mean()) / (sA.std() + 1e-6)
#         sB = (sB - sB.mean()) / (sB.std() + 1e-6)
#
#         # Search horizontal shift
#         for shift in range(-max_shift, max_shift + 1):
#             if shift >= 0:
#                 a = sA[shift:]
#                 b = sB[: len(a)]
#             else:
#                 a = sA[: shift]
#                 b = sB[-shift : -shift + len(a)]
#
#             if len(a) < 100:
#                 continue
#
#             score = float(np.dot(a, b))
#             if score > best_score:
#                 best_score = score
#                 best_dy = dy
#                 best_shift = shift
#
#     return best_dy, best_shift
#

def run_prerecorded_video(
        video_path: str | Path,
        cfg: PelegStereoConfig,
        *,
        show_progress: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build raw left/right cylindrical panoramas from a prerecorded video.

    Responsibility:
    - Read frames
    - Call builder.process_frame
    - (Optional) display progress

    Returns:
    - (left_raw, right_raw)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Create stabilized Peleg pipeline
    pipe = StablePelegPipeline(
        peleg_cfg=cfg,
        pipe_cfg=StablePelegConfig(
        tau=3.0,
        max_iters=1500,
        stabilize_mode="vertical",
        smoothing="exp",
        ma_window=15,
        )
    )

    # Read first frame for initialization
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Video contains no frames.")

    pipe.initialize(first_frame)

    # Main loop
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        stable_frame, (left_pan, right_pan), info = pipe.step(frame)

        if show_progress:
            cv2.imshow("Input Frame", _resize_for_display(frame, cfg.display_scale))
            cv2.imshow("Stabilized Frame", _resize_for_display(stable_frame, cfg.display_scale))

            if left_pan is not None:
                cv2.imshow("Left Panorama (raw)", _resize_for_display(left_pan, cfg.display_scale))
            if right_pan is not None:
                cv2.imshow("Right Panorama (raw)", _resize_for_display(right_pan, cfg.display_scale))

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return left_pan, right_pan


#
# def make_anaglyph(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
#     """
#     Create red-cyan anaglyph from stereo pair.
#
#     left image → red channel
#     right image → green + blue channels
#     """
#     # Ensure same size
#     h = min(left_bgr.shape[0], right_bgr.shape[0])
#     w = min(left_bgr.shape[1], right_bgr.shape[1])
#
#     L = left_bgr[:h, :w]
#     R = right_bgr[:h, :w]
#
#     anaglyph = np.zeros_like(L)
#     anaglyph[:, :, 2] = L[:, :, 2]  # Red from left
#     anaglyph[:, :, 1] = R[:, :, 1]  # Green from right
#     anaglyph[:, :, 0] = R[:, :, 0]  # Blue from right
#
#     return anaglyph
#
#
# def show_anaglyph(left_bgr: np.ndarray, right_bgr: np.ndarray, *, title: str = "Anaglyph") -> None:
#     """
#     Display an anaglyph preview and wait for a keypress.
#     """
#     anaglyph = make_anaglyph(left_bgr, right_bgr)
#     cv2.imshow(title, anaglyph)
#     cv2.waitKey(0)
#     cv2.destroyWindow(title)
#
#
# def show_and_save_anaglyph(
#         left_bgr: np.ndarray,
#         right_bgr: np.ndarray,
#         output_path: Optional[Path] = None,
# ):
#     """
#     Display an anaglyph preview and wait for a keypress.
#     Generate and save a red-cyan anaglyph image.
#
#     Parameters:
#     - left_bgr:
#         Left stereo image (BGR).
#     - right_bgr:
#         Right stereo image (BGR).
#     - output_path:
#         Path to save output image.
#
#     Returns:
#     - Path to saved file.
#     """
#     anaglyph = make_anaglyph(left_bgr, right_bgr)
#
#     cv2.imshow("Anaglyph", anaglyph)
#     cv2.waitKey(0)
#     cv2.destroyWindow("Anaglyph")
#
#     if output_path is not None:
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(output_path), anaglyph)
#         print(f"[saved] {output_path}")

# def validate_epipolar_with_orb_matches(
#         left_bgr: np.ndarray,
#         right_bgr: np.ndarray,
#         *,
#         n_show: int = 20,
#         max_matches: int = 200,
# ) -> dict[str, float]:
#     """
#     Quantitatively validate the epipolar constraint using sparse feature matches.
#
#     We detect features in left/right panoramas and match them.
#     For each match, measure dy = y_right - y_left.
#
#     For rectified/cylindrical stereo, dy should be near 0.
#
#     Returns summary stats.
#     """
#     Lg = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
#     Rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
#
#     orb = cv2.ORB_create(nfeatures=1500)
#
#     kL, dL = orb.detectAndCompute(Lg, None)
#     kR, dR = orb.detectAndCompute(Rg, None)
#
#     if dL is None or dR is None or len(kL) < 10 or len(kR) < 10:
#         return {"num_matches": 0, "median_abs_dy": float("nan"), "mean_abs_dy": float("nan"), "max_abs_dy": float("nan")}
#
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(dL, dR)
#     matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
#
#     dys = []
#     for m in matches:
#         (xL, yL) = kL[m.queryIdx].pt
#         (xR, yR) = kR[m.trainIdx].pt
#         dys.append(float(yR - yL))
#
#     dys = np.array(dys, dtype=np.float64)
#     abs_dy = np.abs(dys)
#
#     stats = {
#         "num_matches": float(len(matches)),
#         "median_abs_dy": float(np.median(abs_dy)),
#         "mean_abs_dy": float(np.mean(abs_dy)),
#         "max_abs_dy": float(np.max(abs_dy)),
#     }
#     return stats


def draw_top_matches_with_row_info(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        *,
        max_draw: int = 20,
) -> np.ndarray:
    """
    Draw top ORB matches between left/right panoramas and annotate dy values.

    Returns a visualization image.
    """
    Lg = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    Rg = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kL, dL = orb.detectAndCompute(Lg, None)
    kR, dR = orb.detectAndCompute(Rg, None)

    if dL is None or dR is None:
        return make_side_by_side(left_bgr, right_bgr)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(dL, dR), key=lambda m: m.distance)[:max_draw]

    vis = cv2.drawMatches(
        left_bgr, kL,
        right_bgr, kR,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Annotate dy for each match near the left keypoint
    hL, wL = left_bgr.shape[:2]
    for idx, m in enumerate(matches):
        xL, yL = kL[m.queryIdx].pt
        xR, yR = kR[m.trainIdx].pt
        dy = yR - yL

        # drawMatches puts right image to the right, so right coords are offset by wL
        p = (int(xL), int(yL))
        cv2.putText(
            vis,
            f"dy={dy:+.1f}",
            (p[0] + 5, p[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis



def main() -> None:
    VIDEO_PATHs = [
        "recorded_video0.mov",
        "recorded_video1.mov",
        # "recorded_video2.mov",
        # "recorded_video3.mov",
    ]

    cfg = PelegStereoConfig(
        strip_offset_px=120,
        strip_width_px=2,
        max_columns=None,
        display_scale=1,
        save_outputs=True,
    )

    for i in range(len(VIDEO_PATHs)):
        VIDEO_PATH = VIDEO_PATHs[i]
        left_raw, right_raw = run_prerecorded_video(VIDEO_PATH, cfg, show_progress=True)
        if left_raw is None or right_raw is None:
            print("No panoramas produced.")
            return

        # Align + crop
        left, right, shift = align_and_crop_pair(left_raw, right_raw, max_shift=200)
        print(f"Estimated vergence shift px: {shift}")


        rows = [left.shape[0] // 4, left.shape[0] // 2, 3 * left.shape[0] // 4]
        for y in rows:
            dy, dx = estimate_row_disparity_shift(left, right, y=y, band=4, max_shift=150)
            print(f"[epipolar check] y={y}: best dy={dy}, best dx={dx}")

        # Epipolar overlay visualization (cylindrical stereo)
        left_epi = overlay_epipolar_lines(left, num_lines=6)
        right_epi = overlay_epipolar_lines(right, num_lines=6)

        sbs_epi = make_side_by_side(left_epi, right_epi)
        cv2.imshow("Cylindrical Stereo with Epipolar Lines (Left | Right)", sbs_epi)

        # Sshow cylindrical anaglyph with lines
        ana_epi = make_anaglyph(left_epi, right_epi)
        cv2.imshow("Cylindrical Anaglyph + Epipolar Lines", ana_epi)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        stats = validate_epipolar_with_orb_matches(left, right, max_matches=200)
        print(f"[epipolar ORB] matches={int(stats['num_matches'])}, "
              f"median|dy|={stats['median_abs_dy']:.2f}px, "
              f"mean|dy|={stats['mean_abs_dy']:.2f}px, "
              f"max|dy|={stats['max_abs_dy']:.2f}px")

        match_vis = draw_top_matches_with_row_info(left, right, max_draw=20)
        cv2.imshow("Top Matches (annotated with dy)", _resize_for_display(match_vis, cfg.display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save
        if cfg.save_outputs:
            out_dir = Path(f"outputs{i}")
            out_dir.mkdir(parents=True, exist_ok=True)

            show_and_save_anaglyph(
                left,
                right,
                out_dir / "peleg_anaglyph.png"
            )

            cv2.imwrite(str(out_dir / "peleg_left_raw.png"), left_raw)
            cv2.imwrite(str(out_dir / "peleg_right_raw.png"), right_raw)
            cv2.imwrite(str(out_dir / "peleg_left.png"), left)
            cv2.imwrite(str(out_dir / "peleg_right.png"), right)

            cv2.imwrite(str(out_dir / "peleg_orb_matches_dy.png"), match_vis)
            print(f"[saved] {out_dir / 'peleg_orb_matches_dy.png'}")

            print(f"[saved] {out_dir / 'peleg_left.png'}")
            print(f"[saved] {out_dir / 'peleg_right.png'}")


if __name__ == "__main__":
    main()