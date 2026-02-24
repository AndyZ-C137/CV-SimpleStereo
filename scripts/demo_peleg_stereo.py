from typing import Tuple, Optional

import cv2
import numpy as np
from pathlib import Path

from stereocv.peleg import PelegStereoConfig, PelegStereo
from stereocv.stabilize import StablePelegPipeline, StablePelegConfig

from stereocv.stereo.vergence import align_and_crop_pair
from stereocv.stereo.validation import run_epipolar_validation_report, draw_top_matches_with_row_info
from stereocv.viz.visulization import (
    resize_for_display,
    show_and_save_anaglyph,
    show_and_save_epipolar_overlay, draw_status_text, make_side_by_side, make_anaglyph,
)

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

def summarize_row_band(row_band: list[dict]) -> tuple[float, float]:
    """
    Given row_band = [{"y":..., "best_dy":..., "best_dx":...}, ...]
    return:
      mean_abs_dy, max_abs_dy
    """
    if not row_band:
        return float("nan"), float("nan")
    dys = np.array([float(r["best_dy"]) for r in row_band], dtype=np.float64)
    return float(np.mean(np.abs(dys))), float(np.max(np.abs(dys)))

def run_prerecorded_video(
        video_path: str | Path,
        cfg: PelegStereoConfig,
        *,
        show_progress: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build stabilized left/right cylindrical panoramas from a prerecorded video.

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

    corr_abs_sum = 0.0
    corr_abs_max = 0.0
    corr_count = 0

    # Create stabilized Peleg pipeline
    pipe = StablePelegPipeline(
        peleg_cfg=cfg,
        pipe_cfg=StablePelegConfig(
            tau=3.0,
            max_iters=1500,
            stabilize_mode="vertical",
            smoothing="exp",
            exp_alpha=0.2,
            trajectory_mode="incremental",
            return_debug_points=True,
            debug_sample_n=80,
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

        # Accumulate stabilization correction magnitude (vertical only)
        stab = info.get("stabilization")
        if stab is not None:
            corr_ty = abs(float(stab.get("corr_ty", 0.0)))
            corr_abs_sum += corr_ty
            corr_abs_max = max(corr_abs_max, corr_ty)
            corr_count += 1

        dbg = info.get("debug_points", None)
        if dbg is not None:
            pts0 = dbg.get("pts0", None)
            pts1 = dbg.get("pts1", None)
            inliers = dbg.get("inliers", None)

            if pts0 is not None and pts1 is not None:
                stable_frame = draw_correspondence_arrows(stable_frame, pts0, pts1, inliers, max_draw=80)

        # Build debug lines from `info`
        lines = []

        trk = info.get("tracking", {})
        lines.append(f"tracked_clean: {trk.get('num_clean', '?')}  refreshed={trk.get('refreshed', False)}")

        if "ransac" in info:
            r = info["ransac"]
            lines.append(f"ransac: inliers={r['num_inliers']}  ratio={r['inlier_ratio']:.3f}")
            lines.append(f"rms={r['rms_error']:.2f}  iters={r['iterations']}  tau={r['threshold']:.1f}")
        else:
            if info.get("ransac_failed", False):
                lines.append("ransac: FAILED")
            elif info.get("tracking_empty", False):
                lines.append("tracking: EMPTY")

        stab = info.get("stabilization", {})
        if stab:
            lines.append(f"corr_ty={stab.get('corr_ty', 0.0):+.2f}  corr_tx={stab.get('corr_tx', 0.0):+.2f}")

        # draw the debug lines on both frames
        frame_vis = draw_status_text(frame, lines)
        stable_vis = draw_status_text(stable_frame, lines)

        if show_progress:
            cv2.imshow("Input Frame", resize_for_display(frame_vis, cfg.display_scale))
            cv2.imshow("Stabilized Frame", resize_for_display(stable_vis, cfg.display_scale))

            if left_pan is not None:
                cv2.imshow("Left Panorama (raw)", resize_for_display(left_pan, cfg.display_scale))
            if right_pan is not None:
                cv2.imshow("Right Panorama (raw)", resize_for_display(right_pan, cfg.display_scale))

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if corr_count > 0:
        print(f"\n[stabilizer] mean|corr_ty|={corr_abs_sum / corr_count:.3f}px, max|corr_ty|={corr_abs_max:.3f}px")
    else:
        print("\n[stabilizer] no corr_ty stats collected (no frames / no stabilization info)")

    return left_pan, right_pan


def run_prerecorded_video_raw(
        video_path: str | Path,
        cfg: PelegStereoConfig,
        *,
        show_progress: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build left/right cylindrical panoramas WITHOUT stabilization.
    Baseline Peleg: just stack strips from each raw frame.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    builder = PelegStereo(cfg)

    left_pan = None
    right_pan = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        builder.process_frame(frame)
        left_pan, right_pan = builder.get_panoramas()

        if show_progress:
            cv2.imshow("Raw Input Frame", resize_for_display(frame, cfg.display_scale))
            if left_pan is not None:
                cv2.imshow("Raw Left Panorama", resize_for_display(left_pan, cfg.display_scale))
            if right_pan is not None:
                cv2.imshow("Raw Right Panorama", resize_for_display(right_pan, cfg.display_scale))

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return left_pan, right_pan


def compare_stablization(VIDEO_PATHs: list[str], cfg: PelegStereoConfig) -> None:
    for i, video_path in enumerate(VIDEO_PATHs):
        out_dir = Path(f"outputs{i}")
        out_dir.mkdir(parents=True, exist_ok=True)

        left_raw0, right_raw0 = build_panoramas(video_path, cfg, stabilized=False, show_progress=False)
        left_raw1, right_raw1 = build_panoramas(video_path, cfg, stabilized=True, show_progress=False)

        if None in (left_raw0, right_raw0, left_raw1, right_raw1):
            print("No panoramas produced for one of the runs.")
            continue

        raw = process_panorama_pair(left_raw0, right_raw0, out_dir=out_dir, tag="raw", cfg=cfg, show=False, save=True)
        stb = process_panorama_pair(left_raw1, right_raw1, out_dir=out_dir, tag="stabilized", cfg=cfg, show=False, save=True)

        # summaries
        m0, M0 = summarize_row_band(raw["report"]["row_band"])
        m1, M1 = summarize_row_band(stb["report"]["row_band"])
        orb0 = raw["report"]["orb"]
        orb1 = stb["report"]["orb"]

        print(f"[RAW]        vergence shift: {raw['shift_px']}")
        print(f"[STABILIZED] vergence shift: {stb['shift_px']}")
        print(f"[RAW row-band]  mean|dy|={m0:.2f}px  max|dy|={M0:.2f}px")
        print(f"[STB row-band]  mean|dy|={m1:.2f}px  max|dy|={M1:.2f}px")
        print(f"[RAW ORB]        median|dy|={orb0['median_abs_dy']:.2f}px  mean|dy|={orb0['mean_abs_dy']:.2f}px  max|dy|={orb0['max_abs_dy']:.2f}px")
        print(f"[STAB ORB]       median|dy|={orb1['median_abs_dy']:.2f}px  mean|dy|={orb1['mean_abs_dy']:.2f}px  max|dy|={orb1['max_abs_dy']:.2f}px")

        # comparisons
        cv2.imwrite(str(out_dir / "compare_left_raw_vs_stab.png"), make_side_by_side(raw["left"], stb["left"]))
        cv2.imwrite(str(out_dir / "compare_right_raw_vs_stab.png"), make_side_by_side(raw["right"], stb["right"]))
        cv2.imwrite(str(out_dir / "compare_anaglyph_raw_vs_stab.png"), make_side_by_side(
            make_anaglyph(raw["left"], raw["right"]),
            make_anaglyph(stb["left"], stb["right"]),
        ))

        # optional quick view
        cv2.imshow("RAW vs STABILIZED (Left panoramas)", resize_for_display(make_side_by_side(raw["left"], stb["left"]), cfg.display_scale))
        cv2.imshow("RAW vs STABILIZED (Anaglyphs)", resize_for_display(make_side_by_side(
            make_anaglyph(raw["left"], raw["right"]),
            make_anaglyph(stb["left"], stb["right"])
        ), cfg.display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def generate_panorama(VIDEO_PATHs: list[str], cfg: PelegStereoConfig) -> None:
    for i, video_path in enumerate(VIDEO_PATHs):
        out_dir = Path(f"outputs{i}")
        if cfg.save_outputs:
            out_dir.mkdir(parents=True, exist_ok=True)

        left_raw, right_raw = build_panoramas(video_path, cfg, stabilized=True, show_progress=True)
        if left_raw is None or right_raw is None:
            print("No panoramas produced.")
            continue

        res = process_panorama_pair(
            left_raw, right_raw,
            out_dir=out_dir,
            tag="stabilized",
            cfg=cfg,
            show=True,
            save=cfg.save_outputs,
        )

        # Print a compact summary (optional)
        orb = res["report"]["orb"]
        m, M = summarize_row_band(res["report"]["row_band"])
        print(f"[stabilized] vergence shift: {res['shift_px']}")
        print(f"[stabilized row-band] mean|dy|={m:.2f}px  max|dy|={M:.2f}px")
        print(f"[stabilized ORB] median|dy|={orb['median_abs_dy']:.2f}px  mean|dy|={orb['mean_abs_dy']:.2f}px  max|dy|={orb['max_abs_dy']:.2f}px")


def build_panoramas(
        video_path: str | Path,
        cfg: PelegStereoConfig,
        *,
        stabilized: bool,
        show_progress: bool,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if stabilized:
        return run_prerecorded_video(video_path, cfg, show_progress=show_progress)
    return run_prerecorded_video_raw(video_path, cfg, show_progress=show_progress)


def process_panorama_pair(
        left_raw: np.ndarray,
        right_raw: np.ndarray,
        *,
        out_dir: Path | None,
        tag: str,
        cfg: PelegStereoConfig,
        show: bool,
        save: bool,
) -> dict:
    # (1) Vergence alignment
    left, right, shift_px = align_and_crop_pair(left_raw, right_raw, max_shift=200)

    # (2) Epipolar validation
    report = run_epipolar_validation_report(left, right, band=4, max_shift=150, max_matches=200)

    # (3) Optional overlays + match vis
    show_and_save_epipolar_overlay(
        left, right,
        out_dir=out_dir if save else None,
        prefix=tag,
        num_lines=8,
        display_scale=cfg.display_scale,
        show=show,
    )

    match_vis = draw_top_matches_with_row_info(left, right, max_draw=10)
    if show:
        cv2.imshow(f"Top Matches ({tag})", resize_for_display(match_vis, cfg.display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save and out_dir is not None:
        cv2.imwrite(str(out_dir / f"{tag}_left_raw.png"), left_raw)
        cv2.imwrite(str(out_dir / f"{tag}_right_raw.png"), right_raw)
        cv2.imwrite(str(out_dir / f"{tag}_left.png"), left)
        cv2.imwrite(str(out_dir / f"{tag}_right.png"), right)
        cv2.imwrite(str(out_dir / f"{tag}_orb_matches_dy.png"), match_vis)

        show_and_save_anaglyph(
            left, right,
            output_path=out_dir / f"{tag}_anaglyph.png",
            display_scale=cfg.display_scale,
            show=show,
        )

    return {
        "left": left,
        "right": right,
        "shift_px": shift_px,
        "report": report,
        "match_vis": match_vis,
    }


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

    generate_panorama(VIDEO_PATHs, cfg)
    # compare_stablization(VIDEO_PATHs, cfg)


if __name__ == "__main__":
    main()