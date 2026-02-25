from typing import Tuple, Optional

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from stereocv.peleg import PelegStereoConfig, PelegStereo
from stereocv.stabilize import StablePelegPipeline, StablePelegConfig

from stereocv.stereo.vergence import align_and_crop_pair
from stereocv.stereo.validation import run_epipolar_validation_report, draw_top_matches_with_row_info, \
    summarize_row_band
from stereocv.viz.visulization import (
    resize_for_display, show_and_save_anaglyph, show_and_save_epipolar_overlay,
    draw_status_text, make_side_by_side, make_anaglyph, render_planar_view_from_cylinder, draw_correspondence_arrows
)


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


@dataclass
class PreparedStereo:
    # raw panoramas

    left_raw: np.ndarray
    right_raw: np.ndarray

    # aligned/cropped panoramas (after vergence horizontal shift)
    left: np.ndarray
    right: np.ndarray
    shift_px: int

    stabilized: bool
    video_path: Path


def prepare_stereo_pair(
        video_path: str | Path,
        cfg: PelegStereoConfig,
        *,
        stabilized: bool,
        show_build_process: bool = False,
        max_shift: int = 200,
) -> Optional[PreparedStereo]:
    """
    Build L/R cylindrical panoramas (raw or stabilized) and apply vergence alignment.
    Returns PreparedStereo or None if panoramas could not be produced.
    """
    left_raw, right_raw = build_panoramas(
        video_path=Path(video_path),
        cfg=cfg,
        stabilized=stabilized,
        show_progress=show_build_process,
    )

    if left_raw is None or right_raw is None:
        return None

    # Vergence alignment
    left, right, shift_px = align_and_crop_pair(left_raw, right_raw, max_shift=max_shift)

    return PreparedStereo(
        left_raw=left_raw,
        right_raw=right_raw,
        left=left,
        right=right,
        shift_px=shift_px,
        stabilized=stabilized,
        video_path=Path(video_path),
    )


def analyze_cylindrical_pair(
        stereo: PreparedStereo,
        *,
        out_dir: Path | None,
        tag: str,
        cfg: PelegStereoConfig,
        show: bool,
        save: bool,
) -> dict:

    left = stereo.left
    right = stereo.right

    # Epipolar validation
    report = run_epipolar_validation_report(left, right, band=4, max_shift=150, max_matches=200)

    # # Overlays + match visulization
    # show_and_save_epipolar_overlay(
    #     left, right,
    #     out_dir=out_dir if save else None,
    #     prefix=tag,
    #     num_lines=8,
    #     display_scale=cfg.display_scale,
    #     show=show,
    # )

    match_vis = draw_top_matches_with_row_info(left, right, max_draw=20)
    if show:
        cv2.imshow(f"Top Matches ({tag})", resize_for_display(match_vis, cfg.display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save and out_dir is not None:
        cv2.imwrite(str(out_dir / f"{tag}_left_raw.png"), stereo.left_raw)
        cv2.imwrite(str(out_dir / f"{tag}_right_raw.png"), stereo.right_raw)
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
        "shift_px": stereo.shift_px,
        "report": report,
        "match_vis": match_vis,
    }


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

    left_pan = None
    right_pan = None

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
            # cv2.imshow("Input Frame", resize_for_display(frame_vis, cfg.display_scale))
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
        print(f"\n[STABILIZED] mean|corr_ty|={corr_abs_sum / corr_count:.3f}px, max|corr_ty|={corr_abs_max:.3f}px")
    else:
        print("\n[STABILIZED] no corr_ty stats collected (no frames / no stabilization info)")

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

#
# def show_planar_view(VIDEO_PATHs: list[str], cfg: PelegStereoConfig) -> None:
#     cv2.destroyAllWindows()
#     # Render planar perspective views
#     yaw_deg = 0.0   # center look direction
#     fov_deg = 60.0
#     pano_fov_deg = 200
#
#     for i, video_path in enumerate(VIDEO_PATHs):
#         stereo = prepare_stereo_pair(video_path, cfg, stabilized=True, show_build_process=False)
#         if stereo is None:
#             print("No panoramas produced.")
#             continue
#
#         while True:
#             Lp = render_planar_view_from_cylinder(
#                 stereo.left, yaw_deg=yaw_deg, fov_deg=fov_deg,
#                 panorama_fov_deg=pano_fov_deg, out_w=960,
#             )
#             Rp = render_planar_view_from_cylinder(
#                 stereo.right, yaw_deg=yaw_deg, fov_deg=fov_deg,
#                 panorama_fov_deg=pano_fov_deg, out_w=960,
#             )
#
#             cv2.imshow("Planar Stereo (L|R)", resize_for_display(make_side_by_side(Lp, Rp), cfg.display_scale))
#             # cv2.imshow("Planar Anaglyph", resize_for_display(make_anaglyph(Lp, Rp), cfg.display_scale))
#
#             key = cv2.waitKey(30) & 0xFF
#             if key in (27, ord("q")):
#                 cv2.destroyAllWindows()
#                 break
#             if key == ord("a"):
#                 yaw_deg -= 2.0
#             elif key == ord("d"):
#                 yaw_deg += 2.0
#             elif key == ord("w"):
#                 fov_deg = min(100.0, fov_deg + 2.0)
#             elif key == ord("s"):
#                 fov_deg = max(20.0, fov_deg - 2.0)

def show_planar_view(VIDEO_PATHs: list[str], cfg: PelegStereoConfig) -> None:
    # Close any lingering windows so key focus is clean
    cv2.destroyAllWindows()

    yaw_deg = 0.0
    fov_deg = 60.0
    pano_fov_deg = 180

    for i, video_path in enumerate(VIDEO_PATHs):
        stereo = prepare_stereo_pair(video_path, cfg, stabilized=True, show_build_process=False)
        if stereo is None:
            print("No panoramas produced.")
            continue

        while True:
            Lp = render_planar_view_from_cylinder(
                stereo.left, yaw_deg=yaw_deg, fov_deg=fov_deg,
                panorama_fov_deg=pano_fov_deg, out_w=960,
            )
            Rp = render_planar_view_from_cylinder(
                stereo.right, yaw_deg=yaw_deg, fov_deg=fov_deg,
                panorama_fov_deg=pano_fov_deg, out_w=960,
            )

            # HUD lines (on-image text)
            hud_lines = [
                f"yaw={yaw_deg:+.1f} deg",
                f"fov={fov_deg:.1f} deg",
                f"pano_fov={pano_fov_deg:.0f} deg",
                "WASD: yaw/fov   q/ESC: quit",
            ]

            # view = make_side_by_side(Lp, Rp)
            # cv2.imshow("Planar Stereo (L|R)", resize_for_display(view, cfg.display_scale))
            ana = make_anaglyph(Lp, Rp)
            ana_vis = draw_status_text(ana, hud_lines)
            cv2.imshow("Planar Anaglyph", resize_for_display(ana_vis, cfg.display_scale))
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                break

            if key == ord("a"):
                yaw_deg -= 2.0
                # print(f"[viewer] yaw={yaw_deg:.1f}  fov={fov_deg:.1f}")
            elif key == ord("d"):
                yaw_deg += 2.0
                # print(f"[viewer] yaw={yaw_deg:.1f}  fov={fov_deg:.1f}")
            elif key == ord("w"):
                fov_deg = min(100.0, fov_deg + 2.0)
                # print(f"[viewer] yaw={yaw_deg:.1f}  fov={fov_deg:.1f}")
            elif key == ord("s"):
                fov_deg = max(20.0, fov_deg - 2.0)
                # print(f"[viewer] yaw={yaw_deg:.1f}  fov={fov_deg:.1f}")


def compare_stabilization(VIDEO_PATHs: list[str], cfg: PelegStereoConfig) -> None:
    for i, video_path in enumerate(VIDEO_PATHs):
        out_dir = Path(f"outputs{i}")
        out_dir.mkdir(parents=True, exist_ok=True)

        original = prepare_stereo_pair(video_path, cfg, stabilized=False, show_build_process=False)
        stabilized = prepare_stereo_pair(video_path, cfg, stabilized=True, show_build_process=False)

        if original is None or stabilized is None:
            print("No panoramas produced for one of the runs.")
            continue

        res_original = analyze_cylindrical_pair(original, out_dir=out_dir, tag="raw", cfg=cfg, show=False, save=True)
        res_stabilized = analyze_cylindrical_pair(stabilized, out_dir=out_dir, tag="stabilized", cfg=cfg, show=False, save=True)

        ori_left = res_original['left']
        ori_right = res_original['right']
        stab_left = res_stabilized['left']
        stab_right = res_stabilized['right']

        # summaries
        m0, M0 = summarize_row_band(res_original["report"]["row_band"])
        m1, M1 = summarize_row_band(res_stabilized["report"]["row_band"])
        orb0 = res_original["report"]["orb"]
        orb1 = res_stabilized["report"]["orb"]

        print(f"[ORI]  vergence shift: {res_original['shift_px']}")
        print(f"[STAB] vergence shift: {res_stabilized['shift_px']}")
        print(f"\n[ORI row-band]  mean|dy|={m0:.2f}px  max|dy|={M0:.2f}px")
        print(f"[STAB row-band] mean|dy|={m1:.2f}px  max|dy|={M1:.2f}px")
        print(f"\n[ORI ORB]  median|dy|={orb0['median_abs_dy']:.2f}px  mean|dy|={orb0['mean_abs_dy']:.2f}px  max|dy|={orb0['max_abs_dy']:.2f}px")
        print(f"[STAB ORB] median|dy|={orb1['median_abs_dy']:.2f}px  mean|dy|={orb1['mean_abs_dy']:.2f}px  max|dy|={orb1['max_abs_dy']:.2f}px")

        # comparisons
        cv2.imwrite(str(out_dir / "compare_left_raw_vs_stab.png"),
                    make_side_by_side(ori_left, stab_left))
        cv2.imwrite(str(out_dir / "compare_right_raw_vs_stab.png"),
                    make_side_by_side(ori_right, stab_right))
        cv2.imwrite(str(out_dir / "compare_anaglyph_raw_vs_stab.png"), make_side_by_side(
            make_anaglyph(ori_left, ori_right),
            make_anaglyph(stab_left, stab_right),
        ))

        cv2.imshow("RAW vs STABILIZED (Left panoramas)", resize_for_display(
            make_side_by_side(ori_left, stab_left), cfg.display_scale))
        cv2.imshow("RAW vs STABILIZED (Right panoramas)", resize_for_display(
            make_side_by_side(ori_right, stab_right), cfg.display_scale))
        cv2.imshow("RAW vs STABILIZED (Anaglyphs)", resize_for_display(
            make_side_by_side(
            make_anaglyph(ori_left, ori_right),
            make_anaglyph(stab_left, stab_right)
        ), cfg.display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generate_panorama(VIDEO_PATHs: list[str], cfg: PelegStereoConfig) -> None:
    for i, video_path in enumerate(VIDEO_PATHs):
        out_dir = Path(f"outputs{i}")
        if cfg.save_outputs:
            out_dir.mkdir(parents=True, exist_ok=True)

        stereo = prepare_stereo_pair(video_path, cfg, stabilized=True, show_build_process=True)
        if stereo is None:
            print("No panoramas produced.")
            continue

        res = analyze_cylindrical_pair(
            stereo,
            out_dir=out_dir,
            tag="stabilized",
            cfg=cfg,
            show=True,
            save=cfg.save_outputs,
        )

        # Print a compact summary
        orb = res["report"]["orb"]
        m, M = summarize_row_band(res["report"]["row_band"])
        print(f"[STABILIZED] vergence shift: {res['shift_px']}")
        print(f"[STABILIZED row-band] mean|dy|={m:.2f}px  max|dy|={M:.2f}px")
        print(f"[STABILIZED ORB] median|dy|={orb['median_abs_dy']:.2f}px  mean|dy|={orb['mean_abs_dy']:.2f}px  max|dy|={orb['max_abs_dy']:.2f}px")

#
# def main() -> None:
#     VIDEO_PATHs = [
#         # "recorded_video0.mov",
#         "recorded_video1.mov",
#     ]
#
#     cfg = PelegStereoConfig(
#         strip_offset_px=120,
#         strip_width_px=2,
#         max_columns=None,
#         display_scale=1,
#         save_outputs=True,
#     )
#
#     print("\nModes:")
#     print("1 - Generate Panorama (stabilized)")
#     print("2 - Compare Stabilization")
#     print("3 - Planar viewer")
#     print("q - Quit")
#
#     while True:
#         key = input("\nSelect mode: ").strip().lower()
#
#         if key == "1":
#             generate_panorama(VIDEO_PATHs, cfg)
#         elif key == "2":
#             compare_stabilization(VIDEO_PATHs, cfg)
#         elif key == "3":
#             show_planar_view(VIDEO_PATHs, cfg)
#         elif key in ("q", "quit"):
#             break
#         cv2.destroyAllWindows()

def main() -> None:
    VIDEO_PATHs = [
        "recorded_video0.mov",
        "recorded_video1.mov",
    ]

    cfg = PelegStereoConfig(
        strip_offset_px=120,
        strip_width_px=2,
        max_columns=None,
        display_scale=1,
        save_outputs=True,
    )

    mode = "generate"  # "generate" | "compare" | "planar"

    print(
        "\nControls:\n"
        "  1 = Generate Panorama (stabilized)\n"
        "  2 = Compare Stabilization\n"
        "  3 = Planar View\n"
        "  q/ESC = quit\n"
    )

    while True:
        if mode == "generate":
            generate_panorama(VIDEO_PATHs, cfg)
        elif mode == "compare":
            compare_stabilization(VIDEO_PATHs, cfg)
        elif mode == "planar":
            show_planar_view(VIDEO_PATHs, cfg)


        canvas = np.zeros((160, 520, 3), dtype=np.uint8)
        cv2.putText(canvas, "Press 1/2/3 to switch mode, q to quit",
                    (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Mode Switch", canvas)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == 27 or key == ord("q"):
            break
        if key == ord("1"):
            mode = "generate"
        elif key == ord("2"):
            mode = "compare"
        elif key == ord("3"):
            mode = "planar"


if __name__ == "__main__":
    main()