# """
# Translation-only video stabilization demo.
#
# Goal:
# - Remove handheld up/down (and optionally left/right) jitter
# - Using your existing:
#     - LKTrackPipeline for correspondences
#     - Generic RANSAC core loop
#     - A new TranslationFitter (2 DOF: tx, ty)
#
# This is intentionally minimal:
# - We estimate a translation each frame
# - We accumulate it (camera path)
# - We smooth it (so it doesn't "buzz")
# - We warp the current frame by the negative smoothed path
# """
#
# from __future__ import annotations
#
# from pathlib import Path
#
# import cv2
# import numpy as np
#
# # LK tracking pipeline that produces (pts0_clean, pts1_clean)
# from stereocv.matching.pipeline import LKTrackPipeline
#
# # generic RANSAC loop
# from stereocv.ransac.core import ransac
#
# # translation-only fitter
# from stereocv.ransac.translation_fitter import TranslationFitter
#
#
# def main(video_path: str | None = None) -> None:
#     """
#     Run translation-only stabilization on a video.
#
#     Controls:
#       - Press 'q' or ESC to quit.
#
#     Notes:
#       - This uses grayscale for LK, but warps/display the original BGR frame.
#       - We use an exponential moving average (EMA) to smooth the estimated camera path.
#     """
#     # ----------------------------
#     # 1) Open video source
#     # ----------------------------
#     if video_path is None:
#         video_path = "recorded_video.mov"
#
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {video_path}")
#
#     # ----------------------------
#     # 2) Read first frame and init pipeline
#     # ----------------------------
#     ok, frame0_bgr = cap.read()
#     if not ok or frame0_bgr is None:
#         raise RuntimeError("Could not read first frame.")
#
#     # Convert first frame to grayscale for LK tracking.
#     gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
#
#     # Create tracking pipeline.
#     pipeline = LKTrackPipeline()
#
#     # Initialize with first grayscale frame (detect corners, set prev state).
#     pipeline.initialize(gray0)
#
#     # ---------- Stabilization state -----------
#     # Maintain an estimate of the camera "path" over time:
#     #   path_x, path_y = cumulative translation of camera motion
#     # Then smooth that path so the stabilization is not jittery.
#     path_x = 0.0
#     path_y = 0.0
#
#     # Smoothed path (EMA)
#     smooth_x = 0.0
#     smooth_y = 0.0
#
#     # EMA smoothing factor:
#     #   smaller -> more smoothing (slower response)
#     #   larger  -> less smoothing (faster response)
#     alpha = 0.05
#
#     # RANSAC parameters (tau is in pixels)
#     tau = 3.0               # inlier threshold (px)
#     max_iters = 1000        # upper bound
#     min_samples = 1         # translation-only model needs 1 correspondence
#
#     # Translation-only model fitter
#     fitter = TranslationFitter()
#
#     # Main loop over frames
#     while True:
#         ok, frame_bgr = cap.read()
#         if not ok or frame_bgr is None:
#             break
#
#         # Convert current frame to grayscale for tracking.
#         gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#
#         # ---------- Get cleaned correspondences from pipeline -----------
#         # pipeline.step(gray) returns:
#         #   pts0_clean: points in previous frame
#         #   pts1_clean: corresponding points in current frame
#         #   info: debug dict (counts, masks, etc.)
#         pts0, pts1, info = pipeline.step(gray)
#
#         # If not enough points, skip stabilization for this frame.
#         # (RANSAC will also return None if n < min_samples.)
#         if pts0.shape[0] < 10:
#             cv2.imshow("original", frame_bgr)
#             cv2.imshow("stabilized", frame_bgr)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27 or key == ord("q"):
#                 break
#             continue
#
#         # ------------------------------------------
#         # Run RANSAC to estimate translation model
#         # ------------------------------------------
#         # ransac() function:
#         #   - samples minimal sets
#         #   - computes residuals
#         #   - sets inliers where err < tau
#         #   - refits on inliers for final model
#         result = ransac(
#             model_fitter=fitter,
#             pts0=pts0,
#             pts1=pts1,
#             min_samples=min_samples,
#             tau=tau,
#             max_iters=max_iters,
#             seed=0,
#         )
#
#         if result is None:
#             # If RANSAC fails, just display unstabilized frame.
#             cv2.imshow("original", frame_bgr)
#             cv2.imshow("stabilized", frame_bgr)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27 or key == ord("q"):
#                 break
#             continue
#
#         # ------------------------------------------
#         # 4c) Extract translation (tx, ty) from model
#         # ------------------------------------------
#         # Translation model is stored as 3x3:
#         #   [1 0 tx]
#         #   [0 1 ty]
#         #   [0 0  1]
#         T = result.model
#         tx = float(T[0, 2])
#         ty = float(T[1, 2])
#
#         # ------------------------------------------
#         # 4d) Update pipeline quality feedback
#         # ------------------------------------------
#         # pipeline.update_quality uses inlier ratio to trigger re-detection
#         # on the next step if tracking quality drops:contentReference[oaicite:9]{index=9}
#         inlier_ratio = result.num_inliers / max(1, pts0.shape[0])
#         pipeline.update_quality(inlier_ratio)
#
#         # ------------------------------------------
#         # 4e) Accumulate camera path
#         # ------------------------------------------
#         # Each frame-to-frame translation is "incremental motion".
#         # Summing it gives an estimated camera trajectory.
#         path_x += tx
#         path_y += ty
#
#         # ------------------------------------------
#         # 4f) Smooth the path (EMA)
#         # ------------------------------------------
#         # EMA:
#         #   smooth = (1-alpha)*smooth + alpha*path
#         smooth_x = (1.0 - alpha) * smooth_x + alpha * path_x
#         smooth_y = (1.0 - alpha) * smooth_y + alpha * path_y
#
#         # ------------------------------------------
#         # 4g) Compute stabilization correction
#         # ------------------------------------------
#         # To stabilizer, we warp current frame by the NEGATIVE smoothed path.
#         # This "undoes" the camera motion and keeps video steady.
#         corr_x = 0.0
#         corr_y = smooth_y - path_y
#
#         corr_y = float(np.clip(corr_y, -5.0, 5.0))
#         print("corr:", corr_x, corr_y)
#
#         # If you only want to fix vertical jitter, you can do:
#         # corr_x = 0.0
#         # corr_y = -smooth_y
#
#         # Build a 2x3 warp matrix for cv2.warpAffine:
#         #   [1 0 corr_x]
#         #   [0 1 corr_y]
#         M = np.array([[1.0, 0.0, corr_x],
#                       [0.0, 1.0, corr_y]], dtype=np.float32)
#
#         # Warp the frame.
#         # borderMode: replicate edges so you don't get black borders immediately.
#         stabilized = cv2.warpAffine(
#             frame_bgr,
#             M,
#             dsize=(frame_bgr.shape[1], frame_bgr.shape[0]),
#             flags=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_REPLICATE,
#         )
#
#         # ------------------------------------------
#         # 4h) Display debug
#         # ------------------------------------------
#         # Show the original vs stabilized frames.
#         cv2.imshow("original", frame_bgr)
#         cv2.imshow("stabilized", stabilized)
#
#         # Optional: print debug numbers occasionally
#         # print(f"tx={tx:.2f} ty={ty:.2f}  inliers={result.num_inliers}/{pts0.shape[0]}")
#
#         # Quit on ESC or q
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27 or key == ord("q"):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()

"""
Translation-only stabilization demo (vertical-only correction).

This is tuned for your Peleg / rotate-in-place videos:

- We estimate per-frame translation (tx, ty) with RANSAC from LK correspondences.
- We IGNORE tx (because horizontal motion is "intended" during rotation).
- We smooth ONLY ty (incremental vertical motion) with an EMA filter.
- We warp the current frame by (-ty_filt) to cancel vertical jitter.

Key idea:
- DO NOT stabilizer the accumulated path for Peleg rotation videos.
  Accumulated-path stabilizers will fight slow drift and can create big warps.
- Instead, stabilizer the incremental vertical motion.

Controls:
- Press 'q' or ESC to quit.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from stereocv.matching.pipeline import LKTrackPipeline
from stereocv.ransac.core import ransac
from stereocv.ransac.translation_fitter import TranslationFitter


def main(video_path: str | None = None) -> None:
    VIDEO_PATHs = [
        # "recorded_video0.mov",
        # "recorded_video1.mov",
        "recorded_video2.mov",
        "recorded_video3.mov",
        # "recorded_video4.mov",
    ]

    for i in range(len(VIDEO_PATHs)):
        video_path = VIDEO_PATHs[i]
        # 1) Open video
        if video_path is None:
            video_path = "recorded_video1.mov"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")


        # 2) First frame + init pipeline
        ok, frame0_bgr = cap.read()
        if not ok or frame0_bgr is None:
            raise RuntimeError("Could not read first frame.")

        gray0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)

        pipeline = LKTrackPipeline()
        pipeline.initialize(gray0)

        # 3) RANSAC + stabilization params
        fitter = TranslationFitter()

        # RANSAC threshold (px).
        tau = 3.0
        max_iters = 800
        min_samples = 1  # translation-only needs 1 correspondence

        # Smooth incremental ty using EMA:
        # larger alpha_ty = less smoothing (tracks changes faster)
        # smaller alpha_ty = more smoothing (more stable but lags)
        alpha_ty = 0.20

        # Filter state for vertical motion only
        ty_filt = 0.0

        # clamp to avoid large jumps from a bad RANSAC frame
        clamp_y = 5.0  # pixels


        # 4) Main loop
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # Get cleaned correspondences (prev->curr)
            pts0, pts1, info = pipeline.step(gray)

            # If too few points, skip stabilization this frame
            if pts0.shape[0] < 20:
                cv2.imshow("original", frame_bgr)
                cv2.imshow("stabilized", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
                continue

            # Robustly estimate translation with RANSAC
            result = ransac(
                model_fitter=fitter,
                pts0=pts0,
                pts1=pts1,
                min_samples=min_samples,
                tau=tau,
                max_iters=max_iters,
                seed=0,
            )

            if result is None:
                cv2.imshow("original", frame_bgr)
                cv2.imshow("stabilized", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
                continue

            # Extract tx, ty from 3x3 translation matrix
            T = result.model
            tx = float(T[0, 2])
            ty = float(T[1, 2])

            # Feed tracking quality back into pipeline
            inlier_ratio = result.num_inliers / max(1, pts0.shape[0])
            pipeline.update_quality(inlier_ratio)


            # Stabilize ONLY vertical jitter (ty)
            # Smooth the incremental ty (per frame) with EMA
            ty_filt = (1.0 - alpha_ty) * ty_filt + alpha_ty * ty

            # Warp by negative filtered motion to cancel it
            corr_x = 0.0
            corr_y = -ty_filt

            # Safety clamp to prevent large warps if a frame estimate is bad
            corr_y = float(np.clip(corr_y, -clamp_y, clamp_y))

            print(f"corr: {corr_x:.1f} {corr_y:.3f} | ty={ty:.3f} | inliers={result.num_inliers}/{pts0.shape[0]}")

            # Build affine warp matrix
            M = np.array([[1.0, 0.0, corr_x],
                          [0.0, 1.0, corr_y]], dtype=np.float32)

            stabilized = cv2.warpAffine(
                frame_bgr,
                M,
                dsize=(frame_bgr.shape[1], frame_bgr.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            # Display
            cv2.imshow("original", frame_bgr)
            cv2.imshow("stabilized", stabilized)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()