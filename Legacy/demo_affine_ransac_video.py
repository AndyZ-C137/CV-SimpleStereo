from __future__ import annotations

import cv2
import numpy as np

from stereocv.matching.pipeline import LKTrackPipeline
from stereocv.ransac.affine_fitter import AffineFitter
from stereocv.ransac.core import ransac


def main(video_path: str | None = None) -> None:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    ok, frame0 = cap.read()

    if not ok:
        raise RuntimeError("Could not read first frame")

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    pipeline = LKTrackPipeline()
    pipeline.initialize(gray0)

    fitter = AffineFitter()

    while True:
        ok, frame1 = cap.read()
        if not ok:
            break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Get CLEAN correspondences directly from pipeline
        pts0, pts1, info = pipeline.step(gray1)

        # Run RANSAC if enough points
        result = None
        inlier_ratio = None

        if pts0.shape[0] >= 3:
            result = ransac(
                fitter,
                pts0,
                pts1,
                min_samples=3,
                tau=3.0,
                max_iters=2000,
                seed=0,
            )

            if result is not None and pts0.shape[0] > 0:
                inlier_ratio = result.num_inliers / float(pts0.shape[0])

        # Adaptive corner refresh feedback (notify pipeline)
        pipeline.update_quality(0.0 if result is None else inlier_ratio)

        # Visualization
        vis = frame1.copy()

        if result is not None:
            inliers = result.inliers  # bool mask over pts0/pts1 (cleaned arrays)

            # Draw correspondences
            for p0, p1, is_in in zip(pts0, pts1, inliers):
                x0, y0 = int(round(p0[0])), int(round(p0[1]))
                x1, y1 = int(round(p1[0])), int(round(p1[1]))

                # Green = inlier, Red = outlier
                color = (0, 255, 0) if is_in else (0, 0, 255)
                cv2.circle(vis, (x1, y1), 2, color, -1)
                cv2.line(vis, (x0, y0), (x1, y1), color, 1)

            ratio_txt = f"{inlier_ratio:.2f}" if inlier_ratio is not None else "NA"
            text = (
                f"inliers={result.num_inliers}/{pts0.shape[0]} "
                f"ratio={inlier_ratio:.2f} rms={result.rms_error:.2f} "
                f"iters={result.iterations} "
                f"refresh={info['refreshed']} cd={info['cooldown_left']}"
            )

        else:
            text = (
                f"RANSAC failed/skip  "
                f"pts={pts0.shape[0]} "
                f"refresh={info['refreshed']} cd={info['cooldown_left']}"
            )

        cv2.putText(
            vis,
            text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Affine RANSAC (LK Pipeline)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(video_path=None)
