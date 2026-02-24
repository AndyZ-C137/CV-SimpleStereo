import numpy as np

from stereocv.ransac.affine import apply_T
from stereocv.ransac.core import ransac
from stereocv.ransac.affine_fitter import AffineFitter


def main() -> None:
    rng = np.random.default_rng(0)

    # True affine transform
    T_true = np.array(
        [[1.05, 0.02, 15.0],
         [-0.01, 0.98, -8.0],
         [0.0,  0.0,  1.0]],
        dtype=np.float64,
    )

    # Generate inlier points
    n_in = 200
    pts0 = rng.uniform([0, 0], [640, 480], size=(n_in, 2)).astype(np.float64)
    pts1 = apply_T(T_true, pts0)

    # Add Gaussian noise (pixel noise)
    pts1 += rng.normal(0.0, 0.8, size=pts1.shape)

    # Add outliers (wrong matches)
    n_out = 80
    o0 = rng.uniform([0, 0], [640, 480], size=(n_out, 2)).astype(np.float64)
    o1 = rng.uniform([0, 0], [640, 480], size=(n_out, 2)).astype(np.float64)

    pts0_all = np.vstack([pts0, o0]).astype(np.float64)
    pts1_all = np.vstack([pts1, o1]).astype(np.float64)

    # Run RANSAC
    res = ransac(
        model_fitter=AffineFitter(),
        pts0=pts0_all,
        pts1=pts1_all,
        min_samples=3,
        tau=3.0,         # your choice
        max_iters=2000,
        seed=42,
    )

    print("T_true:\n", T_true)
    if res is None:
        print("RANSAC failed.")
        return

    print("T_est:\n", res.model)
    print("num_inliers:", res.num_inliers, "/", pts0_all.shape[0])
    print("rms_error:", res.rms_error)
    print("iterations:", res.iterations)


if __name__ == "__main__":
    main()

