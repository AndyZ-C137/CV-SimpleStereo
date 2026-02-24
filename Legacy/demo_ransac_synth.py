import numpy as np
from stereocv.ransac.affine import fit_affine_minimal, apply_T

# Define a known affine transform
T_true = np.array([[1.2, 0.1, 30.0],
                   [-0.2, 0.9, 10.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

pts0 = np.array([[0.0, 0.0],
                 [100.0, 0.0],
                 [0.0, 50.0]], dtype=np.float64)

pts1 = apply_T(T_true, pts0)

T_est = fit_affine_minimal(pts0, pts1)
print("T_true:\n", T_true)
print("T_est:\n", T_est)
