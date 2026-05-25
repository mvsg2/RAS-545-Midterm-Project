# filename: pixel_to_robot_mapper.py
import numpy as np
import cv2

# ================================
# 1) Put your calibration pairs here
#    Image (u,v)  ->  Robot (X,Y)
# ================================

# Clicked at: x=162, y=45
# Clicked at: x=358, y=51
# Clicked at: x=494, y=45
# Clicked at: x=168, y=179
# Clicked at: x=366, y=176
# Clicked at: x=428, y=175
# Clicked at: x=492, y=173

img_pts = np.array([
    [153, 63],
    [484, 58],
    [218, 128],
    [358, 128],
    [226, 192],
    [351, 191],
    [288, 255],
], dtype=np.float64)

rob_xy = np.array([
    [332.8875427246094,  50.575565338134766],
    [331.2615661621094, -94.13094329833984],
    [308.04815673828125,  27.38081169128418],
    [307.2852783203125, -34.91622543334961],
    [278.18890380859375,  24.748804092407227],
    [277.52886962890625, -31.535066604614258],
    [248.09780883789062,  -3.083724021911621]
], dtype=np.float64)

def fit_affine(img_pts, rob_xy):
    """Fit affine [X Y]^T = M * [u v 1]^T using OpenCV."""
    M, inliers = cv2.estimateAffine2D(
        img_pts.reshape(-1,1,2),
        rob_xy.reshape(-1,1,2),
        ransacReprojThreshold=1.0,
        refineIters=1000
    )
    if M is None:
        raise RuntimeError("Affine estimation failed. Points may be degenerate.")
    return M

def fit_homography(img_pts, rob_xy):
    """Fit projective H so that [X Y 1]^T ~ H * [u v 1]^T."""
    H, inliers = cv2.findHomography(img_pts, rob_xy, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if H is None:
        raise RuntimeError("Homography estimation failed. Points may be degenerate.")
    return H

def apply_affine(M, u, v):
    """Apply affine transform (2x3) to a single pixel (u,v) -> (X,Y)."""
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    XY = M @ uv1
    return float(XY[0]), float(XY[1])

def apply_homography(H, u, v):
    """Apply homography (3x3) to a single pixel (u,v) -> (X,Y)."""
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    Xp, Yp, W = H @ uv1
    if abs(W) < 1e-12:
        raise ZeroDivisionError("Homography scale ~ 0 for this point.")
    return float(Xp / W), float(Yp / W)

def rms_error_affine(M, img_pts, rob_xy):
    ones = np.ones((img_pts.shape[0], 1))
    uv1 = np.hstack([img_pts, ones])          # (N,3)
    pred = (uv1 @ M.T)                        # (N,2)
    err = rob_xy - pred
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def rms_error_homography(H, img_pts, rob_xy):
    uv1 = np.hstack([img_pts, np.ones((img_pts.shape[0],1))])  # (N,3)
    proj = (uv1 @ H.T)                                         # (N,3)
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = rob_xy - proj_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))

def main():
    # --- Fit both models ---
    M = fit_affine(img_pts, rob_xy)
    H = fit_homography(img_pts, rob_xy)

    print("Affine matrix M (2x3):\n", M)
    print("\nHomography H (3x3):\n", H)

    # --- Report RMS fit error ---
    aff_rms = rms_error_affine(M, img_pts, rob_xy)
    hom_rms = rms_error_homography(H, img_pts, rob_xy)
    print(f"\nRMS error (affine):    {aff_rms:.6f} (robot units)")
    print(f"RMS error (homography): {hom_rms:.6f} (robot units)")
if __name__ == "__main__":
    main()
