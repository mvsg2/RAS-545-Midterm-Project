import numpy as np
import cv2

def fit_affine(img_pts, rob_xy):
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
    H, inliers = cv2.findHomography(img_pts, rob_xy, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if H is None:
        raise RuntimeError("Homography estimation failed. Points may be degenerate.")
    return H

def apply_affine(M, u, v):
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    XY = M @ uv1
    return float(XY[0]), float(XY[1])

def apply_homography(H, u, v):
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
