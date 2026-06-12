"""
test_normal_rotation.py — Numerical check that augmentation-time normal
rotation is geometrically consistent with the image warp.

Method:
  1. Build a synthetic depth map of a slanted plane (its normal has clear
     in-plane x, y components).
  2. Geometric truth: warp the DEPTH map by theta (the exact cv2 call used in
     JointTrainTransform), then recompute normals from the warped depth.
     A rotation about the principal point is equivalent to a camera roll, so
     recomputed normals are exact ground truth.
  3. Augmentation path: compute normals from the ORIGINAL depth, warp the
     normal image by theta, then apply the vector rotation as done in
     utils/augmentations.py.
  4. Assert the two agree to < 1 degree in the image interior.

Context: the original implementation rotated the vectors by +theta, which in
the y-down pixel-aligned camera frame is the WRONG direction — it produced
up to ~2*theta supervision error on every rotated training sample. The fix
(rotating by -theta at the call site) brings the error to < 0.2 deg.

Run:
    python tests/test_normal_rotation.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.geometry import compute_normals
from utils.augmentations import _rotate_normals


def _angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    cos = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _slanted_plane_depth_mm(H, W, fx, fy, cx, cy, a=0.35, b=-0.20, z0=1.0):
    """Depth (mm) of the plane Z = z0 + a*X + b*Y, solved per pixel."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    denom = 1.0 - a * (u - cx) / fx - b * (v - cy) / fy
    return (1000.0 * z0 / denom).astype(np.float32)


def test_rotation_sign(verbose: bool = True) -> bool:
    H = W = 448
    fx = fy = 525.0
    # Principal point at the rotation center so image rotation == camera roll.
    cx = cy = (W - 1) / 2.0
    depth_mm = _slanted_plane_depth_mm(H, W, fx, fy, cx, cy)

    ok = True
    for theta in (5.0, 10.0, 15.0):
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), theta, 1.0)

        # Geometric truth: recompute normals from the rotated depth.
        depth_rot = cv2.warpAffine(depth_mm, M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        n_true, _ = compute_normals(depth_rot, fx=fx, fy=fy, cx=cx, cy=cy)

        # Augmentation path: warp normal image, then rotate vectors as the
        # transform does (note the call-site negation in JointTrainTransform).
        n0, _ = compute_normals(depth_mm, fx=fx, fy=fy, cx=cx, cy=cy)
        n_warp = cv2.warpAffine(n0, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        n_aug = _rotate_normals(n_warp, -theta)   # matches JointTrainTransform
        n_old = _rotate_normals(n_warp, +theta)   # the pre-fix behaviour

        interior = slice(100, -100)
        valid = depth_rot[interior, interior] > 0
        err_aug = _angular_error_deg(n_aug[interior, interior],
                                     n_true[interior, interior])[valid].mean()
        err_old = _angular_error_deg(n_old[interior, interior],
                                     n_true[interior, interior])[valid].mean()

        if verbose:
            print(f"theta={theta:5.1f}  fixed path: {err_aug:6.2f} deg"
                  f"   pre-fix (+theta): {err_old:6.2f} deg")
        if err_aug > 1.0:
            print(f"  FAIL: fixed path error {err_aug:.2f} deg > 1.0 deg")
            ok = False
        if err_old < err_aug:
            print("  FAIL: old behaviour unexpectedly better — re-examine.")
            ok = False
    return ok


if __name__ == "__main__":
    passed = test_rotation_sign()
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)
